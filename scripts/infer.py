from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import AgricapDataset
from models.agridino import AgriDINO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 inference/evaluation for AgriDINO")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model state_dict (.pt)")
    parser.add_argument("--task", type=str, choices=["retrieval", "zeroshot"], default="retrieval")
    parser.add_argument("--annotations", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--branch", type=str, choices=["global", "fine", "avg"], default="global")
    parser.add_argument("--topk", type=str, default="1,5,10", help="Comma-separated K values for retrieval recall")
    parser.add_argument(
        "--class-text-source",
        type=str,
        choices=["short", "long", "id2label"],
        default="short",
        help="Text source for zero-shot class prototypes",
    )
    parser.add_argument("--id2label", type=str, default=None, help="JSON mapping {label_id: label_text} for zeroshot")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return data


def resolve_device(device_arg: str | None, config: dict) -> torch.device:
    candidate = device_arg if device_arg is not None else str(config.get("device", "cuda"))
    if candidate.startswith("cuda") and torch.cuda.is_available():
        return torch.device(candidate)
    return torch.device("cpu")


def select_branch_features(
    v_g: torch.Tensor,
    v_f: torch.Tensor,
    t_g: torch.Tensor,
    t_f: torch.Tensor,
    branch: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Branch selection for evaluation:
    # - global: use CLS/global branch only
    # - fine: use MAP/fine-grained branch only
    # - avg: average two branches, then L2-normalize again
    if branch == "global":
        return v_g, t_g
    if branch == "fine":
        return v_f, t_f
    v = F.normalize(0.5 * (v_g + v_f), p=2, dim=-1)
    t = F.normalize(0.5 * (t_g + t_f), p=2, dim=-1)
    return v, t


def load_model(args: argparse.Namespace, config: dict, device: torch.device) -> AgriDINO:
    model = AgriDINO(
        dino_weights_path=str(config["dino_weights"]),
        dinov3_repo_path=str(config.get("dinov3_repo", "dinov3")),
        text_model_name_or_path=str(config.get("text_model_name_or_path", "recobo/agriculture-bert-uncased")),
        max_length=int(config.get("max_length", 256)),
        freeze_text_backbone=bool(config.get("freeze_text_backbone", False)),
        logit_scale_max=float(config.get("logit_scale_max", 100.0)),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model


def build_loader(args: argparse.Namespace, config: dict, device: torch.device) -> tuple[AgricapDataset, DataLoader]:
    annotations = args.annotations if args.annotations is not None else str(config.get("annotations", ""))
    image_root = args.image_root if args.image_root is not None else config.get("image_root", None)
    if not annotations:
        raise ValueError("annotations must be provided either in config.yaml or via --annotations")
    if image_root in ("", "null"):
        image_root = None

    dataset = AgricapDataset(
        annotations=annotations,
        image_root=image_root,
        stage="stage2",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    return dataset, loader


def parse_topk(topk_str: str, total: int) -> list[int]:
    values = []
    for part in topk_str.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k > 0:
            values.append(min(k, total))
    return sorted(set(values))


@torch.no_grad()
def run_retrieval(model: AgriDINO, loader: DataLoader, device: torch.device, branch: str, topk: list[int]) -> None:
    all_v = []
    all_t = []

    for batch in tqdm(loader, desc="Extracting features"):
        images = batch["image"].to(device, non_blocking=True)
        short_texts = batch["short_description"]
        long_texts = batch["long_description"]

        v_g, v_f = model.encode_image(images)
        t_g, t_f = model.encode_text(short_texts, long_texts)
        v, t = select_branch_features(v_g, v_f, t_g, t_f, branch)
        all_v.append(v.cpu())
        all_t.append(t.cpu())

    image_feats = torch.cat(all_v, dim=0)
    text_feats = torch.cat(all_t, dim=0)
    # Pairwise similarity over all image-text pairs in the evaluation split.
    sim = image_feats @ text_feats.T

    print(f"Similarity matrix: {tuple(sim.shape)}")
    # Retrieval ground-truth assumes aligned index i <-> i in the same dataset order.
    target = torch.arange(sim.size(0))

    for k in topk:
        i2t_topk = sim.topk(k, dim=1).indices
        t2i_topk = sim.T.topk(k, dim=1).indices
        i2t_hit = (i2t_topk == target.unsqueeze(1)).any(dim=1).float().mean().item()
        t2i_hit = (t2i_topk == target.unsqueeze(1)).any(dim=1).float().mean().item()
        print(f"Recall@{k}: i2t={i2t_hit:.4f}, t2i={t2i_hit:.4f}")


def build_class_texts(dataset: AgricapDataset, source: str, id2label_path: str | None) -> tuple[list[int], list[str]]:
    if source == "id2label":
        if not id2label_path:
            raise ValueError("--id2label is required when --class-text-source id2label")
        with Path(id2label_path).open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError("id2label JSON must be an object mapping id -> text")
        items = sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda x: x[0])
        return [k for k, _ in items], [v for _, v in items]

    id_to_text: dict[int, str] = {}
    # Build one text prototype per disease id from dataset metadata.
    for sample in dataset.samples:
        label_id = int(sample["disease_id"])
        if label_id not in id_to_text:
            id_to_text[label_id] = str(sample["short_description"] if source == "short" else sample["long_description"])
    items = sorted(id_to_text.items(), key=lambda x: x[0])
    return [k for k, _ in items], [v for _, v in items]


@torch.no_grad()
def run_zeroshot(
    model: AgriDINO,
    dataset: AgricapDataset,
    loader: DataLoader,
    device: torch.device,
    branch: str,
    class_text_source: str,
    id2label_path: str | None,
) -> None:
    # Build class text prototypes, then classify each image by max similarity.
    class_ids, class_texts = build_class_texts(dataset, class_text_source, id2label_path)
    if len(class_ids) == 0:
        raise ValueError("No classes found for zero-shot classification")

    t_g, t_f = model.encode_text(class_texts, class_texts)
    if branch == "global":
        class_feats = t_g
    elif branch == "fine":
        class_feats = t_f
    else:
        class_feats = F.normalize(0.5 * (t_g + t_f), p=2, dim=-1)

    class_feats = class_feats.to(device)
    class_id_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)

    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Zero-shot eval"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["disease_id"].to(device, non_blocking=True)
        v_g, v_f = model.encode_image(images)
        if branch == "global":
            image_feats = v_g
        elif branch == "fine":
            image_feats = v_f
        else:
            image_feats = F.normalize(0.5 * (v_g + v_f), p=2, dim=-1)

        # Zero-shot logits: image feature against all class text prototypes.
        logits = image_feats @ class_feats.T
        pred_idx = logits.argmax(dim=1)
        pred_labels = class_id_tensor[pred_idx]
        correct += (pred_labels == labels).sum().item()
        total += labels.numel()

    acc = correct / max(total, 1)
    print(f"Zero-shot disease classification accuracy: {acc:.4f} ({correct}/{total})")
    print(f"Num classes: {len(class_ids)} | text source: {class_text_source}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    device = resolve_device(args.device, config)
    model = load_model(args, config, device)
    dataset, loader = build_loader(args, config, device)

    topk = parse_topk(args.topk, len(dataset))
    if args.task == "retrieval":
        run_retrieval(model, loader, device, args.branch, topk)
    else:
        run_zeroshot(model, dataset, loader, device, args.branch, args.class_text_source, args.id2label)


if __name__ == "__main__":
    main()
