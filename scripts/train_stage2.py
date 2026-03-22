from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import yaml

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import AgricapDataset
from models.agridino import AgriDINO
from utils.losses import taxonomy_aware_soft_target_contrastive_loss


DEFAULT_CONFIG: dict[str, object] = {
    "annotations": "",
    "image_root": None,
    "dino_weights": "",
    "dinov3_repo": "dinov3",
    "text_model_name_or_path": "recobo/agriculture-bert-uncased",
    "max_length": 256,
    "freeze_text_backbone": False,
    "epochs": 10,
    "batch_size": 16,
    "num_workers": 0,
    "lr": 6e-4,
    "weight_decay": 1e-3,
    "drop_last": True,
    "grad_accum_steps": 1,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "min_lr_ratio": 0.01,
    "device": "cuda",
    "seed": 42,
    "output_dir": "outputs/stage2",
    "logit_scale_max": 100.0,
    "init_temperature": 0.07,
}


def _str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 training script for AgriDINO")

    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--annotations", type=str, default=None, help="Path to dataset annotations (.csv or .jsonl)")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for relative image paths")
    parser.add_argument("--dino-weights", type=str, default=None, help="Path to local dinov3_vitl16 checkpoint")
    parser.add_argument("--dinov3-repo", type=str, default=None, help="Path to local dinov3 repo")
    parser.add_argument(
        "--text-model-name-or-path",
        type=str,
        default=None,
        help="HF text model for AgricultureTextEncoder",
    )
    parser.add_argument("--max-length", type=int, default=None, help="Max token length for text encoder")
    parser.add_argument("--freeze-text-backbone", type=_str2bool, default=None, help="Freeze text backbone parameters")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--drop-last", type=_str2bool, default=None, help="Drop last incomplete batch")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    parser.add_argument("--logit-scale-max", type=float, default=None, help="Clamp max for exp(logit_scale)")
    parser.add_argument("--init-temperature", type=float, default=None, help="Initial temperature for logit scale")

    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> LambdaLR:
    warmup_updates = int(total_updates * warmup_ratio)
    warmup_updates = min(max(warmup_updates, 0), max(total_updates - 1, 0))

    def lr_lambda(step_idx: int) -> float:
        if total_updates <= 0:
            return 1.0
        if warmup_updates > 0 and step_idx < warmup_updates:
            return float(step_idx + 1) / float(warmup_updates)

        progress_den = max(total_updates - warmup_updates, 1)
        progress = float(step_idx - warmup_updates) / float(progress_den)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def save_args(args: argparse.Namespace, output_dir: Path) -> None:
    payload = vars(args).copy()
    args_path = output_dir / "train_args.json"
    with args_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_config(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config.yaml must contain a top-level mapping/dictionary.")
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def apply_cli_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    override_map = {
        "annotations": args.annotations,
        "image_root": args.image_root,
        "dino_weights": args.dino_weights,
        "dinov3_repo": args.dinov3_repo,
        "text_model_name_or_path": args.text_model_name_or_path,
        "max_length": args.max_length,
        "freeze_text_backbone": args.freeze_text_backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "drop_last": args.drop_last,
        "grad_accum_steps": args.grad_accum_steps,
        "max_grad_norm": args.max_grad_norm,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "device": args.device,
        "seed": args.seed,
        "output_dir": args.output_dir,
        "logit_scale_max": args.logit_scale_max,
        "init_temperature": args.init_temperature,
    }
    merged = dict(config)
    for key, value in override_map.items():
        if value is not None:
            merged[key] = value
    return merged


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    use_amp: bool,
    epoch_idx: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_loss_g = 0.0
    total_loss_f = 0.0
    num_steps = 0
    num_updates = 0

    progress = tqdm(loader, desc=f"Epoch {epoch_idx + 1}", leave=True)

    for step_idx, batch in enumerate(progress):
        images = batch["image"].to(device, non_blocking=True)
        short_texts = batch["short_description"]
        long_texts = batch["long_description"]
        crop_labels = batch["crop_id"]
        disease_labels = batch["disease_id"]

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            sim_g, sim_f = model(images, short_texts, long_texts)
            loss_g = taxonomy_aware_soft_target_contrastive_loss(sim_g, crop_labels, disease_labels)
            loss_f = taxonomy_aware_soft_target_contrastive_loss(sim_f, crop_labels, disease_labels)
            loss = 0.5 * loss_g + 0.5 * loss_f

        loss_for_backward = loss / grad_accum_steps
        if use_amp:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((step_idx + 1) % grad_accum_steps == 0) or ((step_idx + 1) == len(loader))

        grad_norm_val = 0.0
        if should_step:
            optimizer_stepped = False
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                # If inf/NaN gradients were found, scaler skips optimizer.step and reduces scale.
                optimizer_stepped = scale_after >= scale_before
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norm_val = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                optimizer.step()
                optimizer_stepped = True

            if optimizer_stepped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if optimizer_stepped:
                num_updates += 1

        total_loss += float(loss.detach().item())
        total_loss_g += float(loss_g.detach().item())
        total_loss_f += float(loss_f.detach().item())
        num_steps += 1

        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            loss=f"{loss.detach().item():.4f}",
            loss_g=f"{loss_g.detach().item():.4f}",
            loss_f=f"{loss_f.detach().item():.4f}",
            lr=f"{current_lr:.2e}",
            gnorm=f"{grad_norm_val:.3f}",
        )

    return {
        "loss": total_loss / max(num_steps, 1),
        "loss_g": total_loss_g / max(num_steps, 1),
        "loss_f": total_loss_f / max(num_steps, 1),
        "updates": float(num_updates),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    if not config["annotations"]:
        raise ValueError("Missing required 'annotations' in config.yaml or CLI.")
    if not config["dino_weights"]:
        raise ValueError("Missing required 'dino_weights' in config.yaml or CLI.")

    set_seed(int(config["seed"]))

    device = resolve_device(str(config["device"]))
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_args(argparse.Namespace(**config), output_dir)

    dataset = AgricapDataset(
        annotations=str(config["annotations"]),
        image_root=(None if config["image_root"] in (None, "", "null") else str(config["image_root"])),
        stage="stage2",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=bool(config["drop_last"]),
    )

    model = AgriDINO(
        dino_weights_path=str(config["dino_weights"]),
        dinov3_repo_path=str(config["dinov3_repo"]),
        text_model_name_or_path=str(config["text_model_name_or_path"]),
        max_length=int(config["max_length"]),
        freeze_text_backbone=bool(config["freeze_text_backbone"]),
        logit_scale_max=float(config["logit_scale_max"]),
    ).to(device)
    model.logit_scale.data.fill_(math.log(1.0 / float(config["init_temperature"])))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    grad_accum_steps = max(int(config["grad_accum_steps"]), 1)
    updates_per_epoch = math.ceil(len(loader) / grad_accum_steps)
    total_updates = max(int(config["epochs"]) * updates_per_epoch, 1)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_updates=total_updates,
        warmup_ratio=float(config["warmup_ratio"]),
        min_lr_ratio=float(config["min_lr_ratio"]),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print(f"Train size: {len(dataset)}")
    print(f"Steps/epoch: {len(loader)}, Updates/epoch: {updates_per_epoch}, Total updates: {total_updates}")

    for epoch_idx in range(int(config["epochs"])):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=float(config["max_grad_norm"]),
            use_amp=use_amp,
            epoch_idx=epoch_idx,
        )

        print(
            f"[Epoch {epoch_idx + 1}/{int(config['epochs'])}] "
            f"loss={stats['loss']:.4f}, loss_g={stats['loss_g']:.4f}, "
            f"loss_f={stats['loss_f']:.4f}, lr={stats['lr']:.2e}, updates={int(stats['updates'])}"
        )

        epoch_ckpt = output_dir / f"epoch_{epoch_idx + 1}.pt"
        latest_ckpt = output_dir / "latest.pt"
        torch.save(model.state_dict(), epoch_ckpt)
        torch.save(model.state_dict(), latest_ckpt)

    print("Training completed.")


if __name__ == "__main__":
    main()
