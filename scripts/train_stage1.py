from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_STAGE1_CONFIG: dict[str, Any] = {
    "dinov3_repo": "dinov3",
    "annotations": "",
    "image_root": None,
    "output_dir": "outputs/stage1",
    "resume": True,
    "seed": 0,
    "epochs": 30,
    "batch_size_per_gpu": 16,
    "num_workers": 4,
    "official_epoch_length": 0,  # 0 means auto-compute from dataset size.
    "dino_init_weights": "",
    "lr": 5e-4,
    "min_lr": 1e-6,
    "warmup_epochs": 0,
    "weight_decay": 0.04,
    "weight_decay_end": 0.04,
    "scaling_rule": "none",
    "clip_grad": 3.0,
    "freeze_last_layer_epochs": 1,
    "adamw_beta1": 0.9,
    "adamw_beta2": 0.999,
    "student_arch": "vit_large",
    "patch_size": 16,
    "n_storage_tokens": 4,
    "drop_path_rate": 0.3,
    "teacher_momentum": 0.992,
    "teacher_final_momentum": 1.0,
    "teacher_temp": 0.07,
    "teacher_warmup_temp": 0.04,
    "teacher_warmup_temp_epochs": 30,
    "global_crops_scale": [0.4, 1.0],
    "local_crops_scale": [0.05, 0.4],
    "global_crops_size": 224,
    "local_crops_size": 96,
    "local_crops_number": 8,
    "param_dtype": "bf16",
}

PATH_FIELD_CANDIDATES = ("image_path", "f_path", "path", "img_path", "image")


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
    parser = argparse.ArgumentParser(description="Stage1 SSL training for AgriDINO (DINOv3-style)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--annotations", type=str, default=None, help="Path to CSV/JSONL annotation file")
    parser.add_argument("--image-root", type=str, default=None, help="Root dir for relative image paths")
    parser.add_argument("--dinov3-repo", type=str, default=None, help="Path to local dinov3 repo")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir for checkpoints")
    parser.add_argument("--dino-init-weights", type=str, default=None, help="Initial ViT-L weights path")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size-per-gpu", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--official-epoch-length", type=int, default=None, help="Iteration count per epoch")
    parser.add_argument("--resume", type=_str2bool, default=None, help="Resume from latest checkpoint if exists")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must be a top-level mapping.")
    return payload


def load_stage1_config(config_path: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_yaml(config_path)
    stage1_from_file = payload.get("stage1", {})
    if stage1_from_file is None:
        stage1_from_file = {}
    if not isinstance(stage1_from_file, dict):
        raise ValueError("config.yaml key 'stage1' must be a mapping.")

    cfg = dict(DEFAULT_STAGE1_CONFIG)
    cfg.update(stage1_from_file)

    overrides = {
        "annotations": args.annotations,
        "image_root": args.image_root,
        "dinov3_repo": args.dinov3_repo,
        "output_dir": args.output_dir,
        "dino_init_weights": args.dino_init_weights,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "num_workers": args.num_workers,
        "official_epoch_length": args.official_epoch_length,
        "resume": args.resume,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def ensure_dinov3_importable(dinov3_repo: str | Path) -> None:
    repo = Path(dinov3_repo).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"dinov3 repo path not found: {repo}")
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def sanitize_backbone_state_dict(raw_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = raw_state
    if "teacher" in state and isinstance(state["teacher"], dict):
        state = state["teacher"]
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        norm_key = key
        if "backbone." in norm_key:
            norm_key = norm_key.split("backbone.", 1)[1]
        if norm_key.startswith("module."):
            norm_key = norm_key[len("module.") :]
        out[norm_key] = value
    return out


class AnnotationImageDataset(Dataset):
    def __init__(self, annotations: str | Path, image_root: str | Path | None = None, transform=None) -> None:
        self.annotation_path = Path(annotations)
        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_path}")
        self.image_root = Path(image_root).expanduser().resolve() if image_root not in (None, "", "null") else None
        self.transform = transform
        self.image_paths = self._load_paths(self.annotation_path)
        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image paths found in: {self.annotation_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to open image at index={index}, path='{img_path}'") from exc

        if self.transform is not None:
            image = self.transform(image)
        # Target is unused for SSL, but collate expects (sample, target).
        return image, 0

    def _resolve_path(self, path_value: str) -> Path:
        p = Path(path_value)
        if not p.is_absolute() and self.image_root is not None:
            p = self.image_root / p
        return p.expanduser().resolve()

    def _pick_path_field(self, row: dict[str, Any]) -> str:
        for key in PATH_FIELD_CANDIDATES:
            if key in row and str(row[key]).strip():
                return key
        raise KeyError(f"Cannot find any image path field in row keys: {list(row.keys())}")

    def _load_paths(self, ann_path: Path) -> list[Path]:
        if ann_path.suffix.lower() == ".csv":
            with ann_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        elif ann_path.suffix.lower() == ".jsonl":
            rows = []
            with ann_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no} in {ann_path}") from exc
        else:
            raise ValueError(f"Unsupported annotation format: {ann_path}. Use .csv or .jsonl")

        paths: list[Path] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError(f"Annotation row at index {idx} must be dict, got {type(row).__name__}")
            key = self._pick_path_field(row)
            resolved = self._resolve_path(str(row[key]).strip())
            if not resolved.exists():
                raise FileNotFoundError(f"Image file not found at index={idx}: {resolved}")
            paths.append(resolved)
        return paths


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def export_teacher_backbone(model: torch.nn.Module, output_path: Path) -> None:
    teacher_state = model.model_ema.state_dict()
    backbone_state: dict[str, torch.Tensor] = {}
    for key, value in teacher_state.items():
        if "backbone." not in key:
            continue
        out_key = key.split("backbone.", 1)[1]
        tensor = value
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()  # DTensor compatibility
        backbone_state[out_key] = tensor.detach().cpu()
    if len(backbone_state) == 0:
        raise RuntimeError("No backbone.* keys found in teacher state dict.")
    torch.save(backbone_state, output_path)


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    iteration: int,
    epoch: int,
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, ckpt_dir / f"epoch_{epoch}.pt")
    torch.save(payload, ckpt_dir / "latest.pt")


def try_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    resume: bool,
) -> int:
    if not resume:
        return 0
    latest = output_dir / "checkpoints" / "latest.pt"
    if not latest.exists():
        return 0
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = int(ckpt.get("iteration", -1)) + 1
    print(f"Resumed from checkpoint: {latest} (start_iter={start_iter})")
    return start_iter


def main() -> None:
    args = parse_args()
    stage1_cfg = load_stage1_config(args.config, args)

    if not stage1_cfg["annotations"]:
        raise ValueError("stage1.annotations is required in config.yaml or via --annotations")
    if not stage1_cfg["dino_init_weights"]:
        raise ValueError("stage1.dino_init_weights is required in config.yaml or via --dino-init-weights")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage1 training currently requires CUDA (DINOv3 forward uses .cuda()).")

    ensure_dinov3_importable(stage1_cfg["dinov3_repo"])

    # Imported lazily after dinov3 repo path is inserted.
    from dinov3.configs import get_default_config, write_config
    from dinov3.data import MaskingGenerator, collate_data_and_cast
    from dinov3.train.ssl_meta_arch import SSLMetaArch
    from dinov3.train.train import apply_optim_scheduler, build_optimizer, build_schedulers

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(int(stage1_cfg["seed"]))
    torch.cuda.manual_seed_all(int(stage1_cfg["seed"]))

    cfg = get_default_config()
    cfg.train.output_dir = str(Path(stage1_cfg["output_dir"]).expanduser().resolve())
    cfg.train.seed = int(stage1_cfg["seed"])
    cfg.train.batch_size_per_gpu = int(stage1_cfg["batch_size_per_gpu"])
    cfg.train.num_workers = int(stage1_cfg["num_workers"])
    cfg.train.cache_dataset = False
    cfg.train.checkpointing = False
    cfg.train.compile = False
    cfg.train.cudagraphs = False

    cfg.optim.epochs = int(stage1_cfg["epochs"])
    cfg.optim.lr = float(stage1_cfg["lr"])
    cfg.optim.min_lr = float(stage1_cfg["min_lr"])
    cfg.optim.warmup_epochs = int(stage1_cfg["warmup_epochs"])
    cfg.optim.weight_decay = float(stage1_cfg["weight_decay"])
    cfg.optim.weight_decay_end = float(stage1_cfg["weight_decay_end"])
    cfg.optim.scaling_rule = str(stage1_cfg["scaling_rule"])
    cfg.optim.clip_grad = float(stage1_cfg["clip_grad"])
    cfg.optim.freeze_last_layer_epochs = int(stage1_cfg["freeze_last_layer_epochs"])
    cfg.optim.adamw_beta1 = float(stage1_cfg["adamw_beta1"])
    cfg.optim.adamw_beta2 = float(stage1_cfg["adamw_beta2"])

    cfg.student.arch = str(stage1_cfg["student_arch"])
    cfg.student.patch_size = int(stage1_cfg["patch_size"])
    cfg.student.n_storage_tokens = int(stage1_cfg["n_storage_tokens"])
    cfg.student.drop_path_rate = float(stage1_cfg["drop_path_rate"])

    cfg.teacher.momentum_teacher = float(stage1_cfg["teacher_momentum"])
    cfg.teacher.final_momentum_teacher = float(stage1_cfg["teacher_final_momentum"])
    cfg.teacher.teacher_temp = float(stage1_cfg["teacher_temp"])
    cfg.teacher.warmup_teacher_temp = float(stage1_cfg["teacher_warmup_temp"])
    cfg.teacher.warmup_teacher_temp_epochs = int(stage1_cfg["teacher_warmup_temp_epochs"])

    cfg.crops.global_crops_scale = [float(x) for x in stage1_cfg["global_crops_scale"]]
    cfg.crops.local_crops_scale = [float(x) for x in stage1_cfg["local_crops_scale"]]
    cfg.crops.global_crops_size = int(stage1_cfg["global_crops_size"])
    cfg.crops.local_crops_size = int(stage1_cfg["local_crops_size"])
    cfg.crops.local_crops_number = int(stage1_cfg["local_crops_number"])
    cfg.compute_precision.param_dtype = str(stage1_cfg["param_dtype"])

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model first so we can reuse official DINOv3 augmentation from the model itself.
    model = SSLMetaArch(cfg)
    model.to_empty(device="cuda")
    model.init_weights()

    dino_init_path = Path(str(stage1_cfg["dino_init_weights"])).expanduser().resolve()
    if not dino_init_path.exists():
        raise FileNotFoundError(f"Initial DINO weights not found: {dino_init_path}")
    init_payload = torch.load(dino_init_path, map_location="cpu")
    backbone_state = sanitize_backbone_state_dict(init_payload)
    if len(backbone_state) == 0:
        raise RuntimeError(f"Cannot parse backbone weights from: {dino_init_path}")
    msg_student = model.student["backbone"].load_state_dict(backbone_state, strict=True)
    msg_teacher = model.model_ema["backbone"].load_state_dict(backbone_state, strict=True)
    print(f"Loaded init backbone into student: {msg_student}")
    print(f"Loaded init backbone into teacher: {msg_teacher}")

    transform = model.build_data_augmentation_dino(cfg)
    dataset = AnnotationImageDataset(
        annotations=str(stage1_cfg["annotations"]),
        image_root=stage1_cfg["image_root"],
        transform=transform,
    )

    if int(stage1_cfg["official_epoch_length"]) > 0:
        official_epoch_length = int(stage1_cfg["official_epoch_length"])
    else:
        official_epoch_length = max(math.ceil(len(dataset) / cfg.train.batch_size_per_gpu), 1)
    cfg.train.OFFICIAL_EPOCH_LENGTH = int(official_epoch_length)

    # Persist the resolved DINO config for reproducibility.
    write_config(cfg, str(output_dir), name="stage1_dinov3_resolved.yaml")

    img_size = cfg.crops.global_crops_size
    patch_size = int(cfg.student.patch_size * cfg.crops.teacher_to_student_resolution_scale)
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * (img_size // patch_size) * (img_size // patch_size),
    )
    collate_fn = lambda samples: collate_data_and_cast(
        samples_list=samples,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype={"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.compute_precision.param_dtype],
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        random_circular_shift=cfg.ibot.mask_random_circular_shift,
        local_batch_size=None,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(cfg.train.num_workers > 0),
    )
    loader_iter = cycle_loader(loader)

    optimizer = build_optimizer(cfg, model.get_params_groups())
    lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_schedulers(cfg)
    start_iter = try_resume_checkpoint(model, optimizer, output_dir, bool(stage1_cfg["resume"]))

    model.train()
    max_iter = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
    if start_iter >= max_iter:
        print(f"Training already finished (start_iter={start_iter}, max_iter={max_iter}).")
        return

    print("Stage1 training configuration:")
    print(f"  dataset size: {len(dataset)}")
    print(f"  batch_size_per_gpu: {cfg.train.batch_size_per_gpu}")
    print(f"  official_epoch_length: {cfg.train.OFFICIAL_EPOCH_LENGTH}")
    print(f"  epochs: {cfg.optim.epochs}")
    print(f"  max_iter: {max_iter}")
    print(f"  output_dir: {output_dir}")

    global_batch_size = cfg.train.batch_size_per_gpu
    current_epoch = start_iter // cfg.train.OFFICIAL_EPOCH_LENGTH
    for epoch in range(current_epoch, cfg.optim.epochs):
        epoch_start_iter = max(start_iter, epoch * cfg.train.OFFICIAL_EPOCH_LENGTH)
        epoch_end_iter = (epoch + 1) * cfg.train.OFFICIAL_EPOCH_LENGTH
        progress = tqdm(range(epoch_start_iter, epoch_end_iter), desc=f"Epoch {epoch + 1}/{cfg.optim.epochs}")

        epoch_loss = 0.0
        epoch_steps = 0

        for it in progress:
            data = next(loader_iter)
            data["global_batch_size"] = global_batch_size

            lr = lr_schedule[it]
            wd = wd_schedule[it]
            mom = momentum_schedule[it]
            teacher_temp = teacher_temp_schedule[it]
            last_layer_lr = last_layer_lr_schedule[it]
            apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

            optimizer.zero_grad(set_to_none=True)
            total_loss, metrics_dict = model.forward_backward(data, teacher_temp=teacher_temp, iteration=it)

            if cfg.optim.clip_grad:
                for _, student_module in model.student.items():
                    torch.nn.utils.clip_grad_norm_(student_module.parameters(), max_norm=cfg.optim.clip_grad)

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at iter={it}: {float(total_loss.detach().item())}")

            optimizer.step()
            model.update_ema(mom)

            loss_val = float(total_loss.detach().item())
            epoch_loss += loss_val
            epoch_steps += 1
            progress.set_postfix(
                loss=f"{loss_val:.4f}",
                lr=f"{float(lr):.2e}",
                wd=f"{float(wd):.3f}",
                temp=f"{float(teacher_temp):.4f}",
            )

            _ = metrics_dict  # Keep call contract explicit.

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"[Epoch {epoch + 1}] avg_loss={avg_loss:.4f}")

        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            output_dir=output_dir,
            iteration=epoch_end_iter - 1,
            epoch=epoch + 1,
        )
        export_teacher_backbone(model, output_dir / f"teacher_backbone_epoch_{epoch + 1}.pth")
        export_teacher_backbone(model, output_dir / "teacher_backbone_latest.pth")

    print("Stage1 training completed.")


if __name__ == "__main__":
    main()
