from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class MultiheadAttentionPooling(nn.Module):
    """Cross-attention pooling with a learnable query token."""

    def __init__(self, embed_dim: int, num_heads: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError(f"Expected patch_tokens shape [B, N, D], got {tuple(patch_tokens.shape)}")
        batch_size = patch_tokens.size(0)
        query = self.query_token.expand(batch_size, -1, -1)
        attn_out, _ = self.attn(query, patch_tokens, patch_tokens, need_weights=False)
        return attn_out.squeeze(1)


class DINOv3VisionEncoder(nn.Module):
    def __init__(
        self,
        dinov3_repo_path: str | Path | None = None,
        dino_weights_path: str | Path | None = None,
        freeze_backbone: bool = True,
        map_num_heads: int = 16,
        map_dropout: float = 0.0,
        normalize_by_default: bool = False,
    ) -> None:
        super().__init__()
        if dino_weights_path is None:
            raise ValueError("dino_weights_path is required. Please provide a local DINOv3 checkpoint path.")

        repo_path = Path(dinov3_repo_path) if dinov3_repo_path is not None else Path(__file__).resolve().parents[1] / "dinov3"
        weights_path = Path(dino_weights_path).expanduser().resolve()

        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo path not found: {repo_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights_path}")

        self.backbone = self._load_backbone_from_local_repo(repo_path=repo_path, weights_path=weights_path)
        self.freeze_backbone = freeze_backbone
        self.normalize_by_default = normalize_by_default

        self.embed_dim = int(self.backbone.embed_dim)
        self.map_pool = MultiheadAttentionPooling(
            embed_dim=self.embed_dim,
            num_heads=map_num_heads,
            dropout=map_dropout,
        )

        if self.freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _load_backbone_from_local_repo(repo_path: Path, weights_path: Path) -> nn.Module:
        repo_path = repo_path.resolve()
        package_dir = repo_path / "dinov3"
        if not package_dir.exists():
            raise FileNotFoundError(f"Expected package directory not found: {package_dir}")

        repo_str = str(repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        backbones = importlib.import_module("dinov3.hub.backbones")
        dinov3_vitl16 = getattr(backbones, "dinov3_vitl16")
        return dinov3_vitl16(pretrained=True, weights=str(weights_path))

    def _extract_tokens(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(images)
        token_sequence = features["x_prenorm"]  # [B, 1 + R + N, D]

        register_count = int(getattr(self.backbone, "n_storage_tokens", 0))
        global_cls = token_sequence[:, 0, :]
        patch_tokens = token_sequence[:, 1 + register_count :, :]
        return global_cls, patch_tokens

    def forward(
        self,
        images: torch.Tensor,
        normalize: bool | None = None,
        return_patch_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        do_normalize = self.normalize_by_default if normalize is None else normalize

        if self.freeze_backbone:
            with torch.no_grad():
                global_cls, patch_tokens = self._extract_tokens(images)
        else:
            global_cls, patch_tokens = self._extract_tokens(images)

        map_feature = self.map_pool(patch_tokens)

        if do_normalize:
            global_cls = F.normalize(global_cls, dim=-1)
            map_feature = F.normalize(map_feature, dim=-1)

        output: dict[str, Any] = {
            "global_cls": global_cls,
            "map_feature": map_feature,
        }
        if return_patch_tokens:
            output["patch_tokens"] = patch_tokens
        return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for DINOv3VisionEncoder")
    parser.add_argument("--dinov3-repo", type=str, required=True, help="Path to local dinov3 repo")
    parser.add_argument("--dino-weights", type=str, required=True, help="Path to local dinov3_vitl16 checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    encoder = DINOv3VisionEncoder(
        dinov3_repo_path=args.dinov3_repo,
        dino_weights_path=args.dino_weights,
        freeze_backbone=True,
        normalize_by_default=False,
    ).to(device)

    dummy_images = torch.randn(2, 3, 224, 224, device=device)

    output_default = encoder(dummy_images)
    output_norm_false = encoder(dummy_images, normalize=False)
    output_norm_true = encoder(dummy_images, normalize=True)

    print("default global_cls:", tuple(output_default["global_cls"].shape))
    print("default map_feature:", tuple(output_default["map_feature"].shape))
    print("default patch_tokens:", tuple(output_default["patch_tokens"].shape))
    print("normalize=False global_cls:", tuple(output_norm_false["global_cls"].shape))
    print("normalize=True global_cls:", tuple(output_norm_true["global_cls"].shape))

    assert output_default["global_cls"].shape[0] == 2
    assert output_default["map_feature"].shape[0] == 2
    assert output_default["patch_tokens"].shape[0] == 2
    assert output_default["patch_tokens"].shape[1] == 196  # 224x224 with patch size 16
    assert output_default["global_cls"].shape[-1] == output_default["map_feature"].shape[-1]

    print("Vision encoder forward check passed.")
