from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from models.text import AgricultureTextEncoder
    from models.vision import DINOv3VisionEncoder
except ImportError:
    from text import AgricultureTextEncoder  # type: ignore
    from vision import DINOv3VisionEncoder  # type: ignore


class AgriDINO(nn.Module):
    def __init__(
        self,
        *,
        dino_weights_path: str | Path,
        dinov3_repo_path: str | Path | None = None,
        text_model_name_or_path: str = "recobo/agriculture-bert-uncased",
        max_length: int = 256,
        freeze_text_backbone: bool = False,
        logit_scale_max: float = 100.0,
    ) -> None:
        super().__init__()

        self.vision_encoder = DINOv3VisionEncoder(
            dinov3_repo_path=dinov3_repo_path,
            dino_weights_path=dino_weights_path,
            freeze_backbone=True,
            normalize_by_default=False,
        )
        self.text_encoder = AgricultureTextEncoder(
            model_name_or_path=text_model_name_or_path,
            max_length=max_length,
            freeze_backbone=freeze_text_backbone,
        )

        self.vision_dim = int(self.vision_encoder.embed_dim)
        self.text_dim = int(self.text_encoder.hidden_size)

        self.text_proj_g = nn.Linear(self.text_dim, self.vision_dim, bias=True)
        self.text_proj_f = nn.Linear(self.text_dim, self.vision_dim, bias=True)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_max = float(logit_scale_max)

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vision_outputs = self.vision_encoder(images, normalize=False, return_patch_tokens=False)
        v_g = F.normalize(vision_outputs["global_cls"], p=2, dim=-1)
        v_f = F.normalize(vision_outputs["map_feature"], p=2, dim=-1)
        return v_g, v_f

    def encode_text(
        self,
        short_texts: str | list[str],
        long_texts: str | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_outputs = self.text_encoder(short_text=short_texts, long_text=long_texts)
        t_g = self.text_proj_g(text_outputs["short_text_feature"])
        t_f = self.text_proj_f(text_outputs["long_text_feature"])
        t_g = F.normalize(t_g, p=2, dim=-1)
        t_f = F.normalize(t_f, p=2, dim=-1)
        return t_g, t_f

    def forward(
        self,
        images: torch.Tensor,
        short_texts: str | list[str],
        long_texts: str | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_g, v_f = self.encode_image(images)
        t_g, t_f = self.encode_text(short_texts, long_texts)

        scale = self.logit_scale.exp().clamp(max=self.logit_scale_max)
        sim_g = (v_g @ t_g.T) * scale
        sim_f = (v_f @ t_f.T) * scale
        return sim_g, sim_f


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for AgriDINO")
    parser.add_argument("--dino-weights", type=str, required=True, help="Path to local dinov3_vitl16 checkpoint")
    parser.add_argument("--dinov3-repo", type=str, default=None, help="Path to local dinov3 repo")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    model = AgriDINO(
        dino_weights_path=args.dino_weights,
        dinov3_repo_path=args.dinov3_repo,
    ).to(device)

    dummy_images = torch.randn(2, 3, 224, 224, device=device)
    dummy_short_texts = [
        "tomato early blight",
        "apple scab",
    ]
    dummy_long_texts = [
        "Tomato leaves show concentric necrotic spots and yellowing.",
        "Apple leaves and fruits show olive-brown lesions with rough texture.",
    ]

    with torch.no_grad():
        v_g, v_f = model.encode_image(dummy_images)
        t_g, t_f = model.encode_text(dummy_short_texts, dummy_long_texts)
        sim_g, sim_f = model(dummy_images, dummy_short_texts, dummy_long_texts)

    scale_raw = model.logit_scale.exp().item()
    scale_clamped = model.logit_scale.exp().clamp(max=model.logit_scale_max).item()

    print("encode_image v_g shape:", tuple(v_g.shape))
    print("encode_image v_f shape:", tuple(v_f.shape))
    print("encode_text t_g shape:", tuple(t_g.shape))
    print("encode_text t_f shape:", tuple(t_f.shape))
    print("sim_g shape:", tuple(sim_g.shape))
    print("sim_f shape:", tuple(sim_f.shape))
    print("logit_scale.exp():", scale_raw)
    print("clamped scale:", scale_clamped)

    assert tuple(v_g.shape) == (2, model.vision_dim)
    assert tuple(v_f.shape) == (2, model.vision_dim)
    assert tuple(t_g.shape) == (2, model.vision_dim)
    assert tuple(t_f.shape) == (2, model.vision_dim)
    assert tuple(sim_g.shape) == (2, 2)
    assert tuple(sim_f.shape) == (2, 2)
    assert scale_clamped > 0.0
    assert scale_clamped <= model.logit_scale_max

    print("AgriDINO forward/encode checks passed.")

