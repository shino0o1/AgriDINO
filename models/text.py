from __future__ import annotations

import argparse
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class AgricultureTextEncoder(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "recobo/agriculture-bert-uncased",
        max_length: int = 256,
        freeze_backbone: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.hidden_size = int(self.backbone.config.hidden_size)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

    @staticmethod
    def _ensure_text_list(text: str | list[str], field_name: str) -> list[str]:
        if isinstance(text, str):
            return [text]
        if isinstance(text, list) and all(isinstance(item, str) for item in text):
            return text
        raise TypeError(f"{field_name} must be str or list[str].")

    def _to_model_device(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model_device = next(self.backbone.parameters()).device
        return {k: v.to(model_device) for k, v in inputs.items()}

    def _tokenize(self, text: str | list[str]) -> dict[str, torch.Tensor]:
        text_list = self._ensure_text_list(text, "text")
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return self._to_model_device(encoded)

    def _encode_with_cls(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        return last_hidden[:, 0, :]  # CLS pooling

    def _resolve_inputs(
        self,
        raw_text: str | list[str] | None,
        tokenized_inputs: dict[str, torch.Tensor] | None,
        name: str,
    ) -> dict[str, torch.Tensor]:
        if tokenized_inputs is not None:
            return self._to_model_device(tokenized_inputs)
        if raw_text is not None:
            return self._tokenize(raw_text)
        raise ValueError(f"Either raw text or tokenized inputs must be provided for {name}.")

    def forward(
        self,
        short_text: str | list[str] | None = None,
        long_text: str | list[str] | None = None,
        short_inputs: dict[str, torch.Tensor] | None = None,
        long_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        short_batch = self._resolve_inputs(short_text, short_inputs, "short_text")
        long_batch = self._resolve_inputs(long_text, long_inputs, "long_text")

        short_text_feature = self._encode_with_cls(short_batch)
        long_text_feature = self._encode_with_cls(long_batch)

        return {
            "short_text_feature": short_text_feature,
            "long_text_feature": long_text_feature,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test for AgricultureTextEncoder")
    parser.add_argument("--model-name", type=str, default="recobo/agriculture-bert-uncased")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer from local cache only")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    encoder = AgricultureTextEncoder(
        model_name_or_path=args.model_name,
        freeze_backbone=False,
        local_files_only=args.local_files_only,
    ).to(device)

    short_text = [
        "tomato early blight",
        "apple scab",
    ]
    long_text = [
        "Tomato leaves show concentric necrotic spots and yellowing.",
        "Apple leaves and fruits show olive-brown lesions with rough texture.",
    ]

    # Path 1: raw strings
    output_raw = encoder(short_text=short_text, long_text=long_text)
    print("raw short_text_feature:", tuple(output_raw["short_text_feature"].shape))
    print("raw long_text_feature:", tuple(output_raw["long_text_feature"].shape))

    # Path 2: pre-tokenized inputs
    short_inputs = encoder.tokenizer(
        short_text,
        padding=True,
        truncation=True,
        max_length=encoder.max_length,
        return_tensors="pt",
    )
    long_inputs = encoder.tokenizer(
        long_text,
        padding=True,
        truncation=True,
        max_length=encoder.max_length,
        return_tensors="pt",
    )
    output_tokenized = encoder(short_inputs=short_inputs, long_inputs=long_inputs)
    print("tokenized short_text_feature:", tuple(output_tokenized["short_text_feature"].shape))
    print("tokenized long_text_feature:", tuple(output_tokenized["long_text_feature"].shape))

    assert output_raw["short_text_feature"].shape[0] == len(short_text)
    assert output_raw["long_text_feature"].shape[0] == len(long_text)
    assert output_raw["short_text_feature"].shape[-1] == encoder.hidden_size
    assert output_raw["long_text_feature"].shape[-1] == encoder.hidden_size
    assert output_tokenized["short_text_feature"].shape == output_raw["short_text_feature"].shape
    assert output_tokenized["long_text_feature"].shape == output_raw["long_text_feature"].shape

    print("Text encoder forward check passed.")

