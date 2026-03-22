from __future__ import annotations

import csv
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    from data.transforms import (
        build_stage1_dino_multicrop_transform,
        build_stage2_clip_transform,
    )
except ImportError:
    from transforms import (  # type: ignore
        build_stage1_dino_multicrop_transform,
        build_stage2_clip_transform,
    )


DEFAULT_COLUMN_CANDIDATES = {
    "image_path": ("image_path", "f_path", "image", "img_path", "path"),
    "short_description": ("short_description", "short_desc", "class_name", "category_name"),
    "long_description": ("long_description", "long_desc", "symptom_description", "description"),
    "crop_id": ("Crop_ID", "crop_id"),
    "disease_id": ("Disease_ID", "disease_id"),
    "crop_class": ("crop_class", "crop_name"),
    "pest_disease_class": ("pest_disease_class", "disease_class", "pest_disease_name"),
}


class AgricapDataset(Dataset):
    def __init__(
        self,
        annotations: str | Path | list[dict[str, Any]],
        image_root: str | Path | None = None,
        stage: str = "stage1",
        transform=None,
        column_map: dict[str, str] | None = None,
    ) -> None:
        self.stage = stage.lower()
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError(f"Unsupported stage: {stage}. Expected one of ['stage1', 'stage2'].")

        self.image_root = Path(image_root) if image_root is not None else None
        self.column_map = column_map or {}
        raw_samples = self._load_annotations(annotations)
        self.samples = self._normalize_samples(raw_samples)

        if transform is not None:
            self.transform = transform
        elif self.stage == "stage1":
            self.transform = build_stage1_dino_multicrop_transform(
                global_crops_scale=(0.4, 1.0),
                local_crops_scale=(0.05, 0.4),
                global_crops_number=2,
                local_crops_number=8,
                global_crop_size=224,
                local_crop_size=96,
            )
        else:
            self.transform = build_stage2_clip_transform(image_size=224)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample["image_path"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image at index={index}, path='{image_path}'.") from exc

        transformed = self.transform(image)
        output = {
            "image_path": str(image_path),
            "short_description": sample["short_description"],
            "long_description": sample["long_description"],
            "crop_id": sample["crop_id"],
            "disease_id": sample["disease_id"],
        }

        if self.stage == "stage1":
            if not isinstance(transformed, dict):
                raise TypeError("Stage1 transform must return a dict with 'global_crops' and 'local_crops'.")
            output["global_crops"] = transformed["global_crops"]
            output["local_crops"] = transformed["local_crops"]
        else:
            if isinstance(transformed, dict):
                if "image" not in transformed:
                    raise KeyError("Stage2 transform returned dict but missing 'image' key.")
                output["image"] = transformed["image"]
            else:
                output["image"] = transformed

        return output

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        return path

    def _load_annotations(self, annotations: str | Path | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(annotations, list):
            return annotations

        ann_path = Path(annotations)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        suffix = ann_path.suffix.lower()
        if suffix == ".csv":
            with ann_path.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        if suffix == ".jsonl":
            rows = []
            with ann_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no}: {ann_path}") from exc
            return rows

        raise ValueError(f"Unsupported annotation format: {ann_path}. Use .csv or .jsonl")

    def _resolve_column(self, sample: dict[str, Any], canonical_field: str) -> str:
        if canonical_field in self.column_map:
            mapped = self.column_map[canonical_field]
            if mapped not in sample:
                raise KeyError(
                    f"column_map[{canonical_field!r}]={mapped!r} not found in sample keys: {list(sample.keys())}"
                )
            return mapped

        if canonical_field in sample:
            return canonical_field

        for candidate in DEFAULT_COLUMN_CANDIDATES[canonical_field]:
            if candidate in sample:
                return candidate

        raise KeyError(
            f"Missing field '{canonical_field}' in sample. Available keys: {list(sample.keys())}. "
            f"Try passing column_map."
        )

    def _normalize_samples(self, raw_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        crop_class_to_id: dict[str, int] = {}
        disease_class_to_id: dict[str, int] = {}

        for idx, sample in enumerate(raw_samples):
            if not isinstance(sample, dict):
                raise TypeError(f"Annotation at index={idx} must be dict, got {type(sample).__name__}.")

            image_key = self._resolve_column(sample, "image_path")
            short_key = self._resolve_column(sample, "short_description")
            long_key = self._resolve_column(sample, "long_description")

            crop_key = None
            disease_key = None
            crop_class_key = None
            disease_class_key = None
            try:
                crop_key = self._resolve_column(sample, "crop_id")
            except KeyError:
                crop_class_key = self._resolve_column(sample, "crop_class")
            try:
                disease_key = self._resolve_column(sample, "disease_id")
            except KeyError:
                disease_class_key = self._resolve_column(sample, "pest_disease_class")

            try:
                if crop_key is not None:
                    crop_id = int(sample[crop_key])
                else:
                    crop_name = self._normalize_category_name(str(sample[crop_class_key]))
                    if crop_name not in crop_class_to_id:
                        crop_class_to_id[crop_name] = len(crop_class_to_id)
                    crop_id = crop_class_to_id[crop_name]

                if disease_key is not None:
                    disease_id = int(sample[disease_key])
                else:
                    disease_name = self._normalize_category_name(str(sample[disease_class_key]))
                    if disease_name not in disease_class_to_id:
                        disease_class_to_id[disease_name] = len(disease_class_to_id)
                    disease_id = disease_class_to_id[disease_name]

                normalized.append(
                    {
                        "image_path": str(sample[image_key]).strip(),
                        "short_description": str(sample[short_key]),
                        "long_description": str(sample[long_key]),
                        "crop_id": crop_id,
                        "disease_id": disease_id,
                    }
                )
            except Exception as exc:
                raise ValueError(f"Invalid annotation at index={idx}: {sample}") from exc

        return normalized

    @staticmethod
    def _normalize_category_name(name: str) -> str:
        # Lowercase and remove special characters except spaces.
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


def _make_fake_image(path: Path, size: tuple[int, int]) -> None:
    array = np.random.randint(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _write_fake_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["image_path", "short_description", "long_description", "Crop_ID", "Disease_ID"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_fake_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        image_dir = root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        _make_fake_image(image_dir / "img_0.jpg", (640, 480))
        _make_fake_image(image_dir / "img_1.jpg", (800, 600))

        base_rows = [
            {
                "image_path": "img_0.jpg",
                "short_description": "tomato early blight",
                "long_description": "Leaf spots with concentric rings and yellow halo.",
                "Crop_ID": 1,
                "Disease_ID": 11,
            },
            {
                "image_path": "img_1.jpg",
                "short_description": "apple scab",
                "long_description": "Dark olive lesions on leaves and fruit.",
                "Crop_ID": 2,
                "Disease_ID": 22,
            },
        ]

        csv_path = root / "samples.csv"
        jsonl_path = root / "samples.jsonl"
        _write_fake_csv(csv_path, base_rows)
        _write_fake_jsonl(jsonl_path, base_rows)

        # Stage1 test (CSV)
        stage1_dataset = AgricapDataset(
            annotations=csv_path,
            image_root=image_dir,
            stage="stage1",
        )
        stage1_sample = stage1_dataset[0]
        print("Stage1 keys:", stage1_sample.keys())
        print("Stage1 global crop count:", len(stage1_sample["global_crops"]))
        print("Stage1 local crop count:", len(stage1_sample["local_crops"]))
        print("Stage1 global[0] shape:", tuple(stage1_sample["global_crops"][0].shape))
        print("Stage1 local[0] shape:", tuple(stage1_sample["local_crops"][0].shape))
        assert len(stage1_sample["global_crops"]) == 2
        assert len(stage1_sample["local_crops"]) == 8
        assert tuple(stage1_sample["global_crops"][0].shape) == (3, 224, 224)
        assert tuple(stage1_sample["local_crops"][0].shape) == (3, 96, 96)
        assert isinstance(stage1_sample["crop_id"], int)
        assert isinstance(stage1_sample["disease_id"], int)

        # Stage2 test (JSONL)
        stage2_dataset = AgricapDataset(
            annotations=jsonl_path,
            image_root=image_dir,
            stage="stage2",
        )
        stage2_sample = stage2_dataset[0]
        print("Stage2 keys:", stage2_sample.keys())
        print("Stage2 image shape:", tuple(stage2_sample["image"].shape))
        assert tuple(stage2_sample["image"].shape) == (3, 224, 224)
        assert isinstance(stage2_sample["short_description"], str)
        assert isinstance(stage2_sample["long_description"], str)

        print("All dataset checks passed.")
