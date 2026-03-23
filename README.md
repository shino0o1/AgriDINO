This repository contains the core code implementation and dataset of the AgriDINO paper:

**"A dual-branch vision-language framework with taxonomy-aware alignment for fine-grained crop pest and disease identification"**

## Project Structure

- `data/` dataset loading and image transforms
- `models/` vision encoder, text encoder, and AgriDINO dual-branch model
- `utils/` taxonomy-aware soft-target contrastive loss
- `scripts/` training scripts
- `config.yaml` training hyperparameters and paths

## Environment

Use the provided conda file:

```bash
conda env create -f conda.yaml
conda activate agridino
```

## Dataset

dataset links here:

- Sample dataset Download (Cloud Drive): **[[TBD_LINK_1](https://drive.google.com/file/d/1aeePQgSCjCuFg-6xfJQ3gdutwvgV_YUH/view?usp=sharing)]**

Expected annotation fields include:

- `image_path` (or `f_path`)
- `short_description`
- `long_description`
- labels:
  - either `Crop_ID` / `Disease_ID`
  - or `crop_class` / `pest_disease_class` (auto-encoded by `AgricapDataset`)

## Training (Stage 2)

1. Edit `config.yaml` with your local paths (`annotations`, `image_root`, `dino_weights`).
2. Run:

```bash
python scripts/train_stage2.py --config config.yaml

```

## Training (Stage 1 SSL)

1. Fill `stage1.annotations`, `stage1.image_root`, and `stage1.dino_init_weights` in `config.yaml`.
2. Run:

```bash
python scripts/train_stage1.py --config config.yaml
```

Outputs are saved to `stage1.output_dir`, including:

- `checkpoints/latest.pt` (resume training)
- `teacher_backbone_latest.pth` (directly loadable by `models/vision.py` as `dino_weights_path`)

## Inference and Evaluation

```bash
python scripts/infer.py --config config.yaml --checkpoint <stage2_ckpt.pt> --task retrieval
```

### 1 Image-Text Retrieval

Evaluate image-to-text and text-to-image retrieval with Recall@K:

```bash
python scripts/infer.py \
  --config config.yaml \
  --checkpoint outputs/stage2/latest.pt \
  --task retrieval \
  --branch avg \
  --topk 1,5,10
```

Notes:

- `--branch` controls which representation is used:
  - `global`: CLS/global branch only
  - `fine`: MAP/fine-grained branch only
  - `avg`: average of global and fine branch (then L2-normalized)
- Ground-truth pairing assumes the i-th image matches the i-th text in the same annotation order.

### 2 Zero-shot Classification

Use text prototypes as class prompts and classify images by max similarity:

```bash
python scripts/infer.py \
  --config config.yaml \
  --checkpoint outputs/stage2/latest.pt \
  --task zeroshot \
  --branch avg \
  --class-text-source short
```

`--class-text-source` options:

- `short`: use `short_description` as class prototype text
- `long`: use `long_description` as class prototype text
- `id2label`: load external class text mapping via `--id2label <json>`

Example `id2label` file:

```json
{
  "0": "downy mildew",
  "1": "black rot",
  "2": "flea beetle"
}
```

### 3 Common Inference Arguments

- `--annotations`: override annotation path from `config.yaml`
- `--image-root`: override image root path
- `--batch-size`: inference batch size
- `--num-workers`: dataloader workers
- `--device`: `cuda`

### 4 Expected Outputs

- Retrieval mode prints:
  - similarity matrix shape
  - Recall@1/5/10 for both i2t and t2i
- Zero-shot mode prints:
  - top-1 classification accuracy
  - number of classes and text source used

## Notes

- AMP training is enabled automatically on CUDA.
- Checkpoints are saved to `output_dir` from `config.yaml`.
