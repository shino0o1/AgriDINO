# AgriDINO (Reproduction)

This repository contains the code implementation and dataset of the AgriDINO paper:

**"AgriDINO: A DINO-centric fine-grained vision-language model for identifying agricultural pests and diseases."**

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

Please place your dataset links here:

- Dataset Download (Cloud Drive): **[TBD_LINK_1]**
- Mirror / Backup Link: **[TBD_LINK_2]**

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

You can override config values from CLI, for example:

```bash
python scripts/train_stage2.py --config config.yaml --epochs 5 --batch-size 8
```

## Inference / Evaluation

This repo includes `scripts/infer_stage2.py` for:

- Zero-shot disease classification
- Image-text retrieval (Recall@K)

### 1) Retrieval

```bash
python scripts/infer_stage2.py \
  --config config.yaml \
  --checkpoint outputs/stage2/latest.pt \
  --task retrieval \
  --branch avg \
  --topk 1,5,10
```

### 2) Zero-shot classification

```bash
python scripts/infer_stage2.py \
  --config config.yaml \
  --checkpoint outputs/stage2/latest.pt \
  --task zeroshot \
  --branch avg \
  --class-text-source short
```

Optional: provide custom class names via JSON:

```json
{
  "0": "black rot",
  "1": "flea beetle"
}
```

Then run with:

```bash
python scripts/infer_stage2.py --config config.yaml --checkpoint outputs/stage2/latest.pt --task zeroshot --class-text-source id2label --id2label path/to/id2label.json
```

## Notes

- AMP training is enabled automatically on CUDA.
- Checkpoints are saved to `output_dir` from `config.yaml`.
- This repo focuses on practical reproducibility rather than full production packaging.
