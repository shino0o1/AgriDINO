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

dataset links here:

- Sample dataset Download (Cloud Drive): **[TBD_LINK_1]**

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

## Notes

- AMP training is enabled automatically on CUDA.
- Checkpoints are saved to `output_dir` from `config.yaml`.