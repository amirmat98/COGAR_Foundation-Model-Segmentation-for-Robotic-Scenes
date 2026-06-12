# Isaac Sim Official Unitree G1 Dataset Generator

This folder contains the Isaac Sim generator for the main synthetic dataset.

Current decision:

- Target size: 1000 images.
- Robot: official Unitree G1 USD asset only.
- No surrogate robot fallback for final data.
- Run on AWS or another NVIDIA GPU machine.

Main files:

| Path | Purpose |
| --- | --- |
| `configs/dataset_config_v3_official_g1.json` | 1000-image official-G1 dataset config |
| `src/robotic_sdg/generate_dataset_v2.py` | Isaac Sim Replicator generator |
| `scripts/run_isaac_dataset_v3_official_g1_container.sh` | Strict official-G1 container wrapper |
| `scripts/validate_dataset_preview.py` | Dataset smoke/full validation |
| `scripts/export_isaac_to_coco.py` | COCO instance export |
| `docs/dataset_v3_official_g1_runbook.md` | Current run commands |
| `docs/aws_ssh.md` | AWS SSH setup notes |

Generated data belongs under `data/` and is ignored by Git.
