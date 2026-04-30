# COGAR Foundation-Model Segmentation for Robotic Scenes

This repository contains a reproducible Python benchmark pipeline for
foundation-model segmentation in robotic scenes. The current implemented
pipeline uses the OCID Object Clutter Indoor Dataset and SAM ViT-B with
single-object bounding-box prompts.

The larger COGAR assignment targets zero-shot segmentation benchmarks for SAM,
SAM2, FastSAM, MobileSAM, EfficientSAM, and classical baselines on robotic
scenes with clutter, occlusion, reflective or transparent objects, small parts,
and dynamic scene changes.

## Current Pipeline

- OCID path configuration through `configs/paths.yaml`
- Image-level indexing for `YCB10/table/top/mixed/seq21`
- Object-level indexing and filtering
- Binary ground-truth mask export
- Single-object SAM box-prompt inference
- Batch SAM box-prompt inference over the filtered OCID objects
- Saved predicted masks, visualizations, and IoU results
- CUDA support with CPU fallback for local execution

## Repository Structure

```text
configs/
  paths.yaml
scripts/
  prepare_ocid_debug_dataset.py
  visualize_object_prompt.py
  visualize_binary_gt_mask.py
  run_sam_box_prompt.py
  run_sam_box_prompt_batch.py
  analyze_sam_results.py
src/cogar_seg/
  config.py
  paths.py
  io.py
  datasets/ocid.py
  prompts/boxes.py
  models/sam.py
  metrics/segmentation.py
  visualization/masks.py
  evaluation/sam_box_eval.py
tests/
  test_paths.py
  test_metrics.py
  test_ocid_index.py
```

`scripts/` contains command-line entry points. Reusable code lives under
`src/cogar_seg/`, which makes it easier to add more datasets, prompt types,
models, metrics, plots, and evaluation workflows.

## Ignored Local Assets

Large or generated assets are intentionally ignored by Git:

- `datasets/`, `data/`, `external_data/`, `Raw_Dataset/`, `OCID/`
- `checkpoints/`
- `outputs/`, `results/`, `runs/`
- `*.pth`, `*.pt`
- `.venv/`

Keep OCID data, SAM checkpoints, and generated masks/results in those local
folders or in external storage.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you do not install the package in editable mode, run commands with
`PYTHONPATH=src`.

## Configuration

Edit `configs/paths.yaml` for local paths:

```yaml
ocid_root: "/path/to/OCID-dataset"
ocid_debug_sequence: "/path/to/OCID-dataset/YCB10/table/top/mixed/seq21"
outputs_dir: "outputs"
sam_outputs_dir: "outputs/sam_box_prompt"
```

The code preserves compatibility with existing CSVs that contain older absolute
OCID paths by remapping paths below the `OCID-dataset/` component to the current
configured `ocid_root`.

## Commands

Prepare the OCID debug indexes and binary ground-truth masks:

```bash
PYTHONPATH=src python scripts/prepare_ocid_debug_dataset.py
```

Visualize prompts and binary masks:

```bash
PYTHONPATH=src python scripts/visualize_object_prompt.py 0
PYTHONPATH=src python scripts/visualize_binary_gt_mask.py 5
```

Run SAM on one object:

```bash
PYTHONPATH=src python scripts/run_sam_box_prompt.py \
  --row 0 \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

Run SAM over the filtered OCID object index:

```bash
PYTHONPATH=src python scripts/run_sam_box_prompt_batch.py \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

Summarize the results CSV and print the worst IoU rows:

```bash
PYTHONPATH=src python scripts/analyze_sam_results.py
```

## Tests

The default tests avoid dataset and checkpoint dependencies:

```bash
PYTHONPATH=src pytest -q
```

SAM smoke tests require local OCID data and a SAM checkpoint.

## Extension Points

- Add datasets under `src/cogar_seg/datasets/`.
- Add prompt builders under `src/cogar_seg/prompts/`.
- Add model adapters under `src/cogar_seg/models/`.
- Add metrics under `src/cogar_seg/metrics/`.
- Add plots and visual reports under `src/cogar_seg/visualization/`.
- Add benchmark workflows under `src/cogar_seg/evaluation/`.
