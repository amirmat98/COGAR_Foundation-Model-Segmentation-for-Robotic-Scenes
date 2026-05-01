# COGAR Foundation-Model Segmentation for Robotic Scenes

This repository contains a reproducible Python benchmark pipeline for evaluating
foundation-model segmentation in robotic scenes.

The current implemented pipeline uses the OCID Object Clutter Indoor Dataset and
SAM ViT-B to evaluate single-object segmentation with two prompt types:

- Bounding-box prompts
- Single positive point prompts

The benchmark compares prompt quality on the same filtered OCID object instances
using binary ground-truth masks and Intersection over Union (IoU).

The larger COGAR assignment targets zero-shot segmentation benchmarks for SAM,
SAM2, FastSAM, MobileSAM, EfficientSAM, and classical baselines on robotic scenes
with clutter, occlusion, reflective or transparent objects, small parts, and
dynamic scene changes.

## Current Pipeline

- OCID path configuration through `configs/paths.yaml`
- Image-level indexing for `YCB10/table/top/mixed/seq21`
- Object-level indexing and filtering
- Binary ground-truth mask export
- Single-object SAM box-prompt inference
- Batch SAM box-prompt evaluation
- Box-prompt quantitative analysis
- Single-object SAM point-prompt inference
- Batch SAM point-prompt evaluation
- Generic prompt-result analysis
- Box-vs-point prompt comparison
- Saved predicted masks, visualizations, IoU results, summaries, and plots
- CUDA support with CPU fallback for local execution

## Main Result on the OCID Debug Subset

The current benchmark was run on 52 filtered object instances from:

```text
YCB10/table/top/mixed/seq21
```

Using SAM ViT-B, the box-prompt and point-prompt results were:

| Prompt type | Mean IoU | Wins |
|---|---:|---:|
| Box prompt | 0.8495 | 41 / 52 |
| Single positive point prompt | 0.7282 | 11 / 52 |

The average IoU delta was:

```text
point prompt - box prompt = -0.1214
```

This indicates that single positive point prompts are often effective, but less
stable than bounding-box prompts in cluttered robotic scenes.

The strongest failure case was `object_id = 10`, where point-prompt IoU dropped
to around `0.11`, while box-prompt IoU stayed around `0.89`.

## Repository Structure

```text
configs/
  paths.yaml
  paths.example.yaml
  sim_dataset.yaml

data/
  README.md

outputs/
  README.md

scripts/
  prepare_ocid_debug_dataset.py
  visualize_object_prompt.py
  visualize_binary_gt_mask.py

  run_sam_box_prompt.py
  run_sam_box_prompt_batch.py
  run_sam_point_prompt.py
  run_sam_point_prompt_batch.py

  analyze_sam_results.py
  analyze_prompt_results.py
  compare_prompt_results.py
  generate_sim_dataset_pilot.py

src/cogar_seg/
  config.py
  paths.py
  io.py

  datasets/
    ocid.py
    sim_robotic.py

  prompts/
    boxes.py
    points.py

  models/
    sam.py

  metrics/
    segmentation.py

  visualization/
    masks.py

  evaluation/
    sam_box_eval.py
    sam_point_eval.py

tests/
  test_paths.py
  test_metrics.py
  test_ocid_index.py
  test_sim_robotic_dataset.py

docs/
  roadmap.md
  dataset_plan.md
  ocid_sam_box_prompt_failure_analysis.md
  templates/
```

`scripts/` contains command-line entry points. Reusable code lives under
`src/cogar_seg/`, which makes it easier to add more datasets, prompt types,
models, metrics, plots, and evaluation workflows.

## Ignored Local Assets

Large or generated assets are intentionally ignored by Git:

- `datasets/`, `external_data/`, `Raw_Dataset/`, `OCID/`
- `data/*` except `data/README.md`
- `checkpoints/`
- `outputs/*` except `outputs/README.md`
- `results/`, `runs/`
- `*.pth`, `*.pt`
- `.venv/`

Keep OCID data, SAM checkpoints, generated masks, results, summaries, and plots
in those local folders or in external storage.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

If you do not install the package in editable mode, run commands with:

```bash
PYTHONPATH=src
```

## Configuration

Copy `configs/paths.example.yaml` to `configs/paths.yaml` for a new checkout,
then edit `configs/paths.yaml` for local paths:

```yaml
ocid_root: "/path/to/OCID-dataset"
ocid_debug_sequence: "/path/to/OCID-dataset/YCB10/table/top/mixed/seq21"
outputs_dir: "outputs"
sam_outputs_dir: "outputs/sam_box_prompt"
```

The code preserves compatibility with CSV files that contain older absolute OCID
paths by remapping paths below the `OCID-dataset/` component to the currently
configured `ocid_root`.

The simulated dataset plan is documented in `docs/dataset_plan.md`, with the
current target configuration in `configs/sim_dataset.yaml`.

## Commands

### Generate a small simulated dataset pilot

```bash
PYTHONPATH=src python scripts/generate_sim_dataset_pilot.py --num-images 5
```

This writes a deterministic schema-check dataset under `data/cogar_sim_500/`.
It is intended only as a local smoke test before real Isaac/Gazebo data
generation.

### Prepare OCID indexes and binary ground-truth masks

```bash
PYTHONPATH=src python scripts/prepare_ocid_debug_dataset.py
```

This creates the image-level index, object-level index, filtered object-level
index, and binary ground-truth masks.

Expected local outputs include:

```text
outputs/indexes/ocid_debug_seq21.csv
outputs/indexes/ocid_debug_seq21_objects.csv
outputs/indexes/ocid_debug_seq21_objects_filtered.csv
outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv
outputs/gt_binary_masks/
```

### Visualize object prompts and binary masks

```bash
PYTHONPATH=src python scripts/visualize_object_prompt.py 0
PYTHONPATH=src python scripts/visualize_binary_gt_mask.py 5
```

### Run SAM box prompt on one object

```bash
PYTHONPATH=src python scripts/run_sam_box_prompt.py \
  --row 0 \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

### Run SAM box prompt over all filtered objects

```bash
PYTHONPATH=src python scripts/run_sam_box_prompt_batch.py \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

Expected local outputs include:

```text
outputs/indexes/ocid_debug_seq21_sam_box_results.csv
outputs/sam_box_prompt/masks/
outputs/sam_box_prompt/visualizations/
```

### Analyze box-prompt results

```bash
PYTHONPATH=src python scripts/analyze_sam_results.py
```

Expected local outputs include:

```text
outputs/analysis/sam_box_prompt/
```

### Run SAM point prompt on one object

```bash
PYTHONPATH=src python scripts/run_sam_point_prompt.py \
  --row 1 \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

### Run SAM point prompt over all filtered objects

```bash
PYTHONPATH=src python scripts/run_sam_point_prompt_batch.py \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback
```

Expected local outputs include:

```text
outputs/sam_point_prompt_batch/sam_point_prompt_results.csv
outputs/sam_point_prompt_batch/masks/
outputs/sam_point_prompt_batch/visualizations/
```

### Analyze point-prompt results

```bash
PYTHONPATH=src python scripts/analyze_prompt_results.py \
  --results-csv outputs/sam_point_prompt_batch/sam_point_prompt_results.csv \
  --prompt-name sam_point_prompt \
  --output-dir outputs/analysis/sam_point_prompt \
  --top-k 10
```

Expected local outputs include:

```text
outputs/analysis/sam_point_prompt/sam_point_prompt_global_summary.csv
outputs/analysis/sam_point_prompt/sam_point_prompt_per_object_summary.csv
outputs/analysis/sam_point_prompt/sam_point_prompt_worst_cases.csv
outputs/analysis/sam_point_prompt/sam_point_prompt_best_cases.csv
outputs/analysis/sam_point_prompt/sam_point_prompt_iou_histogram.png
outputs/analysis/sam_point_prompt/sam_point_prompt_sam_score_vs_iou.png
outputs/analysis/sam_point_prompt/sam_point_prompt_per_object_mean_iou.png
```

### Compare box prompts and point prompts

```bash
PYTHONPATH=src python scripts/compare_prompt_results.py \
  --box-results-csv outputs/indexes/ocid_debug_seq21_sam_box_results.csv \
  --point-results-csv outputs/sam_point_prompt_batch/sam_point_prompt_results.csv \
  --output-dir outputs/analysis/prompt_comparison \
  --top-k 10
```

Expected local outputs include:

```text
outputs/analysis/prompt_comparison/box_vs_point_rowwise_comparison.csv
outputs/analysis/prompt_comparison/box_vs_point_global_summary.csv
outputs/analysis/prompt_comparison/box_vs_point_per_object_summary.csv
outputs/analysis/prompt_comparison/point_prompt_strongest_wins.csv
outputs/analysis/prompt_comparison/box_prompt_strongest_wins.csv
outputs/analysis/prompt_comparison/largest_prompt_differences.csv
outputs/analysis/prompt_comparison/box_vs_point_iou_scatter.png
outputs/analysis/prompt_comparison/box_vs_point_iou_delta_histogram.png
outputs/analysis/prompt_comparison/box_vs_point_per_object_delta.png
```

## Tests

The default tests avoid dataset and checkpoint dependencies:

```bash
PYTHONPATH=src pytest -q
```

Current expected result:

```text
13 passed
```

SAM smoke tests require local OCID data and a SAM checkpoint.

## Benchmark Interpretation

The box-vs-point comparison shows that box prompts are more reliable on the
current OCID debug subset.

The key observations are:

- Box prompts achieved higher mean IoU than single positive point prompts.
- Box prompts won on 41 out of 52 object instances.
- Point prompts won on 11 out of 52 object instances, but the gains were small.
- The largest point-prompt failures were severe.
- `object_id = 10` was the clearest failure case for single positive point prompts.
- SAM predicted confidence scores did not always reflect ground-truth IoU failure.

This suggests that bounding boxes provide a stronger spatial constraint for
cluttered robotic scenes, while single positive points can be ambiguous when
multiple objects are nearby or visually similar.

## Extension Points

- Add datasets under `src/cogar_seg/datasets/`.
- Add prompt builders under `src/cogar_seg/prompts/`.
- Add model adapters under `src/cogar_seg/models/`.
- Add metrics under `src/cogar_seg/metrics/`.
- Add plots and visual reports under `src/cogar_seg/visualization/`.
- Add benchmark workflows under `src/cogar_seg/evaluation/`.

## Next Planned Extensions

- Add negative background point prompts.
- Add box-plus-point combined prompts.
- Add multiple-point prompt strategies.
- Compare center point, random point, and farthest-inside point prompts.
- Add MobileSAM, FastSAM, EfficientSAM, and SAM2 adapters.
- Extend from 2D mask evaluation toward 3D semantic mapping experiments.
