# COGAR-Sim Dataset Quality Workflow

This workflow is the source of truth for improving the simulated benchmark
dataset after the v0 sanity run. Generated data, masks, outputs, checkpoints,
raw RGB images, COCO JSON exports, plots, and result CSVs should stay out of
Git.

## v0 Sanity Dataset vs v1/v2 Benchmark Dataset

The 25-image v0 dataset proved that the BlenderProc, normalization, mask export,
finalization, and SAM evaluation path works end to end. It is not the final
benchmark dataset.

Known v0 limitations:

- Some frames are over-cluttered, with more than 25 target objects.
- Some scenes contain too many small screws/connectors, which makes the frame
  chaotic rather than robotic-scene-like.
- Dark, flat, low-texture, or near-wall-only frames need to be excluded.
- Scene-level flags were useful for early analysis, but object-level material
  flags must be derived from the object category.
- Table/support surfaces should not be benchmark target objects by default.

The v1/v2 target is a cleaner robotic-scene dataset with challenge diversity:
reflective metal, transparent glass, partial occlusion, small parts, and
dynamic-scene-style frames.

## Quality Criteria

- Preferred object count: 6 to 18 benchmark objects per image.
- Hard small-parts scenes may be denser, but should usually stay below 20.
- Images with more than 25 target objects are rejected or flagged.
- Empty, dark, flat, low-texture, unreadable, or near-wall-only frames are
  flagged and excluded from the filtered benchmark index.
- Table/support surfaces are excluded from object-level evaluation by default.
- `challenge_primary` remains scene-level.
- Object flags are category/material based:
  - `is_reflective`: `metal_part`, `tool`
  - `is_transparent`: `glass_object`
  - `is_small_part`: `screw`, `connector`
  - `is_dynamic`: `challenge_primary == dynamic_scene`
  - `is_occluded`: occlusion metadata if available, otherwise
    `challenge_primary == partial_occlusion`
- Challenge families and categories should remain balanced enough for analysis.

## Generate Cleaner Candidates

Run BlenderProc through the BlenderProc launcher, not normal Python:

```bash
blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 70 \
  --raw-dataset-name pilot_v3_cleaner
```

Optional deterministic override:

```bash
blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 70 \
  --raw-dataset-name pilot_v3_cleaner \
  --seed 42
```

## Build, Finalize, Validate, Audit, and Filter

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/dataset/build_clean_sim_dataset.py \
  --raw-coco-dir data/cogar_sim_500/raw_blenderproc/pilot_v3_cleaner/coco_data \
  --raw-metadata data/cogar_sim_500/metadata/frame_index_pilot_v2.csv \
  --output-root data/cogar_sim_500 \
  --config configs/blenderproc_dataset.yaml \
  --expected-images 70 \
  --index-output outputs/indexes/cogar_sim_500_objects_v1.csv \
  --mask-dir data/cogar_sim_500/instance_masks/v1 \
  --final-index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1.csv \
  --audit-output-dir outputs/tables/dataset_audit_v1 \
  --filtered-index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --exclude-categories table \
  --min-area 25 \
  --filter-bad
```

The build script runs:

1. `normalize_cogar_sim_500.py`
2. `create_object_index.py`
3. `export_binary_masks.py`
4. `finalize_cogar_sim_index.py`
5. `validate_sim_index.py`
6. `audit_sim_dataset.py`
7. `filter_sim_index.py` when `--filter-bad` is set
8. `validate_sim_index.py` again on the filtered index

Expected audit outputs:

- `image_quality_audit.csv`
- `category_counts.csv`
- `challenge_counts.csv`
- `category_by_challenge.csv`
- `area_by_category.csv`
- `object_flag_counts.csv`
- `bad_images.txt`
- `audit_summary.json`

## Validate and Audit Manually

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/dataset/validate_sim_index.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv
```

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/dataset/audit_sim_dataset.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --output-dir outputs/tables/dataset_audit_v1_filtered
```

If the filtered audit still shows dark, flat, or over-cluttered frames, regenerate
another candidate set or pass explicit file names to `filter_sim_index.py`:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/dataset/filter_sim_index.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1.csv \
  --audit outputs/tables/dataset_audit_v1/image_quality_audit.csv \
  --output data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --exclude-bad \
  --exclude-files 000003.png 000008.png 000013.png
```

## Run SAM ViT-B on the Filtered v1 Index

Box prompt:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/eval/run_sam_box_prompt.py \
  --config configs/paths_sim.yaml \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback \
  --output-dir outputs/sim_sam_vit_b_box_v1 \
  --results-csv outputs/results/sim_sam_vit_b_box_v1.csv \
  --no-visualizations
```

Point prompt:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/eval/run_sam_point_prompt.py \
  --config configs/paths_sim.yaml \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback \
  --output-dir outputs/sim_sam_vit_b_point_v1 \
  --results-csv outputs/results/sim_sam_vit_b_point_v1.csv \
  --no-visualizations
```

Automatic masks:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/eval/run_sam_auto_masks.py \
  --config configs/paths_sim.yaml \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device auto \
  --allow-cpu-fallback \
  --output-dir outputs/sim_sam_vit_b_auto_v1 \
  --results-csv outputs/results/sim_sam_vit_b_auto_v1.csv
```

Summarize:

```bash
PYTHONPATH="$PWD/src:${PYTHONPATH}" python scripts/analysis/summarize_sim_benchmark.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv
```

## Tests

Run the lightweight test suite before generating or publishing results:

```bash
PYTHONPATH=src pytest -q
```
