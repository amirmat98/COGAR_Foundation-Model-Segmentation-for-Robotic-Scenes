# Simulated Robotic Scene Dataset Plan

## Goal

Build a reproducible simulated robotic-scene segmentation dataset with about
500 annotated RGB images for benchmarking SAM, SAM2, FastSAM, MobileSAM,
EfficientSAM, and classical segmentation baselines.

The dataset should stress robotics-specific failure modes:

- reflective metal
- transparent glass or plastic
- partial occlusion
- small screws, washers, connectors, and cables
- dynamic or changed object poses

## Canonical Local Layout

Use `data/cogar_sim_500/` as the canonical local dataset root:

```text
data/cogar_sim_500/
  rgb/
    train/
    val/
    test/
  instance_masks/
    train/
    val/
    test/
  semantic_masks/
    train/
    val/
    test/
  depth/
    train/
    val/
    test/
  annotations/
    categories.json
    sim_robotic_scenes_index.csv
  metadata/
    generation_config.yaml
    generation_summary.json
  assets/
  scenes/
  splits/
```

`sim_dataset/` may be used for disposable pilot exports, but benchmark scripts
should target `data/cogar_sim_500/`.

## Required Object Index Columns

The main benchmark index is:

```text
data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv
```

It should contain one row per object instance with these columns:

```text
image_id
scene_id
frame_id
split
image_path
instance_mask_path
semantic_mask_path
category_id
category_name
object_id
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
point_x
point_y
challenge_primary
challenge_secondary
is_reflective
is_transparent
is_occluded
is_small_part
is_dynamic
camera_name
```

The schema is validated by `cogar_seg.datasets.sim_robotic`.

## Suggested Split

The current configuration in `configs/sim_dataset.yaml` uses:

| Split | Images |
|---|---:|
| train | 350 |
| val | 75 |
| test | 75 |

Foundation-model zero-shot benchmarks can evaluate all splits. Classical
baselines should train on `train`, tune on `val`, and report final results on
`test`.

## Generation Stages

1. Create a 1-5 image dummy/pilot sample that exercises the index schema.
2. Convert a small Isaac/Gazebo/export manifest into the canonical index.
3. Generate a 25-image sanity dataset with real synthetic images and masks.
4. Validate paths, masks, boxes, point prompts, categories, and challenge flags.
5. Scale to the full 500-image dataset.
6. Freeze the generated dataset version and write `generation_summary.json`.

## Validation Checklist

- Every `image_path`, `instance_mask_path`, and `semantic_mask_path` exists.
- All bounding boxes are valid XYXY boxes inside the image.
- Point prompts are inside image bounds and preferably inside the instance mask.
- Split values are only `train`, `val`, or `test`.
- Category IDs match `configs/sim_dataset.yaml`.
- Challenge flags are boolean-like.
- Each challenge category has the target approximate count.
- Small parts and transparent/reflective objects are represented in each split.

## Git Policy

Generated images, masks, depth maps, and benchmark outputs stay ignored. Commit
only code, configs, schemas, templates, tests, and documentation.
