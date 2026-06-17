# Dataset: Isaac Official Unitree G1

## Summary

- Type: synthetic robotic-scene dataset
- Simulator: Isaac Sim
- Robot: official Unitree G1 USD asset
- Images: 1000
- Main challenges: reflective metal, transparent glass, partial occlusion,
  small parts, dynamic scenes, robot close range

## Local Path

Current local copy:

```text
/mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000
```

Default repo path for another user:

```text
Datasets/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000
```

Users may store it elsewhere and update `configs/datasets.yaml`.

## Contents

- `manifest.jsonl`
- `isaac/rgb_*.png`
- `isaac/semantic_segmentation_*.png`
- `isaac/instance_segmentation_*.png`
- `isaac/bounding_box_2d_tight_*.npy`
- `isaac/distance_to_camera_*.npy`
- `annotations/instances_coco.json`

## Validation

Validation date: 2026-06-13

- Manifest rows: 1000
- RGB images: 1000
- Semantic masks: 1000
- Instance masks: 1000
- Robot mode: official Unitree USD for all frames
- COCO annotations: 72,695
- COCO categories: 16
- Result: PASS

## Release URL

Pending public release. Package and upload this dataset using
[public_release.md](public_release.md), then replace this section with the
Zenodo record URL or DOI.
