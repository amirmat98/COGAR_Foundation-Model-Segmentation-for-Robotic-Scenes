# Task 1: Dataset Creation and Curation

Task 1 requires robotic-scene datasets with segmentation annotations and
challenge cases: reflective metal, transparent objects, partial occlusion,
small parts, and dynamic scenes.

Status: complete, audited on 2026-06-13.

## Figure

![Dataset examples](../../outputs/final_benchmark_assets/plots/dataset_examples.png)

## Dataset Set

| Dataset | Role | Annotation type |
| --- | --- | --- |
| Isaac official Unitree G1 | Main synthetic dataset | RGB, semantic masks, instance masks, boxes, depth, COCO |
| BlenderProc COGAR-SimRobotics-1000 | Secondary synthetic dataset | RGB, COCO instance segmentation, metadata |
| OCID | External real-world robustness dataset | RGB, labels, depth, point clouds |

## Current Evidence

Isaac official Unitree G1 has been generated locally:

- Root: `/mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000`
- Manifest rows: 1000
- RGB images: 1000
- Semantic masks: 1000
- Instance masks: 1000
- COCO file: `annotations/instances_coco.json`
- COCO annotations: 72,695
- COCO categories: 16
- Validation: PASS

OCID is available locally:

- Root: `/mnt/Info/COGAR_DATASETs/OCID-dataset`
- RGB frames: 2390
- Label frames: 2390
- Depth frames: 2390
- Point clouds: 2380

BlenderProc final generation target:

```text
/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

The generator and normalizer are in this repository.

BlenderProc target:

- Images: 1000
- COCO categories: 10
- COCO annotations: 8768
- Split: 700 train, 150 validation, 150 test
- Challenge families: 200 images each
- Status: available locally

Local 5-image smoke test: PASS

## Closure Decision

Task 1 is complete. The project now has two local synthetic robotic-scene
datasets with segmentation annotations and explicit coverage of reflective
metal, transparent glass, occlusion, small parts, and moving or dynamic scenes.
OCID is available locally as an external real-world robustness dataset.
