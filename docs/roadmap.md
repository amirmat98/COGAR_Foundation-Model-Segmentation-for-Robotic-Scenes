# Project Roadmap

## Project Title

**Zero-Shot Segmentation Benchmark for Robotic Perception in Simulation**

**Subgroup:** I2 — Foundation Model Segmentation for Robotic Scenes  
**Student ID:** 5884715

## Main Goal

The goal of this project is to build a systematic benchmark for evaluating zero-shot and lightweight segmentation models in simulated robotic scenes.

The project compares foundation-model segmentation methods and classical segmentation baselines on robotic perception challenges such as:

- reflective metal objects
- transparent glass or plastic objects
- partial occlusions
- small parts such as screws, washers, and connectors
- dynamic or changing scenes

The final project should answer this question:

> Which segmentation model should be used for different robotic perception scenarios?

## Project Summary

This project has five major phases:

1. **Phase A — Simulation Dataset**
2. **Phase B — Foundation Model Benchmark**
3. **Phase C — Classical Baseline Benchmark**
4. **Phase D — Metrics, Speed, and Analysis**
5. **Phase E — Final Report and Recommendation Guide**

The current OCID-based SAM pipeline is a prototype and sanity-check pipeline.  
The final assignment should focus on the simulated robotic dataset.

## Final Deliverables

The final project should deliver:

1. A simulated annotated robotic-scene dataset.
2. Full benchmark results with tables and plots.
3. Quantitative evaluation using segmentation metrics.
4. Inference-speed evaluation on GPU and CPU.
5. Qualitative failure-mode analysis.
6. A recommendation guide for robotic segmentation model selection.

## Models to Benchmark

### Foundation Models

These models should be evaluated in zero-shot mode:

| Model | Required | Notes |
|---|---:|---|
| SAM ViT-B | Yes | Already prototyped on OCID |
| SAM ViT-H | Yes | Stronger SAM variant |
| SAM2 | Yes | Useful for image and dynamic/video-style scenes |
| FastSAM | Yes | Speed-oriented SAM-like model |
| MobileSAM | Yes | Lightweight model for edge deployment |
| EfficientSAM | Yes | Lightweight model for edge deployment |

### Classical Baselines

These models should be trained or fine-tuned on a small subset of the simulated dataset:

| Model | Required | Priority |
|---|---:|---:|
| YOLOv8-seg | Yes | High |
| Mask R-CNN | Yes | Medium |
| DeepLabV3+ | Yes | Medium |

If time is limited, YOLOv8-seg should be implemented first because it is usually the most practical baseline to train and evaluate quickly.

## Prompt Modes

The foundation models should be tested with different prompt modes:

| Prompt mode | Description | Status |
|---|---|---|
| Box prompt | Use ground-truth bounding box | Prototyped on OCID |
| Point prompt | Use one positive point | Prototyped on OCID |
| Automatic mask generation | Generate masks without object-specific prompt | Planned |
| Box + point | Combine box and positive point | Planned |
| Positive + negative points | Use foreground and background points | Planned |
| Multi-point prompt | Use multiple positive points | Planned |

## Evaluation Metrics

The final benchmark should include:

| Metric | Purpose |
|---|---|
| IoU | Object-level mask overlap |
| mIoU | Mean segmentation quality |
| Per-category IoU | Category-level robustness |
| Boundary F1 | Boundary quality, useful for small parts |
| Mask AP | Instance segmentation quality |
| GPU FPS | Real-time feasibility on GPU |
| CPU FPS | CPU and edge feasibility |

## Robotic Challenge Categories

The simulated dataset should contain approximately 500 annotated RGB images.

Recommended distribution:

| Challenge | Target images | Purpose |
|---|---:|---|
| Reflective metal | 100 | Test shiny tools and metal parts |
| Transparent objects | 100 | Test glass and clear plastic |
| Partial occlusion | 100 | Test hidden or overlapping objects |
| Small parts | 100 | Test screws, washers, connectors |
| Dynamic scenes | 100 | Test moving objects or changed poses |

## Dataset Structure

The final simulated dataset should follow this structure:

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
    scene_metadata.csv
    sim_robotic_scenes_index.csv

  metadata/
    generation_config.yaml
    generation_summary.json
```

The most important file is:

```text
data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv
```

It should contain one row per object instance.

Required index columns:

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

## Current Status

### Completed Prototype Work

The OCID prototype benchmark is complete enough as a proof of concept.

Completed:

- OCID image-level indexing
- OCID object-level indexing
- binary ground-truth mask export
- SAM ViT-B box-prompt inference
- SAM ViT-B point-prompt inference
- batch evaluation
- IoU computation
- prompt comparison
- result analysis
- README update

### Completed Simulation Infrastructure

Completed:

- simulation dataset plan
- simulation dataset YAML configuration
- simulated dataset index schema
- index validation utilities
- simulated dataset preparation script
- simulated sample visualizer
- dummy simulated sample generator
- Isaac Sim generation plan
- Isaac export converter skeleton
- Isaac manifest template generator
- Isaac Replicator sanity generator skeleton

### Not Completed Yet

Not completed:

- real 25-image Isaac Sim sanity dataset
- real 500-image simulated dataset
- SAM/SAM2/FastSAM/MobileSAM/EfficientSAM on simulated data
- automatic mask generation benchmark
- YOLOv8-seg fine-tuning
- Mask R-CNN baseline
- DeepLabV3+ baseline
- mIoU, boundary F1, mask AP
- GPU/CPU speed benchmark
- final failure analysis
- final recommendation guide

## Phase A — Simulation Dataset

### A1 — Confirm Simulation Environment

**Goal:** Check whether Isaac Sim is installed and runnable.

Tasks:

- find Isaac Sim installation path
- verify `python.sh` or `isaac-sim.sh`
- test minimal Isaac Sim Python script
- decide whether to use Isaac Sim, Gazebo, or fallback dummy/simplified data

Success condition:

> A minimal Isaac Sim script runs successfully.

### A2 — Generate 25-Image Sanity Dataset

**Goal:** Generate a small real synthetic dataset before scaling to 500 images.

Target:

| Challenge | Images |
|---|---:|
| reflective_metal | 5 |
| transparent_objects | 5 |
| partial_occlusion | 5 |
| small_parts | 5 |
| dynamic_scenes | 5 |

Success condition:

> 25 RGB images with masks, boxes, semantic labels, and metadata exist.

### A3 — Convert Isaac Export to Benchmark Format

**Goal:** Convert raw Isaac Sim outputs into `data/cogar_sim_500/`.

Tasks:

- read Isaac export manifest
- copy or map RGB images
- copy or map masks
- compute point prompts
- compute or verify bounding boxes
- write `sim_robotic_scenes_index.csv`
- validate the index

Success condition:

> Converted index passes `validate_sim_index()`.

### A4 — Visual Validation

**Goal:** Check several generated samples visually.

Tasks:

- visualize RGB image
- overlay instance mask
- draw bounding box
- draw point prompt
- verify category and challenge labels

Success condition:

> Random visual samples look correct.

### A5 — Scale to 500 Images

**Goal:** Generate the full assignment dataset.

Target:

> 500 annotated images

Recommended split:

| Split | Images | Purpose |
|---|---:|---|
| train | 350 | Classical baseline training |
| val | 75 | Validation and prompt checks |
| test | 75 | Final benchmark reporting |

Success condition:

> 500 images exist and the full index validates.

## Phase B — Foundation Model Benchmark

### B1 — Run SAM ViT-B on Simulated Dataset

**Goal:** Port the existing OCID SAM ViT-B box/point workflow to `data/cogar_sim_500/`.

Prompt modes:

- box
- point

Success condition:

> SAM ViT-B results CSV exists for simulated data.

### B2 — Run SAM ViT-H

**Goal:** Evaluate stronger SAM checkpoint.

Prompt modes:

- box
- point
- automatic mask generation

Success condition:

> SAM ViT-H results are comparable with SAM ViT-B.

### B3 — Add Automatic Mask Generation

**Goal:** Evaluate foundation models without object-specific prompts.

Tasks:

- generate all masks per image
- match predicted masks to ground-truth instances
- compute matched IoU
- compute false positives and missed objects

Success condition:

> Automatic mask generation can be evaluated against ground truth.

### B4 — Add SAM2

**Goal:** Evaluate SAM2 on static images and dynamic-style scenes.

Modes:

- image mode
- optional video/dynamic sequence mode

Success condition:

> SAM2 results are included in comparison tables.

### B5 — Add FastSAM

**Goal:** Evaluate speed-oriented SAM alternative.

Success condition:

> FastSAM accuracy and FPS are measured.

### B6 — Add MobileSAM

**Goal:** Evaluate lightweight SAM variant for edge deployment.

Success condition:

> MobileSAM accuracy/FPS trade-off is measured.

### B7 — Add EfficientSAM

**Goal:** Evaluate lightweight efficient SAM variant.

Success condition:

> EfficientSAM accuracy/FPS trade-off is measured.

## Phase C — Classical Baselines

### C1 — Prepare Training Subset

**Goal:** Use the train split for supervised baseline training.

Dataset:

> 100 train images

Tasks:

- export YOLO segmentation format
- export COCO format
- verify masks and labels

Success condition:

> Training data can be loaded by YOLOv8-seg and COCO tools.

### C2 — Train YOLOv8-Seg

**Goal:** Create the first supervised baseline.

Tasks:

- fine-tune on train split
- validate on val split
- evaluate on test split

Success condition:

> YOLOv8-seg has test metrics and FPS results.

### C3 — Add Mask R-CNN

**Goal:** Create classical instance segmentation baseline.

Success condition:

> Mask R-CNN results are included if time allows.

### C4 — Add DeepLabV3+

**Goal:** Create semantic segmentation baseline.

Success condition:

> DeepLabV3+ results are included if time allows.

## Phase D — Metrics, Speed, and Analysis

### D1 — Add mIoU and Per-Category IoU

**Goal:** Measure overall and category-specific segmentation quality.

Outputs:

- `global_metrics.csv`
- `per_category_iou.csv`
- `per_challenge_iou.csv`

### D2 — Add Boundary F1

**Goal:** Measure boundary quality, especially for small objects.

Output:

- `boundary_f1_summary.csv`

### D3 — Add Mask AP

**Goal:** Measure instance segmentation performance.

Output:

- `mask_ap_summary.csv`

### D4 — Add GPU Speed Benchmark

**Goal:** Measure real-time feasibility on GPU.

Outputs:

- `gpu_fps.csv`
- `gpu_inference_time.csv`

### D5 — Add CPU Speed Benchmark

**Goal:** Measure CPU and edge feasibility.

Outputs:

- `cpu_fps.csv`
- `cpu_inference_time.csv`

## Phase E — Report and Recommendation Guide

### E1 — Result Tables

Required tables:

- `model_global_summary.csv`
- `per_category_iou.csv`
- `per_challenge_iou.csv`
- `prompt_mode_comparison.csv`
- `speed_benchmark.csv`

### E2 — Result Plots

Required plots:

- `mean_iou_barplot.png`
- `per_category_iou_heatmap.png`
- `per_challenge_iou_heatmap.png`
- `speed_vs_accuracy_scatter.png`
- `gpu_fps_barplot.png`
- `cpu_fps_barplot.png`

### E3 — Failure Mode Analysis

Failure categories:

| Failure mode | Description |
|---|---|
| missed_transparent_object | glass/clear object not segmented |
| reflection_confusion | reflection or highlight segmented incorrectly |
| merged_objects | adjacent objects merged into one mask |
| fragmented_object | one object split into multiple masks |
| small_object_missed | screw/connector missed |
| occlusion_failure | occluded target poorly segmented |
| wrong_point_target | point prompt selects wrong object |
| over_segmentation | mask includes background |
| under_segmentation | mask covers only part of object |
| dynamic_inconsistency | mask unstable across dynamic frames |

Outputs:

- `worst_cases_by_model.csv`
- `worst_cases_by_challenge.csv`
- `failure_case_grid.png`

### E4 — Recommendation Guide

**Goal:** Recommend which segmentation model to use in each robotic scenario.

Example recommendation table:

| Robotic scenario | Recommended model | Reason |
|---|---|---|
| highest accuracy offline | SAM ViT-H or SAM2 | stronger segmentation quality |
| real-time GPU inference | FastSAM or SAM2 variant | better speed |
| edge deployment | MobileSAM or EfficientSAM | lightweight |
| transparent objects | determined by results | challenge-specific |
| small parts | box-prompt SAM or YOLOv8-seg | stronger localization |
| dynamic scenes | SAM2 | image/video segmentation ability |

## Immediate Next Steps

The next practical steps are:

1. Keep repository scripts minimal and focused on reusable benchmark steps.
2. Confirm whether Isaac Sim is installed.
3. If Isaac Sim works, generate a 25-image sanity dataset.
4. If Isaac Sim is not available, choose a fallback:
   - install Isaac Sim
   - use Gazebo/Rviz2
   - use a simplified synthetic generator as temporary fallback
5. Convert the sanity dataset into `data/cogar_sim_500/`.
6. Run SAM ViT-B on the sanity dataset.
7. Only then scale toward 500 images.

## Risk Management

### Risk 1 — Isaac Sim is not installed or too heavy

Mitigation:

> Use Gazebo, Rviz2, or a simplified synthetic dataset as temporary fallback.

### Risk 2 — Dataset generation takes too long

Mitigation:

> Start with 25 images, validate everything, then scale gradually.

### Risk 3 — Too many models for available time

Mitigation priority:

1. SAM ViT-B
2. SAM ViT-H
3. SAM2
4. FastSAM
5. MobileSAM
6. EfficientSAM
7. YOLOv8-seg
8. Mask R-CNN
9. DeepLabV3+

### Risk 4 — Metrics become too complex

Mitigation priority:

1. IoU
2. mIoU
3. per-category IoU
4. FPS
5. boundary F1
6. mask AP

## Definition of Done

The project is complete when:

- the simulated dataset exists
- the dataset has around 500 annotated images
- all required challenge categories are represented
- the benchmark index validates
- SAM-style models have been evaluated
- at least one classical baseline has been evaluated
- mIoU, per-category IoU, boundary F1, mask AP, and FPS are reported
- failure modes are analyzed qualitatively
- a recommendation guide is written

## Current Priority

The current priority is:

> Generate or obtain the first real simulated dataset.

Everything else depends on this.

## References

- NVIDIA Isaac Sim synthetic data generation and Replicator workflows.
- Segment Anything prompt-based segmentation and automatic mask generation.
- SAM2 image/video segmentation.
- Ultralytics YOLO segmentation training and prediction workflows.
