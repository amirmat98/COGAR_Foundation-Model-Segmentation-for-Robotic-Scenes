# Simulation Dataset Plan

## Project

**Subgroup I2: Foundation Model Segmentation for Robotic Scenes**

Assignment:

**Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**

Student ID:

**5884715**

## Objective

The goal of this dataset is to support a zero-shot segmentation benchmark for
robotic scene understanding. The benchmark will evaluate foundation-model
segmentation methods and classical segmentation baselines on simulated robotic
scenes containing perception challenges that are common in robotics.

The main target foundation models are:

- SAM ViT-B
- SAM ViT-H
- SAM2
- FastSAM
- MobileSAM
- EfficientSAM

The classical comparison baselines are:

- Mask R-CNN
- DeepLabV3+
- YOLOv8-seg fine-tuned on a small subset

The main evaluation goals are:

- Segmentation accuracy
- Robustness to robotic-scene challenges
- Prompt sensitivity
- Inference speed on GPU and CPU
- Suitability for real-time or edge deployment

## Why Simulation

The assignment requires a simulated annotated robotic scene dataset. Simulation
makes it possible to control object geometry, materials, lighting, camera
viewpoints, robot context, occlusions, motion, and annotation quality.

The preferred simulation platform is **Isaac Sim**, because it supports robotic
simulation and synthetic data generation through Replicator. Replicator can use
semantic labels to generate annotations such as semantic segmentation,
bounding boxes, and segmentation masks.

Fallback options are:

- Gazebo
- Rviz2

These fallback options may be used only if Isaac Sim is not practical on the
available hardware.

## Dataset Size

Target dataset size:

```text
500 annotated RGB images
```

Recommended split:

| Split | Images | Percentage | Purpose |
|---|---:|---:|---|
| Train subset | 100 | 20% | Small supervised subset for classical baselines |
| Validation | 100 | 20% | Prompt strategy tuning and sanity checks |
| Test benchmark | 300 | 60% | Main zero-shot benchmark split |

The zero-shot foundation models should be evaluated primarily on the test set.
The small training subset is used only for classical baselines such as
YOLOv8-seg, Mask R-CNN, and DeepLabV3+.

## Challenge Categories

The dataset should cover five main robotic perception challenges:

| Challenge | Target images | Description |
|---|---:|---|
| Reflective metal | 100 | Metal tools, shiny parts, and reflective surfaces |
| Transparent objects | 100 | Glass cups, transparent boxes, clear plastic containers |
| Partial occlusion | 100 | Objects partly hidden by other objects or robot parts |
| Small parts | 100 | Screws, connectors, washers, plugs, small tools |
| Dynamic scenes | 100 | Moving objects, robot motion, or changed object poses |

Each image should have at least one primary challenge label. Some images may
contain multiple secondary challenges.

Example:

```text
challenge_primary = "transparent_objects"
challenge_secondary = ["partial_occlusion", "reflective_surface"]
```

## Scene Types

The dataset should include robotic tabletop and workspace scenes.

Planned scene types:

| Scene type | Description |
|---|---|
| Manipulation table | Objects placed on a table near a robot |
| Assembly workspace | Small parts, screws, connectors, tools |
| Inspection scene | Reflective and transparent objects under varied lighting |
| Cluttered bin/table | Overlapping objects and partial occlusions |
| Dynamic interaction | Moving object or robot motion between frames |

## Robot Context

The assignment suggests simulated robotic platforms such as Unitree A2 EDU or
Unitree G1 EDU. The dataset should include a visible or implied robot context.

Preferred robot context:

```text
Unitree G1 EDU-inspired humanoid manipulation scene
```

Alternative context:

```text
Generic robotic arm or mobile manipulator scene
```

The robot does not need to perform full control policies in the first dataset
version. It can be used as a scene element to create realistic robotic
viewpoints, occlusions, and workspace constraints.

Possible robot-scene roles:

| Role | Description |
|---|---|
| Static context | Robot visible near the table/workspace |
| Occluder | Robot hand, gripper, or arm partially blocks objects |
| Dynamic actor | Robot or object changes pose between frames |
| Camera reference | Camera viewpoint approximates robot perception |

## Object Categories

Initial semantic categories:

| Category ID | Category name | Challenge relevance |
|---:|---|---|
| 1 | metal_tool | reflective |
| 2 | metal_part | reflective, small parts |
| 3 | glass_cup | transparent |
| 4 | transparent_box | transparent |
| 5 | plastic_container | transparent, reflective |
| 6 | screw | small parts |
| 7 | connector | small parts |
| 8 | washer | small parts |
| 9 | cable | occlusion, small parts |
| 10 | robot_gripper_or_hand | robot context, occlusion |
| 11 | table | background/support |
| 12 | distractor_object | clutter |

For evaluation, the main foreground categories should be objects 1-9.
Robot parts and table surfaces can be kept as optional categories depending on
the final annotation export.

## Image and Annotation Modalities

Each generated sample should include:

| Modality | Required | Notes |
|---|---|---|
| RGB image | Yes | Main model input |
| Instance mask | Yes | Required for mask AP and object-level IoU |
| Semantic mask | Yes | Required for per-category IoU |
| 2D bounding boxes | Yes | Required for box-prompt generation |
| Point prompts | Derived | Generated from object masks |
| Depth image | Optional | Useful for robotic-scene metadata |
| Camera pose | Optional | Useful for simulation traceability |
| Scene metadata | Yes | Challenge type, scene ID, frame ID |

## Recommended Directory Structure

```text
sim_dataset/
  images/
    train/
    val/
    test/

  masks/
    instance/
      train/
      val/
      test/

    semantic/
      train/
      val/
      test/

  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
    categories.json
    scene_metadata.csv

  metadata/
    camera_poses.csv
    generation_config.yaml
    software_versions.json
```

The final benchmark code should not depend directly on the exact simulator
output format. Instead, simulator outputs should be converted into a clean
benchmark index.

## Benchmark Index Format

The simulated dataset should be converted into a CSV index with one row per
object instance:

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

This format allows the existing SAM box-prompt and point-prompt logic to be
generalized from OCID to simulated data.

## Dataset Split Policy

The split should avoid leakage between nearly identical frames.

Recommended rules:

- Images from the same scene seed should stay in the same split when possible.
- Dynamic sequences should not be split frame-by-frame across train/val/test.
- The test split should contain all five challenge categories.
- Classical baselines may use the train split only.
- Foundation models should be evaluated zero-shot on validation and test.

Suggested split:

```text
train: 100 images
val:   100 images
test:  300 images
```

## Prompt Types

The benchmark should evaluate:

| Prompt mode | Description |
|---|---|
| box | Use ground-truth 2D bounding box |
| point | Use one positive point from the object mask |
| multi_point | Use multiple positive points |
| point_with_negative | Use positive object point and background negative points |
| box_plus_point | Use box and point together |
| automatic | Use automatic mask generation without object-specific prompts |

The current repository already supports:

- box prompt
- single positive point prompt

The next prompt modes to add are:

- automatic mask generation
- point with negative background points
- box plus point

## Point Prompt Generation

For object-level point prompts, the default point should be derived from the
ground-truth instance mask.

Recommended strategies:

| Strategy | Description | Priority |
|---|---|---:|
| mask_center | Center of the object mask bounding box | High |
| centroid | Mean coordinate of mask pixels | High |
| farthest_inside | Pixel farthest from mask boundary | Medium |
| random_inside | Random valid foreground pixel | Medium |
| multi_point | Multiple foreground points | Medium |
| point_with_negative | Foreground point plus background points | High |

The first implementation should use `mask_center`, because it is simple and
already compatible with the current OCID prototype.

## Box Prompt Generation

Box prompts should be generated from the ground-truth object mask or simulator
2D bounding-box annotations.

Required format:

```text
bbox_xmin
bbox_ymin
bbox_xmax
bbox_ymax
```

Coordinate convention:

```text
XYXY pixel format
```

The benchmark should store both the original simulator bounding box and the
mask-derived bounding box if they differ.

## Automatic Mask Generation

Automatic mask generation should be evaluated separately from object-specific
prompts.

For automatic mask generation, the model predicts multiple masks for an image.
The benchmark must then match predicted masks to ground-truth instances.

Recommended matching method:

```text
match each predicted mask to the ground-truth instance with highest IoU
```

Then compute:

- mask AP
- best-match IoU
- per-category IoU
- false positive masks
- missed ground-truth instances

## Models to Benchmark

Foundation models:

| Model | Priority | Notes |
|---|---:|---|
| SAM ViT-B | Already prototyped | Current working baseline |
| SAM ViT-H | High | Required stronger SAM model |
| SAM2 | High | Required by assignment |
| FastSAM | High | Speed-oriented SAM alternative |
| MobileSAM | High | Lightweight SAM variant |
| EfficientSAM | High | Lightweight SAM variant |

Classical baselines:

| Model | Priority | Notes |
|---|---:|---|
| YOLOv8-seg | High | Practical first fine-tuned baseline |
| Mask R-CNN | Medium | Detectron2-based instance baseline |
| DeepLabV3+ | Medium | Semantic segmentation baseline |

## Model Evaluation Modes

Each model should be evaluated in the modes that make sense for its interface.

| Model family | Box prompt | Point prompt | Automatic masks | Fine-tuning |
|---|---:|---:|---:|---:|
| SAM | Yes | Yes | Yes | No |
| SAM2 | Yes | Yes | Yes/optional | No |
| FastSAM | Yes/approx. | Point/approx. | Yes | No |
| MobileSAM | Yes | Yes | Yes | No |
| EfficientSAM | Yes | Yes | Optional | No |
| YOLOv8-seg | No | No | Direct prediction | Yes |
| Mask R-CNN | No | No | Direct prediction | Yes |
| DeepLabV3+ | No | No | Direct prediction | Yes |

## Metrics

Required metrics:

| Metric | Purpose |
|---|---|
| IoU | Object-level overlap between predicted and ground-truth masks |
| mIoU | Overall segmentation quality |
| Per-category IoU | Category-level robustness |
| Boundary F1 | Boundary precision, useful for small parts |
| Mask AP | Instance segmentation quality |
| FPS GPU | Real-time feasibility on GPU |
| FPS CPU | CPU/edge feasibility |

Current repository supports IoU. The next metric additions should be:

- per-category IoU
- boundary F1
- mask AP
- FPS measurement

## Speed Benchmark

Inference speed should be measured separately from accuracy.

Required speed outputs:

| Measurement | Description |
|---|---|
| model_load_time_s | Time to load model/checkpoint |
| preprocessing_time_ms | Image loading and preprocessing |
| encoder_time_ms | Image encoder time, if accessible |
| decoder_time_ms | Prompt decoder or mask decoder time |
| total_inference_time_ms | End-to-end inference time |
| fps | Frames per second |
| gpu_memory_mb | Peak GPU memory usage |
| device | CPU or GPU |

Measure speed on:

```text
GPU
CPU
```

For GPU speed, use CUDA if available. For CPU speed, force CPU inference.

## Failure Mode Analysis

Failure modes to annotate and discuss:

| Failure mode | Description |
|---|---|
| missed_transparent_object | Model ignores glass/clear object |
| reflection_confusion | Model segments reflected highlight or wrong surface |
| merged_objects | Adjacent objects are merged into one mask |
| fragmented_object | One object is split into multiple masks |
| small_object_missed | Screw/connector not segmented |
| occluded_object_failure | Visible part is not segmented correctly |
| wrong_point_target | Point prompt selects neighboring object |
| over_segmentation | Mask includes background or table |
| under_segmentation | Mask covers only part of object |
| motion_related_error | Dynamic object blurred or inconsistently segmented |

Failure analysis should be organized by:

- model
- prompt mode
- challenge type
- object category
- severity

## Expected Output Tables

Final benchmark tables should include:

```text
global_model_summary.csv
per_category_iou.csv
per_challenge_iou.csv
prompt_mode_comparison.csv
speed_benchmark_gpu.csv
speed_benchmark_cpu.csv
mask_ap_summary.csv
boundary_f1_summary.csv
failure_mode_summary.csv
```

## Expected Output Plots

Final plots should include:

```text
model_mean_iou_barplot.png
per_category_iou_heatmap.png
per_challenge_iou_heatmap.png
prompt_mode_comparison.png
speed_vs_accuracy_scatter.png
gpu_fps_barplot.png
cpu_fps_barplot.png
boundary_f1_barplot.png
failure_mode_grid.png
```

## Expected Deliverables

Final deliverables:

- Simulated annotated robotic scene dataset
- Dataset generation/configuration documentation
- Full benchmark result tables
- Accuracy plots and speed plots
- Failure mode analysis report
- Recommendation guide for model choice in robotic scenarios

## Recommendation Guide Target

The final report should include a practical recommendation table.

Example structure:

| Robotic scenario | Recommended model | Reason |
|---|---|---|
| Highest accuracy, offline | SAM ViT-H or SAM2 | Strong segmentation quality |
| Fast GPU inference | FastSAM or SAM2 variant | Better speed/accuracy trade-off |
| Edge deployment | MobileSAM or EfficientSAM | Lightweight design |
| Transparent objects | To be determined | Requires challenge-specific results |
| Small parts | Box-prompt SAM or fine-tuned YOLOv8-seg | Stronger localization may help |
| Dynamic scenes | SAM2 or fast direct predictor | Video/motion robustness may matter |

## Immediate Implementation Plan

The next implementation steps are:

1. Create `configs/simulation_dataset.yaml`.
2. Add a simulated dataset schema/index format.
3. Add `src/cogar_seg/datasets/sim_robotic.py`.
4. Add `scripts/prepare_sim_dataset.py`.
5. Add `scripts/visualize_sim_sample.py`.
6. Only then generalize the benchmark runner from OCID to simulation.

## Notes

The current OCID benchmark should remain in the repository as a prototype and
sanity-check pipeline. It should not be deleted or heavily refactored yet.

The simulation benchmark should be added as a new layer on top of the existing
repository structure.

The guiding rule is:

```text
Do not refactor working code unless the next feature requires it.
```