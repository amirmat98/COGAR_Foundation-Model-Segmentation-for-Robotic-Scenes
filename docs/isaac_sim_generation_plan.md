# Isaac Sim Dataset Generation Plan

## Project

**Subgroup I2: Foundation Model Segmentation for Robotic Scenes**

Assignment:

**Zero-Shot Segmentation Benchmark for Robotic Perception (Simulation)**

Student ID:

**5884715**

## Purpose

This document defines how the simulated robotic-scene dataset will be generated
using Isaac Sim and Replicator.

The goal is to create approximately 500 annotated RGB images for the assignment:

**Zero-Shot Segmentation Benchmark for Robotic Perception**

The generated dataset must support evaluation of:

- SAM ViT-B
- SAM ViT-H
- SAM2
- FastSAM
- MobileSAM
- EfficientSAM
- YOLOv8-seg
- Mask R-CNN
- DeepLabV3+

The dataset should contain robotic-scene challenges such as:

- Reflective metal objects
- Transparent glass or plastic objects
- Partial occlusions
- Small parts such as screws, washers, and connectors
- Dynamic or changing scenes

## Why Isaac Sim / Replicator

Isaac Sim is selected as the preferred simulator because it is designed for
robotics simulation, testing, and synthetic data generation in physically based
virtual environments. :contentReference[oaicite:0]{index=0}

Isaac Sim Replicator provides tools and workflows for synthetic data generation,
including semantic labeling, sensor visualization, GUI-based data recording,
config-file-based workflows, and example scripts. Semantic labels are required
for annotators such as semantic segmentation and bounding boxes to include
semantic information in the generated synthetic data. :contentReference[oaicite:1]{index=1}

Isaac Sim object-generation workflows can take a YAML scene description as input
and output RGB images, 2D/3D bounding boxes, and segmentation masks, which match
the needs of this benchmark. :contentReference[oaicite:2]{index=2}

This matches the required dataset fields in:

```text
configs/simulation_dataset.yaml
```

and the benchmark index schema in:

```text
src/cogar_seg/datasets/sim_robotic.py
```

## Dataset Target

Target size:

```text
500 RGB images
```

Recommended split:

| Split | Images | Purpose |
|---|---:|---|
| train | 100 | Small supervised subset for classical baselines |
| val | 100 | Sanity checks and prompt development |
| test | 300 | Main zero-shot benchmark |

The foundation models should be evaluated zero-shot. The train split is mainly
for classical baselines such as YOLOv8-seg, Mask R-CNN, and DeepLabV3+.

## Challenge Distribution

| Challenge | Target images |
|---|---:|
| reflective_metal | 100 |
| transparent_objects | 100 |
| partial_occlusion | 100 |
| small_parts | 100 |
| dynamic_scenes | 100 |

Each generated image should have one primary challenge and may have secondary
challenge tags.

Example:

```text
challenge_primary = "transparent_objects"
challenge_secondary = "partial_occlusion"
```

## Scene Types

Planned scene types:

| Scene type | Description |
|---|---|
| manipulation_table | Robot workspace with objects on a table |
| assembly_workspace | Screws, connectors, washers, and tools |
| inspection_scene | Reflective and transparent objects under varied lighting |
| cluttered_bin_or_table | Heavy occlusion and adjacent objects |
| dynamic_interaction | Moving object, changed object poses, or changed robot poses |

## Robot Context

The assignment suggests simulated robotic platforms such as Unitree A2 EDU or
Unitree G1 EDU. The first practical dataset version can use a simplified
robotic workspace and later add a more realistic Unitree-inspired platform.

Preferred context:

```text
Unitree G1 EDU-inspired humanoid manipulation workspace
```

Fallback context:

```text
Generic robotic arm or mobile-manipulator workspace
```

Robot-scene roles:

| Role | Description |
|---|---|
| static_context | Robot is visible near the table or workspace |
| occluder | Robot hand, gripper, or arm partially blocks objects |
| dynamic_actor | Robot or object changes pose between frames |
| camera_reference | Camera viewpoint approximates robot perception |

## Required Output Modalities

For each generated image, the exporter should produce:

| Output | Required | Use |
|---|---:|---|
| RGB image | Yes | Model input |
| Instance segmentation mask | Yes | Object-level IoU and mask AP |
| Semantic segmentation mask | Yes | Per-category IoU |
| 2D bounding boxes | Yes | SAM box prompts |
| Metadata | Yes | Challenge/category analysis |
| Depth | Optional | Future 3D extension |
| Camera pose | Optional | Reproducibility and robotics context |

## Semantic Categories

Foreground categories:

| ID | Name | Main challenge |
|---:|---|---|
| 1 | metal_tool | reflective_metal |
| 2 | metal_part | reflective_metal, small_parts |
| 3 | glass_cup | transparent_objects |
| 4 | transparent_box | transparent_objects |
| 5 | plastic_container | transparent_objects, reflective_metal |
| 6 | screw | small_parts |
| 7 | connector | small_parts |
| 8 | washer | small_parts |
| 9 | cable | partial_occlusion, small_parts |

Context categories:

| ID | Name | Role |
|---:|---|---|
| 10 | robot_gripper_or_hand | robot context / occluder |
| 11 | table | support surface |
| 12 | distractor_object | clutter |

## Isaac Sim Semantic Labels

Every foreground object must receive a semantic class label. For example:

```text
class: metal_tool
class: glass_cup
class: screw
```

The semantic labels must map exactly to the category names in:

```text
configs/simulation_dataset.yaml
```

This is necessary because Replicator annotators such as semantic segmentation
and bounding boxes depend on semantic labels. :contentReference[oaicite:3]{index=3}

## Scene Randomization

Each challenge category should randomize different factors. Domain
randomization should vary object poses, lighting conditions, textures, and
camera angles so the generated dataset is diverse and useful for perception
model evaluation. :contentReference[oaicite:4]{index=4}

### Reflective Metal

Randomize:

- Metallic material
- Roughness
- Lighting angle
- Specular highlights
- Object pose
- Camera viewpoint
- Background/table material

Primary objects:

- metal_tool
- metal_part

Expected failure modes:

- Reflections segmented as object parts
- Object boundaries blurred by specular highlights
- Confusion between shiny object and background

### Transparent Objects

Randomize:

- Transparency level
- Roughness
- Background color
- Object overlap
- Lighting intensity
- Camera viewpoint

Primary objects:

- glass_cup
- transparent_box
- plastic_container

Expected failure modes:

- Transparent object missed
- Background segmented through the object
- Incomplete object boundary

### Partial Occlusion

Randomize:

- Occluder object
- Occlusion percentage
- Robot hand/gripper pose
- Object overlap
- Clutter density
- Camera viewpoint

Primary objects:

- metal_tool
- glass_cup
- plastic_container
- connector

Expected failure modes:

- Only visible part segmented
- Target object merged with occluder
- Point prompt selects neighboring object

### Small Parts

Randomize:

- Object scale
- Object count
- Table position
- Camera distance
- Lighting
- Distractor objects

Primary objects:

- screw
- connector
- washer

Expected failure modes:

- Small object missed
- Boundary inaccurate
- Object confused with background texture
- Multiple small objects merged

### Dynamic Scenes

Randomize:

- Object pose between frames
- Robot pose between frames
- Moving object location
- Motion sequence ID
- Frame ID

Primary objects:

- metal_part
- cable
- moving distractor objects

Expected failure modes:

- Inconsistent segmentation across frames
- Motion-related blur
- Changed object position not handled well
- Object confused with robot part

## Image Resolution

Default image size:

```text
640 x 480
```

This matches the current simulation config and the validation logic.

If needed later, the benchmark can add higher-resolution exports such as:

```text
1280 x 720
```

but the first dataset should stay at 640 x 480 for faster iteration.

## Output Directory Mapping

Isaac Sim/Replicator exports should be converted into this repo format:

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
    categories.json
    scene_metadata.csv
    sim_robotic_scenes_index.csv

  metadata/
    generation_config.yaml
    generation_summary.json
```

## Benchmark Index Mapping

The final converted index must contain one row per object instance.

Required columns:

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

The index must validate with:

```text
cogar_seg.datasets.sim_robotic.validate_sim_index
```

## Prompt Generation Rules

### Box Prompt

Source:

```text
2D bounding box from ground-truth annotation
```

Format:

```text
bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
```

Coordinate convention:

```text
XYXY pixel coordinates
```

### Point Prompt

Source:

```text
center of object mask or bounding box
```

Default:

```text
point_x = center x of object bbox
point_y = center y of object bbox
```

Later improvements:

- Mask centroid
- Farthest-inside point
- Random foreground point
- Multiple positive points
- Positive point plus negative background points
- Box plus point

### Automatic Mask Generation

For automatic mask generation, the model predicts all candidate masks in the
image without a target-specific prompt.

The benchmark must then match predicted masks to ground-truth instances using
IoU-based matching.

Required outputs for automatic mode:

- best_match_iou
- matched_gt_object_id
- false_positive_count
- missed_gt_count
- mask_ap
- per_category_iou

## Initial Generation Strategy

The first real Isaac Sim generation should be small:

```text
5 images per challenge category
25 total images
```

Purpose:

- verify semantic labels
- verify RGB export
- verify instance mask export
- verify semantic mask export
- verify bounding-box export
- verify conversion into our benchmark index
- verify visualization script

After this works, scale to:

```text
100 images per challenge category
500 total images
```

## Generation Stages

### Stage 1 — Local sanity dataset

Target:

```text
25 images
```

Breakdown:

| Challenge | Images |
|---|---:|
| reflective_metal | 5 |
| transparent_objects | 5 |
| partial_occlusion | 5 |
| small_parts | 5 |
| dynamic_scenes | 5 |

Success condition:

- all exported images are readable
- all masks are readable
- semantic labels are correct
- bounding boxes are valid
- converted index passes validation
- visualizer works on several rows

### Stage 2 — Full dataset

Target:

```text
500 images
```

Breakdown:

| Challenge | Images |
|---|---:|
| reflective_metal | 100 |
| transparent_objects | 100 |
| partial_occlusion | 100 |
| small_parts | 100 |
| dynamic_scenes | 100 |

Success condition:

- all five challenge categories represented
- train/val/test split created
- no missing RGB/mask files
- one row per object instance in benchmark index
- no invalid bounding boxes
- no invalid prompt points
- visual inspection passes on random samples

## Train / Validation / Test Split

Recommended split:

```text
train: 100 images
val:   100 images
test:  300 images
```

Rules:

- Images from the same scene seed should stay in the same split when possible.
- Dynamic sequences should not be split frame-by-frame across train/val/test.
- The test split should contain all five challenge categories.
- Classical baselines may use only the train split.
- Zero-shot models should be evaluated primarily on validation and test.

## Conversion Requirement

The Isaac Sim raw export should not be used directly by the benchmark scripts.

Instead, we should create a converter:

```text
scripts/convert_isaac_export_to_sim_index.py
```

This converter should:

1. Read Isaac Sim/Replicator output folders.
2. Copy or map RGB images into `sim_dataset/images/{split}/`.
3. Convert instance masks into `sim_dataset/masks/instance/{split}/`.
4. Convert semantic masks into `sim_dataset/masks/semantic/{split}/`.
5. Extract or compute 2D bounding boxes.
6. Compute point prompts.
7. Assign challenge metadata.
8. Write `sim_robotic_scenes_index.csv`.
9. Validate the result with `validate_sim_index`.

## Raw Isaac Export Assumptions

The raw Isaac export may contain:

```text
raw_isaac_exports/
  rgb/
  semantic_segmentation/
  instance_segmentation/
  bounding_box_2d_tight/
  bounding_box_3d/
  camera_params/
  metadata/
```

The exact folder names may vary depending on the Replicator writer or IRO
configuration. The converter should therefore be configurable and should not
hard-code a single raw-export layout too early.

## File Naming Convention

Converted files should use deterministic names.

Example:

```text
img_000001.png
img_000001_obj_0001.png
img_000001_obj_0002.png
```

Recommended paths:

```text
sim_dataset/images/train/img_000001.png
sim_dataset/masks/instance/train/img_000001_obj_0001.png
sim_dataset/masks/semantic/train/img_000001.png
```

## Metadata Requirements

Each generated scene should store:

```text
scene_id
split
scene_type
challenge_primary
challenge_secondary
num_objects
camera_name
robot_context
random_seed
```

Optional metadata:

```text
camera_pose
object_pose
material_type
lighting_condition
occlusion_fraction
motion_sequence_id
```

## Quality Checks

Before using the dataset for benchmarking, run these checks:

- index has required columns
- all image paths exist
- all instance mask paths exist
- all semantic mask paths exist
- all bounding boxes have valid geometry
- all point prompts are inside the image
- all category IDs are known
- all challenge labels are known
- train/val/test split counts are correct
- random samples visualize correctly

## Success Criteria

The dataset generation block is complete when:

- At least 25 real Isaac Sim images are generated.
- Every image has RGB, semantic mask, instance mask, and object metadata.
- The converted benchmark index validates successfully.
- `scripts/visualize_sim_sample.py` works on multiple real rows.
- The index includes all five challenge categories.
- The pipeline can be scaled to 500 images.

The full simulation dataset is complete when:

- 500 generated images exist.
- All expected masks and annotations exist.
- The benchmark index validates.
- The dataset contains all five challenge categories.
- The test set contains 300 images.
- Visual sanity checks pass.
- The dataset is ready for SAM/SAM2/FastSAM/MobileSAM/EfficientSAM evaluation.

## Notes

The current dummy sample is only a local sanity check. It is not part of the
final assignment dataset.

The simulation benchmark should be added as a new layer on top of the existing
repository structure.

The current OCID benchmark should remain in the repository as a prototype and
sanity-check pipeline.

The next implementation step is to create:

```text
configs/isaac_sim_generation.yaml
```

Then we will add:

```text
scripts/convert_isaac_export_to_sim_index.py
```

to convert raw Isaac Sim/Replicator outputs into the benchmark format.