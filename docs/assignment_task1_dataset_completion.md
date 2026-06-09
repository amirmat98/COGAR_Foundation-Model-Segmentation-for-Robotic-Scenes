# Assignment Task 1: Simulated Robotic Scene Dataset

## Requirement

Create or curate a robotic scene dataset in simulation with diverse challenges, including reflective metal, transparent glass, partial occlusions, small screws/connectors, and moving objects, with about 500 annotated images.

## Completion status

Task 1 is complete.

The project created a final simulated robotic-scene dataset for zero-shot segmentation benchmarking. The dataset is the primary dataset used in this assignment and is based on the final 500-image BlenderProc simulation benchmark.

Legacy OCID prototype data is not included in this Task 1 report. It is reserved only as historical context or for a later external real-world generalization test after the main 500-image benchmark is completed.

## Visual evidence

![Representative simulated robotic scenes](/outputs/figures/final_report/dataset/sample_scene_montage.png)

*Figure: Sample scenes from the final simulated benchmark, selected from existing local RGB images and saved as lightweight report thumbnails.*

![Object category counts](/outputs/figures/final_report/dataset/category_counts.png)

*Figure: Final object-instance counts by category for the 4,471-object benchmark index.*

![Challenge distribution](/outputs/figures/final_report/dataset/challenge_distribution.png)

*Figure: Final challenge distribution across small parts, partial occlusion, dynamic scenes, reflective metal, and transparent glass.*

## Final dataset location

Dataset root:

- `data/cogar_sim_500_final/`

Final annotation index:

- `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

## Dataset summary

| Property | Value |
|---|---:|
| Number of RGB images | 500 |
| Number of annotated object instances | 4,471 |
| Number of object categories | 9 |
| Number of robotic challenge groups | 5 |
| Dataset type | Simulated robotic scenes |
| Main benchmark use | Zero-shot segmentation evaluation |
| Annotation level | Instance-level masks, bounding boxes, prompt points, category labels, challenge labels |

## Object categories

The dataset contains 9 object categories relevant to robotic manipulation and scene understanding.

| Category | Object count |
|---|---:|
| robot_gripper | 1042 |
| plastic_object | 627 |
| metal_part | 555 |
| connector | 531 |
| screw | 427 |
| glass_object | 360 |
| box | 352 |
| tool | 296 |
| cable | 281 |

These categories were selected to represent common robotic perception targets in manipulation scenes, including tools, small hardware parts, cables, transparent objects, metallic objects, and the robot end-effector.

## Robotic perception challenge groups

The dataset explicitly includes the challenge types required by the assignment.

| Challenge | Object count |
|---|---:|
| small_parts | 1269 |
| partial_occlusion | 920 |
| dynamic_scene | 797 |
| reflective_metal | 743 |
| transparent_glass | 742 |

## Challenge design

### Reflective metal

Reflective metallic objects are included through the `metal_part` category and the `reflective_metal` challenge label. These objects test whether segmentation models can handle specular highlights, unstable edges, and appearance changes caused by lighting.

### Transparent glass

Transparent or glass-like objects are included through the `glass_object` category and the `transparent_glass` challenge label. These objects test whether models can segment objects whose boundaries are weak, partially see-through, or visually mixed with the background.

### Partial occlusion

Partial occlusion is included through the `partial_occlusion` challenge label. This tests whether models can segment objects that are partly blocked by other objects or by the robot gripper.

### Small screws and connectors

Small parts are represented by categories such as `screw`, `connector`, and `cable`, as well as the `small_parts` challenge label. These objects test model performance on small object areas, thin structures, and fine object boundaries.

### Dynamic scenes

Dynamic or changing configurations are represented by the `dynamic_scene` challenge label. These scenes test whether segmentation models remain robust when object and robot configurations vary across frames.

## Dataset split

The dataset includes train, validation, and test split labels.

| Split | Object count |
|---|---:|
| train | 3120 |
| val | 672 |
| test | 679 |

The split supports two types of evaluation:

- Zero-shot foundation-model evaluation without training on the dataset.
- Small-subset supervised fine-tuning for baseline comparison, such as YOLOv8n-seg.

## Annotation fields

The final annotation CSV includes the fields needed for object-level segmentation evaluation and prompt-based model testing.

Important fields include:

| Field | Purpose |
|---|---|
| `image_id` | Unique image identifier |
| `file_name` | Image filename |
| `scene_id` | Simulated scene identifier |
| `frame_id` | Frame identifier |
| `split` | Train/validation/test split |
| `image_path` | RGB image path |
| `binary_mask_path` | Per-object binary mask path |
| `instance_mask_path` | Instance mask path |
| `semantic_mask_path` | Semantic mask path |
| `category_id` | Numeric object category ID |
| `category_name` | Object category name |
| `object_id` | Object instance ID |
| `bbox_xmin` | Bounding-box minimum x-coordinate |
| `bbox_ymin` | Bounding-box minimum y-coordinate |
| `bbox_xmax` | Bounding-box maximum x-coordinate |
| `bbox_ymax` | Bounding-box maximum y-coordinate |
| `point_x` | Object prompt point x-coordinate |
| `point_y` | Object prompt point y-coordinate |
| `challenge_primary` | Main challenge label |
| `challenge_secondary` | Secondary challenge label |
| `is_reflective` | Reflective-object flag |
| `is_transparent` | Transparent-object flag |
| `is_occluded` | Occlusion flag |
| `is_small_part` | Small-part flag |
| `is_dynamic` | Dynamic-scene flag |
| `area` | Object mask area |

## Prompt compatibility

The dataset was designed to support the required prompt-based segmentation protocols.

### Box prompts

Bounding boxes are available through:

- `bbox_xmin`
- `bbox_ymin`
- `bbox_xmax`
- `bbox_ymax`

These fields were used for box-prompt evaluation of SAM-style models.

### Point prompts

Object prompt points are available through:

- `point_x`
- `point_y`

These fields were used for point-prompt evaluation, especially for SAM ViT-B.

### Automatic mask generation

The RGB images and ground-truth masks support automatic-mask-generation evaluation by matching predicted masks against ground-truth object masks.

## Why this dataset satisfies Task 1

This dataset satisfies Task 1 because it provides:

- Approximately 500 annotated simulated robotic-scene images.
- Instance-level object masks for segmentation evaluation.
- Bounding boxes for box-prompt segmentation.
- Point prompts for point-prompt segmentation.
- Category labels for per-category IoU analysis.
- Challenge labels for robustness analysis.
- Explicit coverage of reflective metal, transparent glass, partial occlusions, small screws/connectors, and dynamic scenes.
- A train/validation/test split for both zero-shot evaluation and small-subset supervised baselines.

## Dataset role in the final benchmark

This dataset is the primary benchmark dataset for Assignment 2.

It is used to evaluate:

- SAM ViT-B
- SAM ViT-H subset
- SAM2.1-Tiny
- FastSAM-S
- MobileSAM
- EfficientSAM-Ti
- Mask R-CNN ResNet-50 FPN
- YOLOv8n-seg supervised fine-tuned baseline

The dataset is also used for:

- Overall mIoU analysis
- Boundary F1 analysis
- Per-category IoU analysis
- Per-challenge robustness analysis
- Failure-mode visualization
- Model recommendation for robotic deployment scenarios

## Task 1 conclusion

Task 1 is complete.

A 500-image simulated robotic-scene dataset was created and organized for zero-shot segmentation benchmarking. It contains 4,471 annotated object instances across 9 robotic object categories and 5 challenge groups. The dataset directly covers the required robotic perception challenges: reflective metal, transparent glass, partial occlusion, small screws/connectors, and dynamic scenes.

This dataset is sufficient as the main simulated benchmark dataset for the assignment.
