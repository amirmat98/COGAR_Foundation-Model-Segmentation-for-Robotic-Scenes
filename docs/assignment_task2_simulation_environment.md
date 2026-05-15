# Assignment Task 2: Simulation Environment for Robotic Scene Generation

## Requirement

Use simulation environments such as Isaac Sim, Gazebo, and/or Rviz2 to generate and organize the robotic scenes.

## Completion status

Task 2 is complete.

The project uses Isaac Sim as the simulation environment for generating and organizing the robotic scene dataset. Gazebo and Rviz2 were not used in the final pipeline because Isaac Sim already provides the required tools for synthetic robotic scene generation, semantic annotation, domain randomization, and dataset export.

## Selected simulation environment

| Environment | Used | Role |
|---|---:|---|
| Isaac Sim | Yes | Main simulation and synthetic data generation environment |
| Gazebo | No | Not required for final dataset generation |
| Rviz2 | No | Not required for final dataset generation |

Isaac Sim was selected because it is well suited for synthetic data generation in robotic perception. It supports simulated scenes, camera capture, object-level semantic information, and Replicator-based synthetic data generation workflows.

## Dataset generated from simulation

The final simulated dataset is:

- `data/cogar_sim_500_final/`

The final annotation index is:

- `data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv`

The dataset contains:

| Property | Value |
|---|---:|
| RGB images | 500 |
| Annotated object instances | 4,471 |
| Object categories | 9 |
| Challenge groups | 5 |

## Organization of generated scenes

The generated scenes were organized into a structured dataset format containing:

- RGB image paths
- binary object mask paths
- instance mask paths
- semantic mask paths
- category IDs and names
- object IDs
- bounding boxes
- point prompts
- challenge labels
- train/validation/test split labels

The final CSV index provides one row per object instance. This structure makes the dataset directly usable for object-level segmentation benchmarking with promptable foundation models.

## Simulation diversity

The simulation dataset was designed to include diverse robotic perception conditions:

| Challenge type | Purpose |
|---|---|
| reflective_metal | Tests segmentation under specular reflections and unstable object boundaries |
| transparent_glass | Tests segmentation of transparent or weak-boundary objects |
| partial_occlusion | Tests segmentation when objects are partially blocked |
| small_parts | Tests segmentation of screws, connectors, cables, and other small objects |
| dynamic_scene | Tests robustness under changing object or robot configurations |

## Why Isaac Sim satisfies the task

Isaac Sim satisfies the simulation requirement because it provides a complete synthetic data generation workflow for perception datasets. The final dataset uses simulation-generated RGB images and object-level annotations, which are then organized into a benchmark-ready CSV index.

The use of Isaac Sim also supports the assignment goal because the dataset includes robotic-scene objects, manipulation-related categories, and challenge types that are difficult for segmentation models in robotic perception.

## Relation to Gazebo and Rviz2

Gazebo and Rviz2 were not used in the final dataset generation. This is acceptable because the assignment asks to use simulation environments such as Isaac Sim, Gazebo, and/or Rviz2. The final pipeline uses Isaac Sim, which is the preferred option mentioned in the task.

## Output of Task 2

Task 2 produced the organized simulated benchmark dataset used by all later evaluation stages:

- SAM ViT-B evaluation
- SAM ViT-H subset evaluation
- FastSAM-S evaluation
- MobileSAM evaluation
- EfficientSAM-Ti evaluation
- YOLOv8n-seg fine-tuned baseline evaluation
- per-category analysis
- per-challenge analysis
- failure-mode visualization

## Task 2 conclusion

Task 2 is complete.

Isaac Sim was used as the simulation environment to generate and organize the robotic scenes. The output is a structured 500-image simulated robotic-scene dataset with 4,471 annotated object instances, object masks, bounding boxes, prompt points, category labels, challenge labels, and train/validation/test splits.
