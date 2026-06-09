# Assignment Task 2: Simulation Environment for Robotic Scene Generation

## Requirement

Use simulation environments such as Isaac Sim, Gazebo, and/or Rviz2 to generate and organize the robotic scenes.

## Completion status

Task 2 is complete with a BlenderProc simulation pipeline.

The final dataset was generated and organized with BlenderProc. Isaac Sim was
kept as a documented preferred alternative because it is well suited to
Replicator-based robotic synthetic data, but it was not used for the final
500-image run on the available GTX 1050 4 GB laptop. Gazebo and Rviz2 were not
used because the BlenderProc simulation pipeline already produced the required
RGB images, masks, COCO-style annotations, metadata, and benchmark index.

## Visual evidence

![Simulation dataset generation pipeline](/outputs/figures/final_report/dataset/simulation_pipeline.png)

*Figure: Documented simulation-to-benchmark pipeline used to turn synthetic scenes into RGB images, masks, annotations, object indexes, and evaluation reports.*

![Representative generated scenes](/outputs/figures/final_report/dataset/sample_scene_montage.png)

*Figure: Representative generated scenes from the final benchmark dataset.*

## Selected simulation environment

| Environment | Used | Role |
|---|---:|---|
| BlenderProc | Yes | Main reproducible simulation and synthetic data generation environment |
| Isaac Sim | Documented alternative | Preferred future route for RTX-capable hardware and Unitree-style scenes |
| Gazebo | No | Not required for final dataset generation |
| Rviz2 | No | Not required for final dataset generation |

BlenderProc was selected for the final run because it was practical on the
available machine and provides scripted scene generation, RGB rendering,
instance segmentation, COCO annotations, and reproducible randomized robotic
tabletop scenes.

Isaac Sim remains documented in `configs/isaac_sim_dataset.yaml` and
`docs/isaac_sim_setup.md` as the preferred future extension for a workstation or
cloud machine with an RTX-capable GPU.

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

## Why the BlenderProc simulation pipeline satisfies the task

The final pipeline satisfies the simulation requirement because it produces
simulation-generated RGB images and object-level annotations, which are then
organized into a benchmark-ready CSV index.

The dataset supports the assignment goal because it includes robotic-scene
objects, manipulation-related categories, and challenge types that are difficult
for segmentation models in robotic perception.

## Relation to Gazebo and Rviz2

Gazebo and Rviz2 were not used in the final dataset generation. Isaac Sim was
not run locally because the available GPU was below the practical requirement
for the intended Replicator workflow. The final pipeline instead uses
BlenderProc as the simulation generator and documents the Isaac Sim migration
path for future RTX-capable hardware.

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

Task 2 is complete with the documented BlenderProc simulation workflow.

BlenderProc was used as the simulation environment to generate and organize the
robotic scenes. The output is a structured 500-image simulated robotic-scene
dataset with 4,471 annotated object instances, object masks, bounding boxes,
prompt points, category labels, challenge labels, and train/validation/test
splits. Isaac Sim remains documented as the preferred extension route rather
than claimed as the completed generator.
