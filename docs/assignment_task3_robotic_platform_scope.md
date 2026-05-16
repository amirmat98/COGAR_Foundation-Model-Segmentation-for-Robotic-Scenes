# Assignment Task 3: Simulated Robotic Platform and Benchmark Scene Design

## Requirement

Use simulated robotic platforms such as Unitree As2 EDU or Unitree G1 EDU to produce meaningful benchmark scenes for segmentation.

## Completion status

Task 3 is partially complete and documented honestly.

The final 500-image benchmark dataset was generated as simulated robotic manipulation/perception scenes. These scenes include a visible robot gripper, manipulation-relevant objects, clutter, occlusion, reflective metal, transparent glass, small parts, and dynamic configurations.

A full Unitree G1 or Unitree As2 embodied simulation was investigated but was not run locally because the available laptop GPU, NVIDIA GTX 1050 with 4 GB VRAM, is below practical Isaac Sim / Isaac Lab rendering requirements.

## Visual evidence

![Robot gripper sample scene](/outputs/figures/final_report/dataset/sample_scenes/robot_gripper.png)

*Figure: Representative simulated manipulation scene containing the `robot_gripper` category from the final benchmark dataset.*

![Robot gripper failure example](/outputs/figures/final_report/failure_modes/worst_04_iou_0.007_robot_gripper_dynamic_scene.png)

*Figure: Example gripper/dynamic-scene failure panel showing why articulated robot parts are a documented segmentation challenge.*

## Main benchmark scene design

The main dataset remains:

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
| Robot-related category | `robot_gripper` |
| Robot gripper instances | 1,042 |

## Robotic platform representation in the main dataset

The dataset includes a simulated robot gripper and robotic manipulation workspace. This provides meaningful robotic segmentation scenes because the robot end-effector appears in the images and interacts visually with the objects.

| Element | Dataset representation |
|---|---|
| Robot component | `robot_gripper` category |
| Manipulation workspace | Simulated tabletop/industrial robotic scenes |
| Manipulable objects | tools, screws, connectors, cables, boxes, plastic objects |
| Difficult robotic materials | reflective metal and transparent glass |
| Occlusion cases | partial object occlusion and gripper-object occlusion |
| Dynamic configurations | changing object/robot scene arrangements |

## Why the scenes are meaningful for segmentation

The benchmark scenes test segmentation problems that are directly relevant to robotic perception:

- separating robot gripper pixels from manipulated objects
- segmenting small screws and connectors
- segmenting thin cables
- handling partial object occlusion
- handling reflective metal surfaces
- handling transparent glass-like objects
- maintaining performance across changing scene configurations

These conditions are common in robotic manipulation, inspection, picking, and scene understanding.

## Unitree G1 / As2 scope clarification

A full Unitree G1 EDU or Unitree As2 EDU embodied simulation was not included in the final 500-image dataset.

Instead, the project uses a simulated robotic manipulation setup with a visible robot gripper. Therefore, the main benchmark should be interpreted as a robotic manipulation/perception benchmark rather than a full Unitree locomotion or humanoid perception benchmark.

## Route A investigation

Route A was to use Isaac Lab with a Unitree G1 environment.

This was considered because Isaac Lab provides Unitree G1 pick-and-place environments. However, the local system did not have Isaac Sim or Isaac Lab installed, and the available GTX 1050 4 GB GPU is below the practical requirements for full Isaac Sim / Isaac Lab rendering workflows.

Therefore, Route A was not used for the final benchmark.

## Route B platform asset preparation

As a reproducible alternative, the project includes a source script for preparing Unitree G1 robot-description assets outside the repository:

- `scripts/simulation/prepare_unitree_platform_assets.sh`

This script downloads public Unitree G1 robot-description assets into:

- `~/Desktop/COGAR/external/g1_description/`

These assets include URDF/MJCF robot descriptions that can be imported into Isaac Sim in a future extension using the Isaac Sim URDF importer.

The external robot assets are not committed to the repository.

## Future work

Future work can add a full Unitree G1 or Unitree As2 platform subset when suitable hardware is available.

Recommended future extension:

1. Import Unitree G1 or Unitree As2 URDF/USD into Isaac Sim.
2. Place the robot in robotic clutter/manipulation scenes.
3. Attach or position RGB/RGB-D cameras from robot-view or third-person viewpoints.
4. Generate 20–50 additional platform-specific images.
5. Export RGB images, segmentation masks, bounding boxes, and metadata.
6. Evaluate the existing SAM/FastSAM/MobileSAM/EfficientSAM pipeline on the platform-specific subset.

## Task 3 conclusion

Task 3 is partially satisfied.

The final dataset contains meaningful simulated robotic manipulation scenes with a visible robot gripper and robotics-relevant objects. A full Unitree G1/As2 embodied simulation was not run due to local hardware limitations. To address this transparently, the project documents the limitation and provides a reproducible Unitree G1 asset-preparation route for future Isaac Sim import.
