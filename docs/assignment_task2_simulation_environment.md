# Assignment Task 2: Simulation Environment for Robotic Scene Generation

## Requirement

Use simulation environments such as Isaac Sim, Gazebo, and/or Rviz2 to generate
and organize the robotic scenes.

## Completion Status

Task 2 is complete with a reproducible BlenderProc simulation pipeline. Isaac
Sim / Replicator was also implemented and tested on AWS, but it was not used as
the final dataset generator because the available affordable GPU setup was not
reliable for producing benchmark-ready Replicator files.

The final 500-image COGAR-SimRobotics-500 dataset used in the current benchmark
tables was generated and organized with BlenderProc. The Isaac Sim work is
reported as an engineering feasibility attempt, not as a source of the final
benchmark data. Gazebo and Rviz2 were not used because the BlenderProc pipeline
already produced the required RGB images, instance masks, semantic/COCO-derived
annotations, metadata, and benchmark indexes.

This report is intentionally explicit about the environment choice: the final
results should not claim Isaac Sim, Gazebo, or Rviz2 execution when the frozen
dataset was generated with BlenderProc.

## Coverage Checklist

| Task 2 expectation | Covered? | Evidence in this project |
|---|---:|---|
| Use a simulation environment to generate robotic scenes | Yes | BlenderProc generated the frozen 500-image benchmark dataset. |
| Organize generated scenes into a benchmark dataset | Yes | RGB, masks, COCO-style annotations, metadata, object indexes, and splits are stored under `data/cogar_sim_500_final/`. |
| Include robotic-scene content | Yes | The dataset contains `robot_gripper` scenes and manipulation objects such as metal parts, glass objects, screws, connectors, cables, tools, boxes, and plastic objects. |
| Cover required visual challenges | Yes | Reflective metal, transparent glass, partial occlusion, small parts, and dynamic-scene groups are explicitly configured and reported. |
| Address Isaac Sim / Gazebo / Rviz2 scope | Addressed, not final | Isaac Sim was implemented and tested on AWS, then documented as impractical on the available affordable hardware; Gazebo and Rviz2 are scoped out because they were not needed for object-level dataset generation. |
| Provide reproducibility | Yes | Generation configs, scripts, normalization commands, validation commands, and AWS Isaac setup notes are tracked. |
| Avoid overclaiming | Yes | The report states that the current frozen benchmark was generated with BlenderProc, while Isaac Sim was attempted but not used for final data. |

## Visual Evidence

![Simulation dataset generation pipeline](/outputs/figures/final_report/dataset/simulation_pipeline.png)

*Figure: Simulation-to-benchmark pipeline used to turn synthetic scenes into RGB
images, masks, annotations, object indexes, and evaluation reports.*

![Representative generated scenes](/outputs/figures/final_report/dataset/sample_scene_montage.png)

*Figure: Representative generated scenes from the final benchmark dataset.*

| Reflective metal | Transparent glass |
|---|---|
| ![Reflective metal scene](/outputs/figures/final_report/dataset/sample_scenes/reflective_metal.png) | ![Transparent glass scene](/outputs/figures/final_report/dataset/sample_scenes/transparent_glass.png) |

| Partial occlusion | Small parts |
|---|---|
| ![Partial occlusion scene](/outputs/figures/final_report/dataset/sample_scenes/partial_occlusion.png) | ![Small parts scene](/outputs/figures/final_report/dataset/sample_scenes/small_parts.png) |

| Dynamic scene | Robot gripper scene |
|---|---|
| ![Dynamic scene](/outputs/figures/final_report/dataset/sample_scenes/dynamic_scene.png) | ![Robot gripper scene](/outputs/figures/final_report/dataset/sample_scenes/robot_gripper.png) |

## Selected Environment

| Environment | Used for final dataset? | Role in this project |
|---|---:|---|
| BlenderProc | Yes | Main reproducible synthetic-data generation environment. |
| Isaac Sim | Attempted, not final | AWS Replicator workflow was implemented and tested, but not used for the frozen dataset. |
| Gazebo | No | Not required for the final dataset because object-level RGB/mask/COCO outputs were produced through BlenderProc. |
| Rviz2 | No | Not required because this project needed dataset generation, not ROS visualization. |

BlenderProc was selected because it supports scripted scene construction,
camera randomization, physically based rendering, instance segmentation, and
COCO-style annotation export in a workflow that was practical on the available
hardware.

## Reproducible Simulation Files

The simulation workflow is represented by these tracked files:

| File | Purpose |
|---|---|
| `configs/blenderproc_dataset.yaml` | Main simulation-generation configuration for COGAR-SimRobotics-500. |
| `scripts/blenderproc/generate_cogar_sim_500.py` | BlenderProc entry point for generating randomized scenes. |
| `src/cogar_seg/generation/blenderproc_scene.py` | Reusable scene-generation implementation. |
| `docs/blenderproc_cogar_sim_500.md` | Short BlenderProc generation notes. |
| `docs/dataset_quality_workflow.md` | Dataset build, audit, filtering, and validation workflow. |
| `configs/isaac_sim_dataset.yaml` | Isaac Sim/Replicator configuration prepared for AWS testing. |
| `scripts/isaac_sim/generate_cogar_isaac_sim_500.py` | Isaac Sim entry point used during AWS feasibility tests. |
| `src/cogar_seg/generation/isaac_sim_scene.py` | Reusable Isaac Sim scene-generation implementation kept for reproducibility and future work. |
| `scripts/aws/run_isaac_sim_dataset_aws.sh` | AWS Docker wrapper used for Isaac Sim setup, smoke tests, and troubleshooting. |
| `docs/aws_isaac_sim_dataset.md` | AWS instructions and troubleshooting notes for the Isaac Sim route. |
| `docs/isaac_sim_setup.md` | Isaac Sim setup note and hardware explanation. |

## Generation Configuration

The main generation config is:

```text
configs/blenderproc_dataset.yaml
```

Key settings:

| Setting | Value |
|---|---:|
| Dataset name | `COGAR-SimRobotics-500` |
| Output directory | `data/cogar_sim_500` |
| Final target images | 500 |
| Image width | 640 |
| Image height | 480 |
| Random seed | 42 |
| Render samples | 32 |
| Final base scenes | 50 |
| Final captures per scene | 10 |

Challenge plan:

| Challenge | Target images |
|---|---:|
| `reflective_metal` | 100 |
| `transparent_glass` | 100 |
| `partial_occlusion` | 100 |
| `small_parts` | 100 |
| `dynamic_scene` | 100 |

## Generation Command

The dataset candidates are generated through the BlenderProc launcher:

```bash
source ~/blenderproc_test/.venv/bin/activate

blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 650 \
  --raw-dataset-name pilot_v5_650_final_candidates
```

The final benchmark was produced from more candidates than needed, then audited
and filtered down to 500 clean images.

## Organization Pipeline

The generated scenes are not just raw images. They are organized into a
benchmark-ready dataset with object-level annotations and prompt fields.

Pipeline:

1. Generate randomized simulated scenes with BlenderProc.
2. Normalize raw BlenderProc outputs into project dataset folders.
3. Export or derive RGB images, masks, metadata, and COCO annotations.
4. Build an object-level CSV index.
5. Export per-object binary masks.
6. Audit image quality and object statistics.
7. Filter bad frames and non-target support surfaces.
8. Freeze the final 500-image dataset.
9. Derive the YOLOv8-seg export from the frozen dataset.

Main commands after generation:

```bash
PYTHONPATH=src python scripts/dataset/normalize_cogar_sim_500.py

PYTHONPATH=src python scripts/dataset/create_object_index.py \
  --dataset cogar_sim_500 \
  --coco data/cogar_sim_500/annotations/instances_all.json \
  --metadata data/cogar_sim_500/metadata/frame_index.csv \
  --rgb-dir data/cogar_sim_500/rgb \
  --output outputs/indexes/cogar_sim_500_objects.csv

PYTHONPATH=src python scripts/dataset/export_binary_masks.py \
  --index outputs/indexes/cogar_sim_500_objects.csv \
  --output-dir data/cogar_sim_500/instance_masks/all
```

Final validation:

```bash
PYTHONPATH=src python scripts/dataset/validate_sim_index.py \
  --index data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv
```

## Final Dataset Produced By The Simulation Pipeline

Final dataset root:

```text
data/cogar_sim_500_final/
```

Final object-level benchmark index:

```text
data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv
```

Final dataset summary:

| Property | Value |
|---|---:|
| RGB images | 500 |
| Annotated object instances | 4,471 |
| Object categories | 9 |
| Robotic challenge groups | 5 |
| Image size | 640 x 480 |

Expected final layout:

```text
data/cogar_sim_500_final/
  annotations/
    instances_all.json
    sim_robotic_scenes_index_final_filtered.csv
  instance_masks/
    final/
  metadata/
    categories.json
    frame_index.csv
  rgb/
  splits/
```

YOLO export derived from the same frozen dataset:

```text
data/yolo_cogar_sim_500_final/
```

## Robotic Scene Content

The simulated scenes were organized around robotic tabletop perception and
manipulation challenges:

| Challenge type | Simulation purpose |
|---|---|
| `reflective_metal` | Tests segmentation under specular surfaces and unstable visual edges. |
| `transparent_glass` | Tests transparent or weak-boundary object segmentation. |
| `partial_occlusion` | Tests object segmentation when objects are blocked by clutter or robot geometry. |
| `small_parts` | Tests screws, connectors, cables, and thin/fine structures. |
| `dynamic_scene` | Tests robustness under changing object and robot-layout configurations. |

The dataset includes manipulation-relevant categories:

```text
robot_gripper, metal_part, glass_object, plastic_object, screw,
connector, cable, tool, box
```

## Why BlenderProc Satisfies Task 2

The requirement says to use simulation environments such as Isaac Sim, Gazebo,
and/or Rviz2. The phrase "such as" allows an equivalent simulation environment
when it produces the needed robotic-scene data.

BlenderProc satisfies the practical dataset-generation requirement because it
produces:

- simulated RGB images;
- object-level masks;
- category labels;
- bounding boxes;
- prompt points;
- COCO-derived annotations;
- train/validation/test splits;
- reproducible randomized scene generation;
- organized benchmark indexes consumed by the evaluation scripts.

For this project, the critical deliverable is an organized simulated robotic
scene dataset for segmentation benchmarking. BlenderProc produced that dataset.

## Isaac Sim Feasibility Attempt

The repository provides an Isaac Sim dataset path, separate from the frozen
benchmark data:

```text
data/cogar_isaac_sim_500/
```

Recommended hardware for a practical future run:

```text
g6e.4xlarge preferred, g6e.2xlarge minimum, g6e.8xlarge if budget/quota allows
```

Main validation commands used during the AWS attempt:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh diagnose
bash scripts/aws/run_isaac_sim_dataset_aws.sh compat
FRAMES=1 PROGRESS_EVERY=1 \
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke1
```

The intended Isaac workflow would write raw Replicator outputs plus metadata:

```text
data/cogar_isaac_sim_500/raw_replicator/final_500/
data/cogar_isaac_sim_500/metadata/frame_index.csv
data/cogar_isaac_sim_500/metadata/categories.json
data/cogar_isaac_sim_500/metadata/dataset_summary.json
```

This route is kept for transparency and future work, but it is not used as a
final deliverable in this submission. If `data/cogar_isaac_sim_500/` ever
becomes the primary dataset, Tasks 4-9 must be rerun on the Isaac-generated
annotations.

## AWS Isaac Sim Attempt And Practical Outcome

The Isaac Sim path was exercised on the available AWS Tesla T4 machine. The
machine was sufficient for SAM-family inference and OCID benchmarking, but it
was not practical for full Isaac Sim data generation.

What was validated:

- Docker and NVIDIA Container Toolkit could expose the GPU to containers.
- The Isaac Sim 6.0 container could be pulled.
- Isaac cache permissions had to be prepared for container UID `1234`.
- The Omniverse Hub cache container was needed to avoid repeated
  `OmniHub: Hub failed to launch` startup loops.
- The AWS runner had to force `/isaac-sim/python.sh` as the Docker entrypoint;
  otherwise the container launched the full streaming app wrapper.
- Low-resolution emergency smoke tests could start Isaac Sim and reach frame
  capture, but Replicator writer finalization did not reliably produce saved
  benchmark image files on the T4 machine.

Observed limitation:

```text
Tesla T4 could start Isaac Sim and execute the prepared Python scene path, but
it was not reliable enough to produce the required saved RGB/mask/annotation
dataset through Replicator.
```

Decision:

```text
Keep data/cogar_sim_500_final/ as the frozen reported benchmark dataset.
Keep the Isaac Sim workflow only as documented future work.
Do not spend more cloud budget for Task 2; close the assignment task with the
completed BlenderProc simulation pipeline.
```

## Isaac Sim, Gazebo, And Rviz2 Scope

Isaac Sim was considered the preferred future environment because Replicator is
well suited for robotic synthetic-data generation. The repository keeps the
Isaac Sim configuration, generator, and AWS runner as a record of the attempt:

```text
configs/isaac_sim_dataset.yaml
scripts/isaac_sim/generate_cogar_isaac_sim_500.py
scripts/aws/run_isaac_sim_dataset_aws.sh
docs/isaac_sim_setup.md
```

It was not used for the currently reported 500-image benchmark because the
available local and affordable AWS hardware was not suitable for reliably
writing Replicator dataset files. A future Isaac dataset should be reported
separately unless all segmentation models and baseline results are rerun on
that data.

Gazebo and Rviz2 were not used because they are better suited here for robot
simulation/visualization than for producing the final object-level RGB/mask/COCO
dataset already generated through BlenderProc.

## Task 2 Output

Task 2 produced the organized simulation dataset used by later stages:

- Task 4 zero-shot prompt and automatic-mask evaluations.
- Task 5 supervised baseline training and comparison.
- Task 6 mIoU, boundary F1, mask AP, and per-category analysis.
- Task 7 FPS measurement.
- Task 8 failure-mode visualization.
- Task 9 lightweight SAM trade-off analysis.

## Conclusion

Task 2 is complete.

The project used BlenderProc as the simulation environment to generate and
organize the currently frozen 500-image COGAR-SimRobotics-500 benchmark
dataset. Isaac Sim was attempted and documented, but the final Task 2 evidence
is the completed BlenderProc simulation pipeline and the organized dataset it
produced.
