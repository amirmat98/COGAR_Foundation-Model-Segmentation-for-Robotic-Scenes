# Main Report: Foundation Model Segmentation Benchmark

Project: Foundation Model Segmentation for Robotic Scenes  
Student id: 5884715

## Scope

This report tracks the Assignment 2 benchmark workflow. Tasks 1-3 cover
dataset preparation, simulation environment use, and robotic platform use.
Task 4 starts the zero-shot SAM-family inference stage.

## Closure Status

Tasks 1-3 are complete as of 2026-06-13 based on the local dataset audit.

| Task | Status | Evidence |
| --- | --- | --- |
| Task 1: Dataset creation/curation | Complete | Isaac and BlenderProc synthetic datasets are available locally with segmentation annotations and the required challenge coverage. |
| Task 2: Simulation environment | Complete | Isaac Sim and BlenderProc generation pipelines are configured and documented. |
| Task 3: Robotic platform | Complete | The main Isaac dataset uses the official Unitree G1 USD asset for all 1000 frames. |

## Dataset Summary

| Dataset | Type | Status | Images | Location setting |
| --- | --- | --- | ---: | --- |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Validated locally | 1000 | `isaac_official_unitree_g1` |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Available locally | 1000 | `blenderproc_cogar_sim` |
| OCID | Real RGB-D clutter dataset | Available locally, external source | 2390 | `ocid` |

Configured paths are in [configs/datasets.yaml](configs/datasets.yaml).

The Isaac dataset validation passed on 2026-06-13. The COCO export contains
1000 images, 72,695 annotations, and 16 categories.

The BlenderProc generator passed a 5-image smoke test on 2026-06-13. The full
dataset was generated, normalized, and copied to the configured local dataset
root with 1000 images and 8768 COCO annotations.

## Public Release

The generated Isaac and BlenderProc datasets need public release URLs before
final submission. The preferred release route is Zenodo because it provides a
citable DOI. Hugging Face Datasets is also suitable for ML users. Google Drive
is acceptable for private/supervisor sharing but is weaker for citation.

Public URLs are currently marked as `TODO_PUBLIC_URL` in
[configs/datasets.yaml](configs/datasets.yaml).

## Task Reports

- [Task 1](docs/tasks/task1_dataset_creation.md)
- [Task 2](docs/tasks/task2_simulation_environment.md)
- [Task 3](docs/tasks/task3_robotic_platform.md)
- [Task 4](docs/tasks/task4_zero_shot_sam.md)
