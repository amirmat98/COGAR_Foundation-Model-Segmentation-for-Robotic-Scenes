# Main Report: Foundation Model Segmentation Benchmark

Project: Foundation Model Segmentation for Robotic Scenes  
Student id: 5884715

## Scope

This report tracks the Assignment 2 benchmark workflow from dataset creation
through evaluation, speed analysis, failure analysis, and lightweight-model
edge-deployment comparison.

## Closure Status

Tasks 1-8 are complete. Task 9 is prepared and ready to run with MobileSAM and
EfficientSAM variants.

| Task | Status | Evidence |
| --- | --- | --- |
| Task 1: Dataset creation/curation | Complete | Isaac and BlenderProc synthetic datasets are available locally with segmentation annotations and the required challenge coverage. |
| Task 2: Simulation environment | Complete | Isaac Sim and BlenderProc generation pipelines are configured and documented. |
| Task 3: Robotic platform | Complete | The main Isaac dataset uses the official Unitree G1 USD asset for all 1000 frames. |
| Task 4: Zero-shot SAM inference | Complete | SAM ViT-H, SAM ViT-B, SAM2, and FastSAM predictions were generated for point, box, and automatic prompts. |
| Task 5: Classical baselines | Complete | YOLOv8-seg, Mask R-CNN, and DeepLabV3+ were trained on small subsets. |
| Task 6: Evaluation metrics | Complete | mIoU, boundary F1, mask AP, and per-category/challenge metrics were produced. |
| Task 7: Inference speed | Complete | GPU and CPU speed summaries were produced for zero-shot and baseline models. |
| Task 8: Failure modes | Complete | Qualitative failure examples and failure-mode summaries were generated. |
| Task 9: Lightweight SAM variants | Ready to run | MobileSAM and EfficientSAM configs, adapters, speed/evaluation setup, and summary tooling are prepared. |

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
- [Task 5](docs/tasks/task5_classical_baselines.md)
- [Task 6](docs/tasks/task6_evaluation.md)
- [Task 7](docs/tasks/task7_inference_speed.md)
- [Task 8](docs/tasks/task8_failure_analysis.md)
- [Task 9](docs/tasks/task9_lightweight_sam.md)
