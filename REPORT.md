# Main Report: Foundation Model Segmentation Benchmark

Project: Foundation Model Segmentation for Robotic Scenes  
Student id: 5884715

## Scope

This report tracks the Assignment 2 benchmark workflow from dataset creation
through evaluation, speed analysis, failure analysis, and lightweight-model
edge-deployment comparison.

## Closure Status

Tasks 1-9 are complete as of 2026-06-17. The remaining non-benchmark project
item is publishing external download links for the generated datasets.

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
| Task 9: Lightweight SAM variants | Complete | MobileSAM and EfficientSAM variants were evaluated and joined with speed/checkpoint-size trade-off summaries. |

## Dataset Summary

| Dataset | Type | Status | Images | Location setting |
| --- | --- | --- | ---: | --- |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Validated locally | 1000 | `isaac_official_unitree_g1` |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Available locally | 1000 | `blenderproc_cogar_sim` |
| OCID | Real RGB-D clutter dataset | Available locally, external source | 2390 | `ocid` |

![Dataset examples](outputs/final_benchmark_assets/plots/dataset_examples.png)

Configured paths are in [configs/datasets.yaml](configs/datasets.yaml).

The Isaac dataset validation passed on 2026-06-13. The COCO export contains
1000 images, 72,695 annotations, and 16 categories.

The BlenderProc generator passed a 5-image smoke test on 2026-06-13. The full
dataset was generated, normalized, and copied to the configured local dataset
root with 1000 images and 8768 COCO annotations.

## Public Release

The generated Isaac and BlenderProc datasets are publicly released through
Zenodo:

https://doi.org/10.5281/zenodo.20736993

The Zenodo record contains the generated dataset archives, SHA256 checksum
files, and release manifests. The packaging workflow is documented in
[docs/datasets/public_release.md](docs/datasets/public_release.md), and the
configured release URLs are recorded in
[configs/datasets.yaml](configs/datasets.yaml).

OCID is not re-hosted by this project. The configured public source is the
upstream OCID project page:
https://www.acin.tuwien.ac.at/object-clutter-indoor-dataset/

## Benchmark Artifact Summary

| Stage | Compact artifact evidence |
| --- | --- |
| Prompt manifests | `outputs/task4_zero_shot_sam/prompts/summary.json` lists 1000 Isaac images, 1000 BlenderProc images, and 2390 OCID images. |
| Zero-shot evaluation | `outputs/task6_evaluation/zero_shot/summary.csv` contains 36 SAM/SAM2/FastSAM metric rows. |
| Baseline evaluation | `outputs/task6_evaluation/baselines/summary.csv` contains 9 YOLOv8-seg, Mask R-CNN, and DeepLabV3+ rows. |
| Inference speed | `outputs/task7_inference_speed/summary.csv` contains 90 GPU/CPU timing rows. |
| Failure analysis | `outputs/task8_failure_analysis/summary.json` records 10 representative failure visualizations and 151 challenge-group rows. |
| Lightweight SAM | `outputs/task9_lightweight_sam/summary/summary.json` records 72 quality rows, 144 speed-quality trade-off rows, and 36 recommendation rows. |
| Final recommendation guide | `outputs/final_benchmark_assets/recommendation_guide.md` summarizes model choices and links six aggregate plots. |

## Final Recommendations

The final recommendation guide is stored at
`outputs/final_benchmark_assets/recommendation_guide.md`.

The generated aggregate plots are embedded below.

![Zero-shot mIoU heatmap](outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png)

![Classical baseline mIoU bars](outputs/final_benchmark_assets/plots/baseline_miou_bars.png)

![GPU speed-quality trade-off](outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png)

![Lightweight SAM trade-off](outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png)

![Robotic challenge group performance](outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png)

![Best zero-shot model by dataset and prompt](outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png)

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
