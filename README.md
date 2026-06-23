# Foundation Model Segmentation for Robotic Scenes

**Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception  
**Student id:** 5884715  

This repository contains a simulation-based benchmark for testing whether
foundation segmentation models can support robotic scene understanding. The
project evaluates SAM-family models, lightweight SAM variants, and supervised
segmentation baselines on robotic scenes containing transparent objects,
reflective metal, partial occlusions, small parts, robot-body visibility,
clutter, and dynamic objects.

The two main entry points are:

| File | Purpose |
| --- | --- |
| [README.md](README.md) | Technical repository guide: datasets, scripts, configs, artifacts, and how to reproduce the benchmark. |
| [REPORT.md](REPORT.md) | Final research report organized according to the required presentation structure. |

Supporting task notes, slide outlines, and reference material are kept under
[docs/](docs) and [report/](report).

---

## Benchmark Overview

The benchmark asks:

> Can promptable foundation segmentation models provide reliable zero-shot
> object masks for robotic perception in challenging simulated scenes?

The study compares:

| Model group | Models |
| --- | --- |
| Heavy zero-shot foundation models | SAM ViT-H, SAM ViT-B, SAM2 Hiera-Large, FastSAM-X |
| Lightweight SAM variants | MobileSAM ViT-T, EfficientSAM-Ti, EfficientSAM-S |
| Supervised baselines | YOLOv8-seg, Mask R-CNN, DeepLabV3+ |

Prompt modes:

- point prompts,
- box prompts,
- automatic mask generation.

Metrics:

- mIoU,
- boundary F1,
- mask AP / AP50 / AP75,
- per-category IoU,
- challenge-group IoU,
- GPU and CPU FPS,
- qualitative failure modes.

## Assignment Compliance Matrix

| Requirement | Repository coverage |
| --- | --- |
| 1. Simulated/curated robotic dataset with reflective metal, transparent glass, occlusion, small parts, and moving objects | Isaac Unitree G1 and BlenderProc synthetic datasets exceed the requested scale, with 1000 images each; OCID adds a real clutter reference. |
| 2. Simulation environment | Isaac Sim is the primary simulator; BlenderProc is used as a secondary synthetic generation pipeline. |
| 3. Simulated robotic platform | The main Isaac dataset uses the official Unitree G1 asset for robot-centered scenes. |
| 4. SAM, SAM2, FastSAM zero-shot evaluation | SAM ViT-H, SAM ViT-B, SAM2 Hiera-Large, and FastSAM-X are evaluated with point, box, and automatic modes. |
| 5. Classical baselines | YOLOv8-seg, Mask R-CNN, and DeepLabV3+ are trained on small labeled subsets. |
| 6. Standard metrics | mIoU, boundary F1, mask AP/AP50/AP75, per-category IoU, and challenge-group summaries are implemented. |
| 7. Inference speed | GPU and CPU FPS/latency benchmarks are stored under `outputs/task7_inference_speed/` and Task 9 speed outputs. |
| 8. Failure modes | Failure analysis and challenge-group summaries are documented in Task 8 and used in the final report. |
| 9. Lightweight SAM variants | MobileSAM and EfficientSAM-Ti/S are evaluated for speed-quality and edge-deployment trade-offs. |

Main software stack:

- Python, PyTorch, OpenCV, COCO-style evaluation tools,
- SAM, SAM2, FastSAM, MobileSAM, EfficientSAM,
- Ultralytics YOLOv8, Mask R-CNN tooling, DeepLabV3+ tooling,
- Isaac Sim and BlenderProc for simulation/synthetic data generation.

---

## Repository Layout

```text
.
├── README.md                         # technical GitHub guide
├── REPORT.md                         # final research report
├── configs/                          # benchmark, dataset, and model configs
├── docs/                             # task-level technical documentation
├── report/                           # supporting report sections and slide outline
├── scripts/
│   ├── analysis/                     # failure analysis and final asset generation
│   ├── baselines/                    # supervised baseline split/training scripts
│   ├── benchmarks/                   # zero-shot inference and speed benchmarks
│   ├── blenderproc/                  # BlenderProc dataset generation
│   ├── datasets/                     # dataset conversion/normalization/release tools
│   ├── evaluation/                   # mIoU, boundary F1, AP evaluation
│   └── aws/                          # artifact sync helpers
├── outputs/                          # compact generated summaries, plots, and tables
└── results/                          # raw predictions/checkpoints; ignored by Git
```

`results/` is intentionally not committed. It stores large raw prediction files
and trained checkpoints. Compact summaries and final plots are stored under
`outputs/`.

---

## Datasets

| Dataset | Type | Role | Images |
| --- | --- | --- | ---: |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Main robot-centered benchmark | 1000 |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Controlled synthetic challenge benchmark | 1000 |
| OCID | Real RGB-D clutter dataset | Real-world clutter/domain-gap reference | 2390 |

![Dataset examples](outputs/final_benchmark_assets/plots/dataset_examples.png)

Dataset paths are configured in [configs/datasets.yaml](configs/datasets.yaml).
Raw datasets are not committed to Git.

Generated Isaac and BlenderProc dataset archives are publicly released on
Zenodo:

<https://doi.org/10.5281/zenodo.20736993>

OCID is external and should be obtained from its upstream source.

---

## Main Technical Workflow

The project is organized as nine benchmark tasks:

| Task | Purpose | Documentation |
| --- | --- | --- |
| Task 1 | Dataset creation and curation | [docs/tasks/task1_dataset_creation.md](docs/tasks/task1_dataset_creation.md) |
| Task 2 | Simulation environment | [docs/tasks/task2_simulation_environment.md](docs/tasks/task2_simulation_environment.md) |
| Task 3 | Robotic platform | [docs/tasks/task3_robotic_platform.md](docs/tasks/task3_robotic_platform.md) |
| Task 4 | Zero-shot SAM-family inference | [docs/tasks/task4_zero_shot_sam.md](docs/tasks/task4_zero_shot_sam.md) |
| Task 5 | Classical supervised baselines | [docs/tasks/task5_classical_baselines.md](docs/tasks/task5_classical_baselines.md) |
| Task 6 | Metric evaluation | [docs/tasks/task6_evaluation.md](docs/tasks/task6_evaluation.md) |
| Task 7 | Inference-speed benchmark | [docs/tasks/task7_inference_speed.md](docs/tasks/task7_inference_speed.md) |
| Task 8 | Failure-mode analysis | [docs/tasks/task8_failure_analysis.md](docs/tasks/task8_failure_analysis.md) |
| Task 9 | Lightweight SAM trade-off | [docs/tasks/task9_lightweight_sam.md](docs/tasks/task9_lightweight_sam.md) |

---

## Environment Setup

Create a local Python environment:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install the required Python dependencies for the task being run. Isaac Sim,
BlenderProc, Detectron2, Ultralytics, SAM/SAM2, FastSAM, MobileSAM, and
EfficientSAM have separate installation requirements depending on the target
machine.

---

## Key Commands

Prepare deterministic supervised splits and COCO subsets:

```bash
python scripts/baselines/prepare_task5_splits.py
```

Evaluate heavy zero-shot models:

```bash
python scripts/evaluation/evaluate_task6_zero_shot.py --split test --rerun-complete
```

Evaluate supervised baselines:

```bash
python scripts/evaluation/evaluate_task6_baselines.py --split test --device 0 --rerun-complete
```

Evaluate lightweight SAM variants:

```bash
python scripts/evaluation/evaluate_task6_zero_shot.py \
  --config configs/task9_evaluation.yaml \
  --split test \
  --rerun-complete
```

Regenerate final lightweight summaries:

```bash
python scripts/analysis/summarize_task9_lightweight_sam.py
```

Regenerate final benchmark plots and tables:

```bash
python scripts/analysis/create_final_benchmark_assets.py
```

---

## Final Benchmark Assets

Compact figures and tables are stored under:

```text
outputs/final_benchmark_assets/
```

Important plots:

| Plot | Path |
| --- | --- |
| Dataset examples | `outputs/final_benchmark_assets/plots/dataset_examples.png` |
| Zero-shot mIoU heatmap | `outputs/final_benchmark_assets/plots/zero_shot_miou_heatmap.png` |
| Baseline mIoU bars | `outputs/final_benchmark_assets/plots/baseline_miou_bars.png` |
| CUDA speed-quality scatter | `outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png` |
| Lightweight SAM trade-off | `outputs/final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png` |
| Challenge group performance | `outputs/final_benchmark_assets/plots/challenge_group_weighted_iou.png` |
| Zero-shot winners | `outputs/final_benchmark_assets/plots/zero_shot_dataset_prompt_winners.png` |

Important tables:

| Table | Path |
| --- | --- |
| Best CUDA quality by dataset | `outputs/final_benchmark_assets/tables/best_cuda_quality_by_dataset.csv` |
| Best CUDA speed-quality trade-off | `outputs/final_benchmark_assets/tables/best_cuda_tradeoff_by_dataset.csv` |
| Best lightweight CUDA trade-off | `outputs/final_benchmark_assets/tables/best_lightweight_cuda_tradeoff.csv` |
| Best zero-shot model by dataset/prompt | `outputs/final_benchmark_assets/tables/best_zero_shot_by_dataset_prompt.csv` |

![CUDA speed-quality trade-off](outputs/final_benchmark_assets/plots/cuda_speed_quality_scatter.png)

---

## Benchmark Protocol Notes

Final comparative results use a common held-out test protocol. The supervised
baselines use train/validation subsets for training and checkpoint selection,
while the final comparison uses reserved test IDs.

Point and box prompt results are oracle-prompt evaluations. They measure
segmentation quality when a target cue is available. In a deployed robot, those
prompts would need to come from an upstream detector, tracker, planner, human
operator, or task prior.

Automatic mask generation is the closest mode to prompt-free object discovery,
but it is slower and less controlled in cluttered robotic scenes.

---

## Final Report

The final research report is [REPORT.md](REPORT.md). It follows the required
lecture structure:

1. Research Problem
2. State of the Art
3. Research Formulation
4. Cognitive Approach
5. Congruence of Results and Conclusions

Additional supporting text and slide planning are available in [report/](report).
