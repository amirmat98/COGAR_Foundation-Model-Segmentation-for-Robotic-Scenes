# Foundation Model Segmentation for Robotic Scenes

**Assignment:** Zero-Shot Segmentation Benchmark for Robotic Perception  
**Student id:** 5884715  

This repository contains a simulation-based benchmark for testing whether
foundation segmentation models can support robotic scene understanding. The
project evaluates SAM-family models, lightweight SAM variants, and supervised
segmentation baselines on robotic scenes containing transparent objects,
reflective metal, partial occlusions, small parts, robot-body visibility,
clutter, and dynamic objects.

The main entry points are:

| File | Purpose |
| --- | --- |
| [README.md](README.md) | Technical repository guide: datasets, scripts, configs, artifacts, and how to reproduce the benchmark. |
| [REPORT.md](REPORT.md) | Final research report organized according to the required presentation structure. |
| [docs/wiki/](docs/wiki) | Repository-local wiki with short navigable pages derived from the report. |

Supporting task notes, slide outlines, shared figure references, and source
material are kept under [docs/](docs) and [report/](report).

> **Result storage notice:** The complete `results/` folder could not be
> included in Git because its raw predictions and model checkpoints are too
> large for practical repository storage. The compact numerical summaries,
> tables, plots, and failure examples needed to inspect the reported findings
> are included under `outputs/`. The full raw folder remains on the benchmark
> machine/AWS storage and can be transferred separately when required.

---

## Benchmark Overview

The benchmark asks:

> To what extent can promptable foundation segmentation models provide reliable
> zero-shot object masks for robotic scene understanding in challenging
> simulated environments, and what trade-offs appear against lightweight and
> supervised alternatives in accuracy, robustness, prompt dependence, and
> real-time feasibility?

The benchmark focuses on four technical questions:

| Question | What is measured |
| --- | --- |
| Mask quality | mIoU, boundary F1, mask AP, AP50/AP75 |
| Robotic robustness | per-category and challenge-group IoU on transparent, reflective, occluded, small-part, dynamic, and robot-centered scenes |
| Deployment feasibility | GPU/CPU FPS and latency |
| Model selection | heavy SAM, lightweight SAM, and supervised baseline trade-offs |

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
| 5. Classical baselines | YOLOv8-seg, TorchVision Mask R-CNN, and DeepLabV3+ are trained on small labeled subsets. |
| 6. Standard metrics | mIoU, boundary F1, mask AP/AP50/AP75, per-category IoU, and challenge-group summaries are implemented. |
| 7. Inference speed | GPU and CPU FPS/latency benchmarks are stored under `outputs/task7_inference_speed/` and Task 9 speed outputs. |
| 8. Failure modes | Failure analysis and challenge-group summaries are documented in Task 8 and used in the final report. |
| 9. Lightweight SAM variants | MobileSAM and EfficientSAM-Ti/S are evaluated for speed-quality and edge-deployment trade-offs. |

Main software stack:

- Python, PyTorch, OpenCV, COCO-style evaluation tools,
- SAM, SAM2, FastSAM, MobileSAM, EfficientSAM,
- Ultralytics YOLOv8, TorchVision Mask R-CNN, DeepLabV3+ tooling,
- Isaac Sim and BlenderProc for simulation/synthetic data generation.

---

## Repository Layout

```text
.
├── README.md                         # technical GitHub guide
├── REPORT.md                         # final research report
├── configs/                          # compact task configs and shared fragments
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

The complete `results/` folder could not be committed to Git because the raw
prediction files and trained checkpoints are too large. Compact summaries,
tables, and final plots are stored under `outputs/`; the full raw results remain
on the benchmark machine/AWS storage.

---

## Datasets

| Dataset | Type | Role | Images |
| --- | --- | --- | ---: |
| Isaac official Unitree G1 | Synthetic, Isaac Sim | Main robot-centered benchmark | 1000 |
| BlenderProc COGAR-SimRobotics-1000 | Synthetic, BlenderProc | Controlled synthetic challenge benchmark | 1000 |
| OCID | Real RGB-D clutter dataset | Real-world clutter/domain-gap reference | 2390 |

![Dataset examples](outputs/final_benchmark_assets/plots/dataset_examples.png)

Dataset paths are configured in [configs/datasets.yaml](configs/datasets.yaml).
Repeated task fragments such as dataset paths, prompt modes, metric settings,
and speed sampling are centralized in [configs/common.yaml](configs/common.yaml).
Raw datasets are not committed to Git.

Generated Isaac and BlenderProc dataset archives are publicly released on
Zenodo:

<https://doi.org/10.5281/zenodo.20736993>

OCID is external and should be obtained from its upstream source.

---

## Benchmark Protocol

| Component | Configuration |
| --- | --- |
| Seed / student ID | 5884715 |
| Simulation datasets | Isaac Sim Unitree G1, BlenderProc COGAR-SimRobotics-1000 |
| Domain-gap reference | OCID converted to COCO-style annotations |
| Robotic challenges | reflective metal, transparent glass, partial occlusion, small parts, cables/connectors, dynamic objects |
| Supervised train subset | 100 images per dataset |
| Validation subset | 50 images per dataset for checkpoint selection |
| Final test subset | 850 Isaac, 850 BlenderProc, 2240 OCID held-out images |
| Zero-shot models | SAM ViT-H, SAM ViT-B, SAM2 Hiera-Large, FastSAM-X |
| Lightweight models | MobileSAM ViT-T, EfficientSAM-Ti, EfficientSAM-S |
| Supervised baselines | YOLOv8-seg, Mask R-CNN, DeepLabV3+ |
| Prompt modes | point, box, automatic mask generation |
| Metrics | mIoU, boundary F1, mask AP, AP50/AP75, per-category IoU, FPS/latency |

Final quality comparisons use the held-out test split. The validation subset is
reserved for supervised checkpoint selection.

Fairness rule: supervised baselines are not treated as zero-shot models.
YOLOv8-seg, Mask R-CNN, and DeepLabV3+ use the 100-image train subset and the
50-image validation subset only for model selection. Final tables compare them
with SAM-family zero-shot models only after filtering all outputs to the same
reserved test image IDs.

---

## Runtime and Timing Environment

The saved speed benchmarks include environment metadata in each `*_speed.json`
file. The recorded timing environment was:

| Component | Recorded setting |
| --- | --- |
| Platform | AWS Linux `6.17.0-1017-aws`, x86_64 |
| GPU | NVIDIA Tesla T4, 15.64 GB |
| CPU/RAM | 4 logical CPU cores, 2 physical cores, 16.55 GB RAM |
| Python | 3.12.3 |
| PyTorch | 2.5.1+cu121 |
| Speed outputs | `outputs/task7_inference_speed/summary.csv`, `outputs/task9_lightweight_sam/inference_speed/summary.csv` |

Timing uses single-image inference. Heavy Task 7 CUDA point/box/baseline rows
use 50 timed images, CUDA automatic rows use 20, CPU point/box/baseline rows
use 3, and CPU automatic rows use 1. Lightweight Task 9 uses 50 CUDA
point/box images, 20 CUDA automatic images, 10 CPU point/box images, and 1 CPU
automatic image. Each summary row records its exact `sample_images`,
`warmup_units`, and `timed_units`.

SAM-family runs use native RGB dataset images with model-specific
preprocessing. Supervised baseline timing uses the configured inference
resolutions: YOLOv8-seg image size 640, DeepLabV3+ 512×512, and Mask R-CNN
640/1024 min/max resizing.

---

## Implementation Notes

The repository keeps the assignment model families intact, but some software
choices are intentionally practical:

| Topic | Implementation choice |
| --- | --- |
| Mask R-CNN | Implemented with TorchVision `maskrcnn_resnet50_fpn`, initialized from COCO weights, instead of Detectron2. This still evaluates the requested Mask R-CNN baseline family. |
| Detectron2 | Not required by the current code path; adding it would duplicate the Mask R-CNN baseline in a second framework. |
| Simulation stack | Isaac Sim is the primary simulator; BlenderProc is used for additional controlled synthetic scenes. Gazebo/Rviz2 are not part of the final measured benchmark. |
| Robot platform | The robot-centered simulation dataset uses the official Unitree G1 asset. |
| FastSAM | Used from a local checkout under `external/FastSAM` because the benchmark imports the repository directly. |
| Dataset scale | The simulation benchmark exceeds the requested 500-image scale with 1000 Isaac and 1000 BlenderProc images. OCID is included only as a real clutter/domain-gap reference. |

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
python -m pip install -r requirements.txt
```

`requirements.txt` is the single Python dependency file for local checks,
evaluation, supervised baselines, zero-shot inference, lightweight SAM
experiments, and plot/report generation.

For GPU runs, install a CUDA-compatible PyTorch/TorchVision wheel first if the
default wheel does not match the machine. FastSAM is used from a local source
checkout under `external/FastSAM`; see the note inside `requirements.txt`.
Isaac Sim is managed separately by NVIDIA's Isaac Sim environment. The current
Mask R-CNN baseline uses torchvision rather than Detectron2.

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

The current `outputs/` directory contains compact evidence for the complete
benchmark:

| Output group | Contents |
| --- | --- |
| `outputs/task4_zero_shot_sam/prompts/` | Prompt manifests for 1000 Isaac images, 1000 BlenderProc images, and 2390 OCID images. |
| `outputs/task5_baselines/` | Supervised split files, COCO subsets, class maps, and training summaries. |
| `outputs/task6_evaluation/` | 36 heavy zero-shot quality rows and 9 supervised baseline quality rows. |
| `outputs/task7_inference_speed/` | 90 GPU/CPU timing rows with FPS and latency summaries. |
| `outputs/task8_failure_analysis/` | 378 category rows, 151 challenge-group rows, 10 representative failure overlays. |
| `outputs/task9_lightweight_sam/` | 72 lightweight quality rows, 144 speed-quality rows, 36 recommendation rows. |
| `outputs/final_benchmark_assets/` | Final aggregate plots, compact tables, and recommendation guide. |

The most important numerical tables are:

| Table | What it summarizes |
| --- | --- |
| `best_cuda_quality_by_dataset.csv` | Highest-quality CUDA model per dataset. |
| `best_cuda_tradeoff_by_dataset.csv` | Best mIoU-FPS trade-off per dataset. |
| `best_lightweight_cuda_tradeoff.csv` | Best lightweight SAM trade-offs by dataset and prompt. |
| `best_zero_shot_by_dataset_prompt.csv` | Best zero-shot model for each dataset/prompt mode. |

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

## Benchmark Interpretation Notes

Final comparative results use a common held-out test protocol. The supervised
baselines use train/validation subsets for training and checkpoint selection,
while the final comparison uses reserved test IDs. Validation metrics are not
used as final benchmark evidence.

Point and box prompt results are oracle-prompt evaluations. They measure
segmentation quality when a target cue is available. In a deployed robot, those
prompts would need to come from an upstream detector, tracker, planner, human
operator, or task prior.

Automatic mask generation is the closest mode to prompt-free object discovery,
but it is slower and less controlled in cluttered robotic scenes.

---

## Threats to Validity

| Type | Main issue | Practical interpretation |
| --- | --- | --- |
| Internal validity | Point and box prompts use oracle ground-truth cues. | Prompted results measure mask quality once a target cue exists, not full autonomy. |
| Internal validity | Supervised baselines are trained, SAM-family models are zero-shot. | Final comparisons use only the shared held-out test IDs. |
| External validity | Simulation does not perfectly match real sensors, lighting, materials, or motion. | Isaac/BlenderProc results should be validated on real robot data before deployment. |
| Construct validity | Segmentation metrics do not measure complete robotic task success. | Grasping, pose, tracking, and planning need separate evaluation. |
| Reproducibility validity | FPS depends on hardware, software versions, image resolution, and preprocessing. | Timing metadata is saved in each `*_speed.json` file and summarized above. |

---

## Key References

The full bibliography is in [REPORT.md](REPORT.md#references). The main
technical sources are:

| Topic | References |
| --- | --- |
| Foundation segmentation | SAM, SAM2, FastSAM, MobileSAM, EfficientSAM |
| Recent SOTA context | SAM3 concept prompting and SAM3D physical-world reconstruction, used only as background |
| Supervised baselines | Mask R-CNN, DeepLabV3+, Ultralytics YOLO segmentation |
| Metrics | COCO instance evaluation, COCO API, Berkeley boundary benchmark |
| Simulation and domain gap | NVIDIA Isaac Sim Replicator, domain randomization |
| Robotic clutter reference | OCID / OCID-Ref |

Primary links:

- SAM: <https://arxiv.org/abs/2304.02643>
- SAM2: <https://arxiv.org/abs/2408.00714>
- SAM3: <https://arxiv.org/abs/2511.16719>
- SAM3D: <https://arxiv.org/abs/2511.16624>
- FastSAM: <https://arxiv.org/abs/2306.12156>
- MobileSAM: <https://arxiv.org/abs/2306.14289>
- EfficientSAM: <https://arxiv.org/abs/2312.00863>
- Mask R-CNN: <https://arxiv.org/abs/1703.06870>
- DeepLabV3+: <https://arxiv.org/abs/1802.02611>
- Ultralytics segmentation docs: <https://docs.ultralytics.com/tasks/segment/>
- COCO: <https://arxiv.org/abs/1405.0312>
- Berkeley boundary benchmark: <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/>
- Isaac Sim Replicator: <https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/index.html>
- Domain randomization: <https://arxiv.org/abs/1703.06907>
- OCID: <https://www.acin.tuwien.ac.at/object-clutter-indoor-dataset/>

---

## Final Report

The final research report is [REPORT.md](REPORT.md). It follows the required
lecture structure:

1. Research Problem
2. State of the Art
3. Research Formulation
4. Cognitive Approach
5. Congruence of Results and Conclusions

`REPORT.md` is the main document. Each major report section links to its
supporting file in [report/](report). Reusable visual evidence is centralized
in [report/figures_and_tables.md](report/figures_and_tables.md), and the
repository-local wiki is available at [docs/wiki/](docs/wiki).
