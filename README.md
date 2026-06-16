# Foundation Model Segmentation for Robotic Scenes

Student id: 5884715

Assignment 2: Zero-Shot Segmentation Benchmark for Robotic Perception.

This repository contains the dataset preparation, model benchmarking, baseline
training, metric evaluation, and inference-speed tooling for the robotic
segmentation benchmark.

## Reports

- [Main report](REPORT.md)
- [Task 1: Dataset creation and curation](docs/tasks/task1_dataset_creation.md)
- [Task 2: Simulation environment](docs/tasks/task2_simulation_environment.md)
- [Task 3: Robotic platform](docs/tasks/task3_robotic_platform.md)
- [Task 4: Zero-shot SAM-family inference](docs/tasks/task4_zero_shot_sam.md)
- [Task 5: Classical baselines](docs/tasks/task5_classical_baselines.md)
- [Task 6: Evaluation metrics](docs/tasks/task6_evaluation.md)
- [Task 7: Inference speed](docs/tasks/task7_inference_speed.md)
- [Task 8: Failure mode analysis](docs/tasks/task8_failure_analysis.md)
- [Task 9: Lightweight SAM edge-deployment trade-off](docs/tasks/task9_lightweight_sam.md)

## Datasets

Dataset paths are configured in [configs/datasets.yaml](configs/datasets.yaml).
Raw datasets are not committed to Git.

| Dataset | Default path | Details |
| --- | --- | --- |
| Isaac official Unitree G1 | `Datasets/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000` | [docs/datasets/isaac_official_g1.md](docs/datasets/isaac_official_g1.md) |
| BlenderProc COGAR-SimRobotics-1000 | `Datasets/BlenderProc_cogar_sim_1000` | [docs/datasets/blenderproc_cogar_sim_1000.md](docs/datasets/blenderproc_cogar_sim_1000.md) |
| OCID | `Datasets/OCID` | [docs/datasets/ocid.md](docs/datasets/ocid.md) |

Users may place datasets elsewhere and edit [configs/datasets.yaml](configs/datasets.yaml).

## Local Python

```text
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Isaac Sim and BlenderProc have their own runtime requirements; see the dataset
documents before generating data.
