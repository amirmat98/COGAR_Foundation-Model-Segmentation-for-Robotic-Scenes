# Foundation Model Segmentation for Robotic Scenes

Student id: 5884715

This repository is being rebuilt task by task for Assignment 2: Zero-Shot
Segmentation Benchmark for Robotic Perception.

Current scope: dataset preparation for tasks 1-3. Benchmark code for SAM,
SAM2, FastSAM, MobileSAM, EfficientSAM, and classical baselines will be added
only after the dataset structure is clean.

## Dataset Sources

The benchmark will use three dataset sources:

| Dataset | Role | Repo policy |
| --- | --- | --- |
| Isaac Sim official Unitree G1 | Main synthetic robotic-scene dataset, 1000 images | Keep generator/configs in Git. Generated images stay ignored. |
| BlenderProc COGAR-SimRobotics | Secondary synthetic dataset regenerated from archived code | Recover only needed code from `Archive` later. Generated images stay ignored. |
| OCID | External real-world RGB-D clutter dataset | Do not copy into Git. Keep path/config/docs only. |

See [docs/datasets.md](docs/datasets.md) and
[configs/datasets.yaml](configs/datasets.yaml).

## Current Layout

```text
configs/                 Dataset registry and future experiment configs
docs/                    Minimal project docs
scripts/datasets/        Dataset indexing/conversion scripts, added as needed
Datasets/Isacc_dataset/  Isaac Sim official Unitree G1 dataset generator
```

## GPU Boundary

Use AWS or another NVIDIA GPU machine for:

- Isaac Sim dataset generation.
- Full SAM-family and baseline benchmark runs.
- Final FPS measurements.

Local CPU is fine for repository cleanup, config work, dataset indexing, and
small validation scripts.
