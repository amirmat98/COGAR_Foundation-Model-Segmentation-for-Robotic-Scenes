# Task 2: Simulation Environment

Task 2 requires simulation-based dataset generation.

## Isaac Sim

Isaac Sim is the primary simulator. The official Unitree G1 dataset was
generated with the Isaac Sim Replicator pipeline in:

```text
Datasets/Isacc_dataset
```

The active runbook is:

```text
Datasets/Isacc_dataset/docs/dataset_v3_official_g1_runbook.md
```

Full Isaac generation should run on AWS or another NVIDIA GPU machine.

## BlenderProc

BlenderProc is the secondary simulator for COGAR-SimRobotics-1000. The target
output path is:

```text
/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

Generation command:

```bash
blenderproc run scripts/blenderproc/generate_cogar_sim.py \
  --config configs/blenderproc_dataset.yaml
```

Normalization command:

```bash
.venv/bin/python scripts/datasets/normalize_blenderproc_cogar_sim.py
```

The local 5-image BlenderProc smoke test passed on 2026-06-13. The full
1000-image dataset should be generated on AWS.
