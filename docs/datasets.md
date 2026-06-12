# Dataset Registry

This project uses three datasets. Raw/generated image data should not be
committed to Git.

## 1. Isaac Sim Official Unitree G1

Decision: use the official Unitree G1 USD asset only, even if generation is
slow. The final target is 1000 images.

Location:

```text
Datasets/Isacc_dataset
```

Main config:

```text
Datasets/Isacc_dataset/configs/dataset_config_v3_official_g1.json
```

Runbook:

```text
Datasets/Isacc_dataset/docs/dataset_v3_official_g1_runbook.md
```

Use AWS or another NVIDIA GPU machine for Isaac Sim. Local CPU is not a
practical target for the full dataset generation.

## 2. BlenderProc COGAR-SimRobotics

The old generated BlenderProc dataset was removed accidentally. The useful
source code still exists in the archive and should be recovered selectively
when this task starts:

```text
/home/amir/Desktop/COGAR/Archive/scripts/blenderproc/generate_cogar_sim_500.py
/home/amir/Desktop/COGAR/Archive/scripts/dataset/normalize_cogar_sim_500.py
/home/amir/Desktop/COGAR/Archive/src/cogar_seg/generation
```

Do not copy old benchmark outputs, plots, or unrelated archive docs back into
this repository.

## 3. OCID

OCID is an external real-world RGB-D clutter dataset. It is used as a
real-world robustness/generalization dataset, not as generated simulation data.

Current local path:

```text
/mnt/Info/COGAR_DATASETs/OCID-dataset
```

Expected extracted layout:

```text
OCID-dataset/
  ARID10/
  ARID20/
  YCB10/
```

Sequence folders contain matching `rgb`, `label`, `depth`, and `pcd` folders.
The current local copy has 2390 files for each modality.

For a future client, download OCID from the upstream Object Cluttered Indoor
Dataset distribution associated with the EasyLabel paper, extract it outside
the Git repository, and point project configs/scripts to the extracted
`OCID-dataset` root. Do not vendor the raw OCID data into this repo.

Reference: Markus Suchi, Timothy Patten, David Fischinger, Markus Vincze,
"EasyLabel: A Semi-Automatic Pixel-wise Object Annotation Tool for Creating
Robotic RGB-D Datasets", arXiv:1902.01626,
https://arxiv.org/abs/1902.01626.

## AWS/GPU Notes

Use AWS/GPU for:

- Isaac Sim full dataset generation.
- Full foundation-model inference benchmark.
- Final FPS measurements on GPU.

CPU/local is enough for:

- OCID indexing.
- Dataset manifest creation.
- Small smoke-test validation.
- Documentation and config edits.
