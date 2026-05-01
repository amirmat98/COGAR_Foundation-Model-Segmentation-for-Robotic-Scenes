# Isaac Sim Dataset Generation Setup

This project uses Isaac Sim Replicator for synthetic robotic-scene dataset generation.

## Local machine status

Local laptop GPU:

- NVIDIA GeForce GTX 1050
- 4 GB VRAM
- no RTX / RT cores

This machine is not suitable for Isaac Sim Replicator dataset generation.

## Required generation machine

Use a workstation or cloud machine with:

- RTX-capable NVIDIA GPU
- preferably 16 GB+ VRAM
- Ubuntu 22.04
- recent NVIDIA driver
- Isaac Sim 5.x

## Workflow

1. Prepare scripts and configs in this repo locally.
2. Push or copy the repo to the RTX machine.
3. Run Isaac Sim Replicator on the RTX machine.
4. Generate the pilot dataset first.
5. Copy the generated dataset back to this repo.
6. Run SAM/SAM2/FastSAM evaluation locally or on GPU machine.