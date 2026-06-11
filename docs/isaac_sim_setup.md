# Isaac Sim Dataset Generation Setup

This project now includes a complete Isaac Sim / Replicator generation path for
a 500-image robotic-scene dataset:

```text
data/cogar_isaac_sim_500/
```

This path is separate from the existing frozen dataset:

```text
data/cogar_sim_500_final/
```

Keeping these roots separate prevents accidental changes to the Task 1 result
that already supports the benchmark reports.

## Local Machine Status

The original local laptop GPU is not suitable for Isaac Sim Replicator dataset
generation:

- NVIDIA GeForce GTX 1050
- 4 GB VRAM
- no RTX / RT cores

The local machine can edit scripts, inspect results, and run lightweight tests,
but the full Isaac Sim dataset should be generated on an RTX-capable cloud or
workstation GPU.

## Required Generation Machine

Use a workstation or cloud machine with:

- RTX-capable NVIDIA GPU with RT cores;
- preferably 16 GB+ VRAM;
- Ubuntu 22.04 or Ubuntu 24.04;
- recent NVIDIA driver;
- Docker with NVIDIA Container Toolkit;
- Isaac Sim 6.0 container image.

Recommended AWS instance:

```text
g6e.2xlarge or larger
```

The Tesla T4 machine used for SAM inference is not the preferred choice for
Isaac Sim. It can run segmentation benchmarks, but Isaac Sim 6.0 targets newer
RTX-class GPUs.

## AWS T4 Attempt Outcome

The project was also tested on an AWS Tesla T4 instance after the Docker and
NVIDIA Container Toolkit setup was repaired.

Observed result:

- `nvidia-smi` worked on the host and inside Docker.
- `nvcr.io/nvidia/isaac-sim:6.0.0` downloaded successfully.
- Isaac Sim cache permission issues were solved by preparing cache folders for
  container UID `1234`.
- Omniverse Hub startup issues were solved by running the Hub Workstation Cache
  container.
- The runner was updated to force `/isaac-sim/python.sh` as the Docker
  entrypoint so the full streaming app wrapper is not launched accidentally.
- Even after those fixes, the T4 machine remained too slow and memory-limited
  for a practical full Isaac Sim dataset generation run.

Conclusion:

```text
Use the T4 instance for SAM/SAM2/FastSAM benchmarking.
Do not use the T4 instance as the final Isaac Sim generation machine.
Use g6e.2xlarge or stronger for the full Isaac Sim dataset.
```

This is why the reported assignment dataset remains
`data/cogar_sim_500_final/`, while the Isaac Sim workflow stays available as a
reproducible route for a future RTX-class AWS run.

## Tracked Files

```text
configs/isaac_sim_dataset.yaml
scripts/isaac_sim/generate_cogar_isaac_sim_500.py
scripts/aws/run_isaac_sim_dataset_aws.sh
src/cogar_seg/generation/isaac_sim_scene.py
docs/aws_isaac_sim_dataset.md
```

## Workflow

1. Launch an RTX AWS instance, preferably `g6e.2xlarge`.
2. Install Docker and NVIDIA Container Toolkit.
3. Clone or pull this repository.
4. Pull the Isaac Sim container.
5. Run a 5-frame smoke test.
6. Run the full 500-frame generation.
7. Package `data/cogar_isaac_sim_500/`.
8. Download it locally.
9. Decide whether to use it as extra Task 2 evidence or rerun all model
   benchmarks on it as a replacement dataset.

See the full command sequence:

```text
docs/aws_isaac_sim_dataset.md
```
