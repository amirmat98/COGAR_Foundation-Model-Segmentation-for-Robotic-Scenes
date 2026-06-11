# Isaac Sim Dataset Generation Setup

This project includes an experimental Isaac Sim / Replicator generation path
that was tested as a possible additional robotic-scene dataset:

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
but it cannot run a practical Isaac Sim / Replicator dataset generation job.

## Required Generation Machine

Use a workstation or cloud machine with:

- RTX-capable NVIDIA GPU with RT cores;
- preferably 16 GB+ VRAM;
- Ubuntu 22.04 or Ubuntu 24.04;
- recent NVIDIA driver;
- Docker with NVIDIA Container Toolkit;
- Isaac Sim 6.0 container image.

Recommended AWS instances:

```text
Best practical choice: g6e.4xlarge
Minimum cost-sensitive choice: g6e.2xlarge
Stronger choice if budget/quota allows: g6e.8xlarge
NVIDIA AWS documentation option: g7e.8xlarge
```

The Tesla T4 machine used for SAM inference is not the preferred choice for
Isaac Sim. It can run segmentation benchmarks, but Isaac Sim 6.0 targets newer
RTX-class GPUs with RT cores. Avoid A100/H100 for this task because Isaac Sim
documents GPUs without RT cores as unsupported.

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
- Low-resolution emergency smoke tests could start Isaac Sim and reach frame
  capture, but Replicator writer finalization did not reliably produce saved
  benchmark image files on the T4 machine.

Conclusion:

```text
Use the T4 instance for SAM/SAM2/FastSAM benchmarking.
Do not use the T4 instance as the final Isaac Sim generation machine.
Do not spend more cloud budget for Task 2; the assignment report closes Task 2
with the completed BlenderProc simulation pipeline.
```

This is why the reported assignment dataset remains
`data/cogar_sim_500_final/`, while the Isaac Sim workflow stays available only
as documented future work.

## Tracked Files

```text
configs/isaac_sim_dataset.yaml
scripts/isaac_sim/generate_cogar_isaac_sim_500.py
scripts/aws/run_isaac_sim_dataset_aws.sh
src/cogar_seg/generation/isaac_sim_scene.py
docs/aws_isaac_sim_dataset.md
```

## Workflow

1. Launch an RTX AWS instance, preferably `g6e.4xlarge`.
2. Install Docker and NVIDIA Container Toolkit.
3. Clone or pull this repository.
4. Run `bash scripts/aws/run_isaac_sim_dataset_aws.sh diagnose`.
5. Pull the Isaac Sim container.
6. Run the compatibility check and a 1-frame smoke test.
7. Stop unless the smoke test reliably writes RGB and annotation files.
8. Treat any future successful Isaac dataset as a separate future extension,
   not as part of the current final benchmark.

See the full command sequence:

```text
docs/aws_isaac_sim_dataset.md
```
