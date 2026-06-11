# AWS Isaac Sim Full Dataset Workflow

This document explains how to generate a complete 500-image Isaac
Sim/Replicator dataset for this project:

```text
data/cogar_isaac_sim_500/
```

This does not delete or overwrite the existing Task 1 result:

```text
data/cogar_sim_500_final/
```

The existing BlenderProc dataset remains the frozen benchmark currently used by
the reports. The Isaac Sim dataset is a stronger replacement candidate for
Task 1/Task 2 after you generate it and, if needed, rerun the benchmark models
against it.

## Why A New AWS Machine Is Needed

Isaac Sim 6.0 is much heavier than SAM inference. A Tesla T4 instance is useful
for SAM/SAM2/FastSAM experiments, but it is not the right target for a full
Isaac Sim synthetic-data run.

This was confirmed experimentally on the available AWS T4 machine. Docker,
NVIDIA drivers, the Isaac Sim image, Replicator startup, Hub cache, and the
project generator could be configured, but the machine spent many minutes in
Isaac startup and extension loading before producing a single frame. The T4 run
is useful as an environment-debugging exercise, not as the recommended full
dataset generation route.

Use an RTX-capable AWS instance. For this project, the practical target is:

```text
Best practical choice: g6e.4xlarge
Minimum cost-sensitive choice: g6e.2xlarge
Stronger choice if budget/quota allows: g6e.8xlarge
NVIDIA AWS documentation option: g7e.8xlarge
Storage: 300 GB gp3 EBS volume
OS: Ubuntu 24.04 LTS or Ubuntu 22.04 LTS
```

Avoid `g4dn` / Tesla T4 for the full Isaac dataset. Also avoid A100/H100
instances for this task: they are excellent training GPUs, but Isaac Sim
requires RT cores and NVIDIA documents A100/H100 as unsupported for Isaac Sim.

Official references:

- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- Isaac Sim container installation: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
- Isaac Sim AWS deployment: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html
- Replicator getting started: https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html

## 1. Create The AWS Instance

In the AWS console:

```text
Region: eu-central-1, or another region where g6e/g7e is available
AMI: Ubuntu 24.04 LTS, Ubuntu 22.04 LTS, or NVIDIA Deep Learning AMI
Instance: g6e.4xlarge preferred, g6e.2xlarge minimum
Storage: 300 GB gp3 EBS recommended
Security group: SSH only
```

After launch, connect:

```bash
ssh -i ~/Downloads/YOUR_KEY.pem ubuntu@YOUR_AWS_PUBLIC_DNS
```

Check the GPU:

```bash
nvidia-smi
```

If `nvidia-smi` does not work, fix the NVIDIA driver before continuing.

## 2. Install Docker And NVIDIA Container Toolkit

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

Install NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Validate Docker GPU access:

```bash
docker run --rm --gpus all \
  nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

## 3. Clone Or Update The Repository

Fresh clone:

```bash
git clone https://github.com/amirmat98/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes.git
cd COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes
```

Existing clone:

```bash
cd COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes
git pull
```

## 4. Diagnose The Machine First

Before pulling the huge Isaac image, run:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh diagnose
```

Use this output to confirm:

- the instance type is `g6e.2xlarge`, `g6e.4xlarge`, `g6e.8xlarge`, or
  another RTX/RT-core instance;
- `nvidia-smi` sees the GPU;
- memory is at least 64 GiB for `g6e.2xlarge`, preferably 128 GiB for
  `g6e.4xlarge`;
- free disk is comfortably above 150 GB before pulling Isaac Sim.

Add swap before testing Isaac. This protects against startup memory spikes:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh setup-swap
```

## 5. Pull Isaac Sim

Pull the Isaac Sim container:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh pull
```

If Docker reports NGC terms or authentication problems, open the container page
in your browser, accept the terms, then try again. You can also try:

```bash
docker logout nvcr.io
bash scripts/aws/run_isaac_sim_dataset_aws.sh pull
```

## 6. Check The Machine

Run:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh check
```

This checks:

- `nvidia-smi`
- Docker availability
- GPU access inside a CUDA container

Then run NVIDIA's Isaac compatibility checker from inside the container:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh compat
```

If the compatibility checker does not pass, fix the GPU driver/container setup
before continuing.

## 7. Prepare Isaac Cache And Hub

Isaac Sim containers write cache/log/data files as container UID `1234`.
Prepare the cache folders before generation:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh fix-permissions
```

Start the Omniverse Hub cache container:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh start-hub
```

If Hub is not running, Isaac Sim may repeat this warning and never reach the
generator:

```text
OmniHub: Hub failed to launch
```

Check Hub:

```bash
docker ps | grep hub-cache
docker logs --tail 100 hub-cache
```

On small machines, add swap before testing Isaac:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh setup-swap
```

## 8. Run A 1-Frame Smoke Test

Before spending money on the full 500 images, run one frame:

```bash
mkdir -p outputs/logs

FRAMES=1 PROGRESS_EVERY=1 \
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke1 2>&1 | tee outputs/logs/isaac_smoke1.log
```

Expected success marker:

```text
[ISAAC] 1/1
[ISAAC] Dataset root: ...
[ISAAC] Frames: 1
```

Only after one frame succeeds, try five frames:

```bash
FRAMES=5 PROGRESS_EVERY=1 \
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke
```

Expected output directory:

```text
data/cogar_isaac_sim_500/
```

Expected files:

```text
data/cogar_isaac_sim_500/raw_replicator/final_500/
data/cogar_isaac_sim_500/metadata/frame_index.csv
data/cogar_isaac_sim_500/metadata/categories.json
data/cogar_isaac_sim_500/metadata/dataset_summary.json
data/cogar_isaac_sim_500/README.md
```

## 9. Run The Full 500-Image Dataset

After the smoke test works:

```bash
FRAMES=500 PROGRESS_EVERY=25 \
  bash scripts/aws/run_isaac_sim_dataset_aws.sh generate
```

To leave it running safely:

```bash
mkdir -p outputs/logs
tmux new -s isaac

FRAMES=500 PROGRESS_EVERY=25 \
  bash scripts/aws/run_isaac_sim_dataset_aws.sh generate 2>&1 | tee outputs/logs/isaac_sim_500.log
```

Detach from tmux:

```text
Ctrl+b then d
```

Reattach later:

```bash
tmux attach -t isaac
```

Check whether it is still running:

```bash
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "generate_cogar_isaac|python.sh|docker" | grep -v grep
```

Run only one Isaac generator at a time. If multiple generator processes exist,
stop them:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh stop-isaac
```

## 10. Package The Result

When generation is finished:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh package
```

This creates:

```text
cogar_isaac_sim_500_dataset.tar.gz
cogar_isaac_sim_500_dataset.tar.gz.sha256
```

Download from your local machine:

```bash
scp -i ~/Downloads/YOUR_KEY.pem \
  ubuntu@YOUR_AWS_PUBLIC_DNS:~/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/cogar_isaac_sim_500_dataset.tar.gz .

scp -i ~/Downloads/YOUR_KEY.pem \
  ubuntu@YOUR_AWS_PUBLIC_DNS:~/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes/cogar_isaac_sim_500_dataset.tar.gz.sha256 .
```

Extract locally:

```bash
tar -xzf cogar_isaac_sim_500_dataset.tar.gz
```

## Troubleshooting From The T4 AWS Attempt

### Docker Pull Looks Stuck

The Isaac Sim image is large. Check progress from another terminal:

```bash
docker system df
sudo du -sh /var/lib/docker 2>/dev/null
df -h /
```

### Permission Denied Under `/isaac-sim/.cache` Or `.local/share/ov`

Run:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh fix-permissions
```

If partial output was created by UID `1234`, either keep it and continue, or
remove only the incomplete Isaac output:

```bash
sudo rm -rf data/cogar_isaac_sim_500
```

Do not delete `data/cogar_sim_500_final/`.

### Full Streaming App Loads Instead Of The Generator

Bad symptom:

```text
Isaac Sim Full Streaming App is loaded.
app ready
```

The runner now forces `/isaac-sim/python.sh` as Docker entrypoint to avoid this.
If this symptom returns, make sure the script contains:

```text
--entrypoint /isaac-sim/python.sh
```

### Headless UI Warnings

These are expected in headless Docker and are not fatal by themselves:

```text
failed to open the default display
GLFW initialization failed
Failed to acquire interface: carb::windowing::IWindowing
```

Continue unless the command exits or a project-script traceback appears.

### T4 Practical Limit

On the tested Tesla T4 instance, Isaac Sim could start, but startup was slow and
memory pressure was high. For the final assignment, keep the existing
BlenderProc dataset as the reported dataset unless you can run Isaac Sim on an
RTX-class instance such as `g6e.4xlarge` or, at minimum, `g6e.2xlarge`.

## 11. What To Do After Generation

After the full Isaac dataset exists locally, decide whether to keep it as:

1. A stronger Task 2 evidence dataset only.
2. A replacement benchmark dataset.

If you use it as a replacement benchmark dataset, the model tables must be
rerun on the Isaac-generated annotations so the report remains consistent.

For now, the safe workflow is:

```text
Keep data/cogar_sim_500_final/ as the frozen reported dataset.
Generate data/cogar_isaac_sim_500/ as the complete Isaac Sim version.
Use the Isaac dataset as improved simulation evidence unless you rerun all benchmarks.
```
