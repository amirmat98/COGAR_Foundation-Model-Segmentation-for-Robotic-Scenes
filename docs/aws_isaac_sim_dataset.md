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

Use an RTX-capable AWS instance.

Recommended:

```text
Instance type: g6e.2xlarge or larger
GPU: RTX-class NVIDIA GPU with RT cores
Storage: 200 GB or larger EBS volume
OS: Ubuntu 24.04 LTS or Ubuntu 22.04 LTS
```

Official references:

- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- Isaac Sim container installation: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
- Isaac Sim AWS deployment: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html
- Replicator getting started: https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html

## 1. Create The AWS Instance

In the AWS console:

```text
Region: eu-central-1, or another region where g6e is available
AMI: Ubuntu 24.04 LTS, Ubuntu 22.04 LTS, or NVIDIA Deep Learning AMI
Instance: g6e.2xlarge preferred
Storage: 200 GB gp3 EBS minimum
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
docker run --rm --runtime=nvidia --gpus all \
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

## 4. Pull Isaac Sim

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

## 5. Check The Machine

Run:

```bash
bash scripts/aws/run_isaac_sim_dataset_aws.sh check
```

This checks:

- `nvidia-smi`
- Docker availability
- GPU access inside a CUDA container

## 6. Run A 5-Frame Smoke Test

Before spending money on the full 500 images, run:

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

## 7. Run The Full 500-Image Dataset

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

## 8. Package The Result

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

## 9. What To Do After Generation

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
