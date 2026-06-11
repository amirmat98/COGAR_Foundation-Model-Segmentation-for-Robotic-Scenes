# AWS OCID Benchmark Runbook

This runbook moves the OCID benchmark to an EC2 GPU instance without relying on
uncommitted local state. It assumes the AWS CLI is configured locally and on the
EC2 instance.

## 1. Package And Upload From Local

Set your S3 destination:

```bash
export S3_URI=s3://cogar-ocid-5884715
```

Package the current repository, OCID dataset, and checkpoints:

```bash
bash scripts/aws/package_ocid_for_aws.sh
```

Useful options:

```bash
SKIP_DATASET=1 bash scripts/aws/package_ocid_for_aws.sh
SKIP_CHECKPOINTS=1 bash scripts/aws/package_ocid_for_aws.sh
OCID_ROOT=/mnt/Info/COGAR_DATASETs/OCID-dataset bash scripts/aws/package_ocid_for_aws.sh
```

## 2. Prepare EC2

Use a GPU instance with at least 24 GB VRAM for comfortable SAM runs, such as
`g5.xlarge`, `g5.2xlarge`, `g6.xlarge`, or `g6.2xlarge`. A Deep Learning AMI is
the simplest base image because NVIDIA drivers and CUDA tooling are already
available.

On the instance:

```bash
sudo apt update
sudo apt install -y awscli git tmux htop

mkdir -p ~/COGAR
aws s3 cp ${S3_URI}/cogar_repo_aws.tar.gz ~/COGAR/
tar -xzf ~/COGAR/cogar_repo_aws.tar.gz -C ~/COGAR

sudo mkdir -p /mnt/Info/COGAR_DATASETs
sudo chown -R $USER:$USER /mnt/Info
aws s3 cp ${S3_URI}/OCID-dataset.tar.gz /tmp/
tar -xzf /tmp/OCID-dataset.tar.gz -C /mnt/Info/COGAR_DATASETs

cd ~/COGAR
mkdir -p checkpoints
aws s3 sync ${S3_URI}/checkpoints/ checkpoints/
```

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt -r requirements-models.txt
```

Check CUDA and paths:

```bash
bash scripts/aws/run_ocid_aws.sh check
```

## 3. Run Jobs

Use `tmux` so the job survives SSH disconnects:

```bash
tmux new -s ocid
source .venv/bin/activate
```

Build or verify the OCID index:

```bash
bash scripts/aws/run_ocid_aws.sh index
```

Run all main SAM ViT-B jobs with the faster automatic-mask setting:

```bash
S3_RESULTS_URI=${S3_URI}/aws_run_$(date +%Y%m%d_%H%M%S) \
bash scripts/aws/run_ocid_aws.sh all-sam-fast
```

For pilots, set limits:

```bash
BOX_LIMIT=500 POINT_LIMIT=500 AUTO_LIMIT=1000 \
bash scripts/aws/run_ocid_aws.sh all-sam-fast
```

For only fast automatic masks:

```bash
S3_RESULTS_URI=${S3_URI}/aws_auto_fast16 \
bash scripts/aws/run_ocid_aws.sh auto-fast16
```

The fast automatic-mask default is:

```text
AUTO_POINTS_PER_SIDE=16
AUTO_PRED_IOU_THRESH=0.90
AUTO_STABILITY_SCORE_THRESH=0.92
NO_SAVE_MASKS=1
```

## 4. Outputs

Main local outputs on EC2:

```text
outputs/ocid_full/results/
outputs/ocid_full/tables/
docs/ocid_massive_benchmark_report.md
```

If `S3_RESULTS_URI` is set, the AWS run script syncs those outputs back to S3
after each job.

## 5. Cost Control

Stop or terminate the EC2 instance immediately after syncing results:

```bash
sudo shutdown -h now
```
