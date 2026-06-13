# BlenderProc AWS Runbook

Target dataset:

```text
/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

## AWS Setup

Use an AWS instance with enough disk space for Blender, the repository, and the
generated dataset. GPU is recommended for speed.

The current config writes to `/mnt/Info/COGAR_DATASETs`. Create it on AWS or
edit `configs/blenderproc_dataset.yaml` and `configs/datasets.yaml` to use the
AWS dataset mount path.

```bash
sudo mkdir -p /mnt/Info/COGAR_DATASETs
sudo chown -R "$USER:$USER" /mnt/Info/COGAR_DATASETs
```

After the repository is on AWS:

```bash
cd COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke Test

```bash
.venv/bin/blenderproc run scripts/blenderproc/generate_cogar_sim.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 5 \
  --raw-dataset-name smoke_5_aws \
  --seed 5884715
```

## Full Generation

```bash
nohup .venv/bin/blenderproc run scripts/blenderproc/generate_cogar_sim.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 1000 \
  --raw-dataset-name cogar_sim_1000_raw \
  --seed 5884715 > blenderproc_cogar_sim_1000.log 2>&1 &
```

Monitor:

```bash
tail -f blenderproc_cogar_sim_1000.log
```

## Normalize

```bash
.venv/bin/python scripts/datasets/normalize_blenderproc_cogar_sim.py
```

## Package

```bash
.venv/bin/python scripts/datasets/package_dataset_release.py \
  /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --name BlenderProc_cogar_sim_1000
```
