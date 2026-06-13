# BlenderProc AWS Runbook

AWS target dataset:

```text
~/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

Final local destination after copying back:

```text
/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
```

## AWS Setup

Use an AWS instance with enough disk space for Blender, the repository, and the
generated dataset. GPU is recommended for speed.

Create a dataset folder outside the Git repository:

```bash
mkdir -p ~/COGAR_DATASETs
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
  --output-root ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --num-images 5 \
  --raw-dataset-name smoke_5_aws \
  --seed 5884715
```

## Full Generation

```bash
nohup .venv/bin/blenderproc run scripts/blenderproc/generate_cogar_sim.py \
  --config configs/blenderproc_dataset.yaml \
  --output-root ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
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
.venv/bin/python scripts/datasets/normalize_blenderproc_cogar_sim.py \
  --output-root ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --raw-coco-dir ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000/raw_blenderproc/cogar_sim_1000_raw/coco_data \
  --raw-metadata ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000/metadata/frame_index_raw.csv \
  --expected-images 1000
```

## Package

```bash
.venv/bin/python scripts/datasets/package_dataset_release.py \
  ~/COGAR_DATASETs/BlenderProc_cogar_sim_1000 \
  --name BlenderProc_cogar_sim_1000
```

Copy back to the local machine from the local terminal:

```bash
mkdir -p /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000
rsync -az --partial --info=progress2 \
  isaac-aws:~/COGAR_DATASETs/BlenderProc_cogar_sim_1000/ \
  /mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/
```
