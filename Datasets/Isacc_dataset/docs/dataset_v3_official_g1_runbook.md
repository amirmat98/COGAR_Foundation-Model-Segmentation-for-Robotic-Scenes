# Isaac Official Unitree G1 Runbook

This is the active route for Assignment 2 tasks 1-3.

## Target

- Simulator: Isaac Sim.
- Robot: official Unitree G1 USD asset only.
- Dataset size: 1000 images.
- Annotation outputs: RGB, semantic masks, instance masks, 2D boxes, depth,
  manifest metadata, and COCO instance JSON.
- Hardware: AWS or another NVIDIA GPU machine.

## Local Preconditions

From this folder:

```bash
bash scripts/download_unitree_assets.sh
```

This should create:

```text
assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd
```

The generated data and Unitree asset files are ignored by Git.

## Five-Frame AWS Smoke Test

Run this first on the AWS GPU machine:

```bash
cd /home/ubuntu/Isacc_dataset
bash scripts/run_isaac_dataset_v3_official_g1_container.sh data/smoke_v3_official_g1_5 5 --sample-mode coverage
```

After copying the smoke output back locally:

```bash
python3 scripts/validate_dataset_preview.py data/smoke_v3_official_g1_5 --expected-images 5 --require-official-robot --require-robot-pose-variation --require-robot-mask
python3 scripts/export_isaac_to_coco.py data/smoke_v3_official_g1_5 --config configs/dataset_config_v3_official_g1.json
```

## Full 1000-Image AWS Run

Use `nohup` so SSH disconnects do not stop the job:

```bash
cd /home/ubuntu/Isacc_dataset
nohup bash scripts/run_isaac_dataset_v3_official_g1_container.sh data/robotic_sdg_v3_official_g1_1000 > generation_v3_official_g1_1000.log 2>&1 &
tail -f generation_v3_official_g1_1000.log
```

Copy the dataset back:

```bash
rsync -az --partial --info=progress2 isaac-aws:/home/ubuntu/Isacc_dataset/data/robotic_sdg_v3_official_g1_1000/ data/robotic_sdg_v3_official_g1_1000/
python3 scripts/validate_dataset_preview.py data/robotic_sdg_v3_official_g1_1000 --expected-images 1000 --require-official-robot --require-robot-pose-variation --require-robot-mask
python3 scripts/export_isaac_to_coco.py data/robotic_sdg_v3_official_g1_1000 --config configs/dataset_config_v3_official_g1.json
```

## Important

The strict wrapper uses `--robot-mode official`. The run should fail if the
official Unitree G1 USD asset is missing or invalid.
