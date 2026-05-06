# BlenderProc COGAR-SimRobotics-500

BlenderProc is used as the current reproducible synthetic-data generator for
COGAR-SimRobotics-500 because it can produce RGB images, instance segmentation,
COCO annotations, and scene metadata from scripted randomized robotic tabletop
scenes.

## Dataset Target

- Dataset size: 500 images
- Image size: 640x480
- Categories: 10
- Challenge balance: 100 images each
  - reflective metal
  - transparent glass
  - partial occlusion
  - small parts
  - dynamic scene proxy

## Generate

```bash
source ~/blenderproc_test/.venv/bin/activate

blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images 500
```

Raw BlenderProc data is written below:

```text
data/cogar_sim_500/raw_blenderproc/
```

## Normalize

```bash
PYTHONPATH=src python scripts/dataset/normalize_cogar_sim_500.py
```

Normalized data is written below:

```text
data/cogar_sim_500/rgb/
data/cogar_sim_500/annotations/
data/cogar_sim_500/metadata/
data/cogar_sim_500/splits/
```

## Create Object Index

```bash
PYTHONPATH=src python scripts/dataset/create_object_index.py \
  --dataset cogar_sim_500 \
  --coco data/cogar_sim_500/annotations/instances_all.json \
  --metadata data/cogar_sim_500/metadata/frame_index.csv \
  --rgb-dir data/cogar_sim_500/rgb \
  --output outputs/indexes/cogar_sim_500_objects.csv
```

## Validate

```bash
python -m py_compile $(find src scripts -name "*.py")
PYTHONPATH=src pytest -q
```

The object-index command validates that COCO image count matches metadata image
count, every normalized RGB file exists, category IDs map to category names, and
every bounding box has positive width and height.

## Git Policy

Generated raw data, normalized images, annotations, metadata, splits, result
CSVs, plots, masks, and checkpoints are ignored by Git. Keep only documentation,
configs, source code, tests, and lightweight placeholders tracked.
