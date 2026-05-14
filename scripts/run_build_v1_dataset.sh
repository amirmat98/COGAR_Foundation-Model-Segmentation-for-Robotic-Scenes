#!/usr/bin/env bash
set -euo pipefail

NUM_IMAGES="${NUM_IMAGES:-70}"
RAW_DATASET_NAME="${RAW_DATASET_NAME:-pilot_v3_cleaner}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-70}"
RAW_METADATA="${RAW_METADATA:-data/cogar_sim_500/metadata/frame_index_pilot_v2.csv}"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

set -x

blenderproc run scripts/blenderproc/generate_cogar_sim_500.py \
  --config configs/blenderproc_dataset.yaml \
  --num-images "${NUM_IMAGES}" \
  --raw-dataset-name "${RAW_DATASET_NAME}"

python scripts/dataset/build_clean_sim_dataset.py \
  --raw-coco-dir "data/cogar_sim_500/raw_blenderproc/${RAW_DATASET_NAME}/coco_data" \
  --raw-metadata "${RAW_METADATA}" \
  --output-root data/cogar_sim_500 \
  --config configs/blenderproc_dataset.yaml \
  --expected-images "${EXPECTED_IMAGES}" \
  --index-output outputs/indexes/cogar_sim_500_objects_v1.csv \
  --mask-dir data/cogar_sim_500/instance_masks/v1 \
  --final-index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1.csv \
  --audit-output-dir outputs/tables/dataset_audit_v1 \
  --filtered-index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --exclude-categories table \
  --min-area 25 \
  --filter-bad

python scripts/dataset/validate_sim_index.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv

python scripts/dataset/audit_sim_dataset.py \
  --index data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv \
  --output-dir outputs/tables/dataset_audit_v1_filtered
