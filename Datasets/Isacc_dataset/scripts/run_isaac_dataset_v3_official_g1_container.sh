#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-datasets/robotic_sdg_v3_official_g1_1000}"
max_images="${2:-}"

if [[ $# -gt 0 ]]; then
  shift
fi
if [[ $# -gt 0 ]]; then
  shift
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$repo_dir/scripts/run_isaac_dataset_container.sh" \
  "$output_dir" \
  "$max_images" \
  "configs/dataset_config_v3_official_g1.json" \
  "src/robotic_sdg/generate_dataset_v2.py" \
  --robot-mode official \
  "$@"
