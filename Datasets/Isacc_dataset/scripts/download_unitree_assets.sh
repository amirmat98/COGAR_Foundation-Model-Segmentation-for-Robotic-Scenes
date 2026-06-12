#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asset_dir="${1:-$repo_dir/assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0}"
base_url="https://huggingface.co/datasets/unitreerobotics/unitree_model/resolve/main/G1/29dof/usd/g1_29dof_rev_1_0"

mkdir -p "$asset_dir/configuration"

download_file() {
  local rel_path="$1"
  local target="$asset_dir/$rel_path"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    echo "exists: $target"
    return
  fi
  echo "download: $rel_path"
  curl -L --fail --retry 3 --retry-delay 2 -o "$target" "$base_url/$rel_path"
}

download_file "g1_29dof_rev_1_0.usd"
download_file "configuration/g1_29dof_rev_1_0_base.usd"
download_file "configuration/g1_29dof_rev_1_0_physics.usd"
download_file "configuration/g1_29dof_rev_1_0_sensor.usd"

echo "Unitree G1 USD assets ready:"
find "$asset_dir" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort
