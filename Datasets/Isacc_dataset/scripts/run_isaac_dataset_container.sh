#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-datasets/smoke_test}"
max_images="${2:-}"
config_path="${3:-configs/dataset_config_v3_official_g1.json}"
generator_script="${4:-src/robotic_sdg/generate_dataset_v2.py}"
extra_generator_args=("${@:5}")
image="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p \
  "$HOME/docker/isaac-sim/cache/main" \
  "$HOME/docker/isaac-sim/cache/computecache" \
  "$HOME/docker/isaac-sim/config" \
  "$HOME/docker/isaac-sim/data" \
  "$HOME/docker/isaac-sim/logs" \
  "$HOME/docker/isaac-sim/pkg" \
  "$HOME/.cache/ov/hub" \
  "$repo_dir/datasets"

sudo chown -R 1234:1234 "$HOME/docker/isaac-sim" "$HOME/.cache/ov/hub"
sudo chown -R "$(id -u):$(id -g)" "$repo_dir/datasets"
chmod -R a+rwX "$repo_dir/datasets"

generator_args=(
  "$generator_script"
  "--config"
  "$config_path"
  "--output"
  "$output_dir"
)

if [[ -n "$max_images" ]]; then
  generator_args+=("--max-images" "$max_images")
fi

if [[ "${#extra_generator_args[@]}" -gt 0 ]]; then
  generator_args+=("${extra_generator_args[@]}")
fi

printf -v generator_command "%q " "${generator_args[@]}"

docker run --name isaac-sim-dataset --entrypoint bash --gpus all --rm --network=host \
  -e ACCEPT_EULA=Y \
  -v "$HOME/docker/isaac-sim/cache/main:/isaac-sim/.cache:rw" \
  -v "$HOME/docker/isaac-sim/cache/computecache:/isaac-sim/.nv/ComputeCache:rw" \
  -v "$HOME/docker/isaac-sim/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "$HOME/docker/isaac-sim/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "$HOME/docker/isaac-sim/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "$HOME/docker/isaac-sim/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "$HOME/.cache/ov/hub:/var/cache/hub:rw" \
  -v "$repo_dir:/workspace/Isacc_dataset:rw" \
  -u 1234:1234 \
  "$image" \
  -lc "cd /workspace/Isacc_dataset && /isaac-sim/python.sh ${generator_command}"
