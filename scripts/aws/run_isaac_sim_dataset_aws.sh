#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Run the complete COGAR-IsaacSimRobotics-500 workflow on an AWS RTX GPU instance.

Usage:
  bash scripts/aws/run_isaac_sim_dataset_aws.sh check
  bash scripts/aws/run_isaac_sim_dataset_aws.sh pull
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke
  bash scripts/aws/run_isaac_sim_dataset_aws.sh generate
  bash scripts/aws/run_isaac_sim_dataset_aws.sh package
  bash scripts/aws/run_isaac_sim_dataset_aws.sh shell

Environment variables:
  ISAAC_SIM_IMAGE   Default: nvcr.io/nvidia/isaac-sim:6.0.0
  CONFIG            Default: configs/isaac_sim_dataset.yaml
  OUTPUT_DIR        Default: data/cogar_isaac_sim_500
  FRAMES            Default: 500 for generate, 5 for smoke
  RAW_DATASET_NAME  Default: final_500
  WIDTH             Optional override
  HEIGHT            Optional override
  RT_SUBFRAMES      Optional override
  PROGRESS_EVERY    Default: 25
USAGE
}

log() {
  printf '[ISAAC-AWS] %s\n' "$*"
}

die() {
  printf '[ISAAC-AWS][ERROR] %s\n' "$*" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cmd="${1:-}"
[[ -n "$cmd" ]] || { usage; exit 1; }
shift || true

ISAAC_SIM_IMAGE="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
CONFIG="${CONFIG:-configs/isaac_sim_dataset.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cogar_isaac_sim_500}"
RAW_DATASET_NAME="${RAW_DATASET_NAME:-final_500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"

cache_root="${HOME}/docker/isaac-sim"
mkdir -p \
  "${cache_root}/cache/main" \
  "${cache_root}/cache/computecache" \
  "${cache_root}/config" \
  "${cache_root}/data" \
  "${cache_root}/logs" \
  "${cache_root}/pkg" \
  "${HOME}/.cache/ov/hub"

docker_args=(
  --rm
  --gpus all
  --network=host
  -e ACCEPT_EULA=Y
  -e PRIVACY_CONSENT=Y
  -v "${cache_root}/cache/main:/isaac-sim/.cache:rw"
  -v "${cache_root}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw"
  -v "${cache_root}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
  -v "${cache_root}/config:/isaac-sim/.nvidia-omniverse/config:rw"
  -v "${cache_root}/data:/isaac-sim/.local/share/ov/data:rw"
  -v "${cache_root}/pkg:/isaac-sim/.local/share/ov/pkg:rw"
  -v "${HOME}/.cache/ov/hub:/var/cache/hub:rw"
  -v "${repo_root}:/workspace/cogar:rw"
  -w /workspace/cogar
)

run_generator() {
  local frame_count="$1"
  [[ -f "$CONFIG" ]] || die "Missing config: $CONFIG"
  mkdir -p "$OUTPUT_DIR"
  chmod -R a+rwX "$OUTPUT_DIR"

  extra_args=()
  [[ -n "${WIDTH:-}" ]] && extra_args+=(--width "$WIDTH")
  [[ -n "${HEIGHT:-}" ]] && extra_args+=(--height "$HEIGHT")
  [[ -n "${RT_SUBFRAMES:-}" ]] && extra_args+=(--rt-subframes "$RT_SUBFRAMES")

  log "Generating ${frame_count} Isaac Sim frames into ${OUTPUT_DIR}"
  docker run "${docker_args[@]}" "$ISAAC_SIM_IMAGE" \
    /isaac-sim/python.sh /workspace/cogar/scripts/isaac_sim/generate_cogar_isaac_sim_500.py \
      --config "/workspace/cogar/${CONFIG}" \
      --output-dir "/workspace/cogar/${OUTPUT_DIR}" \
      --raw-dataset-name "$RAW_DATASET_NAME" \
      --num-images "$frame_count" \
      --headless \
      --progress-every "$PROGRESS_EVERY" \
      "${extra_args[@]}"
}

log "Repo: ${repo_root}"
log "Image: ${ISAAC_SIM_IMAGE}"
log "Config: ${CONFIG}"
log "Output: ${OUTPUT_DIR}"

case "$cmd" in
  check)
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
    command -v docker >/dev/null 2>&1 || die "docker not found"
    nvidia-smi
    docker info >/dev/null
    docker run --rm --runtime=nvidia --gpus all \
      nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
    ;;
  pull)
    docker pull "$ISAAC_SIM_IMAGE"
    ;;
  smoke)
    run_generator "${FRAMES:-5}"
    ;;
  generate)
    run_generator "${FRAMES:-500}"
    ;;
  package)
    [[ -d "$OUTPUT_DIR" ]] || die "Missing output directory: $OUTPUT_DIR"
    package_name="cogar_isaac_sim_500_dataset.tar.gz"
    tar -czf "$package_name" "$OUTPUT_DIR"
    sha256sum "$package_name" > "${package_name}.sha256"
    ls -lh "$package_name" "${package_name}.sha256"
    ;;
  shell)
    docker run -it "${docker_args[@]}" "$ISAAC_SIM_IMAGE" bash
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    die "Unknown command: $cmd"
    ;;
esac
