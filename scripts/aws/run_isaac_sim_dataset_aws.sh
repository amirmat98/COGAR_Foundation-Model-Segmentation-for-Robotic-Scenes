#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Run the complete COGAR-IsaacSimRobotics-500 workflow on an AWS RTX GPU instance.

Usage:
  bash scripts/aws/run_isaac_sim_dataset_aws.sh diagnose
  bash scripts/aws/run_isaac_sim_dataset_aws.sh check
  bash scripts/aws/run_isaac_sim_dataset_aws.sh pull
  bash scripts/aws/run_isaac_sim_dataset_aws.sh compat
  bash scripts/aws/run_isaac_sim_dataset_aws.sh setup-swap
  bash scripts/aws/run_isaac_sim_dataset_aws.sh fix-permissions
  bash scripts/aws/run_isaac_sim_dataset_aws.sh start-hub
  bash scripts/aws/run_isaac_sim_dataset_aws.sh stop-isaac
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke1
  bash scripts/aws/run_isaac_sim_dataset_aws.sh smoke
  bash scripts/aws/run_isaac_sim_dataset_aws.sh generate
  bash scripts/aws/run_isaac_sim_dataset_aws.sh package
  CONFIRM_CLEAN=1 bash scripts/aws/run_isaac_sim_dataset_aws.sh clean-output
  bash scripts/aws/run_isaac_sim_dataset_aws.sh shell

Environment variables:
  ISAAC_SIM_IMAGE      Default: nvcr.io/nvidia/isaac-sim:6.0.0
  ISAAC_CONTAINER_UID  Default: 1234
  CONFIG               Default: configs/isaac_sim_dataset.yaml
  OUTPUT_DIR           Default: data/cogar_isaac_sim_500
  FRAMES               Default: 500 for generate, 5 for smoke, 1 for smoke1
  RAW_DATASET_NAME     Default: final_500
  WIDTH                Optional override
  HEIGHT               Optional override
  RT_SUBFRAMES         Optional override
  PROGRESS_EVERY       Default: 25
  SWAP_SIZE_GB         Default: 16 for setup-swap
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
ISAAC_CONTAINER_UID="${ISAAC_CONTAINER_UID:-1234}"
CONFIG="${CONFIG:-configs/isaac_sim_dataset.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cogar_isaac_sim_500}"
RAW_DATASET_NAME="${RAW_DATASET_NAME:-final_500}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
SWAP_SIZE_GB="${SWAP_SIZE_GB:-16}"

cache_root="${HOME}/docker/isaac-sim"
cache_dirs=(
  "${cache_root}/cache/main"
  "${cache_root}/cache/computecache"
  "${cache_root}/config"
  "${cache_root}/data"
  "${cache_root}/logs"
  "${cache_root}/pkg"
  "${HOME}/.cache/ov/hub"
)
mkdir -p "${cache_dirs[@]}"

docker_args=(
  --rm
  --gpus all
  --network=host
  -u "${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_UID}"
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

aws_metadata() {
  local path="$1"
  local token=""

  command -v curl >/dev/null 2>&1 || return 0
  token="$(curl -fsS -m 1 -X PUT \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
    "http://169.254.169.254/latest/api/token" 2>/dev/null || true)"

  if [[ -n "$token" ]]; then
    curl -fsS -m 1 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      "http://169.254.169.254/latest/meta-data/${path}" 2>/dev/null || true
  else
    curl -fsS -m 1 \
      "http://169.254.169.254/latest/meta-data/${path}" 2>/dev/null || true
  fi
}

diagnose_machine() {
  local instance_type=""
  local gpu_names=""

  instance_type="$(aws_metadata instance-type)"
  [[ -n "$instance_type" ]] || instance_type="unknown/non-EC2"
  log "AWS instance type: ${instance_type}"

  log "Disk:"
  df -h / || true

  log "Memory:"
  free -h || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU:"
    nvidia-smi || true
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if grep -qiE "T4|K80|P100|V100" <<<"$gpu_names"; then
      log "WARNING: This GPU is useful for segmentation inference but is weak for full Isaac Sim generation."
    fi
    if grep -qiE "A100|H100" <<<"$gpu_names"; then
      log "WARNING: Isaac Sim documentation says GPUs without RT cores, including A100/H100, are not supported."
    fi
  else
    log "WARNING: nvidia-smi not found."
  fi

  if command -v docker >/dev/null 2>&1; then
    log "Docker:"
    docker --version || true
    docker system df || true
  else
    log "WARNING: docker not found."
  fi
}

setup_swap() {
  local size_gb="$1"

  command -v sudo >/dev/null 2>&1 || die "sudo not found"
  if swapon --show=NAME --noheadings 2>/dev/null | grep -q '^/swapfile$'; then
    log "/swapfile is already active."
    swapon --show
    free -h
    return 0
  fi

  log "Creating ${size_gb} GB /swapfile. This helps Isaac Sim survive startup memory spikes."
  sudo fallocate -l "${size_gb}G" /swapfile || \
    sudo dd if=/dev/zero of=/swapfile bs=1G count="$size_gb" status=progress
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  swapon --show
  free -h
}

run_generator() {
  local frame_count="$1"
  local -a extra_args=()

  [[ -f "$CONFIG" ]] || die "Missing config: $CONFIG"
  mkdir -p "$OUTPUT_DIR"
  chmod -R a+rwX "$OUTPUT_DIR" 2>/dev/null || \
    log "Output dir has files not owned by this user; continuing for container UID ${ISAAC_CONTAINER_UID}."

  [[ -n "${WIDTH:-}" ]] && extra_args+=(--width "$WIDTH")
  [[ -n "${HEIGHT:-}" ]] && extra_args+=(--height "$HEIGHT")
  [[ -n "${RT_SUBFRAMES:-}" ]] && extra_args+=(--rt-subframes "$RT_SUBFRAMES")

  log "Generating ${frame_count} Isaac Sim frames into ${OUTPUT_DIR}"
  log "Using /isaac-sim/python.sh as Docker entrypoint to avoid the full streaming app wrapper."
  docker run "${docker_args[@]}" \
    --entrypoint /isaac-sim/python.sh \
    "$ISAAC_SIM_IMAGE" \
      /workspace/cogar/scripts/isaac_sim/generate_cogar_isaac_sim_500.py \
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
log "Container UID: ${ISAAC_CONTAINER_UID}"
log "Config: ${CONFIG}"
log "Output: ${OUTPUT_DIR}"

case "$cmd" in
  diagnose)
    diagnose_machine
    ;;
  check)
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
    command -v docker >/dev/null 2>&1 || die "docker not found"
    diagnose_machine
    nvidia-smi
    docker info >/dev/null
    docker run --rm --gpus all \
      nvcr.io/nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
    ;;
  pull)
    docker pull "$ISAAC_SIM_IMAGE"
    ;;
  compat)
    docker run "${docker_args[@]}" \
      --entrypoint bash \
      "$ISAAC_SIM_IMAGE" \
      /isaac-sim/isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
    ;;
  setup-swap)
    setup_swap "$SWAP_SIZE_GB"
    ;;
  fix-permissions)
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo mkdir -p "${cache_dirs[@]}"
    sudo chown -R "${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_UID}" "$cache_root" "${HOME}/.cache/ov/hub"
    sudo chmod -R a+rwX "$cache_root" "${HOME}/.cache/ov/hub"
    if [[ -d "$OUTPUT_DIR" ]]; then
      sudo chmod -R a+rwX "$OUTPUT_DIR" || true
    fi
    log "Prepared Isaac Sim cache directories for container UID ${ISAAC_CONTAINER_UID}."
    ;;
  start-hub)
    docker rm -f hub-cache >/dev/null 2>&1 || true
    docker pull nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
    docker run --name hub-cache --rm -d --network=host \
      -v "${HOME}/.cache/ov/hub:/var/cache/hub:rw" \
      -u "${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_UID}" \
      nvcr.io/nvidia/omniverse/hub_workstation_cache:2.0.0
    sleep 3
    docker ps | grep hub-cache
    ;;
  stop-isaac)
    ids="$(docker ps -q --filter ancestor="$ISAAC_SIM_IMAGE")"
    if [[ -n "$ids" ]]; then
      docker kill $ids 2>/dev/null || true
    fi
    pkill -f generate_cogar_isaac_sim_500.py 2>/dev/null || true
    log "Stopped active Isaac Sim generator containers/processes if any were running."
    ;;
  smoke1)
    run_generator "${FRAMES:-1}"
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
    if ! tar -czf "$package_name" "$OUTPUT_DIR"; then
      log "Normal tar failed, likely because generated files are owned by container UID ${ISAAC_CONTAINER_UID}. Retrying with sudo."
      command -v sudo >/dev/null 2>&1 || die "sudo not found"
      sudo tar -czf "$package_name" "$OUTPUT_DIR"
      sudo chown "$(id -u):$(id -g)" "$package_name"
    fi
    sha256sum "$package_name" > "${package_name}.sha256"
    ls -lh "$package_name" "${package_name}.sha256"
    ;;
  clean-output)
    [[ "${CONFIRM_CLEAN:-0}" == "1" ]] || \
      die "Refusing to remove ${OUTPUT_DIR}. Re-run with CONFIRM_CLEAN=1 when you intentionally want to delete only this Isaac output."
    command -v sudo >/dev/null 2>&1 || die "sudo not found"
    sudo rm -rf "$OUTPUT_DIR"
    log "Removed ${OUTPUT_DIR}. Frozen BlenderProc dataset data/cogar_sim_500_final/ was not touched."
    ;;
  shell)
    docker run -it "${docker_args[@]}" --entrypoint bash "$ISAAC_SIM_IMAGE"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    die "Unknown command: $cmd"
    ;;
esac
