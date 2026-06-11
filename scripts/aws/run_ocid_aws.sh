#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run OCID benchmark jobs on an AWS EC2 GPU instance.

Usage:
  bash scripts/aws/run_ocid_aws.sh check
  bash scripts/aws/run_ocid_aws.sh index
  bash scripts/aws/run_ocid_aws.sh box
  bash scripts/aws/run_ocid_aws.sh point
  bash scripts/aws/run_ocid_aws.sh auto-fast16
  bash scripts/aws/run_ocid_aws.sh report
  bash scripts/aws/run_ocid_aws.sh all-sam-fast

Common environment variables:
  OCID_ROOT=/mnt/Info/COGAR_DATASETs/OCID-dataset
  CONFIG=configs/paths.yaml
  INDEX=outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv
  CHECKPOINT=checkpoints/sam_vit_b_01ec64.pth
  MODEL_TYPE=vit_b
  DEVICE=cuda
  PROGRESS_EVERY=500
  S3_RESULTS_URI=s3://your-bucket/prefix/results

Speed/disk controls:
  NO_SAVE_MASKS=1
  AUTO_POINTS_PER_SIDE=16
  AUTO_PRED_IOU_THRESH=0.90
  AUTO_STABILITY_SCORE_THRESH=0.92

Limit controls for pilots:
  BOX_LIMIT=500
  POINT_LIMIT=500
  AUTO_LIMIT=1000
EOF
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "[ERROR] Missing required path: $1" >&2
    exit 1
  fi
}

run_cmd() {
  echo
  echo "[AWS] $*"
  "$@"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

JOB="${1:-${JOB:-help}}"

OCID_ROOT="${OCID_ROOT:-/mnt/Info/COGAR_DATASETs/OCID-dataset}"
CONFIG="${CONFIG:-configs/paths.yaml}"
INDEX="${INDEX:-outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv}"
CHECKPOINT="${CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
DEVICE="${DEVICE:-cuda}"
PROGRESS_EVERY="${PROGRESS_EVERY:-500}"
NO_SAVE_MASKS="${NO_SAVE_MASKS:-1}"

BOX_OUTPUT_DIR="${BOX_OUTPUT_DIR:-outputs/ocid_full/aws/sam_vit_b_box}"
POINT_OUTPUT_DIR="${POINT_OUTPUT_DIR:-outputs/ocid_full/aws/sam_vit_b_point}"
AUTO_OUTPUT_DIR="${AUTO_OUTPUT_DIR:-outputs/ocid_full/aws/sam_vit_b_auto_fast16}"

BOX_RESULTS="${BOX_RESULTS:-outputs/ocid_full/results/sam_vit_b_box.csv}"
POINT_RESULTS="${POINT_RESULTS:-outputs/ocid_full/results/sam_vit_b_point.csv}"
AUTO_FAST_RESULTS="${AUTO_FAST_RESULTS:-outputs/ocid_full/results/sam_vit_b_auto_fast16.csv}"
AUTO_DEFAULT_RESULTS="${AUTO_DEFAULT_RESULTS:-outputs/ocid_full/results/sam_vit_b_auto.csv}"

AUTO_POINTS_PER_SIDE="${AUTO_POINTS_PER_SIDE:-16}"
AUTO_PRED_IOU_THRESH="${AUTO_PRED_IOU_THRESH:-0.90}"
AUTO_STABILITY_SCORE_THRESH="${AUTO_STABILITY_SCORE_THRESH:-0.92}"
AUTO_CROP_N_LAYERS="${AUTO_CROP_N_LAYERS:-0}"

print_environment() {
  echo "[AWS] Repo: ${REPO_ROOT}"
  echo "[AWS] OCID_ROOT: ${OCID_ROOT}"
  echo "[AWS] CONFIG: ${CONFIG}"
  echo "[AWS] INDEX: ${INDEX}"
  echo "[AWS] CHECKPOINT: ${CHECKPOINT}"
  echo "[AWS] MODEL_TYPE: ${MODEL_TYPE}"
  echo "[AWS] DEVICE: ${DEVICE}"
  echo "[AWS] PROGRESS_EVERY: ${PROGRESS_EVERY}"
  echo "[AWS] NO_SAVE_MASKS: ${NO_SAVE_MASKS}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "[WARN] nvidia-smi not found"
  fi
}

ensure_index() {
  if [[ -f "${INDEX}" && "${FORCE_INDEX:-0}" != "1" ]]; then
    echo "[AWS] Reusing existing OCID index: ${INDEX}"
    return
  fi

  require_file "${CONFIG}"
  require_file "${OCID_ROOT}"
  strict_args=()
  if [[ "${STRICT_INDEX:-0}" == "1" ]]; then
    strict_args+=(--strict)
  fi
  run_cmd python3 scripts/dataset/create_object_index.py \
    --dataset ocid_full \
    --config "${CONFIG}" \
    --ocid-root "${OCID_ROOT}" \
    --progress-every "${INDEX_PROGRESS_EVERY:-250}" \
    "${strict_args[@]}"
}

mask_args=()
if [[ "${NO_SAVE_MASKS}" == "1" ]]; then
  mask_args+=(--no-save-masks)
fi

run_box() {
  ensure_index
  require_file "${CHECKPOINT}"
  args=(
    python3 scripts/eval/run_sam_box_prompt.py
    --config "${CONFIG}"
    --index "${INDEX}"
    --checkpoint "${CHECKPOINT}"
    --model-type "${MODEL_TYPE}"
    --device "${DEVICE}"
    --output-dir "${BOX_OUTPUT_DIR}"
    --results-csv "${BOX_RESULTS}"
    --no-visualizations
    --progress-every "${PROGRESS_EVERY}"
    "${mask_args[@]}"
  )
  if [[ -n "${BOX_LIMIT:-}" ]]; then
    args+=(--limit "${BOX_LIMIT}")
  fi
  run_cmd "${args[@]}"
}

run_point() {
  ensure_index
  require_file "${CHECKPOINT}"
  args=(
    python3 scripts/eval/run_sam_point_prompt.py
    --config "${CONFIG}"
    --index "${INDEX}"
    --checkpoint "${CHECKPOINT}"
    --model-type "${MODEL_TYPE}"
    --device "${DEVICE}"
    --output-dir "${POINT_OUTPUT_DIR}"
    --results-csv "${POINT_RESULTS}"
    --no-visualizations
    --progress-every "${PROGRESS_EVERY}"
    "${mask_args[@]}"
  )
  if [[ -n "${POINT_LIMIT:-}" ]]; then
    args+=(--limit "${POINT_LIMIT}")
  fi
  run_cmd "${args[@]}"
}

run_auto_fast16() {
  ensure_index
  require_file "${CHECKPOINT}"
  args=(
    python3 scripts/eval/run_sam_auto_masks.py
    --config "${CONFIG}"
    --index "${INDEX}"
    --checkpoint "${CHECKPOINT}"
    --model-type "${MODEL_TYPE}"
    --device "${DEVICE}"
    --output-dir "${AUTO_OUTPUT_DIR}"
    --results-csv "${AUTO_FAST_RESULTS}"
    --points-per-side "${AUTO_POINTS_PER_SIDE}"
    --pred-iou-thresh "${AUTO_PRED_IOU_THRESH}"
    --stability-score-thresh "${AUTO_STABILITY_SCORE_THRESH}"
    --crop-n-layers "${AUTO_CROP_N_LAYERS}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ "${NO_SAVE_MASKS}" == "1" ]]; then
    args+=(--no-save-masks)
  fi
  if [[ -n "${AUTO_LIMIT:-}" ]]; then
    args+=(--limit "${AUTO_LIMIT}")
  fi
  run_cmd "${args[@]}"
}

run_auto_default() {
  ensure_index
  require_file "${CHECKPOINT}"
  args=(
    python3 scripts/eval/run_sam_auto_masks.py
    --config "${CONFIG}"
    --index "${INDEX}"
    --checkpoint "${CHECKPOINT}"
    --model-type "${MODEL_TYPE}"
    --device "${DEVICE}"
    --output-dir "${AUTO_DEFAULT_OUTPUT_DIR:-outputs/ocid_full/aws/sam_vit_b_auto_default}"
    --results-csv "${AUTO_DEFAULT_RESULTS}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ "${NO_SAVE_MASKS}" == "1" ]]; then
    args+=(--no-save-masks)
  fi
  if [[ -n "${AUTO_LIMIT:-}" ]]; then
    args+=(--limit "${AUTO_LIMIT}")
  fi
  run_cmd "${args[@]}"
}

run_report() {
  ensure_index
  result_args=()
  for csv in \
    "${BOX_RESULTS}" \
    "${POINT_RESULTS}" \
    "${AUTO_FAST_RESULTS}" \
    "${AUTO_DEFAULT_RESULTS}" \
    outputs/ocid_full/results/fastsam_s_box.csv \
    outputs/ocid_full/results/mobilesam_box.csv; do
    if [[ -s "${csv}" ]]; then
      result_args+=("${csv}")
    else
      echo "[AWS] Report skip missing/empty result: ${csv}"
    fi
  done

  args=(python3 scripts/analysis/summarize_ocid_massive_benchmark.py --debug)
  if [[ "${STRICT_REPORT:-0}" == "1" ]]; then
    args+=(--strict-results)
  fi
  if [[ "${#result_args[@]}" -gt 0 ]]; then
    args+=(--results "${result_args[@]}")
  fi
  run_cmd "${args[@]}"
}

sync_results() {
  if [[ -z "${S3_RESULTS_URI:-}" ]]; then
    return
  fi
  if ! command -v aws >/dev/null 2>&1; then
    echo "[WARN] aws CLI not found; cannot sync results"
    return
  fi
  run_cmd aws s3 sync outputs/ocid_full/results "${S3_RESULTS_URI%/}/results/"
  run_cmd aws s3 sync outputs/ocid_full/tables "${S3_RESULTS_URI%/}/tables/"
  if [[ -f docs/ocid_massive_benchmark_report.md ]]; then
    run_cmd aws s3 cp docs/ocid_massive_benchmark_report.md \
      "${S3_RESULTS_URI%/}/docs/ocid_massive_benchmark_report.md"
  fi
}

case "${JOB}" in
  -h|--help|help)
    usage
    ;;
  check)
    print_environment
    require_file "${CONFIG}"
    require_file "${OCID_ROOT}"
    require_file "${CHECKPOINT}"
    run_cmd python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY
    ;;
  index)
    print_environment
    if [[ -z "${FORCE_INDEX:-}" ]]; then
      FORCE_INDEX=1
    fi
    ensure_index
    ;;
  box)
    print_environment
    run_box
    sync_results
    ;;
  point)
    print_environment
    run_point
    sync_results
    ;;
  auto-fast16)
    print_environment
    run_auto_fast16
    sync_results
    ;;
  auto-default)
    print_environment
    run_auto_default
    sync_results
    ;;
  report)
    print_environment
    run_report
    sync_results
    ;;
  all-sam-fast)
    print_environment
    ensure_index
    run_box
    run_point
    run_auto_fast16
    run_report
    sync_results
    ;;
  *)
    usage
    echo "[ERROR] Unknown job: ${JOB}" >&2
    exit 1
    ;;
esac
