#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Package this repository, OCID, and checkpoints for an AWS EC2 benchmark run.

Required:
  S3_URI=s3://your-bucket/prefix

Optional:
  OCID_ROOT=/mnt/Info/COGAR_DATASETs/OCID-dataset
  CHECKPOINT_DIR=checkpoints
  REPO_TAR=/tmp/cogar_repo_aws.tar.gz
  DATASET_TAR=/tmp/OCID-dataset.tar.gz
  SKIP_DATASET=1
  SKIP_CHECKPOINTS=1

Example:
  S3_URI=s3://cogar-ocid-5884715 bash scripts/aws/package_ocid_for_aws.sh
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

S3_URI="${S3_URI:-}"
if [[ -z "${S3_URI}" ]]; then
  usage
  echo "[ERROR] Set S3_URI=s3://bucket/prefix" >&2
  exit 1
fi

require_cmd aws
require_cmd tar

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OCID_ROOT="${OCID_ROOT:-/mnt/Info/COGAR_DATASETs/OCID-dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
REPO_TAR="${REPO_TAR:-/tmp/cogar_repo_aws.tar.gz}"
DATASET_TAR="${DATASET_TAR:-/tmp/OCID-dataset.tar.gz}"

cd "${REPO_ROOT}"

echo "[AWS] Packaging repo: ${REPO_TAR}"
tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='venv' \
  --exclude='env' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='outputs' \
  --exclude='data' \
  --exclude='runs' \
  --exclude='checkpoints' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.tar.gz' \
  -czf "${REPO_TAR}" .

aws s3 cp "${REPO_TAR}" "${S3_URI%/}/cogar_repo_aws.tar.gz"

if [[ "${SKIP_DATASET:-0}" != "1" ]]; then
  if [[ ! -d "${OCID_ROOT}" ]]; then
    echo "[ERROR] OCID_ROOT does not exist: ${OCID_ROOT}" >&2
    exit 1
  fi
  echo "[AWS] Packaging OCID dataset: ${DATASET_TAR}"
  tar -C "$(dirname "${OCID_ROOT}")" -czf "${DATASET_TAR}" "$(basename "${OCID_ROOT}")"
  aws s3 cp "${DATASET_TAR}" "${S3_URI%/}/OCID-dataset.tar.gz"
else
  echo "[AWS] SKIP_DATASET=1, not uploading OCID dataset"
fi

if [[ "${SKIP_CHECKPOINTS:-0}" != "1" ]]; then
  if [[ -d "${CHECKPOINT_DIR}" ]]; then
    echo "[AWS] Syncing checkpoints directory"
    aws s3 sync "${CHECKPOINT_DIR}" "${S3_URI%/}/checkpoints/"
  else
    echo "[WARN] Checkpoint directory not found: ${CHECKPOINT_DIR}"
  fi

  for weight_file in FastSAM-s.pt mobile_sam.pt yolov8n-seg.pt yolo26n.pt; do
    if [[ -f "${weight_file}" ]]; then
      echo "[AWS] Uploading root weight: ${weight_file}"
      aws s3 cp "${weight_file}" "${S3_URI%/}/checkpoints/${weight_file}"
    fi
  done
else
  echo "[AWS] SKIP_CHECKPOINTS=1, not uploading checkpoints"
fi

echo "[AWS] Package upload complete: ${S3_URI%/}"
