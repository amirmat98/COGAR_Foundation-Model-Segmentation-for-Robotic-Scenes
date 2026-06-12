#!/usr/bin/env bash
set -euo pipefail

remote="${1:?Usage: scripts/sync_to_remote.sh <ssh-alias> [remote-dir]}"
remote_dir="${2:-~/Isacc_dataset}"

rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "data/" \
  --exclude "datasets/" \
  --exclude "outputs/" \
  --exclude "_out*/" \
  ./ "${remote}:${remote_dir}/"

echo "Synced to ${remote}:${remote_dir}"
