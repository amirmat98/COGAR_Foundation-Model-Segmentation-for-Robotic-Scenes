#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Package the frozen COGAR-SimRobotics-500 dataset for GitHub Release upload.

Default command:
  bash scripts/dataset/package_release_datasets.sh

Options:
  --output-dir DIR        Directory for release assets.
                          Default: /tmp/cogar_dataset_release
  --dataset-dir DIR       Frozen dataset directory.
                          Default: data/cogar_sim_500_final
  --yolo-dir DIR          YOLOv8-seg export directory.
                          Default: data/yolo_cogar_sim_500_final
  --skip-yolo             Package only the main frozen dataset.
  --force                 Overwrite existing archive files.
  --dry-run               Validate inputs and print the plan without archiving.
  -h, --help              Show this help.

Outputs:
  COGAR-SimRobotics-500_dataset.tar.gz
  COGAR-SimRobotics-500_yolov8seg_export.tar.gz
  SHA256SUMS.txt
  RELEASE_NOTES_TEMPLATE.md
USAGE
}

log() {
  printf '[DATASET-RELEASE] %s\n' "$*"
}

die() {
  printf '[DATASET-RELEASE][ERROR] %s\n' "$*" >&2
  exit 1
}

require_path() {
  local path_value="$1"
  local label="$2"
  if [[ ! -e "$path_value" ]]; then
    die "Missing ${label}: ${path_value}"
  fi
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

output_dir="${OUTPUT_DIR:-/tmp/cogar_dataset_release}"
dataset_dir="data/cogar_sim_500_final"
yolo_dir="data/yolo_cogar_sim_500_final"
include_yolo=1
force=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a value"
      output_dir="$2"
      shift 2
      ;;
    --dataset-dir)
      [[ $# -ge 2 ]] || die "--dataset-dir requires a value"
      dataset_dir="$2"
      shift 2
      ;;
    --yolo-dir)
      [[ $# -ge 2 ]] || die "--yolo-dir requires a value"
      yolo_dir="$2"
      shift 2
      ;;
    --skip-yolo)
      include_yolo=0
      shift
      ;;
    --force)
      force=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

dataset_index="${dataset_dir}/annotations/sim_robotic_scenes_index_final_filtered.csv"
dataset_archive_name="COGAR-SimRobotics-500_dataset.tar.gz"
yolo_archive_name="COGAR-SimRobotics-500_yolov8seg_export.tar.gz"
dataset_archive="${output_dir}/${dataset_archive_name}"
yolo_archive="${output_dir}/${yolo_archive_name}"
checksums_file="${output_dir}/SHA256SUMS.txt"
release_notes_file="${output_dir}/RELEASE_NOTES_TEMPLATE.md"

log "Repo: ${repo_root}"
log "Dataset dir: ${dataset_dir}"
log "YOLO dir: ${yolo_dir}"
log "Output dir: ${output_dir}"
log "Include YOLO export: ${include_yolo}"
log "Dry run: ${dry_run}"

require_path "$dataset_dir" "frozen dataset directory"
require_path "$dataset_index" "final dataset index"
require_path "${dataset_dir}/rgb" "dataset RGB directory"
require_path "${dataset_dir}/instance_masks/final" "dataset final mask directory"
require_path "${dataset_dir}/metadata/categories.json" "dataset category metadata"
require_path "${dataset_dir}/metadata/frame_index.csv" "dataset frame metadata"
require_path "${dataset_dir}/splits" "dataset split directory"

if [[ "$include_yolo" -eq 1 ]]; then
  require_path "$yolo_dir" "YOLO export directory"
  yolo_yaml="${yolo_dir}/data.yaml"
  if [[ ! -f "$yolo_yaml" ]]; then
    yolo_yaml="$(find "$yolo_dir" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.yml' \) | sort | head -n 1)"
  fi
  [[ -n "$yolo_yaml" ]] || die "Missing YOLO YAML metadata in ${yolo_dir}"
  require_path "$yolo_yaml" "YOLO YAML metadata"
  require_path "${yolo_dir}/images" "YOLO images directory"
  require_path "${yolo_dir}/labels" "YOLO labels directory"
  yolo_yaml_in_release="data/$(basename "$yolo_dir")/$(basename "$yolo_yaml")"
else
  yolo_yaml_in_release=""
fi

python3 - "$dataset_index" <<'PY'
import csv
import sys
from collections import Counter

index_path = sys.argv[1]
with open(index_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not rows:
    raise SystemExit(f"Index is empty: {index_path}")

image_ids = {row.get("image_id", "") for row in rows}
categories = Counter(row.get("category_name", "") for row in rows)
challenges = Counter(row.get("challenge_primary", "") for row in rows)

print(f"[DATASET-RELEASE] Index rows: {len(rows):,}")
print(f"[DATASET-RELEASE] Unique images: {len(image_ids):,}")
print("[DATASET-RELEASE] Categories:")
for name, count in categories.most_common():
    print(f"[DATASET-RELEASE]   {name}: {count:,}")
print("[DATASET-RELEASE] Challenges:")
for name, count in challenges.most_common():
    print(f"[DATASET-RELEASE]   {name}: {count:,}")

if len(image_ids) != 500:
    raise SystemExit(f"Expected 500 images, found {len(image_ids)}")
if len(rows) != 4471:
    raise SystemExit(f"Expected 4,471 object rows, found {len(rows)}")
PY

if [[ "$dry_run" -eq 1 ]]; then
  log "Dry run complete. No archives were created."
  log "Planned dataset archive: ${dataset_archive}"
  if [[ "$include_yolo" -eq 1 ]]; then
    log "Planned YOLO archive: ${yolo_archive}"
  fi
  exit 0
fi

mkdir -p "$output_dir"

if [[ -e "$dataset_archive" && "$force" -ne 1 ]]; then
  die "Archive already exists: ${dataset_archive}. Pass --force to overwrite."
fi
if [[ "$include_yolo" -eq 1 && -e "$yolo_archive" && "$force" -ne 1 ]]; then
  die "Archive already exists: ${yolo_archive}. Pass --force to overwrite."
fi

log "Creating dataset archive..."
tar -C "$(dirname "$dataset_dir")" -czf "$dataset_archive" "$(basename "$dataset_dir")"

archive_names=("$dataset_archive_name")
if [[ "$include_yolo" -eq 1 ]]; then
  log "Creating YOLO export archive..."
  tar -C "$(dirname "$yolo_dir")" -czf "$yolo_archive" "$(basename "$yolo_dir")"
  archive_names+=("$yolo_archive_name")
fi

log "Writing checksums..."
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$output_dir" && sha256sum "${archive_names[@]}" > "$(basename "$checksums_file")")
elif command -v shasum >/dev/null 2>&1; then
  (cd "$output_dir" && shasum -a 256 "${archive_names[@]}" > "$(basename "$checksums_file")")
else
  die "Neither sha256sum nor shasum is available."
fi

log "Writing release notes template..."
python3 - "$dataset_index" "$release_notes_file" "$include_yolo" "$yolo_yaml_in_release" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

index_path = sys.argv[1]
output_path = Path(sys.argv[2])
include_yolo = sys.argv[3] == "1"
yolo_yaml_in_release = sys.argv[4]

with open(index_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

images = len({row.get("image_id", "") for row in rows})
objects = len(rows)
categories = Counter(row.get("category_name", "") for row in rows)
challenges = Counter(row.get("challenge_primary", "") for row in rows)

asset_lines = [
    "- `COGAR-SimRobotics-500_dataset.tar.gz`: full frozen benchmark dataset.",
]
extract_yolo = ""
verify_yolo = ""
if include_yolo:
    asset_lines.append(
        "- `COGAR-SimRobotics-500_yolov8seg_export.tar.gz`: YOLOv8-seg export."
    )
    extract_yolo = "tar -C data -xzf COGAR-SimRobotics-500_yolov8seg_export.tar.gz\n"
    verify_yolo = f"test -f {yolo_yaml_in_release}\n"
asset_lines.append("- `SHA256SUMS.txt`: archive checksums.")

category_lines = "\n".join(f"- `{name}`: {count:,}" for name, count in categories.most_common())
challenge_lines = "\n".join(f"- `{name}`: {count:,}" for name, count in challenges.most_common())
assets = "\n".join(asset_lines)

text = f"""# COGAR-SimRobotics-500 Dataset Release

Frozen simulated robotic-scene dataset for Assignment 2, Task 1.

## Contents

{assets}

## Dataset Summary

- Images: {images:,}
- Object instances: {objects:,}
- Object categories: {len(categories):,}
- Robotic challenge groups: {len(challenges):,}
- Image size: 640 x 480
- Simulation tool: BlenderProc

## Categories

{category_lines}

## Challenges

{challenge_lines}

## Extract

```bash
mkdir -p data
tar -C data -xzf COGAR-SimRobotics-500_dataset.tar.gz
{extract_yolo.rstrip()}
```

## Verify

```bash
sha256sum -c SHA256SUMS.txt
test -f data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv
test -d data/cogar_sim_500_final/rgb
test -d data/cogar_sim_500_final/instance_masks/final
{verify_yolo.rstrip()}
```

OCID is not included in this release. It is an external dataset used for the
separate massive robustness benchmark.
"""

output_path.write_text(text, encoding="utf-8")
PY

log "Release assets ready:"
ls -lh "$output_dir"
