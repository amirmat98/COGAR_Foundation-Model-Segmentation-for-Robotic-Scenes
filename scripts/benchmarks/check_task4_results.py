"""Validate Task 4 prediction JSONL files against prompt manifest counts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
PROMPT_MODES = ("point", "box", "automatic")

from cogar_seg.config import load_config as load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_zero_shot_sam.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompt-modes", nargs="*", choices=PROMPT_MODES, default=None)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing files without returning a failing exit code.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return load_project_config(path)


def selected_keys(
    items: dict[str, Any],
    selected: list[str] | None,
    enabled_only: bool = True,
) -> list[str]:
    keys = []
    for key, value in items.items():
        if selected is not None and key not in selected:
            continue
        if enabled_only and not value.get("enabled", False):
            continue
        keys.append(key)
    return keys


def prompt_manifest_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["task"]["prompt_manifest_dir"]) / f"{dataset}_instances.jsonl"


def output_path(
    config: dict[str, Any],
    dataset: str,
    model: str,
    prompt_mode: str,
) -> Path:
    return (
        Path(config["task"]["output_root"])
        / dataset
        / model
        / f"{prompt_mode}_predictions.jsonl"
    )


def prompt_stats(path: Path) -> dict[str, int]:
    records = 0
    image_ids: OrderedDict[Any, None] = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            image_ids.setdefault(record["image_id"], None)
    return {"records": records, "images": len(image_ids)}


def expected_count(stats: dict[str, int], prompt_mode: str) -> int:
    if prompt_mode == "automatic":
        return stats["images"]
    return stats["records"]


def validate_jsonl(path: Path, prompt_mode: str) -> dict[str, Any]:
    valid_rows = 0
    blank_rows = 0
    bad_rows: list[tuple[int, str, str]] = []
    record_indexes: list[Any] = []
    image_ids: list[Any] = []
    annotation_ids: list[Any] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                blank_rows += 1
                continue
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                bad_rows.append((line_no, repr(exc), line[:120]))
                continue
            valid_rows += 1
            record_indexes.append(row.get("record_index"))
            image_ids.append(row.get("image_id"))
            if prompt_mode != "automatic":
                annotation_ids.append(row.get("annotation_id"))

    duplicate_record_indexes = [
        key for key, value in Counter(record_indexes).items() if value > 1
    ]
    duplicate_keys = duplicate_record_indexes
    if prompt_mode == "automatic":
        duplicate_image_ids = [key for key, value in Counter(image_ids).items() if value > 1]
        duplicate_keys.extend([f"image_id:{key}" for key in duplicate_image_ids])
    else:
        duplicate_annotation_ids = [
            key for key, value in Counter(annotation_ids).items() if value > 1
        ]
        duplicate_keys.extend([f"annotation_id:{key}" for key in duplicate_annotation_ids])

    return {
        "valid_rows": valid_rows,
        "blank_rows": blank_rows,
        "bad_rows": bad_rows,
        "duplicate_keys": duplicate_keys,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = selected_keys(config["datasets"], args.datasets)
    models = selected_keys(config["models"], args.models)
    prompt_modes = args.prompt_modes or list(PROMPT_MODES)

    failed = False
    for dataset in datasets:
        manifest = prompt_manifest_path(config, dataset)
        if not manifest.exists():
            print(f"[MISSING] {dataset}: prompt manifest {manifest}", flush=True)
            failed = True
            continue
        stats = prompt_stats(manifest)
        print(
            f"[DATASET] {dataset}: {stats['records']} prompted records, "
            f"{stats['images']} unique images",
            flush=True,
        )

        for model in models:
            for prompt_mode in prompt_modes:
                expected = expected_count(stats, prompt_mode)
                path = output_path(config, dataset, model, prompt_mode)
                label = f"{dataset}/{model}/{prompt_mode}"
                if not path.exists():
                    print(f"[MISSING] {label}: expected={expected} path={path}", flush=True)
                    if not args.allow_missing:
                        failed = True
                    continue

                validation = validate_jsonl(path, prompt_mode)
                valid_rows = validation["valid_rows"]
                bad_count = len(validation["bad_rows"])
                duplicate_count = len(validation["duplicate_keys"])
                status = "OK"
                if valid_rows != expected or bad_count or duplicate_count:
                    status = "BAD"
                    failed = True
                print(
                    f"[{status}] {label}: expected={expected} valid={valid_rows} "
                    f"bad={bad_count} blank={validation['blank_rows']} "
                    f"duplicates={duplicate_count} path={path}",
                    flush=True,
                )
                if bad_count:
                    print(f"  bad_rows_sample={validation['bad_rows'][:3]}", flush=True)
                if duplicate_count:
                    print(
                        f"  duplicate_sample={validation['duplicate_keys'][:10]}",
                        flush=True,
                    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
