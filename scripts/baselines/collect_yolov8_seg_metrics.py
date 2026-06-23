"""Collect compact YOLOv8-seg training metrics from Ultralytics run folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cogar_seg.config import load_config as load_project_config  # noqa: E402

METRIC_PRIORITY = (
    "metrics/mAP50-95(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
)

COMPACT_KEYS = (
    "epoch",
    "time",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "val/box_loss",
    "val/seg_loss",
    "val/cls_loss",
    "val/dfl_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task5_yolov8_seg.yaml")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return load_project_config(resolve_repo_path(path))


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_value(value: str | None) -> int | float | str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    if number.is_integer():
        return int(number)
    return number


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_results_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            rows.append({key.strip(): parse_value(value) for key, value in raw_row.items()})
    return rows


def pick_best_row(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    for metric_name in METRIC_PRIORITY:
        if any(is_number(row.get(metric_name)) for row in rows):
            best = max(
                rows,
                key=lambda row: row.get(metric_name)
                if is_number(row.get(metric_name))
                else float("-inf"),
            )
            return best, metric_name
    raise ValueError("No YOLOv8 validation metric column found in results.csv")


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in COMPACT_KEYS if key in row}


def collect_record(record: dict[str, Any]) -> dict[str, Any]:
    dataset_name = record["dataset"]
    run_dir = resolve_repo_path(record["result_save_dir"])
    results_csv = run_dir / "results.csv"
    best_weight = resolve_repo_path(record["best_weight"])

    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results.csv for {dataset_name}: {results_csv}")
    if not best_weight.exists():
        raise FileNotFoundError(f"Missing best.pt for {dataset_name}: {best_weight}")

    rows = read_results_csv(results_csv)
    if not rows:
        raise ValueError(f"Empty results.csv for {dataset_name}: {results_csv}")

    best_row, selection_metric = pick_best_row(rows)
    final_row = rows[-1]

    return {
        "dataset": dataset_name,
        "status": record.get("status"),
        "epochs_recorded": len(rows),
        "selection_metric": selection_metric,
        "best_epoch": best_row.get("epoch"),
        "best_metrics": compact_metrics(best_row),
        "final_epoch": final_row.get("epoch"),
        "final_metrics": compact_metrics(final_row),
        "elapsed_s": record.get("elapsed_s"),
        "run_dir": relative_to_repo(run_dir),
        "results_csv": relative_to_repo(results_csv),
        "best_weight": relative_to_repo(best_weight),
    }


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    metric_keys = set()
    for record in records:
        metric_keys.update(f"best/{key}" for key in record["best_metrics"])
        metric_keys.update(f"final/{key}" for key in record["final_metrics"])

    fieldnames = [
        "dataset",
        "status",
        "epochs_recorded",
        "selection_metric",
        "best_epoch",
        "final_epoch",
        "elapsed_s",
        "run_dir",
        "best_weight",
        *sorted(metric_keys),
    ]

    for record in records:
        row = {key: record.get(key) for key in fieldnames}
        for key, value in record["best_metrics"].items():
            row[f"best/{key}"] = value
        for key, value in record["final_metrics"].items():
            row[f"final/{key}"] = value
        rows.append(row)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = resolve_repo_path(config["task"]["output_root"])
    summary_path = resolve_repo_path(args.summary or output_root / "summary.json")
    output_json = resolve_repo_path(args.output_json or output_root / "metrics_summary.json")
    output_csv = resolve_repo_path(args.output_csv or output_root / "metrics_summary.csv")

    training_records = load_json(summary_path)
    metric_records = []
    for record in training_records:
        print(f"[START] collecting YOLOv8 metrics for {record['dataset']}", flush=True)
        metric_records.append(collect_record(record))
        print(f"[OK] {record['dataset']}", flush=True)

    write_json(output_json, metric_records)
    write_csv(output_csv, metric_records)
    print(f"[DONE] wrote {relative_to_repo(output_json)}", flush=True)
    print(f"[DONE] wrote {relative_to_repo(output_csv)}", flush=True)


if __name__ == "__main__":
    main()
