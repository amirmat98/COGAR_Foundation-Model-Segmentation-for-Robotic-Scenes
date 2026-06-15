"""Train YOLOv8-seg baselines on Task 5A small subsets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task5_yolov8_seg.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for setup validation.")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selected_datasets(
    config: dict[str, Any],
    selected_names: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    datasets = []
    for name, dataset_config in config["datasets"].items():
        if selected_names is not None and name not in selected_names:
            continue
        if dataset_config.get("enabled", False):
            datasets.append((name, dataset_config))
    return datasets


def run_name(dataset_name: str, config: dict[str, Any], epochs: int, smoke: bool) -> str:
    suffix = config["model"].get("run_suffix", "yolov8_seg")
    if smoke:
        return f"{dataset_name}_{suffix}_smoke"
    return f"{dataset_name}_{suffix}_e{epochs}"


def expected_best_weight(results_root: Path, name: str) -> Path:
    return results_root / name / "weights" / "best.pt"


def build_train_kwargs(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    epochs = 1 if args.smoke else int(args.epochs or training["epochs"])
    batch = int(args.batch or training["batch"])
    image_size = int(args.image_size or training["image_size"])
    workers = int(args.workers or training["workers"])
    device = args.device if args.device is not None else training["device"]
    results_root = Path(config["task"]["results_root"])
    name = run_name(dataset_name, config, epochs, args.smoke)

    return {
        "data": str(Path(dataset_config["data_yaml"])),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch,
        "device": device,
        "workers": workers,
        "project": str(results_root),
        "name": name,
        "exist_ok": True,
        "seed": int(config["task"]["seed"]),
        "patience": int(training["patience"]),
        "optimizer": training.get("optimizer", "auto"),
        "cache": bool(training.get("cache", False)),
        "verbose": True,
    }


def train_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = config["model"]["checkpoint"]
    kwargs = build_train_kwargs(args, config, dataset_name, dataset_config)
    results_root = Path(config["task"]["results_root"])
    best_weight = expected_best_weight(results_root, kwargs["name"])

    print(
        f"[START] YOLOv8-seg dataset={dataset_name} checkpoint={checkpoint} "
        f"epochs={kwargs['epochs']} batch={kwargs['batch']} imgsz={kwargs['imgsz']} "
        f"device={kwargs['device']} output={results_root / kwargs['name']}",
        flush=True,
    )

    if not Path(kwargs["data"]).exists():
        raise FileNotFoundError(f"Missing YOLO dataset YAML: {kwargs['data']}")

    if best_weight.exists() and not args.rerun_complete:
        print(f"[SKIP] {dataset_name}: existing best weight {best_weight}", flush=True)
        return {
            "dataset": dataset_name,
            "status": "skipped_existing",
            "best_weight": str(best_weight),
            "train_kwargs": kwargs,
        }

    if args.dry_run:
        print(f"[DRY RUN] would train {dataset_name}: {kwargs}", flush=True)
        return {
            "dataset": dataset_name,
            "status": "dry_run",
            "best_weight": str(best_weight),
            "train_kwargs": kwargs,
        }

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Missing ultralytics. Install Task 5 GPU requirements before training."
        ) from exc

    started_at = time.perf_counter()
    model = YOLO(checkpoint)
    train_result = model.train(**kwargs)
    elapsed_s = time.perf_counter() - started_at

    status = "ok" if best_weight.exists() else "missing_best_weight"
    summary = {
        "dataset": dataset_name,
        "status": status,
        "elapsed_s": elapsed_s,
        "best_weight": str(best_weight),
        "train_kwargs": kwargs,
        "result_save_dir": str(getattr(train_result, "save_dir", "")),
    }
    print(
        f"[DONE] {dataset_name}: status={status} elapsed={elapsed_s / 60.0:.1f}min "
        f"best={best_weight}",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = Path(config["task"]["output_root"])
    summaries = []

    for dataset_name, dataset_config in selected_datasets(config, args.datasets):
        summary = train_dataset(args, config, dataset_name, dataset_config)
        summaries.append(summary)
        if not args.dry_run:
            write_json(output_root / f"{dataset_name}_train_summary.json", summary)

    if not args.dry_run:
        write_json(output_root / "summary.json", summaries)
    print("[DONE] YOLOv8-seg training wrapper", flush=True)


if __name__ == "__main__":
    main()
