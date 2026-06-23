"""Summarize Task 9 lightweight SAM quality/speed trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cogar_seg.config import load_config as load_project_config  # noqa: E402

CHECKPOINT_SIZE_MB_FALLBACK = {
    "sam_vit_h": 2564.55,
    "sam_vit_b": 375.04,
    "sam2_hiera_large": 898.08,
    "fastsam_x": 144.94,
    "mobile_sam_vit_t": 40.73,
    "efficient_sam_ti": 40.98,
    "efficient_sam_s": 105.74,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task9-config", default="configs/task9_lightweight_sam.yaml")
    parser.add_argument("--task4-config", default="configs/task4_zero_shot_sam.yaml")
    parser.add_argument(
        "--task9-eval",
        default="outputs/task9_lightweight_sam/evaluation/zero_shot/test/summary.csv",
    )
    parser.add_argument(
        "--task9-speed",
        default="outputs/task9_lightweight_sam/inference_speed/summary.csv",
    )
    parser.add_argument(
        "--task6-zero-shot",
        default="outputs/task6_evaluation/zero_shot/test/summary.csv",
    )
    parser.add_argument(
        "--task6-baselines",
        default="outputs/task6_evaluation/baselines/test/summary.csv",
    )
    parser.add_argument("--task7-speed", default="outputs/task7_inference_speed/summary.csv")
    parser.add_argument("--output-root", default="outputs/task9_lightweight_sam/summary")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    return load_project_config(resolve(path))


def read_csv(path: str | Path, required: bool = True) -> list[dict[str, str]]:
    resolved = resolve(path)
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"Missing input CSV: {resolved}")
        print(f"[WARN] missing optional CSV: {relative(resolved)}", flush=True)
        return []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: str | Path, data: Any) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def model_size_mb(configs: list[dict[str, Any]]) -> dict[str, float | None]:
    sizes: dict[str, float | None] = {}
    for config in configs:
        for model_name, model_config in config.get("models", {}).items():
            checkpoint = model_config.get("checkpoint")
            if not checkpoint:
                sizes[model_name] = CHECKPOINT_SIZE_MB_FALLBACK.get(model_name)
                continue
            path = resolve(checkpoint)
            sizes[model_name] = (
                round(path.stat().st_size / 1e6, 2)
                if path.exists()
                else CHECKPOINT_SIZE_MB_FALLBACK.get(model_name)
            )
    return sizes


def normalize_zero_shot_quality(
    rows: list[dict[str, str]],
    source_group: str,
    lightweight_models: set[str],
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        model = row.get("model", "")
        normalized.append(
            {
                "dataset": row.get("dataset", ""),
                "model": model,
                "model_group": "lightweight_sam" if model in lightweight_models else source_group,
                "prompt_mode": row.get("prompt_mode", ""),
                "split": row.get("split", ""),
                "evaluation_images": row.get("evaluation_images", ""),
                "split_sha256": row.get("split_sha256", ""),
                "status": row.get("status", ""),
                "evaluation_type": "zero_shot",
                "mIoU": as_float(row.get("mIoU")),
                "boundary_f1": as_float(row.get("boundary_f1")),
                "mask_AP": as_float(row.get("mask_AP")),
                "mask_AP50": as_float(row.get("mask_AP50")),
                "mask_AP75": as_float(row.get("mask_AP75")),
                "quality_metrics_file": row.get("metrics_file", ""),
            }
        )
    return normalized


def normalize_baseline_quality(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        model = row.get("baseline", "")
        normalized.append(
            {
                "dataset": row.get("dataset", ""),
                "model": model,
                "model_group": "baseline",
                "prompt_mode": "inference",
                "split": row.get("split", ""),
                "evaluation_images": row.get("evaluation_images", ""),
                "split_sha256": row.get("split_sha256", ""),
                "status": row.get("status", ""),
                "evaluation_type": row.get("evaluation_type", "baseline"),
                "mIoU": as_float(row.get("mIoU")),
                "boundary_f1": as_float(row.get("boundary_f1")),
                "mask_AP": as_float(row.get("mask_AP")),
                "mask_AP50": as_float(row.get("mask_AP50")),
                "mask_AP75": as_float(row.get("mask_AP75")),
                "quality_metrics_file": row.get("metrics_file", ""),
            }
        )
    return normalized


def normalize_speed(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    speed = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("model", ""),
            row.get("prompt_mode", ""),
            row.get("device", ""),
        )
        if key in speed:
            continue
        speed[key] = {
            "device": row.get("device", ""),
            "fps": as_float(row.get("fps")),
            "latency_mean_ms": as_float(row.get("latency_mean_ms")),
            "latency_p95_ms": as_float(row.get("latency_p95_ms")),
            "sample_images": as_float(row.get("sample_images")),
            "speed_metrics_file": row.get("metrics_file", ""),
            "speed_status": row.get("status", ""),
        }
    return speed


def combined_rows(
    quality_rows: list[dict[str, Any]],
    speed_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    size_by_model: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows = []
    devices = sorted({key[3] for key in speed_by_key if key[3]})
    for quality in quality_rows:
        for device in devices:
            key = (
                str(quality["dataset"]),
                str(quality["model"]),
                str(quality["prompt_mode"]),
                device,
            )
            speed = speed_by_key.get(key)
            if not speed:
                continue
            miou = quality.get("mIoU")
            fps = speed.get("fps")
            latency = speed.get("latency_mean_ms")
            row = {
                **quality,
                **speed,
                "checkpoint_size_mb": size_by_model.get(str(quality["model"])),
                "real_time_30fps": bool(fps is not None and fps >= 30.0),
                "miou_fps_product": miou * fps if miou is not None and fps is not None else None,
                "miou_per_ms": miou / latency if miou is not None and latency else None,
            }
            rows.append(row)
    return rows


def best_rows(rows: list[dict[str, Any]], group_filter: str, metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model_group") != group_filter:
            continue
        value = row.get(metric)
        if value is None:
            continue
        grouped[(row["dataset"], row["prompt_mode"], row["device"])].append(row)

    best = []
    for key, candidates in sorted(grouped.items()):
        chosen = max(candidates, key=lambda item: item.get(metric) or -1.0)
        best.append(
            {
                "dataset": key[0],
                "prompt_mode": key[1],
                "device": key[2],
                "selection_metric": metric,
                "model": chosen["model"],
                "mIoU": chosen.get("mIoU"),
                "boundary_f1": chosen.get("boundary_f1"),
                "mask_AP": chosen.get("mask_AP"),
                "fps": chosen.get("fps"),
                "latency_mean_ms": chosen.get("latency_mean_ms"),
                "checkpoint_size_mb": chosen.get("checkpoint_size_mb"),
                "real_time_30fps": chosen.get("real_time_30fps"),
                "miou_fps_product": chosen.get("miou_fps_product"),
            }
        )
    return best


def write_report(path: Path, tradeoff_rows: list[dict[str, Any]], recommendations: list[dict[str, Any]]) -> None:
    total_light = sum(1 for row in tradeoff_rows if row.get("model_group") == "lightweight_sam")
    realtime_light = sum(
        1
        for row in tradeoff_rows
        if row.get("model_group") == "lightweight_sam" and row.get("real_time_30fps")
    )
    lines = [
        "# Task 9: Lightweight SAM Edge-Deployment Trade-Off",
        "",
        "This report compares MobileSAM and EfficientSAM variants against the heavier Task 4 models and the supervised Task 5 baselines where matching quality/speed rows are available.",
        "",
        "EfficientSAM automatic mode is evaluated as grid-prompt automatic proposal generation, because the official EfficientSAM API exposes direct point/box tensor inference rather than the same automatic-mask-generator class used by SAM and MobileSAM.",
        "",
        "## Output Tables",
        "",
        "- `lightweight_quality.csv`: Task 9 quality metrics only.",
        "- `speed_quality_tradeoff.csv`: quality joined with GPU/CPU speed and checkpoint size.",
        "- `recommendations.csv`: best lightweight model per dataset, prompt mode, and device by mIoU and by mIoU-FPS product.",
        "",
        "## Figure",
        "",
        "![Lightweight SAM CUDA speed-quality trade-off](../../final_benchmark_assets/plots/lightweight_sam_tradeoff_cuda.png)",
        "",
        "## Compact Summary",
        "",
        f"- Joined lightweight rows: {total_light}",
        f"- Lightweight rows at or above 30 FPS: {realtime_light}",
        "",
        "## Recommended Lightweight Choices",
        "",
        "| Dataset | Prompt | Device | Metric | Model | mIoU | FPS | Checkpoint MB |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in recommendations[:36]:
        lines.append(
            "| {dataset} | {prompt_mode} | {device} | {selection_metric} | {model} | "
            "{miou:.4f} | {fps:.2f} | {size} |".format(
                dataset=row["dataset"],
                prompt_mode=row["prompt_mode"],
                device=row["device"],
                selection_metric=row["selection_metric"],
                model=row["model"],
                miou=float(row["mIoU"] or 0.0),
                fps=float(row["fps"] or 0.0),
                size="" if row.get("checkpoint_size_mb") is None else row["checkpoint_size_mb"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    task9_config = load_yaml(args.task9_config)
    task4_config = load_yaml(args.task4_config)
    lightweight_models = set(task9_config["models"].keys())

    quality_rows = []
    quality_rows.extend(
        normalize_zero_shot_quality(
            read_csv(args.task9_eval),
            source_group="lightweight_sam",
            lightweight_models=lightweight_models,
        )
    )
    quality_rows.extend(
        normalize_zero_shot_quality(
            read_csv(args.task6_zero_shot, required=False),
            source_group="heavy_sam",
            lightweight_models=lightweight_models,
        )
    )
    quality_rows.extend(normalize_baseline_quality(read_csv(args.task6_baselines, required=False)))

    invalid_splits = sorted({row.get("split", "") for row in quality_rows if row.get("split") != "test"})
    if invalid_splits:
        raise ValueError(
            "Final speed-quality comparisons require test-only quality rows; "
            f"found splits: {invalid_splits}"
        )

    speed_rows = read_csv(args.task9_speed)
    speed_rows.extend(read_csv(args.task7_speed, required=False))
    speed_by_key = normalize_speed(speed_rows)

    size_by_model = model_size_mb([task9_config, task4_config])
    tradeoff = combined_rows(quality_rows, speed_by_key, size_by_model)
    recommendations = best_rows(tradeoff, "lightweight_sam", "mIoU")
    recommendations.extend(best_rows(tradeoff, "lightweight_sam", "miou_fps_product"))

    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_fields = [
        "dataset",
        "model",
        "model_group",
        "prompt_mode",
        "split",
        "evaluation_images",
        "split_sha256",
        "status",
        "evaluation_type",
        "mIoU",
        "boundary_f1",
        "mask_AP",
        "mask_AP50",
        "mask_AP75",
        "quality_metrics_file",
    ]
    tradeoff_fields = [
        *quality_fields,
        "device",
        "speed_status",
        "fps",
        "latency_mean_ms",
        "latency_p95_ms",
        "sample_images",
        "checkpoint_size_mb",
        "real_time_30fps",
        "miou_fps_product",
        "miou_per_ms",
        "speed_metrics_file",
    ]
    recommendation_fields = [
        "dataset",
        "prompt_mode",
        "device",
        "selection_metric",
        "model",
        "mIoU",
        "boundary_f1",
        "mask_AP",
        "fps",
        "latency_mean_ms",
        "checkpoint_size_mb",
        "real_time_30fps",
        "miou_fps_product",
    ]
    write_csv(output_root / "lightweight_quality.csv", quality_rows, quality_fields)
    write_csv(output_root / "speed_quality_tradeoff.csv", tradeoff, tradeoff_fields)
    write_csv(output_root / "recommendations.csv", recommendations, recommendation_fields)
    write_json(
        output_root / "summary.json",
        {
            "quality_rows": len(quality_rows),
            "tradeoff_rows": len(tradeoff),
            "recommendation_rows": len(recommendations),
            "lightweight_models": sorted(lightweight_models),
            "outputs": {
                "lightweight_quality": relative(output_root / "lightweight_quality.csv"),
                "speed_quality_tradeoff": relative(output_root / "speed_quality_tradeoff.csv"),
                "recommendations": relative(output_root / "recommendations.csv"),
                "report": relative(output_root / "task9_lightweight_sam_report.md"),
            },
        },
    )
    write_report(output_root / "task9_lightweight_sam_report.md", tradeoff, recommendations)
    print(f"[DONE] wrote {relative(output_root)}", flush=True)


if __name__ == "__main__":
    main()
