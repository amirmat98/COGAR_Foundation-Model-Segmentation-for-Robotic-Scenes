"""Evaluate Task 5 supervised baselines for Task 6."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from segmentation_metrics import (  # noqa: E402
    annotation_to_rle,
    annotations_by_image,
    boundary_f1,
    category_name_by_id,
    encode_binary_mask,
    evaluate_coco_predictions,
    image_size_by_id,
    load_json,
    load_semantic_png,
    rle_to_mask,
    semantic_boundary_f1,
    semantic_confusion,
    semantic_metrics_from_confusion,
    summarize_instance_matches,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task6_evaluation.yaml")
    parser.add_argument("--baselines", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
    return yaml.safe_load(resolve_repo_path(path).read_text(encoding="utf-8"))


def selected_names(all_names: list[str], selected: list[str] | None) -> list[str]:
    if selected is None:
        return all_names
    selected_set = set(selected)
    return [name for name in all_names if name in selected_set]


def val_coco_path(dataset_name: str) -> Path:
    return REPO_ROOT / "outputs" / "task5_baselines" / "coco" / dataset_name / "instances_val_small.json"


def metric_output_path(config: dict[str, Any], baseline: str, dataset_name: str) -> Path:
    return (
        resolve_repo_path(config["task"]["output_root"])
        / "baselines"
        / baseline
        / f"{dataset_name}_metrics.json"
    )


def compact_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline": record["baseline"],
        "dataset": record["dataset"],
        "status": record["status"],
        "evaluation_type": record["evaluation_type"],
        "mIoU": record.get("mIoU"),
        "boundary_f1": record.get("boundary_f1"),
        "mask_AP": record.get("mask_AP"),
        "mask_AP50": record.get("mask_AP50"),
        "mask_AP75": record.get("mask_AP75"),
        "elapsed_s": record["elapsed_s"],
        "metrics_file": record["metrics_file"],
    }


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "baseline",
        "dataset",
        "status",
        "evaluation_type",
        "mIoU",
        "boundary_f1",
        "mask_AP",
        "mask_AP50",
        "mask_AP75",
        "elapsed_s",
        "metrics_file",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_summary_records(path: str | Path) -> dict[str, dict[str, Any]]:
    records = load_json(resolve_repo_path(path))
    return {record["dataset"]: record for record in records}


def evaluate_instance_predictions(
    baseline: str,
    dataset_name: str,
    annotation_file: Path,
    predictions_file: Path,
    tolerance_px: int,
    output_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    coco = load_json(annotation_file)
    predictions = load_json(predictions_file)
    gt_by_image = annotations_by_image(coco)
    image_sizes = image_size_by_id(coco)
    names_by_category = category_name_by_id(coco)

    predictions_by_image_category: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image_category[
            (int(prediction["image_id"]), int(prediction["category_id"]))
        ].append(prediction)

    matches = []
    for image_id, annotations in gt_by_image.items():
        height, width = image_sizes[image_id]
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            gt_rle = annotation_to_rle(annotation, height, width)
            candidates = predictions_by_image_category.get((image_id, category_id), [])
            if candidates:
                pred_rles = [candidate["segmentation"] for candidate in candidates]
                ious = mask_utils.iou(pred_rles, [gt_rle], [0]).reshape(-1)
                best_index = int(np.argmax(ious))
                iou = float(ious[best_index])
                bf1 = boundary_f1(
                    rle_to_mask(gt_rle),
                    rle_to_mask(pred_rles[best_index]),
                    tolerance_px,
                )
            else:
                iou = 0.0
                bf1 = 0.0
            matches.append(
                {
                    "annotation_id": int(annotation["id"]),
                    "image_id": image_id,
                    "category_id": category_id,
                    "category_name": names_by_category.get(category_id, str(category_id)),
                    "iou": iou,
                    "boundary_f1": bf1,
                }
            )

    image_ids = sorted(gt_by_image)
    mask_ap = evaluate_coco_predictions(annotation_file, predictions, iou_type="segm", image_ids=image_ids)
    instance_metrics = summarize_instance_matches(matches)
    summary = {
        "baseline": baseline,
        "dataset": dataset_name,
        "status": "ok",
        "evaluation_type": "instance_segmentation",
        "annotation_file": relative_to_repo(annotation_file),
        "prediction_file": relative_to_repo(predictions_file),
        "instance_metrics": instance_metrics,
        "mIoU": instance_metrics["mIoU"],
        "boundary_f1": instance_metrics["boundary_f1"],
        "mask_AP": mask_ap["AP"],
        "mask_AP50": mask_ap["AP50"],
        "mask_AP75": mask_ap["AP75"],
        "mask_AP_stats": mask_ap,
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary["metrics_file"] = relative_to_repo(output_path)
    write_json(output_path, summary)
    return summary


def yolo_category_map(dataset_name: str, config: dict[str, Any]) -> dict[int, int]:
    class_map_path = (
        resolve_repo_path(config["baselines"]["yolo8_seg"]["class_map_dir"])
        / f"{dataset_name}_class_map.json"
    )
    class_map = load_json(class_map_path)
    return {
        int(category["yolo_class_id"]): int(category["category_id"])
        for category in class_map["categories"]
    }


def resolve_image_path_from_coco(annotation_file: Path, image: dict[str, Any]) -> Path:
    dataset_name = annotation_file.parent.name
    task5_config = load_yaml("configs/task5_baselines.yaml")
    dataset_config = task5_config["datasets"][dataset_name]
    root = Path(dataset_config["root"])
    mode = dataset_config.get("image_path", {}).get("mode", "coco_file_name_relative_to_root")
    if mode == "coco_file_name_relative_to_root":
        return root / image["file_name"]
    if mode == "basename_in_image_dir":
        return Path(dataset_config["image_path"]["image_dir"]) / Path(image["file_name"]).name
    raise ValueError(f"Unsupported image path mode: {mode}")


def generate_yolo_predictions(
    dataset_name: str,
    record: dict[str, Any],
    config: dict[str, Any],
    annotation_file: Path,
    output_file: Path,
    args: argparse.Namespace,
) -> Path:
    if output_file.exists() and not args.rerun_complete:
        print(f"[SKIP] existing YOLO predictions {relative_to_repo(output_file)}", flush=True)
        return output_file

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing ultralytics. Install requirements-task5-gpu.txt.") from exc

    coco = load_json(annotation_file)
    category_by_yolo_class = yolo_category_map(dataset_name, config)
    baseline_config = config["baselines"]["yolo8_seg"]
    image_size = int(baseline_config["image_size"])
    confidence = float(baseline_config["confidence"])
    device = args.device if args.device is not None else 0
    weight = resolve_repo_path(record["best_weight"])
    model = YOLO(str(weight))
    predictions = []

    started_at = time.perf_counter()
    for index, image in enumerate(coco["images"], start=1):
        image_path = resolve_image_path_from_coco(annotation_file, image)
        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            device=device,
            retina_masks=True,
            verbose=False,
        )[0]
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            target_size = (int(image["height"]), int(image["width"]))
            for mask, box, cls_id, score in zip(masks, boxes, classes, scores):
                if mask.shape != target_size:
                    mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
                    mask_image = mask_image.resize((target_size[1], target_size[0]), Image.NEAREST)
                    mask = np.asarray(mask_image) > 0
                rle = encode_binary_mask(mask)
                x1, y1, x2, y2 = [float(value) for value in box]
                category_id = category_by_yolo_class.get(int(cls_id))
                if category_id is None:
                    continue
                predictions.append(
                    {
                        "image_id": int(image["id"]),
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                        "segmentation": rle,
                    }
                )
        if args.log_every and index % args.log_every == 0:
            print(
                f"[PROGRESS] yolo8_seg/{dataset_name}: predicted {index}/{len(coco['images'])} images",
                flush=True,
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(predictions), encoding="utf-8")
    print(
        f"[OK] yolo8_seg/{dataset_name}: wrote {len(predictions)} predictions "
        f"in {time.perf_counter() - started_at:.1f}s",
        flush=True,
    )
    return output_file


def evaluate_yolo(dataset_name: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    summary_records = load_summary_records(config["baselines"]["yolo8_seg"]["summary"])
    record = summary_records[dataset_name]
    annotation_file = val_coco_path(dataset_name)
    output_path = metric_output_path(config, "yolo8_seg", dataset_name)
    prediction_file = (
        resolve_repo_path(config["baselines"]["yolo8_seg"]["prediction_output_root"])
        / f"{dataset_name}_val_predictions.json"
    )
    if output_path.exists() and not args.rerun_complete:
        print(f"[SKIP] existing metrics {relative_to_repo(output_path)}", flush=True)
        return load_json(output_path)
    if args.dry_run:
        return {
            "baseline": "yolo8_seg",
            "dataset": dataset_name,
            "status": "dry_run",
            "prediction_file": relative_to_repo(prediction_file),
            "metrics_file": relative_to_repo(output_path),
        }
    prediction_file = generate_yolo_predictions(
        dataset_name,
        record,
        config,
        annotation_file,
        prediction_file,
        args,
    )
    return evaluate_instance_predictions(
        "yolo8_seg",
        dataset_name,
        annotation_file,
        prediction_file,
        int(config["metrics"]["boundary_tolerance_px"]),
        output_path,
    )


def evaluate_mask_rcnn(dataset_name: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    summary_records = load_summary_records(config["baselines"]["mask_rcnn"]["summary"])
    record = summary_records[dataset_name]
    prediction_file = resolve_repo_path(record["best_metrics"]["predictions"])
    annotation_file = val_coco_path(dataset_name)
    output_path = metric_output_path(config, "mask_rcnn", dataset_name)
    if output_path.exists() and not args.rerun_complete:
        print(f"[SKIP] existing metrics {relative_to_repo(output_path)}", flush=True)
        return load_json(output_path)
    if args.dry_run:
        return {
            "baseline": "mask_rcnn",
            "dataset": dataset_name,
            "status": "dry_run",
            "prediction_file": relative_to_repo(prediction_file),
            "metrics_file": relative_to_repo(output_path),
        }
    return evaluate_instance_predictions(
        "mask_rcnn",
        dataset_name,
        annotation_file,
        prediction_file,
        int(config["metrics"]["boundary_tolerance_px"]),
        output_path,
    )


def resolve_semantic_root(dataset_name: str, dataset_yaml: dict[str, Any]) -> Path:
    root = Path(dataset_yaml["root"])
    if root.exists():
        return root
    fallback = REPO_ROOT / "data" / "task5_baselines" / "deeplabv3plus" / dataset_name
    return fallback if fallback.exists() else root


def class_names_from_deeplab_summary(record: dict[str, Any]) -> list[str]:
    return list(record.get("class_names") or [])


def evaluate_deeplab(dataset_name: str, config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    summary_records = load_summary_records(config["baselines"]["deeplabv3plus"]["summary"])
    record = summary_records[dataset_name]
    output_path = metric_output_path(config, "deeplabv3plus", dataset_name)
    if output_path.exists() and not args.rerun_complete:
        print(f"[SKIP] existing metrics {relative_to_repo(output_path)}", flush=True)
        return load_json(output_path)
    dataset_yaml = load_yaml(
        resolve_repo_path(config["baselines"]["deeplabv3plus"]["data_config_dir"])
        / f"{dataset_name}.yaml"
    )
    predictions_dir = resolve_repo_path(record["best_metrics"]["predictions_dir"])
    root = resolve_semantic_root(dataset_name, dataset_yaml)
    mask_dir = root / dataset_yaml["val_masks"]
    class_names = class_names_from_deeplab_summary(record)
    num_classes = int(dataset_yaml["num_classes"])
    if not class_names:
        class_names = ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]

    if args.dry_run:
        return {
            "baseline": "deeplabv3plus",
            "dataset": dataset_name,
            "status": "dry_run",
            "predictions_dir": relative_to_repo(predictions_dir),
            "metrics_file": relative_to_repo(output_path),
        }

    started_at = time.perf_counter()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_values = []
    prediction_paths = sorted(predictions_dir.glob("*.png"))
    for index, prediction_path in enumerate(prediction_paths, start=1):
        target_path = mask_dir / prediction_path.name
        prediction = load_semantic_png(prediction_path)
        target = load_semantic_png(target_path)
        if target.shape != prediction.shape:
            target_image = Image.fromarray(target.astype(np.uint16), mode="I;16")
            target_image = target_image.resize((prediction.shape[1], prediction.shape[0]), Image.NEAREST)
            target = np.asarray(target_image, dtype=np.int64)
        confusion += semantic_confusion(prediction, target, num_classes)
        boundary_values.append(
            semantic_boundary_f1(
                prediction,
                target,
                class_names,
                int(config["metrics"]["boundary_tolerance_px"]),
            )
        )
        if args.log_every and index % args.log_every == 0:
            print(
                f"[PROGRESS] deeplabv3plus/{dataset_name}: evaluated {index}/{len(prediction_paths)} masks",
                flush=True,
            )

    semantic_metrics = semantic_metrics_from_confusion(confusion, class_names)
    boundary_f1_values = [item["boundary_f1"] for item in boundary_values]
    foreground_boundary_f1_values = [item["foreground_boundary_f1"] for item in boundary_values]
    per_category_boundary: dict[str, list[float]] = defaultdict(list)
    for item in boundary_values:
        for category_name, value in item["per_category_boundary_f1"].items():
            per_category_boundary[category_name].append(float(value))

    boundary_summary = {
        "boundary_f1": float(np.mean(boundary_f1_values)) if boundary_f1_values else 0.0,
        "foreground_boundary_f1": float(np.mean(foreground_boundary_f1_values))
        if foreground_boundary_f1_values
        else 0.0,
        "per_category_boundary_f1": {
            category_name: float(np.mean(values))
            for category_name, values in sorted(per_category_boundary.items())
        },
    }
    summary = {
        "baseline": "deeplabv3plus",
        "dataset": dataset_name,
        "status": "ok",
        "evaluation_type": "semantic_segmentation",
        "prediction_count": len(prediction_paths),
        "predictions_dir": relative_to_repo(predictions_dir),
        "mIoU": semantic_metrics["mIoU"],
        "foreground_mIoU": semantic_metrics["foreground_mIoU"],
        "boundary_f1": boundary_summary["boundary_f1"],
        "foreground_boundary_f1": boundary_summary["foreground_boundary_f1"],
        "mask_AP": None,
        "mask_AP50": None,
        "mask_AP75": None,
        "mask_AP_note": "not_applicable_for_semantic_segmentation",
        "semantic_metrics": semantic_metrics,
        "boundary_metrics": boundary_summary,
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary["metrics_file"] = relative_to_repo(output_path)
    write_json(output_path, summary)
    return summary


def evaluate_baseline(
    baseline: str,
    dataset_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"[START] evaluating {baseline}/{dataset_name}", flush=True)
    if baseline == "yolo8_seg":
        return evaluate_yolo(dataset_name, config, args)
    if baseline == "mask_rcnn":
        return evaluate_mask_rcnn(dataset_name, config, args)
    if baseline == "deeplabv3plus":
        return evaluate_deeplab(dataset_name, config, args)
    raise ValueError(f"Unsupported baseline: {baseline}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task5_config = load_yaml("configs/task5_baselines.yaml")

    baseline_names = selected_names(
        [
            name
            for name, baseline_config in config["baselines"].items()
            if baseline_config.get("enabled", False)
        ],
        args.baselines,
    )
    dataset_names = selected_names(
        [
            name
            for name, dataset_config in task5_config["datasets"].items()
            if dataset_config.get("enabled", False)
        ],
        args.datasets,
    )

    rows = []
    for baseline in baseline_names:
        for dataset_name in dataset_names:
            summary = evaluate_baseline(baseline, dataset_name, config, args)
            rows.append(compact_row(summary) if not args.dry_run else summary)

    if not args.dry_run:
        output_root = resolve_repo_path(config["task"]["output_root"]) / "baselines"
        write_json(output_root / "summary.json", rows)
        write_summary_csv(output_root / "summary.csv", rows)
        print(f"[DONE] wrote {relative_to_repo(output_root / 'summary.csv')}", flush=True)
    print("[DONE] Task 6 baseline evaluation", flush=True)


if __name__ == "__main__":
    main()
