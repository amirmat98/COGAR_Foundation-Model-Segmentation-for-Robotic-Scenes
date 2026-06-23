"""Evaluate Task 4 zero-shot SAM-family predictions for Task 6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from segmentation_metrics import (  # noqa: E402
    annotation_to_rle,
    annotations_by_id,
    annotations_by_image,
    boundary_f1,
    category_name_by_id,
    class_agnostic_coco,
    evaluate_coco_predictions,
    image_size_by_id,
    load_json,
    rle_iou,
    rle_to_mask,
    select_prediction,
    summarize_instance_matches,
    write_json,
    write_temp_coco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task6_evaluation.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompt-modes", nargs="*", default=None)
    parser.add_argument(
        "--split",
        choices=("full", "train", "val", "test"),
        default=None,
        help="Evaluation split. Defaults to evaluation.split in the config.",
    )
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
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


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def selected_names(all_names: list[str], selected: list[str] | None) -> list[str]:
    if selected is None:
        return all_names
    return [name for name in all_names if name in set(selected)]


def prediction_path(task4_config: dict[str, Any], dataset: str, model: str, prompt_mode: str) -> Path:
    return (
        resolve_repo_path(task4_config["task"]["output_root"])
        / dataset
        / model
        / f"{prompt_mode}_predictions.jsonl"
    )


def evaluation_output_path(config: dict[str, Any], dataset: str, model: str, prompt_mode: str) -> Path:
    split = str(config.get("evaluation", {}).get("split", "full"))
    root = resolve_repo_path(config["task"]["output_root"]) / "zero_shot"
    if split != "full":
        root = root / split
    return root / dataset / model / f"{prompt_mode}_metrics.json"


def evaluation_split(config: dict[str, Any], args: argparse.Namespace) -> str:
    return str(args.split or config.get("evaluation", {}).get("split", "full"))


def split_image_ids_path(config: dict[str, Any], dataset: str, split: str) -> Path | None:
    if split == "full":
        return None
    split_root = resolve_repo_path(
        config.get("evaluation", {}).get(
            "split_root",
            "outputs/task5_baselines/splits",
        )
    )
    return split_root / dataset / f"{split}_image_ids.txt"


def load_evaluation_image_ids(
    config: dict[str, Any],
    dataset: str,
    split: str,
) -> tuple[set[int] | None, Path | None]:
    path = split_image_ids_path(config, dataset, split)
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(f"Missing {split} split file: {path}")
    image_ids = {
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not image_ids:
        raise ValueError(f"Empty {split} split file: {path}")
    return image_ids, path


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_observed_images(
    observed: set[int],
    expected: set[int] | None,
    args: argparse.Namespace,
    run_name: str,
) -> None:
    if expected is None or args.max_records is not None:
        return
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{run_name} does not cover the configured test split: "
            f"missing={len(missing)} extra={len(extra)} "
            f"missing_sample={missing[:10]} extra_sample={extra[:10]}"
        )


def compact_row(record: dict[str, Any]) -> dict[str, Any]:
    mask_ap = record.get("mask_AP") or {}
    return {
        "dataset": record["dataset"],
        "model": record["model"],
        "prompt_mode": record["prompt_mode"],
        "split": record.get("split", "full"),
        "evaluation_images": record.get("evaluation_images"),
        "split_sha256": record.get("split_sha256"),
        "status": record["status"],
        "records": record["records"],
        "instances": record["instance_metrics"]["instances"],
        "mIoU": record["instance_metrics"]["mIoU"],
        "boundary_f1": record["instance_metrics"]["boundary_f1"],
        "mask_AP": mask_ap.get("AP"),
        "mask_AP50": mask_ap.get("AP50"),
        "mask_AP75": mask_ap.get("AP75"),
        "elapsed_s": record["elapsed_s"],
        "metrics_file": record["metrics_file"],
    }


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "prompt_mode",
        "split",
        "evaluation_images",
        "split_sha256",
        "status",
        "records",
        "instances",
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


def build_detection(
    row: dict[str, Any],
    prediction: dict[str, Any],
    category_id: int,
    default_score: float,
) -> dict[str, Any]:
    segmentation = prediction["segmentation"]
    return {
        "image_id": int(row["image_id"]),
        "category_id": int(category_id),
        "segmentation": segmentation,
        "bbox": [float(value) for value in prediction.get("bbox", mask_utils.toBbox(segmentation))],
        "score": float(prediction.get("score") if prediction.get("score") is not None else default_score),
    }


def evaluate_prompted(
    prediction_file: Path,
    coco: dict[str, Any],
    annotation_file: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    allowed_image_ids: set[int] | None,
) -> dict[str, Any]:
    gt_by_id = annotations_by_id(coco)
    image_sizes = image_size_by_id(coco)
    names_by_category = category_name_by_id(coco)
    tolerance_px = int(metadata["boundary_tolerance_px"])
    default_score = float(metadata["score_default"])

    matches = []
    detections = []
    processed = 0
    started_at = time.perf_counter()
    progress_every = args.log_every or 1000

    for row in iter_jsonl(prediction_file):
        image_id = int(row["image_id"])
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            continue
        if args.max_records is not None and processed >= args.max_records:
            break
        processed += 1
        annotation_id = int(row["annotation_id"])
        gt_annotation = gt_by_id[annotation_id]
        height, width = image_sizes[int(row["image_id"])]
        gt_rle = annotation_to_rle(gt_annotation, height, width)
        prediction = select_prediction(row.get("predictions", []), default_score)

        if prediction is None:
            iou = 0.0
            bf1 = 0.0
        else:
            pred_rle = prediction["segmentation"]
            iou = rle_iou(gt_rle, pred_rle)
            bf1 = boundary_f1(rle_to_mask(gt_rle), rle_to_mask(pred_rle), tolerance_px)
            detections.append(build_detection(row, prediction, int(row["category_id"]), default_score))

        matches.append(
            {
                "annotation_id": annotation_id,
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "category_name": names_by_category.get(int(row["category_id"]), str(row["category_id"])),
                "iou": iou,
                "boundary_f1": bf1,
            }
        )
        if progress_every and processed % progress_every == 0:
            print(
                f"[PROGRESS] {metadata['run_name']}: {processed} prompted records, "
                f"elapsed={time.perf_counter() - started_at:.1f}s",
                flush=True,
            )

    image_ids = sorted({item["image_id"] for item in matches})
    validate_observed_images(set(image_ids), allowed_image_ids, args, metadata["run_name"])
    print(
        f"[AP] {metadata['run_name']}: evaluating COCO mask AP with "
        f"{len(detections)} detections on {len(image_ids)} images",
        flush=True,
    )
    mask_ap = evaluate_coco_predictions(annotation_file, detections, iou_type="segm", image_ids=image_ids)
    summary = {
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "prompt_mode": metadata["prompt_mode"],
        "status": "ok",
        "records": processed,
        "prediction_file": relative_to_repo(prediction_file),
        "annotation_file": str(annotation_file),
        "evaluation_type": "prompted_instance",
        "instance_metrics": summarize_instance_matches(matches),
        "mask_AP": mask_ap,
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary["metrics_file"] = relative_to_repo(output_path)
    write_json(output_path, summary)
    return summary


def evaluate_automatic(
    prediction_file: Path,
    coco: dict[str, Any],
    annotation_file: Path,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    allowed_image_ids: set[int] | None,
) -> dict[str, Any]:
    gt_by_image = annotations_by_image(coco)
    image_sizes = image_size_by_id(coco)
    names_by_category = category_name_by_id(coco)
    tolerance_px = int(metadata["boundary_tolerance_px"])
    default_score = float(metadata["score_default"])

    matches = []
    detections = []
    processed = 0
    started_at = time.perf_counter()
    progress_every = min(args.log_every, 100) if args.log_every else 100

    for row in iter_jsonl(prediction_file):
        image_id = int(row["image_id"])
        if allowed_image_ids is not None and image_id not in allowed_image_ids:
            continue
        if args.max_records is not None and processed >= args.max_records:
            break
        processed += 1
        gt_annotations = gt_by_image.get(image_id, [])
        if not gt_annotations:
            continue
        height, width = image_sizes[image_id]
        gt_rles = [annotation_to_rle(annotation, height, width) for annotation in gt_annotations]
        predictions = row.get("predictions", [])
        pred_rles = [prediction["segmentation"] for prediction in predictions]

        for prediction in predictions:
            detections.append(build_detection(row, prediction, 1, default_score))

        if pred_rles:
            iou_matrix = mask_utils.iou(gt_rles, pred_rles, [0] * len(pred_rles))
        else:
            iou_matrix = np.zeros((len(gt_rles), 0), dtype=np.float32)

        for gt_index, annotation in enumerate(gt_annotations):
            if pred_rles:
                pred_index = int(np.argmax(iou_matrix[gt_index]))
                iou = float(iou_matrix[gt_index, pred_index])
                bf1 = boundary_f1(
                    rle_to_mask(gt_rles[gt_index]),
                    rle_to_mask(pred_rles[pred_index]),
                    tolerance_px,
                )
            else:
                iou = 0.0
                bf1 = 0.0
            category_id = int(annotation["category_id"])
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

        if progress_every and processed % progress_every == 0:
            print(
                f"[PROGRESS] {metadata['run_name']}: {processed} automatic images, "
                f"matches={len(matches)}, detections={len(detections)}, "
                f"elapsed={time.perf_counter() - started_at:.1f}s",
                flush=True,
            )

    image_ids = sorted({item["image_id"] for item in matches})
    validate_observed_images(set(image_ids), allowed_image_ids, args, metadata["run_name"])
    class_agnostic = class_agnostic_coco(coco)
    for detection in detections:
        detection["category_id"] = 1
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_annotation_file = write_temp_coco(Path(tmpdir) / "class_agnostic_gt.json", class_agnostic)
        print(
            f"[AP] {metadata['run_name']}: evaluating class-agnostic COCO mask AP "
            f"with {len(detections)} detections on {len(image_ids)} images",
            flush=True,
        )
        mask_ap = evaluate_coco_predictions(
            temp_annotation_file,
            detections,
            iou_type="segm",
            image_ids=image_ids,
        )

    summary = {
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "prompt_mode": metadata["prompt_mode"],
        "status": "ok",
        "records": processed,
        "prediction_file": relative_to_repo(prediction_file),
        "annotation_file": str(annotation_file),
        "evaluation_type": "automatic_class_agnostic_proposal",
        "instance_metrics": summarize_instance_matches(matches),
        "mask_AP": mask_ap,
        "mask_AP_note": "class_agnostic_AP_all_categories_mapped_to_object",
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary["metrics_file"] = relative_to_repo(output_path)
    write_json(output_path, summary)
    return summary


def evaluate_run(
    config: dict[str, Any],
    task4_config: dict[str, Any],
    dataset: str,
    model: str,
    prompt_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    split = evaluation_split(config, args)
    config = dict(config)
    config["evaluation"] = dict(config.get("evaluation", {}), split=split)
    pred_path = prediction_path(task4_config, dataset, model, prompt_mode)
    out_path = evaluation_output_path(config, dataset, model, prompt_mode)
    if out_path.exists() and not args.rerun_complete:
        print(f"[SKIP] existing metrics {relative_to_repo(out_path)}", flush=True)
        return load_json(out_path)

    dataset_config = task4_config["datasets"][dataset]
    annotation_file = Path(dataset_config["annotation_file"])
    allowed_image_ids, split_file = load_evaluation_image_ids(config, dataset, split)
    metadata = {
        "dataset": dataset,
        "model": model,
        "prompt_mode": prompt_mode,
        "run_name": f"{dataset}/{model}/{prompt_mode}",
        "boundary_tolerance_px": config["metrics"]["boundary_tolerance_px"],
        "score_default": config["metrics"]["score_default"],
        "split": split,
        "split_file": None if split_file is None else relative_to_repo(split_file),
        "split_sha256": sha256_file(split_file),
        "evaluation_images": None if allowed_image_ids is None else len(allowed_image_ids),
    }
    print(f"[START] evaluating {metadata['run_name']} from {relative_to_repo(pred_path)}", flush=True)
    if args.dry_run:
        return {
            "dataset": dataset,
            "model": model,
            "prompt_mode": prompt_mode,
            "status": "dry_run",
            "split": split,
            "evaluation_images": None if allowed_image_ids is None else len(allowed_image_ids),
            "split_sha256": metadata["split_sha256"],
            "prediction_file": relative_to_repo(pred_path),
            "metrics_file": relative_to_repo(out_path),
        }
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    coco = load_json(annotation_file)
    if prompt_mode == "automatic":
        summary = evaluate_automatic(
            pred_path, coco, annotation_file, metadata, args, out_path, allowed_image_ids
        )
    else:
        summary = evaluate_prompted(
            pred_path, coco, annotation_file, metadata, args, out_path, allowed_image_ids
        )
    summary["split"] = split
    summary["split_file"] = metadata["split_file"]
    summary["split_sha256"] = metadata["split_sha256"]
    summary["evaluation_images"] = (
        len(allowed_image_ids) if allowed_image_ids is not None else len(coco.get("images", []))
    )
    write_json(out_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task4_config = load_yaml(config["zero_shot"]["config"])

    dataset_names = selected_names(
        [name for name, item in task4_config["datasets"].items() if item.get("enabled", False)],
        args.datasets,
    )
    model_names = selected_names(config["zero_shot"]["models"], args.models)
    prompt_modes = selected_names(config["zero_shot"]["prompt_modes"], args.prompt_modes)

    rows = []
    for dataset in dataset_names:
        for model in model_names:
            for prompt_mode in prompt_modes:
                summary = evaluate_run(config, task4_config, dataset, model, prompt_mode, args)
                rows.append(compact_row(summary) if not args.dry_run else summary)

    if not args.dry_run:
        split = evaluation_split(config, args)
        output_root = resolve_repo_path(config["task"]["output_root"]) / "zero_shot"
        if split != "full":
            output_root = output_root / split
        write_json(output_root / "summary.json", rows)
        write_summary_csv(output_root / "summary.csv", rows)
        print(f"[DONE] wrote {relative_to_repo(output_root / 'summary.csv')}", flush=True)
    print("[DONE] Task 6 zero-shot evaluation", flush=True)


if __name__ == "__main__":
    main()
