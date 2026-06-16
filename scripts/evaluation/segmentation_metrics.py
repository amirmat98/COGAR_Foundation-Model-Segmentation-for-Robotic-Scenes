"""Shared segmentation metric helpers for Task 6."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


COCO_STATS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    binary = (mask > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def normalize_rle(rle: dict[str, Any]) -> dict[str, Any]:
    normalized = {"size": [int(v) for v in rle["size"]], "counts": rle["counts"]}
    if isinstance(normalized["counts"], bytes):
        normalized["counts"] = normalized["counts"].decode("ascii")
    return normalized


def annotation_to_rle(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            return normalize_rle(mask_utils.frPyObjects(segmentation, height, width))
        return normalize_rle(segmentation)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        merged = mask_utils.merge(rles)
        return normalize_rle(merged)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)!r}")


def rle_to_mask(rle: dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(bool)


def rle_iou(gt_rle: dict[str, Any], pred_rle: dict[str, Any]) -> float:
    return float(mask_utils.iou([pred_rle], [gt_rle], [0])[0, 0])


def select_prediction(predictions: list[dict[str, Any]], default_score: float) -> dict[str, Any] | None:
    if not predictions:
        return None

    def score(item: dict[str, Any]) -> tuple[float, int]:
        raw_score = item.get("score")
        if raw_score is None:
            raw_score = default_score
        return float(raw_score), -int(item.get("candidate_index", 0))

    return max(predictions, key=score)


def binary_erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    eroded = np.ones(mask.shape, dtype=bool)
    for y_offset in range(3):
        for x_offset in range(3):
            eroded &= padded[y_offset : y_offset + mask.shape[0], x_offset : x_offset + mask.shape[1]]
    return eroded


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.astype(bool)
    binary = mask.astype(bool)
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    dilated = np.zeros(binary.shape, dtype=bool)
    for y_offset in range(-radius, radius + 1):
        for x_offset in range(-radius, radius + 1):
            if y_offset * y_offset + x_offset * x_offset > radius * radius:
                continue
            y_start = radius + y_offset
            x_start = radius + x_offset
            dilated |= padded[y_start : y_start + binary.shape[0], x_start : x_start + binary.shape[1]]
    return dilated


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(binary, dtype=bool)
    try:
        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        return (binary - eroded).astype(bool)
    except ImportError:
        pass
    eroded = binary_erode(binary)
    return np.logical_and(binary.astype(bool), ~eroded)


def boundary_f1(gt_mask: np.ndarray, pred_mask: np.ndarray, tolerance_px: int) -> float:
    gt_boundary = binary_boundary(gt_mask)
    pred_boundary = binary_boundary(pred_mask)
    gt_count = int(gt_boundary.sum())
    pred_count = int(pred_boundary.sum())

    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 or pred_count == 0:
        return 0.0

    try:
        import cv2

        kernel_size = max(1, int(tolerance_px) * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
        pred_dilated = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        gt_dilated = binary_dilate(gt_boundary, int(tolerance_px))
        pred_dilated = binary_dilate(pred_boundary, int(tolerance_px))

    precision = float(np.logical_and(pred_boundary, gt_dilated).sum()) / float(pred_count)
    recall = float(np.logical_and(gt_boundary, pred_dilated).sum()) / float(gt_count)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def category_name_by_id(coco: dict[str, Any]) -> dict[int, str]:
    return {int(category["id"]): str(category["name"]) for category in coco.get("categories", [])}


def image_size_by_id(coco: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {
        int(image["id"]): (int(image["height"]), int(image["width"]))
        for image in coco.get("images", [])
    }


def annotations_by_id(coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(annotation["id"]): annotation for annotation in coco.get("annotations", [])}


def annotations_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        if int(annotation.get("iscrowd", 0)):
            continue
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def summarize_instance_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "instances": 0,
            "mIoU": 0.0,
            "boundary_f1": 0.0,
            "per_category_iou": {},
            "per_category_boundary_f1": {},
        }

    ious = np.asarray([float(item["iou"]) for item in matches], dtype=np.float64)
    bf1 = np.asarray([float(item["boundary_f1"]) for item in matches], dtype=np.float64)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in matches:
        by_category[item["category_name"]].append(item)

    per_category_iou = {}
    per_category_bf1 = {}
    per_category_count = {}
    for category_name, items in sorted(by_category.items()):
        per_category_iou[category_name] = float(np.mean([item["iou"] for item in items]))
        per_category_bf1[category_name] = float(np.mean([item["boundary_f1"] for item in items]))
        per_category_count[category_name] = len(items)

    return {
        "instances": len(matches),
        "mIoU": float(ious.mean()),
        "boundary_f1": float(bf1.mean()),
        "per_category_iou": per_category_iou,
        "per_category_boundary_f1": per_category_bf1,
        "per_category_count": per_category_count,
    }


def coco_stats_to_dict(stats: Any) -> dict[str, float]:
    return {name: float(stats[index]) for index, name in enumerate(COCO_STATS)}


def evaluate_coco_predictions(
    annotation_file: str | Path,
    detections: list[dict[str, Any]],
    iou_type: str = "segm",
    image_ids: list[int] | None = None,
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {name: 0.0 for name in COCO_STATS}

    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if image_ids is not None:
        coco_eval.params.imgIds = sorted(set(int(image_id) for image_id in image_ids))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_stats_to_dict(coco_eval.stats)


def class_agnostic_coco(coco: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(coco)
    converted["categories"] = [{"id": 1, "name": "object"}]
    for annotation in converted.get("annotations", []):
        annotation["category_id"] = 1
    return converted


def write_temp_coco(path: str | Path, coco: dict[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(coco), encoding="utf-8")
    return resolved


def semantic_confusion(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def semantic_metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    true_positive = np.diag(confusion).astype(np.float64)
    row_sum = confusion.sum(axis=1).astype(np.float64)
    col_sum = confusion.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=union > 0,
    )
    accuracy = np.divide(
        true_positive,
        row_sum,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )
    valid_iou = iou[~np.isnan(iou)]
    foreground_iou = iou[1:][~np.isnan(iou[1:])]
    valid_accuracy = accuracy[~np.isnan(accuracy)]

    return {
        "mIoU": float(valid_iou.mean()) if valid_iou.size else 0.0,
        "foreground_mIoU": float(foreground_iou.mean()) if foreground_iou.size else 0.0,
        "pixel_accuracy": float(true_positive.sum() / max(confusion.sum(), 1.0)),
        "mean_accuracy": float(valid_accuracy.mean()) if valid_accuracy.size else 0.0,
        "per_category_iou": {
            class_names[index] if index < len(class_names) else f"class_{index}": None
            if np.isnan(value)
            else float(value)
            for index, value in enumerate(iou)
        },
    }


def semantic_boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: list[str],
    tolerance_px: int,
) -> dict[str, Any]:
    values = []
    per_category = {}
    for class_id, class_name in enumerate(class_names):
        gt_mask = target == class_id
        pred_mask = prediction == class_id
        if not gt_mask.any() and not pred_mask.any():
            continue
        score = boundary_f1(gt_mask, pred_mask, tolerance_px)
        per_category[class_name] = score
        values.append(score)
    foreground_values = [
        value for index, value in enumerate(per_category.values()) if index > 0
    ]
    return {
        "boundary_f1": float(np.mean(values)) if values else 0.0,
        "foreground_boundary_f1": float(np.mean(foreground_values)) if foreground_values else 0.0,
        "per_category_boundary_f1": per_category,
    }


def load_semantic_png(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int64)
