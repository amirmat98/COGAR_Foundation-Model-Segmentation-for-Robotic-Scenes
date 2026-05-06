"""Reusable SAM automatic-mask evaluation workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogar_seg.config import load_config
from cogar_seg.io import load_binary_mask, load_rgb_image, save_binary_mask
from cogar_seg.metrics import compute_iou
from cogar_seg.models.sam import (
    DeviceMode,
    load_sam_automatic_mask_generator,
    select_device,
)
from cogar_seg.paths import remap_ocid_path, resolve_project_path


AUTO_REQUIRED_COLUMNS = [
    "image_path",
    "binary_mask_path",
    "object_id",
]


@dataclass(frozen=True)
class BatchAutomaticMaskRun:
    """Outputs from a batch automatic-mask evaluation run."""

    config_path: Path
    index_path: Path
    project_root: Path
    ocid_root: Path
    checkpoint_path: Path
    output_dir: Path
    masks_dir: Path
    results_csv_path: Path
    results: pd.DataFrame
    device: str
    model_type: str


ProgressCallback = Callable[[dict[str, Any], int, int], None]


def validate_auto_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if automatic-mask evaluation columns are missing."""
    missing = [col for col in AUTO_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns for automatic masks: {missing}")


def select_best_mask_for_gt(
    generated_masks: list[dict[str, Any]],
    gt_mask: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Select the generated mask with the highest IoU against one GT mask."""
    if not generated_masks:
        return np.zeros(gt_mask.shape, dtype=bool), 0.0, {}

    best_mask = None
    best_iou = -1.0
    best_record: dict[str, Any] = {}

    for mask_record in generated_masks:
        candidate = np.asarray(mask_record["segmentation"], dtype=bool)
        candidate_iou = compute_iou(candidate, gt_mask)
        if candidate_iou > best_iou:
            best_iou = candidate_iou
            best_mask = candidate
            best_record = mask_record

    if best_mask is None:
        return np.zeros(gt_mask.shape, dtype=bool), 0.0, {}

    return best_mask, float(best_iou), best_record


def run_batch_sam_automatic_masks(
    config: str | Path,
    index: str | Path,
    checkpoint: str | Path,
    model_type: str,
    device: DeviceMode,
    allow_cpu_fallback: bool,
    output_dir: str | Path | None = None,
    results_csv: str | Path | None = None,
    max_rows: int | None = None,
    start_row: int = 0,
    split: str = "all",
    project_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchAutomaticMaskRun:
    """Run SAM automatic mask generation and score best matching masks."""
    root = Path.cwd() if project_root is None else Path(project_root)
    config_path = resolve_project_path(config, root)
    index_path = resolve_project_path(index, root)
    checkpoint_path = resolve_project_path(checkpoint, root)
    cfg = load_config(config_path)
    ocid_root = Path(cfg["ocid_root"])

    if output_dir is None:
        resolved_output_dir = resolve_project_path("outputs/sam_auto_masks", root)
    else:
        resolved_output_dir = resolve_project_path(output_dir, root)

    masks_dir = resolved_output_dir / "masks"
    if results_csv is None:
        results_csv_path = resolved_output_dir / "sam_auto_mask_results.csv"
    else:
        results_csv_path = resolve_project_path(results_csv, root)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint_path}")

    df = pd.read_csv(index_path)
    validate_auto_required_columns(df)

    process_df = df
    if split != "all":
        if "split" not in process_df.columns:
            raise ValueError("--split requires an index CSV with a 'split' column")
        process_df = process_df[process_df["split"].astype(str) == split]

    if process_df.empty:
        raise ValueError("No rows matched the requested batch selection")

    if start_row < 0 or start_row >= len(process_df):
        raise IndexError(
            f"start-row {start_row} is outside valid range 0 to {len(process_df) - 1}"
        )

    process_df = process_df.iloc[start_row:].copy()
    if max_rows is not None:
        process_df = process_df.head(max_rows)

    selected_device = select_device(
        requested_device=device,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    generator = load_sam_automatic_mask_generator(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=selected_device,
    )

    records: list[dict[str, Any]] = []
    current_image_path: Path | None = None
    current_masks: list[dict[str, Any]] | None = None
    total_rows = len(process_df)

    for counter, (row_index, row) in enumerate(process_df.iterrows(), start=1):
        image_path = remap_ocid_path(row["image_path"], ocid_root)
        gt_mask_path = resolve_project_path(row["binary_mask_path"], root)

        if current_image_path != image_path:
            image_rgb = load_rgb_image(image_path)
            current_masks = generator.generate(image_rgb)
            current_image_path = image_path

        gt_mask = load_binary_mask(gt_mask_path)
        best_mask, iou, best_record = select_best_mask_for_gt(current_masks or [], gt_mask)

        object_id = int(row["object_id"])
        mask_output_path = (
            masks_dir / f"row_{row_index:04d}_object_{object_id}_sam_auto_mask.png"
        )
        save_binary_mask(best_mask, mask_output_path)

        result = {
            "row_index": int(row_index),
            "file_name": row.get("file_name", ""),
            "object_id": object_id,
            "image_path": str(image_path),
            "gt_mask_path": str(gt_mask_path),
            "sam_mask_path": str(mask_output_path),
            "sam_score": float(best_record.get("predicted_iou", 0.0)),
            "stability_score": float(best_record.get("stability_score", 0.0)),
            "generated_mask_count": len(current_masks or []),
            "iou": iou,
            "device": selected_device,
            "model_type": model_type,
        }
        records.append(result)

        if progress_callback is not None:
            progress_callback(result, counter, total_rows)

    results_df = pd.DataFrame(records)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv_path, index=False)

    return BatchAutomaticMaskRun(
        config_path=config_path,
        index_path=index_path,
        project_root=root,
        ocid_root=ocid_root,
        checkpoint_path=checkpoint_path,
        output_dir=resolved_output_dir,
        masks_dir=masks_dir,
        results_csv_path=results_csv_path,
        results=results_df,
        device=selected_device,
        model_type=model_type,
    )
