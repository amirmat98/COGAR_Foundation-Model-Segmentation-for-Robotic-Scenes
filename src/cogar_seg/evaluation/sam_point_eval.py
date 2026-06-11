"""Reusable SAM point-prompt evaluation workflows."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import pandas as pd

from cogar_seg.config import load_config
from cogar_seg.io import load_binary_mask, load_rgb_image, save_binary_mask
from cogar_seg.metrics import compute_boundary_f1, compute_iou
from cogar_seg.models.sam import (
    DeviceMode,
    load_sam_predictor,
    run_sam_for_point,
    select_device,
)
from cogar_seg.paths import remap_ocid_path, resolve_project_path
from cogar_seg.prompts import make_positive_point_prompt
from cogar_seg.visualization import save_sam_point_visualization


POINT_REQUIRED_COLUMNS = [
    "image_path",
    "binary_mask_path",
    "object_id",
    "point_x",
    "point_y",
]

ProgressCallback = Callable[[dict[str, Any], int, int], None]


@dataclass(frozen=True)
class SingleSamPointResult:
    """Outputs and metrics from one SAM point-prompt run."""

    config_path: Path
    index_path: Path
    project_root: Path
    ocid_root: Path
    image_path: Path
    gt_mask_path: Path
    checkpoint_path: Path
    output_dir: Path
    row_index: int
    object_id: int
    original_image_path: str
    point_coords: list[list[float]]
    point_labels: list[int]
    mask_output_path: Path
    visualization_output_path: Path
    sam_score: float
    iou: float
    boundary_f1: float
    latency_sec: float
    device: str
    model_type: str


def validate_point_required_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if the object-index CSV misses point-prompt columns."""
    missing = [col for col in POINT_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns for point prompt: {missing}")


def run_single_sam_point(
    config: str | Path,
    index: str | Path,
    row_index: int,
    checkpoint: str | Path,
    model_type: str,
    device: DeviceMode,
    allow_cpu_fallback: bool,
    output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> SingleSamPointResult:
    """Run SAM point-prompt inference for one object-index row."""
    root = Path.cwd() if project_root is None else Path(project_root)

    config_path = resolve_project_path(config, root)
    index_path = resolve_project_path(index, root)
    checkpoint_path = resolve_project_path(checkpoint, root)

    cfg = load_config(config_path)
    ocid_root = Path(cfg["ocid_root"])

    if output_dir is None:
        resolved_output_dir = resolve_project_path("outputs/sam_point_prompt", root)
    else:
        resolved_output_dir = resolve_project_path(output_dir, root)

    df = pd.read_csv(index_path)
    validate_point_required_columns(df)

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row {row_index} is outside valid range 0 to {len(df) - 1}")

    row = df.iloc[row_index]

    image_path = remap_ocid_path(row["image_path"], ocid_root)
    gt_mask_path = resolve_project_path(row["binary_mask_path"], root)

    if not image_path.exists():
        raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")

    if not gt_mask_path.exists():
        raise FileNotFoundError(f"GT mask path does not exist: {gt_mask_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint_path}")

    selected_device = select_device(
        requested_device=device,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    image_rgb = load_rgb_image(image_path)
    gt_mask = load_binary_mask(gt_mask_path)

    point_coords, point_labels = make_positive_point_prompt(row)

    predictor = load_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=selected_device,
    )

    predictor.set_image(image_rgb)

    t_start = time.perf_counter()
    sam_mask, sam_score = run_sam_for_point(
        predictor=predictor,
        point_coords=point_coords,
        point_labels=point_labels,
    )
    if selected_device == "cuda":
        import torch
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    latency = t_end - t_start

    iou = compute_iou(sam_mask, gt_mask)
    boundary_f1 = compute_boundary_f1(sam_mask, gt_mask)

    object_id = int(row["object_id"])

    mask_output_path = (
        resolved_output_dir / f"row_{row_index:04d}_object_{object_id}_sam_point_mask.png"
    )

    vis_output_path = (
        resolved_output_dir
        / f"row_{row_index:04d}_object_{object_id}_sam_point_visualization.png"
    )

    save_binary_mask(sam_mask, mask_output_path)

    save_sam_point_visualization(
        image_rgb=image_rgb,
        gt_mask=gt_mask,
        pred_mask=sam_mask,
        point_coords=point_coords,
        output_path=vis_output_path,
        iou=iou,
        model_score=sam_score,
        row_index=row_index,
        object_id=object_id,
    )

    return SingleSamPointResult(
        config_path=config_path,
        index_path=index_path,
        project_root=root,
        ocid_root=ocid_root,
        image_path=image_path,
        gt_mask_path=gt_mask_path,
        checkpoint_path=checkpoint_path,
        output_dir=resolved_output_dir,
        row_index=row_index,
        object_id=object_id,
        original_image_path=str(row["image_path"]),
        point_coords=point_coords.astype(float).tolist(),
        point_labels=point_labels.astype(int).tolist(),
        mask_output_path=mask_output_path,
        visualization_output_path=vis_output_path,
        sam_score=sam_score,
        iou=iou,
        boundary_f1=boundary_f1,
        latency_sec=latency,
        device=selected_device,
        model_type=model_type,
    )


@dataclass(frozen=True)
class BatchSamPointRun:
    """Summary from a batch SAM point-prompt evaluation run."""

    config_path: Path
    index_path: Path
    project_root: Path
    ocid_root: Path
    checkpoint_path: Path
    output_dir: Path
    results_csv_path: Path
    num_rows: int
    mean_iou: float
    median_iou: float
    mean_boundary_f1: float
    mean_sam_score: float
    mean_latency_sec: float
    mean_fps: float
    results: pd.DataFrame
    device: str
    model_type: str


def run_batch_sam_point(
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
    save_visualizations: bool = True,
    save_masks: bool = True,
    project_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchSamPointRun:
    """Run SAM point-prompt inference for many object-index rows."""
    root = Path.cwd() if project_root is None else Path(project_root)

    config_path = resolve_project_path(config, root)
    index_path = resolve_project_path(index, root)
    checkpoint_path = resolve_project_path(checkpoint, root)

    cfg = load_config(config_path)
    ocid_root = Path(cfg["ocid_root"])

    if output_dir is None:
        resolved_output_dir = resolve_project_path("outputs/sam_point_prompt_batch", root)
    else:
        resolved_output_dir = resolve_project_path(output_dir, root)

    mask_dir = resolved_output_dir / "masks"
    vis_dir = resolved_output_dir / "visualizations"
    if results_csv is None:
        results_csv_path = resolved_output_dir / "sam_point_prompt_results.csv"
    else:
        results_csv_path = resolve_project_path(results_csv, root)

    df = pd.read_csv(index_path)
    validate_point_required_columns(df)

    df_to_run = df
    if split != "all":
        if "split" not in df_to_run.columns:
            raise ValueError("--split requires an index CSV with a 'split' column")
        df_to_run = df_to_run[df_to_run["split"].astype(str) == split]

    if df_to_run.empty:
        raise ValueError("No rows matched the requested batch selection")

    if start_row < 0 or start_row >= len(df_to_run):
        raise IndexError(
            f"start-row {start_row} is outside valid range 0 to {len(df_to_run) - 1}"
        )

    df_to_run = df_to_run.iloc[start_row:].copy()

    if max_rows is not None:
        df_to_run = df_to_run.head(max_rows)

    selected_device = select_device(
        requested_device=device,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {checkpoint_path}")

    predictor = load_sam_predictor(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=selected_device,
    )

    records: list[dict[str, Any]] = []

    total = len(df_to_run)

    current_image_path: Path | None = None
    current_image_rgb = None

    for count, (row_index, row) in enumerate(df_to_run.iterrows(), start=1):
        object_id = int(row["object_id"])

        image_path = remap_ocid_path(row["image_path"], ocid_root)
        gt_mask_path = resolve_project_path(row["binary_mask_path"], root)

        if not image_path.exists():
            raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask path does not exist: {gt_mask_path}")

        if current_image_path != image_path:
            current_image_rgb = load_rgb_image(image_path)
            predictor.set_image(current_image_rgb)
            current_image_path = image_path

        gt_mask = load_binary_mask(gt_mask_path)

        point_coords, point_labels = make_positive_point_prompt(row)

        t_start = time.perf_counter()
        sam_mask, sam_score = run_sam_for_point(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
        )
        if selected_device == "cuda":
            import torch
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        latency = t_end - t_start
        fps = 1.0 / latency if latency > 0 else 0.0

        iou = compute_iou(sam_mask, gt_mask)
        boundary_f1 = compute_boundary_f1(sam_mask, gt_mask)

        if save_masks:
            mask_output_path = (
                mask_dir / f"row_{row_index:04d}_object_{object_id}_sam_point_mask.png"
            )
            save_binary_mask(sam_mask, mask_output_path)
            mask_output_path_str = str(mask_output_path)
        else:
            mask_output_path_str = ""

        if save_visualizations:
            vis_output_path = (
                vis_dir
                / f"row_{row_index:04d}_object_{object_id}_sam_point_visualization.png"
            )
        else:
            vis_output_path = ""

        if save_visualizations:
            save_sam_point_visualization(
                image_rgb=current_image_rgb,
                gt_mask=gt_mask,
                pred_mask=sam_mask,
                point_coords=point_coords,
                output_path=vis_output_path,
                iou=iou,
                model_score=sam_score,
                row_index=int(row_index),
                object_id=object_id,
            )

        record = {
            "row_index": int(row_index),
            "file_name": row.get("file_name", ""),
            "object_id": object_id,
            "image_path": str(image_path),
            "gt_mask_path": str(gt_mask_path),
            "point_x": float(point_coords[0, 0]),
            "point_y": float(point_coords[0, 1]),
            "sam_score": float(sam_score),
            "iou": float(iou),
            "boundary_f1": float(boundary_f1),
            "latency_sec": latency,
            "fps": fps,
            "mask_output_path": mask_output_path_str,
            "visualization_output_path": str(vis_output_path),
            "device": selected_device,
            "model_type": model_type,
        }
        records.append(record)

        if progress_callback is not None:
            progress_callback(record, count, total)

    results_df = pd.DataFrame(records)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv_path, index=False)

    mean_iou = float(results_df["iou"].mean())
    median_iou = float(results_df["iou"].median())
    mean_boundary_f1 = float(results_df["boundary_f1"].mean())
    mean_sam_score = float(results_df["sam_score"].mean())
    mean_latency_sec = float(results_df["latency_sec"].mean())
    mean_fps = float(results_df["fps"].mean())

    return BatchSamPointRun(
        config_path=config_path,
        index_path=index_path,
        project_root=root,
        ocid_root=ocid_root,
        checkpoint_path=checkpoint_path,
        output_dir=resolved_output_dir,
        results_csv_path=results_csv_path,
        num_rows=len(results_df),
        mean_iou=mean_iou,
        median_iou=median_iou,
        mean_boundary_f1=mean_boundary_f1,
        mean_sam_score=mean_sam_score,
        mean_latency_sec=mean_latency_sec,
        mean_fps=mean_fps,
        results=results_df,
        device=selected_device,
        model_type=model_type,
    )
