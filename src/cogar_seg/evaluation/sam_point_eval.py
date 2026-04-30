"""Reusable SAM point-prompt evaluation workflows."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cogar_seg.config import load_config
from cogar_seg.io import load_binary_mask, load_rgb_image, save_binary_mask
from cogar_seg.metrics import compute_iou
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

    sam_mask, sam_score = run_sam_for_point(
        predictor=predictor,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    iou = compute_iou(sam_mask, gt_mask)

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
    mean_sam_score: float
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
    max_rows: int | None = None,
    project_root: str | Path | None = None,
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
    results_csv_path = resolved_output_dir / "sam_point_prompt_results.csv"

    df = pd.read_csv(index_path)
    validate_point_required_columns(df)

    if max_rows is not None:
        df_to_run = df.head(max_rows)
    else:
        df_to_run = df

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

    for count, (row_index, row) in enumerate(df_to_run.iterrows(), start=1):
        object_id = int(row["object_id"])

        image_path = remap_ocid_path(row["image_path"], ocid_root)
        gt_mask_path = resolve_project_path(row["binary_mask_path"], root)

        if not image_path.exists():
            raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask path does not exist: {gt_mask_path}")

        image_rgb = load_rgb_image(image_path)
        gt_mask = load_binary_mask(gt_mask_path)

        point_coords, point_labels = make_positive_point_prompt(row)

        predictor.set_image(image_rgb)

        sam_mask, sam_score = run_sam_for_point(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        iou = compute_iou(sam_mask, gt_mask)

        mask_output_path = (
            mask_dir / f"row_{row_index:04d}_object_{object_id}_sam_point_mask.png"
        )
        vis_output_path = (
            vis_dir / f"row_{row_index:04d}_object_{object_id}_sam_point_visualization.png"
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
            row_index=int(row_index),
            object_id=object_id,
        )

        records.append(
            {
                "row_index": int(row_index),
                "object_id": object_id,
                "image_path": str(image_path),
                "gt_mask_path": str(gt_mask_path),
                "point_x": float(point_coords[0, 0]),
                "point_y": float(point_coords[0, 1]),
                "sam_score": float(sam_score),
                "iou": float(iou),
                "mask_output_path": str(mask_output_path),
                "visualization_output_path": str(vis_output_path),
                "device": selected_device,
                "model_type": model_type,
            }
        )

        print(
            f"[{count:03d}/{total:03d}] "
            f"row={row_index}, obj={object_id}, "
            f"score={sam_score:.4f}, IoU={iou:.4f}"
        )

    results_df = pd.DataFrame(records)
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_csv_path, index=False)

    mean_iou = float(results_df["iou"].mean())
    median_iou = float(results_df["iou"].median())
    mean_sam_score = float(results_df["sam_score"].mean())

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
        mean_sam_score=mean_sam_score,
        device=selected_device,
        model_type=model_type,
    )
