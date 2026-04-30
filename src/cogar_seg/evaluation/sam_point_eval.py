"""Reusable SAM point-prompt evaluation workflows."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cogar_seg.config import load_config
from cogar_seg.io import load_binary_mask, load_rgb_image, save_binary_mask
from cogar_seg.metrics import compute_iou
from cogar_seg.models.sam import DeviceMode, load_sam_predictor, run_sam_for_point, select_device
from cogar_seg.paths import default_results_csv, remap_ocid_path, resolve_project_path
from cogar_seg.prompts.points import make_positive_point_prompt
from cogar_seg.visualization import save_sam_point_visualization


BATCH_REQUIRED_COLUMNS = [
    "image_path",
    "binary_mask_path",
    "file_name",
    "object_id",
    "point_x",
    "point_y",
]


@dataclass(frozen=True)
class BatchSamPointConfig:
    """Resolved paths and settings for a batch SAM point-prompt run."""

    project_root: Path
    config_path: Path
    index_path: Path
    checkpoint_path: Path
    output_dir: Path
    masks_dir: Path
    visualizations_dir: Path
    results_csv_path: Path
    ocid_root: Path


@dataclass(frozen=True)
class BatchSamPointRun:
    """Outputs from a batch SAM point-prompt run."""

    config: BatchSamPointConfig
    results: pd.DataFrame
    device: str
    model_type: str


ProgressCallback = Callable[[dict[str, Any], int, int], None]


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if an object-index CSV misses required columns."""
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def build_batch_config(
    config: str | Path,
    index: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path | None,
    results_csv: str | Path | None,
    project_root: str | Path | None = None,
) -> BatchSamPointConfig:
    """Resolve all paths for a batch SAM point-prompt run."""
    root = Path.cwd() if project_root is None else Path(project_root)
    config_path = resolve_project_path(config, root)
    cfg = load_config(config_path)

    index_path = resolve_project_path(index, root)
    checkpoint_path = resolve_project_path(checkpoint, root)

    if output_dir is None:
        resolved_output_dir = resolve_project_path("outputs/sam_point_prompt", root)
    else:
        resolved_output_dir = resolve_project_path(output_dir, root)

    if results_csv is None:
        results_csv_path = default_results_csv(root, "ocid_debug_seq21_sam_point_results.csv")
    else:
        results_csv_path = resolve_project_path(results_csv, root)

    return BatchSamPointConfig(
        project_root=root,
        config_path=config_path,
        index_path=index_path,
        checkpoint_path=checkpoint_path,
        output_dir=resolved_output_dir,
        masks_dir=resolved_output_dir / "masks",
        visualizations_dir=resolved_output_dir / "visualizations",
        results_csv_path=results_csv_path,
        ocid_root=Path(cfg["ocid_root"]),
    )


def validate_batch_config(batch_cfg: BatchSamPointConfig) -> None:
    """Validate input paths and create output directories for batch evaluation."""
    if not batch_cfg.index_path.exists():
        raise FileNotFoundError(f"Index CSV does not exist: {batch_cfg.index_path}")

    if not batch_cfg.checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {batch_cfg.checkpoint_path}")

    if not batch_cfg.ocid_root.exists():
        raise FileNotFoundError(f"OCID root does not exist: {batch_cfg.ocid_root}")

    batch_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    batch_cfg.masks_dir.mkdir(parents=True, exist_ok=True)
    batch_cfg.visualizations_dir.mkdir(parents=True, exist_ok=True)
    batch_cfg.results_csv_path.parent.mkdir(parents=True, exist_ok=True)


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
    save_visualizations: bool = True,
    project_root: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> BatchSamPointRun:
    """Run SAM point-prompt inference over an object-index CSV."""
    batch_cfg = build_batch_config(
        config=config,
        index=index,
        checkpoint=checkpoint,
        output_dir=output_dir,
        results_csv=results_csv,
        project_root=project_root,
    )
    validate_batch_config(batch_cfg)

    selected_device = select_device(
        requested_device=device,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    df = pd.read_csv(batch_cfg.index_path)
    validate_required_columns(df, BATCH_REQUIRED_COLUMNS)

    if start_row < 0 or start_row >= len(df):
        raise IndexError(f"start-row {start_row} is outside valid range 0 to {len(df) - 1}")

    process_df = df.iloc[start_row:].copy()

    if max_rows is not None:
        process_df = process_df.head(max_rows)

    predictor = load_sam_predictor(
        checkpoint_path=batch_cfg.checkpoint_path,
        model_type=model_type,
        device=selected_device,
    )

    results = []
    current_image_path: Path | None = None
    current_image_rgb = None
    total_rows = len(process_df)

    for counter, (row_idx, row) in enumerate(process_df.iterrows(), start=1):
        image_path = remap_ocid_path(row["image_path"], batch_cfg.ocid_root)
        gt_mask_path = resolve_project_path(row["binary_mask_path"], batch_cfg.project_root)

        if not image_path.exists():
            raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask path does not exist: {gt_mask_path}")

        object_id = int(row["object_id"])
        point_coords, point_labels = make_positive_point_prompt(row)

        if current_image_path != image_path:
            current_image_rgb = load_rgb_image(image_path)
            predictor.set_image(current_image_rgb)
            current_image_path = image_path

        gt_mask = load_binary_mask(gt_mask_path)
        sam_mask, sam_score = run_sam_for_point(
            predictor=predictor,
            point_coords=point_coords,
            point_labels=point_labels,
        )
        iou = compute_iou(sam_mask, gt_mask)

        mask_output_path = (
            batch_cfg.masks_dir
            / f"row_{row_idx:04d}_object_{object_id}_sam_point_mask.png"
        )
        save_binary_mask(sam_mask, mask_output_path)

        if save_visualizations:
            vis_output_path = (
                batch_cfg.visualizations_dir
                / f"row_{row_idx:04d}_object_{object_id}_sam_point_visualization.png"
            )
            save_sam_point_visualization(
                image_rgb=current_image_rgb,
                gt_mask=gt_mask,
                pred_mask=sam_mask,
                point_coords=point_coords,
                output_path=vis_output_path,
                iou=iou,
                model_score=sam_score,
                row_index=int(row_idx),
                object_id=object_id,
            )
        else:
            vis_output_path = ""

        result = {
            "row_index": row_idx,
            "file_name": row["file_name"],
            "object_id": object_id,
            "image_path": str(image_path),
            "gt_mask_path": str(gt_mask_path),
            "point_x": float(row["point_x"]),
            "point_y": float(row["point_y"]),
            "sam_mask_path": str(mask_output_path),
            "sam_visualization_path": str(vis_output_path),
            "sam_score": sam_score,
            "iou": iou,
            "device": selected_device,
            "model_type": model_type,
            "prompt_type": "point",
        }
        results.append(result)

        if progress_callback is not None:
            progress_callback(result, counter, total_rows)

    results_df = pd.DataFrame(results)
    results_df.to_csv(batch_cfg.results_csv_path, index=False)

    return BatchSamPointRun(
        config=batch_cfg,
        results=results_df,
        device=selected_device,
        model_type=model_type,
    )