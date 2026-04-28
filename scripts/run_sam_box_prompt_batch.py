import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from segment_anything import SamPredictor, sam_model_registry


DeviceMode = Literal["auto", "cpu", "cuda"]


@dataclass
class BatchConfig:
    project_root: Path
    config_path: Path
    index_path: Path
    checkpoint_path: Path
    output_dir: Path
    masks_dir: Path
    visualizations_dir: Path
    results_csv_path: Path
    ocid_root: Path


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def remap_ocid_path(path_value: str | Path, ocid_root: Path) -> Path:
    """
    Remap old absolute OCID paths stored in the CSV to the current OCID root.

    Example old CSV path:
        /mnt/Info/COGAR_DATASETs/OCID/OCID-dataset/YCB10/...

    Current root from configs/paths.yaml:
        /mnt/Info/COGAR_DATASETs/OCID-dataset
    """
    path = Path(path_value)

    if path.exists():
        return path

    parts = path.parts

    if "OCID-dataset" not in parts:
        return path

    ocid_idx = parts.index("OCID-dataset")
    relative_inside_ocid = Path(*parts[ocid_idx + 1:])

    return ocid_root / relative_inside_ocid


def select_device(requested_device: DeviceMode, allow_cpu_fallback: bool) -> str:
    if requested_device == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        if requested_device == "cuda" and not allow_cpu_fallback:
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        print("CUDA is not available. Falling back to CPU.")
        return "cpu"

    try:
        test_tensor = torch.empty(1, device="cuda")
        test_tensor += 1
        torch.cuda.synchronize()
        return "cuda"

    except Exception as exc:
        message = (
            "CUDA is visible, but a real CUDA tensor test failed.\n"
            "This usually means your PyTorch CUDA build is not compatible with your GPU.\n"
            f"Original CUDA error: {exc}"
        )

        if requested_device == "cuda" and not allow_cpu_fallback:
            raise RuntimeError(message) from exc

        print(message)
        print("Falling back to CPU.")
        return "cpu"


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "image_path",
        "binary_mask_path",
        "file_name",
        "object_id",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def load_rgb_image(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"OpenCV could not read GT mask: {mask_path}")

    return mask > 0


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255)


def save_visualization(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    sam_mask: np.ndarray,
    box: np.ndarray,
    output_path: Path,
    iou: float,
    sam_score: float,
    row_index: int,
    object_id: int,
) -> None:
    x_min, y_min, x_max, y_max = box.astype(int)

    overlay = image_rgb.copy()
    green_mask = np.zeros_like(image_rgb)
    green_mask[:, :, 1] = 255

    overlay = np.where(
        sam_mask[:, :, None],
        (0.6 * overlay + 0.4 * green_mask).astype(np.uint8),
        overlay,
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"RGB + box | row={row_index}, obj={object_id}")
    axes[0].add_patch(
        plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(sam_mask, cmap="gray")
    axes[2].set_title("SAM predicted mask")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay | IoU={iou:.3f}, score={sam_score:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_batch_config(args: argparse.Namespace) -> BatchConfig:
    project_root = Path.cwd()

    config_path = resolve_project_path(args.config, project_root)
    cfg = load_config(config_path)

    index_path = resolve_project_path(args.index, project_root)
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)

    if args.output_dir is None:
        output_dir = resolve_project_path(cfg["sam_outputs_dir"], project_root)
    else:
        output_dir = resolve_project_path(args.output_dir, project_root)

    masks_dir = output_dir / "masks"
    visualizations_dir = output_dir / "visualizations"

    if args.results_csv is None:
        results_csv_path = project_root / "outputs/indexes/ocid_debug_seq21_sam_box_results.csv"
    else:
        results_csv_path = resolve_project_path(args.results_csv, project_root)

    ocid_root = Path(cfg["ocid_root"])

    return BatchConfig(
        project_root=project_root,
        config_path=config_path,
        index_path=index_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        masks_dir=masks_dir,
        visualizations_dir=visualizations_dir,
        results_csv_path=results_csv_path,
        ocid_root=ocid_root,
    )


def validate_batch_config(batch_cfg: BatchConfig) -> None:
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


def load_sam_predictor(checkpoint_path: Path, model_type: str, device: str) -> SamPredictor:
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()

    return SamPredictor(sam)


def make_box_from_row(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        ],
        dtype=np.float32,
    )


def run_sam_for_box(
    predictor: SamPredictor,
    box_xyxy: np.ndarray,
) -> tuple[np.ndarray, float]:
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            box=box_xyxy,
            multimask_output=False,
        )

    return masks[0], float(scores[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM box-prompt inference over all filtered OCID objects."
    )

    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Path to project paths YAML file.",
    )

    parser.add_argument(
        "--index",
        default="outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv",
        help="Path to final object-level CSV.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint.",
    )

    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )

    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA fails, continue on CPU instead of stopping.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, uses sam_outputs_dir from configs/paths.yaml.",
    )

    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to output results CSV.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process for debugging.",
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start processing from this row index.",
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable saving visualization PNGs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    batch_cfg = build_batch_config(args)
    validate_batch_config(batch_cfg)

    device = select_device(
        requested_device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )

    print("Using config:", batch_cfg.config_path)
    print("Project root:", batch_cfg.project_root)
    print("OCID root:", batch_cfg.ocid_root)
    print("Index path:", batch_cfg.index_path)
    print("Checkpoint path:", batch_cfg.checkpoint_path)
    print("Output dir:", batch_cfg.output_dir)
    print("Results CSV:", batch_cfg.results_csv_path)
    print("Device:", device)
    print("Model type:", args.model_type)

    df = pd.read_csv(batch_cfg.index_path)
    validate_required_columns(df)

    if args.start_row < 0 or args.start_row >= len(df):
        raise IndexError(f"start-row {args.start_row} is outside valid range 0 to {len(df) - 1}")

    process_df = df.iloc[args.start_row:].copy()

    if args.max_rows is not None:
        process_df = process_df.head(args.max_rows)

    print()
    print(f"Total rows in index: {len(df)}")
    print(f"Rows to process: {len(process_df)}")
    print()

    predictor = load_sam_predictor(
        checkpoint_path=batch_cfg.checkpoint_path,
        model_type=args.model_type,
        device=device,
    )

    results = []

    current_image_path = None
    current_image_rgb = None

    for counter, (row_index, row) in enumerate(process_df.iterrows(), start=1):
        image_path = remap_ocid_path(row["image_path"], batch_cfg.ocid_root)
        gt_mask_path = resolve_project_path(row["binary_mask_path"], batch_cfg.project_root)

        if not image_path.exists():
            raise FileNotFoundError(f"Resolved image path does not exist: {image_path}")

        if not gt_mask_path.exists():
            raise FileNotFoundError(f"GT mask path does not exist: {gt_mask_path}")

        object_id = int(row["object_id"])
        box_xyxy = make_box_from_row(row)

        if current_image_path != image_path:
            current_image_rgb = load_rgb_image(image_path)
            predictor.set_image(current_image_rgb)
            current_image_path = image_path

        gt_mask = load_binary_mask(gt_mask_path)

        sam_mask, sam_score = run_sam_for_box(
            predictor=predictor,
            box_xyxy=box_xyxy,
        )

        iou = compute_iou(sam_mask, gt_mask)

        mask_output_path = (
            batch_cfg.masks_dir
            / f"row_{row_index:04d}_object_{object_id}_sam_mask.png"
        )

        save_mask(sam_mask, mask_output_path)

        if args.no_visualizations:
            vis_output_path = ""
        else:
            vis_output_path = (
                batch_cfg.visualizations_dir
                / f"row_{row_index:04d}_object_{object_id}_sam_visualization.png"
            )

            save_visualization(
                image_rgb=current_image_rgb,
                gt_mask=gt_mask,
                sam_mask=sam_mask,
                box=box_xyxy,
                output_path=vis_output_path,
                iou=iou,
                sam_score=sam_score,
                row_index=row_index,
                object_id=object_id,
            )

        results.append(
            {
                "row_index": row_index,
                "file_name": row["file_name"],
                "object_id": object_id,
                "image_path": str(image_path),
                "gt_mask_path": str(gt_mask_path),
                "bbox_xmin": float(row["bbox_xmin"]),
                "bbox_ymin": float(row["bbox_ymin"]),
                "bbox_xmax": float(row["bbox_xmax"]),
                "bbox_ymax": float(row["bbox_ymax"]),
                "sam_mask_path": str(mask_output_path),
                "sam_visualization_path": str(vis_output_path),
                "sam_score": sam_score,
                "iou": iou,
                "device": device,
                "model_type": args.model_type,
            }
        )

        print(
            f"[{counter:03d}/{len(process_df):03d}] "
            f"row={row_index:04d} "
            f"obj={object_id} "
            f"score={sam_score:.4f} "
            f"IoU={iou:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(batch_cfg.results_csv_path, index=False)

    print()
    print("Done.")
    print("Saved results CSV:", batch_cfg.results_csv_path)
    print("Saved masks dir:", batch_cfg.masks_dir)

    if not args.no_visualizations:
        print("Saved visualizations dir:", batch_cfg.visualizations_dir)

    print()
    print("Summary:")
    print(f"Number of evaluated objects: {len(results_df)}")
    print(f"Mean IoU: {results_df['iou'].mean():.4f}")
    print(f"Median IoU: {results_df['iou'].median():.4f}")
    print(f"Min IoU: {results_df['iou'].min():.4f}")
    print(f"Max IoU: {results_df['iou'].max():.4f}")
    print(f"Mean SAM score: {results_df['sam_score'].mean():.4f}")


if __name__ == "__main__":
    main()