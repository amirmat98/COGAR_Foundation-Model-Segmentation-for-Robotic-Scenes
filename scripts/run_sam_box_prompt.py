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
class SamplePaths:
    image_path: Path
    gt_mask_path: Path
    checkpoint_path: Path
    output_dir: Path


def load_config(config_path: Path) -> dict:
    """Load project paths from configs/paths.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_project_path(path_value: str | Path, project_root: Path) -> Path:
    """Resolve repo-relative paths such as outputs/... or checkpoints/..."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def remap_ocid_path(path_value: str | Path, ocid_root: Path) -> Path:
    """
    Remap an old absolute OCID path from the CSV to the current OCID root.

    Old CSV example:
        /mnt/Info/COGAR_DATASETs/OCID/OCID-dataset/YCB10/...

    Current root from configs/paths.yaml:
        /mnt/Info/COGAR_DATASETs/OCID-dataset

    The function keeps everything after 'OCID-dataset/' and attaches it
    to ocid_root.
    """
    path = Path(path_value)

    if path.exists():
        return path

    parts = path.parts

    if "OCID-dataset" not in parts:
        return path

    ocid_idx = parts.index("OCID-dataset")
    relative_inside_ocid = Path(*parts[ocid_idx + 1 :])

    return ocid_root / relative_inside_ocid


def select_device(requested_device: DeviceMode, allow_cpu_fallback: bool) -> str:
    """
    Select CPU or CUDA.

    This function does a real CUDA tensor test, not only torch.cuda.is_available().
    That is important because your GTX 1050 can be detected by PyTorch while still
    being incompatible with the installed PyTorch CUDA build.
    """
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
    """Check that the CSV contains all columns needed by Block 3."""
    required_columns = [
        "image_path",
        "binary_mask_path",
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
    """Load RGB image as HWC uint8."""
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load GT binary mask as boolean array."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"OpenCV could not read GT mask: {mask_path}")

    return mask > 0


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """Save a boolean mask as a binary PNG."""
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
) -> None:
    """
    Save a 4-panel visualization:
    1. RGB image with box prompt
    2. Ground-truth object mask
    3. SAM predicted mask
    4. SAM mask overlay on RGB image
    """
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
    axes[0].set_title("RGB + box prompt")
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


def build_sample_paths(
    row: pd.Series,
    cfg: dict,
    project_root: Path,
    checkpoint_arg: str,
    output_dir_arg: str | None,
) -> SamplePaths:
    """Resolve all filesystem paths for one selected object row."""
    ocid_root = Path(cfg["ocid_root"])

    image_path = remap_ocid_path(row["image_path"], ocid_root)
    gt_mask_path = resolve_project_path(row["binary_mask_path"], project_root)
    checkpoint_path = resolve_project_path(checkpoint_arg, project_root)

    if output_dir_arg is None:
        output_dir = resolve_project_path(cfg["sam_outputs_dir"], project_root)
    else:
        output_dir = resolve_project_path(output_dir_arg, project_root)

    return SamplePaths(
        image_path=image_path,
        gt_mask_path=gt_mask_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )


def validate_paths(paths: SamplePaths) -> None:
    """Fail early with clear path errors."""
    if not paths.image_path.exists():
        raise FileNotFoundError(f"Resolved image path does not exist: {paths.image_path}")

    if not paths.gt_mask_path.exists():
        raise FileNotFoundError(f"GT mask path does not exist: {paths.gt_mask_path}")

    if not paths.checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint does not exist: {paths.checkpoint_path}")


def run_sam_box_prompt(
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    checkpoint_path: Path,
    model_type: str,
    device: str,
) -> tuple[np.ndarray, float]:
    """Run SAM prediction for one RGB image and one XYXY box prompt."""
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    sam.eval()

    predictor = SamPredictor(sam)

    with torch.inference_mode():
        predictor.set_image(image_rgb)

        masks, scores, _ = predictor.predict(
            box=box_xyxy,
            multimask_output=False,
        )

    sam_mask = masks[0]
    sam_score = float(scores[0])

    return sam_mask, sam_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM on one OCID object using a bounding-box prompt."
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
        "--row",
        type=int,
        default=0,
        help="Object row number to test.",
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
        help="Inference device. Use 'cuda' to require GPU, 'auto' to use GPU if compatible.",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path.cwd()
    config_path = resolve_project_path(args.config, project_root)
    index_path = resolve_project_path(args.index, project_root)

    cfg = load_config(config_path)

    df = pd.read_csv(index_path)
    validate_required_columns(df)

    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"Row {args.row} is outside valid range 0 to {len(df) - 1}")

    row = df.iloc[args.row]

    paths = build_sample_paths(
        row=row,
        cfg=cfg,
        project_root=project_root,
        checkpoint_arg=args.checkpoint,
        output_dir_arg=args.output_dir,
    )

    validate_paths(paths)

    box_xyxy = np.array(
        [
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        ],
        dtype=np.float32,
    )

    device = select_device(
        requested_device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )

    print("Using config:", config_path)
    print("Project root:", project_root)
    print("OCID root:", cfg["ocid_root"])
    print("Index path:", index_path)
    print("Checkpoint path:", paths.checkpoint_path)
    print("Output dir:", paths.output_dir)
    print()
    print("Selected row:", args.row)
    print("Object ID:", row["object_id"])
    print("Original CSV image path:", row["image_path"])
    print("Resolved image path:", paths.image_path)
    print("GT mask path:", paths.gt_mask_path)
    print("Box prompt XYXY:", box_xyxy.tolist())
    print("Device:", device)

    image_rgb = load_rgb_image(paths.image_path)
    gt_mask = load_binary_mask(paths.gt_mask_path)

    sam_mask, sam_score = run_sam_box_prompt(
        image_rgb=image_rgb,
        box_xyxy=box_xyxy,
        checkpoint_path=paths.checkpoint_path,
        model_type=args.model_type,
        device=device,
    )

    iou = compute_iou(sam_mask, gt_mask)

    object_id = int(row["object_id"])
    mask_output_path = paths.output_dir / f"row_{args.row:04d}_object_{object_id}_sam_mask.png"
    vis_output_path = paths.output_dir / f"row_{args.row:04d}_object_{object_id}_sam_visualization.png"

    save_mask(sam_mask, mask_output_path)

    save_visualization(
        image_rgb=image_rgb,
        gt_mask=gt_mask,
        sam_mask=sam_mask,
        box=box_xyxy,
        output_path=vis_output_path,
        iou=iou,
        sam_score=sam_score,
    )

    print()
    print("Done.")
    print("Saved SAM mask:", mask_output_path)
    print("Saved visualization:", vis_output_path)
    print(f"SAM score: {sam_score:.4f}")
    print(f"IoU with GT mask: {iou:.4f}")


if __name__ == "__main__":
    main()