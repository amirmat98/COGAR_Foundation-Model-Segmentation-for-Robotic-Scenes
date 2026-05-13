"""Mask and prompt visualizations."""

from pathlib import Path
import os
from typing import Any

import numpy as np

from cogar_seg.cv_compat import cv2


def _pyplot():
    """Import pyplot after ensuring Matplotlib has a writable cache directory."""
    if "MPLCONFIGDIR" not in os.environ:
        default_config_dir = Path.home() / ".config" / "matplotlib"
        writable_target = default_config_dir if default_config_dir.exists() else default_config_dir.parent

        if not os.access(writable_target, os.W_OK):
            cache_dir = Path("/tmp/cogar_matplotlib")
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(cache_dir)

    import matplotlib.pyplot as plt

    return plt


def draw_box_and_point(
    image_bgr: np.ndarray,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    point_x: int,
    point_y: int,
) -> np.ndarray:
    """Draw a green box prompt and red point prompt on a BGR image."""
    output = image_bgr.copy()

    cv2.rectangle(output, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.circle(output, (point_x, point_y), 6, (0, 0, 255), -1)

    return output


def visualize_object_prompt_from_row(row: dict[str, Any], row_index: int) -> None:
    """Visualize one object prompt from one CSV row."""
    plt = _pyplot()

    image_path = row["image_path"]
    object_id = int(row["object_id"])

    xmin = int(row["bbox_xmin"])
    ymin = int(row["bbox_ymin"])
    xmax = int(row["bbox_xmax"])
    ymax = int(row["bbox_ymax"])

    point_x = int(row["point_x"])
    point_y = int(row["point_y"])

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    image_with_prompt_bgr = draw_box_and_point(
        image_bgr=image_bgr,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        point_x=point_x,
        point_y=point_y,
    )

    image_rgb = cv2.cvtColor(image_with_prompt_bgr, cv2.COLOR_BGR2RGB)

    print("Image:", image_path)
    print("Row index:", row_index)
    print("Object ID:", object_id)
    print("Bounding box:", xmin, ymin, xmax, ymax)
    print("Point prompt:", point_x, point_y)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Object {object_id}: box prompt + point prompt")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_binary_mask_from_row(row: dict[str, Any], row_index: int) -> None:
    """Visualize an RGB image and binary ground-truth mask side by side."""
    plt = _pyplot()

    image_path = row["image_path"]
    binary_mask_path = row["binary_mask_path"]
    object_id = row["object_id"]

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Could not read binary mask: {binary_mask_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Row index:", row_index)
    print("Object ID:", object_id)
    print("Image:", image_path)
    print("Binary mask:", binary_mask_path)
    print("Mask unique values:", sorted(set(int(v) for v in mask.flatten())))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("RGB image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Binary GT mask for object {object_id}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_sam_box_visualization(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    box_xyxy: np.ndarray,
    output_path: str | Path,
    iou: float,
    model_score: float,
    row_index: int | None = None,
    object_id: int | None = None,
    model_name: str = "SAM",
) -> None:
    """Save a four-panel box-prompt segmentation comparison."""
    plt = _pyplot()

    output_path = Path(output_path)
    x_min, y_min, x_max, y_max = box_xyxy.astype(int)

    overlay = image_rgb.copy()
    green_mask = np.zeros_like(image_rgb)
    green_mask[:, :, 1] = 255

    overlay = np.where(
        pred_mask[:, :, None],
        (0.6 * overlay + 0.4 * green_mask).astype(np.uint8),
        overlay,
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    title = "RGB + box prompt"
    if row_index is not None and object_id is not None:
        title = f"RGB + box | row={row_index}, obj={object_id}"

    axes[0].imshow(image_rgb)
    axes[0].set_title(title)
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

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"{model_name} predicted mask")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay | IoU={iou:.3f}, score={model_score:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_sam_point_visualization(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    point_coords: np.ndarray,
    output_path: str | Path,
    iou: float,
    model_score: float,
    row_index: int | None = None,
    object_id: int | None = None,
    model_name: str = "SAM",
) -> None:
    """Save a four-panel point-prompt segmentation comparison."""
    plt = _pyplot()

    output_path = Path(output_path)
    point_x, point_y = point_coords[0].astype(int)

    overlay = image_rgb.copy()
    green_mask = np.zeros_like(image_rgb)
    green_mask[:, :, 1] = 255

    overlay = np.where(
        pred_mask[:, :, None],
        (0.6 * overlay + 0.4 * green_mask).astype(np.uint8),
        overlay,
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    title = "RGB + point prompt"
    if row_index is not None and object_id is not None:
        title = f"RGB + point | row={row_index}, obj={object_id}"

    axes[0].imshow(image_rgb)
    axes[0].set_title(title)
    axes[0].scatter(
        [point_x],
        [point_y],
        c="red",
        s=80,
        marker="*",
        edgecolors="white",
        linewidths=1.5,
    )

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")

    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"{model_name} predicted mask")

    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay | IoU={iou:.3f}, score={model_score:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
