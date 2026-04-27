import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor


def compute_iou(mask_a, mask_b):
    """
    Compute Intersection over Union between two binary masks.
    """
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def save_visualization(image_rgb, gt_mask, sam_mask, box, output_path, iou, sam_score):
    """
    Save a 4-panel figure:
    1. RGB image with box prompt
    2. Ground-truth binary object mask
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
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

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
        "--output-dir",
        default="outputs/sam_box_prompt",
        help="Directory for output masks and visualizations.",
    )

    args = parser.parse_args()

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)

    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"Row {args.row} is outside valid range 0 to {len(df) - 1}")

    row = df.iloc[args.row]

    image_path = Path(row["image_path"])
    gt_mask_path = Path(row["binary_mask_path"])

    box = np.array(
        [
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        ],
        dtype=np.float32,
    )

    print("Selected row:", args.row)
    print("Image path:", image_path)
    print("GT mask path:", gt_mask_path)
    print("Object ID:", row["object_id"])
    print("Box prompt XYXY:", box.tolist())

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"Could not read GT mask: {gt_mask_path}")

    gt_mask = gt_mask > 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # SAM expects RGB image in HWC uint8 format.
    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=False,
    )

    sam_mask = masks[0]
    sam_score = float(scores[0])

    iou = compute_iou(sam_mask, gt_mask)

    mask_output_path = output_dir / f"row_{args.row:04d}_object_{int(row['object_id'])}_sam_mask.png"
    vis_output_path = output_dir / f"row_{args.row:04d}_object_{int(row['object_id'])}_sam_visualization.png"

    cv2.imwrite(str(mask_output_path), sam_mask.astype(np.uint8) * 255)

    save_visualization(
        image_rgb=image_rgb,
        gt_mask=gt_mask,
        sam_mask=sam_mask,
        box=box,
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