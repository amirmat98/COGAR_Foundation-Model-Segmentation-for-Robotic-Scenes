#!/usr/bin/env python3
"""
Evaluate SAM ViT-B automatic mask generation on COGAR-SimRobotics-500.

For each image:
1. Generate automatic masks once using SamAutomaticMaskGenerator.
2. For every clean GT object in that image, find the generated mask with maximum IoU.
3. Save one result row per GT object.

This is an oracle proposal-matching evaluation:
it measures whether SAM AMG generates a good candidate mask for each object.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogar_seg.cv_compat import cv2
from cogar_seg.metrics import compute_iou
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read GT mask: {path}")
    return mask > 0


def bbox_xywh_to_xyxy(bbox: list[float] | tuple[float, ...] | None) -> tuple[float, float, float, float]:
    if bbox is None or len(bbox) != 4:
        return (-1.0, -1.0, -1.0, -1.0)
    x, y, w, h = bbox
    return (float(x), float(y), float(x + w), float(y + h))


def make_mask_generator(args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    sam = sam_model_registry[args.model_type](checkpoint=str(checkpoint))
    sam.to(device=args.device)

    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )


def save_best_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (mask.astype(np.uint8) * 255))


def evaluate(args: argparse.Namespace) -> None:
    index_csv = resolve_path(args.index_csv)
    output_csv = resolve_path(args.output_csv)
    output_dir = resolve_path(args.output_dir)
    best_mask_dir = output_dir / "best_object_masks"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_csv)

    required_cols = [
        "row_index",
        "file_name",
        "object_id",
        "image_path",
        "gt_mask_path",
        "iou",
        "sam_score",
        "category_name",
        "primary_challenge",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in index CSV: {missing}")

    # Optional quick/debug subset by number of unique images.
    unique_images = list(df["image_path"].drop_duplicates())
    if args.max_images is not None:
        selected_images = set(unique_images[: args.max_images])
        df = df[df["image_path"].isin(selected_images)].copy()

    existing_rows: list[dict[str, Any]] = []
    processed_row_indexes: set[int] = set()

    if args.resume and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        existing_rows = existing_df.to_dict("records")
        processed_row_indexes = set(existing_df["row_index"].astype(int).tolist())
        print(f"Resume enabled: found {len(processed_row_indexes)} existing evaluated objects.")

    generator = make_mask_generator(args)

    config = {
        "model_type": args.model_type,
        "checkpoint": str(resolve_path(args.checkpoint)),
        "device": args.device,
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "index_csv": str(index_csv),
        "output_csv": str(output_csv),
        "max_images": args.max_images,
    }
    (output_dir / "sam_auto_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = existing_rows
    image_groups = list(df.groupby("image_path", sort=False))

    start_all = time.time()

    for image_idx, (image_path_raw, group) in enumerate(image_groups, start=1):
        remaining_group = group[~group["row_index"].astype(int).isin(processed_row_indexes)]
        if remaining_group.empty:
            print(f"[{image_idx:04d}/{len(image_groups):04d}] Skipping already processed image: {image_path_raw}")
            continue

        image_path = resolve_path(image_path_raw)
        image = read_rgb(image_path)

        start = time.time()
        masks = generator.generate(image)
        elapsed = time.time() - start

        print(
            f"[{image_idx:04d}/{len(image_groups):04d}] "
            f"{Path(image_path_raw).name} | generated masks: {len(masks)} | "
            f"objects: {len(remaining_group)} | time: {elapsed:.2f}s"
        )

        for _, row in remaining_group.iterrows():
            gt_mask_path = resolve_path(row["gt_mask_path"])
            gt_mask = read_binary_mask(gt_mask_path)

            best_iou = 0.0
            best_mask_idx = -1
            best_mask = None
            best_meta: dict[str, Any] = {}

            for mask_idx, mask_dict in enumerate(masks):
                pred_mask = mask_dict["segmentation"].astype(bool)
                iou = compute_iou(gt_mask, pred_mask)

                if iou > best_iou:
                    best_iou = iou
                    best_mask_idx = mask_idx
                    best_mask = pred_mask
                    best_meta = mask_dict

            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox_xywh_to_xyxy(best_meta.get("bbox"))

            best_mask_path = ""
            if args.save_best_masks and best_mask is not None:
                best_mask_path_obj = best_mask_dir / f"row_{int(row['row_index']):06d}_object_{int(row['object_id'])}_sam_auto_best.png"
                save_best_mask(best_mask, best_mask_path_obj)
                best_mask_path = str(best_mask_path_obj.relative_to(REPO_ROOT))

            result = {
                "row_index": int(row["row_index"]),
                "file_name": row["file_name"],
                "object_id": int(row["object_id"]),
                "image_path": row["image_path"],
                "gt_mask_path": row["gt_mask_path"],
                "category_name": row["category_name"],
                "primary_challenge": row["primary_challenge"],
                "box_prompt_iou": float(row["iou"]),
                "box_prompt_sam_score": float(row["sam_score"]),
                "sam_auto_best_iou": float(best_iou),
                "sam_auto_best_mask_index": int(best_mask_idx),
                "sam_auto_num_masks_image": int(len(masks)),
                "sam_auto_predicted_iou": float(best_meta.get("predicted_iou", np.nan)),
                "sam_auto_stability_score": float(best_meta.get("stability_score", np.nan)),
                "sam_auto_mask_area": float(best_meta.get("area", np.nan)),
                "sam_auto_bbox_xmin": bbox_xmin,
                "sam_auto_bbox_ymin": bbox_ymin,
                "sam_auto_bbox_xmax": bbox_xmax,
                "sam_auto_bbox_ymax": bbox_ymax,
                "sam_auto_best_mask_path": best_mask_path,
                "device": args.device,
                "model_type": args.model_type,
                "points_per_side": args.points_per_side,
                "pred_iou_thresh": args.pred_iou_thresh,
                "stability_score_thresh": args.stability_score_thresh,
                "crop_n_layers": args.crop_n_layers,
                "min_mask_region_area": args.min_mask_region_area,
            }

            results.append(result)
            processed_row_indexes.add(int(row["row_index"]))

        pd.DataFrame(results).sort_values("row_index").to_csv(output_csv, index=False)

        if args.device == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - start_all
    print("\nDone.")
    print(f"Saved results: {output_csv}")
    print(f"Evaluated objects: {len(results)}")
    print(f"Total time: {total_elapsed:.2f}s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-csv",
        default="outputs/indexes/cogar_sim_500_sam_box_clean_results.csv",
        help="Clean object-level CSV with GT masks and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="SAM checkpoint path.",
    )
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cogar_sim_500/sam_auto_masks",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/indexes/cogar_sim_500_sam_auto_clean_results.csv",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-best-masks", action="store_true")

    # GTX 1050-safe AMG defaults.
    parser.add_argument("--points-per-side", type=int, default=16)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
