import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "image_id",
    "file_name",
    "scene_id",
    "frame_id",
    "split",
    "image_path",
    "binary_mask_path",
    "instance_mask_path",
    "semantic_mask_path",
    "category_id",
    "category_name",
    "object_id",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "point_x",
    "point_y",
    "challenge_primary",
    "challenge_secondary",
    "is_reflective",
    "is_transparent",
    "is_occluded",
    "is_small_part",
    "is_dynamic",
    "area",
]


def bool_like(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def choose_point_inside_mask(mask_path: str, fallback_x: float, fallback_y: float) -> tuple[float, float]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None or mask.max() == 0:
        return float(fallback_x), float(fallback_y)

    binary = (mask > 0).astype(np.uint8)

    # Choose the safest interior point, not just the bbox center.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)

    return float(x), float(y)


def assign_splits_by_image(df: pd.DataFrame) -> pd.Series:
    image_ids = sorted(df["image_id"].astype(int).unique())
    n = len(image_ids)

    train_cut = int(0.70 * n)
    val_cut = int(0.85 * n)

    split_by_image = {}

    for i, image_id in enumerate(image_ids):
        if i < train_cut:
            split_by_image[image_id] = "train"
        elif i < val_cut:
            split_by_image[image_id] = "val"
        else:
            split_by_image[image_id] = "test"

    return df["image_id"].astype(int).map(split_by_image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata", default="data/cogar_sim_500/metadata/frame_index.csv")
    parser.add_argument("--output", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--exclude-categories", nargs="*", default=["table"])
    parser.add_argument("--exclude-files", nargs="*", default=[])
    parser.add_argument("--min-area", type=float, default=25.0)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    metadata = pd.read_csv(args.metadata)

    required = {
        "image_id",
        "file_name",
        "image_path",
        "annotation_id",
        "category_id",
        "category_name",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "area",
        "gt_mask_path",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    before = len(df)

    if args.exclude_categories:
        df = df[~df["category_name"].astype(str).isin(args.exclude_categories)].copy()

    if args.exclude_files:
        df = df[~df["file_name"].astype(str).isin(args.exclude_files)].copy()

    df = df[df["area"].astype(float) >= args.min_area].copy()

    if df.empty:
        raise ValueError("Finalized index would be empty after filtering.")

    meta_cols = [
        "image_id",
        "scene_id",
        "scene_family",
        "primary_challenge",
        "reflective",
        "transparent",
        "occlusion",
        "small_parts",
        "dynamic",
        "camera_view",
        "lighting_condition",
    ]
    available_meta_cols = [c for c in meta_cols if c in metadata.columns]

    df = df.merge(
        metadata[available_meta_cols],
        on="image_id",
        how="left",
        suffixes=("", "_meta"),
    )

    out = pd.DataFrame()

    out["image_id"] = df["image_id"].astype(int)
    out["file_name"] = df["file_name"].astype(str)
    out["scene_id"] = df.get("scene_id", "unknown")
    out["frame_id"] = df["image_id"].astype(int)
    out["split"] = assign_splits_by_image(df)

    out["image_path"] = df["image_path"].astype(str)
    out["binary_mask_path"] = df["gt_mask_path"].astype(str)
    out["instance_mask_path"] = df["gt_mask_path"].astype(str)
    out["semantic_mask_path"] = ""

    out["category_id"] = df["category_id"].astype(int)
    out["category_name"] = df["category_name"].astype(str)
    out["object_id"] = df["annotation_id"].astype(int)

    x = df["bbox_x"].astype(float)
    y = df["bbox_y"].astype(float)
    w = df["bbox_w"].astype(float)
    h = df["bbox_h"].astype(float)

    out["bbox_xmin"] = x
    out["bbox_ymin"] = y
    out["bbox_xmax"] = x + w
    out["bbox_ymax"] = y + h

    points = [
        choose_point_inside_mask(mask_path, bx + bw / 2.0, by + bh / 2.0)
        for mask_path, bx, by, bw, bh in zip(df["gt_mask_path"], x, y, w, h)
    ]

    out["point_x"] = [p[0] for p in points]
    out["point_y"] = [p[1] for p in points]

    if "primary_challenge_meta" in df.columns:
        out["challenge_primary"] = df["primary_challenge_meta"].fillna(df.get("primary_challenge", "unknown"))
    else:
        out["challenge_primary"] = df.get("primary_challenge", "unknown")

    out["challenge_secondary"] = ""

    for src, dst in [
        ("reflective", "is_reflective"),
        ("transparent", "is_transparent"),
        ("occlusion", "is_occluded"),
        ("small_parts", "is_small_part"),
        ("dynamic", "is_dynamic"),
    ]:
        if src in df.columns:
            out[dst] = df[src].map(bool_like)
        else:
            out[dst] = False

    out["area"] = df["area"].astype(float)

    out = out[OUTPUT_COLUMNS]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"[OK] Wrote finalized index: {output_path}")
    print(f"Rows before filtering: {before}")
    print(f"Rows after filtering: {len(out)}")
    print("\nSplit counts:")
    print(out["split"].value_counts())
    print("\nCategory counts:")
    print(out["category_name"].value_counts())
    print("\nChallenge counts:")
    print(out["challenge_primary"].value_counts())


if __name__ == "__main__":
    main()
