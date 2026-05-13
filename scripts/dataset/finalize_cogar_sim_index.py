import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cogar_seg.cv_compat import cv2


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

REQUIRED_INPUT_COLUMNS = {
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

REFLECTIVE_CATEGORIES = {"metal_part", "tool"}
TRANSPARENT_CATEGORIES = {"glass_object"}
SMALL_PART_CATEGORIES = {"screw", "connector"}


def bool_like(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def convert_bbox_xywh_to_xyxy(
    bbox_x: pd.Series,
    bbox_y: pd.Series,
    bbox_w: pd.Series,
    bbox_h: pd.Series,
) -> pd.DataFrame:
    """Convert COCO xywh boxes to xyxy boxes."""
    x = bbox_x.astype(float)
    y = bbox_y.astype(float)
    w = bbox_w.astype(float)
    h = bbox_h.astype(float)
    return pd.DataFrame(
        {
            "bbox_xmin": x,
            "bbox_ymin": y,
            "bbox_xmax": x + w,
            "bbox_ymax": y + h,
        }
    )


def object_flags_for_category(
    category_name: str,
    challenge_primary: str,
    occlusion_value=None,
) -> dict[str, bool]:
    """Create object-level material/size flags without copying scene flags."""
    category = str(category_name)
    challenge = str(challenge_primary)
    is_occluded = (
        bool_like(occlusion_value)
        if occlusion_value is not None and not pd.isna(occlusion_value)
        else challenge == "partial_occlusion"
    )
    return {
        "is_reflective": category in REFLECTIVE_CATEGORIES,
        "is_transparent": category in TRANSPARENT_CATEGORIES,
        "is_small_part": category in SMALL_PART_CATEGORIES,
        "is_dynamic": challenge == "dynamic_scene",
        "is_occluded": is_occluded or challenge == "partial_occlusion",
    }


def split_labels_for_images(image_ids: Iterable[int]) -> dict[int, str]:
    """Assign stable 70/15/15 splits by sorted image id, with sane tiny-set behavior."""
    ids = sorted({int(image_id) for image_id in image_ids})
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        counts = (1, 0, 0)
    elif n == 2:
        counts = (1, 0, 1)
    elif n == 3:
        counts = (1, 1, 1)
    else:
        train_count = max(1, int(round(0.70 * n)))
        val_count = max(1, int(round(0.15 * n)))
        if train_count + val_count >= n:
            val_count = max(1, n - train_count - 1)
        test_count = n - train_count - val_count
        if test_count <= 0:
            test_count = 1
            train_count = max(1, n - val_count - test_count)
        counts = (train_count, val_count, test_count)

    split_by_image: dict[int, str] = {}
    train_count, val_count, _ = counts
    for pos, image_id in enumerate(ids):
        if pos < train_count:
            split_by_image[image_id] = "train"
        elif pos < train_count + val_count:
            split_by_image[image_id] = "val"
        else:
            split_by_image[image_id] = "test"
    return split_by_image


def assign_splits_by_image(df: pd.DataFrame) -> pd.Series:
    split_by_image = split_labels_for_images(df["image_id"].astype(int).unique())
    return df["image_id"].astype(int).map(split_by_image)


def choose_point_inside_mask(
    mask_path: str | Path,
    fallback_x: float,
    fallback_y: float,
) -> tuple[float, float]:
    """Choose a prompt point at the deepest mask interior using distance transform."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None or mask.max() == 0:
        return float(fallback_x), float(fallback_y)

    binary = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return float(x), float(y)


def _metadata_challenge(df: pd.DataFrame) -> pd.Series:
    if "primary_challenge_meta" in df.columns:
        fallback = df["primary_challenge"] if "primary_challenge" in df.columns else "unknown"
        return df["primary_challenge_meta"].fillna(fallback)
    if "challenge_primary_meta" in df.columns:
        fallback = df["challenge_primary"] if "challenge_primary" in df.columns else "unknown"
        return df["challenge_primary_meta"].fillna(fallback)
    if "primary_challenge" in df.columns:
        return df["primary_challenge"].fillna("unknown")
    if "challenge_primary" in df.columns:
        return df["challenge_primary"].fillna("unknown")
    return pd.Series(["unknown"] * len(df), index=df.index)


def finalize_index_dataframe(
    object_index: pd.DataFrame,
    metadata: pd.DataFrame,
    exclude_categories: Iterable[str] | None = None,
    exclude_files: Iterable[str] | None = None,
    min_area: float = 25.0,
) -> tuple[pd.DataFrame, int]:
    """Build the SAM-compatible COGAR-Sim index from the mask-export CSV."""
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(object_index.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = object_index.copy()
    before = len(df)

    excluded_categories = set(exclude_categories or [])
    if excluded_categories:
        df = df[~df["category_name"].astype(str).isin(excluded_categories)].copy()

    excluded_files = set(exclude_files or [])
    if excluded_files:
        df = df[~df["file_name"].astype(str).isin(excluded_files)].copy()

    df = df[df["area"].astype(float) >= float(min_area)].copy()
    if df.empty:
        raise ValueError("Finalized index would be empty after filtering.")

    meta_cols = [
        "image_id",
        "scene_id",
        "scene_family",
        "primary_challenge",
        "challenge_primary",
        "reflective",
        "transparent",
        "occlusion",
        "small_parts",
        "dynamic",
        "camera_view",
        "lighting_condition",
        "occlusion_level",
    ]
    available_meta_cols = [c for c in meta_cols if c in metadata.columns]
    if "image_id" in available_meta_cols:
        df = df.merge(
            metadata[available_meta_cols],
            on="image_id",
            how="left",
            suffixes=("", "_meta"),
        )

    out = pd.DataFrame(index=df.index)
    out["image_id"] = df["image_id"].astype(int)
    out["file_name"] = df["file_name"].astype(str)
    if "scene_id" in df.columns:
        out["scene_id"] = df["scene_id"].fillna("unknown").astype(str)
    else:
        out["scene_id"] = "unknown"
    out["frame_id"] = df["image_id"].astype(int)
    out["split"] = assign_splits_by_image(df)

    out["image_path"] = df["image_path"].astype(str)
    out["binary_mask_path"] = df["gt_mask_path"].astype(str)
    out["instance_mask_path"] = df["gt_mask_path"].astype(str)
    out["semantic_mask_path"] = ""

    out["category_id"] = df["category_id"].astype(int)
    out["category_name"] = df["category_name"].astype(str)
    out["object_id"] = df["annotation_id"].astype(int)

    bbox = convert_bbox_xywh_to_xyxy(
        df["bbox_x"],
        df["bbox_y"],
        df["bbox_w"],
        df["bbox_h"],
    )
    for col in ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]:
        out[col] = bbox[col].values

    points = [
        choose_point_inside_mask(mask_path, bx + bw / 2.0, by + bh / 2.0)
        for mask_path, bx, by, bw, bh in zip(
            df["gt_mask_path"],
            df["bbox_x"].astype(float),
            df["bbox_y"].astype(float),
            df["bbox_w"].astype(float),
            df["bbox_h"].astype(float),
        )
    ]
    out["point_x"] = [p[0] for p in points]
    out["point_y"] = [p[1] for p in points]

    out["challenge_primary"] = _metadata_challenge(df).astype(str)
    out["challenge_secondary"] = ""

    occlusion_values = df["occlusion"] if "occlusion" in df.columns else [None] * len(df)
    flags = [
        object_flags_for_category(category, challenge, occlusion)
        for category, challenge, occlusion in zip(
            out["category_name"],
            out["challenge_primary"],
            occlusion_values,
        )
    ]
    for col in [
        "is_reflective",
        "is_transparent",
        "is_occluded",
        "is_small_part",
        "is_dynamic",
    ]:
        out[col] = [flag[col] for flag in flags]

    out["area"] = df["area"].astype(float)
    out = out[OUTPUT_COLUMNS].reset_index(drop=True)
    return out, before


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize a COGAR-Sim object index into the SAM-compatible schema."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--metadata", default="data/cogar_sim_500/metadata/frame_index.csv")
    parser.add_argument("--output", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--exclude-categories", nargs="*", default=["table"])
    parser.add_argument("--exclude-files", nargs="*", default=[])
    parser.add_argument("--min-area", type=float, default=25.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    metadata = pd.read_csv(args.metadata)
    out, before = finalize_index_dataframe(
        object_index=df,
        metadata=metadata,
        exclude_categories=args.exclude_categories,
        exclude_files=args.exclude_files,
        min_area=args.min_area,
    )

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

    print("\nObject-level reflective counts:")
    print(out[out["is_reflective"]]["category_name"].value_counts())

    print("\nObject-level transparent counts:")
    print(out[out["is_transparent"]]["category_name"].value_counts())

    print("\nObject-level small-part counts:")
    print(out[out["is_small_part"]]["category_name"].value_counts())


if __name__ == "__main__":
    main()
