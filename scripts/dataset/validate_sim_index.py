import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from cogar_seg.cv_compat import cv2


REQUIRED_COLUMNS = [
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

BOOLEAN_COLUMNS = [
    "is_reflective",
    "is_transparent",
    "is_occluded",
    "is_small_part",
    "is_dynamic",
]

VALID_SPLITS = {"train", "val", "test"}
BOOLEAN_VALUES = {True, False, 0, 1, "0", "1", "true", "false", "True", "False"}


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_boolean_columns(df: pd.DataFrame) -> None:
    for col in BOOLEAN_COLUMNS:
        invalid = sorted(set(df[col].dropna().tolist()) - BOOLEAN_VALUES, key=str)
        if invalid:
            raise ValueError(f"Invalid boolean-like values in {col}: {invalid}")


def validate_sim_index_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    validate_required_columns(df)
    if df.empty:
        raise ValueError("Index is empty")

    bad_splits = sorted(set(df["split"].astype(str)) - VALID_SPLITS)
    if bad_splits:
        raise ValueError(f"Invalid split values: {bad_splits}")
    validate_boolean_columns(df)

    warnings: list[str] = []
    if "table" in set(df["category_name"].astype(str)):
        warnings.append("table category is present in the benchmark index")

    for i, row in df.iterrows():
        image_path = Path(str(row["image_path"]))
        mask_path = Path(str(row["binary_mask_path"]))

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image at row {i}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask at row {i}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable image at row {i}: {image_path}")
        if mask is None:
            raise ValueError(f"Unreadable mask at row {i}: {mask_path}")

        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            raise ValueError(
                f"Mask shape does not match image at row {i}: "
                f"mask={mask.shape[:2]}, image={(h, w)}"
            )

        xmin = float(row["bbox_xmin"])
        ymin = float(row["bbox_ymin"])
        xmax = float(row["bbox_xmax"])
        ymax = float(row["bbox_ymax"])
        px = int(round(float(row["point_x"])))
        py = int(round(float(row["point_y"])))
        area = float(row["area"])

        if not (0 <= xmin < xmax <= w):
            raise ValueError(f"Invalid bbox x at row {i}: xmin={xmin}, xmax={xmax}, width={w}")
        if not (0 <= ymin < ymax <= h):
            raise ValueError(f"Invalid bbox y at row {i}: ymin={ymin}, ymax={ymax}, height={h}")
        if not (0 <= px < w and 0 <= py < h):
            raise ValueError(f"Invalid point at row {i}: point=({px}, {py}), image=({w}, {h})")
        if mask.max() == 0:
            raise ValueError(f"Mask is empty at row {i}: {mask_path}")
        if mask[py, px] == 0:
            raise ValueError(f"Point is not inside mask at row {i}: point=({px}, {py})")
        if area <= 0:
            raise ValueError(f"Area must be positive at row {i}: area={area}")

    num_images = int(df["file_name"].nunique())
    num_objects = int(len(df))
    if num_images < 10:
        warnings.append(f"image count seems low for a benchmark: {num_images}")
    if num_objects < 50:
        warnings.append(f"object count seems low for a benchmark: {num_objects}")

    summary = {
        "rows": num_objects,
        "images": num_images,
        "splits": df["split"].value_counts().to_dict(),
        "categories": df["category_name"].value_counts().to_dict(),
        "challenges": df["challenge_primary"].value_counts().to_dict(),
        "warnings": warnings,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a finalized COGAR-Sim index.")
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.index)
    summary = validate_sim_index_dataframe(df)

    print(f"[OK] Valid index: {args.index}")
    print(f"Rows: {summary['rows']}")
    print(f"Images: {summary['images']}")
    print("\nSplit counts:")
    print(pd.Series(summary["splits"]))
    print("\nCategory counts:")
    print(pd.Series(summary["categories"]))
    print("\nChallenge counts:")
    print(pd.Series(summary["challenges"]))
    for warning in summary["warnings"]:
        print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
