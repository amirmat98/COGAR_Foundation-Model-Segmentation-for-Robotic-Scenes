"""Utilities for simulated robotic-scene benchmark indexes."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cogar_seg.datasets.coco_utils import (
    load_categories_from_yaml,
    load_json,
    write_json,
)


REQUIRED_SIM_INDEX_COLUMNS = [
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

VALID_SIM_SPLITS = {"train", "val", "test"}

BBOX_COLUMNS = [
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
]

POINT_COLUMNS = [
    "point_x",
    "point_y",
]

BOOLEAN_COLUMNS = [
    "is_reflective",
    "is_transparent",
    "is_occluded",
    "is_small_part",
    "is_dynamic",
]

__all__ = [
    "BBOX_COLUMNS",
    "BOOLEAN_COLUMNS",
    "NormalizedCogarSim",
    "POINT_COLUMNS",
    "REQUIRED_SIM_INDEX_COLUMNS",
    "VALID_SIM_SPLITS",
    "load_sim_index",
    "normalize_cogar_sim",
    "summarize_sim_index",
    "validate_sim_boolean_columns",
    "validate_sim_bounding_boxes",
    "validate_sim_category_ids",
    "validate_sim_index",
    "validate_sim_index_columns",
    "validate_sim_points",
    "validate_sim_splits",
]


def validate_sim_index_columns(df: pd.DataFrame) -> None:
    """Validate that all required simulation-index columns are present."""
    missing = [col for col in REQUIRED_SIM_INDEX_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required simulation-index columns: {missing}")


def validate_sim_splits(
    df: pd.DataFrame,
    valid_splits: set[str] | None = None,
) -> None:
    """Validate that all split names are known."""
    if valid_splits is None:
        valid_splits = VALID_SIM_SPLITS

    invalid_splits = sorted(set(df["split"].astype(str)) - valid_splits)

    if invalid_splits:
        raise ValueError(
            "Invalid split values found. "
            f"Expected one of {sorted(valid_splits)}, got {invalid_splits}"
        )


def validate_sim_bounding_boxes(
    df: pd.DataFrame,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Validate XYXY bounding boxes."""
    for col in BBOX_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Bounding-box column must be numeric: {col}")

    invalid_geometry = df[
        (df["bbox_xmin"] >= df["bbox_xmax"])
        | (df["bbox_ymin"] >= df["bbox_ymax"])
    ]

    if not invalid_geometry.empty:
        rows = invalid_geometry.index.tolist()
        raise ValueError(f"Invalid bounding-box geometry in rows: {rows}")

    negative_coords = df[
        (df["bbox_xmin"] < 0)
        | (df["bbox_ymin"] < 0)
        | (df["bbox_xmax"] < 0)
        | (df["bbox_ymax"] < 0)
    ]

    if not negative_coords.empty:
        rows = negative_coords.index.tolist()
        raise ValueError(f"Negative bounding-box coordinates in rows: {rows}")

    if image_width is not None:
        too_wide = df[df["bbox_xmax"] > image_width]
        if not too_wide.empty:
            rows = too_wide.index.tolist()
            raise ValueError(
                f"Bounding boxes exceed image width {image_width} in rows: {rows}"
            )

    if image_height is not None:
        too_tall = df[df["bbox_ymax"] > image_height]
        if not too_tall.empty:
            rows = too_tall.index.tolist()
            raise ValueError(
                f"Bounding boxes exceed image height {image_height} in rows: {rows}"
            )


def validate_sim_points(
    df: pd.DataFrame,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """Validate point-prompt coordinates."""
    for col in POINT_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Point column must be numeric: {col}")

    negative_points = df[(df["point_x"] < 0) | (df["point_y"] < 0)]

    if not negative_points.empty:
        rows = negative_points.index.tolist()
        raise ValueError(f"Negative point coordinates in rows: {rows}")

    if image_width is not None:
        too_wide = df[df["point_x"] >= image_width]
        if not too_wide.empty:
            rows = too_wide.index.tolist()
            raise ValueError(
                f"Point x coordinate exceeds image width {image_width} in rows: {rows}"
            )

    if image_height is not None:
        too_tall = df[df["point_y"] >= image_height]
        if not too_tall.empty:
            rows = too_tall.index.tolist()
            raise ValueError(
                f"Point y coordinate exceeds image height {image_height} in rows: {rows}"
            )


def validate_sim_boolean_columns(df: pd.DataFrame) -> None:
    """Validate boolean challenge-flag columns.

    Accepted values:
    - Python booleans: True, False
    - Integers: 0, 1
    - Strings: true, false, 0, 1
    """
    accepted = {True, False, 0, 1, "true", "false", "True", "False", "0", "1"}

    for col in BOOLEAN_COLUMNS:
        invalid_values = sorted(set(df[col].dropna().tolist()) - accepted, key=str)

        if invalid_values:
            raise ValueError(
                f"Invalid boolean-like values in column '{col}': {invalid_values}"
            )


def validate_sim_category_ids(
    df: pd.DataFrame,
    allowed_category_ids: set[int] | None = None,
) -> None:
    """Validate category IDs when an allowed set is provided."""
    if not pd.api.types.is_numeric_dtype(df["category_id"]):
        raise ValueError("category_id must be numeric")

    if allowed_category_ids is None:
        return

    observed = set(df["category_id"].astype(int).tolist())
    invalid = sorted(observed - allowed_category_ids)

    if invalid:
        raise ValueError(f"Invalid category IDs found: {invalid}")


def validate_sim_index(
    df: pd.DataFrame,
    image_width: int | None = None,
    image_height: int | None = None,
    allowed_category_ids: set[int] | None = None,
) -> None:
    """Run all validation checks for a simulated robotic-scene index.

    Empty placeholder indexes are allowed during dataset setup. For an empty
    index, only the column schema is validated because pandas may load empty CSV
    columns with object dtype.
    """
    validate_sim_index_columns(df)

    if df.empty:
        return

    validate_sim_splits(df)
    validate_sim_bounding_boxes(
        df=df,
        image_width=image_width,
        image_height=image_height,
    )
    validate_sim_points(
        df=df,
        image_width=image_width,
        image_height=image_height,
    )
    validate_sim_boolean_columns(df)
    validate_sim_category_ids(df, allowed_category_ids=allowed_category_ids)


def load_sim_index(
    index_path: str | Path,
    validate: bool = True,
    image_width: int | None = None,
    image_height: int | None = None,
    allowed_category_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Load a simulated robotic-scene benchmark index CSV."""
    path = Path(index_path)

    if not path.exists():
        raise FileNotFoundError(f"Simulation index CSV not found: {path}")

    df = pd.read_csv(path)

    if validate:
        validate_sim_index(
            df=df,
            image_width=image_width,
            image_height=image_height,
            allowed_category_ids=allowed_category_ids,
        )

    return df


def summarize_sim_index(df: pd.DataFrame) -> dict[str, Any]:
    """Create a compact summary of a simulated dataset index."""
    validate_sim_index_columns(df)

    return {
        "num_object_instances": int(len(df)),
        "num_images": int(df["image_id"].nunique()),
        "num_scenes": int(df["scene_id"].nunique()),
        "splits": df["split"].value_counts().sort_index().to_dict(),
        "categories": df["category_name"].value_counts().sort_index().to_dict(),
        "primary_challenges": (
            df["challenge_primary"].value_counts().sort_index().to_dict()
        ),
    }


@dataclass(frozen=True)
class NormalizedCogarSim:
    """Paths and counts from normalizing BlenderProc COGAR-Sim output."""

    root: Path
    rgb_dir: Path
    annotations_path: Path
    metadata_path: Path
    categories_path: Path
    splits_dir: Path
    num_images: int
    num_annotations: int
    num_metadata_rows: int


def normalize_cogar_sim(
    raw_coco_dir: str | Path = (
        "/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/raw_blenderproc/cogar_sim_1000_raw/coco_data"
    ),
    raw_metadata_path: str | Path = "/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/metadata/frame_index_raw.csv",
    output_root: str | Path = "/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000",
    config_path: str | Path = "configs/blenderproc_dataset.yaml",
    expected_images: int = 1000,
    clean_rgb_dir: bool = True,
) -> NormalizedCogarSim:
    """Normalize raw BlenderProc COCO output into the COGAR-Sim layout."""
    raw_coco_dir = Path(raw_coco_dir)
    raw_images_dir = raw_coco_dir / "images"
    raw_coco_path = raw_coco_dir / "coco_annotations.json"
    raw_metadata_path = Path(raw_metadata_path)
    root = Path(output_root)

    if not raw_coco_path.exists():
        raise FileNotFoundError(f"Missing COCO file: {raw_coco_path}")

    if not raw_images_dir.exists():
        raise FileNotFoundError(f"Missing image folder: {raw_images_dir}")

    if not raw_metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {raw_metadata_path}")

    coco = load_json(raw_coco_path)
    categories = load_categories_from_yaml(config_path)
    raw_images = sorted(raw_images_dir.glob("*.png"))
    with raw_metadata_path.open("r", newline="", encoding="utf-8") as f:
        metadata_rows = list(csv.DictReader(f))

    if len(raw_images) != expected_images:
        raise ValueError(f"Expected {expected_images} raw images, found {len(raw_images)}")

    if len(coco["images"]) != expected_images:
        raise ValueError(
            f"Expected {expected_images} COCO images, found {len(coco['images'])}"
        )

    if len(metadata_rows) != expected_images:
        raise ValueError(
            f"Expected {expected_images} metadata rows, found {len(metadata_rows)}"
        )

    if len(coco["annotations"]) <= expected_images:
        raise ValueError(
            "Too few COCO annotations for an object-instance dataset: "
            f"{len(coco['annotations'])}"
        )

    rgb_dir = root / "rgb"
    ann_dir = root / "annotations"
    meta_dir = root / "metadata"
    split_dir = root / "splits"

    for output_dir in [rgb_dir, ann_dir, meta_dir, split_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)

    if clean_rgb_dir:
        for old_png in rgb_dir.glob("*.png"):
            old_png.unlink()

    for img_path in raw_images:
        shutil.copy2(img_path, rgb_dir / img_path.name)

    coco["categories"] = categories
    annotations_path = ann_dir / "instances_all.json"
    categories_path = meta_dir / "categories.json"
    metadata_path = meta_dir / "frame_index.csv"

    write_json(coco, annotations_path)
    write_json(categories, categories_path)
    shutil.copy2(raw_metadata_path, metadata_path)

    ids = [f"{i:06d}" for i in range(expected_images)]
    train_count = int(round(0.70 * expected_images))
    val_count = int(round(0.15 * expected_images))
    if expected_images >= 3:
        train_count = max(1, min(train_count, expected_images - 2))
        val_count = max(1, min(val_count, expected_images - train_count - 1))
    else:
        train_count = max(1, expected_images)
        val_count = 0
    train = ids[:train_count]
    val = ids[train_count : train_count + val_count]
    test = ids[train_count + val_count : expected_images]

    (split_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (split_dir / "all.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    return NormalizedCogarSim(
        root=root,
        rgb_dir=rgb_dir,
        annotations_path=annotations_path,
        metadata_path=metadata_path,
        categories_path=categories_path,
        splits_dir=split_dir,
        num_images=len(raw_images),
        num_annotations=len(coco["annotations"]),
        num_metadata_rows=len(metadata_rows),
    )
