"""Utilities for simulated robotic-scene benchmark indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_SIM_INDEX_COLUMNS = [
    "image_id",
    "scene_id",
    "frame_id",
    "split",
    "image_path",
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
    "camera_name",
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
