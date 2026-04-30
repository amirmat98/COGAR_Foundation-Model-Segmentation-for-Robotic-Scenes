"""Bounding-box prompt helpers."""

from typing import Any, Mapping

import numpy as np


def make_box_from_row(row: Mapping[str, Any]) -> np.ndarray:
    """Build an XYXY float32 box prompt from an object-index row."""
    return np.array(
        [
            row["bbox_xmin"],
            row["bbox_ymin"],
            row["bbox_xmax"],
            row["bbox_ymax"],
        ],
        dtype=np.float32,
    )
