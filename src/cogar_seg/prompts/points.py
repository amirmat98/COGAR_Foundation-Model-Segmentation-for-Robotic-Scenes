from __future__ import annotations

from typing import Any

import numpy as np


def make_positive_point_prompt(row: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a SAM positive point prompt from an object-level CSV row.

    The row must contain:
        - point_x
        - point_y

    Returns:
        point_coords:
            NumPy array with shape (1, 2), containing the point in XY pixel format.

        point_labels:
            NumPy array with shape (1,), where label 1 means foreground point.
    """
    point_x = float(row["point_x"])
    point_y = float(row["point_y"])

    point_coords = np.array([[point_x, point_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    return point_coords, point_labels