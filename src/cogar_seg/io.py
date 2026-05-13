"""Small I/O helpers used by scripts and evaluation code."""

from pathlib import Path
import csv

import numpy as np

from cogar_seg.cv_compat import cv2


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file and return rows as dictionaries."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an RGB image as an HWC uint8 array."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask as a boolean array."""
    mask_path = Path(mask_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"OpenCV could not read mask: {mask_path}")

    return mask > 0


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> None:
    """Save a boolean-like mask as a binary PNG using 255 for foreground."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255)

    if not success:
        raise RuntimeError(f"OpenCV could not write mask: {output_path}")
