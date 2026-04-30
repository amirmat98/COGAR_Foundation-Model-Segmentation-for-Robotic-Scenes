"""Metrics for binary and instance segmentation masks."""

import numpy as np


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)
