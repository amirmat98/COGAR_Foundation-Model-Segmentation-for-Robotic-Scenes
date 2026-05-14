"""Metrics for binary and instance segmentation masks."""

import numpy as np

from cogar_seg.cv_compat import cv2


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute intersection-over-union between two binary masks."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_boundary_f1(
    mask_a: np.ndarray, mask_b: np.ndarray, bound_thresh: int = 2
) -> float:
    """Compute Boundary F1 score between two binary masks using morphological operations.

    Essential for measuring contour precision on thin edges and small robotic parts.
    """
    mask_a = (mask_a > 0).astype(np.uint8)
    mask_b = (mask_b > 0).astype(np.uint8)

    if mask_a.sum() == 0 and mask_b.sum() == 0:
        return 1.0
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return 0.0

    # Extract internal boundaries
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary_a = mask_a ^ cv2.erode(mask_a, kernel_small)
    boundary_b = mask_b ^ cv2.erode(mask_b, kernel_small)

    if boundary_a.sum() == 0 and boundary_b.sum() == 0:
        return 1.0
    if boundary_a.sum() == 0 or boundary_b.sum() == 0:
        return 0.0

    # Dilate boundaries to account for allowable pixel distance threshold
    kernel_thresh = cv2.getStructuringElement(
        cv2.MORPH_RECT, (bound_thresh * 2 + 1, bound_thresh * 2 + 1)
    )
    dilated_boundary_a = cv2.dilate(boundary_a, kernel_thresh)
    dilated_boundary_b = cv2.dilate(boundary_b, kernel_thresh)

    # Compute matches within threshold
    matches_a = np.logical_and(boundary_a, dilated_boundary_b).sum()
    matches_b = np.logical_and(boundary_b, dilated_boundary_a).sum()

    precision = matches_a / boundary_a.sum() if boundary_a.sum() > 0 else 0.0
    recall = matches_b / boundary_b.sum() if boundary_b.sum() > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))
