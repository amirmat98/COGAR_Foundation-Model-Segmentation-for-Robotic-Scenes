import numpy as np

from cogar_seg.metrics import compute_iou


def test_compute_iou_partial_overlap() -> None:
    mask_a = np.array([[1, 1], [0, 0]], dtype=bool)
    mask_b = np.array([[1, 0], [1, 0]], dtype=bool)

    assert compute_iou(mask_a, mask_b) == 1 / 3


def test_compute_iou_empty_union() -> None:
    mask_a = np.zeros((2, 2), dtype=bool)
    mask_b = np.zeros((2, 2), dtype=bool)

    assert compute_iou(mask_a, mask_b) == 0.0
