import numpy as np

from cogar_seg.metrics import compute_boundary_f1, compute_boundary_f1_ratio, compute_iou


def test_compute_iou_partial_overlap() -> None:
    mask_a = np.array([[1, 1], [0, 0]], dtype=bool)
    mask_b = np.array([[1, 0], [1, 0]], dtype=bool)

    assert compute_iou(mask_a, mask_b) == 1 / 3


def test_compute_iou_empty_union() -> None:
    mask_a = np.zeros((2, 2), dtype=bool)
    mask_b = np.zeros((2, 2), dtype=bool)

    assert compute_iou(mask_a, mask_b) == 0.0


def test_compute_iou_empty_union_custom_value() -> None:
    mask_a = np.zeros((2, 2), dtype=bool)
    mask_b = np.zeros((2, 2), dtype=bool)

    assert compute_iou(mask_a, mask_b, empty_value=1.0) == 1.0


def test_compute_boundary_f1_identical() -> None:
    mask_a = np.ones((10, 10), dtype=bool)
    mask_b = np.ones((10, 10), dtype=bool)

    assert compute_boundary_f1(mask_a, mask_b) == 1.0


def test_compute_boundary_f1_disjoint() -> None:
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[1:4, 1:4] = True

    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[6:9, 6:9] = True

    assert compute_boundary_f1(mask_a, mask_b) == 0.0


def test_compute_boundary_f1_ratio_identical() -> None:
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:5, 2:5] = True

    assert compute_boundary_f1_ratio(mask_a, mask_a) == 1.0
