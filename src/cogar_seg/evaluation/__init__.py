"""Evaluation workflows."""

from cogar_seg.evaluation.sam_box_eval import (
    BatchSamBoxRun,
    SingleSamBoxResult,
    run_batch_sam_box,
    run_single_sam_box,
)
from cogar_seg.evaluation.sam_point_eval import (
    BatchSamPointRun,
    SingleSamPointResult,
    run_batch_sam_point,
    run_single_sam_point,
)

__all__ = [
    "BatchSamBoxRun",
    "BatchSamPointRun",
    "SingleSamBoxResult",
    "SingleSamPointResult",
    "run_batch_sam_box",
    "run_batch_sam_point",
    "run_single_sam_box",
    "run_single_sam_point",
]
