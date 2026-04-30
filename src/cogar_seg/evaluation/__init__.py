"""Evaluation workflows."""

from cogar_seg.evaluation.sam_box_eval import (
    BatchSamBoxRun,
    SingleSamBoxResult,
    run_batch_sam_box,
    run_single_sam_box,
)

__all__ = [
    "BatchSamBoxRun",
    "SingleSamBoxResult",
    "run_batch_sam_box",
    "run_single_sam_box",
]
