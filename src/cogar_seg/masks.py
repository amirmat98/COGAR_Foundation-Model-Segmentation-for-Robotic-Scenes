"""Backward-compatible imports for OCID mask-index helpers.

New code should import from ``cogar_seg.datasets.ocid``.
"""

from cogar_seg.datasets.ocid import (  # noqa: F401
    OCID_IMAGE_HEIGHT as IMAGE_HEIGHT,
    OCID_IMAGE_WIDTH as IMAGE_WIDTH,
    export_binary_gt_masks,
    filter_object_index,
    make_binary_mask_filename,
)

IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT

__all__ = [
    "IMAGE_AREA",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "export_binary_gt_masks",
    "filter_object_index",
    "make_binary_mask_filename",
]
