"""Backward-compatible imports for OCID dataset helpers.

New code should import from ``cogar_seg.datasets.ocid``.
"""

from cogar_seg.datasets.ocid import (  # noqa: F401
    compute_object_properties,
    create_image_index,
    create_object_index,
    get_rgb_label_dirs,
)

__all__ = [
    "compute_object_properties",
    "create_image_index",
    "create_object_index",
    "get_rgb_label_dirs",
]
