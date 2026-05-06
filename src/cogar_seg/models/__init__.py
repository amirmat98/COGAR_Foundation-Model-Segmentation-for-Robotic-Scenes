"""Model adapters."""

from cogar_seg.models.registry import MODEL_BACKENDS, ModelBackend, get_model_backend
from cogar_seg.models.sam import (
    DeviceMode,
    load_sam_automatic_mask_generator,
    load_sam_model,
    load_sam_predictor,
    run_sam_box_prompt,
    run_sam_for_box,
    run_sam_for_point,
    select_device,
)

__all__ = [
    "DeviceMode",
    "MODEL_BACKENDS",
    "ModelBackend",
    "get_model_backend",
    "load_sam_automatic_mask_generator",
    "load_sam_model",
    "load_sam_predictor",
    "run_sam_box_prompt",
    "run_sam_for_box",
    "run_sam_for_point",
    "select_device",
]
