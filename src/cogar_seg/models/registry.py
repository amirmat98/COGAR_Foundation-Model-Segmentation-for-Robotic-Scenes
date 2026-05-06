"""Model backend registry for evaluation entry points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelBackend:
    """Minimal metadata for a supported segmentation backend."""

    name: str
    supports_box_prompt: bool
    supports_point_prompt: bool
    supports_automatic_masks: bool


MODEL_BACKENDS = {
    "sam": ModelBackend(
        name="sam",
        supports_box_prompt=True,
        supports_point_prompt=True,
        supports_automatic_masks=True,
    ),
    "sam2": ModelBackend(
        name="sam2",
        supports_box_prompt=True,
        supports_point_prompt=True,
        supports_automatic_masks=True,
    ),
    "fastsam": ModelBackend(
        name="fastsam",
        supports_box_prompt=True,
        supports_point_prompt=True,
        supports_automatic_masks=True,
    ),
}


def get_model_backend(name: str) -> ModelBackend:
    """Return backend metadata by name."""
    try:
        return MODEL_BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model backend: {name}") from exc
