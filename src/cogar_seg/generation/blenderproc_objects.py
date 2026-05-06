"""BlenderProc object creation helpers for COGAR scenes."""

from __future__ import annotations

import random
from typing import Any


def tag_object(obj: Any, category: dict[str, Any], name_suffix: str = "") -> Any:
    """Attach semantic metadata expected by BlenderProc COCO export."""
    del name_suffix
    obj.set_name(category["name"])
    obj.set_cp("category_id", category["id"])
    obj.set_cp("supercategory", category["supercategory"])
    return obj


def create_primitive(
    category: dict[str, Any],
    primitive_type: str,
    location: list[float],
    scale: list[float],
    material: Any,
    rotation: list[float] | None = None,
    name_suffix: str = "",
) -> Any:
    """Create and semantically tag a BlenderProc primitive."""
    import blenderproc as bproc

    obj = bproc.object.create_primitive(primitive_type)
    obj.set_location(location)
    obj.set_scale(scale)

    if rotation is not None:
        obj.set_rotation_euler(rotation)

    obj.replace_materials(material)
    tag_object(obj, category, name_suffix=name_suffix)
    return obj


def category_lookup(categories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index category dictionaries by semantic category name."""
    return {cat["name"]: cat for cat in categories}


def random_xy(
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-0.65, 0.65),
) -> tuple[float, float]:
    """Sample a random tabletop XY position."""
    return random.uniform(*xlim), random.uniform(*ylim)
