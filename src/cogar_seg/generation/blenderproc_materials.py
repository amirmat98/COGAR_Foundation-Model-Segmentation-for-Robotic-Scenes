"""BlenderProc material and renderer-environment helpers."""

from __future__ import annotations

import random


def ensure_world_background() -> None:
    """Restore a valid world background after BlenderProc cleanup."""
    import blenderproc as bproc
    import bpy

    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    bpy.context.scene.world.use_nodes = True
    bproc.renderer.set_world_background([0.03, 0.03, 0.03], strength=0.8)


def make_material(
    name: str,
    color: list[float],
    metallic: float = 0.0,
    roughness: float = 0.5,
    alpha: float = 1.0,
):
    """Create a BlenderProc material with principled-shader defaults."""
    import blenderproc as bproc

    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Metallic", metallic)
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Alpha", alpha)
    return mat


def create_material_pool() -> dict[str, object]:
    """Create the randomized material palette used by COGAR-Sim scenes."""
    table_colors = [
        [0.72, 0.65, 0.55, 1.0],
        [0.55, 0.48, 0.40, 1.0],
        [0.35, 0.35, 0.35, 1.0],
        [0.80, 0.76, 0.68, 1.0],
        [0.42, 0.30, 0.18, 1.0],
    ]
    plastic_colors = [
        [0.05, 0.20, 0.85, 1.0],
        [0.85, 0.10, 0.08, 1.0],
        [0.10, 0.55, 0.18, 1.0],
        [0.95, 0.75, 0.10, 1.0],
        [0.75, 0.15, 0.85, 1.0],
        [0.05, 0.05, 0.05, 1.0],
    ]

    return {
        "table": make_material(
            "mat_table",
            random.choice(table_colors),
            metallic=0.0,
            roughness=random.uniform(0.55, 0.95),
        ),
        "wall": make_material(
            "mat_wall",
            [random.uniform(0.35, 0.75)] * 3 + [1.0],
            metallic=0.0,
            roughness=random.uniform(0.6, 1.0),
        ),
        "robot": make_material(
            "mat_robot_dark",
            [0.015, 0.015, 0.018, 1.0],
            metallic=random.uniform(0.0, 0.35),
            roughness=random.uniform(0.25, 0.65),
        ),
        "metal": make_material(
            "mat_reflective_metal",
            [random.uniform(0.55, 0.9)] * 3 + [1.0],
            metallic=random.uniform(0.85, 1.0),
            roughness=random.uniform(0.02, 0.18),
        ),
        "glass": make_material(
            "mat_transparent_glass",
            [0.65, 0.9, 1.0, random.uniform(0.18, 0.42)],
            metallic=0.0,
            roughness=random.uniform(0.0, 0.08),
            alpha=random.uniform(0.18, 0.42),
        ),
        "plastic": make_material(
            "mat_plastic",
            random.choice(plastic_colors),
            metallic=0.0,
            roughness=random.uniform(0.3, 0.8),
        ),
        "plastic2": make_material(
            "mat_plastic2",
            random.choice(plastic_colors),
            metallic=0.0,
            roughness=random.uniform(0.3, 0.8),
        ),
        "cable": make_material(
            "mat_black_cable",
            [0.005, 0.005, 0.005, 1.0],
            metallic=0.0,
            roughness=random.uniform(0.35, 0.75),
        ),
        "cardboard": make_material(
            "mat_cardboard",
            [0.58, 0.40, 0.20, 1.0],
            metallic=0.0,
            roughness=random.uniform(0.75, 1.0),
        ),
    }
