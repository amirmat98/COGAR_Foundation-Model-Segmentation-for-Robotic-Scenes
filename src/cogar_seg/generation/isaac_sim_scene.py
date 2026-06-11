"""Isaac Sim Replicator generation for COGAR-IsaacSimRobotics-500.

The functions in this module are imported by a thin script entry point. Isaac
Sim and Replicator imports stay inside runtime functions so normal unit tests
and syntax checks can run on machines without Isaac Sim installed.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CHALLENGE_ORDER = [
    "reflective_metal",
    "transparent_glass",
    "partial_occlusion",
    "small_parts",
    "dynamic_scene",
]


@dataclass(frozen=True)
class IsaacSimGenerationRun:
    """Paths and counts from an Isaac Sim dataset generation run."""

    output_dir: Path
    raw_output_dir: Path
    metadata_dir: Path
    num_frames: int
    width: int
    height: int
    elapsed_seconds: float


def load_isaac_sim_config(path: str | Path) -> dict[str, Any]:
    """Load the Isaac Sim dataset configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Isaac Sim config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def _bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _challenge_sequence(final_plan: dict[str, int], num_frames: int) -> list[str]:
    sequence: list[str] = []
    for challenge in CHALLENGE_ORDER:
        count = int(final_plan.get(challenge, 0))
        sequence.extend([challenge] * count)

    if not sequence:
        for challenge, count in final_plan.items():
            sequence.extend([challenge] * int(count))

    if not sequence:
        sequence = CHALLENGE_ORDER.copy()

    while len(sequence) < num_frames:
        sequence.extend(sequence)

    return sequence[:num_frames]


def _split_for_frame(frame_index: int, num_frames: int) -> str:
    train_end = int(round(0.70 * num_frames))
    val_end = train_end + int(round(0.15 * num_frames))
    if frame_index < train_end:
        return "train"
    if frame_index < val_end:
        return "val"
    return "test"


def _challenge_flags(challenge: str) -> dict[str, bool]:
    return {
        "is_reflective": challenge == "reflective_metal",
        "is_transparent": challenge == "transparent_glass",
        "is_occluded": challenge == "partial_occlusion",
        "is_small_part": challenge == "small_parts",
        "is_dynamic": challenge == "dynamic_scene",
    }


def _random_pose(
    rng: random.Random,
    pos_min: list[float],
    pos_max: list[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position = (
        rng.uniform(pos_min[0], pos_max[0]),
        rng.uniform(pos_min[1], pos_max[1]),
        rng.uniform(pos_min[2], pos_max[2]),
    )
    rotation = (
        rng.uniform(-12.0, 12.0),
        rng.uniform(-12.0, 12.0),
        rng.uniform(0.0, 360.0),
    )
    return position, rotation


def _prepare_dirs(
    output_dir: Path,
    raw_output_dir: Path,
    clean: bool,
) -> tuple[Path, Path]:
    if clean and raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)

    metadata_dir = output_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return raw_output_dir, metadata_dir


def _write_text_metadata(
    output_dir: Path,
    metadata_dir: Path,
    dataset_name: str,
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    width: int,
    height: int,
    raw_output_dir: Path,
) -> None:
    frame_index_path = metadata_dir / "frame_index.csv"
    with frame_index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "frame_index",
                "file_name_hint",
                "split",
                "challenge_primary",
                "challenge_secondary",
                "is_reflective",
                "is_transparent",
                "is_occluded",
                "is_small_part",
                "is_dynamic",
                "image_width",
                "image_height",
                "simulation_environment",
                "writer",
                "raw_output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (metadata_dir / "categories.json").write_text(
        json.dumps(categories, indent=2), encoding="utf-8"
    )
    (metadata_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "images": len(rows),
                "image_width": width,
                "image_height": height,
                "simulation_environment": "Isaac Sim Replicator",
                "writer": "BasicWriter",
                "raw_output_dir": str(raw_output_dir),
                "challenge_counts": _count_values(
                    [str(row["challenge_primary"]) for row in rows]
                ),
                "status": "generated_raw_replicator_dataset",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        f"""# {dataset_name}

Complete Isaac Sim / Replicator synthetic robotic-scene dataset.

This dataset is generated separately from `data/cogar_sim_500_final/` so the
existing BlenderProc Task 1 result remains intact.

Expected contents:

```text
raw_replicator/final_500/
metadata/frame_index.csv
metadata/categories.json
metadata/dataset_summary.json
```

The Replicator `BasicWriter` output contains RGB images, semantic
segmentation, instance-id segmentation, and tight 2D boxes. The metadata CSV
records split labels and the primary robotics challenge for each frame.
""",
        encoding="utf-8",
    )


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _start_simulation_app(headless: bool, renderer: str):
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp  # type: ignore

    return SimulationApp(launch_config={"headless": headless, "renderer": renderer})


def _apply_transform(rep: Any, prim: Any, position: Any, rotation: Any, scale: Any) -> None:
    modify = rep.functional.modify
    if hasattr(modify, "pose"):
        try:
            modify.pose(
                prim,
                position_value=position,
                rotation_value=rotation,
                scale_value=scale,
            )
            return
        except TypeError:
            pass

    if hasattr(modify, "position"):
        modify.position(prim, position)
    if hasattr(modify, "rotation"):
        modify.rotation(prim, rotation)
    if hasattr(modify, "scale"):
        modify.scale(prim, scale)


def _apply_optional_material(rep: Any, prim: Any, category: str) -> None:
    """Best-effort visual material binding.

    Isaac Sim material APIs vary across versions. Failing to bind material hints
    should not fail dataset annotation generation, because semantic and instance
    labels remain the critical output.
    """
    if not hasattr(rep, "create") or not hasattr(rep.create, "material_omnipbr"):
        return

    try:
        if category == "metal_part":
            material = rep.create.material_omnipbr(
                diffuse=(0.75, 0.75, 0.78),
                metallic=1.0,
                roughness=0.08,
            )
        elif category == "glass_object":
            material = rep.create.material_omnipbr(
                diffuse=(0.60, 0.85, 1.0),
                roughness=0.02,
                opacity=0.35,
            )
        elif category == "screw":
            material = rep.create.material_omnipbr(
                diffuse=(0.50, 0.50, 0.52),
                metallic=1.0,
                roughness=0.18,
            )
        else:
            return

        with prim:
            rep.modify.material(material)
    except Exception as exc:  # pragma: no cover - Isaac-version-specific.
        print(f"[ISAAC][WARN] Material hint skipped for {category}: {exc}")


def generate_cogar_isaac_sim_500(
    config_path: str | Path = "configs/isaac_sim_dataset.yaml",
    num_frames: int | None = None,
    output_dir: str | Path | None = None,
    raw_dataset_name: str = "final_500",
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    rt_subframes: int | None = None,
    renderer: str | None = None,
    headless: bool | None = None,
    clean: bool = True,
    progress_every: int = 25,
) -> IsaacSimGenerationRun:
    """Generate a complete 500-frame Isaac Sim robotic scene dataset."""
    config = load_isaac_sim_config(config_path)
    dataset_cfg = config["dataset"]
    repl_cfg = config["replicator"]
    scene_cfg = config["scene"]

    resolved_output_dir = Path(output_dir or dataset_cfg["output_dir"]).resolve()
    resolved_raw_dir = resolved_output_dir / "raw_replicator" / raw_dataset_name
    if output_dir is None and dataset_cfg.get("raw_output_dir"):
        resolved_raw_dir = Path(dataset_cfg["raw_output_dir"]).resolve()

    total_frames = int(num_frames or dataset_cfg.get("final_images", 500))
    image_width = int(width or dataset_cfg.get("image_width", 640))
    image_height = int(height or dataset_cfg.get("image_height", 480))
    seed_value = int(seed if seed is not None else repl_cfg.get("seed", 42))
    subframes = int(rt_subframes or repl_cfg.get("rt_subframes", 16))
    renderer_value = str(renderer or repl_cfg.get("renderer", "RayTracedLighting"))
    headless_value = (
        _bool_value(repl_cfg.get("headless", True)) if headless is None else headless
    )
    print(
        "[ISAAC] Starting generation "
        f"frames={total_frames} size={image_width}x{image_height} "
        f"renderer={renderer_value} subframes={subframes} "
        f"output={resolved_output_dir}",
        flush=True,
    )

    raw_dir, metadata_dir = _prepare_dirs(
        output_dir=resolved_output_dir,
        raw_output_dir=resolved_raw_dir,
        clean=clean,
    )

    rng = random.Random(seed_value)
    categories = list(config["categories"])
    categories_by_name = {str(cat["name"]): cat for cat in categories}
    challenges = _challenge_sequence(config["final_plan"], total_frames)
    pos_min = list(scene_cfg["object_position_min"])
    pos_max = list(scene_cfg["object_position_max"])
    enable_material_hints = _bool_value(scene_cfg.get("enable_material_hints", True))

    simulation_app = _start_simulation_app(
        headless=headless_value,
        renderer=renderer_value,
    )
    print("[ISAAC] SimulationApp started", flush=True)
    start = time.perf_counter()

    try:
        print("[ISAAC] Importing Replicator modules", flush=True)
        import carb.settings
        import omni.replicator.core as rep
        import omni.usd

        print("[ISAAC] Creating stage and scene objects", flush=True)
        omni.usd.get_context().new_stage()
        rep.orchestrator.set_capture_on_play(False)
        rep.set_global_seed(seed_value)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        rep.functional.create.xform(name="World")
        rep.functional.create.xform(parent="/World", name="Objects")

        try:
            rep.create.light(light_type="Dome", intensity=650)
        except Exception as exc:  # pragma: no cover - Isaac-version-specific.
            print(f"[ISAAC][WARN] Dome light creation skipped: {exc}")

        table = rep.functional.create.cube(parent="/World", name="WorkTable")
        _apply_transform(
            rep,
            table,
            position=(0.0, 0.0, -0.06),
            rotation=(0.0, 0.0, 0.0),
            scale=tuple(scene_cfg.get("table_size", [2.8, 1.8, 0.08])),
        )
        rep.functional.modify.semantics(table, {"class": "table"}, mode="add")

        objects: list[dict[str, Any]] = []
        for spec in config["object_specs"]:
            category = str(spec["category"])
            if category not in categories_by_name:
                raise ValueError(f"Object spec has unknown category: {category}")
            for instance_idx in range(int(spec["count"])):
                prim = rep.functional.create.cube(
                    parent="/World/Objects",
                    name=f"{category}_{instance_idx:02d}",
                )
                scale = tuple(float(v) for v in spec["scale"])
                rep.functional.modify.semantics(prim, {"class": category}, mode="add")
                _apply_transform(
                    rep,
                    prim,
                    position=(0.0, 0.0, 0.2),
                    rotation=(0.0, 0.0, 0.0),
                    scale=scale,
                )
                if enable_material_hints:
                    _apply_optional_material(rep, prim, category)
                objects.append({"prim": prim, "category": category, "scale": scale})

        camera = rep.functional.create.camera(
            position=tuple(scene_cfg["camera_position"]),
            look_at=tuple(scene_cfg["camera_look_at"]),
            parent="/World",
            name="Camera",
        )
        render_product = rep.create.render_product(
            camera,
            (image_width, image_height),
            name="cogar_isaac_sim_500",
        )

        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(raw_dir))
        writer = rep.writers.get("BasicWriter")
        writer.initialize(
            backend=backend,
            rgb=True,
            semantic_segmentation=True,
            colorize_semantic_segmentation=True,
            instance_id_segmentation=True,
            colorize_instance_id_segmentation=True,
            bounding_box_2d_tight=True,
        )
        writer.attach(render_product)
        print("[ISAAC] Writer attached; starting frame capture", flush=True)

        rows: list[dict[str, Any]] = []
        for frame_idx, challenge in enumerate(challenges):
            for obj in objects:
                position, rotation = _random_pose(rng, pos_min, pos_max)
                category = obj["category"]

                if challenge == "small_parts" and category in {"screw", "connector"}:
                    position = (
                        rng.uniform(-0.45, 0.45),
                        rng.uniform(-0.35, 0.35),
                        rng.uniform(0.03, 0.08),
                    )
                elif challenge == "partial_occlusion" and category == "robot_gripper":
                    position = (
                        rng.uniform(-0.25, 0.25),
                        rng.uniform(-0.15, 0.20),
                        rng.uniform(0.30, 0.55),
                    )
                elif challenge == "dynamic_scene":
                    drift = 0.25 * ((frame_idx % 10) - 5) / 5.0
                    position = (position[0] + drift, position[1] - drift, position[2])

                _apply_transform(rep, obj["prim"], position, rotation, obj["scale"])

            if frame_idx == 0 or (frame_idx + 1) % max(1, progress_every) == 0:
                elapsed = time.perf_counter() - start
                done = frame_idx + 1
                fps = done / elapsed if elapsed > 0 else 0.0
                remaining = (total_frames - done) / fps if fps > 0 else 0.0
                print(
                    "[ISAAC] "
                    f"{done}/{total_frames} challenge={challenge} "
                    f"elapsed_min={elapsed / 60.0:.1f} eta_min={remaining / 60.0:.1f}",
                    flush=True,
                )

            rep.orchestrator.step(rt_subframes=subframes)
            flags = _challenge_flags(challenge)
            rows.append(
                {
                    "image_id": f"img_{frame_idx:06d}",
                    "frame_index": frame_idx,
                    "file_name_hint": f"{frame_idx:06d}",
                    "split": _split_for_frame(frame_idx, total_frames),
                    "challenge_primary": challenge,
                    "challenge_secondary": "",
                    "image_width": image_width,
                    "image_height": image_height,
                    "simulation_environment": "Isaac Sim Replicator",
                    "writer": "BasicWriter",
                    "raw_output_dir": str(raw_dir),
                    **flags,
                }
            )

        rep.orchestrator.wait_until_complete()
        if hasattr(writer, "detach"):
            writer.detach()
        if hasattr(render_product, "destroy"):
            render_product.destroy()

        elapsed_total = time.perf_counter() - start
        _write_text_metadata(
            output_dir=resolved_output_dir,
            metadata_dir=metadata_dir,
            dataset_name=str(dataset_cfg["name"]),
            rows=rows,
            categories=categories,
            width=image_width,
            height=image_height,
            raw_output_dir=raw_dir,
        )

        print(f"[ISAAC] Dataset root: {resolved_output_dir}", flush=True)
        print(f"[ISAAC] Raw Replicator output: {raw_dir}", flush=True)
        print(f"[ISAAC] Metadata: {metadata_dir}", flush=True)
        print(f"[ISAAC] Frames: {total_frames}", flush=True)
        print(f"[ISAAC] Elapsed seconds: {elapsed_total:.1f}", flush=True)

        return IsaacSimGenerationRun(
            output_dir=resolved_output_dir,
            raw_output_dir=raw_dir,
            metadata_dir=metadata_dir,
            num_frames=total_frames,
            width=image_width,
            height=image_height,
            elapsed_seconds=elapsed_total,
        )
    finally:
        simulation_app.close()
