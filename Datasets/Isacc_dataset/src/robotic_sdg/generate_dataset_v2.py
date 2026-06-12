"""Generate the official Unitree G1 robotic segmentation dataset.

This generator targets a robotic workcell with the official Unitree G1 USD
asset, richer materials, stronger clutter, partial occlusion, transparent
panels, reflective parts, small screws/connectors, cables, and dynamic
conveyor-like objects.

Run with Isaac Sim Python, for example:

    /path/to/isaac-sim/python.sh src/robotic_sdg/generate_dataset_v2.py \
      --config configs/dataset_config_v3_official_g1.json \
      --output data/robotic_sdg_v3_official_g1_1000 \
      --robot-mode official
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from isaacsim import SimulationApp


TABLE_TOP_Z = 0.62
HIDE_TRANSLATE = (4.0, 4.0, 0.2)
OFFICIAL_G1_ROOT_Z = 0.82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_config_v3_official_g1.json", help="Path to dataset config JSON.")
    parser.add_argument("--output", default="data/robotic_sdg_v3_official_g1_1000", help="Output dataset directory.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional smoke-test limit after scenario shuffling.")
    parser.add_argument(
        "--robot-mode",
        choices=["official"],
        default="official",
        help="Require the official Unitree G1 USD asset.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["shuffled", "coverage"],
        default="shuffled",
        help="Use coverage with --max-images to preview distinct scenario families.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for scenario in config["scenarios"]:
        for _ in range(int(scenario["count"])):
            scenarios.append({"name": scenario["name"], "challenge_tags": list(scenario.get("challenge_tags", []))})

    target = int(config["dataset"]["num_images"])
    if len(scenarios) != target:
        raise ValueError(f"Scenario counts total {len(scenarios)} but dataset.num_images is {target}")
    return scenarios


def select_scenarios(scenarios: list[dict[str, Any]], max_images: int, sample_mode: str) -> list[dict[str, Any]]:
    if sample_mode == "shuffled":
        return scenarios[:max_images]

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario["name"] in seen:
            continue
        picked.append(scenario)
        seen.add(scenario["name"])
        if len(picked) == max_images:
            return picked

    for scenario in scenarios:
        picked.append(scenario)
        if len(picked) == max_images:
            return picked
    return picked


def launch_sim(config: dict[str, Any]) -> SimulationApp:
    render_config = config.get("render", {})
    return SimulationApp(launch_config={"headless": bool(render_config.get("headless", True))})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation_app = launch_sim(config)

    import carb.settings
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    add_update_semantics = load_semantics_helper()

    rng = random.Random(int(config["dataset"]["seed"]))
    scenarios = expand_scenarios(config)
    rng.shuffle(scenarios)
    if args.max_images is not None:
        scenarios = select_scenarios(scenarios, args.max_images, args.sample_mode)

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    rep.orchestrator.set_capture_on_play(False)
    if hasattr(rep, "set_global_seed"):
        rep.set_global_seed(int(config["dataset"]["seed"]))

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/dlss/execMode", int(config.get("render", {}).get("dlss_quality_mode", 2)))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    materials = create_materials(stage, UsdShade, Sdf, Gf)
    records, lights, robot_info = build_scene(stage, config, rng, materials, add_update_semantics, rep, UsdLux, args.robot_mode)

    width, height = config["dataset"]["resolution"]
    camera = rep.functional.create.camera(position=(3.0, -3.4, 2.0), look_at=(0, 0, 0.75), parent="/World", name="Camera")
    render_product = rep.create.render_product(camera, (int(width), int(height)), name="RoboticDatasetRenderProductV2")

    writer = make_writer(rep, output_dir / "isaac", config.get("writer", {}))
    writer.attach(render_product)

    manifest_path = output_dir / "manifest.jsonl"
    rt_subframes = int(config.get("render", {}).get("rt_subframes", 12))

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for frame_id, scenario in enumerate(scenarios):
            frame_meta = randomize_frame(
                frame_id=frame_id,
                scenario=scenario,
                rng=rng,
                records=records,
                lights=lights,
                camera=camera,
                rep=rep,
            )
            frame_meta["frame_id"] = frame_id
            frame_meta["dataset_name"] = config["dataset"]["name"]
            frame_meta["robot_platform"] = config["robot"]["platform"]
            frame_meta["robot_asset"] = robot_info
            frame_meta["source_environment"] = "isaac_sim"
            manifest.write(json.dumps(frame_meta, sort_keys=True) + "\n")
            print(f"[{frame_id + 1:04d}/{len(scenarios):04d}] {scenario['name']}", flush=True)
            simulation_app.update()
            rep.orchestrator.step(rt_subframes=rt_subframes)
            rep.orchestrator.wait_until_complete()

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()
    simulation_app.close()


def load_semantics_helper():
    try:
        from isaacsim.core.utils.semantics import add_update_semantics

        return add_update_semantics
    except Exception:
        try:
            from omni.isaac.core.utils.semantics import add_update_semantics

            return add_update_semantics
        except Exception:
            return None


def create_materials(stage, UsdShade, Sdf, Gf) -> dict[str, Any]:
    return {
        "robot": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/robot", (0.26, 0.29, 0.32), 0.0, 0.35, 1.0),
        "floor": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/floor", (0.42, 0.43, 0.41), 0.0, 0.60, 1.0),
        "wall": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/wall", (0.68, 0.69, 0.66), 0.0, 0.68, 1.0),
        "workbench": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/workbench", (0.38, 0.36, 0.32), 0.0, 0.45, 1.0),
        "bench_top": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/bench_top", (0.58, 0.56, 0.51), 0.0, 0.36, 1.0),
        "metal_polished": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/metal_polished", (0.82, 0.82, 0.78), 1.0, 0.04, 1.0),
        "metal_brushed": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/metal_brushed", (0.64, 0.65, 0.62), 1.0, 0.16, 1.0),
        "metal_dark": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/metal_dark", (0.19, 0.20, 0.21), 1.0, 0.22, 1.0),
        "glass_clear": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/glass_clear", (0.78, 0.92, 1.0), 0.0, 0.01, 0.18),
        "glass_green": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/glass_green", (0.62, 0.98, 0.86), 0.0, 0.02, 0.28),
        "glass_smoked": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/glass_smoked", (0.36, 0.46, 0.52), 0.0, 0.04, 0.36),
        "screw": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/screw", (0.50, 0.49, 0.45), 1.0, 0.12, 1.0),
        "connector_blue": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/connector_blue", (0.02, 0.22, 0.82), 0.0, 0.32, 1.0),
        "connector_orange": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/connector_orange", (0.95, 0.38, 0.06), 0.0, 0.34, 1.0),
        "connector_green": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/connector_green", (0.08, 0.48, 0.24), 0.0, 0.36, 1.0),
        "cable_black": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/cable_black", (0.01, 0.012, 0.014), 0.0, 0.50, 1.0),
        "cable_yellow": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/cable_yellow", (0.95, 0.75, 0.04), 0.0, 0.42, 1.0),
        "moving_red": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/moving_red", (0.92, 0.12, 0.06), 0.0, 0.28, 1.0),
        "moving_teal": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/moving_teal", (0.0, 0.62, 0.72), 0.0, 0.30, 1.0),
        "occluder_cardboard": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/occluder_cardboard", (0.55, 0.46, 0.34), 0.0, 0.74, 1.0),
        "occluder_dark": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/occluder_dark", (0.16, 0.17, 0.18), 0.0, 0.66, 1.0),
        "tool_yellow": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/tool_yellow", (0.92, 0.66, 0.04), 0.0, 0.38, 1.0),
        "tool_black": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/tool_black", (0.02, 0.025, 0.03), 0.0, 0.44, 1.0),
        "bin_blue": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/bin_blue", (0.06, 0.30, 0.78), 0.0, 0.48, 1.0),
        "bin_green": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/bin_green", (0.08, 0.45, 0.26), 0.0, 0.52, 1.0),
        "bin_gray": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/bin_gray", (0.34, 0.38, 0.40), 0.0, 0.55, 1.0),
        "pcb_green": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/pcb_green", (0.02, 0.34, 0.18), 0.0, 0.30, 1.0),
        "pcb_dark": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/pcb_dark", (0.01, 0.12, 0.08), 0.0, 0.35, 1.0),
        "rubber_black": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/rubber_black", (0.006, 0.007, 0.008), 0.0, 0.72, 1.0),
        "rubber_gray": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/rubber_gray", (0.08, 0.09, 0.09), 0.0, 0.66, 1.0),
        "sensor_body": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/sensor_body", (0.08, 0.10, 0.12), 0.0, 0.28, 1.0),
        "sensor_lens": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/sensor_lens", (0.02, 0.08, 0.11), 0.0, 0.03, 0.72),
        "marker_white": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/marker_white", (0.92, 0.91, 0.84), 0.0, 0.46, 1.0),
        "marker_black": make_preview_surface(stage, UsdShade, Sdf, Gf, "/World/Materials/marker_black", (0.015, 0.015, 0.015), 0.0, 0.48, 1.0),
    }


def make_preview_surface(stage, UsdShade, Sdf, Gf, path: str, color: tuple[float, float, float], metallic: float, roughness: float, opacity: float):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI(prim).Bind(material)


def set_semantics(prim, label: str, add_update_semantics, rep) -> None:
    if add_update_semantics is not None:
        add_update_semantics(prim, label, type_label="class")
        return
    rep.functional.modify.semantics(prim, {"class": label}, mode="add")


def define_prim(stage, kind: str, path: str):
    from pxr import UsdGeom

    if kind == "cube":
        return UsdGeom.Cube.Define(stage, path).GetPrim()
    if kind == "sphere":
        return UsdGeom.Sphere.Define(stage, path).GetPrim()
    if kind == "cylinder":
        return UsdGeom.Cylinder.Define(stage, path).GetPrim()
    raise ValueError(f"Unsupported primitive kind: {kind}")


def set_transform_ops(prim, translate, rotate, scale):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()
    scale_op = xform.AddScaleOp()
    translate_op.Set(Gf.Vec3d(*translate))
    rotate_op.Set(Gf.Vec3f(*rotate))
    scale_op.Set(Gf.Vec3f(*scale))
    return {"translate": translate_op, "rotate": rotate_op, "scale": scale_op}


def make_object(
    stage,
    records: dict[str, list[dict[str, Any]]],
    group: str,
    kind: str,
    path: str,
    label: str,
    material,
    translate,
    rotate,
    scale,
    add_update_semantics,
    rep,
    material_options=None,
):
    prim = define_prim(stage, kind, path)
    ops = set_transform_ops(prim, translate, rotate, scale)
    bind_material(prim, material)
    set_semantics(prim, label, add_update_semantics, rep)
    record = {
        "path": path,
        "label": label,
        "kind": kind,
        "group": group,
        "prim": prim,
        "ops": ops,
        "base_scale": scale,
        "material_options": material_options or [material],
    }
    records.setdefault(group, []).append(record)
    records.setdefault("all", []).append(record)
    return record


def build_scene(stage, config, rng, materials, add_update_semantics, rep, UsdLux, robot_mode: str):
    from pxr import Gf, UsdGeom

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Lights")
    records: dict[str, list[dict[str, Any]]] = {
        "all": [],
        "fixed": [],
        "metal": [],
        "glass": [],
        "screws": [],
        "connectors": [],
        "cables": [],
        "moving": [],
        "occluders": [],
        "tools": [],
        "bins": [],
        "pcbs": [],
        "rubber": [],
        "sensors": [],
        "markers": [],
        "robot": [],
    }

    lights = create_lights(stage, UsdLux, Gf)
    add_fixed_workcell(stage, records, materials, add_update_semantics, rep)

    robot_records, robot_info = add_robot(stage, config, records, materials, add_update_semantics, rep, robot_mode)
    for robot_record in robot_records:
        if robot_record not in records["robot"]:
            records["robot"].append(robot_record)
        if robot_record not in records["all"]:
            records["all"].append(robot_record)

    for idx in range(12):
        kind = "cube" if idx % 3 else "cylinder"
        scale = (0.20, 0.045, 0.018) if kind == "cube" else (0.026, 0.026, 0.24)
        record = make_object(
            stage,
            records,
            "metal",
            kind,
            f"/World/Parts/Metal_{idx:02d}",
            "metal_part",
            materials["metal_polished"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            scale,
            add_update_semantics,
            rep,
            [materials["metal_polished"], materials["metal_brushed"], materials["metal_dark"]],
        )
        record["object_role"] = "plate" if kind == "cube" else "rod"

    for idx in range(10):
        kind = "cube" if idx < 6 else "cylinder"
        scale = (0.22, 0.010, 0.16) if kind == "cube" else (0.055, 0.055, 0.12)
        record = make_object(
            stage,
            records,
            "glass",
            kind,
            f"/World/Parts/Glass_{idx:02d}",
            "transparent_glass",
            materials["glass_clear"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            scale,
            add_update_semantics,
            rep,
            [materials["glass_clear"], materials["glass_green"], materials["glass_smoked"]],
        )
        record["object_role"] = "pane" if kind == "cube" else "cup_like_cylinder"

    for idx in range(28):
        make_object(
            stage,
            records,
            "screws",
            "cylinder",
            f"/World/Parts/Screw_{idx:02d}",
            "screw",
            materials["screw"],
            HIDE_TRANSLATE,
            (90, 0, 0),
            (0.010, 0.010, 0.042),
            add_update_semantics,
            rep,
            [materials["screw"], materials["metal_dark"]],
        )

    for idx in range(16):
        make_object(
            stage,
            records,
            "connectors",
            "cube",
            f"/World/Parts/Connector_{idx:02d}",
            "connector",
            materials["connector_blue"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.045, 0.032, 0.025),
            add_update_semantics,
            rep,
            [materials["connector_blue"], materials["connector_orange"], materials["connector_green"]],
        )

    for idx in range(10):
        make_object(
            stage,
            records,
            "cables",
            "cylinder",
            f"/World/Parts/Cable_{idx:02d}",
            "cable",
            materials["cable_black"],
            HIDE_TRANSLATE,
            (90, 0, 0),
            (0.008, 0.008, 0.34),
            add_update_semantics,
            rep,
            [materials["cable_black"], materials["cable_yellow"]],
        )

    for idx in range(5):
        make_object(
            stage,
            records,
            "moving",
            "sphere" if idx % 2 == 0 else "cube",
            f"/World/Parts/MovingObject_{idx:02d}",
            "moving_object",
            materials["moving_red"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.060, 0.060, 0.060),
            add_update_semantics,
            rep,
            [materials["moving_red"], materials["moving_teal"]],
        )

    for idx in range(6):
        make_object(
            stage,
            records,
            "occluders",
            "cube",
            f"/World/Parts/Occluder_{idx:02d}",
            "occluder",
            materials["occluder_cardboard"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.16, 0.035, 0.18),
            add_update_semantics,
            rep,
            [materials["occluder_cardboard"], materials["occluder_dark"]],
        )

    for idx in range(6):
        make_object(
            stage,
            records,
            "tools",
            "cube" if idx % 2 == 0 else "cylinder",
            f"/World/Parts/Tool_{idx:02d}",
            "tool",
            materials["tool_yellow"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.16, 0.028, 0.026) if idx % 2 == 0 else (0.018, 0.018, 0.22),
            add_update_semantics,
            rep,
            [materials["tool_yellow"], materials["tool_black"], materials["metal_dark"]],
        )

    for idx in range(5):
        make_object(
            stage,
            records,
            "bins",
            "cube",
            f"/World/Parts/StorageBin_{idx:02d}",
            "storage_bin",
            materials["bin_blue"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.18, 0.13, 0.075),
            add_update_semantics,
            rep,
            [materials["bin_blue"], materials["bin_green"], materials["bin_gray"]],
        )

    for idx in range(10):
        record = make_object(
            stage,
            records,
            "pcbs",
            "cube",
            f"/World/Parts/Pcb_{idx:02d}",
            "pcb",
            materials["pcb_green"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.10, 0.065, 0.006),
            add_update_semantics,
            rep,
            [materials["pcb_green"], materials["pcb_dark"]],
        )
        record["object_role"] = "electronics_board"

    for idx in range(10):
        record = make_object(
            stage,
            records,
            "rubber",
            "cylinder" if idx % 2 == 0 else "cube",
            f"/World/Parts/RubberPart_{idx:02d}",
            "rubber_part",
            materials["rubber_black"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.030, 0.030, 0.018) if idx % 2 == 0 else (0.055, 0.026, 0.018),
            add_update_semantics,
            rep,
            [materials["rubber_black"], materials["rubber_gray"]],
        )
        record["object_role"] = "seal_or_gasket"

    for idx in range(8):
        record = make_object(
            stage,
            records,
            "sensors",
            "cube" if idx % 3 else "sphere",
            f"/World/Parts/SensorModule_{idx:02d}",
            "sensor_module",
            materials["sensor_body"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.055, 0.040, 0.026) if idx % 3 else (0.036, 0.036, 0.036),
            add_update_semantics,
            rep,
            [materials["sensor_body"], materials["sensor_lens"]],
        )
        record["object_role"] = "camera_or_lidar_module"

    for idx in range(12):
        record = make_object(
            stage,
            records,
            "markers",
            "cube",
            f"/World/Parts/FiducialMarker_{idx:02d}",
            "fiducial_marker",
            materials["marker_white"],
            HIDE_TRANSLATE,
            (0, 0, 0),
            (0.038, 0.038, 0.004),
            add_update_semantics,
            rep,
            [materials["marker_white"], materials["marker_black"]],
        )
        record["object_role"] = "fiducial_or_label"

    for record in records["all"]:
        if record["group"] not in {"fixed", "robot"}:
            record["ops"]["translate"].Set(Gf.Vec3d(*HIDE_TRANSLATE))

    return records, lights, robot_info


def create_lights(stage, UsdLux, Gf) -> dict[str, Any]:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome_intensity = dome.CreateIntensityAttr(450)

    key = UsdLux.RectLight.Define(stage, "/World/Lights/KeySoftbox")
    key_intensity = key.CreateIntensityAttr(900)
    key.CreateWidthAttr(3.2)
    key.CreateHeightAttr(2.0)
    set_transform_ops(key.GetPrim(), (0.0, -2.4, 3.2), (-55, 0, 0), (1, 1, 1))

    rim = UsdLux.RectLight.Define(stage, "/World/Lights/RimStrip")
    rim_intensity = rim.CreateIntensityAttr(300)
    rim.CreateWidthAttr(1.2)
    rim.CreateHeightAttr(3.0)
    set_transform_ops(rim.GetPrim(), (-2.1, 1.2, 2.2), (-20, 0, -55), (1, 1, 1))

    return {
        "dome_intensity": dome_intensity,
        "key_intensity": key_intensity,
        "rim_intensity": rim_intensity,
    }


def add_fixed_workcell(stage, records, materials, add_update_semantics, rep) -> None:
    fixed = [
        ("cube", "/World/Room/Floor", "background_fixture", materials["floor"], (0, 0, -0.04), (0, 0, 0), (3.8, 3.2, 0.04)),
        ("cube", "/World/Room/BackWall", "background_fixture", materials["wall"], (0, 1.55, 1.10), (0, 0, 0), (3.8, 0.04, 1.10)),
        ("cube", "/World/Room/LeftWall", "background_fixture", materials["wall"], (-1.9, 0, 1.10), (0, 0, 0), (0.04, 3.2, 1.10)),
        ("cube", "/World/Workbench/Base", "workbench", materials["workbench"], (0.10, 0.02, 0.32), (0, 0, 0), (1.55, 1.02, 0.32)),
        ("cube", "/World/Workbench/Top", "workbench", materials["bench_top"], (0.10, 0.02, 0.56), (0, 0, 0), (1.62, 1.10, 0.06)),
        ("cube", "/World/Workbench/Conveyor", "workbench", materials["metal_dark"], (0.18, -0.56, 0.66), (0, 0, 0), (1.25, 0.12, 0.035)),
        ("cube", "/World/Fixtures/PegBoard", "background_fixture", materials["wall"], (0.10, 1.18, 1.05), (0, 0, 0), (1.35, 0.035, 0.52)),
        ("cube", "/World/Fixtures/ShelfTop", "background_fixture", materials["metal_dark"], (0.15, 1.03, 1.42), (0, 0, 0), (1.15, 0.09, 0.035)),
        ("cube", "/World/Fixtures/ShelfMid", "background_fixture", materials["metal_dark"], (0.15, 1.03, 1.08), (0, 0, 0), (1.15, 0.08, 0.030)),
        ("cube", "/World/Fixtures/ShelfLeftPost", "background_fixture", materials["metal_dark"], (-1.02, 1.03, 1.05), (0, 0, 0), (0.035, 0.045, 0.58)),
        ("cube", "/World/Fixtures/ShelfRightPost", "background_fixture", materials["metal_dark"], (1.32, 1.03, 1.05), (0, 0, 0), (0.035, 0.045, 0.58)),
        ("cube", "/World/Fixtures/CalibrationPanel", "background_fixture", materials["glass_smoked"], (1.18, 0.82, 0.91), (0, 0, -18), (0.22, 0.014, 0.18)),
    ]
    for kind, path, label, material, translate, rotate, scale in fixed:
        make_object(stage, records, "fixed", kind, path, label, material, translate, rotate, scale, add_update_semantics, rep)

    for idx, x in enumerate([-0.68, -0.28, 0.12, 0.52, 0.92]):
        make_object(
            stage,
            records,
            "fixed",
            "cube",
            f"/World/Fixtures/ShelfBin_{idx:02d}",
            "storage_bin",
            [materials["bin_blue"], materials["bin_green"], materials["bin_gray"]][idx % 3],
            (x, 1.00, 1.20),
            (0, 0, 0),
            (0.16, 0.11, 0.09),
            add_update_semantics,
            rep,
        )


def add_robot(stage, config, records, materials, add_update_semantics, rep, robot_mode: str):
    if robot_mode != "official":
        raise ValueError(f"Unsupported robot mode for this project: {robot_mode}")
    asset_path = config.get("robot", {}).get("unitree_usd_path", "")
    resolved_asset = resolve_asset_path(asset_path)
    if not resolved_asset or not resolved_asset.exists():
        raise FileNotFoundError(f"Official Unitree G1 USD asset was not found: {asset_path}")
    if is_lfs_pointer(resolved_asset):
        raise RuntimeError(f"Official Unitree G1 USD asset is still a Git LFS pointer: {resolved_asset}")
    return add_referenced_robot(stage, resolved_asset, add_update_semantics, rep, require_pose_targets=True)


def resolve_asset_path(asset_path: str) -> Path | None:
    if not asset_path:
        return None
    path = Path(asset_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def is_lfs_pointer(path: Path) -> bool:
    try:
        data = path.read_bytes()[:256]
    except OSError:
        return False
    return b"git-lfs.github.com/spec" in data


OFFICIAL_G1_POSE_LIBRARY: dict[str, dict[str, tuple[float, float, float]]] = {
    "neutral": {},
    "left_reach": {
        "left_shoulder_pitch": (-28, 0, 0),
        "left_shoulder_roll": (0, 0, 16),
        "left_elbow": (0, 34, 0),
        "waist_yaw": (0, 0, 8),
    },
    "right_reach": {
        "right_shoulder_pitch": (-28, 0, 0),
        "right_shoulder_roll": (0, 0, -16),
        "right_elbow": (0, -34, 0),
        "waist_yaw": (0, 0, -8),
    },
    "inspection_lean": {
        "waist_pitch": (10, 0, 0),
        "left_shoulder_pitch": (-16, 0, 0),
        "right_shoulder_pitch": (-16, 0, 0),
        "left_elbow": (0, 22, 0),
        "right_elbow": (0, -22, 0),
    },
    "crouch_view": {
        "left_hip_pitch": (-9, 0, 0),
        "right_hip_pitch": (-9, 0, 0),
        "left_knee": (18, 0, 0),
        "right_knee": (18, 0, 0),
        "left_ankle_pitch": (-8, 0, 0),
        "right_ankle_pitch": (-8, 0, 0),
    },
    "occluded_side_reach": {
        "waist_yaw": (0, 0, 15),
        "right_shoulder_pitch": (-22, 0, 0),
        "right_shoulder_yaw": (0, 18, 0),
        "right_elbow": (0, -30, 0),
    },
}


OFFICIAL_G1_POSE_TARGET_NAMES = tuple(
    sorted(
        {target for pose_targets in OFFICIAL_G1_POSE_LIBRARY.values() for target in pose_targets},
        key=len,
        reverse=True,
    )
)


def add_referenced_robot(stage, asset_path: Path, add_update_semantics, rep, require_pose_targets: bool = False):
    from pxr import UsdGeom

    root = UsdGeom.Xform.Define(stage, "/World/RobotRig").GetPrim()
    asset_root = UsdGeom.Xform.Define(stage, "/World/RobotRig/RobotAsset").GetPrim()
    asset_root.GetReferences().AddReference(str(asset_path))
    ops = set_transform_ops(root, (-0.82, 0.68, OFFICIAL_G1_ROOT_Z), (0, 0, -25), (1.0, 1.0, 1.0))
    apply_robot_semantics(stage, "/World/RobotRig", add_update_semantics, rep)

    pose_ops = collect_official_g1_pose_ops(stage, "/World/RobotRig/RobotAsset")
    if require_pose_targets and not pose_ops:
        raise RuntimeError("Official Unitree G1 asset loaded, but no editable G1 pose targets were found.")

    robot_info = {
        "mode": "official_unitree_usd",
        "asset_path": str(asset_path),
        "asset_exists": True,
        "fallback_used": False,
        "pose_target_count": len(pose_ops),
    }
    return [
        {
            "path": "/World/RobotRig",
            "label": "robot",
            "kind": "referenced_usd",
            "group": "robot",
            "prim": root,
            "ops": ops,
            "pose_ops": pose_ops,
            "base_scale": (1, 1, 1),
        }
    ], robot_info


def collect_official_g1_pose_ops(stage, asset_root_path: str) -> dict[str, Any]:
    from pxr import UsdGeom

    pose_ops: dict[str, Any] = {}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(asset_root_path):
            continue
        prim_name = prim.GetName().lower()
        matched_name = next((target for target in OFFICIAL_G1_POSE_TARGET_NAMES if target in prim_name), None)
        if not matched_name or matched_name in pose_ops:
            continue
        if not prim.IsA(UsdGeom.Xformable):
            continue
        try:
            xformable = UsdGeom.Xformable(prim)
            pose_ops[matched_name] = add_or_get_rotate_xyz_op(xformable, "datasetPose")
        except Exception as exc:
            print(f"[warn] Could not create pose op for {prim_path}: {exc}", flush=True)
    return pose_ops


def add_or_get_rotate_xyz_op(xformable, suffix: str):
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == f"xformOp:rotateXYZ:{suffix}":
            return op
    return xformable.AddRotateXYZOp(opSuffix=suffix)


def apply_robot_semantics(stage, root_path: str, add_update_semantics, rep) -> None:
    from pxr import UsdGeom

    # Unitree USD visual branches are instanceable Xforms. Mark them
    # non-instanceable so stronger semantic opinions can be authored locally.
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(root_path) and prim.IsInstanceable():
            prim.SetInstanceable(False)

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root_path):
            continue
        if "/Looks" in path or "/joints" in path:
            continue
        if prim.IsA(UsdGeom.Imageable) or prim.IsA(UsdGeom.Gprim):
            try:
                set_semantics(prim, "robot", add_update_semantics, rep)
            except Exception as exc:
                print(f"[warn] Could not assign robot semantics to {path}: {exc}", flush=True)


SCENARIO_PROBABILITIES: dict[str, dict[str, float]] = {
    "reflective_metal_workcell": {
        "metal": 0.96,
        "tools": 0.65,
        "connectors": 0.30,
        "screws": 0.28,
        "glass": 0.18,
        "bins": 0.45,
        "cables": 0.24,
        "pcbs": 0.22,
        "sensors": 0.30,
        "markers": 0.22,
    },
    "transparent_inspection": {
        "glass": 0.96,
        "metal": 0.36,
        "connectors": 0.44,
        "screws": 0.22,
        "occluders": 0.25,
        "bins": 0.35,
        "cables": 0.18,
        "sensors": 0.34,
        "markers": 0.28,
    },
    "occluded_bin_picking": {
        "occluders": 0.92,
        "bins": 0.86,
        "connectors": 0.72,
        "screws": 0.62,
        "metal": 0.48,
        "glass": 0.38,
        "cables": 0.42,
        "tools": 0.32,
        "rubber": 0.42,
        "pcbs": 0.35,
    },
    "dense_small_parts": {
        "screws": 0.94,
        "connectors": 0.88,
        "cables": 0.78,
        "tools": 0.42,
        "metal": 0.35,
        "bins": 0.32,
        "pcbs": 0.48,
        "rubber": 0.58,
        "markers": 0.34,
    },
    "dynamic_conveyor": {
        "moving": 0.95,
        "metal": 0.45,
        "connectors": 0.46,
        "screws": 0.32,
        "cables": 0.28,
        "glass": 0.20,
        "bins": 0.35,
        "sensors": 0.34,
        "rubber": 0.36,
    },
    "mixed_hard_workcell": {
        "metal": 0.92,
        "glass": 0.88,
        "screws": 0.88,
        "connectors": 0.84,
        "cables": 0.76,
        "moving": 0.82,
        "occluders": 0.80,
        "tools": 0.68,
        "bins": 0.72,
        "pcbs": 0.76,
        "rubber": 0.72,
        "sensors": 0.68,
        "markers": 0.58,
    },
    "robot_close_range": {
        "metal": 0.72,
        "glass": 0.62,
        "screws": 0.72,
        "connectors": 0.74,
        "cables": 0.60,
        "occluders": 0.55,
        "tools": 0.70,
        "moving": 0.25,
        "bins": 0.42,
        "pcbs": 0.56,
        "sensors": 0.62,
        "rubber": 0.40,
        "markers": 0.42,
    },
    "micro_assembly_electronics": {
        "pcbs": 0.96,
        "screws": 0.90,
        "connectors": 0.88,
        "cables": 0.82,
        "sensors": 0.78,
        "markers": 0.62,
        "metal": 0.42,
        "glass": 0.28,
        "rubber": 0.52,
        "tools": 0.48,
    },
    "handover_pose_occlusion": {
        "occluders": 0.86,
        "tools": 0.74,
        "metal": 0.66,
        "glass": 0.58,
        "connectors": 0.70,
        "screws": 0.62,
        "pcbs": 0.52,
        "sensors": 0.54,
        "cables": 0.58,
        "moving": 0.34,
        "bins": 0.44,
    },
    "glass_reflection_robot_closeup": {
        "glass": 0.98,
        "metal": 0.88,
        "sensors": 0.62,
        "markers": 0.58,
        "connectors": 0.58,
        "screws": 0.46,
        "cables": 0.42,
        "occluders": 0.52,
        "tools": 0.42,
    },
    "low_light_motion_blur": {
        "moving": 0.94,
        "metal": 0.60,
        "glass": 0.42,
        "connectors": 0.54,
        "screws": 0.50,
        "cables": 0.46,
        "rubber": 0.42,
        "sensors": 0.38,
        "bins": 0.34,
    },
    "shelf_bin_clutter": {
        "bins": 0.94,
        "occluders": 0.80,
        "metal": 0.64,
        "glass": 0.46,
        "screws": 0.72,
        "connectors": 0.78,
        "cables": 0.58,
        "rubber": 0.60,
        "pcbs": 0.54,
        "tools": 0.50,
        "markers": 0.48,
    },
}


def randomize_frame(frame_id, scenario, rng, records, lights, camera, rep) -> dict[str, Any]:
    from pxr import Gf

    scenario_name = scenario["name"]
    probabilities = SCENARIO_PROBABILITIES[scenario_name]
    randomized_objects: list[dict[str, Any]] = []

    robot_pose_meta = randomize_robot(records["robot"], rng, scenario_name)
    for group_name, group_records in records.items():
        if group_name in {"all", "fixed", "robot"}:
            continue
        probability = probabilities.get(group_name, 0.0)
        for index, record in enumerate(group_records):
            enabled = rng.random() < probability
            if not enabled:
                record["ops"]["translate"].Set(Gf.Vec3d(*HIDE_TRANSLATE))
                continue

            sx, sy, sz = jitter_scale(record["base_scale"], rng, record["label"], scenario_name)
            x, y, z = sample_position(record, group_name, scenario_name, frame_id, index, rng, sx, sy, sz)
            roll, pitch, yaw = sample_rotation(record, group_name, rng)

            record["ops"]["translate"].Set(Gf.Vec3d(x, y, z))
            record["ops"]["rotate"].Set(Gf.Vec3f(roll, pitch, yaw))
            record["ops"]["scale"].Set(Gf.Vec3f(sx, sy, sz))
            material = rng.choice(record.get("material_options") or [])
            if material:
                bind_material(record_prim(record), material)

            randomized_objects.append(
                {
                    "path": record["path"],
                    "label": record["label"],
                    "group": group_name,
                    "kind": record["kind"],
                    "position": [round(x, 4), round(y, 4), round(z, 4)],
                    "rotation": [round(roll, 3), round(pitch, 3), round(yaw, 3)],
                    "scale": [round(sx, 4), round(sy, 4), round(sz, 4)],
                    "object_role": record.get("object_role", ""),
                }
            )

    camera_position, camera_target = sample_camera_pose(scenario_name, rng)
    rep.functional.modify.pose(camera, position_value=camera_position, look_at_value=camera_target, write_to_usd=True)
    lighting_meta = randomize_lighting(lights, rng, scenario_name)

    return {
        "scenario": scenario_name,
        "challenge_tags": scenario["challenge_tags"],
        "camera": {
            "position": [round(v, 4) for v in camera_position],
            "look_at": [round(v, 4) for v in camera_target],
        },
        "lighting": lighting_meta,
        "robot_pose": robot_pose_meta,
        "objects": randomized_objects,
    }


def record_prim(record):
    return record["prim"]


def randomize_robot(robot_records: list[dict[str, Any]], rng, scenario_name: str) -> dict[str, Any]:
    from pxr import Gf

    if not robot_records:
        return {"mode": "none"}

    if scenario_name == "robot_close_range":
        center = (rng.uniform(-0.74, -0.34), rng.uniform(0.20, 0.56), OFFICIAL_G1_ROOT_Z)
        yaw = rng.uniform(-42, 18)
    else:
        center = (rng.uniform(-0.98, -0.54), rng.uniform(0.46, 0.88), OFFICIAL_G1_ROOT_Z)
        yaw = rng.uniform(-42, -5)

    if len(robot_records) == 1 and robot_records[0]["kind"] == "referenced_usd":
        pose_name = choose_robot_pose(scenario_name, rng)
        root_pitch = rng.uniform(-2.0, 2.0)
        root_roll = rng.uniform(-1.5, 1.5)
        robot_records[0]["ops"]["translate"].Set(Gf.Vec3d(*center))
        robot_records[0]["ops"]["rotate"].Set(Gf.Vec3f(root_roll, root_pitch, yaw))
        applied_targets = apply_official_g1_pose(robot_records[0].get("pose_ops", {}), pose_name, rng)
        return {
            "mode": "official_unitree_g1",
            "pose_name": pose_name,
            "position": [round(v, 4) for v in center],
            "root_rotation": [round(root_roll, 3), round(root_pitch, 3), round(yaw, 3)],
            "pose_target_count": len(robot_records[0].get("pose_ops", {})),
            "applied_pose_targets": applied_targets,
        }

    raise RuntimeError("Official Unitree G1 robot record was not loaded.")


def choose_robot_pose(scenario_name: str, rng) -> str:
    if scenario_name in {"robot_close_range", "handover_pose_occlusion", "glass_reflection_robot_closeup"}:
        choices = ["left_reach", "right_reach", "inspection_lean", "occluded_side_reach"]
    elif scenario_name in {"occluded_bin_picking", "shelf_bin_clutter"}:
        choices = ["inspection_lean", "crouch_view", "left_reach", "right_reach"]
    elif scenario_name in {"dense_small_parts", "micro_assembly_electronics"}:
        choices = ["inspection_lean", "left_reach", "right_reach", "neutral"]
    else:
        choices = list(OFFICIAL_G1_POSE_LIBRARY)
    return rng.choice(choices)


def apply_official_g1_pose(pose_ops: dict[str, Any], pose_name: str, rng) -> list[str]:
    from pxr import Gf

    pose_targets = OFFICIAL_G1_POSE_LIBRARY.get(pose_name, {})
    applied: list[str] = []
    for target_name, op in pose_ops.items():
        roll, pitch, yaw = pose_targets.get(target_name, (0.0, 0.0, 0.0))
        if target_name in pose_targets:
            roll += rng.uniform(-2.5, 2.5)
            pitch += rng.uniform(-2.5, 2.5)
            yaw += rng.uniform(-2.5, 2.5)
            applied.append(target_name)
        op.Set(Gf.Vec3f(float(roll), float(pitch), float(yaw)))
    return sorted(applied)


def sample_position(record, group_name: str, scenario_name: str, frame_id: int, index: int, rng, sx: float, sy: float, sz: float):
    if group_name == "moving":
        base_x = -0.95 + 0.28 * index
        phase = frame_id * 0.21 + index * 0.65
        x = base_x + math.sin(phase) * 0.18 + rng.uniform(-0.025, 0.025)
        y = -0.56 + math.cos(phase * 1.2) * 0.035
        z = TABLE_TOP_Z + sz + 0.035
        return x, y, z

    if group_name == "occluders":
        x = rng.uniform(-0.86, 0.86)
        y = rng.uniform(-0.70, -0.25)
        z = TABLE_TOP_Z + sz + rng.uniform(0.02, 0.18)
        return x, y, z

    if group_name == "bins":
        x = rng.uniform(-0.95, 0.95)
        y = rng.uniform(-0.40, 0.48)
        z = TABLE_TOP_Z + sz
        return x, y, z

    if scenario_name == "robot_close_range":
        x = rng.uniform(-0.65, 0.45)
        y = rng.uniform(-0.12, 0.45)
    elif scenario_name == "dense_small_parts":
        x = rng.uniform(-0.95, 1.05)
        y = rng.uniform(-0.52, 0.42)
    else:
        x = rng.uniform(-1.08, 1.12)
        y = rng.uniform(-0.60, 0.55)

    z = TABLE_TOP_Z + sz
    if record["label"] == "transparent_glass":
        z += rng.uniform(0.04, 0.16)
    if record["label"] == "cable":
        z = TABLE_TOP_Z + max(sx, sy) + 0.012
    return x, y, z


def sample_rotation(record, group_name: str, rng) -> tuple[float, float, float]:
    label = record["label"]
    yaw = rng.uniform(-180, 180)
    if label == "cable":
        return 90 + rng.uniform(-5, 5), rng.uniform(-8, 8), yaw
    if label == "screw":
        if rng.random() < 0.72:
            return 90 + rng.uniform(-10, 10), rng.uniform(-8, 8), yaw
        return 0, 0, yaw
    if label == "transparent_glass":
        return rng.uniform(-5, 5), rng.uniform(-10, 10), yaw
    if label in {"pcb", "fiducial_marker"}:
        return rng.uniform(-2, 2), rng.uniform(-2, 2), yaw
    if label in {"sensor_module", "rubber_part"}:
        return rng.uniform(-10, 10), rng.uniform(-10, 10), yaw
    if label in {"metal_part", "tool"}:
        return rng.uniform(-12, 12), rng.uniform(-12, 12), yaw
    if group_name == "occluders":
        return rng.uniform(-4, 4), rng.uniform(-5, 5), rng.uniform(-35, 35)
    return 0, 0, yaw


def jitter_scale(base_scale, rng, label: str, scenario_name: str) -> tuple[float, float, float]:
    sx, sy, sz = base_scale
    if label == "screw":
        factor = rng.uniform(0.70, 1.30)
        return sx * factor, sy * factor, sz * rng.uniform(0.65, 1.35)
    if label == "cable":
        return sx * rng.uniform(0.85, 1.25), sy * rng.uniform(0.85, 1.25), sz * rng.uniform(0.45, 1.35)
    if label == "transparent_glass":
        return sx * rng.uniform(0.70, 1.45), sy, sz * rng.uniform(0.65, 1.35)
    if label in {"pcb", "fiducial_marker"}:
        return sx * rng.uniform(0.80, 1.20), sy * rng.uniform(0.80, 1.20), sz
    if label in {"sensor_module", "rubber_part"}:
        return sx * rng.uniform(0.80, 1.30), sy * rng.uniform(0.80, 1.30), sz * rng.uniform(0.80, 1.25)
    if label == "occluder":
        height_gain = 1.25 if scenario_name in {"occluded_bin_picking", "mixed_hard_workcell", "robot_close_range"} else 1.0
        return sx * rng.uniform(0.75, 1.35), sy * rng.uniform(0.8, 1.25), sz * rng.uniform(0.8, 1.35) * height_gain
    if label == "storage_bin":
        return sx * rng.uniform(0.80, 1.20), sy * rng.uniform(0.80, 1.20), sz * rng.uniform(0.80, 1.10)
    return sx * rng.uniform(0.80, 1.22), sy * rng.uniform(0.80, 1.22), sz * rng.uniform(0.80, 1.22)


def sample_camera_pose(scenario_name: str, rng) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if scenario_name in {"robot_close_range", "handover_pose_occlusion", "glass_reflection_robot_closeup"}:
        return (
            (rng.uniform(1.35, 2.15), rng.uniform(-1.92, -1.28), rng.uniform(1.10, 1.65)),
            (rng.uniform(-0.55, 0.22), rng.uniform(0.04, 0.48), rng.uniform(0.66, 1.02)),
        )
    if scenario_name in {"dense_small_parts", "micro_assembly_electronics"}:
        return (
            (rng.uniform(1.55, 2.55), rng.uniform(-2.35, -1.65), rng.uniform(1.25, 1.85)),
            (rng.uniform(-0.05, 0.32), rng.uniform(-0.18, 0.28), rng.uniform(0.62, 0.82)),
        )
    if scenario_name in {"dynamic_conveyor", "low_light_motion_blur"}:
        return (
            (rng.uniform(2.15, 3.05), rng.uniform(-2.95, -2.35), rng.uniform(1.40, 2.00)),
            (rng.uniform(-0.10, 0.38), rng.uniform(-0.45, -0.20), rng.uniform(0.66, 0.88)),
        )
    if scenario_name == "shelf_bin_clutter":
        return (
            (rng.uniform(1.80, 2.80), rng.uniform(-2.20, -1.45), rng.uniform(1.45, 2.20)),
            (rng.uniform(-0.20, 0.45), rng.uniform(0.32, 0.86), rng.uniform(0.82, 1.25)),
        )
    return (
        (rng.uniform(2.35, 3.55), rng.uniform(-3.55, -2.45), rng.uniform(1.55, 2.35)),
        (rng.uniform(-0.12, 0.28), rng.uniform(-0.05, 0.35), rng.uniform(0.68, 1.02)),
    )


def randomize_lighting(lights: dict[str, Any], rng, scenario_name: str) -> dict[str, float]:
    dome = rng.uniform(260, 540)
    key = rng.uniform(650, 1250)
    rim = rng.uniform(160, 520)
    if scenario_name in {"reflective_metal_workcell", "transparent_inspection", "mixed_hard_workcell", "glass_reflection_robot_closeup"}:
        key *= 1.18
        rim *= 1.25
    if scenario_name == "low_light_motion_blur":
        dome *= 0.42
        key *= 0.52
        rim *= 0.65

    lights["dome_intensity"].Set(float(dome))
    lights["key_intensity"].Set(float(key))
    lights["rim_intensity"].Set(float(rim))
    return {
        "dome_intensity": round(dome, 3),
        "key_intensity": round(key, 3),
        "rim_intensity": round(rim, 3),
    }


def make_writer(rep, output_dir: Path, writer_config: dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(output_dir))

    writer = rep.writers.get("BasicWriter")
    requested = dict(writer_config)
    requested.setdefault("rgb", True)
    requested.setdefault("semantic_segmentation", True)
    requested.setdefault("colorize_semantic_segmentation", True)
    try:
        writer.initialize(backend=backend, **requested)
    except TypeError as exc:
        print(f"[warn] BasicWriter rejected full writer config: {exc}")
        minimal = {
            "rgb": True,
            "semantic_segmentation": True,
            "colorize_semantic_segmentation": True,
            "bounding_box_2d_tight": True,
        }
        writer.initialize(backend=backend, **minimal)
    return writer


if __name__ == "__main__":
    main()
