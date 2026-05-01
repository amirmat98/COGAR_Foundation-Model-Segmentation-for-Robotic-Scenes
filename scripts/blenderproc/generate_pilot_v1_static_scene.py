import blenderproc as bproc

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_material(name, color, metallic=0.0, roughness=0.5, alpha=1.0):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Metallic", metallic)
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Alpha", alpha)
    return mat


def tag_object(obj, category_name, category_id, supercategory):
    obj.set_name(category_name)
    obj.set_cp("category_id", category_id)
    obj.set_cp("supercategory", supercategory)
    return obj


def create_primitive(category, primitive_type, location, scale, material, rotation=None):
    obj = bproc.object.create_primitive(primitive_type)
    obj.set_location(location)
    obj.set_scale(scale)
    if rotation is not None:
        obj.set_rotation_euler(rotation)

    obj.replace_materials(material)

    tag_object(
        obj=obj,
        category_name=category["name"],
        category_id=category["id"],
        supercategory=category["supercategory"],
    )
    return obj


def category_lookup(categories):
    return {cat["name"]: cat for cat in categories}


def random_xy(xlim=(-1.0, 1.0), ylim=(-0.55, 0.55)):
    return [random.uniform(*xlim), random.uniform(*ylim)]


def add_camera_pose(location, target=(0.0, 0.0, 0.15)):
    location_np = np.array(location)
    target_np = np.array(target)
    forward_vec = target_np - location_np
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world_matrix = bproc.math.build_transformation_mat(location_np, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def create_scene(categories, render_samples):
    cats = category_lookup(categories)

    # Materials
    mat_table = make_material("mat_table", [0.45, 0.35, 0.25, 1.0], metallic=0.0, roughness=0.8)
    mat_robot = make_material("mat_robot_dark", [0.02, 0.02, 0.025, 1.0], metallic=0.2, roughness=0.35)
    mat_metal = make_material("mat_reflective_metal", [0.75, 0.75, 0.72, 1.0], metallic=1.0, roughness=0.08)
    mat_glass = make_material("mat_transparent_glass", [0.65, 0.90, 1.0, 0.35], metallic=0.0, roughness=0.02, alpha=0.35)
    mat_plastic = make_material("mat_blue_plastic", [0.05, 0.20, 0.85, 1.0], metallic=0.0, roughness=0.45)
    mat_cable = make_material("mat_black_cable", [0.005, 0.005, 0.005, 1.0], metallic=0.0, roughness=0.6)
    mat_tool = make_material("mat_tool", [0.60, 0.58, 0.52, 1.0], metallic=1.0, roughness=0.18)
    mat_box = make_material("mat_cardboard", [0.55, 0.37, 0.18, 1.0], metallic=0.0, roughness=0.9)

    # Renderer settings: low enough for laptop, acceptable for pilot.
    bproc.camera.set_resolution(640, 480)
    bproc.renderer.set_max_amount_of_samples(render_samples)

    # Table
    create_primitive(
        category=cats["table"],
        primitive_type="CUBE",
        location=[0.0, 0.0, -0.05],
        scale=[2.8, 1.8, 0.08],
        material=mat_table,
    )

    # Robot gripper-like occluder
    create_primitive(
        category=cats["robot_gripper"],
        primitive_type="CUBE",
        location=[0.0, -0.10, 0.65],
        scale=[0.45, 0.10, 0.10],
        material=mat_robot,
    )
    create_primitive(
        category=cats["robot_gripper"],
        primitive_type="CUBE",
        location=[-0.18, 0.05, 0.43],
        scale=[0.06, 0.08, 0.32],
        material=mat_robot,
    )
    create_primitive(
        category=cats["robot_gripper"],
        primitive_type="CUBE",
        location=[0.18, 0.05, 0.43],
        scale=[0.06, 0.08, 0.32],
        material=mat_robot,
    )

    # Reflective metal parts
    for i in range(3):
        x, y = random_xy()
        create_primitive(
            category=cats["metal_part"],
            primitive_type="CYLINDER",
            location=[x, y, 0.18],
            scale=[0.10, 0.10, 0.22],
            material=mat_metal,
            rotation=[0.0, 0.0, random.uniform(0, math.pi)],
        )

    # Transparent glass-like objects
    for i in range(2):
        x, y = random_xy()
        create_primitive(
            category=cats["glass_object"],
            primitive_type="CYLINDER",
            location=[x, y, 0.25],
            scale=[0.14, 0.14, 0.32],
            material=mat_glass,
        )

    # Plastic object
    x, y = random_xy()
    create_primitive(
        category=cats["plastic_object"],
        primitive_type="SPHERE",
        location=[x, y, 0.17],
        scale=[0.16, 0.16, 0.16],
        material=mat_plastic,
    )

    # Small screws
    for i in range(10):
        x, y = random_xy()
        create_primitive(
            category=cats["screw"],
            primitive_type="CYLINDER",
            location=[x, y, 0.055],
            scale=[0.025, 0.025, 0.035],
            material=mat_metal,
            rotation=[0.0, 0.0, random.uniform(0, math.pi)],
        )

    # Connectors
    for i in range(4):
        x, y = random_xy()
        create_primitive(
            category=cats["connector"],
            primitive_type="CUBE",
            location=[x, y, 0.08],
            scale=[0.13, 0.055, 0.045],
            material=mat_plastic,
            rotation=[0.0, 0.0, random.uniform(0, math.pi)],
        )

    # Cable-like dark cylinders
    for i in range(2):
        x, y = random_xy()
        create_primitive(
            category=cats["cable"],
            primitive_type="CYLINDER",
            location=[x, y, 0.065],
            scale=[0.025, 0.025, 0.55],
            material=mat_cable,
            rotation=[math.pi / 2.0, 0.0, random.uniform(0, math.pi)],
        )

    # Tool-like metal bar
    x, y = random_xy()
    create_primitive(
        category=cats["tool"],
        primitive_type="CUBE",
        location=[x, y, 0.08],
        scale=[0.42, 0.055, 0.045],
        material=mat_tool,
        rotation=[0.0, 0.0, random.uniform(0, math.pi)],
    )

    # Box
    x, y = random_xy()
    create_primitive(
        category=cats["box"],
        primitive_type="CUBE",
        location=[x, y, 0.16],
        scale=[0.25, 0.20, 0.18],
        material=mat_box,
        rotation=[0.0, 0.0, random.uniform(0, math.pi)],
    )

    # Dynamic-scene proxy: the same object class appears in multiple positions.
    # In the final generator we can implement true frame-wise motion.
    for x in [-0.75, -0.35, 0.05, 0.45, 0.85]:
        create_primitive(
            category=cats["plastic_object"],
            primitive_type="SPHERE",
            location=[x, -0.68, 0.12],
            scale=[0.09, 0.09, 0.09],
            material=mat_plastic,
        )

    # Lighting
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([1.5, -2.0, 3.0])
    light.set_energy(450)

    light2 = bproc.types.Light()
    light2.set_type("AREA")
    light2.set_location([-2.0, 1.5, 4.0])
    light2.set_energy(250)

    # Camera poses: 20 views.
    camera_locations = [
        [0.0, -3.2, 2.0],
        [1.8, -2.6, 2.1],
        [-1.8, -2.6, 2.1],
        [0.0, -0.2, 4.0],
    ]

    for i in range(20):
        base = camera_locations[i % len(camera_locations)]
        jitter = [
            random.uniform(-0.18, 0.18),
            random.uniform(-0.18, 0.18),
            random.uniform(-0.10, 0.10),
        ]
        loc = [base[0] + jitter[0], base[1] + jitter[1], base[2] + jitter[2]]
        add_camera_pose(loc)


def write_metadata(output_root: Path, categories, num_images: int):
    metadata_dir = output_root / "metadata"
    splits_dir = output_root / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    challenges = (
        ["reflective_metal"] * 4
        + ["transparent_glass"] * 4
        + ["partial_occlusion"] * 4
        + ["small_parts"] * 4
        + ["dynamic_scene"] * 4
    )

    rows = []
    for i in range(num_images):
        challenge = challenges[i]
        rows.append({
            "image_id": i + 1,
            "file_name": f"frame_{i:06d}",
            "scene_id": f"pilot_scene_{i // 4:03d}",
            "primary_challenge": challenge,
            "reflective": int(challenge == "reflective_metal"),
            "transparent": int(challenge == "transparent_glass"),
            "occlusion": int(challenge == "partial_occlusion"),
            "small_parts": int(challenge == "small_parts"),
            "dynamic": int(challenge == "dynamic_scene"),
            "num_objects": 30,
            "camera_view": ["front", "oblique_right", "oblique_left", "top"][i % 4],
            "lighting_condition": "mixed_point_area",
        })

    csv_path = metadata_dir / "frame_index_pilot.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "scene_id",
                "primary_challenge",
                "reflective",
                "transparent",
                "occlusion",
                "small_parts",
                "dynamic",
                "num_objects",
                "camera_view",
                "lighting_condition",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    categories_path = metadata_dir / "categories.json"
    with categories_path.open("w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)

    split_path = splits_dir / "pilot.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for i in range(num_images):
            f.write(f"{i + 1:06d}\n")

    print(f"[OK] Metadata CSV: {csv_path}")
    print(f"[OK] Categories JSON: {categories_path}")
    print(f"[OK] Pilot split: {split_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/blenderproc_dataset.yaml",
        help="Path to dataset config YAML.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    config = load_config(repo_root / args.config)

    dataset_cfg = config["dataset"]
    gen_cfg = config["generation"]
    categories = config["categories"]

    output_root = repo_root / dataset_cfg["output_dir"]
    raw_output_dir = output_root / "raw_blenderproc" / "pilot_20"
    coco_output_dir = raw_output_dir / "coco_data"

    output_root.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(gen_cfg["seed"])
    np.random.seed(gen_cfg["seed"])

    bproc.init()

    create_scene(
        categories=categories,
        render_samples=int(gen_cfg.get("render_samples", 32)),
    )

    print("[INFO] Rendering RGB images...")
    data = bproc.renderer.render()

    print("[INFO] Rendering segmentation maps...")
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

    print("[INFO] Writing COCO annotations...")
    bproc.writer.write_coco_annotations(
        str(coco_output_dir),
        instance_segmaps=seg_data["instance_segmaps"],
        instance_attribute_maps=seg_data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="PNG",
    )

    write_metadata(
        output_root=output_root,
        categories=categories,
        num_images=int(dataset_cfg["pilot_images"]),
    )

    print(f"[OK] BlenderProc raw pilot output: {raw_output_dir}")
    print(f"[OK] COCO output folder: {coco_output_dir}")


if __name__ == "__main__":
    main()