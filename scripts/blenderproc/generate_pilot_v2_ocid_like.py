import blenderproc as bproc

import argparse
import csv
import json
import math
import random
from pathlib import Path

import bpy
import numpy as np
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def ensure_world_background():
    """
    bproc.clean_up() may remove Blender's world background.
    render_segmap expects scene.world.node_tree to exist.
    """
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")

    bpy.context.scene.world.use_nodes = True
    bproc.renderer.set_world_background([0.03, 0.03, 0.03], strength=0.8)


def make_material(name, color, metallic=0.0, roughness=0.5, alpha=1.0):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Metallic", metallic)
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Alpha", alpha)
    return mat


def tag_object(obj, category, name_suffix=""):
    name = category["name"] if not name_suffix else f"{category['name']}_{name_suffix}"
    obj.set_name(name)
    obj.set_cp("category_id", category["id"])
    obj.set_cp("supercategory", category["supercategory"])
    return obj


def create_primitive(category, primitive_type, location, scale, material, rotation=None, name_suffix=""):
    obj = bproc.object.create_primitive(primitive_type)
    obj.set_location(location)
    obj.set_scale(scale)

    if rotation is not None:
        obj.set_rotation_euler(rotation)

    obj.replace_materials(material)
    tag_object(obj, category, name_suffix=name_suffix)
    return obj


def category_lookup(categories):
    return {cat["name"]: cat for cat in categories}


def random_xy(xlim=(-1.0, 1.0), ylim=(-0.65, 0.65)):
    return random.uniform(*xlim), random.uniform(*ylim)


def add_camera_pose_random(view_type):
    if view_type == "top":
        location = np.array([
            random.uniform(-0.25, 0.25),
            random.uniform(-0.25, 0.25),
            random.uniform(3.0, 4.2),
        ])
    else:
        azimuth = random.uniform(0.0, 2.0 * math.pi)
        distance = random.uniform(2.2, 3.8)
        height = random.uniform(1.4, 2.8)

        location = np.array([
            distance * math.cos(azimuth),
            distance * math.sin(azimuth),
            height,
        ])

    target = np.array([
        random.uniform(-0.15, 0.15),
        random.uniform(-0.10, 0.10),
        random.uniform(0.05, 0.25),
    ])

    forward_vec = target - location
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward_vec)
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)


def create_material_pool():
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


def add_table_and_background(cats, mats):
    create_primitive(
        cats["table"],
        "CUBE",
        location=[0.0, 0.0, -0.06],
        scale=[3.0, 2.0, 0.08],
        material=mats["table"],
        name_suffix="surface",
    )

    # Simple wall/backdrop to avoid pure empty background.
    if random.random() < 0.7:
        create_primitive(
            cats["table"],
            "CUBE",
            location=[0.0, 1.05, 0.95],
            scale=[3.0, 0.06, 1.1],
            material=mats["wall"],
            name_suffix="backdrop",
        )


def add_robot_gripper(cats, mats, occlusion_level):
    if occlusion_level == "none":
        return 0

    x = random.uniform(-0.35, 0.35)
    y = random.uniform(-0.25, 0.25)
    z = random.uniform(0.45, 0.75)
    yaw = random.uniform(-0.6, 0.6)

    create_primitive(
        cats["robot_gripper"],
        "CUBE",
        location=[x, y, z],
        scale=[random.uniform(0.35, 0.60), 0.09, 0.10],
        material=mats["robot"],
        rotation=[0.0, 0.0, yaw],
        name_suffix="palm",
    )

    create_primitive(
        cats["robot_gripper"],
        "CUBE",
        location=[x - 0.17, y + 0.08, z - 0.18],
        scale=[0.055, 0.07, random.uniform(0.20, 0.36)],
        material=mats["robot"],
        rotation=[0.0, 0.0, yaw],
        name_suffix="finger_l",
    )

    create_primitive(
        cats["robot_gripper"],
        "CUBE",
        location=[x + 0.17, y + 0.08, z - 0.18],
        scale=[0.055, 0.07, random.uniform(0.20, 0.36)],
        material=mats["robot"],
        rotation=[0.0, 0.0, yaw],
        name_suffix="finger_r",
    )

    return 3


def add_random_object(cats, mats, obj_type, idx):
    x, y = random_xy()
    yaw = random.uniform(0, math.pi)

    if obj_type == "box":
        return create_primitive(
            cats["box"],
            "CUBE",
            location=[x, y, random.uniform(0.11, 0.20)],
            scale=[
                random.uniform(0.14, 0.34),
                random.uniform(0.12, 0.28),
                random.uniform(0.10, 0.26),
            ],
            material=mats["cardboard"],
            rotation=[0.0, 0.0, yaw],
            name_suffix=str(idx),
        )

    if obj_type == "metal_part":
        primitive = random.choice(["CYLINDER", "CUBE", "SPHERE"])
        return create_primitive(
            cats["metal_part"],
            primitive,
            location=[x, y, random.uniform(0.07, 0.22)],
            scale=[
                random.uniform(0.05, 0.18),
                random.uniform(0.05, 0.18),
                random.uniform(0.06, 0.28),
            ],
            material=mats["metal"],
            rotation=[random.uniform(0, 0.5), random.uniform(0, 0.5), yaw],
            name_suffix=str(idx),
        )

    if obj_type == "glass_object":
        return create_primitive(
            cats["glass_object"],
            random.choice(["CYLINDER", "CUBE"]),
            location=[x, y, random.uniform(0.16, 0.32)],
            scale=[
                random.uniform(0.10, 0.20),
                random.uniform(0.10, 0.20),
                random.uniform(0.20, 0.44),
            ],
            material=mats["glass"],
            rotation=[0.0, 0.0, yaw],
            name_suffix=str(idx),
        )

    if obj_type == "plastic_object":
        return create_primitive(
            cats["plastic_object"],
            random.choice(["SPHERE", "CUBE", "CYLINDER"]),
            location=[x, y, random.uniform(0.08, 0.20)],
            scale=[
                random.uniform(0.07, 0.18),
                random.uniform(0.07, 0.18),
                random.uniform(0.07, 0.20),
            ],
            material=random.choice([mats["plastic"], mats["plastic2"]]),
            rotation=[0.0, 0.0, yaw],
            name_suffix=str(idx),
        )

    if obj_type == "connector":
        return create_primitive(
            cats["connector"],
            "CUBE",
            location=[x, y, random.uniform(0.055, 0.09)],
            scale=[
                random.uniform(0.08, 0.18),
                random.uniform(0.035, 0.075),
                random.uniform(0.030, 0.060),
            ],
            material=random.choice([mats["plastic"], mats["plastic2"]]),
            rotation=[0.0, 0.0, yaw],
            name_suffix=str(idx),
        )

    if obj_type == "screw":
        return create_primitive(
            cats["screw"],
            "CYLINDER",
            location=[x, y, random.uniform(0.035, 0.065)],
            scale=[
                random.uniform(0.018, 0.040),
                random.uniform(0.018, 0.040),
                random.uniform(0.025, 0.060),
            ],
            material=mats["metal"],
            rotation=[random.uniform(0, math.pi), 0.0, yaw],
            name_suffix=str(idx),
        )

    if obj_type == "cable":
        return create_primitive(
            cats["cable"],
            "CYLINDER",
            location=[x, y, random.uniform(0.055, 0.085)],
            scale=[
                random.uniform(0.018, 0.035),
                random.uniform(0.018, 0.035),
                random.uniform(0.35, 0.85),
            ],
            material=mats["cable"],
            rotation=[math.pi / 2.0, random.uniform(-0.3, 0.3), yaw],
            name_suffix=str(idx),
        )

    if obj_type == "tool":
        return create_primitive(
            cats["tool"],
            "CUBE",
            location=[x, y, random.uniform(0.055, 0.10)],
            scale=[
                random.uniform(0.30, 0.65),
                random.uniform(0.035, 0.075),
                random.uniform(0.030, 0.060),
            ],
            material=mats["metal"],
            rotation=[0.0, 0.0, yaw],
            name_suffix=str(idx),
        )

    raise ValueError(f"Unknown object type: {obj_type}")


def sample_scene_family(image_id):
    families = [
        "low_clutter",
        "medium_clutter",
        "high_clutter",
        "transparent_reflective",
        "small_parts",
        "partial_occlusion",
        "dynamic_scene",
    ]
    return families[image_id % len(families)]


def object_plan_for_family(family):
    base_pool = ["box", "metal_part", "glass_object", "plastic_object", "connector", "screw", "cable", "tool"]

    if family == "low_clutter":
        return random.choices(base_pool, k=random.randint(5, 8)), "none"

    if family == "medium_clutter":
        return random.choices(base_pool, k=random.randint(9, 15)), random.choice(["none", "mild"])

    if family == "high_clutter":
        return random.choices(base_pool, k=random.randint(16, 25)), random.choice(["mild", "medium"])

    if family == "transparent_reflective":
        plan = (
            random.choices(["glass_object"], k=random.randint(3, 6))
            + random.choices(["metal_part", "tool"], k=random.randint(3, 6))
            + random.choices(["box", "plastic_object", "connector"], k=random.randint(3, 7))
        )
        random.shuffle(plan)
        return plan, random.choice(["none", "mild"])

    if family == "small_parts":
        plan = (
            random.choices(["screw"], k=random.randint(10, 22))
            + random.choices(["connector"], k=random.randint(4, 10))
            + random.choices(["cable", "metal_part", "plastic_object"], k=random.randint(3, 8))
        )
        random.shuffle(plan)
        return plan, random.choice(["none", "mild"])

    if family == "partial_occlusion":
        return random.choices(base_pool, k=random.randint(10, 18)), random.choice(["medium", "severe"])

    if family == "dynamic_scene":
        # For now this is a dynamic-scene proxy: several possible positions of a moving object.
        plan = random.choices(base_pool, k=random.randint(8, 14))
        plan += ["plastic_object"] * random.randint(3, 6)
        random.shuffle(plan)
        return plan, random.choice(["none", "mild", "medium"])

    return random.choices(base_pool, k=10), "none"


def add_lighting():
    light = bproc.types.Light()
    light.set_type(random.choice(["POINT", "AREA"]))
    light.set_location([
        random.uniform(-2.5, 2.5),
        random.uniform(-2.5, 2.5),
        random.uniform(2.5, 5.0),
    ])
    light.set_energy(random.uniform(250, 700))

    if random.random() < 0.5:
        light2 = bproc.types.Light()
        light2.set_type("POINT")
        light2.set_location([
            random.uniform(-2.5, 2.5),
            random.uniform(-2.5, 2.5),
            random.uniform(2.0, 4.5),
        ])
        light2.set_energy(random.uniform(80, 300))


def build_random_scene(image_id, categories, render_samples):
    cats = category_lookup(categories)

    bproc.clean_up(clean_up_camera=True)
    ensure_world_background()

    bproc.camera.set_resolution(640, 480)
    bproc.renderer.set_max_amount_of_samples(render_samples)

    mats = create_material_pool()
    add_table_and_background(cats, mats)

    family = sample_scene_family(image_id)
    object_plan, occlusion_level = object_plan_for_family(family)

    num_objects = 0

    for idx, obj_type in enumerate(object_plan):
        add_random_object(cats, mats, obj_type, idx)
        num_objects += 1

    num_objects += add_robot_gripper(cats, mats, occlusion_level)

    add_lighting()

    view_type = random.choice(["front", "oblique", "oblique", "top"])
    add_camera_pose_random(view_type)

    primary_challenge = {
        "low_clutter": "mixed",
        "medium_clutter": "mixed",
        "high_clutter": "partial_occlusion",
        "transparent_reflective": "transparent_glass",
        "small_parts": "small_parts",
        "partial_occlusion": "partial_occlusion",
        "dynamic_scene": "dynamic_scene",
    }[family]

    metadata = {
        "image_id": image_id + 1,
        "file_name": f"{image_id:06d}.png",
        "scene_id": f"pilot_v2_scene_{image_id:03d}",
        "scene_family": family,
        "primary_challenge": primary_challenge,
        "reflective": int("metal_part" in object_plan or "tool" in object_plan),
        "transparent": int("glass_object" in object_plan),
        "occlusion": int(occlusion_level != "none"),
        "small_parts": int("screw" in object_plan or "connector" in object_plan),
        "dynamic": int(family == "dynamic_scene"),
        "num_objects": num_objects,
        "camera_view": view_type,
        "lighting_condition": "randomized",
    }

    return metadata


def write_metadata(output_root: Path, categories, rows):
    metadata_dir = output_root / "metadata"
    splits_dir = output_root / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / "frame_index_pilot_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "scene_id",
                "scene_family",
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

    split_path = splits_dir / "pilot_v2.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_id']:06d}\n")

    print(f"[OK] Metadata CSV: {csv_path}")
    print(f"[OK] Categories JSON: {categories_path}")
    print(f"[OK] Pilot v2 split: {split_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/blenderproc_dataset.yaml",
        help="Path to dataset config YAML.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of independent randomized scenes to render.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    config = load_config(repo_root / args.config)

    dataset_cfg = config["dataset"]
    gen_cfg = config["generation"]
    categories = config["categories"]

    output_root = repo_root / dataset_cfg["output_dir"]
    raw_output_dir = output_root / "raw_blenderproc" / "pilot_v2_ocid_like"
    coco_output_dir = raw_output_dir / "coco_data"

    output_root.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    coco_output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(gen_cfg["seed"])
    np.random.seed(gen_cfg["seed"])

    bproc.init()
    ensure_world_background()

    all_colors = []
    all_instance_segmaps = []
    all_instance_attribute_maps = []
    metadata_rows = []

    render_samples = int(gen_cfg.get("render_samples", 32))

    for image_id in range(args.num_images):
        print(f"[INFO] Building independent scene {image_id + 1}/{args.num_images}")

        metadata = build_random_scene(
            image_id=image_id,
            categories=categories,
            render_samples=render_samples,
        )

        print("[INFO] Rendering RGB...")
        data = bproc.renderer.render()

        print("[INFO] Rendering segmentation...")
        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

        all_colors.append(data["colors"][0])
        all_instance_segmaps.append(seg_data["instance_segmaps"][0])
        all_instance_attribute_maps.append(seg_data["instance_attribute_maps"][0])
        metadata_rows.append(metadata)

    print("[INFO] Writing COCO annotations...")
    bproc.writer.write_coco_annotations(
        str(coco_output_dir),
        instance_segmaps=all_instance_segmaps,
        instance_attribute_maps=all_instance_attribute_maps,
        colors=all_colors,
        color_file_format="PNG",
    )

    write_metadata(
        output_root=output_root,
        categories=categories,
        rows=metadata_rows,
    )

    print(f"[OK] BlenderProc OCID-like pilot output: {raw_output_dir}")
    print(f"[OK] COCO output folder: {coco_output_dir}")


if __name__ == "__main__":
    main()