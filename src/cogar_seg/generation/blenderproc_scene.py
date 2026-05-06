import blenderproc as bproc

import math
import random
import shutil
from pathlib import Path

import numpy as np
import yaml

from cogar_seg.generation.blenderproc_materials import (
    create_material_pool,
    ensure_world_background,
)
from cogar_seg.generation.blenderproc_metadata import (
    write_ocid_like_metadata as write_metadata,
)
from cogar_seg.generation.blenderproc_objects import (
    category_lookup,
    create_primitive,
    random_xy,
)


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

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
        "reflective_metal",
        "transparent_glass",
        "partial_occlusion",
        "small_parts",
        "dynamic_scene",
    ]
    return families[image_id % len(families)]


def object_plan_for_family(family):
    base_pool = ["box", "metal_part", "glass_object", "plastic_object", "connector", "screw", "cable", "tool"]

    if family == "reflective_metal":
        plan = (
            random.choices(["metal_part", "tool"], k=random.randint(5, 10))
            + random.choices(["box", "plastic_object", "connector", "screw", "cable"], k=random.randint(5, 12))
        )
        random.shuffle(plan)
        return plan, random.choice(["none", "mild"])

    if family == "transparent_glass":
        plan = (
            random.choices(["glass_object"], k=random.randint(4, 8))
            + random.choices(["metal_part", "box", "plastic_object", "connector", "cable"], k=random.randint(5, 12))
        )
        random.shuffle(plan)
        return plan, random.choice(["none", "mild", "medium"])

    if family == "partial_occlusion":
        return random.choices(base_pool, k=random.randint(12, 22)), random.choice(["medium", "severe"])

    if family == "small_parts":
        plan = (
            random.choices(["screw"], k=random.randint(12, 25))
            + random.choices(["connector"], k=random.randint(5, 12))
            + random.choices(["cable", "metal_part", "plastic_object", "box"], k=random.randint(3, 8))
        )
        random.shuffle(plan)
        return plan, random.choice(["none", "mild", "medium"])

    if family == "dynamic_scene":
        plan = random.choices(base_pool, k=random.randint(8, 16))
        plan += ["plastic_object"] * random.randint(4, 8)
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

    primary_challenge = family

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


def generate_cogar_sim_500(
    config_path: str | Path = "configs/blenderproc_dataset.yaml",
    num_images: int | None = None,
    repo_root: str | Path | None = None,
    raw_dataset_name: str = "pilot_v2_ocid_like",
    clean: bool = True,
) -> tuple[Path, Path]:
    """Generate randomized BlenderProc scenes for the COGAR-Sim 500 pipeline."""
    root = Path.cwd() if repo_root is None else Path(repo_root)
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path

    config = load_config(resolved_config_path)

    dataset_cfg = config["dataset"]
    gen_cfg = config["generation"]
    categories = config["categories"]

    if num_images is None:
        num_images = int(dataset_cfg.get("final_images", 500))

    output_root = root / dataset_cfg["output_dir"]
    raw_output_dir = output_root / "raw_blenderproc" / raw_dataset_name
    coco_output_dir = raw_output_dir / "coco_data"

    if clean and raw_output_dir.exists():
        shutil.rmtree(raw_output_dir)

    raw_output_dir.mkdir(parents=True, exist_ok=True)
    coco_output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(gen_cfg["seed"])
    np.random.seed(gen_cfg["seed"])

    bproc.init()
    ensure_world_background()

    metadata_rows = []
    render_samples = int(gen_cfg.get("render_samples", 32))

    for image_id in range(num_images):
        print(f"[INFO] Building independent scene {image_id + 1}/{num_images}")

        metadata = build_random_scene(
            image_id=image_id,
            categories=categories,
            render_samples=render_samples,
        )

        print("[INFO] Rendering RGB...")
        data = bproc.renderer.render()

        print("[INFO] Rendering segmentation...")
        seg_data = bproc.renderer.render_segmap(map_by=["instance", "class"])

        print(f"[INFO] Writing COCO frame {image_id + 1}/{args.num_images}...")
        bproc.writer.write_coco_annotations(
            str(coco_output_dir),
            instance_segmaps=seg_data["instance_segmaps"],
            instance_attribute_maps=seg_data["instance_attribute_maps"],
            colors=data["colors"],
            color_file_format="PNG",
            append_to_existing_output=(image_id != 0),
        )

        metadata_rows.append(metadata)

    write_metadata(
        output_root=output_root,
        categories=categories,
        rows=metadata_rows,
    )

    print(f"[OK] BlenderProc OCID-like pilot output: {raw_output_dir}")
    print(f"[OK] COCO output folder: {coco_output_dir}")

    return raw_output_dir, coco_output_dir
