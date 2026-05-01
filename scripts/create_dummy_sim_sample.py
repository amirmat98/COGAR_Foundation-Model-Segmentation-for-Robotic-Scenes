import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from cogar_seg.datasets.sim_robotic import (
    REQUIRED_SIM_INDEX_COLUMNS,
    validate_sim_index,
)


DEFAULT_INDEX_PATH = Path("sim_dataset/annotations/sim_robotic_scenes_index.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one dummy simulated robotic-scene sample."
    )

    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help="Path to the simulated benchmark index CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dummy files and index row.",
    )

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_rgb_image(path: Path) -> None:
    ensure_parent(path)

    image = Image.new("RGB", (640, 480), (235, 235, 235))
    draw = ImageDraw.Draw(image)

    # Table/workspace background.
    draw.rectangle((0, 300, 639, 479), fill=(180, 170, 150))

    # Simple robot/context shape.
    draw.rectangle((40, 60, 120, 270), fill=(90, 90, 100))
    draw.rectangle((90, 180, 220, 230), fill=(110, 110, 120))

    # Main foreground object: reflective metal tool placeholder.
    draw.rounded_rectangle(
        (250, 180, 430, 280),
        radius=18,
        fill=(120, 130, 140),
        outline=(70, 80, 90),
        width=3,
    )

    # Specular highlights.
    draw.line((280, 205, 395, 205), fill=(230, 230, 235), width=4)
    draw.line((300, 230, 410, 230), fill=(210, 210, 220), width=2)

    # Small distractor objects.
    draw.ellipse((470, 250, 510, 290), fill=(80, 120, 180))
    draw.rectangle((520, 200, 570, 245), fill=(160, 100, 90))

    image.save(path)


def create_instance_mask(path: Path) -> None:
    ensure_parent(path)

    mask = Image.new("L", (640, 480), 0)
    draw = ImageDraw.Draw(mask)

    # Object instance ID/value 1.
    draw.rounded_rectangle((250, 180, 430, 280), radius=18, fill=255)

    mask.save(path)


def create_semantic_mask(path: Path) -> None:
    ensure_parent(path)

    mask = Image.new("L", (640, 480), 0)
    draw = ImageDraw.Draw(mask)

    # Semantic category ID 1 = metal_tool.
    draw.rounded_rectangle((250, 180, 430, 280), radius=18, fill=1)

    # Context classes are optional, but useful for sanity inspection.
    # 10 = robot_gripper_or_hand, 11 = table, 12 = distractor_object.
    draw.rectangle((0, 300, 639, 479), fill=11)
    draw.rectangle((40, 60, 120, 270), fill=10)
    draw.rectangle((90, 180, 220, 230), fill=10)
    draw.ellipse((470, 250, 510, 290), fill=12)
    draw.rectangle((520, 200, 570, 245), fill=12)

    mask.save(path)


def make_dummy_row() -> dict[str, object]:
    return {
        "image_id": "img_000001",
        "scene_id": "scene_0001",
        "frame_id": 0,
        "split": "train",
        "image_path": "sim_dataset/images/train/img_000001.png",
        "instance_mask_path": (
            "sim_dataset/masks/instance/train/img_000001_obj_1.png"
        ),
        "semantic_mask_path": "sim_dataset/masks/semantic/train/img_000001.png",
        "category_id": 1,
        "category_name": "metal_tool",
        "object_id": 1,
        "bbox_xmin": 250,
        "bbox_ymin": 180,
        "bbox_xmax": 430,
        "bbox_ymax": 280,
        "point_x": 340,
        "point_y": 230,
        "challenge_primary": "reflective_metal",
        "challenge_secondary": "partial_occlusion",
        "is_reflective": True,
        "is_transparent": False,
        "is_occluded": True,
        "is_small_part": False,
        "is_dynamic": False,
        "camera_name": "front_table",
    }


def write_index_row(index_path: Path, row: dict[str, object], overwrite: bool) -> None:
    ensure_parent(index_path)

    if index_path.exists() and not overwrite:
        existing = pd.read_csv(index_path)

        if not existing.empty and str(row["image_id"]) in set(existing["image_id"]):
            print(f"Keeping existing row for image_id={row['image_id']}")
            return

        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=REQUIRED_SIM_INDEX_COLUMNS)

    df = df[REQUIRED_SIM_INDEX_COLUMNS]
    validate_sim_index(df, image_width=640, image_height=480)

    df.to_csv(index_path, index=False)
    print(f"Saved index:", index_path)


def main() -> None:
    args = parse_args()

    index_path = Path(args.index)
    row = make_dummy_row()

    image_path = Path(str(row["image_path"]))
    instance_mask_path = Path(str(row["instance_mask_path"]))
    semantic_mask_path = Path(str(row["semantic_mask_path"]))

    if args.overwrite or not image_path.exists():
        create_rgb_image(image_path)
        print("Saved RGB image:", image_path)
    else:
        print("Keeping existing RGB image:", image_path)

    if args.overwrite or not instance_mask_path.exists():
        create_instance_mask(instance_mask_path)
        print("Saved instance mask:", instance_mask_path)
    else:
        print("Keeping existing instance mask:", instance_mask_path)

    if args.overwrite or not semantic_mask_path.exists():
        create_semantic_mask(semantic_mask_path)
        print("Saved semantic mask:", semantic_mask_path)
    else:
        print("Keeping existing semantic mask:", semantic_mask_path)

    write_index_row(index_path, row, overwrite=args.overwrite)

    print()
    print("Dummy simulated sample created.")
    print("Image:", image_path)
    print("Instance mask:", instance_mask_path)
    print("Semantic mask:", semantic_mask_path)
    print("Index:", index_path)


if __name__ == "__main__":
    main()
