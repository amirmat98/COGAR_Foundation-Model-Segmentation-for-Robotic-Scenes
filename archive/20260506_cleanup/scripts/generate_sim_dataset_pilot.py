"""Generate a tiny deterministic simulated-dataset pilot.

This is a lightweight fallback/pipeline smoke test, not a replacement for
Isaac/Gazebo-generated data. It writes RGB images, instance masks, semantic
masks, metadata, and an object-level index that matches the project schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from cogar_seg.datasets.sim_robotic import REQUIRED_SIM_INDEX_COLUMNS, validate_sim_index


CHALLENGE_FLAGS = {
    "reflective_metal": {
        "is_reflective": True,
        "is_transparent": False,
        "is_occluded": False,
        "is_small_part": False,
        "is_dynamic": False,
    },
    "transparent_glass": {
        "is_reflective": False,
        "is_transparent": True,
        "is_occluded": False,
        "is_small_part": False,
        "is_dynamic": False,
    },
    "partial_occlusion": {
        "is_reflective": False,
        "is_transparent": False,
        "is_occluded": True,
        "is_small_part": False,
        "is_dynamic": False,
    },
    "small_parts": {
        "is_reflective": False,
        "is_transparent": False,
        "is_occluded": False,
        "is_small_part": True,
        "is_dynamic": False,
    },
    "dynamic_scene": {
        "is_reflective": False,
        "is_transparent": False,
        "is_occluded": False,
        "is_small_part": False,
        "is_dynamic": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small deterministic pilot dataset for schema checks."
    )
    parser.add_argument(
        "--config",
        default="configs/sim_dataset.yaml",
        help="Path to the simulated dataset YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory. Defaults to dataset.output_dir from config.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of pilot images to generate.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split for pilot images.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")

    return config


def category_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    categories = config.get("categories", [])
    return {str(category["name"]): category for category in categories}


def ensure_dirs(output_dir: Path, split: str) -> None:
    for relative in [
        f"rgb/{split}",
        f"instance_masks/{split}",
        f"semantic_masks/{split}",
        "annotations",
        "metadata",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def draw_pilot_object(
    image: np.ndarray,
    instance_mask: np.ndarray,
    semantic_mask: np.ndarray,
    challenge: str,
    category_id: int,
    object_id: int,
    frame_index: int,
) -> tuple[int, int, int, int]:
    height, width = instance_mask.shape

    center_x = int(width * 0.35 + frame_index * 22)
    center_y = int(height * 0.48)

    if challenge == "small_parts":
        half_w, half_h = 18, 10
    else:
        half_w, half_h = 82, 42

    xmin = max(0, center_x - half_w)
    ymin = max(0, center_y - half_h)
    xmax = min(width - 1, center_x + half_w)
    ymax = min(height - 1, center_y + half_h)

    if challenge == "transparent_glass":
        color = (190, 220, 235)
    elif challenge == "reflective_metal":
        color = (205, 205, 190)
    elif challenge == "small_parts":
        color = (45, 45, 45)
    else:
        color = (70, 130, 180)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness=-1)
    cv2.rectangle(instance_mask, (xmin, ymin), (xmax, ymax), object_id, thickness=-1)
    cv2.rectangle(semantic_mask, (xmin, ymin), (xmax, ymax), category_id, thickness=-1)

    if challenge == "partial_occlusion":
        occluder_xmin = xmin + int((xmax - xmin) * 0.55)
        cv2.rectangle(
            image,
            (occluder_xmin, ymin - 8),
            (xmax + 18, ymax + 8),
            (35, 35, 35),
            thickness=-1,
        )
        cv2.rectangle(
            instance_mask,
            (occluder_xmin, ymin),
            (xmax, ymax),
            0,
            thickness=-1,
        )
        cv2.rectangle(
            semantic_mask,
            (occluder_xmin, ymin),
            (xmax, ymax),
            0,
            thickness=-1,
        )
        xmax = occluder_xmin - 1

    return xmin, ymin, xmax, ymax


def write_categories(config: dict[str, Any], output_dir: Path) -> None:
    categories_path = output_dir / "annotations" / "categories.json"
    with categories_path.open("w") as f:
        json.dump(config.get("categories", []), f, indent=2)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    dataset_cfg = config["dataset"]
    output_dir = Path(args.output_dir or dataset_cfg["output_dir"])
    image_width = int(dataset_cfg["image_width"])
    image_height = int(dataset_cfg["image_height"])
    split = args.split

    ensure_dirs(output_dir, split)
    write_categories(config, output_dir)

    categories = category_lookup(config)
    category = categories.get("metal_part", {"id": 3, "name": "metal_part"})
    category_id = int(category["id"])
    category_name = str(category["name"])
    challenges = list(CHALLENGE_FLAGS.keys())

    rows: list[dict[str, Any]] = []

    for frame_index in range(args.num_images):
        challenge = challenges[frame_index % len(challenges)]
        image_id = f"pilot_{frame_index:06d}"
        scene_id = f"pilot_scene_{frame_index // len(challenges):04d}"
        object_id = 1

        image = np.full((image_height, image_width, 3), 155, dtype=np.uint8)
        instance_mask = np.zeros((image_height, image_width), dtype=np.uint16)
        semantic_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        cv2.rectangle(
            image,
            (0, int(image_height * 0.62)),
            (image_width, image_height),
            (95, 100, 105),
            thickness=-1,
        )

        xmin, ymin, xmax, ymax = draw_pilot_object(
            image=image,
            instance_mask=instance_mask,
            semantic_mask=semantic_mask,
            challenge=challenge,
            category_id=category_id,
            object_id=object_id,
            frame_index=frame_index,
        )

        point_x = int((xmin + xmax) / 2)
        point_y = int((ymin + ymax) / 2)

        image_path = output_dir / "rgb" / split / f"{image_id}.png"
        instance_mask_path = (
            output_dir / "instance_masks" / split / f"{image_id}_obj_{object_id:04d}.png"
        )
        semantic_mask_path = output_dir / "semantic_masks" / split / f"{image_id}.png"

        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(instance_mask_path), instance_mask)
        cv2.imwrite(str(semantic_mask_path), semantic_mask)

        row = {
            "image_id": image_id,
            "scene_id": scene_id,
            "frame_id": frame_index,
            "split": split,
            "image_path": str(image_path),
            "instance_mask_path": str(instance_mask_path),
            "semantic_mask_path": str(semantic_mask_path),
            "category_id": category_id,
            "category_name": category_name,
            "object_id": object_id,
            "bbox_xmin": xmin,
            "bbox_ymin": ymin,
            "bbox_xmax": xmax,
            "bbox_ymax": ymax,
            "point_x": point_x,
            "point_y": point_y,
            "challenge_primary": challenge,
            "challenge_secondary": "",
            "camera_name": "pilot_front",
            **CHALLENGE_FLAGS[challenge],
        }
        rows.append(row)

    index = pd.DataFrame(rows, columns=REQUIRED_SIM_INDEX_COLUMNS)
    validate_sim_index(
        index,
        image_width=image_width,
        image_height=image_height,
        allowed_category_ids={int(category["id"]) for category in config.get("categories", [])},
    )

    index_path = output_dir / "annotations" / "sim_robotic_scenes_index.csv"
    index.to_csv(index_path, index=False)

    summary = {
        "dataset_name": dataset_cfg["name"],
        "num_images": int(index["image_id"].nunique()),
        "num_object_instances": int(len(index)),
        "split": split,
        "index_path": str(index_path),
    }
    with (output_dir / "metadata" / "generation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved pilot dataset to: {output_dir}")
    print(f"Saved index: {index_path}")
    print(f"Rows: {len(index)}")


if __name__ == "__main__":
    main()
