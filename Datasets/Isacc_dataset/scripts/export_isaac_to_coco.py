#!/usr/bin/env python3
"""Export Isaac BasicWriter instance masks to COCO instance JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - fallback for minimal environments
    mask_utils = None


SKIP_CLASSES = {"BACKGROUND", "UNLABELLED", "", None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Dataset root containing manifest.jsonl and isaac/")
    parser.add_argument("--config", default=None, help="Optional dataset config JSON for category order.")
    parser.add_argument("--output", default=None, help="Output COCO JSON path.")
    parser.add_argument("--min-area", type=int, default=8, help="Skip tiny masks below this pixel area.")
    return parser.parse_args()


def load_config_categories(config_path: str | None) -> list[str]:
    if not config_path:
        return []
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = json.load(f)
    return list(config.get("classes", []))


def parse_color_key(key: str) -> tuple[int, int, int] | None:
    values = [int(v) for v in re.findall(r"\d+", key)]
    if len(values) < 3:
        return None
    return values[0], values[1], values[2]


def color_to_code(color: tuple[int, int, int]) -> int:
    return int(color[0]) | (int(color[1]) << 8) | (int(color[2]) << 16)


def load_color_class_mapping(path: Path) -> dict[tuple[int, int, int], str]:
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, dict[str, Any]] = json.load(f)
    mapping: dict[tuple[int, int, int], str] = {}
    for key, value in raw.items():
        color = parse_color_key(key)
        if color is None:
            continue
        mapping[color] = str(value.get("class", ""))
    return mapping


def encode_instance_image(instance: np.ndarray) -> np.ndarray:
    rgb = instance.astype(np.uint32, copy=False)
    return rgb[:, :, 0] | (rgb[:, :, 1] << 8) | (rgb[:, :, 2] << 16)


def encode_positions_rle(positions: np.ndarray, height: int, width: int) -> dict[str, Any]:
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return {"size": [height, width], "counts": [height * width]}

    counts: list[int] = [int(positions[0])]
    run_length = 1
    previous = int(positions[0])
    for raw_position in positions[1:]:
        position = int(raw_position)
        if position == previous + 1:
            run_length += 1
        else:
            counts.append(run_length)
            counts.append(position - previous - 1)
            run_length = 1
        previous = position
    counts.append(run_length)

    trailing = height * width - previous - 1
    if trailing > 0:
        counts.append(trailing)
    return {"size": [height, width], "counts": counts}


def positions_bbox(positions: np.ndarray, height: int) -> list[float]:
    ys = positions % height
    xs = positions // height
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    flat = np.asfortranarray(mask.astype(np.uint8)).reshape(-1, order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for pixel in flat:
        value = int(pixel)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = value
    counts.append(run_length)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def mask_bbox(mask: np.ndarray) -> tuple[list[float], int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    area = int(mask.sum())
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)], area


def encode_mask(mask: np.ndarray) -> tuple[dict[str, Any], list[float], int]:
    if mask_utils is not None:
        rle_raw = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        bbox = [float(v) for v in mask_utils.toBbox(rle_raw).tolist()]
        area = int(mask_utils.area(rle_raw))
        return {
            "size": [int(v) for v in rle_raw["size"]],
            "counts": rle_raw["counts"].decode("ascii"),
        }, bbox, area

    bbox, area = mask_bbox(mask)
    return encode_uncompressed_rle(mask), bbox, area


def get_rgb_files(isaac_dir: Path) -> list[Path]:
    return sorted(isaac_dir.glob("rgb_*.png"))


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    isaac_dir = dataset_dir / "isaac"
    output_path = Path(args.output) if args.output else dataset_dir / "annotations" / "instances_coco.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    category_names = load_config_categories(args.config)
    category_to_id = {name: idx + 1 for idx, name in enumerate(category_names)}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    rgb_files = get_rgb_files(isaac_dir)
    print("using one-pass positional uncompressed RLE", flush=True)

    for image_id, rgb_path in enumerate(rgb_files, start=1):
        if image_id % 100 == 0:
            print(f"processed {image_id}/{len(rgb_files)} images", flush=True)
        frame = rgb_path.stem.split("_")[-1]
        rgb = Image.open(rgb_path).convert("RGB")
        width, height = rgb.size
        images.append(
            {
                "id": image_id,
                "file_name": str(Path("isaac") / rgb_path.name),
                "width": width,
                "height": height,
            }
        )

        instance_path = isaac_dir / f"instance_segmentation_{frame}.png"
        mapping_path = isaac_dir / f"instance_segmentation_semantics_mapping_{frame}.json"
        if not instance_path.exists() or not mapping_path.exists():
            continue

        instance = np.asarray(Image.open(instance_path).convert("RGB"))
        color_to_class = load_color_class_mapping(mapping_path)
        code_to_class = {color_to_code(color): class_name for color, class_name in color_to_class.items()}
        flat_codes = np.ravel(encode_instance_image(instance), order="F")
        order = np.argsort(flat_codes, kind="stable")
        sorted_codes = flat_codes[order]
        unique_codes, starts, counts = np.unique(sorted_codes, return_index=True, return_counts=True)

        for code, start, count in zip(unique_codes, starts, counts):
            class_name = code_to_class.get(int(code))
            if class_name in SKIP_CLASSES:
                continue
            if class_name not in category_to_id:
                category_to_id[class_name] = len(category_to_id) + 1

            area = int(count)
            if area < args.min_area:
                continue
            positions = order[int(start) : int(start) + area]
            segmentation = encode_positions_rle(positions, height, width)
            bbox = positions_bbox(positions, height)

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[class_name],
                    "segmentation": segmentation,
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    categories = [{"id": category_id, "name": name, "supercategory": "robotic_scene"} for name, category_id in sorted(category_to_id.items(), key=lambda item: item[1])]
    coco = {
        "info": {
            "description": "Isaac Sim robotic segmentation dataset exported to COCO",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"wrote {output_path}")
    print(f"images={len(images)} annotations={len(annotations)} categories={len(categories)}")


if __name__ == "__main__":
    main()
