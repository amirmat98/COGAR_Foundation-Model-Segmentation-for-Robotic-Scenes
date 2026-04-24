from pathlib import Path
import csv
from typing import Any

import cv2
import numpy as np


def get_rgb_label_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """
    Return:
    - sequence path
    - RGB directory
    - label directory
    """
    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])

    seq_path = ocid_root / sequence
    rgb_dir = seq_path / config["rgb_folder_name"]
    label_dir = seq_path / config["label_folder_name"]

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    return seq_path, rgb_dir, label_dir


def create_image_index(config: dict[str, Any], output_csv: Path) -> int:
    """
    Create an image-level CSV index:
    one row per RGB image and matching label mask.
    """
    _, rgb_dir, label_dir = get_rgb_label_dirs(config)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rgb_files = sorted(rgb_dir.glob("*.png"))

    for rgb_path in rgb_files:
        label_path = label_dir / rgb_path.name

        if not label_path.exists():
            print(f"Skipping {rgb_path.name}: matching label not found")
            continue

        rows.append(
            {
                "image_path": str(rgb_path),
                "label_path": str(label_path),
                "sequence": config["ocid_debug_sequence"],
                "file_name": rgb_path.name,
            }
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label_path", "sequence", "file_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def compute_object_properties(label: np.ndarray, object_id: int) -> dict[str, int] | None:
    """
    For one object ID inside a label image, compute:
    - area
    - bounding box
    - point prompt
    """
    binary_mask = label == object_id

    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    area = int(binary_mask.sum())

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    point_x = int(xs.mean())
    point_y = int(ys.mean())

    return {
        "area": area,
        "bbox_xmin": xmin,
        "bbox_ymin": ymin,
        "bbox_xmax": xmax,
        "bbox_ymax": ymax,
        "point_x": point_x,
        "point_y": point_y,
    }


def create_object_index(image_index_csv: Path, output_csv: Path) -> int:
    """
    Create an object-level CSV:
    one row per object instance.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    object_rows = []

    with open(image_index_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label_path = row["label_path"]
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            if label is None:
                print(f"Warning: could not read label mask: {label_path}")
                continue

            unique_ids = np.unique(label)

            for object_id in unique_ids:
                object_id = int(object_id)

                if object_id == 0:
                    continue

                props = compute_object_properties(label, object_id)

                if props is None:
                    continue

                object_rows.append(
                    {
                        "image_path": row["image_path"],
                        "label_path": row["label_path"],
                        "sequence": row["sequence"],
                        "file_name": row["file_name"],
                        "object_id": object_id,
                        **props,
                    }
                )

    fieldnames = [
        "image_path",
        "label_path",
        "sequence",
        "file_name",
        "object_id",
        "area",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "point_x",
        "point_y",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    return len(object_rows)