from pathlib import Path
import csv

import cv2
import numpy as np


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT


def filter_object_index(
    input_csv: Path,
    output_csv: Path,
    min_area: int = 500,
    max_area_ratio: float = 0.08,
    max_bbox_area_ratio: float = 0.15,
) -> int:
    """
    Filter object rows to remove tiny regions and large background/table-like regions.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    filtered_rows = []

    with open(input_csv, "r") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        area = int(row["area"])

        xmin = int(row["bbox_xmin"])
        ymin = int(row["bbox_ymin"])
        xmax = int(row["bbox_xmax"])
        ymax = int(row["bbox_ymax"])

        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        bbox_area = bbox_width * bbox_height

        area_ratio = area / IMAGE_AREA
        bbox_area_ratio = bbox_area / IMAGE_AREA
        bbox_width_ratio = bbox_width / IMAGE_WIDTH
        bbox_height_ratio = bbox_height / IMAGE_HEIGHT

        if area < min_area:
            continue

        if area_ratio > max_area_ratio:
            continue

        if bbox_area_ratio > max_bbox_area_ratio:
            continue

        if bbox_width_ratio > 0.75 and bbox_height_ratio < 0.30:
            continue

        row["area_ratio"] = f"{area_ratio:.6f}"
        row["bbox_area_ratio"] = f"{bbox_area_ratio:.6f}"
        row["bbox_width_ratio"] = f"{bbox_width_ratio:.6f}"
        row["bbox_height_ratio"] = f"{bbox_height_ratio:.6f}"

        filtered_rows.append(row)

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
        "area_ratio",
        "bbox_area_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    return len(filtered_rows)


def make_binary_mask_filename(row_index: int, file_name: str, object_id: int) -> str:
    stem = Path(file_name).stem
    return f"{row_index:05d}_{stem}_obj{object_id}.png"


def export_binary_gt_masks(
    input_csv: Path,
    output_csv: Path,
    output_mask_dir: Path,
) -> int:
    """
    Export one binary ground-truth mask for each object row.

    Mask convention:
    - 255 = target object
    - 0   = background
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    updated_rows = []

    with open(input_csv, "r") as f:
        rows = list(csv.DictReader(f))

    for row_index, row in enumerate(rows):
        label_path = row["label_path"]
        file_name = row["file_name"]
        object_id = int(row["object_id"])

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if label is None:
            print(f"Warning: could not read label mask: {label_path}")
            continue

        binary_mask = label == object_id
        binary_mask_uint8 = binary_mask.astype(np.uint8) * 255

        mask_filename = make_binary_mask_filename(row_index, file_name, object_id)
        mask_path = output_mask_dir / mask_filename

        success = cv2.imwrite(str(mask_path), binary_mask_uint8)

        if not success:
            print(f"Warning: could not write mask: {mask_path}")
            continue

        row["binary_mask_path"] = str(mask_path)
        updated_rows.append(row)

    fieldnames = list(updated_rows[0].keys())

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return len(updated_rows)