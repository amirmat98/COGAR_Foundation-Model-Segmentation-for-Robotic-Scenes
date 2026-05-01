"""OCID indexing and ground-truth mask utilities."""

from pathlib import Path
import csv
from typing import Any

import numpy as np

from cogar_seg.paths import resolve_ocid_sequence_path

OCID_IMAGE_WIDTH = 640
OCID_IMAGE_HEIGHT = 480


def get_rgb_label_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Return the sequence path, RGB directory, and label directory."""
    seq_path = resolve_ocid_sequence_path(config)
    rgb_dir = seq_path / config["rgb_folder_name"]
    label_dir = seq_path / config["label_folder_name"]

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    return seq_path, rgb_dir, label_dir


def create_image_index(config: dict[str, Any], output_csv: Path) -> int:
    """Create an image-level CSV index with one RGB/label pair per row."""
    _, rgb_dir, label_dir = get_rgb_label_dirs(config)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for rgb_path in sorted(rgb_dir.glob("*.png")):
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

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label_path", "sequence", "file_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def compute_object_properties(label: np.ndarray, object_id: int) -> dict[str, int] | None:
    """Compute area, bounding box, and centroid point for one object ID."""
    binary_mask = label == object_id
    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return {
        "area": int(binary_mask.sum()),
        "bbox_xmin": int(xs.min()),
        "bbox_ymin": int(ys.min()),
        "bbox_xmax": int(xs.max()),
        "bbox_ymax": int(ys.max()),
        "point_x": int(xs.mean()),
        "point_y": int(ys.mean()),
    }


def create_object_index(image_index_csv: Path, output_csv: Path) -> int:
    """Create an object-level CSV with one row per object instance."""
    import cv2

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    object_rows = []

    with image_index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label_path = row["label_path"]
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            if label is None:
                print(f"Warning: could not read label mask: {label_path}")
                continue

            for object_id_value in np.unique(label):
                object_id = int(object_id_value)

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

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    return len(object_rows)


def filter_object_index(
    input_csv: Path,
    output_csv: Path,
    min_area: int = 500,
    max_area_ratio: float = 0.08,
    max_bbox_area_ratio: float = 0.15,
    image_width: int = OCID_IMAGE_WIDTH,
    image_height: int = OCID_IMAGE_HEIGHT,
) -> int:
    """Filter object rows to remove tiny regions and large table-like regions."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_area = image_width * image_height
    filtered_rows = []

    with input_csv.open("r", newline="") as f:
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

        area_ratio = area / image_area
        bbox_area_ratio = bbox_area / image_area
        bbox_width_ratio = bbox_width / image_width
        bbox_height_ratio = bbox_height / image_height

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

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    return len(filtered_rows)


def make_binary_mask_filename(row_index: int, file_name: str, object_id: int) -> str:
    """Create the deterministic binary-mask filename used by the OCID pipeline."""
    stem = Path(file_name).stem
    return f"{row_index:05d}_{stem}_obj{object_id}.png"


def export_binary_gt_masks(
    input_csv: Path,
    output_csv: Path,
    output_mask_dir: Path,
) -> int:
    """Export one binary ground-truth mask for each object row."""
    import cv2

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    updated_rows = []

    with input_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    for row_index, row in enumerate(rows):
        label_path = row["label_path"]
        file_name = row["file_name"]
        object_id = int(row["object_id"])

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if label is None:
            print(f"Warning: could not read label mask: {label_path}")
            continue

        binary_mask_uint8 = (label == object_id).astype(np.uint8) * 255
        mask_filename = make_binary_mask_filename(row_index, file_name, object_id)
        mask_path = output_mask_dir / mask_filename

        success = cv2.imwrite(str(mask_path), binary_mask_uint8)

        if not success:
            print(f"Warning: could not write mask: {mask_path}")
            continue

        row["binary_mask_path"] = str(mask_path)
        updated_rows.append(row)

    if not updated_rows:
        raise RuntimeError(f"No binary masks were exported from {input_csv}")

    fieldnames = list(updated_rows[0].keys())

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return len(updated_rows)
