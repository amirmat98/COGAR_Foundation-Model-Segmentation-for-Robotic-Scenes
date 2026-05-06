"""COGAR-SimRobotics-500 object-index creation from normalized COCO output."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COGAR_SIM_OBJECT_INDEX_COLUMNS = [
    "image_id",
    "file_name",
    "image_path",
    "annotation_id",
    "category_id",
    "category_name",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "primary_challenge",
    "reflective",
    "transparent",
    "occlusion",
    "small_parts",
    "dynamic",
]

METADATA_COLUMNS = [
    "image_id",
    "file_name",
    "primary_challenge",
    "reflective",
    "transparent",
    "occlusion",
    "small_parts",
    "dynamic",
]


@dataclass(frozen=True)
class CogarSimObjectIndexRun:
    """Paths and counts from creating a COGAR-Sim object index."""

    coco_path: Path
    metadata_path: Path
    rgb_dir: Path
    output_csv: Path
    num_images: int
    num_metadata_rows: int
    num_annotations: int
    num_rows: int


def load_coco_annotations(coco_path: str | Path) -> dict[str, Any]:
    """Load a COCO annotation JSON object from disk."""
    path = Path(coco_path)
    if not path.exists():
        raise FileNotFoundError(f"COCO annotations not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata_by_file_name(metadata_path: str | Path) -> dict[str, dict[str, str]]:
    """Load frame metadata keyed by normalized image file basename."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Frame metadata not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Frame metadata CSV is empty: {path}")

    missing = sorted(set(METADATA_COLUMNS) - set(rows[0].keys()))
    if missing:
        raise ValueError(f"Frame metadata is missing columns: {missing}")

    metadata_by_file = {Path(row["file_name"]).name: row for row in rows}
    if len(metadata_by_file) != len(rows):
        raise ValueError("Frame metadata contains duplicate file_name values")

    return metadata_by_file


def validate_cogar_sim_coco(coco: dict[str, Any]) -> None:
    """Validate the minimal COCO fields required for object indexing."""
    required_keys = {"images", "annotations", "categories"}
    missing = sorted(required_keys - set(coco.keys()))
    if missing:
        raise ValueError(f"COCO JSON is missing keys: {missing}")

    category_ids = {int(cat["id"]) for cat in coco["categories"]}
    for annotation in coco["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id not in category_ids:
            raise ValueError(f"Annotation uses unknown category_id: {category_id}")

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Annotation has invalid bbox: {annotation.get('id')}")

        _, _, bbox_w, bbox_h = [float(value) for value in bbox]
        if bbox_w <= 0 or bbox_h <= 0:
            raise ValueError(
                "Annotation bbox must have positive width and height: "
                f"{annotation.get('id')}"
            )


def create_cogar_sim_object_rows(
    coco: dict[str, Any],
    metadata_by_file: dict[str, dict[str, str]],
    rgb_dir: str | Path,
) -> list[dict[str, Any]]:
    """Create object-index rows from COCO annotations and frame metadata."""
    validate_cogar_sim_coco(coco)

    rgb_root = Path(rgb_dir)
    category_name_by_id = {
        int(category["id"]): str(category["name"]) for category in coco["categories"]
    }
    image_by_id = {int(image["id"]): image for image in coco["images"]}

    if len(image_by_id) != len(coco["images"]):
        raise ValueError("COCO images contain duplicate IDs")

    if len(image_by_id) != len(metadata_by_file):
        raise ValueError(
            "COCO image count does not match metadata image count: "
            f"coco={len(image_by_id)}, metadata={len(metadata_by_file)}"
        )

    for image in coco["images"]:
        file_name = Path(str(image["file_name"])).name
        if file_name not in metadata_by_file:
            raise ValueError(f"COCO image missing matching metadata row: {file_name}")

        image_path = rgb_root / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Normalized RGB image not found: {image_path}")

    rows: list[dict[str, Any]] = []

    for annotation in coco["annotations"]:
        image = image_by_id[int(annotation["image_id"])]
        file_name = Path(str(image["file_name"])).name
        metadata = metadata_by_file[file_name]
        bbox_x, bbox_y, bbox_w, bbox_h = [float(value) for value in annotation["bbox"]]
        category_id = int(annotation["category_id"])

        rows.append(
            {
                "image_id": metadata["image_id"],
                "file_name": file_name,
                "image_path": str(rgb_root / file_name),
                "annotation_id": int(annotation["id"]),
                "category_id": category_id,
                "category_name": category_name_by_id[category_id],
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "area": float(annotation.get("area", bbox_w * bbox_h)),
                "primary_challenge": metadata["primary_challenge"],
                "reflective": metadata["reflective"],
                "transparent": metadata["transparent"],
                "occlusion": metadata["occlusion"],
                "small_parts": metadata["small_parts"],
                "dynamic": metadata["dynamic"],
            }
        )

    return rows


def create_cogar_sim_object_index(
    coco_path: str | Path = "data/cogar_sim_500/annotations/instances_all.json",
    metadata_path: str | Path = "data/cogar_sim_500/metadata/frame_index.csv",
    rgb_dir: str | Path = "data/cogar_sim_500/rgb",
    output_csv: str | Path = "outputs/indexes/cogar_sim_500_objects.csv",
) -> CogarSimObjectIndexRun:
    """Create the normalized COGAR-Sim object-level CSV index."""
    resolved_coco_path = Path(coco_path)
    resolved_metadata_path = Path(metadata_path)
    resolved_rgb_dir = Path(rgb_dir)
    resolved_output_csv = Path(output_csv)

    coco = load_coco_annotations(resolved_coco_path)
    metadata_by_file = load_metadata_by_file_name(resolved_metadata_path)
    rows = create_cogar_sim_object_rows(
        coco=coco,
        metadata_by_file=metadata_by_file,
        rgb_dir=resolved_rgb_dir,
    )

    resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COGAR_SIM_OBJECT_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return CogarSimObjectIndexRun(
        coco_path=resolved_coco_path,
        metadata_path=resolved_metadata_path,
        rgb_dir=resolved_rgb_dir,
        output_csv=resolved_output_csv,
        num_images=len(coco["images"]),
        num_metadata_rows=len(metadata_by_file),
        num_annotations=len(coco["annotations"]),
        num_rows=len(rows),
    )
