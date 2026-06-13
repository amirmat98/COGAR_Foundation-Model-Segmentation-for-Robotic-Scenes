"""Metadata writers for BlenderProc dataset generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OCID_LIKE_FIELDNAMES = [
    "image_id",
    "file_name",
    "candidate_image_id",
    "accepted_image_id",
    "scene_id",
    "scene_family",
    "primary_challenge",
    "challenge_primary",
    "reflective",
    "transparent",
    "occlusion",
    "small_parts",
    "dynamic",
    "num_objects",
    "num_requested_objects",
    "num_created_objects",
    "occlusion_level",
    "seed",
    "generator_version",
    "camera_view",
    "lighting_condition",
]

STATIC_PILOT_FIELDNAMES = [
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
]


def write_categories(metadata_dir: Path, categories: list[dict[str, Any]]) -> Path:
    """Write semantic category metadata and return its path."""
    categories_path = metadata_dir / "categories.json"
    with categories_path.open("w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)
    return categories_path


def write_ocid_like_metadata(
    output_root: Path,
    categories: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    """Write frame metadata for the randomized OCID-like BlenderProc generator."""
    metadata_dir = output_root / "metadata"
    splits_dir = output_root / "splits"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metadata_dir / "frame_index_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OCID_LIKE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    categories_path = write_categories(metadata_dir, categories)

    split_path = splits_dir / "all_raw.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['image_id']:06d}\n")

    print(f"[OK] Metadata CSV: {csv_path}")
    print(f"[OK] Categories JSON: {categories_path}")
    print(f"[OK] Raw split: {split_path}")


def write_static_pilot_metadata(
    output_root: Path,
    categories: list[dict[str, Any]],
    num_images: int,
) -> None:
    """Write frame metadata for the fixed-camera pilot generator."""
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
        rows.append(
            {
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
            }
        )

    csv_path = metadata_dir / "frame_index_pilot.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATIC_PILOT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    categories_path = write_categories(metadata_dir, categories)

    split_path = splits_dir / "pilot.txt"
    with split_path.open("w", encoding="utf-8") as f:
        for i in range(num_images):
            f.write(f"{i + 1:06d}\n")

    print(f"[OK] Metadata CSV: {csv_path}")
    print(f"[OK] Categories JSON: {categories_path}")
    print(f"[OK] Pilot split: {split_path}")
