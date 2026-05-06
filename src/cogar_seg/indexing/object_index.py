"""Object-index creation workflows built on dataset-specific index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cogar_seg.config import get_outputs_dir, load_config
from cogar_seg.datasets.ocid import (
    create_image_index,
    create_object_index,
    export_binary_gt_masks,
    filter_object_index,
)


@dataclass(frozen=True)
class OcidDebugIndexPaths:
    """Canonical output paths for the OCID debug indexing pipeline."""

    outputs_dir: Path
    index_dir: Path
    mask_dir: Path
    image_index_csv: Path
    object_index_csv: Path
    filtered_object_index_csv: Path
    final_object_index_csv: Path


@dataclass(frozen=True)
class OcidDebugIndexRun:
    """Counts and paths from preparing OCID debug indexes and masks."""

    paths: OcidDebugIndexPaths
    num_images: int
    num_objects: int
    num_filtered_objects: int
    num_masks: int


def build_ocid_debug_index_paths(config: dict) -> OcidDebugIndexPaths:
    """Build the canonical OCID debug index and mask output paths."""
    outputs_dir = get_outputs_dir(config)
    index_dir = outputs_dir / "indexes"
    mask_dir = outputs_dir / "gt_binary_masks"

    return OcidDebugIndexPaths(
        outputs_dir=outputs_dir,
        index_dir=index_dir,
        mask_dir=mask_dir,
        image_index_csv=index_dir / "ocid_debug_seq21.csv",
        object_index_csv=index_dir / "ocid_debug_seq21_objects.csv",
        filtered_object_index_csv=index_dir / "ocid_debug_seq21_objects_filtered.csv",
        final_object_index_csv=(
            index_dir / "ocid_debug_seq21_objects_filtered_with_masks.csv"
        ),
    )


def create_ocid_object_index(config_path: str | Path) -> tuple[OcidDebugIndexPaths, int, int, int]:
    """Create image, object, and filtered object indexes for the OCID debug sequence."""
    config = load_config(config_path)
    paths = build_ocid_debug_index_paths(config)

    num_images = create_image_index(config, paths.image_index_csv)
    num_objects = create_object_index(paths.image_index_csv, paths.object_index_csv)
    num_filtered = filter_object_index(
        paths.object_index_csv,
        paths.filtered_object_index_csv,
    )

    return paths, num_images, num_objects, num_filtered


def prepare_ocid_debug_dataset(config_path: str | Path) -> OcidDebugIndexRun:
    """Create OCID debug indexes and export binary ground-truth masks."""
    paths, num_images, num_objects, num_filtered = create_ocid_object_index(config_path)
    num_masks = export_binary_gt_masks(
        input_csv=paths.filtered_object_index_csv,
        output_csv=paths.final_object_index_csv,
        output_mask_dir=paths.mask_dir,
    )

    return OcidDebugIndexRun(
        paths=paths,
        num_images=num_images,
        num_objects=num_objects,
        num_filtered_objects=num_filtered,
        num_masks=num_masks,
    )
