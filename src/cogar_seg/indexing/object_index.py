"""Object-index creation workflows built on dataset-specific index helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cogar_seg.config import get_outputs_dir, load_config
from cogar_seg.datasets.ocid import (
    create_full_image_index,
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


@dataclass(frozen=True)
class OcidFullIndexPaths:
    """Canonical output paths for the full OCID indexing pipeline."""

    outputs_dir: Path
    index_dir: Path
    mask_dir: Path
    image_index_csv: Path
    object_index_csv: Path
    filtered_object_index_csv: Path
    final_object_index_csv: Path


@dataclass(frozen=True)
class OcidFullIndexRun:
    """Counts and paths from preparing the full OCID object benchmark."""

    paths: OcidFullIndexPaths
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


def build_ocid_full_index_paths(config: dict) -> OcidFullIndexPaths:
    """Build canonical paths for full-OCID indexes and exported binary masks."""
    outputs_dir = get_outputs_dir(config) / "ocid_full"
    index_dir = outputs_dir / "indexes"
    mask_dir = outputs_dir / "gt_binary_masks"

    return OcidFullIndexPaths(
        outputs_dir=outputs_dir,
        index_dir=index_dir,
        mask_dir=mask_dir,
        image_index_csv=index_dir / "ocid_full_images.csv",
        object_index_csv=index_dir / "ocid_full_objects.csv",
        filtered_object_index_csv=index_dir / "ocid_full_objects_filtered.csv",
        final_object_index_csv=index_dir / "ocid_full_objects_filtered_with_masks.csv",
    )


def create_ocid_object_index(
    config_path: str | Path,
    progress: bool = False,
    progress_every: int = 100,
    debug: bool = False,
    strict: bool = False,
) -> tuple[OcidDebugIndexPaths, int, int, int]:
    """Create image, object, and filtered object indexes for the OCID debug sequence."""
    config = load_config(config_path)
    paths = build_ocid_debug_index_paths(config)

    num_images = create_image_index(config, paths.image_index_csv)
    num_objects = create_object_index(
        paths.image_index_csv,
        paths.object_index_csv,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )
    num_filtered = filter_object_index(
        paths.object_index_csv,
        paths.filtered_object_index_csv,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
    )

    return paths, num_images, num_objects, num_filtered


def prepare_ocid_debug_dataset(
    config_path: str | Path,
    progress: bool = False,
    progress_every: int = 100,
    debug: bool = False,
    strict: bool = False,
) -> OcidDebugIndexRun:
    """Create OCID debug indexes and export binary ground-truth masks."""
    paths, num_images, num_objects, num_filtered = create_ocid_object_index(
        config_path,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )
    num_masks = export_binary_gt_masks(
        input_csv=paths.filtered_object_index_csv,
        output_csv=paths.final_object_index_csv,
        output_mask_dir=paths.mask_dir,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )

    return OcidDebugIndexRun(
        paths=paths,
        num_images=num_images,
        num_objects=num_objects,
        num_filtered_objects=num_filtered,
        num_masks=num_masks,
    )


def prepare_ocid_full_dataset(
    config_path: str | Path,
    progress: bool = False,
    progress_every: int = 250,
    debug: bool = False,
    strict: bool = False,
) -> OcidFullIndexRun:
    """Create full OCID image/object indexes and exported binary GT masks."""
    config = load_config(config_path)
    paths = build_ocid_full_index_paths(config)

    rgb_folder_name = config.get("rgb_folder_name", "rgb")
    label_folder_name = config.get("label_folder_name", "label")
    min_area = int(config.get("ocid_min_area", 500))
    max_area_ratio = float(config.get("ocid_max_area_ratio", 0.08))
    max_bbox_area_ratio = float(config.get("ocid_max_bbox_area_ratio", 0.15))

    num_images = create_full_image_index(
        ocid_root=config["ocid_root"],
        output_csv=paths.image_index_csv,
        rgb_folder_name=rgb_folder_name,
        label_folder_name=label_folder_name,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )
    num_objects = create_object_index(
        paths.image_index_csv,
        paths.object_index_csv,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )
    num_filtered = filter_object_index(
        input_csv=paths.object_index_csv,
        output_csv=paths.filtered_object_index_csv,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        max_bbox_area_ratio=max_bbox_area_ratio,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
    )
    num_masks = export_binary_gt_masks(
        input_csv=paths.filtered_object_index_csv,
        output_csv=paths.final_object_index_csv,
        output_mask_dir=paths.mask_dir,
        progress=progress,
        progress_every=progress_every,
        debug=debug,
        strict=strict,
    )

    return OcidFullIndexRun(
        paths=paths,
        num_images=num_images,
        num_objects=num_objects,
        num_filtered_objects=num_filtered,
        num_masks=num_masks,
    )
