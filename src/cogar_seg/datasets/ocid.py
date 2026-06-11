"""OCID indexing and ground-truth mask utilities."""

from pathlib import Path
import csv
import time
from typing import Any

import numpy as np

from cogar_seg.cv_compat import cv2
from cogar_seg.paths import resolve_ocid_sequence_path

OCID_IMAGE_WIDTH = 640
OCID_IMAGE_HEIGHT = 480
OCID_OBJECT_FIELDS = [
    "object_id",
    "area",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "point_x",
    "point_y",
]
OCID_RATIO_FIELDS = [
    "area_ratio",
    "bbox_area_ratio",
    "bbox_width_ratio",
    "bbox_height_ratio",
]


def _emit_progress(enabled: bool, message: str) -> None:
    """Print a progress/debug message when progress output is enabled."""
    if enabled:
        print(message, flush=True)


def _should_report_progress(count: int, total: int | None, progress_every: int) -> bool:
    """Return whether a loop should emit a progress line for this count."""
    if count <= 0:
        return False
    if count == 1:
        return True
    if total is not None and count == total:
        return True
    return progress_every > 0 and count % progress_every == 0


def _elapsed_s(start_time: float) -> str:
    """Format elapsed seconds for progress messages."""
    return f"{time.perf_counter() - start_time:.1f}s"


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


def normalize_label_array(label: np.ndarray) -> np.ndarray:
    """Return a 2D integer instance-label array from an OCID label image."""
    if label.ndim == 2:
        return label

    if label.ndim == 3:
        if label.shape[2] == 1:
            return label[:, :, 0]

        first = label[:, :, 0]
        if np.all(label[:, :, : label.shape[2]] == first[:, :, None]):
            return first

        # Fallback for color-coded labels: pack BGR/RGB channels into a stable ID.
        channels = label[:, :, :3].astype(np.uint32)
        return (
            channels[:, :, 0]
            + (channels[:, :, 1] << 8)
            + (channels[:, :, 2] << 16)
        ).astype(np.uint32)

    raise ValueError(f"Unsupported OCID label shape: {label.shape}")


def discover_ocid_sequences(
    ocid_root: str | Path,
    rgb_folder_name: str = "rgb",
    label_folder_name: str = "label",
) -> list[Path]:
    """Find OCID sequence directories containing matching RGB and label folders."""
    root = Path(ocid_root)
    if not root.exists():
        raise FileNotFoundError(f"OCID root does not exist: {root}")

    sequences = []
    for rgb_dir in root.glob(f"*/**/{rgb_folder_name}"):
        seq_dir = rgb_dir.parent
        label_dir = seq_dir / label_folder_name
        if label_dir.is_dir():
            sequences.append(seq_dir)

    return sorted(sequences)


def parse_sequence_metadata(sequence_path: Path, ocid_root: str | Path) -> dict[str, str]:
    """Parse common OCID path components into metadata fields."""
    relative = sequence_path.relative_to(Path(ocid_root))
    parts = relative.parts
    metadata = {
        "sequence": str(relative),
        "object_set": parts[0] if len(parts) > 0 else "",
        "surface": parts[1] if len(parts) > 1 else "",
        "camera_view": parts[2] if len(parts) > 2 else "",
        "scene_type": parts[3] if len(parts) > 3 else "",
        "sequence_id": parts[4] if len(parts) > 4 else sequence_path.name,
    }
    return metadata


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


def create_full_image_index(
    ocid_root: str | Path,
    output_csv: str | Path,
    rgb_folder_name: str = "rgb",
    label_folder_name: str = "label",
    progress: bool = False,
    progress_every: int = 100,
    debug: bool = False,
    strict: bool = False,
) -> int:
    """Create a full-OCID image-level CSV from every discovered sequence."""
    root = Path(ocid_root)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    _emit_progress(progress, f"[OCID] Scanning sequences under: {root}")
    sequences = discover_ocid_sequences(
        root,
        rgb_folder_name=rgb_folder_name,
        label_folder_name=label_folder_name,
    )
    _emit_progress(progress, f"[OCID] Found {len(sequences):,} sequence directories")
    if strict and not sequences:
        raise RuntimeError(f"No OCID sequences found under: {root}")
    if debug:
        for sequence_path in sequences[:10]:
            _emit_progress(progress, f"[OCID][debug] sequence: {sequence_path}")
        if len(sequences) > 10:
            _emit_progress(progress, f"[OCID][debug] ... {len(sequences) - 10:,} more sequences")

    rows = []
    image_id = 0
    missing_labels = 0
    total_frames_seen = 0
    for sequence_index, sequence_path in enumerate(sequences, start=1):
        metadata = parse_sequence_metadata(sequence_path, root)
        rgb_dir = sequence_path / rgb_folder_name
        label_dir = sequence_path / label_folder_name
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        total_frames_seen += len(rgb_paths)
        if debug or _should_report_progress(sequence_index, len(sequences), progress_every):
            _emit_progress(
                progress,
                (
                    f"[OCID] Sequence {sequence_index:,}/{len(sequences):,}: "
                    f"{metadata['sequence']} ({len(rgb_paths):,} RGB frames)"
                ),
            )

        for frame_index, rgb_path in enumerate(rgb_paths):
            label_path = label_dir / rgb_path.name
            if not label_path.exists():
                missing_labels += 1
                message = f"Skipping {rgb_path}: matching label not found"
                if strict:
                    raise FileNotFoundError(message)
                _emit_progress(progress, f"[OCID][warn] {message}")
                continue

            rows.append(
                {
                    "image_id": image_id,
                    "frame_index": frame_index,
                    "image_path": str(rgb_path),
                    "label_path": str(label_path),
                    "file_name": rgb_path.name,
                    **metadata,
                }
            )
            image_id += 1
            if _should_report_progress(image_id, None, progress_every):
                _emit_progress(
                    progress,
                    f"[OCID] Indexed {image_id:,} RGB-label pairs in {_elapsed_s(start_time)}",
                )

    if strict and not rows:
        raise RuntimeError(f"No RGB-label pairs were indexed from OCID root: {root}")

    fieldnames = [
        "image_id",
        "frame_index",
        "image_path",
        "label_path",
        "file_name",
        "sequence",
        "object_set",
        "surface",
        "camera_view",
        "scene_type",
        "sequence_id",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _emit_progress(
        progress,
        (
            f"[OCID] Wrote image index: {output_csv} "
            f"({len(rows):,}/{total_frames_seen:,} pairs, "
            f"missing_labels={missing_labels:,}, elapsed={_elapsed_s(start_time)})"
        ),
    )
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


def create_object_index(
    image_index_csv: Path,
    output_csv: Path,
    progress: bool = False,
    progress_every: int = 100,
    debug: bool = False,
    strict: bool = False,
) -> int:
    """Create an object-level CSV with one row per object instance."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    object_rows = []
    start_time = time.perf_counter()

    with image_index_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        image_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    _emit_progress(
        progress,
        f"[OCID] Building object index from {len(rows):,} image rows: {image_index_csv}",
    )
    if strict and not rows:
        raise RuntimeError(f"Image index is empty: {image_index_csv}")

    unreadable_labels = 0
    for row_index, row in enumerate(rows, start=1):
        label_path = row["label_path"]
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if label is None:
            unreadable_labels += 1
            message = f"could not read label mask: {label_path}"
            if strict:
                raise FileNotFoundError(message)
            _emit_progress(progress, f"[OCID][warn] {message}")
            continue
        label = normalize_label_array(label)

        object_ids = np.unique(label)
        if debug and _should_report_progress(row_index, len(rows), progress_every):
            _emit_progress(
                progress,
                (
                    f"[OCID][debug] row {row_index:,}/{len(rows):,}: "
                    f"{len(object_ids):,} raw labels in {label_path}"
                ),
            )

        for object_id_value in object_ids:
            object_id = int(object_id_value)

            if object_id == 0:
                continue

            props = compute_object_properties(label, object_id)

            if props is None:
                continue

            object_rows.append(
                {
                    **row,
                    "object_id": object_id,
                    "category_name": row.get("scene_type", "ocid_object") or "ocid_object",
                    **props,
                }
            )

        if _should_report_progress(row_index, len(rows), progress_every):
            _emit_progress(
                progress,
                (
                    f"[OCID] Processed {row_index:,}/{len(rows):,} labels; "
                    f"objects={len(object_rows):,}; elapsed={_elapsed_s(start_time)}"
                ),
            )

    if strict and not object_rows:
        raise RuntimeError(f"No object rows were created from image index: {image_index_csv}")

    fieldnames = list(image_fieldnames)
    for field in ["category_name", *OCID_OBJECT_FIELDS]:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    _emit_progress(
        progress,
        (
            f"[OCID] Wrote object index: {output_csv} "
            f"(objects={len(object_rows):,}, unreadable_labels={unreadable_labels:,}, "
            f"elapsed={_elapsed_s(start_time)})"
        ),
    )
    return len(object_rows)


def filter_object_index(
    input_csv: Path,
    output_csv: Path,
    min_area: int = 500,
    max_area_ratio: float = 0.08,
    max_bbox_area_ratio: float = 0.15,
    image_width: int = OCID_IMAGE_WIDTH,
    image_height: int = OCID_IMAGE_HEIGHT,
    progress: bool = False,
    progress_every: int = 1000,
    debug: bool = False,
) -> int:
    """Filter object rows to remove tiny regions and large table-like regions."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    image_area = image_width * image_height
    filtered_rows = []
    start_time = time.perf_counter()

    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        input_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    _emit_progress(
        progress,
        (
            f"[OCID] Filtering {len(rows):,} object rows "
            f"(min_area={min_area}, max_area_ratio={max_area_ratio}, "
            f"max_bbox_area_ratio={max_bbox_area_ratio})"
        ),
    )
    rejected_tiny = 0
    rejected_area_ratio = 0
    rejected_bbox_area_ratio = 0
    rejected_wide_region = 0

    for row_index, row in enumerate(rows, start=1):
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

        keep_row = True
        if area < min_area:
            rejected_tiny += 1
            keep_row = False
        elif area_ratio > max_area_ratio:
            rejected_area_ratio += 1
            keep_row = False
        elif bbox_area_ratio > max_bbox_area_ratio:
            rejected_bbox_area_ratio += 1
            keep_row = False
        elif bbox_width_ratio > 0.75 and bbox_height_ratio < 0.30:
            rejected_wide_region += 1
            keep_row = False

        if keep_row:
            row["area_ratio"] = f"{area_ratio:.6f}"
            row["bbox_area_ratio"] = f"{bbox_area_ratio:.6f}"
            row["bbox_width_ratio"] = f"{bbox_width_ratio:.6f}"
            row["bbox_height_ratio"] = f"{bbox_height_ratio:.6f}"

            filtered_rows.append(row)

        if _should_report_progress(row_index, len(rows), progress_every):
            _emit_progress(
                progress,
                (
                    f"[OCID] Filtered {row_index:,}/{len(rows):,} source rows; "
                    f"kept={len(filtered_rows):,}; elapsed={_elapsed_s(start_time)}"
                ),
            )

    fieldnames = list(input_fieldnames)
    for field in OCID_RATIO_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    _emit_progress(
        progress,
        (
            f"[OCID] Wrote filtered object index: {output_csv} "
            f"(kept={len(filtered_rows):,}/{len(rows):,}, "
            f"tiny={rejected_tiny:,}, area_ratio={rejected_area_ratio:,}, "
            f"bbox_area_ratio={rejected_bbox_area_ratio:,}, "
            f"wide_region={rejected_wide_region:,}, elapsed={_elapsed_s(start_time)})"
        ),
    )
    if debug and filtered_rows:
        _emit_progress(progress, f"[OCID][debug] First kept row: {filtered_rows[0]}")
    return len(filtered_rows)


def make_binary_mask_filename(row_index: int, file_name: str, object_id: int) -> str:
    """Create the deterministic binary-mask filename used by the OCID pipeline."""
    stem = Path(file_name).stem
    return f"{row_index:05d}_{stem}_obj{object_id}.png"


def export_binary_gt_masks(
    input_csv: Path,
    output_csv: Path,
    output_mask_dir: Path,
    progress: bool = False,
    progress_every: int = 1000,
    debug: bool = False,
    strict: bool = False,
) -> int:
    """Export one binary ground-truth mask for each object row."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    updated_rows = []
    start_time = time.perf_counter()

    with input_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    _emit_progress(
        progress,
        f"[OCID] Exporting binary masks for {len(rows):,} object rows to {output_mask_dir}",
    )
    cached_label_path = None
    cached_label = None
    labels_read = 0
    unreadable_labels = 0
    write_failures = 0
    for row_index, row in enumerate(rows):
        label_path = row["label_path"]
        file_name = row["file_name"]
        object_id = int(row["object_id"])

        if label_path != cached_label_path:
            cached_label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            cached_label_path = label_path
            labels_read += 1

        if cached_label is None:
            unreadable_labels += 1
            message = f"could not read label mask: {label_path}"
            if strict:
                raise FileNotFoundError(message)
            _emit_progress(progress, f"[OCID][warn] {message}")
            continue
        label = normalize_label_array(cached_label)

        binary_mask_uint8 = (label == object_id).astype(np.uint8) * 255
        mask_filename = make_binary_mask_filename(row_index, file_name, object_id)
        mask_path = output_mask_dir / mask_filename

        success = cv2.imwrite(str(mask_path), binary_mask_uint8)

        if not success:
            write_failures += 1
            message = f"could not write mask: {mask_path}"
            if strict:
                raise OSError(message)
            _emit_progress(progress, f"[OCID][warn] {message}")
            continue

        row["binary_mask_path"] = str(mask_path)
        updated_rows.append(row)

        count = row_index + 1
        if _should_report_progress(count, len(rows), progress_every):
            _emit_progress(
                progress,
                (
                    f"[OCID] Exported {len(updated_rows):,}/{len(rows):,} masks; "
                    f"labels_read={labels_read:,}; elapsed={_elapsed_s(start_time)}"
                ),
            )

    if not updated_rows:
        raise RuntimeError(f"No binary masks were exported from {input_csv}")

    fieldnames = list(updated_rows[0].keys())

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    _emit_progress(
        progress,
        (
            f"[OCID] Wrote final object index: {output_csv} "
            f"(masks={len(updated_rows):,}, labels_read={labels_read:,}, "
            f"unreadable_labels={unreadable_labels:,}, write_failures={write_failures:,}, "
            f"elapsed={_elapsed_s(start_time)})"
        ),
    )
    if debug:
        _emit_progress(progress, f"[OCID][debug] First mask row: {updated_rows[0]}")
    return len(updated_rows)
