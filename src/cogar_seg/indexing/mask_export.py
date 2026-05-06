"""Binary-mask export workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cogar_seg.config import load_config
from cogar_seg.datasets.ocid import export_binary_gt_masks
from cogar_seg.indexing.object_index import build_ocid_debug_index_paths


@dataclass(frozen=True)
class BinaryMaskExportRun:
    """Paths and count from exporting binary masks for an object index."""

    input_csv: Path
    output_csv: Path
    output_mask_dir: Path
    num_masks: int


def export_binary_masks(
    input_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_mask_dir: str | Path | None = None,
    config_path: str | Path = "configs/paths.yaml",
) -> BinaryMaskExportRun:
    """Export binary masks using explicit paths or the OCID debug defaults."""
    if input_csv is None or output_csv is None or output_mask_dir is None:
        config = load_config(config_path)
        paths = build_ocid_debug_index_paths(config)
        resolved_input_csv = Path(input_csv or paths.filtered_object_index_csv)
        resolved_output_csv = Path(output_csv or paths.final_object_index_csv)
        resolved_mask_dir = Path(output_mask_dir or paths.mask_dir)
    else:
        resolved_input_csv = Path(input_csv)
        resolved_output_csv = Path(output_csv)
        resolved_mask_dir = Path(output_mask_dir)

    num_masks = export_binary_gt_masks(
        input_csv=resolved_input_csv,
        output_csv=resolved_output_csv,
        output_mask_dir=resolved_mask_dir,
    )

    return BinaryMaskExportRun(
        input_csv=resolved_input_csv,
        output_csv=resolved_output_csv,
        output_mask_dir=resolved_mask_dir,
        num_masks=num_masks,
    )
