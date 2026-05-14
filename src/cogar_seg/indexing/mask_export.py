"""Binary-mask export workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cogar_seg.cv_compat import cv2
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


def _safe_name(value: object) -> str:
    """Convert category names into filesystem-safe tokens."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def export_cogar_sim_binary_masks(
    coco_path: str | Path,
    object_index_csv: str | Path,
    output_csv: str | Path,
    output_mask_dir: str | Path,
) -> BinaryMaskExportRun:
    """Export one binary PNG mask per COCO annotation listed in the object index.

    The output CSV is the input object index plus a new gt_mask_path column.
    """
    from pycocotools.coco import COCO

    coco_path = Path(coco_path)
    object_index_csv = Path(object_index_csv)
    output_csv = Path(output_csv)
    output_mask_dir = Path(output_mask_dir)

    if not coco_path.exists():
        raise FileNotFoundError(f"Missing COCO annotations file: {coco_path}")

    if not object_index_csv.exists():
        raise FileNotFoundError(f"Missing object index CSV: {object_index_csv}")

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(coco_path))
    df = pd.read_csv(object_index_csv)

    required_columns = {
        "image_id",
        "file_name",
        "image_path",
        "annotation_id",
        "category_id",
        "category_name",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Object index is missing required columns: {missing}")

    ann_by_id = {int(ann_id): ann for ann_id, ann in coco.anns.items()}

    missing_ann_ids = sorted(
        set(df["annotation_id"].astype(int).tolist()) - set(ann_by_id.keys())
    )
    if missing_ann_ids:
        raise ValueError(
            "Some annotation_id values from the object index are missing in COCO. "
            f"First missing IDs: {missing_ann_ids[:20]}"
        )

    gt_mask_paths: list[str] = []

    for row_idx, row in df.iterrows():
        ann_id = int(row["annotation_id"])
        ann = ann_by_id[ann_id]

        image_id = int(ann["image_id"])
        if image_id not in coco.imgs:
            raise ValueError(f"COCO annotation {ann_id} points to missing image_id={image_id}")

        img_info = coco.imgs[image_id]
        expected_height = int(img_info["height"])
        expected_width = int(img_info["width"])

        mask = coco.annToMask(ann)
        if mask.shape != (expected_height, expected_width):
            raise ValueError(
                f"Mask shape mismatch for annotation {ann_id}: "
                f"got {mask.shape}, expected {(expected_height, expected_width)}"
            )

        mask_u8 = (mask > 0).astype(np.uint8) * 255

        file_stem = Path(str(img_info["file_name"])).stem
        category_name = _safe_name(row["category_name"])
        mask_name = f"ann_{ann_id:08d}_img_{file_stem}_cat_{category_name}.png"
        mask_path = output_mask_dir / mask_name

        ok = cv2.imwrite(str(mask_path), mask_u8)
        if not ok:
            raise RuntimeError(f"Failed to write mask: {mask_path}")

        if not mask_path.exists():
            raise RuntimeError(f"Mask path was not created: {mask_path}")

        gt_mask_paths.append(str(mask_path))

        if (row_idx + 1) % 1000 == 0:
            print(f"[INFO] Exported {row_idx + 1}/{len(df)} masks")

    out_df = df.copy()
    out_df["gt_mask_path"] = gt_mask_paths
    out_df.to_csv(output_csv, index=False)

    if len(out_df) != len(df):
        raise RuntimeError("Output CSV row count does not match input object index row count")

    print(f"[OK] Exported COGAR-Sim binary masks: {len(out_df)}")
    print(f"[OK] Output CSV: {output_csv}")
    print(f"[OK] Mask directory: {output_mask_dir}")

    return BinaryMaskExportRun(
        input_csv=object_index_csv,
        output_csv=output_csv,
        output_mask_dir=output_mask_dir,
        num_masks=len(out_df),
    )
