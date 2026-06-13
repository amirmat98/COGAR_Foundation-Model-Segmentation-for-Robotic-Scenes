"""Convert OCID integer instance-label PNGs to COCO instance annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/mnt/Info/COGAR_DATASETs/OCID-dataset",
        help="OCID dataset root.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output COCO JSON. Defaults to <root>/annotations/instances_all.json.",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Frame metadata CSV. Defaults to <root>/metadata/frame_index.csv.",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Split output directory. Defaults to <root>/splits.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-area", type=int, default=1)
    return parser.parse_args()


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(v) for v in rle["size"]], "counts": counts}


def bbox_from_rle(rle: dict[str, Any]) -> list[float]:
    encoded = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return [float(v) for v in mask_utils.toBbox(encoded)]


def area_from_rle(rle: dict[str, Any]) -> float:
    encoded = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    return float(mask_utils.area(encoded))


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sequence_fields(label_path: Path, root: Path) -> dict[str, str]:
    rel = label_path.relative_to(root)
    parts = rel.parts
    return {
        "subset": parts[0] if len(parts) > 0 else "",
        "surface": parts[1] if len(parts) > 1 else "",
        "view": parts[2] if len(parts) > 2 else "",
        "scene_type": parts[3] if len(parts) > 3 else "",
        "sequence": parts[4] if len(parts) > 4 else "",
    }


def matching_rgb_path(label_path: Path) -> Path:
    return label_path.parent.parent / "rgb" / label_path.name


def iter_label_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/label/*.png"))


def split_ids(image_ids: list[int]) -> dict[str, list[int]]:
    total = len(image_ids)
    train_count = int(round(0.70 * total))
    val_count = int(round(0.15 * total))
    if total >= 3:
        train_count = max(1, min(train_count, total - 2))
        val_count = max(1, min(val_count, total - train_count - 1))
    else:
        train_count = total
        val_count = 0
    return {
        "train": image_ids[:train_count],
        "val": image_ids[train_count : train_count + val_count],
        "test": image_ids[train_count + val_count :],
        "all": image_ids,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_file = (
        Path(args.output_file)
        if args.output_file
        else root / "annotations" / "instances_all.json"
    )
    metadata_file = (
        Path(args.metadata_file)
        if args.metadata_file
        else root / "metadata" / "frame_index.csv"
    )
    split_dir = Path(args.splits_dir) if args.splits_dir else root / "splits"

    label_paths = iter_label_paths(root)
    if args.max_images is not None:
        label_paths = label_paths[: args.max_images]

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    skipped_missing_rgb = 0
    skipped_small = 0
    annotation_id = 1

    for image_id, label_path in enumerate(label_paths, start=1):
        rgb_path = matching_rgb_path(label_path)
        if not rgb_path.exists():
            skipped_missing_rgb += 1
            continue

        label = np.array(Image.open(label_path))
        if label.ndim != 2:
            raise ValueError(f"Expected 2D label image: {label_path}")
        height, width = label.shape
        fields = sequence_fields(label_path, root)

        images.append(
            {
                "id": image_id,
                "file_name": relative_posix(rgb_path, root),
                "width": int(width),
                "height": int(height),
            }
        )
        metadata_rows.append(
            {
                "image_id": image_id,
                "file_name": relative_posix(rgb_path, root),
                "label_file": relative_posix(label_path, root),
                **fields,
            }
        )

        for instance_id in sorted(int(v) for v in np.unique(label) if int(v) > 0):
            mask = label == instance_id
            area = int(mask.sum())
            if area < args.min_area:
                skipped_small += 1
                continue
            rle = encode_binary_mask(mask)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "iscrowd": 0,
                    "instance_id": instance_id,
                    "area": float(area),
                    "bbox": bbox_from_rle(rle),
                    "segmentation": rle,
                }
            )
            annotation_id += 1

    categories = [{"id": 1, "name": "object", "supercategory": "ocid"}]
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    with metadata_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id",
            "file_name",
            "label_file",
            "subset",
            "surface",
            "view",
            "scene_type",
            "sequence",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    split_dir.mkdir(parents=True, exist_ok=True)
    id_strings = {image["id"]: f"{image['id']:06d}" for image in images}
    for split_name, ids in split_ids([image["id"] for image in images]).items():
        lines = [id_strings[image_id] for image_id in ids]
        (split_dir / f"{split_name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    print("[OK] Converted OCID to COCO")
    print(f"root: {root}")
    print(f"annotations: {output_file}")
    print(f"metadata: {metadata_file}")
    print(f"images: {len(images)}")
    print(f"annotations_count: {len(annotations)}")
    print(f"categories: {len(categories)}")
    print(f"skipped_missing_rgb: {skipped_missing_rgb}")
    print(f"skipped_small_annotations: {skipped_small}")


if __name__ == "__main__":
    main()
