"""Build point and box prompt manifests from COCO instance annotations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cogar_seg.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_zero_shot_sam.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Print progress after this many COCO annotations per dataset.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_image_path(dataset_config: dict[str, Any], file_name: str) -> str:
    root = Path(dataset_config["root"])
    image_path_config = dataset_config.get("image_path", {})
    mode = image_path_config.get("mode", "coco_file_name_relative_to_root")

    if mode == "coco_file_name_relative_to_root":
        return str(root / file_name)

    if mode == "basename_in_image_dir":
        image_dir = Path(image_path_config["image_dir"])
        return str(image_dir / Path(file_name).name)

    raise ValueError(f"Unsupported image path mode: {mode}")


def decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rles)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(bool)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)!r}")


def point_nearest_centroid(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build point prompt from empty mask")

    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    distances = (xs - centroid_x) ** 2 + (ys - centroid_y) ** 2
    selected = int(np.argmin(distances))
    return [int(xs[selected]), int(ys[selected])]


def bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def iter_enabled_datasets(
    config: dict[str, Any], selected_names: list[str] | None
) -> list[tuple[str, dict[str, Any]]]:
    datasets = []
    for name, dataset_config in config["datasets"].items():
        if selected_names is not None and name not in selected_names:
            continue
        if dataset_config.get("enabled", False):
            datasets.append((name, dataset_config))
    return datasets


def build_dataset_manifest(
    dataset_name: str,
    dataset_config: dict[str, Any],
    output_dir: Path,
    max_images: int | None,
    min_area: float,
    log_every: int,
) -> dict[str, Any]:
    coco = load_json(dataset_config["annotation_file"])
    images = {image["id"]: image for image in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    selected_image_ids = set(images)
    if max_images is not None:
        selected_image_ids = set(list(images.keys())[:max_images])

    output_path = output_dir / f"{dataset_name}_instances.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_annotations = len(coco["annotations"])
    selected_images_count = len(selected_image_ids)
    started_at = time.perf_counter()
    print(
        f"[START] {dataset_name}: {total_annotations} annotations, "
        f"{selected_images_count} selected images -> {output_path}",
        flush=True,
    )

    count = 0
    skipped = 0

    def maybe_log_progress(annotation_index: int) -> None:
        if log_every > 0 and annotation_index % log_every == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"[PROGRESS] {dataset_name}: "
                f"{annotation_index}/{total_annotations} annotations scanned, "
                f"{count} prompts, {skipped} skipped, {elapsed:.1f}s",
                flush=True,
            )

    with output_path.open("w", encoding="utf-8") as f:
        for annotation_index, annotation in enumerate(coco["annotations"], start=1):
            image_id = annotation["image_id"]
            if image_id not in selected_image_ids:
                maybe_log_progress(annotation_index)
                continue
            if annotation.get("iscrowd", 0):
                skipped += 1
                maybe_log_progress(annotation_index)
                continue
            if float(annotation.get("area", 0.0)) < min_area:
                skipped += 1
                maybe_log_progress(annotation_index)
                continue

            image = images[image_id]
            height = int(image["height"])
            width = int(image["width"])
            mask = decode_mask(annotation["segmentation"], height, width)
            if not mask.any():
                skipped += 1
                maybe_log_progress(annotation_index)
                continue

            point_xy = point_nearest_centroid(mask)
            record = {
                "dataset": dataset_name,
                "image_id": image_id,
                "annotation_id": annotation["id"],
                "category_id": annotation["category_id"],
                "category_name": categories.get(annotation["category_id"], "unknown"),
                "file_name": image["file_name"],
                "image_path": resolve_image_path(dataset_config, image["file_name"]),
                "height": height,
                "width": width,
                "area": float(annotation.get("area", 0.0)),
                "point_prompt": {
                    "points": [point_xy],
                    "labels": [1],
                },
                "box_prompt": {
                    "box_xyxy": bbox_xywh_to_xyxy(annotation["bbox"]),
                    "source_bbox_xywh": [float(v) for v in annotation["bbox"]],
                },
            }
            f.write(json.dumps(record) + "\n")
            count += 1
            maybe_log_progress(annotation_index)

    summary = {
        "dataset": dataset_name,
        "manifest": str(output_path),
        "images": len(selected_image_ids),
        "instances": count,
        "skipped_annotations": skipped,
        "min_area": min_area,
    }
    write_json(output_dir / f"{dataset_name}_summary.json", summary)
    elapsed = time.perf_counter() - started_at
    print(
        f"[DONE] {dataset_name}: {count} prompts, {skipped} skipped, "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["task"]["prompt_manifest_dir"])
    summaries = []

    for dataset_name, dataset_config in iter_enabled_datasets(config, args.datasets):
        summaries.append(
            build_dataset_manifest(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                output_dir=output_dir,
                max_images=args.max_images,
                min_area=args.min_area,
                log_every=args.log_every,
            )
        )

    write_json(output_dir / "summary.json", summaries)
    for summary in summaries:
        print(
            f"[OK] {summary['dataset']}: "
            f"{summary['instances']} prompts from {summary['images']} images"
        )


if __name__ == "__main__":
    main()
