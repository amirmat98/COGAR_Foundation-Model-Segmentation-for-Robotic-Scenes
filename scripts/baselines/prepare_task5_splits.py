"""Prepare small supervised train/val subsets for Task 5 baselines."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

TRAINING_SPLITS = ("train", "val")
ALL_SPLITS = ("train", "val", "test")
FORMATS = ("coco", "yolo", "deeplab")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task5_baselines.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--formats", nargs="*", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--train-images", type=int, default=None)
    parser.add_argument("--val-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def resolve_image_path(dataset_config: dict[str, Any], file_name: str) -> Path:
    root = Path(dataset_config["root"])
    image_path_config = dataset_config.get("image_path", {})
    mode = image_path_config.get("mode", "coco_file_name_relative_to_root")

    if mode == "coco_file_name_relative_to_root":
        return root / file_name

    if mode == "basename_in_image_dir":
        return Path(image_path_config["image_dir"]) / Path(file_name).name

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


def iter_enabled_datasets(
    config: dict[str, Any],
    selected_names: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    datasets = []
    for name, dataset_config in config["datasets"].items():
        if selected_names is not None and name not in selected_names:
            continue
        if dataset_config.get("enabled", False):
            datasets.append((name, dataset_config))
    return datasets


def category_maps(coco: dict[str, Any]) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    yolo_class_by_category = {int(category["id"]): idx for idx, category in enumerate(categories)}
    semantic_id_by_category = {
        int(category["id"]): idx + 1 for idx, category in enumerate(categories)
    }
    metadata = {
        "background": {"semantic_id": 0, "name": "background"},
        "categories": [
            {
                "category_id": int(category["id"]),
                "name": category["name"],
                "yolo_class_id": yolo_class_by_category[int(category["id"])],
                "semantic_id": semantic_id_by_category[int(category["id"])],
            }
            for category in categories
        ],
    }
    return yolo_class_by_category, semantic_id_by_category, metadata


def valid_annotation(annotation: dict[str, Any], min_area: float) -> bool:
    if annotation.get("iscrowd", 0):
        return False
    if float(annotation.get("area", 0.0)) < min_area:
        return False
    return True


def split_image_ids(
    coco: dict[str, Any],
    train_count: int,
    val_count: int,
    seed: int,
    min_area: float,
    require_annotations: bool,
    dataset_name: str,
) -> dict[str, list[int]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if valid_annotation(annotation, min_area):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    candidate_ids = [int(image["id"]) for image in coco["images"]]
    if require_annotations:
        candidate_ids = [image_id for image_id in candidate_ids if annotations_by_image[image_id]]

    if len(candidate_ids) < train_count + val_count:
        raise ValueError(
            f"{dataset_name} has only {len(candidate_ids)} candidate images, "
            f"but {train_count + val_count} are requested."
        )

    stable_offset = sum(ord(char) for char in dataset_name)
    rng = random.Random(seed + stable_offset)
    shuffled = candidate_ids[:]
    rng.shuffle(shuffled)

    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count : train_count + val_count]
    test_ids = shuffled[train_count + val_count :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def subset_coco(
    coco: dict[str, Any],
    image_ids: list[int],
    min_area: float,
) -> dict[str, Any]:
    selected = set(image_ids)
    images = [image for image in coco["images"] if int(image["id"]) in selected]
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if int(annotation["image_id"]) in selected and valid_annotation(annotation, min_area)
    ]
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_images:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def yolo_polygon_from_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    min_contour_points: int,
) -> list[float] | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "YOLO label generation requires opencv-python. "
            "Install requirements.txt in the active environment."
        ) from exc

    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    points = contour.reshape(-1, 2)
    if len(points) < min_contour_points:
        return None

    normalized: list[float] = []
    for x, y in points:
        normalized.append(min(max(float(x) / float(width), 0.0), 1.0))
        normalized.append(min(max(float(y) / float(height), 0.0), 1.0))
    return normalized


def write_yolo_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any],
    coco: dict[str, Any],
    splits: dict[str, list[int]],
    output_root: Path,
    data_root: Path,
    yolo_class_by_category: dict[int, int],
    category_metadata: dict[str, Any],
    min_area: float,
    min_contour_points: int,
    copy_images: bool,
    dry_run: bool,
) -> dict[str, Any]:
    yolo_root = data_root / "yolo8_seg" / dataset_name
    config_path = output_root / "yolo8_seg" / f"{dataset_name}.yaml"
    images_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if valid_annotation(annotation, min_area):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    counts = {"images": 0, "labels": 0, "skipped_annotations": 0}
    if not dry_run:
        for split in TRAINING_SPLITS:
            for image_id in splits[split]:
                image = images_by_id[image_id]
                source = resolve_image_path(dataset_config, image["file_name"])
                suffix = Path(image["file_name"]).suffix or ".png"
                stem = f"{image_id:06d}"
                image_target = yolo_root / "images" / split / f"{stem}{suffix}"
                label_target = yolo_root / "labels" / split / f"{stem}.txt"
                link_or_copy_image(source, image_target, copy_images)

                height = int(image["height"])
                width = int(image["width"])
                label_lines = []
                for annotation in annotations_by_image[image_id]:
                    mask = decode_mask(annotation["segmentation"], height, width)
                    polygon = yolo_polygon_from_mask(
                        mask,
                        width=width,
                        height=height,
                        min_contour_points=min_contour_points,
                    )
                    if polygon is None:
                        counts["skipped_annotations"] += 1
                        continue
                    class_id = yolo_class_by_category[int(annotation["category_id"])]
                    coords = " ".join(f"{value:.6f}" for value in polygon)
                    label_lines.append(f"{class_id} {coords}")

                label_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                counts["images"] += 1
                counts["labels"] += len(label_lines)

        names = {
            item["yolo_class_id"]: item["name"] for item in category_metadata["categories"]
        }
        yolo_config = {
            "path": str(yolo_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": names,
        }
        write_json(output_root / "yolo8_seg" / f"{dataset_name}_class_map.json", category_metadata)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(yolo_config, sort_keys=False), encoding="utf-8")

    return {"format": "yolo8_seg", "config": str(config_path), **counts}


def write_deeplab_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any],
    coco: dict[str, Any],
    splits: dict[str, list[int]],
    output_root: Path,
    data_root: Path,
    semantic_id_by_category: dict[int, int],
    category_metadata: dict[str, Any],
    min_area: float,
    copy_images: bool,
    dry_run: bool,
) -> dict[str, Any]:
    semantic_root = data_root / "deeplabv3plus" / dataset_name
    config_path = output_root / "deeplabv3plus" / f"{dataset_name}.yaml"
    images_by_id = {int(image["id"]): image for image in coco["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if valid_annotation(annotation, min_area):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    counts = {"images": 0, "masks": 0}
    if not dry_run:
        for split in TRAINING_SPLITS:
            for image_id in splits[split]:
                image = images_by_id[image_id]
                source = resolve_image_path(dataset_config, image["file_name"])
                suffix = Path(image["file_name"]).suffix or ".png"
                stem = f"{image_id:06d}"
                image_target = semantic_root / "images" / split / f"{stem}{suffix}"
                mask_target = semantic_root / "masks" / split / f"{stem}.png"
                link_or_copy_image(source, image_target, copy_images)

                height = int(image["height"])
                width = int(image["width"])
                semantic = np.zeros((height, width), dtype=np.uint16)
                annotations = sorted(
                    annotations_by_image[image_id],
                    key=lambda item: float(item.get("area", 0.0)),
                    reverse=True,
                )
                for annotation in annotations:
                    mask = decode_mask(annotation["segmentation"], height, width)
                    semantic_id = semantic_id_by_category[int(annotation["category_id"])]
                    semantic[mask] = semantic_id

                mask_target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(semantic, mode="I;16").save(mask_target)
                counts["images"] += 1
                counts["masks"] += 1

        deeplab_config = {
            "root": str(semantic_root.resolve()),
            "train_images": "images/train",
            "train_masks": "masks/train",
            "val_images": "images/val",
            "val_masks": "masks/val",
            "num_classes": len(category_metadata["categories"]) + 1,
            "background_label": 0,
            "class_map": str((output_root / "deeplabv3plus" / f"{dataset_name}_class_map.json").resolve()),
        }
        write_json(output_root / "deeplabv3plus" / f"{dataset_name}_class_map.json", category_metadata)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(deeplab_config, sort_keys=False), encoding="utf-8")

    return {"format": "deeplabv3plus", "config": str(config_path), **counts}


def write_coco_subsets(
    dataset_name: str,
    coco: dict[str, Any],
    splits: dict[str, list[int]],
    output_root: Path,
    min_area: float,
    dry_run: bool,
) -> dict[str, Any]:
    subset_dir = output_root / "coco" / dataset_name
    counts: dict[str, Any] = {"format": "coco_instance", "files": {}}
    for split in ALL_SPLITS:
        subset = subset_coco(coco, splits[split], min_area)
        filename = f"instances_{split}_small.json" if split in TRAINING_SPLITS else "instances_test.json"
        path = subset_dir / filename
        counts["files"][split] = {
            "path": str(path),
            "images": len(subset["images"]),
            "annotations": len(subset["annotations"]),
        }
        if not dry_run:
            write_json(path, subset)
    return counts


def write_split_files(
    dataset_name: str,
    splits: dict[str, list[int]],
    output_root: Path,
    dry_run: bool,
) -> dict[str, str]:
    split_dir = output_root / "splits" / dataset_name
    paths = {}
    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)
    for split, image_ids in splits.items():
        path = split_dir / f"{split}_image_ids.txt"
        paths[split] = str(path)
        if not dry_run:
            path.write_text(
                "".join(f"{image_id}\n" for image_id in image_ids),
                encoding="utf-8",
            )
    return paths


def prepare_dataset(
    dataset_name: str,
    dataset_config: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_root = Path(config["task"]["output_root"])
    data_root = Path(config["task"]["data_root"])
    split_policy = config["split_policy"]
    train_count = args.train_images or int(split_policy["train_images"])
    val_count = args.val_images or int(split_policy["val_images"])
    seed = args.seed or int(config["task"]["seed"])
    min_area = float(split_policy.get("min_area", 1.0))
    require_annotations = bool(split_policy.get("require_annotations", True))

    coco = load_json(dataset_config["annotation_file"])
    splits = split_image_ids(
        coco=coco,
        train_count=train_count,
        val_count=val_count,
        seed=seed,
        min_area=min_area,
        require_annotations=require_annotations,
        dataset_name=dataset_name,
    )
    yolo_class_by_category, semantic_id_by_category, category_metadata = category_maps(coco)

    print(
        f"[START] {dataset_name}: train={len(splits['train'])} "
        f"val={len(splits['val'])} test_remaining={len(splits['test'])}",
        flush=True,
    )

    outputs: list[dict[str, Any]] = []
    split_paths = write_split_files(dataset_name, splits, output_root, args.dry_run)
    if "coco" in args.formats:
        outputs.append(write_coco_subsets(dataset_name, coco, splits, output_root, min_area, args.dry_run))
    if "yolo" in args.formats:
        min_contour_points = int(config["baselines"]["yolo8_seg"].get("min_contour_points", 3))
        outputs.append(
            write_yolo_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                coco=coco,
                splits=splits,
                output_root=output_root,
                data_root=data_root,
                yolo_class_by_category=yolo_class_by_category,
                category_metadata=category_metadata,
                min_area=min_area,
                min_contour_points=min_contour_points,
                copy_images=args.copy_images,
                dry_run=args.dry_run,
            )
        )
    if "deeplab" in args.formats:
        outputs.append(
            write_deeplab_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                coco=coco,
                splits=splits,
                output_root=output_root,
                data_root=data_root,
                semantic_id_by_category=semantic_id_by_category,
                category_metadata=category_metadata,
                min_area=min_area,
                copy_images=args.copy_images,
                dry_run=args.dry_run,
            )
        )

    summary_path = output_root / "summaries" / f"{dataset_name}_summary.json"
    if set(args.formats) != set(FORMATS) and summary_path.exists():
        existing_summary = load_json(summary_path)
        updated_by_format = {item["format"]: item for item in outputs}
        for item in existing_summary.get("formats", []):
            updated_by_format.setdefault(item["format"], item)
        outputs = [updated_by_format[name] for name in ("coco_instance", "yolo8_seg", "deeplabv3plus") if name in updated_by_format]

    summary = {
        "dataset": dataset_name,
        "seed": seed,
        "train_images": len(splits["train"]),
        "val_images": len(splits["val"]),
        "test_images": len(splits["test"]),
        "test_remaining_images": len(splits["test"]),
        "split_files": split_paths,
        "formats": outputs,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        write_json(summary_path, summary)
    print(f"[OK] {dataset_name}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summaries = []
    for dataset_name, dataset_config in iter_enabled_datasets(config, args.datasets):
        summaries.append(prepare_dataset(dataset_name, dataset_config, config, args))

    if not args.dry_run:
        summary_path = Path(config["task"]["output_root"]) / "summaries" / "summary.json"
        if (args.datasets is not None or set(args.formats) != set(FORMATS)) and summary_path.exists():
            merged = {item["dataset"]: item for item in load_json(summary_path)}
            merged.update({item["dataset"]: item for item in summaries})
            summaries = [
                merged[name]
                for name in config["datasets"]
                if name in merged
            ]
        write_json(summary_path, summaries)
    print("[DONE] Task 5A split preparation", flush=True)


if __name__ == "__main__":
    main()
