import csv
import json
import shutil
from pathlib import Path

import yaml


RAW_COCO_DIR = Path("data/cogar_sim_500/raw_blenderproc/pilot_v2_ocid_like/coco_data")
RAW_IMAGES_DIR = RAW_COCO_DIR / "images"
RAW_COCO_PATH = RAW_COCO_DIR / "coco_annotations.json"

ROOT = Path("data/cogar_sim_500")
RGB_DIR = ROOT / "rgb"
ANN_DIR = ROOT / "annotations"
META_DIR = ROOT / "metadata"
SPLIT_DIR = ROOT / "splits"

CONFIG_PATH = Path("configs/blenderproc_dataset.yaml")
RAW_METADATA_PATH = META_DIR / "frame_index_pilot_v2.csv"


def load_categories():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return [
        {
            "id": int(cat["id"]),
            "name": cat["name"],
            "supercategory": cat.get("supercategory", "object"),
        }
        for cat in config["categories"]
    ]


def main():
    if not RAW_COCO_PATH.exists():
        raise FileNotFoundError(f"Missing COCO file: {RAW_COCO_PATH}")

    if not RAW_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {RAW_IMAGES_DIR}")

    if not RAW_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {RAW_METADATA_PATH}")

    coco = json.loads(RAW_COCO_PATH.read_text())
    categories = load_categories()

    raw_images = sorted(RAW_IMAGES_DIR.glob("*.png"))
    metadata_rows = list(csv.DictReader(RAW_METADATA_PATH.open()))

    print("Raw images:", len(raw_images))
    print("COCO images:", len(coco["images"]))
    print("COCO annotations:", len(coco["annotations"]))
    print("Metadata rows:", len(metadata_rows))
    print("Original categories:", len(coco["categories"]))

    assert len(raw_images) == 500, "Expected 500 raw images"
    assert len(coco["images"]) == 500, "Expected 500 COCO images"
    assert len(metadata_rows) == 500, "Expected 500 metadata rows"
    assert len(coco["annotations"]) > 500, "Too few annotations"

    RGB_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean normalized RGB folder before copying.
    for old_png in RGB_DIR.glob("*.png"):
        old_png.unlink()

    for img_path in raw_images:
        shutil.copy2(img_path, RGB_DIR / img_path.name)

    # Fix COCO category names from numeric labels to semantic labels.
    coco["categories"] = categories

    (ANN_DIR / "instances_all.json").write_text(
        json.dumps(coco, indent=2),
        encoding="utf-8",
    )

    (META_DIR / "categories.json").write_text(
        json.dumps(categories, indent=2),
        encoding="utf-8",
    )

    shutil.copy2(RAW_METADATA_PATH, META_DIR / "frame_index.csv")

    ids = [f"{i:06d}" for i in range(500)]

    train = ids[:350]
    val = ids[350:425]
    test = ids[425:500]

    (SPLIT_DIR / "train.txt").write_text("\n".join(train) + "\n")
    (SPLIT_DIR / "val.txt").write_text("\n".join(val) + "\n")
    (SPLIT_DIR / "test.txt").write_text("\n".join(test) + "\n")
    (SPLIT_DIR / "all.txt").write_text("\n".join(ids) + "\n")

    print("\n[OK] Normalized COGAR-SimRobotics-500 dataset created.")
    print("RGB:", RGB_DIR)
    print("Annotations:", ANN_DIR / "instances_all.json")
    print("Metadata:", META_DIR / "frame_index.csv")
    print("Categories:", META_DIR / "categories.json")
    print("Splits:", SPLIT_DIR)


if __name__ == "__main__":
    main()
