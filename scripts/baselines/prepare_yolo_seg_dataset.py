import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


DEFAULT_CLASSES = [
    "box",
    "cable",
    "connector",
    "glass_object",
    "metal_part",
    "plastic_object",
    "robot_gripper",
    "screw",
    "tool",
]


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def mask_to_yolo_polygon(mask: np.ndarray, min_points: int = 3):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None

    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    points = approx.reshape(-1, 2)
    if len(points) < min_points:
        return None

    h, w = mask.shape[:2]
    points = points.astype(float)
    points[:, 0] = np.clip(points[:, 0] / w, 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1] / h, 0.0, 1.0)

    return points.reshape(-1).tolist()


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-root", default="data/yolo_cogar_sim_500_final")
    parser.add_argument("--max-train-images", type=int, default=120)
    parser.add_argument("--max-val-images", type=int, default=80)
    parser.add_argument("--max-test-images", type=int, default=80)
    parser.add_argument("--symlink", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)

    required = [
        "file_name",
        "image_path",
        "binary_mask_path",
        "category_name",
        "split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    classes = DEFAULT_CLASSES
    class_to_id = {name: i for i, name in enumerate(classes)}

    unknown = sorted(set(df["category_name"]) - set(classes))
    if unknown:
        raise ValueError(f"Unknown categories: {unknown}")

    limits = {
        "train": args.max_train_images,
        "val": args.max_val_images,
        "test": args.max_test_images,
    }

    selected_frames = []

    for split, limit in limits.items():
        split_files = (
            df[df["split"] == split]["file_name"]
            .drop_duplicates()
            .sort_values()
            .head(limit)
            .tolist()
        )
        selected_frames.extend([(split, f) for f in split_files])

    selected = pd.DataFrame(selected_frames, columns=["split", "file_name"])
    selected.to_csv(output_root / "selected_images.csv", index=False)

    rows_written = 0
    labels_written = 0
    skipped_masks = 0

    for split, file_name in selected_frames:
        frame_df = df[(df["split"] == split) & (df["file_name"] == file_name)].copy()
        if frame_df.empty:
            continue

        image_path = Path(str(frame_df.iloc[0]["image_path"]))
        image_dst = output_root / "images" / split / image_path.name
        label_dst = output_root / "labels" / split / f"{image_path.stem}.txt"

        copy_or_link(image_path, image_dst, args.symlink)

        label_lines = []

        for _, row in frame_df.iterrows():
            category = str(row["category_name"])
            class_id = class_to_id[category]

            mask = read_mask(str(row["binary_mask_path"]))
            polygon = mask_to_yolo_polygon(mask)

            if polygon is None or len(polygon) < 6:
                skipped_masks += 1
                continue

            values = [str(class_id)] + [f"{v:.6f}" for v in polygon]
            label_lines.append(" ".join(values))
            labels_written += 1

        label_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

        rows_written += 1

    yaml_path = output_root / "cogar_sim_500_yolov8seg.yaml"
    yaml_data = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }

    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))

    print("[OK] Prepared YOLOv8-seg dataset")
    print("Output root:", output_root)
    print("YAML:", yaml_path)
    print("Images selected:", rows_written)
    print("Labels written:", labels_written)
    print("Masks skipped:", skipped_masks)
    print("\nSelected image counts:")
    print(selected["split"].value_counts())


if __name__ == "__main__":
    main()
