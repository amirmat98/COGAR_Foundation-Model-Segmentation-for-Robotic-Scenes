import argparse
from pathlib import Path

import cv2
import pandas as pd


REQUIRED_COLUMNS = [
    "image_id",
    "file_name",
    "split",
    "image_path",
    "binary_mask_path",
    "category_id",
    "category_name",
    "object_id",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "point_x",
    "point_y",
    "challenge_primary",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.index)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    bad_splits = sorted(set(df["split"].astype(str)) - {"train", "val", "test"})
    if bad_splits:
        raise ValueError(f"Invalid split values: {bad_splits}")

    for i, row in df.iterrows():
        image_path = Path(row["image_path"])
        mask_path = Path(row["binary_mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image at row {i}: {image_path}")

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask at row {i}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Unreadable image at row {i}: {image_path}")

        if mask is None:
            raise ValueError(f"Unreadable mask at row {i}: {mask_path}")

        h, w = image.shape[:2]

        xmin = float(row["bbox_xmin"])
        ymin = float(row["bbox_ymin"])
        xmax = float(row["bbox_xmax"])
        ymax = float(row["bbox_ymax"])
        px = int(round(float(row["point_x"])))
        py = int(round(float(row["point_y"])))

        if not (0 <= xmin < xmax <= w):
            raise ValueError(f"Invalid bbox x at row {i}: xmin={xmin}, xmax={xmax}, width={w}")

        if not (0 <= ymin < ymax <= h):
            raise ValueError(f"Invalid bbox y at row {i}: ymin={ymin}, ymax={ymax}, height={h}")

        if not (0 <= px < w and 0 <= py < h):
            raise ValueError(f"Invalid point at row {i}: point=({px}, {py}), image=({w}, {h})")

        if mask[py, px] == 0:
            raise ValueError(f"Point is not inside mask at row {i}: point=({px}, {py})")

    print(f"[OK] Valid index: {args.index}")
    print(f"Rows: {len(df)}")
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nCategory counts:")
    print(df["category_name"].value_counts())
    print("\nChallenge counts:")
    print(df["challenge_primary"].value_counts())


if __name__ == "__main__":
    main()
