from pathlib import Path
import csv
import cv2
import numpy as np


INPUT_CSV = Path("outputs/indexes/ocid_debug_seq21.csv")
OUTPUT_CSV = Path("outputs/indexes/ocid_debug_seq21_objects.csv")


def compute_object_properties(label: np.ndarray, object_id: int):
    """
    Given a label image and one object ID, compute:
    - binary mask area
    - bounding box
    - simple center point prompt
    """

    binary_mask = label == object_id

    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    area = int(binary_mask.sum())

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    # Simple point prompt:
    # use the mean pixel location of the object's mask.
    point_x = int(xs.mean())
    point_y = int(ys.mean())

    return {
        "area": area,
        "bbox_xmin": xmin,
        "bbox_ymin": ymin,
        "bbox_xmax": xmax,
        "bbox_ymax": ymax,
        "point_x": point_x,
        "point_y": point_y,
    }


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    object_rows = []

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_path = row["image_path"]
            label_path = row["label_path"]

            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            if label is None:
                print(f"Warning: could not read label mask: {label_path}")
                continue

            unique_ids = np.unique(label)

            for object_id in unique_ids:
                object_id = int(object_id)

                # 0 is background, not an object.
                if object_id == 0:
                    continue

                props = compute_object_properties(label, object_id)

                if props is None:
                    continue

                object_rows.append({
                    "image_path": image_path,
                    "label_path": label_path,
                    "sequence": row["sequence"],
                    "file_name": row["file_name"],
                    "object_id": object_id,
                    **props,
                })

    fieldnames = [
        "image_path",
        "label_path",
        "sequence",
        "file_name",
        "object_id",
        "area",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "point_x",
        "point_y",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(object_rows)

    print(f"Saved object-level index to: {OUTPUT_CSV}")
    print(f"Number of object instances: {len(object_rows)}")


if __name__ == "__main__":
    main()