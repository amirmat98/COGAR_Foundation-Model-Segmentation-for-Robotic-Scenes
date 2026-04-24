from pathlib import Path
import sys
import csv
import cv2
import matplotlib.pyplot as plt


OBJECT_INDEX_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered.csv")


def main():
    with open(OBJECT_INDEX_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError("Object index is empty.")

    # Pick one object instance.

    row_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    row = rows[row_index]

    image_path = row["image_path"]
    object_id = int(row["object_id"])

    xmin = int(row["bbox_xmin"])
    ymin = int(row["bbox_ymin"])
    xmax = int(row["bbox_xmax"])
    ymax = int(row["bbox_ymax"])

    point_x = int(row["point_x"])
    point_y = int(row["point_y"])

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # Draw bounding box.
    cv2.rectangle(
        image_bgr,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2,
    )

    # Draw point prompt.
    cv2.circle(
        image_bgr,
        (point_x, point_y),
        6,
        (0, 0, 255),
        -1,
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Image:", image_path)
    print("Row index:", row_index)
    print("Object ID:", object_id)
    print("Bounding box:", xmin, ymin, xmax, ymax)
    print("Point prompt:", point_x, point_y)

    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(f"Object {object_id}: box prompt + point prompt")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()