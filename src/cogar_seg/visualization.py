from pathlib import Path
import csv

import cv2
import matplotlib.pyplot as plt


def read_csv_rows(csv_path: str | Path) -> list[dict]:
    """
    Read a CSV file and return its rows as dictionaries.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r") as f:
        return list(csv.DictReader(f))


def draw_box_and_point(
    image_bgr,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    point_x: int,
    point_y: int,
):
    """
    Draw a bounding box and point prompt on a BGR image.

    Green rectangle = box prompt
    Red dot = point prompt
    """
    output = image_bgr.copy()

    cv2.rectangle(
        output,
        (xmin, ymin),
        (xmax, ymax),
        (0, 255, 0),
        2,
    )

    cv2.circle(
        output,
        (point_x, point_y),
        6,
        (0, 0, 255),
        -1,
    )

    return output


def visualize_object_prompt_from_row(row: dict, row_index: int):
    """
    Visualize one object prompt from one CSV row.
    """
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

    image_with_prompt_bgr = draw_box_and_point(
        image_bgr=image_bgr,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        point_x=point_x,
        point_y=point_y,
    )

    image_rgb = cv2.cvtColor(image_with_prompt_bgr, cv2.COLOR_BGR2RGB)

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


def visualize_binary_mask_from_row(row: dict, row_index: int):
    """
    Visualize RGB image and binary ground-truth mask side by side.
    """
    image_path = row["image_path"]
    binary_mask_path = row["binary_mask_path"]
    object_id = row["object_id"]

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    if mask is None:
        raise RuntimeError(f"Could not read binary mask: {binary_mask_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Row index:", row_index)
    print("Object ID:", object_id)
    print("Image:", image_path)
    print("Binary mask:", binary_mask_path)
    print("Mask unique values:", sorted(set(int(v) for v in mask.flatten())))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("RGB image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Binary GT mask for object {object_id}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()