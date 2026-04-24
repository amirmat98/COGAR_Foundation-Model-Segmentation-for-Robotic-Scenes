from pathlib import Path
import csv


INPUT_CSV = Path("outputs/indexes/ocid_debug_seq21_objects.csv")
OUTPUT_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered.csv")


# OCID image size
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT


def main():
    filtered_rows = []

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        area = int(row["area"])

        xmin = int(row["bbox_xmin"])
        ymin = int(row["bbox_ymin"])
        xmax = int(row["bbox_xmax"])
        ymax = int(row["bbox_ymax"])

        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        bbox_area = bbox_width * bbox_height

        area_ratio = area / IMAGE_AREA
        bbox_area_ratio = bbox_area / IMAGE_AREA
        bbox_width_ratio = bbox_width / IMAGE_WIDTH
        bbox_height_ratio = bbox_height / IMAGE_HEIGHT

        # Remove very tiny masks.
        if area < 500:
            continue

        # Remove very large masks.
        # For this first debug benchmark, we want object-like regions,
        # not table/wall/background-like regions.
        if area_ratio > 0.08:
            continue

        # Remove boxes that cover a large part of the image.
        if bbox_area_ratio > 0.15:
            continue

        # Remove long horizontal/vertical background-like strips.
        if bbox_width_ratio > 0.75 and bbox_height_ratio < 0.30:
            continue

        row["area_ratio"] = f"{area_ratio:.6f}"
        row["bbox_area_ratio"] = f"{bbox_area_ratio:.6f}"
        row["bbox_width_ratio"] = f"{bbox_width_ratio:.6f}"
        row["bbox_height_ratio"] = f"{bbox_height_ratio:.6f}"

        filtered_rows.append(row)

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
        "area_ratio",
        "bbox_area_ratio",
        "bbox_width_ratio",
        "bbox_height_ratio",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

    print(f"Input object instances: {len(rows)}")
    print(f"Filtered object instances: {len(filtered_rows)}")
    print(f"Saved filtered index to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()