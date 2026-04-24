from pathlib import Path
import csv
import cv2
import numpy as np


OBJECT_INDEX_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered.csv")
OUTPUT_MASK_DIR = Path("outputs/gt_binary_masks")


def make_mask_filename(row_index: int, file_name: str, object_id: int) -> str:
    stem = Path(file_name).stem
    return f"{row_index:05d}_{stem}_obj{object_id}.png"


def main():
    OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    updated_rows = []

    with open(OBJECT_INDEX_CSV, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row_index, row in enumerate(rows):
        label_path = row["label_path"]
        file_name = row["file_name"]
        object_id = int(row["object_id"])

        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        if label is None:
            print(f"Warning: could not read label mask: {label_path}")
            continue

        # Binary mask:
        # True where label == object_id, False elsewhere.
        binary_mask = label == object_id

        # Save as normal 8-bit image:
        # object pixels = 255, background = 0.
        binary_mask_uint8 = (binary_mask.astype(np.uint8)) * 255

        mask_filename = make_mask_filename(row_index, file_name, object_id)
        mask_path = OUTPUT_MASK_DIR / mask_filename

        success = cv2.imwrite(str(mask_path), binary_mask_uint8)

        if not success:
            print(f"Warning: could not write mask: {mask_path}")
            continue

        row["binary_mask_path"] = str(mask_path)
        updated_rows.append(row)

    output_csv = Path("outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv")

    fieldnames = list(updated_rows[0].keys())

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Input filtered objects: {len(rows)}")
    print(f"Saved binary masks: {len(updated_rows)}")
    print(f"Binary mask folder: {OUTPUT_MASK_DIR}")
    print(f"Updated index saved to: {output_csv}")


if __name__ == "__main__":
    main()