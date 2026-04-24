import sys
from pathlib import Path

from cogar_seg.visualization import (
    read_csv_rows,
    visualize_binary_mask_from_row,
)


INDEX_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv")


def main():
    rows = read_csv_rows(INDEX_CSV)

    if not rows:
        raise RuntimeError("Index CSV is empty.")

    row_index = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Row index {row_index} is outside valid range 0-{len(rows)-1}")

    row = rows[row_index]
    visualize_binary_mask_from_row(row, row_index)


if __name__ == "__main__":
    main()