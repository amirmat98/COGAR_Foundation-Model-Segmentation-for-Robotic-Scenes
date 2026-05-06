import argparse
from pathlib import Path

from cogar_seg.io import read_csv_rows
from cogar_seg.visualization import (
    visualize_binary_mask_from_row,
)


INDEX_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an OCID RGB image and one exported binary GT mask."
    )
    parser.add_argument(
        "row",
        nargs="?",
        type=int,
        default=5,
        help="Object row number to visualize.",
    )
    parser.add_argument(
        "--index",
        default=INDEX_CSV,
        help="Path to object-level CSV with binary_mask_path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv_rows(args.index)

    if not rows:
        raise RuntimeError("Index CSV is empty.")

    row_index = args.row

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Row index {row_index} is outside valid range 0-{len(rows)-1}")

    row = rows[row_index]
    visualize_binary_mask_from_row(row, row_index)


if __name__ == "__main__":
    main()
