import argparse
from pathlib import Path

from cogar_seg.io import read_csv_rows
from cogar_seg.visualization import (
    visualize_object_prompt_from_row,
)


OBJECT_INDEX_CSV = Path("outputs/indexes/ocid_debug_seq21_objects_filtered.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize an OCID object row with its box and point prompts."
    )
    parser.add_argument(
        "row",
        nargs="?",
        type=int,
        default=0,
        help="Object row number to visualize.",
    )
    parser.add_argument(
        "--index",
        default=OBJECT_INDEX_CSV,
        help="Path to object-level CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv_rows(args.index)

    if not rows:
        raise RuntimeError("Object index is empty.")

    row_index = args.row

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"Row index {row_index} is outside valid range 0-{len(rows)-1}")

    row = rows[row_index]
    visualize_object_prompt_from_row(row, row_index)


if __name__ == "__main__":
    main()
