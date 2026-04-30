import argparse
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_CSV = Path("outputs/indexes/ocid_debug_seq21_sam_box_results.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a SAM box-prompt results CSV."
    )
    parser.add_argument(
        "--results-csv",
        default=DEFAULT_RESULTS_CSV,
        help="Path to SAM results CSV.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        help="Number of lowest-IoU rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv)

    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    required_columns = {"row_index", "file_name", "object_id", "sam_score", "iou"}
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(f"Missing required result columns: {missing}")

    print("Results CSV:", results_csv)
    print(f"Number of evaluated objects: {len(df)}")
    print(f"Mean IoU: {df['iou'].mean():.4f}")
    print(f"Median IoU: {df['iou'].median():.4f}")
    print(f"Min IoU: {df['iou'].min():.4f}")
    print(f"Max IoU: {df['iou'].max():.4f}")
    print(f"Mean SAM score: {df['sam_score'].mean():.4f}")

    if args.worst <= 0:
        return

    print()
    print(f"Worst {min(args.worst, len(df))} rows by IoU:")

    columns = ["row_index", "file_name", "object_id", "iou", "sam_score"]
    worst_rows = df.sort_values("iou", ascending=True).head(args.worst)
    print(worst_rows[columns].to_string(index=False))


if __name__ == "__main__":
    main()
