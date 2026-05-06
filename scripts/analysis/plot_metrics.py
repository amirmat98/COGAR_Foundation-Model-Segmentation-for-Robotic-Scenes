import argparse
from pathlib import Path

import pandas as pd

from cogar_seg.analysis.results import (
    plot_iou_histogram,
    plot_per_object_mean_iou,
    plot_score_vs_iou,
    save_per_object_summary,
    validate_results_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from a result CSV.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    validate_results_dataframe(df)
    per_object = save_per_object_summary(df, output_dir, args.prompt_name)
    plot_iou_histogram(df, output_dir, args.prompt_name)
    plot_score_vs_iou(df, output_dir, args.prompt_name)
    plot_per_object_mean_iou(per_object, output_dir, args.prompt_name)

    print("Saved plots in:", output_dir)


if __name__ == "__main__":
    main()
