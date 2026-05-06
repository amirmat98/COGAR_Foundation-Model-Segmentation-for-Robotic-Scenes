import argparse

from cogar_seg.analysis.comparison import (
    compare_prompt_results,
    print_prompt_comparison_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM box-prompt and point-prompt result CSVs."
    )
    parser.add_argument(
        "--box-results-csv",
        default="outputs/indexes/ocid_debug_seq21_sam_box_results.csv",
    )
    parser.add_argument(
        "--point-results-csv",
        default="outputs/sam_point_prompt_batch/sam_point_prompt_results.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/prompt_comparison")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = compare_prompt_results(
        box_results_csv=args.box_results_csv,
        point_results_csv=args.point_results_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        save_plots=not args.no_plots,
    )
    print_prompt_comparison_summary(run)
    print()
    print("Saved comparison files in:", run.output_dir)


if __name__ == "__main__":
    main()
