import argparse

from cogar_seg.analysis.results import (
    analyze_prompt_results,
    print_prompt_analysis_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze segmentation prompt results.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = analyze_prompt_results(
        results_csv=args.results_csv,
        output_dir=args.output_dir,
        prompt_name=args.prompt_name,
        top_k=args.top_k,
        save_plots=not args.no_plots,
    )
    print_prompt_analysis_summary(run)
    print()
    print("Saved analysis files in:", run.output_dir)


if __name__ == "__main__":
    main()
