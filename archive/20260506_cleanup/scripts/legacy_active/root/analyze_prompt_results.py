import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze segmentation prompt results from a SAM result CSV."
    )

    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to a prompt-evaluation results CSV.",
    )
    parser.add_argument(
        "--prompt-name",
        required=True,
        help="Name used in output files, e.g. sam_point_prompt or sam_box_prompt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where analysis CSVs and plots will be saved.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of best/worst rows to save.",
    )

    return parser.parse_args()


def validate_results_dataframe(df: pd.DataFrame) -> None:
    required_columns = {
        "row_index",
        "object_id",
        "sam_score",
        "iou",
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(f"Missing required result columns: {missing}")


def save_global_summary(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "prompt_name": prompt_name,
                "num_objects": len(df),
                "mean_iou": df["iou"].mean(),
                "median_iou": df["iou"].median(),
                "min_iou": df["iou"].min(),
                "max_iou": df["iou"].max(),
                "std_iou": df["iou"].std(),
                "mean_sam_score": df["sam_score"].mean(),
                "median_sam_score": df["sam_score"].median(),
                "min_sam_score": df["sam_score"].min(),
                "max_sam_score": df["sam_score"].max(),
            }
        ]
    )

    summary_path = output_dir / f"{prompt_name}_global_summary.csv"
    summary.to_csv(summary_path, index=False)

    return summary


def save_per_object_summary(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> pd.DataFrame:
    per_object = (
        df.groupby("object_id")
        .agg(
            count=("iou", "count"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            min_iou=("iou", "min"),
            max_iou=("iou", "max"),
            std_iou=("iou", "std"),
            mean_sam_score=("sam_score", "mean"),
        )
        .reset_index()
        .sort_values("mean_iou", ascending=True)
    )

    per_object_path = output_dir / f"{prompt_name}_per_object_summary.csv"
    per_object.to_csv(per_object_path, index=False)

    return per_object


def save_best_and_worst_cases(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    preferred_columns = [
        "row_index",
        "object_id",
        "iou",
        "sam_score",
        "image_path",
        "gt_mask_path",
        "mask_output_path",
        "visualization_output_path",
        "sam_mask_path",
        "sam_visualization_path",
    ]

    selected_columns = [col for col in preferred_columns if col in df.columns]

    worst_cases = df.sort_values("iou", ascending=True).head(top_k)[selected_columns]
    best_cases = df.sort_values("iou", ascending=False).head(top_k)[selected_columns]

    worst_cases.to_csv(output_dir / f"{prompt_name}_worst_cases.csv", index=False)
    best_cases.to_csv(output_dir / f"{prompt_name}_best_cases.csv", index=False)

    return worst_cases, best_cases


def plot_iou_histogram(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["iou"], bins=10)
    plt.xlabel("IoU")
    plt.ylabel("Number of objects")
    plt.title(f"{prompt_name}: IoU Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prompt_name}_iou_histogram.png", dpi=150)
    plt.close()


def plot_score_vs_iou(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["sam_score"], df["iou"])
    plt.xlabel("SAM predicted score")
    plt.ylabel("Ground-truth IoU")
    plt.title(f"{prompt_name}: SAM Score vs Ground-Truth IoU")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prompt_name}_sam_score_vs_iou.png", dpi=150)
    plt.close()


def plot_per_object_mean_iou(
    per_object: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> None:
    sorted_df = per_object.sort_values("object_id")

    plt.figure(figsize=(8, 5))
    plt.bar(sorted_df["object_id"].astype(str), sorted_df["mean_iou"])
    plt.xlabel("Object ID")
    plt.ylabel("Mean IoU")
    plt.title(f"{prompt_name}: Mean IoU per Object ID")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prompt_name}_per_object_mean_iou.png", dpi=150)
    plt.close()


def print_summary(
    results_csv: Path,
    output_dir: Path,
    prompt_name: str,
    summary: pd.DataFrame,
    per_object: pd.DataFrame,
    worst_cases: pd.DataFrame,
    best_cases: pd.DataFrame,
) -> None:
    row = summary.iloc[0]

    print("Prompt name:", prompt_name)
    print("Results CSV:", results_csv)
    print("Analysis output dir:", output_dir)
    print()
    print("Global summary:")
    print(f"Number of evaluated objects: {int(row['num_objects'])}")
    print(f"Mean IoU: {row['mean_iou']:.4f}")
    print(f"Median IoU: {row['median_iou']:.4f}")
    print(f"Min IoU: {row['min_iou']:.4f}")
    print(f"Max IoU: {row['max_iou']:.4f}")
    print(f"Std IoU: {row['std_iou']:.4f}")
    print(f"Mean SAM score: {row['mean_sam_score']:.4f}")

    print()
    print("Per-object summary sorted by lowest mean IoU:")
    print(per_object.to_string(index=False))

    print()
    print(f"Worst {len(worst_cases)} cases:")
    print(worst_cases.to_string(index=False))

    print()
    print(f"Best {len(best_cases)} cases:")
    print(best_cases.to_string(index=False))


def main() -> None:
    args = parse_args()

    results_csv = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    prompt_name = args.prompt_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    validate_results_dataframe(df)

    summary = save_global_summary(df, output_dir, prompt_name)
    per_object = save_per_object_summary(df, output_dir, prompt_name)
    worst_cases, best_cases = save_best_and_worst_cases(
        df=df,
        output_dir=output_dir,
        prompt_name=prompt_name,
        top_k=args.top_k,
    )

    plot_iou_histogram(df, output_dir, prompt_name)
    plot_score_vs_iou(df, output_dir, prompt_name)
    plot_per_object_mean_iou(per_object, output_dir, prompt_name)

    print_summary(
        results_csv=results_csv,
        output_dir=output_dir,
        prompt_name=prompt_name,
        summary=summary,
        per_object=per_object,
        worst_cases=worst_cases,
        best_cases=best_cases,
    )

    print()
    print("Saved analysis files in:", output_dir)


if __name__ == "__main__":
    main()
