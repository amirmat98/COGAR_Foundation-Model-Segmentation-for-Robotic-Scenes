"""Analysis utilities for prompt-evaluation result CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PromptAnalysisRun:
    """Outputs from one prompt-results analysis run."""

    results_csv: Path
    output_dir: Path
    prompt_name: str
    summary: pd.DataFrame
    per_object: pd.DataFrame
    worst_cases: pd.DataFrame
    best_cases: pd.DataFrame


def validate_results_dataframe(df: pd.DataFrame) -> None:
    """Validate the common columns required for prompt result analysis."""
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
    """Save global IoU and SAM-score summary metrics."""
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
    summary.to_csv(output_dir / f"{prompt_name}_global_summary.csv", index=False)
    return summary


def save_per_object_summary(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
) -> pd.DataFrame:
    """Save per-object summary metrics sorted by lowest mean IoU."""
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
    per_object.to_csv(output_dir / f"{prompt_name}_per_object_summary.csv", index=False)
    return per_object


def save_best_and_worst_cases(
    df: pd.DataFrame,
    output_dir: Path,
    prompt_name: str,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save the best and worst IoU cases for inspection."""
    preferred_columns = [
        "row_index",
        "file_name",
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
    """Plot a prompt IoU histogram."""
    import matplotlib.pyplot as plt

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
    """Plot SAM predicted score against ground-truth IoU."""
    import matplotlib.pyplot as plt

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
    """Plot mean IoU for each object ID."""
    import matplotlib.pyplot as plt

    sorted_df = per_object.sort_values("object_id")

    plt.figure(figsize=(8, 5))
    plt.bar(sorted_df["object_id"].astype(str), sorted_df["mean_iou"])
    plt.xlabel("Object ID")
    plt.ylabel("Mean IoU")
    plt.title(f"{prompt_name}: Mean IoU per Object ID")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prompt_name}_per_object_mean_iou.png", dpi=150)
    plt.close()


def analyze_prompt_results(
    results_csv: str | Path,
    output_dir: str | Path,
    prompt_name: str,
    top_k: int = 10,
    save_plots: bool = True,
) -> PromptAnalysisRun:
    """Analyze one SAM prompt-results CSV and write summary artifacts."""
    resolved_results_csv = Path(results_csv)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {resolved_results_csv}")

    df = pd.read_csv(resolved_results_csv)
    validate_results_dataframe(df)

    summary = save_global_summary(df, resolved_output_dir, prompt_name)
    per_object = save_per_object_summary(df, resolved_output_dir, prompt_name)
    worst_cases, best_cases = save_best_and_worst_cases(
        df=df,
        output_dir=resolved_output_dir,
        prompt_name=prompt_name,
        top_k=top_k,
    )

    if save_plots:
        plot_iou_histogram(df, resolved_output_dir, prompt_name)
        plot_score_vs_iou(df, resolved_output_dir, prompt_name)
        plot_per_object_mean_iou(per_object, resolved_output_dir, prompt_name)

    return PromptAnalysisRun(
        results_csv=resolved_results_csv,
        output_dir=resolved_output_dir,
        prompt_name=prompt_name,
        summary=summary,
        per_object=per_object,
        worst_cases=worst_cases,
        best_cases=best_cases,
    )


def print_prompt_analysis_summary(run: PromptAnalysisRun) -> None:
    """Print the compact CLI summary for an analysis run."""
    row = run.summary.iloc[0]

    print("Prompt name:", run.prompt_name)
    print("Results CSV:", run.results_csv)
    print("Analysis output dir:", run.output_dir)
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
    print(run.per_object.to_string(index=False))
    print()
    print(f"Worst {len(run.worst_cases)} cases:")
    print(run.worst_cases.to_string(index=False))
    print()
    print(f"Best {len(run.best_cases)} cases:")
    print(run.best_cases.to_string(index=False))
