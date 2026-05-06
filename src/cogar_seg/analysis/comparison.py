"""Comparison utilities for SAM prompt evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_MERGE_KEYS = [
    "row_index",
    "object_id",
    "image_path",
    "gt_mask_path",
]


@dataclass(frozen=True)
class PromptComparisonRun:
    """Outputs from comparing box-prompt and point-prompt result CSVs."""

    box_results_csv: Path
    point_results_csv: Path
    output_dir: Path
    comparison: pd.DataFrame
    summary: pd.DataFrame
    per_object: pd.DataFrame
    point_best: pd.DataFrame
    box_best: pd.DataFrame
    biggest_differences: pd.DataFrame


def validate_columns(df: pd.DataFrame, required_columns: set[str], name: str) -> None:
    """Validate required columns for a named result DataFrame."""
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{name} results CSV is missing columns: {missing}")


def prepare_box_results(
    box_df: pd.DataFrame,
    merge_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Select and rename box-prompt columns for row-wise comparison."""
    keys = DEFAULT_MERGE_KEYS if merge_keys is None else merge_keys
    required = {
        *keys,
        "file_name",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "sam_score",
        "iou",
        "sam_mask_path",
        "sam_visualization_path",
    }
    validate_columns(box_df, required, "Box-prompt")

    optional_columns = [col for col in ["device", "model_type"] if col in box_df.columns]
    selected = box_df[
        [
            *keys,
            "file_name",
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
            "sam_score",
            "iou",
            "sam_mask_path",
            "sam_visualization_path",
            *optional_columns,
        ]
    ].copy()

    return selected.rename(
        columns={
            "sam_score": "box_sam_score",
            "iou": "box_iou",
            "sam_mask_path": "box_mask_path",
            "sam_visualization_path": "box_visualization_path",
            "device": "box_device",
            "model_type": "box_model_type",
        }
    )


def prepare_point_results(
    point_df: pd.DataFrame,
    merge_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Select and rename point-prompt columns for row-wise comparison."""
    keys = DEFAULT_MERGE_KEYS if merge_keys is None else merge_keys
    required = {
        *keys,
        "point_x",
        "point_y",
        "sam_score",
        "iou",
        "mask_output_path",
        "visualization_output_path",
    }
    validate_columns(point_df, required, "Point-prompt")

    optional_columns = [col for col in ["device", "model_type"] if col in point_df.columns]
    selected = point_df[
        [
            *keys,
            "point_x",
            "point_y",
            "sam_score",
            "iou",
            "mask_output_path",
            "visualization_output_path",
            *optional_columns,
        ]
    ].copy()

    return selected.rename(
        columns={
            "sam_score": "point_sam_score",
            "iou": "point_iou",
            "mask_output_path": "point_mask_path",
            "visualization_output_path": "point_visualization_path",
            "device": "point_device",
            "model_type": "point_model_type",
        }
    )


def compare_results(
    box_df: pd.DataFrame,
    point_df: pd.DataFrame,
    merge_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Merge box and point result rows and compute prompt deltas."""
    keys = DEFAULT_MERGE_KEYS if merge_keys is None else merge_keys
    box_prepared = prepare_box_results(box_df, keys)
    point_prepared = prepare_point_results(point_df, keys)

    merged = box_prepared.merge(
        point_prepared,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(box_df) or len(merged) != len(point_df):
        raise ValueError(
            "Merged row count does not match input row counts. "
            f"box={len(box_df)}, point={len(point_df)}, merged={len(merged)}"
        )

    merged["iou_delta_point_minus_box"] = merged["point_iou"] - merged["box_iou"]
    merged["abs_iou_delta"] = merged["iou_delta_point_minus_box"].abs()
    merged["sam_score_delta_point_minus_box"] = (
        merged["point_sam_score"] - merged["box_sam_score"]
    )
    merged["winner"] = "tie"
    merged.loc[merged["iou_delta_point_minus_box"] > 0, "winner"] = "point"
    merged.loc[merged["iou_delta_point_minus_box"] < 0, "winner"] = "box"

    return merged.sort_values("row_index")


def save_global_summary(comparison: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save global prompt-comparison summary metrics."""
    point_better = int((comparison["winner"] == "point").sum())
    box_better = int((comparison["winner"] == "box").sum())
    ties = int((comparison["winner"] == "tie").sum())

    summary = pd.DataFrame(
        [
            {
                "num_objects": len(comparison),
                "box_mean_iou": comparison["box_iou"].mean(),
                "box_median_iou": comparison["box_iou"].median(),
                "box_min_iou": comparison["box_iou"].min(),
                "box_max_iou": comparison["box_iou"].max(),
                "box_std_iou": comparison["box_iou"].std(),
                "point_mean_iou": comparison["point_iou"].mean(),
                "point_median_iou": comparison["point_iou"].median(),
                "point_min_iou": comparison["point_iou"].min(),
                "point_max_iou": comparison["point_iou"].max(),
                "point_std_iou": comparison["point_iou"].std(),
                "mean_iou_delta_point_minus_box": comparison[
                    "iou_delta_point_minus_box"
                ].mean(),
                "median_iou_delta_point_minus_box": comparison[
                    "iou_delta_point_minus_box"
                ].median(),
                "point_better_count": point_better,
                "box_better_count": box_better,
                "tie_count": ties,
                "point_better_fraction": point_better / len(comparison),
                "box_better_fraction": box_better / len(comparison),
                "tie_fraction": ties / len(comparison),
                "box_mean_sam_score": comparison["box_sam_score"].mean(),
                "point_mean_sam_score": comparison["point_sam_score"].mean(),
            }
        ]
    )
    summary.to_csv(output_dir / "box_vs_point_global_summary.csv", index=False)
    return summary


def save_per_object_summary(comparison: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Save per-object prompt comparison metrics."""
    per_object = (
        comparison.groupby("object_id")
        .agg(
            count=("row_index", "count"),
            box_mean_iou=("box_iou", "mean"),
            point_mean_iou=("point_iou", "mean"),
            mean_iou_delta_point_minus_box=("iou_delta_point_minus_box", "mean"),
            median_iou_delta_point_minus_box=("iou_delta_point_minus_box", "median"),
            min_iou_delta_point_minus_box=("iou_delta_point_minus_box", "min"),
            max_iou_delta_point_minus_box=("iou_delta_point_minus_box", "max"),
            box_wins=("winner", lambda s: int((s == "box").sum())),
            point_wins=("winner", lambda s: int((s == "point").sum())),
            ties=("winner", lambda s: int((s == "tie").sum())),
        )
        .reset_index()
        .sort_values("mean_iou_delta_point_minus_box", ascending=True)
    )
    per_object.to_csv(output_dir / "box_vs_point_per_object_summary.csv", index=False)
    return per_object


def save_best_worst_prompt_differences(
    comparison: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Save strongest prompt wins and largest absolute prompt differences."""
    selected_columns = [
        "row_index",
        "file_name",
        "object_id",
        "box_iou",
        "point_iou",
        "iou_delta_point_minus_box",
        "box_sam_score",
        "point_sam_score",
        "winner",
        "box_visualization_path",
        "point_visualization_path",
    ]

    point_best = (
        comparison.sort_values("iou_delta_point_minus_box", ascending=False)
        .head(top_k)[selected_columns]
    )
    box_best = (
        comparison.sort_values("iou_delta_point_minus_box", ascending=True)
        .head(top_k)[selected_columns]
    )
    biggest_differences = (
        comparison.sort_values("abs_iou_delta", ascending=False)
        .head(top_k)[selected_columns]
    )

    point_best.to_csv(output_dir / "point_prompt_strongest_wins.csv", index=False)
    box_best.to_csv(output_dir / "box_prompt_strongest_wins.csv", index=False)
    biggest_differences.to_csv(output_dir / "largest_prompt_differences.csv", index=False)

    return point_best, box_best, biggest_differences


def plot_box_vs_point_scatter(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Plot box-prompt IoU against point-prompt IoU."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(comparison["box_iou"], comparison["point_iou"])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Box-prompt IoU")
    plt.ylabel("Point-prompt IoU")
    plt.title("SAM Box Prompt vs Point Prompt IoU")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "box_vs_point_iou_scatter.png", dpi=150)
    plt.close()


def plot_iou_delta_histogram(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Plot the distribution of point-minus-box IoU deltas."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(comparison["iou_delta_point_minus_box"], bins=12)
    plt.axvline(0, linestyle="--")
    plt.xlabel("IoU delta: point prompt - box prompt")
    plt.ylabel("Number of objects")
    plt.title("Distribution of Prompt IoU Difference")
    plt.tight_layout()
    plt.savefig(output_dir / "box_vs_point_iou_delta_histogram.png", dpi=150)
    plt.close()


def plot_per_object_delta(per_object: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean point-minus-box IoU delta per object ID."""
    import matplotlib.pyplot as plt

    sorted_df = per_object.sort_values("object_id")

    plt.figure(figsize=(8, 5))
    plt.bar(
        sorted_df["object_id"].astype(str),
        sorted_df["mean_iou_delta_point_minus_box"],
    )
    plt.axhline(0, linestyle="--")
    plt.xlabel("Object ID")
    plt.ylabel("Mean IoU delta: point - box")
    plt.title("Mean Prompt Difference per Object ID")
    plt.tight_layout()
    plt.savefig(output_dir / "box_vs_point_per_object_delta.png", dpi=150)
    plt.close()


def compare_prompt_results(
    box_results_csv: str | Path,
    point_results_csv: str | Path,
    output_dir: str | Path,
    top_k: int = 10,
    save_plots: bool = True,
) -> PromptComparisonRun:
    """Compare SAM box-prompt and point-prompt result CSVs row by row."""
    resolved_box_csv = Path(box_results_csv)
    resolved_point_csv = Path(point_results_csv)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_box_csv.exists():
        raise FileNotFoundError(f"Box results CSV not found: {resolved_box_csv}")

    if not resolved_point_csv.exists():
        raise FileNotFoundError(f"Point results CSV not found: {resolved_point_csv}")

    box_df = pd.read_csv(resolved_box_csv)
    point_df = pd.read_csv(resolved_point_csv)

    comparison = compare_results(box_df, point_df)
    comparison.to_csv(
        resolved_output_dir / "box_vs_point_rowwise_comparison.csv",
        index=False,
    )

    summary = save_global_summary(comparison, resolved_output_dir)
    per_object = save_per_object_summary(comparison, resolved_output_dir)
    point_best, box_best, biggest_differences = save_best_worst_prompt_differences(
        comparison=comparison,
        output_dir=resolved_output_dir,
        top_k=top_k,
    )

    if save_plots:
        plot_box_vs_point_scatter(comparison, resolved_output_dir)
        plot_iou_delta_histogram(comparison, resolved_output_dir)
        plot_per_object_delta(per_object, resolved_output_dir)

    return PromptComparisonRun(
        box_results_csv=resolved_box_csv,
        point_results_csv=resolved_point_csv,
        output_dir=resolved_output_dir,
        comparison=comparison,
        summary=summary,
        per_object=per_object,
        point_best=point_best,
        box_best=box_best,
        biggest_differences=biggest_differences,
    )


def print_prompt_comparison_summary(run: PromptComparisonRun) -> None:
    """Print the compact CLI summary for a prompt-comparison run."""
    row = run.summary.iloc[0]

    print("Prompt comparison summary")
    print("Output dir:", run.output_dir)
    print()
    print(f"Objects compared: {int(row['num_objects'])}")
    print(f"Box mean IoU: {row['box_mean_iou']:.4f}")
    print(f"Point mean IoU: {row['point_mean_iou']:.4f}")
    print(
        "Mean IoU delta, point - box: "
        f"{row['mean_iou_delta_point_minus_box']:.4f}"
    )
    print(f"Box wins: {int(row['box_better_count'])}")
    print(f"Point wins: {int(row['point_better_count'])}")
    print(f"Ties: {int(row['tie_count'])}")
    print()
    print("Per-object comparison:")
    print(run.per_object.to_string(index=False))
    print()
    print("Strongest point-prompt wins:")
    print(run.point_best.to_string(index=False))
    print()
    print("Strongest box-prompt wins:")
    print(run.box_best.to_string(index=False))
    print()
    print("Largest absolute prompt differences:")
    print(run.biggest_differences.to_string(index=False))
