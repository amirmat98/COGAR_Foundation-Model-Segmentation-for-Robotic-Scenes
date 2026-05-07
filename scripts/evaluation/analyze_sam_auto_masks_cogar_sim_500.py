#!/usr/bin/env python3
"""
Analyze SAM automatic-mask results for COGAR-SimRobotics-500.
Creates overall, per-category, and per-challenge tables and figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def save_barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values(value_col, ascending=True)

    plt.figure(figsize=(10, max(4, 0.45 * len(plot_df))))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col])
    plt.xlabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Wrote figure: {output_path}")


def main(args: argparse.Namespace) -> None:
    input_csv = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    required = [
        "category_name",
        "primary_challenge",
        "box_prompt_iou",
        "sam_auto_best_iou",
        "sam_auto_num_masks_image",
        "image_path",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    overall = pd.DataFrame([
        {
            "objects": len(df),
            "images": df["image_path"].nunique(),
            "auto_mean_iou": df["sam_auto_best_iou"].mean(),
            "auto_median_iou": df["sam_auto_best_iou"].median(),
            "box_mean_iou": df["box_prompt_iou"].mean(),
            "box_median_iou": df["box_prompt_iou"].median(),
            "mean_iou_drop_auto_minus_box": df["sam_auto_best_iou"].mean() - df["box_prompt_iou"].mean(),
            "mean_generated_masks_per_image": df.groupby("image_path")["sam_auto_num_masks_image"].first().mean(),
            "median_generated_masks_per_image": df.groupby("image_path")["sam_auto_num_masks_image"].first().median(),
        }
    ])

    by_category = (
        df.groupby("category_name", dropna=False)
        .agg(
            count=("sam_auto_best_iou", "size"),
            auto_mean_iou=("sam_auto_best_iou", "mean"),
            auto_median_iou=("sam_auto_best_iou", "median"),
            box_mean_iou=("box_prompt_iou", "mean"),
            box_median_iou=("box_prompt_iou", "median"),
            mean_generated_masks_per_image=("sam_auto_num_masks_image", "mean"),
        )
        .reset_index()
    )
    by_category["mean_iou_drop_auto_minus_box"] = by_category["auto_mean_iou"] - by_category["box_mean_iou"]
    by_category = by_category.sort_values("auto_mean_iou", ascending=True)

    by_challenge = (
        df.groupby("primary_challenge", dropna=False)
        .agg(
            count=("sam_auto_best_iou", "size"),
            auto_mean_iou=("sam_auto_best_iou", "mean"),
            auto_median_iou=("sam_auto_best_iou", "median"),
            box_mean_iou=("box_prompt_iou", "mean"),
            box_median_iou=("box_prompt_iou", "median"),
            mean_generated_masks_per_image=("sam_auto_num_masks_image", "mean"),
        )
        .reset_index()
    )
    by_challenge["mean_iou_drop_auto_minus_box"] = by_challenge["auto_mean_iou"] - by_challenge["box_mean_iou"]
    by_challenge = by_challenge.sort_values("auto_mean_iou", ascending=True)

    overall_path = output_dir / "sam_auto_overall.csv"
    by_category_path = output_dir / "sam_auto_by_category.csv"
    by_challenge_path = output_dir / "sam_auto_by_challenge.csv"

    overall.to_csv(overall_path, index=False)
    by_category.to_csv(by_category_path, index=False)
    by_challenge.to_csv(by_challenge_path, index=False)

    print(f"Wrote: {overall_path}")
    print(f"Wrote: {by_category_path}")
    print(f"Wrote: {by_challenge_path}")

    save_barh(
        by_category,
        "category_name",
        "auto_mean_iou",
        "SAM ViT-B Automatic Masks: Mean IoU by Category",
        figures_dir / "sam_auto_by_category.png",
    )

    save_barh(
        by_challenge,
        "primary_challenge",
        "auto_mean_iou",
        "SAM ViT-B Automatic Masks: Mean IoU by Challenge",
        figures_dir / "sam_auto_by_challenge.png",
    )

    print("\nOverall:")
    print(overall.to_string(index=False))

    print("\nWorst categories:")
    print(by_category.head(10).to_string(index=False))

    print("\nWorst challenges:")
    print(by_challenge.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="outputs/indexes/cogar_sim_500_sam_auto_clean_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cogar_sim_500/analysis_sam_auto_masks",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
