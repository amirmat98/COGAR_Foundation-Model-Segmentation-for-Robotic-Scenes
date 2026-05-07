#!/usr/bin/env python3
"""
Export a report-ready COGAR-SimRobotics-500 result package.

This copies curated benchmark CSVs and figures from ignored outputs/
into docs/results/cogar_sim_500/, then writes a short markdown report.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

DOCS_DIR = REPO_ROOT / "docs" / "results" / "cogar_sim_500"
TABLES_DIR = DOCS_DIR / "tables"
FIGURES_DIR = DOCS_DIR / "figures"

OUTPUTS_DIR = REPO_ROOT / "outputs"
INDEXES_DIR = OUTPUTS_DIR / "indexes"
PROMPT_COMPARISON_DIR = OUTPUTS_DIR / "cogar_sim_500" / "analysis_prompt_comparison"

BOX_RESULTS = INDEXES_DIR / "cogar_sim_500_sam_box_clean_results.csv"
POINT_RESULTS = INDEXES_DIR / "cogar_sim_500_sam_point_clean_results.csv"

COMPARISON_OVERALL = PROMPT_COMPARISON_DIR / "sam_box_vs_point_overall.csv"
COMPARISON_BY_CATEGORY = PROMPT_COMPARISON_DIR / "sam_box_vs_point_by_category.csv"
COMPARISON_BY_CHALLENGE = PROMPT_COMPARISON_DIR / "sam_box_vs_point_by_challenge.csv"
COMPARISON_FIGURES_DIR = PROMPT_COMPARISON_DIR / "figures"


KNOWN_DATASET_STATS = {
    "dataset": "COGAR-SimRobotics-500",
    "images": 500,
    "coco_annotations": 8570,
    "categories": 10,
    "clean_filter": "area >= 100, bbox_w >= 5, bbox_h >= 5, visible_ratio >= 0.05",
    "clean_instances": 7274,
}

KNOWN_RESULTS = {
    "box": {
        "prompt": "SAM ViT-B box prompt",
        "objects": 7274,
        "mean_iou": 0.8914,
        "median_iou": 0.9427,
        "mean_sam_score": 0.9523,
    },
    "point": {
        "prompt": "SAM ViT-B single positive point prompt",
        "objects": 7274,
        "mean_iou": 0.8040,
        "median_iou": 0.9126,
        "mean_sam_score": 0.8784,
    },
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def copy_file(src: Path, dst: Path) -> None:
    require_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols:
            return cols[candidate.lower()]
    return None


def summarize_result_csv(path: Path, fallback: dict) -> dict:
    if not path.exists():
        return dict(fallback)

    df = pd.read_csv(path)
    iou_col = find_col(df, ["iou", "mask_iou", "sam_iou", "gt_iou"])
    score_col = find_col(df, ["sam_score", "score", "predicted_iou"])

    summary = dict(fallback)
    summary["objects"] = int(len(df))

    if iou_col is not None:
        summary["mean_iou"] = float(df[iou_col].mean())
        summary["median_iou"] = float(df[iou_col].median())

    if score_col is not None:
        summary["mean_sam_score"] = float(df[score_col].mean())

    return summary


def write_one_row_summary(path: Path, summary: dict) -> None:
    pd.DataFrame([summary]).to_csv(path, index=False)
    print(f"Wrote: {path}")


def write_group_summary(
    input_csv: Path,
    output_csv: Path,
    group_candidates: list[str],
    label: str,
) -> None:
    if not input_csv.exists():
        print(f"Skipped {label}: missing {input_csv}")
        return

    df = pd.read_csv(input_csv)
    iou_col = find_col(df, ["iou", "mask_iou", "sam_iou", "gt_iou"])
    score_col = find_col(df, ["sam_score", "score", "predicted_iou"])
    group_col = find_col(df, group_candidates)

    if iou_col is None or group_col is None:
        print(f"Skipped {label}: missing group or IoU column")
        return

    agg_spec = {
        "objects": (iou_col, "size"),
        "mean_iou": (iou_col, "mean"),
        "median_iou": (iou_col, "median"),
    }

    grouped = df.groupby(group_col, dropna=False).agg(**agg_spec).reset_index()

    if score_col is not None:
        score_grouped = df.groupby(group_col, dropna=False)[score_col].mean().reset_index()
        score_grouped = score_grouped.rename(columns={score_col: "mean_sam_score"})
        grouped = grouped.merge(score_grouped, on=group_col, how="left")

    grouped = grouped.sort_values("mean_iou", ascending=True)
    grouped.to_csv(output_csv, index=False)
    print(f"Wrote: {output_csv}")


def copy_figures() -> list[Path]:
    copied: list[Path] = []

    if not COMPARISON_FIGURES_DIR.exists():
        print(f"No comparison figures directory found: {COMPARISON_FIGURES_DIR}")
        return copied

    for src in sorted(COMPARISON_FIGURES_DIR.iterdir()):
        if src.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            continue
        dst = FIGURES_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"Copied figure: {src} -> {dst}")

    return copied


def markdown_table_from_summary(box: dict, point: dict) -> str:
    delta = point["mean_iou"] - box["mean_iou"]

    rows = [
        "| Prompt | Objects | Mean IoU | Median IoU | Mean SAM score |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Box prompt | {box['objects']} | "
            f"{box['mean_iou']:.4f} | {box['median_iou']:.4f} | {box['mean_sam_score']:.4f} |"
        ),
        (
            f"| Single positive point prompt | {point['objects']} | "
            f"{point['mean_iou']:.4f} | {point['median_iou']:.4f} | {point['mean_sam_score']:.4f} |"
        ),
        (
            f"| Point minus box | — | "
            f"{delta:.4f} | — | {point['mean_sam_score'] - box['mean_sam_score']:.4f} |"
        ),
    ]

    return "\n".join(rows)


def write_report(box: dict, point: dict, copied_figures: list[Path]) -> None:
    delta = point["mean_iou"] - box["mean_iou"]

    figures_section = "\n".join(
        f"- `figures/{fig.name}`" for fig in copied_figures
    ) or "- No figures were copied. Check `outputs/cogar_sim_500/analysis_prompt_comparison/figures/`."

    report = f"""# COGAR-SimRobotics-500 SAM ViT-B Prompt Benchmark

## Dataset

This report summarizes the SAM ViT-B prompt benchmark on **{KNOWN_DATASET_STATS['dataset']}**.

- Images: {KNOWN_DATASET_STATS['images']}
- COCO annotations: {KNOWN_DATASET_STATS['coco_annotations']}
- Categories: {KNOWN_DATASET_STATS['categories']}
- Clean benchmark filter: `{KNOWN_DATASET_STATS['clean_filter']}`
- Clean benchmark object instances: {KNOWN_DATASET_STATS['clean_instances']}

## Evaluated prompts

The benchmark compares two zero-shot prompt types on the same clean non-table object instances:

1. **Box prompt**: the model receives a ground-truth object bounding box.
2. **Single positive point prompt**: the model receives one foreground point sampled from the object mask.

## Main quantitative result

{markdown_table_from_summary(box, point)}

The mean IoU drops from **{box['mean_iou']:.4f}** with box prompts to **{point['mean_iou']:.4f}** with single positive point prompts.
The absolute mean IoU difference is **{delta:.4f}**.

## Interpretation

SAM ViT-B performs better and more consistently with bounding-box prompts than with single foreground point prompts on COGAR-SimRobotics-500.

The result is expected because a box prompt gives SAM stronger spatial constraints around the object extent, while a single point prompt gives weaker information in cluttered robotic scenes. This is especially important for objects with ambiguous boundaries, transparency, specular highlights, thin structures, or heavy occlusion.

## Harder categories for point prompts

The most problematic categories for point prompts compared with box prompts are:

- `glass_object`
- `robot_gripper`
- `tool`
- `cable`

## Harder challenge types

The most difficult challenge groups are:

- `transparent_glass`
- `partial_occlusion`
- `reflective_metal`

## Included clean tables

- `tables/sam_box_clean_results.csv`
- `tables/sam_point_clean_results.csv`
- `tables/sam_box_summary.csv`
- `tables/sam_point_summary.csv`
- `tables/sam_box_by_category.csv`
- `tables/sam_point_by_category.csv`
- `tables/sam_box_by_challenge.csv`
- `tables/sam_point_by_challenge.csv`
- `tables/sam_box_vs_point_overall.csv`
- `tables/sam_box_vs_point_by_category.csv`
- `tables/sam_box_vs_point_by_challenge.csv`

## Included figures

{figures_section}

## Report conclusion

For the current COGAR-SimRobotics-500 benchmark, **SAM ViT-B box prompting is the stronger baseline**.
Single positive point prompting remains useful, but it is less stable on transparent, reflective, occluded, and thin robotic-scene objects.
"""

    report_path = DOCS_DIR / "README.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote: {report_path}")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    required = [
        BOX_RESULTS,
        POINT_RESULTS,
        COMPARISON_OVERALL,
        COMPARISON_BY_CATEGORY,
        COMPARISON_BY_CHALLENGE,
    ]
    for path in required:
        require_file(path)

    copy_file(BOX_RESULTS, TABLES_DIR / "sam_box_clean_results.csv")
    copy_file(POINT_RESULTS, TABLES_DIR / "sam_point_clean_results.csv")
    copy_file(COMPARISON_OVERALL, TABLES_DIR / "sam_box_vs_point_overall.csv")
    copy_file(COMPARISON_BY_CATEGORY, TABLES_DIR / "sam_box_vs_point_by_category.csv")
    copy_file(COMPARISON_BY_CHALLENGE, TABLES_DIR / "sam_box_vs_point_by_challenge.csv")

    box_summary = summarize_result_csv(BOX_RESULTS, KNOWN_RESULTS["box"])
    point_summary = summarize_result_csv(POINT_RESULTS, KNOWN_RESULTS["point"])

    write_one_row_summary(TABLES_DIR / "sam_box_summary.csv", box_summary)
    write_one_row_summary(TABLES_DIR / "sam_point_summary.csv", point_summary)

    write_group_summary(
        BOX_RESULTS,
        TABLES_DIR / "sam_box_by_category.csv",
        ["category_name", "category", "class_name", "label"],
        "box by category",
    )
    write_group_summary(
        POINT_RESULTS,
        TABLES_DIR / "sam_point_by_category.csv",
        ["category_name", "category", "class_name", "label"],
        "point by category",
    )
    write_group_summary(
        BOX_RESULTS,
        TABLES_DIR / "sam_box_by_challenge.csv",
        ["primary_challenge", "challenge", "scene_challenge"],
        "box by challenge",
    )
    write_group_summary(
        POINT_RESULTS,
        TABLES_DIR / "sam_point_by_challenge.csv",
        ["primary_challenge", "challenge", "scene_challenge"],
        "point by challenge",
    )

    copied_figures = copy_figures()
    write_report(box_summary, point_summary, copied_figures)

    print("\nDone. Report package created at:")
    print(DOCS_DIR)


if __name__ == "__main__":
    main()
