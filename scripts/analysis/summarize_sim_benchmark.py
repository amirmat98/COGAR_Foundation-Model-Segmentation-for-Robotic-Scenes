import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = [
    "prompt",
    "group",
    "value",
    "count",
    "mean_iou",
    "median_iou",
    "min_iou",
    "max_iou",
    "std_iou",
    "mean_boundary_f1",
    "median_boundary_f1",
    "mean_sam_score",
    "median_sam_score",
    "mean_latency",
    "mean_fps",
    "success_rate_iou_ge_0_90",
    "success_rate_iou_ge_0_75",
    "success_rate_iou_ge_0_50",
    "catastrophic_failure_rate_iou_lt_0_10",
]


def load_result(path: str | Path, prompt: str) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing result file: {path}")
        return None
    df = pd.read_csv(path)
    df["prompt"] = prompt
    if "latency_sec" not in df.columns and "image_gen_latency_sec" in df.columns:
        df["latency_sec"] = df["image_gen_latency_sec"]
    if "fps" not in df.columns and "image_gen_fps" in df.columns:
        df["fps"] = df["image_gen_fps"]
    return df


def attach_index_metadata(results: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    index_meta = index.reset_index(names="row_index_index")
    index_meta["row_index"] = index_meta["row_index_index"].astype(int)
    meta_cols = [
        "row_index",
        "object_id",
        "category_name",
        "challenge_primary",
        "is_reflective",
        "is_transparent",
        "is_occluded",
        "is_small_part",
        "is_dynamic",
        "area",
    ]
    available = [col for col in meta_cols if col in index_meta.columns]

    if "row_index" in results.columns:
        merged = results.merge(
            index_meta[available],
            on="row_index",
            how="left",
            suffixes=("", "_index"),
        )
    elif "object_id" in results.columns:
        merged = results.merge(
            index_meta[available],
            on="object_id",
            how="left",
            suffixes=("", "_index"),
        )
    else:
        merged = results.copy()

    for col in ["category_name", "challenge_primary"]:
        index_col = f"{col}_index"
        if index_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[index_col])
            else:
                merged[col] = merged[index_col]
    return merged


def summarize_subset(df: pd.DataFrame, prompt: str, group: str, value: str) -> dict:
    iou = pd.to_numeric(df["iou"], errors="coerce")
    boundary_f1 = pd.to_numeric(df.get("boundary_f1", pd.Series(dtype=float)), errors="coerce")
    sam_score = pd.to_numeric(df.get("sam_score", pd.Series(dtype=float)), errors="coerce")
    latency = pd.to_numeric(df.get("latency_sec", pd.Series(dtype=float)), errors="coerce")
    fps = pd.to_numeric(df.get("fps", pd.Series(dtype=float)), errors="coerce")

    return {
        "prompt": prompt,
        "group": group,
        "value": value,
        "count": int(iou.count()),
        "mean_iou": float(iou.mean()),
        "median_iou": float(iou.median()),
        "min_iou": float(iou.min()),
        "max_iou": float(iou.max()),
        "std_iou": float(iou.std(ddof=0)),
        "mean_boundary_f1": float(boundary_f1.mean()),
        "median_boundary_f1": float(boundary_f1.median()),
        "mean_sam_score": float(sam_score.mean()),
        "median_sam_score": float(sam_score.median()),
        "mean_latency": float(latency.mean()),
        "mean_fps": float(fps.mean()),
        "success_rate_iou_ge_0_90": float((iou >= 0.90).mean()),
        "success_rate_iou_ge_0_75": float((iou >= 0.75).mean()),
        "success_rate_iou_ge_0_50": float((iou >= 0.50).mean()),
        "catastrophic_failure_rate_iou_lt_0_10": float((iou < 0.10).mean()),
    }


def summarize_by_group(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    if group_col is None:
        for prompt, prompt_df in df.groupby("prompt", sort=True):
            rows.append(summarize_subset(prompt_df, str(prompt), "global", "all"))
    else:
        if group_col not in df.columns:
            return pd.DataFrame(columns=SUMMARY_COLUMNS)
        for (prompt, value), group_df in df.groupby(["prompt", group_col], sort=True):
            rows.append(summarize_subset(group_df, str(prompt), group_col, str(value)))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize COGAR-Sim SAM benchmark CSVs.")
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1_filtered.csv")
    parser.add_argument("--box", default="outputs/results/sim_sam_vit_b_box_v1.csv")
    parser.add_argument("--point", default="outputs/results/sim_sam_vit_b_point_v1.csv")
    parser.add_argument("--auto", default="outputs/results/sim_sam_vit_b_auto_v1.csv")
    parser.add_argument("--output-dir", default="outputs/tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.index)
    result_frames = [
        load_result(args.box, "box"),
        load_result(args.point, "point"),
        load_result(args.auto, "auto"),
    ]
    result_frames = [df for df in result_frames if df is not None]
    if not result_frames:
        raise ValueError("No result files found.")

    merged = attach_index_metadata(pd.concat(result_frames, ignore_index=True), index)
    if "iou" not in merged.columns:
        raise ValueError("Result CSVs must contain an iou column")

    global_summary = summarize_by_group(merged)
    per_category = summarize_by_group(merged, "category_name")
    per_challenge = summarize_by_group(merged, "challenge_primary")

    global_summary.to_csv(output_dir / "sim_sam_vit_b_global_summary.csv", index=False)
    per_category.to_csv(output_dir / "sim_sam_vit_b_per_category.csv", index=False)
    per_challenge.to_csv(output_dir / "sim_sam_vit_b_per_challenge.csv", index=False)

    failure_cases = merged.sort_values(["prompt", "iou"]).groupby("prompt").head(25)
    failure_cases.to_csv(output_dir / "sim_sam_vit_b_failure_cases.csv", index=False)

    print("[OK] Wrote:")
    print(output_dir / "sim_sam_vit_b_global_summary.csv")
    print(output_dir / "sim_sam_vit_b_per_category.csv")
    print(output_dir / "sim_sam_vit_b_per_challenge.csv")
    print(output_dir / "sim_sam_vit_b_failure_cases.csv")
    print("\nGlobal summary:")
    print(global_summary)


if __name__ == "__main__":
    main()
