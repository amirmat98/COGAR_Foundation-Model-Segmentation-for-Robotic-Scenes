import argparse
from pathlib import Path

import pandas as pd


def load_result(path: str, prompt: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prompt"] = prompt
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--box", default="outputs/results/sim_sam_vit_b_box_25.csv")
    parser.add_argument("--point", default="outputs/results/sim_sam_vit_b_point_25.csv")
    parser.add_argument("--auto", default="outputs/results/sim_sam_vit_b_auto_25.csv")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.index)

    results = []
    for prompt, path in [
        ("box", args.box),
        ("point", args.point),
        ("auto", args.auto),
    ]:
        if Path(path).exists():
            results.append(load_result(path, prompt))
        else:
            print(f"[WARN] Missing result file: {path}")

    if not results:
        raise ValueError("No result files found.")

    results = pd.concat(results, ignore_index=True)

    # Results already contain object_id and row_index. Merge object metadata by object_id.
    meta_cols = [
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

    merged = results.merge(
        index[meta_cols],
        on="object_id",
        how="left",
        suffixes=("", "_index"),
    )

    # Prefer metadata from finalized index if duplicate columns exist.
    if "category_name_index" in merged.columns:
        merged["category_name"] = merged["category_name"].fillna(merged["category_name_index"])
    if "challenge_primary_index" in merged.columns:
        merged["challenge_primary"] = merged["challenge_primary"].fillna(merged["challenge_primary_index"])

    latency_col = "latency_sec" if "latency_sec" in merged.columns else None
    fps_col = "fps" if "fps" in merged.columns else None

    agg_dict = {
        "iou": ["count", "mean", "median", "min", "max", "std"],
        "boundary_f1": ["mean", "median"],
        "sam_score": ["mean", "median"],
    }

    if latency_col:
        agg_dict[latency_col] = ["mean", "median"]
    if fps_col:
        agg_dict[fps_col] = ["mean", "median"]

    global_summary = merged.groupby("prompt").agg(agg_dict)
    global_summary.to_csv(output_dir / "sim_sam_vit_b_global_summary.csv")

    category_summary = merged.groupby(["prompt", "category_name"]).agg(agg_dict)
    category_summary.to_csv(output_dir / "sim_sam_vit_b_per_category.csv")

    challenge_summary = merged.groupby(["prompt", "challenge_primary"]).agg(agg_dict)
    challenge_summary.to_csv(output_dir / "sim_sam_vit_b_per_challenge.csv")

    # Practical failure tables.
    failures = merged.sort_values("iou").groupby("prompt").head(25)
    failures.to_csv(output_dir / "sim_sam_vit_b_failure_cases.csv", index=False)

    print("[OK] Wrote:")
    print(output_dir / "sim_sam_vit_b_global_summary.csv")
    print(output_dir / "sim_sam_vit_b_per_category.csv")
    print(output_dir / "sim_sam_vit_b_per_challenge.csv")
    print(output_dir / "sim_sam_vit_b_failure_cases.csv")

    print("\nGlobal summary:")
    print(global_summary)


if __name__ == "__main__":
    main()

