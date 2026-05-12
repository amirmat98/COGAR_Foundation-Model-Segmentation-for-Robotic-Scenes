import argparse
from pathlib import Path

import cv2
import pandas as pd


def image_stats(image_path: Path) -> dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        return {
            "readable": False,
            "mean_gray": None,
            "std_gray": None,
            "width": None,
            "height": None,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    return {
        "readable": True,
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "width": int(w),
        "height": int(h),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--output-dir", default="outputs/tables/dataset_audit")
    parser.add_argument("--dark-mean-threshold", type=float, default=35.0)
    parser.add_argument("--flat-std-threshold", type=float, default=8.0)
    parser.add_argument("--min-objects-per-image", type=int, default=3)
    parser.add_argument("--max-objects-per-image", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)

    image_rows = []

    for file_name, group in df.groupby("file_name"):
        image_path = Path(group["image_path"].iloc[0])
        stats = image_stats(image_path)

        num_objects = len(group)
        total_area = float(group["area"].sum()) if "area" in group.columns else None

        bad_reasons = []

        if not stats["readable"]:
            bad_reasons.append("unreadable")
        else:
            if stats["mean_gray"] < args.dark_mean_threshold:
                bad_reasons.append("dark")
            if stats["std_gray"] < args.flat_std_threshold:
                bad_reasons.append("low_texture_or_empty")

        if num_objects < args.min_objects_per_image:
            bad_reasons.append("too_few_objects")

        if num_objects > args.max_objects_per_image:
            bad_reasons.append("too_many_objects")

        image_rows.append(
            {
                "file_name": file_name,
                "image_path": str(image_path),
                "num_objects": num_objects,
                "total_mask_area": total_area,
                "mean_gray": stats["mean_gray"],
                "std_gray": stats["std_gray"],
                "readable": stats["readable"],
                "bad_reasons": ";".join(bad_reasons),
                "is_bad": len(bad_reasons) > 0,
            }
        )

    image_audit = pd.DataFrame(image_rows)
    image_audit.to_csv(output_dir / "image_quality_audit.csv", index=False)

    category_counts = (
        df.groupby("category_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    category_counts.to_csv(output_dir / "category_counts.csv", index=False)

    challenge_counts = (
        df.groupby("challenge_primary")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    challenge_counts.to_csv(output_dir / "challenge_counts.csv", index=False)

    category_by_challenge = pd.crosstab(df["challenge_primary"], df["category_name"])
    category_by_challenge.to_csv(output_dir / "category_by_challenge.csv")

    if "area" in df.columns:
        area_summary = (
            df.groupby("category_name")["area"]
            .agg(["count", "mean", "median", "min", "max"])
            .sort_values("median")
        )
        area_summary.to_csv(output_dir / "area_by_category.csv")

    print("[OK] Wrote dataset audit files to:", output_dir)

    print("\nBad images:")
    bad = image_audit[image_audit["is_bad"]]
    if len(bad) == 0:
        print("None")
    else:
        print(bad[["file_name", "num_objects", "mean_gray", "std_gray", "bad_reasons"]])

    print("\nCategory counts:")
    print(category_counts)

    print("\nChallenge counts:")
    print(challenge_counts)


if __name__ == "__main__":
    main()