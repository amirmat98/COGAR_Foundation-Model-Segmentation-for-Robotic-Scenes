import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cogar_seg.cv_compat import cv2


FLAG_COLUMNS = [
    "is_reflective",
    "is_transparent",
    "is_small_part",
    "is_occluded",
    "is_dynamic",
]

TARGET_CATEGORIES = {
    "robot_gripper",
    "metal_part",
    "glass_object",
    "plastic_object",
    "screw",
    "connector",
    "cable",
    "tool",
    "box",
}


def bool_like(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def image_stats(image_path: Path) -> dict[str, Any]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "readable": False,
            "mean_gray": np.nan,
            "std_gray": np.nan,
            "width": np.nan,
            "height": np.nan,
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


def add_size_band(area: float) -> str:
    if area < 100:
        return "tiny"
    if area < 1000:
        return "small"
    if area < 10000:
        return "medium"
    return "large"


def verify_object_flags(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "category_name" not in df.columns:
        return ["category_name column missing; cannot verify object flags"]

    category = df["category_name"].astype(str)
    checks = [
        ("is_transparent", category.eq("glass_object"), "glass_object"),
        ("is_small_part", category.isin(["screw", "connector"]), "screw/connector"),
        ("is_reflective", category.isin(["metal_part", "tool"]), "metal_part/tool"),
    ]
    for col, allowed_mask, label in checks:
        if col not in df.columns:
            warnings.append(f"{col} column missing; cannot verify flag correctness")
            continue
        true_mask = df[col].map(bool_like)
        bad = df[true_mask & ~allowed_mask]
        if not bad.empty:
            categories = sorted(bad["category_name"].astype(str).unique().tolist())
            warnings.append(
                f"{col} is true for categories outside {label}: {categories}"
            )
    return warnings


def audit_sim_dataset(
    index_path: str | Path,
    output_dir: str | Path,
    dark_mean_threshold: float = 35.0,
    flat_std_threshold: float = 8.0,
    min_objects_per_image: int = 3,
    max_objects_per_image: int = 25,
    min_mask_area: float = 25.0,
    target_min_objects: int = 6,
    target_max_objects: int = 18,
    max_mask_area_ratio: float = 0.95,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)
    if df.empty:
        raise ValueError(f"Index is empty: {index_path}")

    if "file_name" not in df.columns or "image_path" not in df.columns:
        raise ValueError("Index must contain file_name and image_path columns")

    if "area" not in df.columns:
        df = df.copy()
        df["area"] = np.nan

    image_rows: list[dict[str, Any]] = []
    for file_name, group in df.groupby("file_name", sort=True):
        image_path = Path(str(group["image_path"].iloc[0]))
        stats = image_stats(image_path)

        categories = group["category_name"].astype(str) if "category_name" in group else pd.Series([])
        num_objects = int(len(group))
        num_categories = int(categories.nunique()) if not categories.empty else 0
        total_area = float(pd.to_numeric(group["area"], errors="coerce").fillna(0).sum())
        challenge = (
            str(group["challenge_primary"].iloc[0])
            if "challenge_primary" in group.columns
            else ""
        )

        bad_reasons: list[str] = []
        if not stats["readable"]:
            bad_reasons.append("unreadable")
        else:
            if float(stats["mean_gray"]) < dark_mean_threshold:
                bad_reasons.append("dark")
            if float(stats["std_gray"]) < flat_std_threshold:
                bad_reasons.append("low_texture_or_empty")
            image_area = float(stats["width"]) * float(stats["height"])
            if image_area > 0 and total_area / image_area > max_mask_area_ratio:
                bad_reasons.append("excessive_mask_area")

        if num_objects < min_objects_per_image:
            bad_reasons.append("too_few_objects")
        if num_objects > max_objects_per_image:
            bad_reasons.append("too_many_objects")
        if num_categories < 2:
            bad_reasons.append("too_few_categories")
        if categories.empty or not categories.isin(TARGET_CATEGORIES).any():
            bad_reasons.append("no_target_objects")

        object_areas = pd.to_numeric(group["area"], errors="coerce").fillna(0)
        if num_objects > 0 and float((object_areas < min_mask_area).mean()) >= 0.5:
            bad_reasons.append("tiny_objects_dominant")

        image_rows.append(
            {
                "file_name": file_name,
                "image_path": str(image_path),
                "num_objects": num_objects,
                "num_categories": num_categories,
                "total_mask_area": total_area,
                "mean_gray": stats["mean_gray"],
                "std_gray": stats["std_gray"],
                "readable": bool(stats["readable"]),
                "challenge_primary": challenge,
                "bad_reasons": ";".join(bad_reasons),
                "is_bad": bool(bad_reasons),
            }
        )

    image_audit = pd.DataFrame(image_rows)
    image_audit.to_csv(output_path / "image_quality_audit.csv", index=False)

    category_counts = (
        df.groupby("category_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if "category_name" in df.columns
        else pd.DataFrame(columns=["category_name", "count"])
    )
    category_counts.to_csv(output_path / "category_counts.csv", index=False)

    challenge_counts = (
        df.groupby("challenge_primary")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        if "challenge_primary" in df.columns
        else pd.DataFrame(columns=["challenge_primary", "count"])
    )
    challenge_counts.to_csv(output_path / "challenge_counts.csv", index=False)

    if {"challenge_primary", "category_name"}.issubset(df.columns):
        category_by_challenge = pd.crosstab(df["challenge_primary"], df["category_name"])
    else:
        category_by_challenge = pd.DataFrame()
    category_by_challenge.to_csv(output_path / "category_by_challenge.csv")

    area_df = df.copy()
    area_df["area"] = pd.to_numeric(area_df["area"], errors="coerce").fillna(0)
    area_df["size_band"] = area_df["area"].map(add_size_band)
    if "category_name" in area_df.columns:
        area_stats = area_df.groupby("category_name")["area"].agg(
            ["count", "min", "median", "mean", "max"]
        )
        size_bands = pd.crosstab(area_df["category_name"], area_df["size_band"])
        for col in ["tiny", "small", "medium", "large"]:
            if col not in size_bands.columns:
                size_bands[col] = 0
        area_by_category = area_stats.join(size_bands[["tiny", "small", "medium", "large"]])
        area_by_category = area_by_category.sort_values("median")
    else:
        area_by_category = pd.DataFrame()
    area_by_category.to_csv(output_path / "area_by_category.csv")

    flag_rows = []
    for col in FLAG_COLUMNS:
        if col in df.columns:
            flag_rows.append({"flag": col, "count_true": int(df[col].map(bool_like).sum())})
        else:
            flag_rows.append({"flag": col, "count_true": 0})
    object_flag_counts = pd.DataFrame(flag_rows)
    object_flag_counts.to_csv(output_path / "object_flag_counts.csv", index=False)

    bad_images = image_audit[image_audit["is_bad"]]["file_name"].astype(str).tolist()
    (output_path / "bad_images.txt").write_text(
        "\n".join(bad_images) + ("\n" if bad_images else ""),
        encoding="utf-8",
    )

    warnings = verify_object_flags(df)
    total_objects = int(len(df))
    if total_objects > 0:
        for _, row in challenge_counts.iterrows():
            if float(row["count"]) / total_objects < 0.10:
                warnings.append(
                    f"challenge {row['challenge_primary']} has fewer than 10% of objects"
                )
        for _, row in category_counts.iterrows():
            if float(row["count"]) / total_objects < 0.03:
                warnings.append(
                    f"category {row['category_name']} has fewer than 3% of objects"
                )
    if (image_audit["num_objects"] > max_objects_per_image).any():
        warnings.append(f"at least one image has more than {max_objects_per_image} objects")

    object_counts = image_audit["num_objects"].astype(float)
    summary = {
        "index": str(index_path),
        "total_images": int(image_audit["file_name"].nunique()),
        "total_objects": total_objects,
        "bad_images_count": int(image_audit["is_bad"].sum()),
        "bad_images": bad_images,
        "object_count_per_image": {
            "min": float(object_counts.min()),
            "median": float(object_counts.median()),
            "mean": float(object_counts.mean()),
            "max": float(object_counts.max()),
            "target_min": int(target_min_objects),
            "target_max": int(target_max_objects),
        },
        "challenge_distribution": challenge_counts.set_index("challenge_primary")["count"].to_dict()
        if not challenge_counts.empty
        else {},
        "category_distribution": category_counts.set_index("category_name")["count"].to_dict()
        if not category_counts.empty
        else {},
        "warnings": warnings,
    }
    (output_path / "audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("[OK] Wrote dataset audit files to:", output_path)
    print(f"Total images: {summary['total_images']}")
    print(f"Total objects: {summary['total_objects']}")
    print(f"Bad images: {summary['bad_images_count']}")
    if bad_images:
        print("Bad image names:", ", ".join(bad_images))
    counts = summary["object_count_per_image"]
    print(
        "Objects per image: "
        f"min={counts['min']:.0f}, median={counts['median']:.1f}, "
        f"mean={counts['mean']:.2f}, max={counts['max']:.0f}"
    )
    print("\nChallenge distribution:")
    print(challenge_counts if not challenge_counts.empty else "No challenge_primary column")
    print("\nCategory distribution:")
    print(category_counts if not category_counts.empty else "No category_name column")
    for warning in warnings:
        print(f"[WARN] {warning}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit COGAR-Sim dataset quality.")
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--output-dir", default="outputs/tables/dataset_audit")
    parser.add_argument("--dark-mean-threshold", type=float, default=35.0)
    parser.add_argument("--flat-std-threshold", type=float, default=8.0)
    parser.add_argument("--min-objects-per-image", type=int, default=3)
    parser.add_argument("--max-objects-per-image", type=int, default=25)
    parser.add_argument("--min-mask-area", type=float, default=25.0)
    parser.add_argument("--target-min-objects", type=int, default=6)
    parser.add_argument("--target-max-objects", type=int, default=18)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_sim_dataset(
        index_path=args.index,
        output_dir=args.output_dir,
        dark_mean_threshold=args.dark_mean_threshold,
        flat_std_threshold=args.flat_std_threshold,
        min_objects_per_image=args.min_objects_per_image,
        max_objects_per_image=args.max_objects_per_image,
        min_mask_area=args.min_mask_area,
        target_min_objects=args.target_min_objects,
        target_max_objects=args.target_max_objects,
        max_mask_area_ratio=args.max_mask_area_ratio,
    )


if __name__ == "__main__":
    main()
