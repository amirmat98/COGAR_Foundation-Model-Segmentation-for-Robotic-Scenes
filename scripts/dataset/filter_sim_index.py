import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import pandas as pd


PATH_COLUMNS = ["image_path", "binary_mask_path"]


def bool_like(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def split_labels_for_images(image_ids: Iterable[int]) -> dict[int, str]:
    ids = sorted({int(image_id) for image_id in image_ids})
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        train_count, val_count = 1, 0
    elif n == 2:
        train_count, val_count = 1, 0
    elif n == 3:
        train_count, val_count = 1, 1
    else:
        train_count = max(1, int(round(0.70 * n)))
        val_count = max(1, int(round(0.15 * n)))
        if train_count + val_count >= n:
            val_count = max(1, n - train_count - 1)
        if n - train_count - val_count <= 0:
            train_count = max(1, n - val_count - 1)

    result: dict[int, str] = {}
    for idx, image_id in enumerate(ids):
        if idx < train_count:
            result[image_id] = "train"
        elif idx < train_count + val_count:
            result[image_id] = "val"
        else:
            result[image_id] = "test"
    return result


def image_challenge_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for image_id, group in df.groupby("image_id", sort=True):
        challenge = (
            str(group["challenge_primary"].mode().iloc[0])
            if "challenge_primary" in group.columns and not group["challenge_primary"].empty
            else "unknown"
        )
        rows.append(
            {
                "image_id": int(image_id),
                "file_name": str(group["file_name"].iloc[0]),
                "challenge_primary": challenge,
                "num_objects": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def select_balanced_images(
    image_table: pd.DataFrame,
    target_images: int | None,
    rebalance: bool,
) -> set[int]:
    if image_table.empty:
        return set()

    if rebalance and target_images is None:
        counts = image_table["challenge_primary"].value_counts()
        target_images = int(counts.min() * counts.size)

    if target_images is None or target_images >= len(image_table):
        return set(image_table["image_id"].astype(int).tolist())

    groups: dict[str, deque[int]] = defaultdict(deque)
    for _, row in image_table.sort_values(["challenge_primary", "image_id"]).iterrows():
        groups[str(row["challenge_primary"])].append(int(row["image_id"]))

    selected: list[int] = []
    while len(selected) < target_images and any(groups.values()):
        for challenge in sorted(groups):
            if groups[challenge] and len(selected) < target_images:
                selected.append(groups[challenge].popleft())

    return set(selected)


def validate_paths_exist(df: pd.DataFrame) -> None:
    for col in PATH_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Filtered index is missing path column: {col}")
        missing = sorted({path for path in df[col].astype(str) if not Path(path).exists()})
        if missing:
            preview = missing[:10]
            raise FileNotFoundError(f"Missing paths in {col}: {preview}")


def filter_sim_index(
    index_path: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
    exclude_bad: bool = False,
    exclude_files: Iterable[str] | None = None,
    max_objects_per_image: int = 25,
    min_objects_per_image: int = 3,
    rebalance: bool = False,
    target_images: int | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(index_path)
    audit = pd.read_csv(audit_path)
    before_images = int(df["file_name"].nunique())
    before_objects = int(len(df))

    excluded = set(str(value) for value in (exclude_files or []))
    if exclude_bad:
        if "is_bad" not in audit.columns or "file_name" not in audit.columns:
            raise ValueError("Audit CSV must contain file_name and is_bad for --exclude-bad")
        excluded.update(
            audit.loc[audit["is_bad"].map(bool_like), "file_name"].astype(str).tolist()
        )

    filtered = df[~df["file_name"].astype(str).isin(excluded)].copy()

    per_image_counts = filtered.groupby("file_name").size()
    count_excluded = per_image_counts[
        (per_image_counts < min_objects_per_image)
        | (per_image_counts > max_objects_per_image)
    ].index.astype(str)
    if len(count_excluded) > 0:
        filtered = filtered[~filtered["file_name"].astype(str).isin(set(count_excluded))].copy()

    if filtered.empty:
        raise ValueError("Filtered index is empty after exclusions.")

    selected_images = select_balanced_images(
        image_table=image_challenge_table(filtered),
        target_images=target_images,
        rebalance=rebalance,
    )
    if selected_images:
        filtered = filtered[filtered["image_id"].astype(int).isin(selected_images)].copy()

    if filtered.empty:
        raise ValueError("Filtered index is empty after target-image selection.")

    split_map = split_labels_for_images(filtered["image_id"].astype(int).unique())
    filtered["split"] = filtered["image_id"].astype(int).map(split_map)
    filtered = filtered.reset_index(drop=True)

    validate_paths_exist(filtered)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output, index=False)

    print(f"[OK] Wrote filtered index: {output}")
    print(f"Images before: {before_images}")
    print(f"Objects before: {before_objects}")
    print(f"Images after: {filtered['file_name'].nunique()}")
    print(f"Objects after: {len(filtered)}")
    print("\nCategory counts:")
    print(filtered["category_name"].value_counts() if "category_name" in filtered else "missing")
    print("\nChallenge counts:")
    print(filtered["challenge_primary"].value_counts() if "challenge_primary" in filtered else "missing")
    print("\nSplit counts:")
    print(filtered["split"].value_counts())

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a finalized COGAR-Sim index.")
    parser.add_argument("--index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index.csv")
    parser.add_argument("--audit", default="outputs/tables/dataset_audit/image_quality_audit.csv")
    parser.add_argument("--output", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index_filtered.csv")
    parser.add_argument("--exclude-bad", action="store_true")
    parser.add_argument("--exclude-files", nargs="*", default=[])
    parser.add_argument("--max-objects-per-image", type=int, default=25)
    parser.add_argument("--min-objects-per-image", type=int, default=3)
    parser.add_argument("--rebalance", action="store_true")
    parser.add_argument("--target-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filter_sim_index(
        index_path=args.index,
        audit_path=args.audit,
        output_path=args.output,
        exclude_bad=args.exclude_bad,
        exclude_files=args.exclude_files,
        max_objects_per_image=args.max_objects_per_image,
        min_objects_per_image=args.min_objects_per_image,
        rebalance=args.rebalance,
        target_images=args.target_images,
    )


if __name__ == "__main__":
    main()
