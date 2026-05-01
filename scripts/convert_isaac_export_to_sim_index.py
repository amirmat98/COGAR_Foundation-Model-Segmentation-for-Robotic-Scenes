import argparse
import shutil
from pathlib import Path

import pandas as pd

from cogar_seg.datasets.sim_robotic import (
    REQUIRED_SIM_INDEX_COLUMNS,
    validate_sim_index,
)


DEFAULT_MANIFEST_PATH = Path("raw_isaac_exports/isaac_export_manifest.csv")
DEFAULT_OUTPUT_ROOT = Path("sim_dataset")
DEFAULT_INDEX_PATH = Path("sim_dataset/annotations/sim_robotic_scenes_index.csv")


REQUIRED_MANIFEST_COLUMNS = [
    "image_id",
    "scene_id",
    "frame_id",
    "split",
    "raw_image_path",
    "raw_instance_mask_path",
    "raw_semantic_mask_path",
    "category_id",
    "category_name",
    "object_id",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "challenge_primary",
    "challenge_secondary",
    "is_reflective",
    "is_transparent",
    "is_occluded",
    "is_small_part",
    "is_dynamic",
    "camera_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a manifest-described Isaac Sim export into the standardized "
            "simulated robotic-scene benchmark index."
        )
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the raw Isaac export manifest CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for the converted sim_dataset structure.",
    )
    parser.add_argument(
        "--index-output",
        default=DEFAULT_INDEX_PATH,
        help="Output path for the converted benchmark index CSV.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Image width used for validation.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Image height used for validation.",
    )
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy raw files into sim_dataset/. Without this, only mapped paths are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite copied files and existing index output.",
    )

    return parser.parse_args()


def validate_manifest_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_MANIFEST_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")


def ensure_file_exists(path: str | Path, column_name: str) -> None:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file from column {column_name}: {file_path}")


def make_converted_paths(row: pd.Series, output_root: Path) -> dict[str, Path]:
    split = str(row["split"])
    image_id = str(row["image_id"])
    object_id = int(row["object_id"])

    image_path = output_root / "images" / split / f"{image_id}.png"
    instance_mask_path = (
        output_root
        / "masks"
        / "instance"
        / split
        / f"{image_id}_obj_{object_id:04d}.png"
    )
    semantic_mask_path = output_root / "masks" / "semantic" / split / f"{image_id}.png"

    return {
        "image_path": image_path,
        "instance_mask_path": instance_mask_path,
        "semantic_mask_path": semantic_mask_path,
    }


def copy_if_needed(src: str | Path, dst: Path, copy_files: bool, overwrite: bool) -> None:
    src_path = Path(src)

    if not copy_files:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return

    shutil.copy2(src_path, dst)


def compute_bbox_center(row: pd.Series) -> tuple[float, float]:
    point_x = (float(row["bbox_xmin"]) + float(row["bbox_xmax"])) / 2.0
    point_y = (float(row["bbox_ymin"]) + float(row["bbox_ymax"])) / 2.0

    return point_x, point_y


def convert_manifest_row(
    row: pd.Series,
    output_root: Path,
    copy_files: bool,
    overwrite: bool,
) -> dict[str, object]:
    ensure_file_exists(row["raw_image_path"], "raw_image_path")
    ensure_file_exists(row["raw_instance_mask_path"], "raw_instance_mask_path")
    ensure_file_exists(row["raw_semantic_mask_path"], "raw_semantic_mask_path")

    paths = make_converted_paths(row, output_root)

    copy_if_needed(
        src=row["raw_image_path"],
        dst=paths["image_path"],
        copy_files=copy_files,
        overwrite=overwrite,
    )
    copy_if_needed(
        src=row["raw_instance_mask_path"],
        dst=paths["instance_mask_path"],
        copy_files=copy_files,
        overwrite=overwrite,
    )
    copy_if_needed(
        src=row["raw_semantic_mask_path"],
        dst=paths["semantic_mask_path"],
        copy_files=copy_files,
        overwrite=overwrite,
    )

    point_x, point_y = compute_bbox_center(row)

    converted = {
        "image_id": str(row["image_id"]),
        "scene_id": str(row["scene_id"]),
        "frame_id": int(row["frame_id"]),
        "split": str(row["split"]),
        "image_path": str(paths["image_path"]),
        "instance_mask_path": str(paths["instance_mask_path"]),
        "semantic_mask_path": str(paths["semantic_mask_path"]),
        "category_id": int(row["category_id"]),
        "category_name": str(row["category_name"]),
        "object_id": int(row["object_id"]),
        "bbox_xmin": float(row["bbox_xmin"]),
        "bbox_ymin": float(row["bbox_ymin"]),
        "bbox_xmax": float(row["bbox_xmax"]),
        "bbox_ymax": float(row["bbox_ymax"]),
        "point_x": point_x,
        "point_y": point_y,
        "challenge_primary": str(row["challenge_primary"]),
        "challenge_secondary": str(row["challenge_secondary"]),
        "is_reflective": row["is_reflective"],
        "is_transparent": row["is_transparent"],
        "is_occluded": row["is_occluded"],
        "is_small_part": row["is_small_part"],
        "is_dynamic": row["is_dynamic"],
        "camera_name": str(row["camera_name"]),
    }

    return converted


def convert_manifest(
    manifest_path: Path,
    output_root: Path,
    index_output: Path,
    image_width: int,
    image_height: int,
    copy_files: bool,
    overwrite: bool,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    if index_output.exists() and not overwrite:
        raise FileExistsError(
            f"Index output already exists: {index_output}. "
            "Use --overwrite to replace it."
        )

    manifest_df = pd.read_csv(manifest_path)
    validate_manifest_columns(manifest_df)

    rows = [
        convert_manifest_row(
            row=row,
            output_root=output_root,
            copy_files=copy_files,
            overwrite=overwrite,
        )
        for _, row in manifest_df.iterrows()
    ]

    index_df = pd.DataFrame(rows, columns=REQUIRED_SIM_INDEX_COLUMNS)

    validate_sim_index(
        index_df,
        image_width=image_width,
        image_height=image_height,
    )

    index_output.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_output, index=False)

    return index_df


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    index_output = Path(args.index_output)

    index_df = convert_manifest(
        manifest_path=manifest_path,
        output_root=output_root,
        index_output=index_output,
        image_width=args.image_width,
        image_height=args.image_height,
        copy_files=args.copy_files,
        overwrite=args.overwrite,
    )

    print("Converted Isaac export manifest.")
    print("Manifest:", manifest_path)
    print("Output root:", output_root)
    print("Index output:", index_output)
    print("Rows:", len(index_df))
    print("Images:", index_df["image_id"].nunique())
    print("Categories:", index_df["category_name"].nunique())


if __name__ == "__main__":
    main()
