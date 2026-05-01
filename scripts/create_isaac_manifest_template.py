import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_PATH = Path("raw_isaac_exports/isaac_export_manifest_template.csv")


MANIFEST_COLUMNS = [
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
        description="Create a CSV manifest template for Isaac Sim raw exports."
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the Isaac export manifest template CSV.",
    )
    parser.add_argument(
        "--with-example",
        action="store_true",
        help="Include one example row using the current dummy simulated sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing template file.",
    )

    return parser.parse_args()


def make_empty_template() -> pd.DataFrame:
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def make_example_template() -> pd.DataFrame:
    row = {
        "image_id": "img_000001",
        "scene_id": "scene_0001",
        "frame_id": 0,
        "split": "train",
        "raw_image_path": "sim_dataset/images/train/img_000001.png",
        "raw_instance_mask_path": (
            "sim_dataset/masks/instance/train/img_000001_obj_1.png"
        ),
        "raw_semantic_mask_path": "sim_dataset/masks/semantic/train/img_000001.png",
        "category_id": 1,
        "category_name": "metal_tool",
        "object_id": 1,
        "bbox_xmin": 250,
        "bbox_ymin": 180,
        "bbox_xmax": 430,
        "bbox_ymax": 280,
        "challenge_primary": "reflective_metal",
        "challenge_secondary": "partial_occlusion",
        "is_reflective": True,
        "is_transparent": False,
        "is_occluded": True,
        "is_small_part": False,
        "is_dynamic": False,
        "camera_name": "front_table",
    }

    return pd.DataFrame([row], columns=MANIFEST_COLUMNS)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.with_example:
        df = make_example_template()
    else:
        df = make_empty_template()

    df.to_csv(output_path, index=False)

    print("Saved Isaac manifest template:", output_path)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()
