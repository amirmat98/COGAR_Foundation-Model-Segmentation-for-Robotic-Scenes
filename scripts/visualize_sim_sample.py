import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from cogar_seg.datasets.sim_robotic import load_sim_index


DEFAULT_INDEX_PATH = Path("sim_dataset/annotations/sim_robotic_scenes_index.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/sim_dataset_visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one simulated robotic-scene benchmark sample."
    )

    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help="Path to the simulated benchmark index CSV.",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the visualization will be saved.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
        help="Expected image width for index validation.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
        help="Expected image height for index validation.",
    )
    parser.add_argument(
        "--allow-empty-index",
        action="store_true",
        help="Allow an empty index and exit without error.",
    )

    return parser.parse_args()


def load_image(path: str | Path) -> Image.Image:
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return Image.open(image_path).convert("RGB")


def load_mask(path: str | Path) -> Image.Image:
    mask_path = Path(path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    return Image.open(mask_path)


def build_title(row: pd.Series) -> str:
    return (
        f"image_id={row['image_id']} | "
        f"object_id={row['object_id']} | "
        f"category={row['category_name']} | "
        f"split={row['split']} | "
        f"challenge={row['challenge_primary']}"
    )


def visualize_row(row: pd.Series, output_path: Path) -> None:
    image = load_image(row["image_path"])
    instance_mask = load_mask(row["instance_mask_path"])
    semantic_mask = load_mask(row["semantic_mask_path"])

    bbox_xmin = float(row["bbox_xmin"])
    bbox_ymin = float(row["bbox_ymin"])
    bbox_xmax = float(row["bbox_xmax"])
    bbox_ymax = float(row["bbox_ymax"])

    point_x = float(row["point_x"])
    point_y = float(row["point_y"])

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB + box + point")
    axes[0].axis("off")

    rect = patches.Rectangle(
        (bbox_xmin, bbox_ymin),
        bbox_xmax - bbox_xmin,
        bbox_ymax - bbox_ymin,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    axes[0].add_patch(rect)
    axes[0].scatter([point_x], [point_y], s=60, c="red", marker="x")

    axes[1].imshow(instance_mask)
    axes[1].set_title("Instance mask")
    axes[1].axis("off")

    axes[2].imshow(semantic_mask)
    axes[2].set_title("Semantic mask")
    axes[2].axis("off")

    axes[3].imshow(image)
    axes[3].imshow(instance_mask, alpha=0.45)
    axes[3].set_title("RGB + instance overlay")
    axes[3].axis("off")

    fig.suptitle(build_title(row), fontsize=11)

    metadata_text = (
        f"scene_id: {row['scene_id']}\n"
        f"frame_id: {row['frame_id']}\n"
        f"camera: {row['camera_name']}\n"
        f"secondary challenge: {row['challenge_secondary']}\n"
        f"reflective: {row['is_reflective']}\n"
        f"transparent: {row['is_transparent']}\n"
        f"occluded: {row['is_occluded']}\n"
        f"small_part: {row['is_small_part']}\n"
        f"dynamic: {row['is_dynamic']}"
    )

    fig.text(0.01, 0.01, metadata_text, fontsize=9, va="bottom", ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0.12, 1, 0.92))
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    index_path = Path(args.index)
    output_dir = Path(args.output_dir)

    df = load_sim_index(
        index_path=index_path,
        validate=True,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    if df.empty:
        message = f"Simulation index is empty: {index_path}"
        if args.allow_empty_index:
            print(message)
            print("No visualization created.")
            return
        raise ValueError(message)

    if args.row < 0 or args.row >= len(df):
        raise IndexError(f"Row {args.row} is outside valid range 0 to {len(df) - 1}")

    row = df.iloc[args.row]

    output_path = output_dir / f"sim_sample_row_{args.row:04d}.png"

    visualize_row(row, output_path)

    print("Saved visualization:", output_path)


if __name__ == "__main__":
    main()
