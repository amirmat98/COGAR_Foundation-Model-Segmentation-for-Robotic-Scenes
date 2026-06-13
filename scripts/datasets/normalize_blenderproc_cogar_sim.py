"""Normalize raw BlenderProc output into the final COGAR-SimRobotics layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument(
        "--output-root",
        default="/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000",
    )
    parser.add_argument(
        "--raw-coco-dir",
        default="/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/raw_blenderproc/cogar_sim_1000_raw/coco_data",
    )
    parser.add_argument(
        "--raw-metadata",
        default="/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/metadata/frame_index_raw.csv",
    )
    parser.add_argument("--expected-images", type=int, default=1000)
    parser.add_argument("--keep-existing-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    from cogar_seg.datasets import normalize_cogar_sim

    args = parse_args()
    result = normalize_cogar_sim(
        raw_coco_dir=args.raw_coco_dir,
        raw_metadata_path=args.raw_metadata,
        output_root=args.output_root,
        config_path=args.config,
        expected_images=args.expected_images,
        clean_rgb_dir=not args.keep_existing_rgb,
    )

    print("[OK] Normalized BlenderProc dataset")
    print(f"root: {result.root}")
    print(f"rgb: {result.rgb_dir}")
    print(f"annotations: {result.annotations_path}")
    print(f"metadata: {result.metadata_path}")
    print(f"images: {result.num_images}")
    print(f"annotations_count: {result.num_annotations}")


if __name__ == "__main__":
    main()
