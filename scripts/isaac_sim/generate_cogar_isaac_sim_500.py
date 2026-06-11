#!/usr/bin/env python3
"""Generate the complete COGAR-IsaacSimRobotics-500 dataset.

Run this inside the Isaac Sim Python environment, for example:

    /isaac-sim/python.sh scripts/isaac_sim/generate_cogar_isaac_sim_500.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate COGAR-IsaacSimRobotics-500 with Isaac Sim Replicator."
    )
    parser.add_argument("--config", default="configs/isaac_sim_dataset.yaml")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--raw-dataset-name", default="final_500")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--rt-subframes", type=int, default=None)
    parser.add_argument("--renderer", default=None)
    parser.add_argument(
        "--writer-mode",
        choices=("full", "seg", "rgb"),
        default=None,
        help="Replicator writer outputs: full=rgb+segmentation+bboxes, seg=rgb+segmentation, rgb=rgb only.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument(
        "--skip-writer-wait",
        action="store_true",
        help="Skip Replicator wait_until_complete; useful when weak GPUs hang during writer finalization.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help="Optional cap on scene objects for weak GPU smoke tests.",
    )
    parser.add_argument(
        "--disable-materials",
        action="store_true",
        help="Skip optional material hints for weak GPU smoke tests.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.isaac_sim_scene import generate_cogar_isaac_sim_500

    headless = None
    if args.headless:
        headless = True
    if args.show_window:
        headless = False

    generate_cogar_isaac_sim_500(
        config_path=args.config,
        num_frames=args.num_images,
        output_dir=args.output_dir,
        raw_dataset_name=args.raw_dataset_name,
        seed=args.seed,
        width=args.width,
        height=args.height,
        rt_subframes=args.rt_subframes,
        renderer=args.renderer,
        writer_mode=args.writer_mode,
        headless=headless,
        clean=not args.no_clean,
        skip_writer_wait=args.skip_writer_wait,
        max_objects=args.max_objects,
        disable_materials=args.disable_materials,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
