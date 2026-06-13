import blenderproc as bproc  # noqa: F401

"""Generate the BlenderProc COGAR-SimRobotics dataset.

Run with:

    blenderproc run scripts/blenderproc/generate_cogar_sim.py \
      --config configs/blenderproc_dataset.yaml
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate COGAR-SimRobotics with BlenderProc.")
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--raw-dataset-name", default="cogar_sim_1000_raw")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.blenderproc_scene import generate_cogar_sim

    generate_cogar_sim(
        config_path=args.config,
        output_root=args.output_root,
        num_images=args.num_images,
        repo_root=REPO_ROOT,
        raw_dataset_name=args.raw_dataset_name,
        seed=args.seed,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
