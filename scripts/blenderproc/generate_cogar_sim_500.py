import blenderproc as bproc
import sys
from pathlib import Path

# BlenderProc entry scripts must be run with:
# blenderproc run scripts/blenderproc/generate_cogar_sim_500.py ...
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate randomized BlenderProc scenes for COGAR-SimRobotics-500."
    )
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument("--num-images", dest="requested_image_count", type=int, default=None)
    parser.add_argument("--raw-dataset-name", default="pilot_v2_ocid_like")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.blenderproc_scene import generate_cogar_sim_500

    generate_cogar_sim_500(
        config_path=args.config,
        num_images=args.requested_image_count,
        raw_dataset_name=args.raw_dataset_name,
        seed=args.seed,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
