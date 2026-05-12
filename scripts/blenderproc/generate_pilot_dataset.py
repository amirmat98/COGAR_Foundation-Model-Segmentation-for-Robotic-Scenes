import blenderproc as bproc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed-camera BlenderProc pilot dataset."
    )
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.blenderproc_pilot import generate_pilot_dataset

    generate_pilot_dataset(config_path=args.config)


if __name__ == "__main__":
    main()
