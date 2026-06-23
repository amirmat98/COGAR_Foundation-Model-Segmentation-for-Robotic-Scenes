"""Check configured dataset paths without loading heavy images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cogar_seg.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets.yaml")
    return parser.parse_args()


def path_status(path_value: str | None) -> str:
    if not path_value:
        return "not configured"
    path = Path(path_value)
    return "exists" if path.exists() else "missing"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for name, dataset in config["datasets"].items():
        print(f"[{name}]")
        for key in ["repo_default_root", "local_root"]:
            value = dataset.get(key)
            print(f"  {key}: {value} ({path_status(value)})")


if __name__ == "__main__":
    main()
