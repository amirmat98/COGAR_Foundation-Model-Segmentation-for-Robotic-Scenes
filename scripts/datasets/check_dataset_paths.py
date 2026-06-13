"""Check configured dataset paths without loading heavy images."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


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
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for name, dataset in config["datasets"].items():
        print(f"[{name}]")
        for key in ["repo_default_root", "local_root"]:
            value = dataset.get(key)
            print(f"  {key}: {value} ({path_status(value)})")


if __name__ == "__main__":
    main()

