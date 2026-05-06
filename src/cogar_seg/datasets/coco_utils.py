"""Small COCO JSON helpers used by dataset conversion scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write JSON data to disk, creating the parent directory when needed."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(data, indent=indent), encoding="utf-8")


def load_categories_from_yaml(config_path: str | Path) -> list[dict[str, Any]]:
    """Load dataset categories from a YAML config in COCO category format."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return [
        {
            "id": int(cat["id"]),
            "name": cat["name"],
            "supercategory": cat.get("supercategory", "object"),
        }
        for cat in config["categories"]
    ]
