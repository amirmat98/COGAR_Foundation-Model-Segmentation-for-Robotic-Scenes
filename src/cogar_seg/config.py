"""YAML configuration loading with small local ``$ref`` support.

The project keeps task configs readable while avoiding repeated dataset,
prompt, metric, and sampling blocks.  A mapping of the form

    key:
      $ref: common.yaml#/path/to/fragment

is replaced by the referenced YAML fragment.  If other keys are present next
to ``$ref``, they override the referenced mapping.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and resolve local ``$ref`` fragments."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    data = _load_yaml_file(resolved)
    return _resolve_refs(data, resolved.parent, stack=[])


def _load_yaml_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_refs(node: Any, base_dir: Path, stack: list[tuple[Path, str]]) -> Any:
    if isinstance(node, list):
        return [_resolve_refs(item, base_dir, stack) for item in node]

    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        ref = str(node["$ref"])
        ref_path, pointer = _parse_ref(ref, base_dir)
        key = (ref_path.resolve(), pointer)
        if key in stack:
            chain = " -> ".join(f"{path}#{ptr}" for path, ptr in stack + [key])
            raise ValueError(f"Circular config reference: {chain}")

        ref_data = _load_yaml_file(ref_path)
        ref_node = _select_pointer(ref_data, pointer)
        resolved_ref = _resolve_refs(copy.deepcopy(ref_node), ref_path.parent, stack + [key])

        overrides = {k: v for k, v in node.items() if k != "$ref"}
        if not overrides:
            return resolved_ref
        if not isinstance(resolved_ref, dict):
            raise TypeError(f"Cannot override non-mapping config reference: {ref}")
        resolved_overrides = _resolve_refs(overrides, base_dir, stack)
        return _deep_merge(resolved_ref, resolved_overrides)

    return {key: _resolve_refs(value, base_dir, stack) for key, value in node.items()}


def _parse_ref(ref: str, base_dir: Path) -> tuple[Path, str]:
    path_part, separator, pointer = ref.partition("#")
    if not separator:
        pointer = ""
    ref_path = base_dir / path_part if path_part else base_dir
    return ref_path.resolve(), pointer


def _select_pointer(data: Any, pointer: str) -> Any:
    if not pointer:
        return data

    parts = pointer.lstrip("/").split("/")
    current = data
    for raw_part in parts:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
