"""Path resolution helpers for project-relative and dataset paths."""

from pathlib import Path
from typing import Any


def resolve_project_path(path_value: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    path = Path(path_value)

    if path.is_absolute():
        return path

    root = Path.cwd() if project_root is None else Path(project_root)
    return root / path


def resolve_ocid_sequence_path(config: dict[str, Any]) -> Path:
    """Resolve the OCID debug sequence from ``configs/paths.yaml``."""
    sequence = Path(config["ocid_debug_sequence"])

    if sequence.is_absolute():
        return sequence

    return Path(config["ocid_root"]) / sequence


def remap_ocid_path(path_value: str | Path, ocid_root: str | Path) -> Path:
    """
    Remap an old absolute OCID path to the configured OCID root.

    Existing paths are returned unchanged. If the stored path contains an
    ``OCID-dataset`` component, everything below that component is attached to
    the current OCID root. This preserves compatibility with existing CSVs.
    """
    path = Path(path_value)

    if path.exists():
        return path

    parts = path.parts

    if "OCID-dataset" not in parts:
        return path

    ocid_idx = parts.index("OCID-dataset")
    relative_inside_ocid = Path(*parts[ocid_idx + 1 :])
    return Path(ocid_root) / relative_inside_ocid


def default_results_csv(project_root: str | Path, name: str) -> Path:
    """Return a default CSV path under ``outputs/indexes``."""
    return Path(project_root) / "outputs" / "indexes" / name
