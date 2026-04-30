import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path = "configs/paths.yaml") -> dict[str, Any]:
    """Load project configuration from a YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    return config


def get_ocid_sequence_path(config: dict[str, Any]) -> Path:
    """Return the configured OCID debug sequence path."""
    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])

    if sequence.is_absolute():
        return sequence

    return ocid_root / sequence


def get_outputs_dir(config: dict[str, Any]) -> Path:
    """Return the output directory path."""
    return Path(config.get("outputs_dir", "outputs"))
