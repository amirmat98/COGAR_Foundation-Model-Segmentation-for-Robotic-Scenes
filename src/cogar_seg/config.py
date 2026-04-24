from pathlib import Path
import yaml


def load_config(config_path: str | Path = "configs/paths.yaml") -> dict:
    """
    Load project configuration from a YAML file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_ocid_sequence_path(config: dict) -> Path:
    """
    Build the full OCID debug sequence path from the config.
    """
    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])
    return ocid_root / sequence


def get_outputs_dir(config: dict) -> Path:
    """
    Return the output directory path.
    """
    return Path(config.get("outputs_dir", "outputs"))
