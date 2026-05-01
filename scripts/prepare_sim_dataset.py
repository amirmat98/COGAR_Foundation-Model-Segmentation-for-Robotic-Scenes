import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cogar_seg.datasets.sim_robotic import (
    REQUIRED_SIM_INDEX_COLUMNS,
    validate_sim_index_columns,
)


DEFAULT_CONFIG_PATH = Path("configs/simulation_dataset.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the simulated robotic-scene dataset directory layout."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the simulation dataset YAML configuration.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated placeholder CSV/JSON files if they already exist.",
    )

    return parser.parse_args()


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Simulation config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Simulation config is not a YAML mapping: {path}")

    return data


def ensure_directories(config: dict[str, Any]) -> list[Path]:
    dataset_root = Path(config["dataset"]["root_dir"])
    outputs = config["outputs"]

    split_names = list(config["dataset"]["splits"].keys())

    dirs: list[Path] = []

    for split in split_names:
        dirs.append(dataset_root / "images" / split)
        dirs.append(dataset_root / "masks" / "instance" / split)
        dirs.append(dataset_root / "masks" / "semantic" / split)

    dirs.extend(
        [
            dataset_root / "annotations",
            dataset_root / "metadata",
            Path(outputs["benchmark_results_dir"]),
            Path(outputs["analysis_dir"]),
            Path(outputs["failure_analysis_dir"]),
        ]
    )

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    return dirs


def collect_categories(config: dict[str, Any]) -> list[dict[str, Any]]:
    foreground = config["categories"]["foreground"]
    context = config["categories"]["context"]

    categories = []

    for item in foreground:
        categories.append(
            {
                "id": int(item["id"]),
                "name": str(item["name"]),
                "supercategory": "foreground",
                "challenge_tags": list(item.get("challenge_tags", [])),
            }
        )

    for item in context:
        categories.append(
            {
                "id": int(item["id"]),
                "name": str(item["name"]),
                "supercategory": "context",
                "challenge_tags": list(item.get("challenge_tags", [])),
            }
        )

    return categories


def write_json_if_needed(path: Path, data: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Keeping existing file: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {path}")


def write_csv_if_needed(path: Path, df: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Keeping existing file: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"Saved: {path}")


def create_empty_benchmark_index(config: dict[str, Any]) -> pd.DataFrame:
    columns_from_config = config["annotations"]["required_index_columns"]

    if columns_from_config != REQUIRED_SIM_INDEX_COLUMNS:
        raise ValueError(
            "Config required_index_columns does not match code schema. "
            "Update configs/simulation_dataset.yaml or sim_robotic.py."
        )

    df = pd.DataFrame(columns=REQUIRED_SIM_INDEX_COLUMNS)
    validate_sim_index_columns(df)

    return df


def create_empty_scene_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "scene_id",
            "split",
            "scene_type",
            "challenge_primary",
            "challenge_secondary",
            "num_objects",
            "camera_name",
            "robot_context",
            "random_seed",
        ]
    )


def create_generation_summary(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    splits = dataset["splits"]

    return {
        "dataset_name": dataset["name"],
        "target_num_images": dataset["target_num_images"],
        "preferred_simulator": dataset["preferred_simulator"],
        "splits": {
            split_name: split_cfg["num_images"]
            for split_name, split_cfg in splits.items()
        },
        "foreground_categories": len(config["categories"]["foreground"]),
        "context_categories": len(config["categories"]["context"]),
        "required_index_columns": len(config["annotations"]["required_index_columns"]),
        "random_seed": config["reproducibility"]["random_seed"],
    }


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    dataset_root = Path(config["dataset"]["root_dir"])
    annotations_dir = dataset_root / "annotations"
    metadata_dir = dataset_root / "metadata"

    created_dirs = ensure_directories(config)

    categories = collect_categories(config)
    categories_path = annotations_dir / "categories.json"

    index_df = create_empty_benchmark_index(config)
    index_path = annotations_dir / "sim_robotic_scenes_index.csv"

    metadata_df = create_empty_scene_metadata()
    metadata_path = annotations_dir / "scene_metadata.csv"

    generation_summary = create_generation_summary(config)
    generation_summary_path = metadata_dir / "generation_summary.json"

    write_json_if_needed(categories_path, categories, overwrite=args.overwrite)
    write_csv_if_needed(index_path, index_df, overwrite=args.overwrite)
    write_csv_if_needed(metadata_path, metadata_df, overwrite=args.overwrite)
    write_json_if_needed(
        generation_summary_path,
        generation_summary,
        overwrite=args.overwrite,
    )

    print()
    print("Simulation dataset preparation complete.")
    print("Config:", config_path)
    print("Dataset root:", dataset_root)
    print("Created/checked directories:", len(created_dirs))
    print("Categories:", len(categories))
    print("Benchmark index:", index_path)
    print("Scene metadata:", metadata_path)
    print("Generation summary:", generation_summary_path)


if __name__ == "__main__":
    main()
