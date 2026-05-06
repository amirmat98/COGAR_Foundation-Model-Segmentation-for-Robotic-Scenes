from pathlib import Path

import yaml

import cogar_seg


def test_package_imports() -> None:
    assert cogar_seg.__version__


def test_blenderproc_category_config_loads() -> None:
    config_path = Path("configs/blenderproc_dataset.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["dataset"]["final_images"] == 500
    assert config["generation"]["render_samples"] == 32
    assert len(config["categories"]) == 10
