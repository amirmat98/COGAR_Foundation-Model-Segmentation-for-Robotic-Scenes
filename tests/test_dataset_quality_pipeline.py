from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

from cogar_seg.cv_compat import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


finalize = load_script("scripts/dataset/finalize_cogar_sim_index.py", "finalize_cogar_sim_index")
audit = load_script("scripts/dataset/audit_sim_dataset.py", "audit_sim_dataset")
filter_script = load_script("scripts/dataset/filter_sim_index.py", "filter_sim_index")
validate = load_script("scripts/dataset/validate_sim_index.py", "validate_sim_index")


def write_image(path: Path, value: int = 128, textured: bool = False) -> None:
    if textured:
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[:, :6] = 40
        image[:, 6:] = 220
    else:
        image = np.full((12, 12, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def write_mask(path: Path) -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 255
    assert cv2.imwrite(str(path), mask)


def valid_index_rows(tmp_path: Path, num_images: int = 3, objects_per_image: int = 3) -> list[dict]:
    rows = []
    categories = ["metal_part", "glass_object", "connector"]
    challenges = ["reflective_metal", "transparent_glass", "small_parts"]
    for image_idx in range(num_images):
        image_path = tmp_path / f"{image_idx:06d}.png"
        write_image(image_path, textured=True)
        for obj_idx in range(objects_per_image):
            mask_path = tmp_path / f"{image_idx:06d}_{obj_idx}.png"
            write_mask(mask_path)
            rows.append(
                {
                    "image_id": image_idx + 1,
                    "file_name": image_path.name,
                    "scene_id": f"scene_{image_idx:03d}",
                    "frame_id": image_idx + 1,
                    "split": "train",
                    "image_path": str(image_path),
                    "binary_mask_path": str(mask_path),
                    "instance_mask_path": str(mask_path),
                    "semantic_mask_path": "",
                    "category_id": obj_idx + 3,
                    "category_name": categories[obj_idx % len(categories)],
                    "object_id": image_idx * 10 + obj_idx,
                    "bbox_xmin": 3,
                    "bbox_ymin": 2,
                    "bbox_xmax": 9,
                    "bbox_ymax": 8,
                    "point_x": 5,
                    "point_y": 5,
                    "challenge_primary": challenges[image_idx % len(challenges)],
                    "challenge_secondary": "",
                    "is_reflective": categories[obj_idx % len(categories)] == "metal_part",
                    "is_transparent": categories[obj_idx % len(categories)] == "glass_object",
                    "is_occluded": False,
                    "is_small_part": categories[obj_idx % len(categories)] == "connector",
                    "is_dynamic": False,
                    "area": 36.0,
                }
            )
    return rows


def test_finalize_helpers_convert_bbox_flags_and_splits() -> None:
    bbox = finalize.convert_bbox_xywh_to_xyxy(
        pd.Series([10]),
        pd.Series([20]),
        pd.Series([30]),
        pd.Series([40]),
    )
    assert bbox.iloc[0].to_dict() == {
        "bbox_xmin": 10.0,
        "bbox_ymin": 20.0,
        "bbox_xmax": 40.0,
        "bbox_ymax": 60.0,
    }

    flags = finalize.object_flags_for_category("tool", "dynamic_scene", "0")
    assert flags["is_reflective"] is True
    assert flags["is_dynamic"] is True
    assert flags["is_transparent"] is False

    split_map = finalize.split_labels_for_images([3, 1, 2, 4, 5])
    assert split_map[1] == "train"
    assert set(split_map.values()) == {"train", "val", "test"}


def test_audit_classifies_bad_images_and_writes_counts(tmp_path: Path) -> None:
    dark_image = tmp_path / "000000.png"
    good_image = tmp_path / "000001.png"
    write_image(dark_image, value=0)
    write_image(good_image, textured=True)

    rows = []
    for idx, image_path in enumerate([dark_image, good_image]):
        for category in ["metal_part", "tool", "connector"]:
            rows.append(
                {
                    "file_name": image_path.name,
                    "image_path": str(image_path),
                    "category_name": category,
                    "challenge_primary": "reflective_metal",
                    "area": 36.0,
                    "is_reflective": category in {"metal_part", "tool"},
                    "is_transparent": False,
                    "is_small_part": category == "connector",
                    "is_occluded": False,
                    "is_dynamic": False,
                    "object_id": idx,
                }
            )
    index_path = tmp_path / "index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)

    output_dir = tmp_path / "audit"
    summary = audit.audit_sim_dataset(index_path, output_dir)

    image_audit = pd.read_csv(output_dir / "image_quality_audit.csv")
    dark_reasons = image_audit.loc[image_audit["file_name"] == "000000.png", "bad_reasons"].iloc[0]
    assert "dark" in dark_reasons
    assert summary["bad_images_count"] >= 1

    category_counts = pd.read_csv(output_dir / "category_counts.csv")
    assert set(category_counts["category_name"]) == {"metal_part", "tool", "connector"}
    assert (output_dir / "challenge_counts.csv").exists()


def test_filter_excludes_bad_files_recomputes_splits_and_preserves_columns(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    audit_path = tmp_path / "audit.csv"
    output_path = tmp_path / "filtered.csv"

    df = pd.DataFrame(valid_index_rows(tmp_path, num_images=4, objects_per_image=3))
    df.to_csv(index_path, index=False)
    pd.DataFrame(
        [
            {"file_name": "000000.png", "is_bad": True},
            {"file_name": "000001.png", "is_bad": False},
            {"file_name": "000002.png", "is_bad": False},
            {"file_name": "000003.png", "is_bad": False},
        ]
    ).to_csv(audit_path, index=False)

    filtered = filter_script.filter_sim_index(
        index_path=index_path,
        audit_path=audit_path,
        output_path=output_path,
        exclude_bad=True,
    )

    assert "000000.png" not in set(filtered["file_name"])
    assert set(validate.REQUIRED_COLUMNS).issubset(filtered.columns)
    assert set(filtered["split"]) == {"train", "val", "test"}
    assert output_path.exists()


def test_validate_catches_missing_path_invalid_bbox_and_point_outside_mask(tmp_path: Path) -> None:
    df = pd.DataFrame(valid_index_rows(tmp_path, num_images=1, objects_per_image=1))

    missing_path = df.copy()
    missing_path.loc[0, "image_path"] = str(tmp_path / "missing.png")
    with pytest.raises(FileNotFoundError, match="Missing image"):
        validate.validate_sim_index_dataframe(missing_path)

    invalid_bbox = df.copy()
    invalid_bbox.loc[0, "bbox_xmax"] = 2
    with pytest.raises(ValueError, match="Invalid bbox x"):
        validate.validate_sim_index_dataframe(invalid_bbox)

    point_outside = df.copy()
    point_outside.loc[0, "point_x"] = 10
    point_outside.loc[0, "point_y"] = 10
    with pytest.raises(ValueError, match="Point is not inside mask"):
        validate.validate_sim_index_dataframe(point_outside)
