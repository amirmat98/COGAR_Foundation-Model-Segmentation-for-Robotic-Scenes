import pandas as pd
import pytest

from cogar_seg.datasets.sim_robotic import (
    REQUIRED_SIM_INDEX_COLUMNS,
    load_sim_index,
    summarize_sim_index,
    validate_sim_bounding_boxes,
    validate_sim_index,
    validate_sim_index_columns,
    validate_sim_splits,
)


def make_valid_sim_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "img_000001",
                "scene_id": "scene_0001",
                "frame_id": 0,
                "split": "train",
                "image_path": "sim_dataset/images/train/img_000001.png",
                "instance_mask_path": (
                    "sim_dataset/masks/instance/train/img_000001_obj_1.png"
                ),
                "semantic_mask_path": (
                    "sim_dataset/masks/semantic/train/img_000001.png"
                ),
                "category_id": 1,
                "category_name": "metal_tool",
                "object_id": 1,
                "bbox_xmin": 10,
                "bbox_ymin": 20,
                "bbox_xmax": 100,
                "bbox_ymax": 120,
                "point_x": 55,
                "point_y": 70,
                "challenge_primary": "reflective_metal",
                "challenge_secondary": "partial_occlusion",
                "is_reflective": True,
                "is_transparent": False,
                "is_occluded": True,
                "is_small_part": False,
                "is_dynamic": False,
                "camera_name": "front_table",
            },
            {
                "image_id": "img_000002",
                "scene_id": "scene_0002",
                "frame_id": 0,
                "split": "test",
                "image_path": "sim_dataset/images/test/img_000002.png",
                "instance_mask_path": (
                    "sim_dataset/masks/instance/test/img_000002_obj_3.png"
                ),
                "semantic_mask_path": "sim_dataset/masks/semantic/test/img_000002.png",
                "category_id": 3,
                "category_name": "glass_cup",
                "object_id": 3,
                "bbox_xmin": 200,
                "bbox_ymin": 100,
                "bbox_xmax": 300,
                "bbox_ymax": 250,
                "point_x": 250,
                "point_y": 175,
                "challenge_primary": "transparent_objects",
                "challenge_secondary": "none",
                "is_reflective": False,
                "is_transparent": True,
                "is_occluded": False,
                "is_small_part": False,
                "is_dynamic": False,
                "camera_name": "top_down",
            },
        ]
    )


def test_required_columns_match_expected_count():
    assert len(REQUIRED_SIM_INDEX_COLUMNS) == 24


def test_validate_sim_index_accepts_valid_dataframe():
    df = make_valid_sim_index()

    validate_sim_index(df, image_width=640, image_height=480)


def test_validate_sim_index_columns_rejects_missing_column():
    df = make_valid_sim_index().drop(columns=["image_path"])

    with pytest.raises(ValueError, match="Missing required simulation-index columns"):
        validate_sim_index_columns(df)


def test_validate_sim_splits_rejects_invalid_split():
    df = make_valid_sim_index()
    df.loc[0, "split"] = "debug"

    with pytest.raises(ValueError, match="Invalid split values"):
        validate_sim_splits(df)


def test_validate_sim_bounding_boxes_rejects_invalid_geometry():
    df = make_valid_sim_index()
    df.loc[0, "bbox_xmin"] = 120
    df.loc[0, "bbox_xmax"] = 100

    with pytest.raises(ValueError, match="Invalid bounding-box geometry"):
        validate_sim_bounding_boxes(df)


def test_validate_sim_bounding_boxes_rejects_out_of_bounds_box():
    df = make_valid_sim_index()
    df.loc[0, "bbox_xmax"] = 700

    with pytest.raises(ValueError, match="exceed image width"):
        validate_sim_bounding_boxes(df, image_width=640, image_height=480)


def test_load_sim_index_reads_and_validates_csv(tmp_path):
    df = make_valid_sim_index()
    csv_path = tmp_path / "sim_index.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_sim_index(csv_path, image_width=640, image_height=480)

    assert len(loaded) == 2
    assert loaded["category_name"].tolist() == ["metal_tool", "glass_cup"]


def test_summarize_sim_index_returns_expected_counts():
    df = make_valid_sim_index()

    summary = summarize_sim_index(df)

    assert summary["num_object_instances"] == 2
    assert summary["num_images"] == 2
    assert summary["num_scenes"] == 2
    assert summary["splits"] == {"test": 1, "train": 1}
    assert summary["categories"] == {"glass_cup": 1, "metal_tool": 1}


def test_validate_sim_index_allows_empty_placeholder_index():
    df = pd.DataFrame(columns=REQUIRED_SIM_INDEX_COLUMNS)

    validate_sim_index(df, image_width=640, image_height=480)
