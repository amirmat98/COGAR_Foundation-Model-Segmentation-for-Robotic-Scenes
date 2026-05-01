import pandas as pd
import pytest

from cogar_seg.datasets.sim_robotic import (
    REQUIRED_SIM_INDEX_COLUMNS,
    summarize_sim_index,
    validate_sim_index,
)


def make_valid_sim_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": "img_000001",
                "scene_id": "scene_0001",
                "frame_id": 0,
                "split": "train",
                "image_path": "data/cogar_sim_500/rgb/train/img_000001.png",
                "instance_mask_path": (
                    "data/cogar_sim_500/instance_masks/train/img_000001_obj_0001.png"
                ),
                "semantic_mask_path": (
                    "data/cogar_sim_500/semantic_masks/train/img_000001.png"
                ),
                "category_id": 3,
                "category_name": "metal_part",
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
                "camera_name": "front",
            }
        ],
        columns=REQUIRED_SIM_INDEX_COLUMNS,
    )


def test_validate_sim_index_accepts_valid_rows() -> None:
    df = make_valid_sim_index()

    validate_sim_index(
        df,
        image_width=640,
        image_height=480,
        allowed_category_ids={1, 2, 3},
    )


def test_validate_sim_index_rejects_missing_columns() -> None:
    df = make_valid_sim_index().drop(columns=["camera_name"])

    with pytest.raises(ValueError, match="Missing required simulation-index columns"):
        validate_sim_index(df)


def test_validate_sim_index_rejects_invalid_bbox_geometry() -> None:
    df = make_valid_sim_index()
    df.loc[0, "bbox_xmax"] = 5

    with pytest.raises(ValueError, match="Invalid bounding-box geometry"):
        validate_sim_index(df)


def test_summarize_sim_index_counts_core_fields() -> None:
    summary = summarize_sim_index(make_valid_sim_index())

    assert summary["num_object_instances"] == 1
    assert summary["num_images"] == 1
    assert summary["splits"] == {"train": 1}
    assert summary["categories"] == {"metal_part": 1}
