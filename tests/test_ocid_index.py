from pathlib import Path
import csv

import numpy as np
from PIL import Image
import pytest

from cogar_seg.datasets.ocid import (
    compute_object_properties,
    create_full_image_index,
    discover_ocid_sequences,
    filter_object_index,
    make_binary_mask_filename,
    parse_sequence_metadata,
)


def test_compute_object_properties() -> None:
    label = np.array(
        [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
        ]
    )

    props = compute_object_properties(label, object_id=3)

    assert props == {
        "area": 3,
        "bbox_xmin": 1,
        "bbox_ymin": 1,
        "bbox_xmax": 2,
        "bbox_ymax": 2,
        "point_x": 1,
        "point_y": 1,
    }


def test_make_binary_mask_filename() -> None:
    assert (
        make_binary_mask_filename(12, "result_2018-08-24-13-13-32.png", 3)
        == "00012_result_2018-08-24-13-13-32_obj3.png"
    )


def test_discover_ocid_sequences_and_full_image_index(tmp_path: Path) -> None:
    seq = tmp_path / "OCID-dataset" / "YCB10" / "table" / "top" / "mixed" / "seq21"
    rgb_dir = seq / "rgb"
    label_dir = seq / "label"
    rgb_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    label = np.array([[0, 1], [0, 2]], dtype=np.uint8)
    Image.fromarray(image).save(rgb_dir / "frame.png")
    Image.fromarray(label).save(label_dir / "frame.png")

    root = tmp_path / "OCID-dataset"
    sequences = discover_ocid_sequences(root)
    assert sequences == [seq]

    metadata = parse_sequence_metadata(seq, root)
    assert metadata["object_set"] == "YCB10"
    assert metadata["surface"] == "table"
    assert metadata["camera_view"] == "top"
    assert metadata["scene_type"] == "mixed"
    assert metadata["sequence_id"] == "seq21"

    output_csv = tmp_path / "images.csv"
    count = create_full_image_index(root, output_csv)

    with output_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    assert count == 1
    assert rows[0]["sequence"] == "YCB10/table/top/mixed/seq21"
    assert rows[0]["file_name"] == "frame.png"


def test_create_full_image_index_progress_output(tmp_path: Path, capsys) -> None:
    seq = tmp_path / "OCID-dataset" / "YCB10" / "table" / "top" / "mixed" / "seq21"
    rgb_dir = seq / "rgb"
    label_dir = seq / "label"
    rgb_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    label = np.array([[0, 1], [0, 2]], dtype=np.uint8)
    Image.fromarray(image).save(rgb_dir / "frame.png")
    Image.fromarray(label).save(label_dir / "frame.png")

    count = create_full_image_index(
        tmp_path / "OCID-dataset",
        tmp_path / "images.csv",
        progress=True,
        progress_every=1,
    )

    captured = capsys.readouterr()
    assert count == 1
    assert "[OCID] Found 1 sequence directories" in captured.out
    assert "[OCID] Wrote image index:" in captured.out


def test_create_full_image_index_strict_missing_label(tmp_path: Path) -> None:
    seq = tmp_path / "OCID-dataset" / "YCB10" / "table" / "top" / "mixed" / "seq21"
    rgb_dir = seq / "rgb"
    label_dir = seq / "label"
    rgb_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    Image.fromarray(image).save(rgb_dir / "frame.png")

    with pytest.raises(FileNotFoundError):
        create_full_image_index(
            tmp_path / "OCID-dataset",
            tmp_path / "images.csv",
            strict=True,
        )


def test_filter_object_index_keeps_valid_rows(tmp_path: Path) -> None:
    input_csv = tmp_path / "objects.csv"
    output_csv = tmp_path / "filtered.csv"

    fieldnames = [
        "image_path",
        "label_path",
        "sequence",
        "file_name",
        "object_id",
        "area",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "point_x",
        "point_y",
    ]
    rows = [
        {
            "image_path": "rgb.png",
            "label_path": "label.png",
            "sequence": "seq",
            "file_name": "rgb.png",
            "object_id": "1",
            "area": "25",
            "bbox_xmin": "0",
            "bbox_ymin": "0",
            "bbox_xmax": "4",
            "bbox_ymax": "4",
            "point_x": "2",
            "point_y": "2",
        },
        {
            "image_path": "rgb.png",
            "label_path": "label.png",
            "sequence": "seq",
            "file_name": "rgb.png",
            "object_id": "2",
            "area": "1",
            "bbox_xmin": "0",
            "bbox_ymin": "0",
            "bbox_xmax": "0",
            "bbox_ymax": "0",
            "point_x": "0",
            "point_y": "0",
        },
    ]

    with input_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    count = filter_object_index(
        input_csv=input_csv,
        output_csv=output_csv,
        min_area=10,
        max_area_ratio=1.0,
        max_bbox_area_ratio=1.0,
        image_width=10,
        image_height=10,
    )

    with output_csv.open("r", newline="") as f:
        filtered_rows = list(csv.DictReader(f))

    assert count == 1
    assert filtered_rows[0]["object_id"] == "1"
    assert filtered_rows[0]["area_ratio"] == "0.250000"
