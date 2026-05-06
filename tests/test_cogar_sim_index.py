import csv
import json
from pathlib import Path

import pytest

from cogar_seg.indexing.cogar_sim_index import create_cogar_sim_object_index


def write_tiny_cogar_sim_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    (rgb_dir / "000000.png").write_bytes(b"not-a-real-png")

    coco_path = tmp_path / "instances_all.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "images/000000.png"}],
                "categories": [{"id": 3, "name": "metal_part"}],
                "annotations": [
                    {
                        "id": 7,
                        "image_id": 0,
                        "category_id": 3,
                        "bbox": [10, 20, 30, 40],
                        "area": 1200,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    metadata_path = tmp_path / "frame_index.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "primary_challenge",
                "reflective",
                "transparent",
                "occlusion",
                "small_parts",
                "dynamic",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "1",
                "file_name": "000000.png",
                "primary_challenge": "reflective_metal",
                "reflective": "1",
                "transparent": "0",
                "occlusion": "0",
                "small_parts": "0",
                "dynamic": "0",
            }
        )

    return coco_path, metadata_path, rgb_dir


def test_create_cogar_sim_object_index_from_tiny_coco(tmp_path: Path) -> None:
    coco_path, metadata_path, rgb_dir = write_tiny_cogar_sim_fixture(tmp_path)
    output_csv = tmp_path / "indexes" / "cogar_sim_500_objects.csv"

    run = create_cogar_sim_object_index(
        coco_path=coco_path,
        metadata_path=metadata_path,
        rgb_dir=rgb_dir,
        output_csv=output_csv,
    )

    with output_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert run.num_images == 1
    assert run.num_rows == 1
    assert rows[0]["annotation_id"] == "7"
    assert rows[0]["category_name"] == "metal_part"
    assert rows[0]["bbox_w"] == "30.0"
    assert rows[0]["primary_challenge"] == "reflective_metal"


def test_create_cogar_sim_object_index_rejects_invalid_bbox(tmp_path: Path) -> None:
    coco_path, metadata_path, rgb_dir = write_tiny_cogar_sim_fixture(tmp_path)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    coco["annotations"][0]["bbox"] = [10, 20, 0, 40]
    coco_path.write_text(json.dumps(coco), encoding="utf-8")

    with pytest.raises(ValueError, match="positive width and height"):
        create_cogar_sim_object_index(
            coco_path=coco_path,
            metadata_path=metadata_path,
            rgb_dir=rgb_dir,
            output_csv=tmp_path / "objects.csv",
        )
