from pathlib import Path

from cogar_seg.paths import remap_ocid_path, resolve_ocid_sequence_path, resolve_project_path


def test_resolve_project_path_keeps_absolute_path() -> None:
    absolute = Path("/tmp/example")

    assert resolve_project_path(absolute, Path("/project")) == absolute


def test_resolve_project_path_uses_project_root() -> None:
    assert resolve_project_path("outputs/file.csv", Path("/project")) == Path(
        "/project/outputs/file.csv"
    )


def test_resolve_ocid_sequence_accepts_absolute_sequence() -> None:
    config = {
        "ocid_root": "/data/OCID-dataset",
        "ocid_debug_sequence": "/data/OCID-dataset/YCB10/table/top/mixed/seq21",
    }

    assert resolve_ocid_sequence_path(config) == Path(
        "/data/OCID-dataset/YCB10/table/top/mixed/seq21"
    )


def test_remap_ocid_path_replaces_old_root() -> None:
    old_path = "/old/location/OCID-dataset/YCB10/table/top/mixed/seq21/rgb/image.png"

    assert remap_ocid_path(old_path, "/new/OCID-dataset") == Path(
        "/new/OCID-dataset/YCB10/table/top/mixed/seq21/rgb/image.png"
    )
