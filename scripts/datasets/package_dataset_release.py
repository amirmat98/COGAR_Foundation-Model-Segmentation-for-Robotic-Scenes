"""Package a dataset directory as a tar.gz with a SHA256 checksum."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Dataset directory to package.")
    parser.add_argument("--output-dir", default="/mnt/Info/COGAR_DATASETs/releases")
    parser.add_argument("--name", default=None, help="Release archive base name.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    release_name = args.name or dataset_root.name
    archive_path = output_dir / f"{release_name}.tar.gz"
    checksum_path = output_dir / f"{release_name}.tar.gz.sha256"

    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(dataset_root, arcname=dataset_root.name)

    checksum = sha256_file(archive_path)
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")

    print(f"archive: {archive_path}")
    print(f"sha256: {checksum_path}")


if __name__ == "__main__":
    main()

