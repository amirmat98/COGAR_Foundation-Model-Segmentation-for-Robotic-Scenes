"""Package a dataset directory for public release.

The script writes a tar.gz archive, SHA256 checksum, and small release manifest.
Keep the archive output outside the Git repository.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

DEFAULT_EXCLUDES = (
    ".DS_Store",
    "*/.DS_Store",
    "__MACOSX/*",
    "*/__MACOSX/*",
    "Thumbs.db",
    "*/Thumbs.db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", help="Dataset directory to package.")
    parser.add_argument("--output-dir", default="/mnt/Info/COGAR_DATASETs/releases")
    parser.add_argument("--name", default=None, help="Release archive base name.")
    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        choices=range(0, 10),
        metavar="0-9",
        help="gzip compression level. Use 1 for faster packaging, 9 for smaller archives.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional fnmatch pattern to exclude, relative to the dataset root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and print planned outputs only.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing archive.")
    parser.add_argument("--log-every-files", type=int, default=500)
    parser.add_argument("--log-every-seconds", type=float, default=20.0)
    return parser.parse_args()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    raise AssertionError("unreachable")


def is_excluded(relative_path: Path, patterns: list[str]) -> bool:
    text = relative_path.as_posix()
    return any(fnmatch.fnmatch(text, pattern) for pattern in patterns)


def collect_files(dataset_root: Path, patterns: list[str]) -> list[Path]:
    files = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(dataset_root)
        if is_excluded(relative_path, patterns):
            continue
        files.append(path)
    return files


def sha256_file(path: Path, log_every_seconds: float) -> str:
    digest = hashlib.sha256()
    bytes_read = 0
    last_log = time.monotonic()
    print(f"[SHA256] start {path}")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
            bytes_read += len(chunk)
            now = time.monotonic()
            if now - last_log >= log_every_seconds:
                print(f"[SHA256] hashed {format_bytes(bytes_read)}")
                last_log = now
    print(f"[SHA256] done {format_bytes(bytes_read)}")
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir = Path(args.output_dir).resolve()
    release_name = args.name or dataset_root.name
    archive_path = output_dir / f"{release_name}.tar.gz"
    checksum_path = output_dir / f"{release_name}.tar.gz.sha256"
    manifest_path = output_dir / f"{release_name}_release_manifest.json"
    temp_archive_path = output_dir / f".{archive_path.name}.{os.getpid()}.tmp"

    if archive_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Archive already exists. Use --force to overwrite: {archive_path}")

    exclude_patterns = list(DEFAULT_EXCLUDES) + list(args.exclude)
    print(f"[SCAN] dataset_root={dataset_root}")
    files = collect_files(dataset_root, exclude_patterns)
    total_bytes = sum(path.stat().st_size for path in files)
    if not files:
        raise RuntimeError(f"No files found under {dataset_root}")

    print(f"[SCAN] files={len(files)} bytes={total_bytes} ({format_bytes(total_bytes)})")
    print(f"[PLAN] archive={archive_path}")
    print(f"[PLAN] checksum={checksum_path}")
    print(f"[PLAN] manifest={manifest_path}")
    if args.dry_run:
        print("[DRY RUN] no files written")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    completed_files = 0
    completed_bytes = 0
    started_at = time.monotonic()
    last_log = started_at

    try:
        with tarfile.open(temp_archive_path, "w:gz", compresslevel=args.compression_level) as tar:
            for index, path in enumerate(files, start=1):
                relative_path = path.relative_to(dataset_root)
                arcname = Path(dataset_root.name) / relative_path
                size = path.stat().st_size
                tar.add(path, arcname=str(arcname), recursive=False)
                completed_files = index
                completed_bytes += size
                now = time.monotonic()
                should_log = (
                    index == 1
                    or index == len(files)
                    or index % max(args.log_every_files, 1) == 0
                    or now - last_log >= args.log_every_seconds
                )
                if should_log:
                    elapsed = now - started_at
                    print(
                        "[PROGRESS] "
                        f"{index}/{len(files)} files, "
                        f"{format_bytes(completed_bytes)}/{format_bytes(total_bytes)}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                    last_log = now
        temp_archive_path.replace(archive_path)
    except Exception:
        if temp_archive_path.exists():
            print(f"[ERROR] incomplete archive kept at {temp_archive_path}")
        print(f"[ERROR] stopped after {completed_files}/{len(files)} files")
        raise

    archive_bytes = archive_path.stat().st_size
    checksum = sha256_file(archive_path, args.log_every_seconds)
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")

    manifest = {
        "release_name": release_name,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "dataset_directory_name": dataset_root.name,
        "file_count": len(files),
        "source_bytes": total_bytes,
        "source_human_size": format_bytes(total_bytes),
        "archive": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_human_size": format_bytes(archive_bytes),
        "sha256": checksum,
        "checksum_file": str(checksum_path),
        "compression": "tar.gz",
        "compression_level": args.compression_level,
        "excluded_patterns": exclude_patterns,
        "upload_files": [
            archive_path.name,
            checksum_path.name,
            manifest_path.name,
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] archive: {archive_path} ({format_bytes(archive_bytes)})")
    print(f"[OK] sha256: {checksum_path}")
    print(f"[OK] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
