"""Validate local Markdown image links used by the final reports."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 5 * 1024 * 1024
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def resolve_image_path(link: str) -> Path | None:
    clean = link.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean or clean.startswith(("http://", "https://")):
        return None
    if clean.startswith(("/home/", "/Users/", "C:\\")):
        return Path(clean)
    if clean.startswith("/"):
        return ROOT / clean.lstrip("/")
    return None


def main() -> int:
    problems: list[str] = []
    for md_path in markdown_files():
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        for match in IMAGE_RE.finditer(text):
            link = match.group(1).strip()
            path = resolve_image_path(link)
            if path is None:
                continue
            if link.startswith(("/home/", "/Users/")) or re.match(r"^[A-Za-z]:\\", link):
                problems.append(f"{md_path.relative_to(ROOT)}: absolute local image path: {link}")
                continue
            if not path.exists():
                problems.append(f"{md_path.relative_to(ROOT)}: missing image: {link}")
                continue
            if path.stat().st_size > MAX_BYTES:
                size_mb = path.stat().st_size / (1024 * 1024)
                problems.append(f"{md_path.relative_to(ROOT)}: oversized image {link} ({size_mb:.2f} MB)")

    if problems:
        print("Markdown image validation failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Markdown image validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
