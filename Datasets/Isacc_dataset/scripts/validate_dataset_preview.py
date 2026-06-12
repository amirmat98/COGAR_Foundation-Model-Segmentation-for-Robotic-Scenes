#!/usr/bin/env python3
"""Validate a small Isaac Sim dataset preview.

This is intentionally stricter than a simple file-count check. It catches the
most common synthetic-data failure modes: empty masks, missing depth, and saved
frames that all show the same final scene despite different manifest rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="Dataset directory containing manifest.jsonl and isaac/")
    parser.add_argument("--expected-images", type=int, default=None)
    parser.add_argument("--min-mean-rgb-diff", type=float, default=8.0)
    parser.add_argument("--require-official-robot", action="store_true")
    parser.add_argument("--require-robot-pose-variation", action="store_true")
    parser.add_argument("--require-robot-mask", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_at_least_colors(image: Image.Image, min_colors: int) -> bool:
    # Pillow returns None when there are more than maxcolors unique values.
    return image.getcolors(maxcolors=min_colors - 1) is None


def parse_color_key(key: str) -> tuple[int, int, int] | None:
    values = []
    for chunk in key.strip("()").split(","):
        chunk = chunk.strip()
        if chunk.isdigit():
            values.append(int(chunk))
    if len(values) < 3:
        return None
    return values[0], values[1], values[2]


def print_numeric_summary(name: str, values: list[float]) -> None:
    if len(values) <= 20:
        print(f"{name}:", [round(v, 4) for v in values])
        return
    array = np.asarray(values, dtype=np.float64)
    print(
        f"{name}: count={len(values)} "
        f"min={array.min():.4f} mean={array.mean():.4f} max={array.max():.4f} "
        f"first5={[round(v, 4) for v in values[:5]]} "
        f"last5={[round(v, 4) for v in values[-5:]]}"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_dir)
    isaac = root / "isaac"
    manifest = load_manifest(root / "manifest.jsonl")
    expected = args.expected_images or len(manifest)

    errors: list[str] = []
    if len(manifest) != expected:
        errors.append(f"manifest rows: expected {expected}, got {len(manifest)}")

    required_patterns = {
        "rgb": "rgb_*.png",
        "semantic": "semantic_segmentation_*.png",
        "instance": "instance_segmentation_*.png",
        "bbox": "bounding_box_2d_tight_*.npy",
        "depth": "distance_to_camera_*.npy",
    }
    for name, pattern in required_patterns.items():
        count = len(list(isaac.glob(pattern)))
        if count != expected:
            errors.append(f"{name} files: expected {expected}, got {count}")

    first_rgb: np.ndarray | None = None
    mean_diffs: list[float] = []
    robot_area_fracs: list[float] = []
    for frame_id in range(expected):
        rgb_path = isaac / f"rgb_{frame_id:04d}.png"
        semantic_path = isaac / f"semantic_segmentation_{frame_id:04d}.png"
        instance_path = isaac / f"instance_segmentation_{frame_id:04d}.png"
        depth_path = isaac / f"distance_to_camera_{frame_id:04d}.npy"
        bbox_path = isaac / f"bounding_box_2d_tight_{frame_id:04d}.npy"

        rgb = Image.open(rgb_path).convert("RGB")
        semantic = Image.open(semantic_path)
        instance = Image.open(instance_path)
        depth = np.load(depth_path)
        boxes = np.load(bbox_path, allow_pickle=True)

        rgb_array = np.asarray(rgb, dtype=np.int16)
        if first_rgb is None:
            first_rgb = rgb_array
        else:
            mean_diffs.append(float(np.abs(rgb_array - first_rgb).mean()))
        if args.require_robot_mask:
            labels_path = isaac / f"semantic_segmentation_labels_{frame_id:04d}.json"
            with labels_path.open("r", encoding="utf-8") as f:
                label_map = json.load(f)
            robot_colors = [
                color
                for key, value in label_map.items()
                if value.get("class") == "robot"
                for color in [parse_color_key(key)]
                if color is not None
            ]
            semantic_rgb = np.asarray(semantic.convert("RGB"))
            robot_area = 0
            for color in robot_colors:
                robot_area += int(np.all(semantic_rgb == np.asarray(color, dtype=np.uint8), axis=2).sum())
            robot_area_fracs.append(robot_area / float(semantic_rgb.shape[0] * semantic_rgb.shape[1]))
            if robot_area == 0:
                errors.append(f"frame {frame_id:04d}: robot class has zero semantic-mask pixels")
        if not has_at_least_colors(semantic, 4):
            errors.append(f"frame {frame_id:04d}: semantic mask has too few colors")
        if not has_at_least_colors(instance, 4):
            errors.append(f"frame {frame_id:04d}: instance mask has too few colors")
        if not np.isfinite(depth).any():
            errors.append(f"frame {frame_id:04d}: depth has no finite values")
        if len(boxes) == 0:
            errors.append(f"frame {frame_id:04d}: no bounding boxes")

    if mean_diffs:
        if max(mean_diffs) < args.min_mean_rgb_diff:
            errors.append(
                "frame diversity is too low: "
                f"max mean RGB diff from frame 0000 is {max(mean_diffs):.3f}, "
                f"threshold is {args.min_mean_rgb_diff:.3f}"
            )

    scenario_counts = Counter(row["scenario"] for row in manifest)
    robot_modes = Counter(row.get("robot_asset", {}).get("mode", "") for row in manifest)
    robot_pose_names = Counter(row.get("robot_pose", {}).get("pose_name", "") for row in manifest)
    pose_target_counts = [int(row.get("robot_asset", {}).get("pose_target_count", 0)) for row in manifest]
    label_counts = Counter()
    role_counts = Counter()
    for row in manifest:
        for obj in row.get("objects", []):
            label_counts[obj.get("label", "")] += 1
            role = obj.get("object_role", "")
            if role:
                role_counts[role] += 1

    print(f"dataset: {root}")
    print(f"manifest_rows: {len(manifest)}")
    print(f"scenarios: {dict(sorted(scenario_counts.items()))}")
    print(f"robot_modes: {dict(sorted(robot_modes.items()))}")
    if robot_pose_names:
        print(f"robot_poses: {dict(sorted(robot_pose_names.items()))}")
    print(f"labels: {dict(sorted(label_counts.items()))}")
    if role_counts:
        print(f"object_roles: {dict(sorted(role_counts.items()))}")
    if mean_diffs:
        print_numeric_summary("mean_rgb_diffs_from_frame_0000", mean_diffs)
    if robot_area_fracs:
        print_numeric_summary("robot_area_fracs", robot_area_fracs)

    if args.require_official_robot:
        if set(robot_modes) != {"official_unitree_usd"}:
            errors.append(f"official robot required, got robot modes {dict(robot_modes)}")
        if min(pose_target_counts or [0]) <= 0:
            errors.append("official robot required, but pose_target_count is zero")
    if args.require_robot_pose_variation and len(robot_pose_names) < 2:
        errors.append(f"robot pose variation required, got pose names {dict(robot_pose_names)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("validation: PASS")


if __name__ == "__main__":
    main()
