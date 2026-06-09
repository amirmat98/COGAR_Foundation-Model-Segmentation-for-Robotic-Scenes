#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cogar_seg.metrics import compute_boundary_f1, compute_iou
from PIL import Image
from tqdm import tqdm


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates):
    norm_to_col = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def truthy(v):
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) != 0.0
    return str(v).strip().lower() in {"1", "true", "yes", "y", "present"}


def resolve_path(value, project_root):
    raw = os.path.expanduser(str(value).strip())
    p = Path(raw)
    if p.is_absolute():
        return p

    candidates = [
        project_root / p,
        project_root / "data" / "cogar_sim_500_final" / p,
        project_root / "data" / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def load_gt_mask(row, df, project_root, image_hw):
    mask_col = find_col(df, [
        "binary_mask_path",
        "mask_path",
        "gt_mask_path",
        "segmentation_mask_path",
        "mask_file",
    ])
    if mask_col is None:
        raise RuntimeError("No binary mask column found.")

    p = resolve_path(row[mask_col], project_root)
    if p is None or not p.exists():
        raise FileNotFoundError(f"Missing mask: {p}")

    h, w = image_hw
    m = np.array(Image.open(p).convert("L")) > 0

    if m.shape[:2] != (h, w):
        m = np.array(
            Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        ) > 0

    return m


def get_category(row, df):
    col = find_col(df, ["category_name", "category", "class_name", "class", "label"])
    return str(row[col]) if col is not None else "unknown"


def get_split(row, df):
    col = find_col(df, ["split"])
    return str(row[col]) if col is not None else "all"


def get_instance_id(row, df, fallback):
    col = find_col(df, ["object_id", "instance_id", "annotation_id", "id"])
    return str(row[col]) if col is not None else str(fallback)


def get_challenge(row, df):
    found = []

    primary_col = find_col(df, ["challenge_primary", "primary_challenge", "challenge"])
    secondary_col = find_col(df, ["challenge_secondary", "secondary_challenge"])

    for col in [primary_col, secondary_col]:
        if col is not None and not pd.isna(row[col]):
            value = str(row[col]).strip()
            if value and value.lower() not in {"none", "nan", "null"}:
                found.append(value)

    flag_map = {
        "is_reflective": "reflective_metal",
        "is_transparent": "transparent_glass",
        "is_occluded": "partial_occlusion",
        "is_small_part": "small_parts",
        "is_dynamic": "dynamic_scene",
    }

    for flag_col, challenge_name in flag_map.items():
        col = find_col(df, [flag_col])
        if col is not None and truthy(row[col]):
            found.append(challenge_name)

    deduped = []
    for x in found:
        if x not in deduped:
            deduped.append(x)

    return ";".join(deduped) if deduped else "none"


def normalize_generated_masks(mask_records, gt_shape):
    out = []

    for rec in mask_records:
        if isinstance(rec, dict):
            seg = rec.get("segmentation", None)
            score = rec.get("predicted_iou", rec.get("stability_score", 0.0))
            area = rec.get("area", 0)
        else:
            seg = rec
            score = 0.0
            area = 0

        if seg is None:
            continue

        m = np.asarray(seg).astype(bool)

        if m.shape != gt_shape:
            m = np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (gt_shape[1], gt_shape[0]), Image.NEAREST
                )
            ) > 0

        out.append((m, float(score), int(area) if area is not None else int(m.sum())))

    return out


def best_mask_by_iou(candidates, gt_mask):
    if not candidates:
        return np.zeros_like(gt_mask, dtype=bool), 0.0, 0.0

    best_mask = None
    best_iou = -1.0
    best_score = 0.0

    for m, score, _area in candidates:
        iou = compute_iou(m, gt_mask, empty_value=1.0)
        if iou > best_iou:
            best_iou = iou
            best_mask = m
            best_score = score

    if best_mask is None:
        return np.zeros_like(gt_mask, dtype=bool), 0.0, 0.0

    return best_mask, float(best_iou), float(best_score)


def summarize(results, out_dir, device_name):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance = out_dir / "sam2_1-tiny_auto_per_instance.csv"
    results.to_csv(per_instance, index=False)

    total_time = float(results["elapsed_s"].sum())
    n = int(len(results))
    mean_fps = float(n / total_time) if total_time > 0 else 0.0

    overall = pd.DataFrame([{
        "model": "SAM2.1-Tiny",
        "prompt_type": "auto",
        "num_objects": n,
        "mean_iou": results["iou"].mean(),
        "median_iou": results["iou"].median(),
        "mean_boundary_f1": results["boundary_f1"].mean(),
        "iou_ge_90": (results["iou"] >= 0.90).mean(),
        "iou_ge_75": (results["iou"] >= 0.75).mean(),
        "iou_ge_50": (results["iou"] >= 0.50).mean(),
        "iou_lt_10": (results["iou"] < 0.10).mean(),
        "mean_predicted_iou": results["predicted_iou"].mean(),
        "mean_fps": mean_fps,
        "total_model_time_s": total_time,
        "device": device_name,
    }])

    overall.to_csv(out_dir / "overall_summary.csv", index=False)

    cat = (
        results.groupby("category", dropna=False)
        .agg(
            count=("iou", "size"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_boundary_f1=("boundary_f1", "mean"),
            iou_ge_75=("iou", lambda s: float((s >= 0.75).mean())),
            iou_lt_10=("iou", lambda s: float((s < 0.10).mean())),
        )
        .reset_index()
        .sort_values("mean_iou")
    )
    cat.to_csv(out_dir / "mean_iou_by_category.csv", index=False)

    expanded = []
    for _, r in results.iterrows():
        for ch in str(r["challenge"]).split(";"):
            rr = r.copy()
            rr["challenge"] = ch
            expanded.append(rr)

    exp = pd.DataFrame(expanded)
    chal = (
        exp.groupby("challenge", dropna=False)
        .agg(
            count=("iou", "size"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_boundary_f1=("boundary_f1", "mean"),
            iou_ge_75=("iou", lambda s: float((s >= 0.75).mean())),
            iou_lt_10=("iou", lambda s: float((s < 0.10).mean())),
        )
        .reset_index()
        .sort_values("mean_iou")
    )
    chal.to_csv(out_dir / "mean_iou_by_challenge.csv", index=False)

    print()
    print("Saved:")
    print(" ", per_instance)
    print(" ", out_dir / "overall_summary.csv")
    print(" ", out_dir / "mean_iou_by_category.csv")
    print(" ", out_dir / "mean_iou_by_challenge.csv")
    print()
    print(overall.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--sam2-repo", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.80)
    ap.add_argument("--stability-score-thresh", type=float, default=0.90)
    ap.add_argument("--crop-n-layers", type=int, default=0)
    ap.add_argument("--min-mask-region-area", type=int, default=0)
    ap.add_argument("--boundary-tolerance-px", type=int, default=2)
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    sam2_repo = Path(args.sam2_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    sys.path.insert(0, str(sam2_repo))

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    index_csv = Path(args.index_csv)
    if not index_csv.is_absolute():
        index_csv = project_root / index_csv

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"

    print("Device:", device_name)
    print("SAM2 repo:", sam2_repo)
    print("Checkpoint:", checkpoint)
    print("Config:", args.config)
    print("Index CSV:", index_csv)
    print("points_per_side:", args.points_per_side)

    df = pd.read_csv(index_csv)
    print("CSV columns:", list(df.columns))

    image_col = find_col(df, ["image_path", "rgb_path", "file_name", "filename"])
    if image_col is None:
        raise RuntimeError("No image path column found.")

    grouped_items = list(df.groupby(image_col, sort=False))
    if args.limit_images is not None:
        grouped_items = grouped_items[:args.limit_images]

    model = build_sam2(args.config, str(checkpoint), device=device_name)

    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )

    rows = []

    with torch.inference_mode():
        for image_value, g in tqdm(grouped_items, desc="SAM2.1-Tiny auto"):
            image_path = resolve_path(image_value, project_root)
            if image_path is None or not image_path.exists():
                print(f"[WARN] missing image: {image_value}")
                continue

            image = np.array(Image.open(image_path).convert("RGB"))
            h, w = image.shape[:2]

            if device_name == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            mask_records = generator.generate(image)

            if device_name == "cuda":
                torch.cuda.synchronize()
            elapsed_image = time.perf_counter() - t0

            first_gt_shape = (h, w)
            candidates = normalize_generated_masks(mask_records, first_gt_shape)
            per_obj_time = elapsed_image / max(len(g), 1)

            for idx, row in g.iterrows():
                gt_mask = load_gt_mask(row, df, project_root, (h, w))

                pred_mask, iou_val, pred_score = best_mask_by_iou(candidates, gt_mask)
                bf1 = compute_boundary_f1(
                    pred_mask, gt_mask, bound_thresh=args.boundary_tolerance_px
                )

                rows.append({
                    "model": "SAM2.1-Tiny",
                    "prompt_type": "auto",
                    "split": get_split(row, df),
                    "image_path": str(image_path.relative_to(project_root)) if str(image_path).startswith(str(project_root)) else str(image_path),
                    "instance_id": get_instance_id(row, df, idx),
                    "category": get_category(row, df),
                    "challenge": get_challenge(row, df),
                    "iou": iou_val,
                    "boundary_f1": bf1,
                    "predicted_iou": pred_score,
                    "elapsed_s": per_obj_time,
                    "fps": 1.0 / per_obj_time if per_obj_time > 0 else 0.0,
                    "device": device_name,
                    "num_generated_masks": len(candidates),
                })

            if device_name == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No results generated.")

    results = pd.DataFrame(rows)
    summarize(results, out_dir, device_name)


if __name__ == "__main__":
    main()
