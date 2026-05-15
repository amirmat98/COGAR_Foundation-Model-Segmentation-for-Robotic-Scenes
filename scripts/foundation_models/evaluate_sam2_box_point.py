#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
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


def get_bbox(row, df):
    colmap = {norm_name(c): c for c in df.columns}

    keys = ["bboxxmin", "bboxymin", "bboxxmax", "bboxymax"]
    if all(k in colmap for k in keys):
        return np.array([
            float(row[colmap["bboxxmin"]]),
            float(row[colmap["bboxymin"]]),
            float(row[colmap["bboxxmax"]]),
            float(row[colmap["bboxymax"]]),
        ], dtype=np.float32)

    keys = ["xmin", "ymin", "xmax", "ymax"]
    if all(k in colmap for k in keys):
        return np.array([
            float(row[colmap["xmin"]]),
            float(row[colmap["ymin"]]),
            float(row[colmap["xmax"]]),
            float(row[colmap["ymax"]]),
        ], dtype=np.float32)

    raise RuntimeError("No bbox columns found.")


def get_point(row, df):
    px_col = find_col(df, ["point_x", "prompt_x", "center_x", "cx"])
    py_col = find_col(df, ["point_y", "prompt_y", "center_y", "cy"])

    if px_col is not None and py_col is not None:
        return np.array([[float(row[px_col]), float(row[py_col])]], dtype=np.float32)

    x1, y1, x2, y2 = get_bbox(row, df)
    return np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)


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


def mask_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if inter == 0 else 0.0

    return float(inter / union)


def mask_boundary(mask):
    mask = mask.astype(np.uint8)

    if mask.sum() == 0:
        return mask.astype(bool)

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool) & ~eroded.astype(bool)


def boundary_f1(pred, gt, tolerance_px=2):
    pred_b = mask_boundary(pred)
    gt_b = mask_boundary(gt)

    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0

    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    pred_d = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=tolerance_px).astype(bool)
    gt_d = cv2.dilate(gt_b.astype(np.uint8), kernel, iterations=tolerance_px).astype(bool)

    precision = np.logical_and(pred_b, gt_d).sum() / max(pred_b.sum(), 1)
    recall = np.logical_and(gt_b, pred_d).sum() / max(gt_b.sum(), 1)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def pick_best_mask(masks, scores, gt_mask):
    if masks is None or len(masks) == 0:
        return np.zeros_like(gt_mask, dtype=bool), 0.0, 0.0

    best_idx = 0
    best_iou = -1.0

    for i, m in enumerate(masks):
        pred = m.astype(bool)
        if pred.shape != gt_mask.shape:
            pred = np.array(
                Image.fromarray(pred.astype(np.uint8) * 255).resize(
                    (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                )
            ) > 0

        iou = mask_iou(pred, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    best_mask = masks[best_idx].astype(bool)
    if best_mask.shape != gt_mask.shape:
        best_mask = np.array(
            Image.fromarray(best_mask.astype(np.uint8) * 255).resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
            )
        ) > 0

    score = float(scores[best_idx]) if scores is not None and len(scores) > best_idx else 0.0
    return best_mask, float(best_iou), score


def summarize(results, out_dir, model_name, prompt_type, device_name):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance = out_dir / f"{model_name.lower().replace('.', '_')}_{prompt_type}_per_instance.csv"
    results.to_csv(per_instance, index=False)

    total_time = float(results["elapsed_s"].sum())
    n = int(len(results))
    mean_fps = float(n / total_time) if total_time > 0 else 0.0

    overall = pd.DataFrame([{
        "model": model_name,
        "prompt_type": prompt_type,
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
    ap.add_argument("--prompt-type", choices=["box", "point"], required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boundary-tolerance-px", type=int, default=2)
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    sam2_repo = Path(args.sam2_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()

    sys.path.insert(0, str(sam2_repo))

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

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

    df = pd.read_csv(index_csv)
    print("CSV columns:", list(df.columns))

    if args.limit is not None:
        df = df.head(args.limit).copy()

    image_col = find_col(df, ["image_path", "rgb_path", "file_name", "filename"])
    if image_col is None:
        raise RuntimeError("No image path column found.")

    model = build_sam2(args.config, str(checkpoint), device=device_name)
    predictor = SAM2ImagePredictor(model)
    model_name = "SAM2.1-Tiny"

    rows = []

    with torch.inference_mode():
        for image_value, g in tqdm(df.groupby(image_col, sort=False), desc=f"{model_name} {args.prompt_type}"):
            image_path = resolve_path(image_value, project_root)
            if image_path is None or not image_path.exists():
                print(f"[WARN] missing image: {image_value}")
                continue

            image = np.array(Image.open(image_path).convert("RGB"))
            h, w = image.shape[:2]

            if device_name == "cuda":
                torch.cuda.synchronize()
            t_set0 = time.perf_counter()
            predictor.set_image(image)
            if device_name == "cuda":
                torch.cuda.synchronize()
            set_image_time = time.perf_counter() - t_set0

            per_obj_set_time = set_image_time / max(len(g), 1)

            for idx, row in g.iterrows():
                gt_mask = load_gt_mask(row, df, project_root, (h, w))

                if args.prompt_type == "box":
                    box = get_bbox(row, df)
                    prompt_kwargs = {"box": box, "multimask_output": True}
                else:
                    point_coords = get_point(row, df)
                    point_labels = np.array([1], dtype=np.int32)
                    prompt_kwargs = {
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                        "multimask_output": True,
                    }

                if device_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                masks, scores, logits = predictor.predict(**prompt_kwargs)

                if device_name == "cuda":
                    torch.cuda.synchronize()
                pred_time = time.perf_counter() - t0

                pred_mask, iou_val, pred_score = pick_best_mask(masks, scores, gt_mask)
                bf1 = boundary_f1(pred_mask, gt_mask, tolerance_px=args.boundary_tolerance_px)

                elapsed = per_obj_set_time + pred_time

                rows.append({
                    "model": model_name,
                    "prompt_type": args.prompt_type,
                    "split": get_split(row, df),
                    "image_path": str(image_path.relative_to(project_root)) if str(image_path).startswith(str(project_root)) else str(image_path),
                    "instance_id": get_instance_id(row, df, idx),
                    "category": get_category(row, df),
                    "challenge": get_challenge(row, df),
                    "iou": iou_val,
                    "boundary_f1": bf1,
                    "predicted_iou": pred_score,
                    "elapsed_s": elapsed,
                    "fps": 1.0 / elapsed if elapsed > 0 else 0.0,
                    "device": device_name,
                })

            if device_name == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No results generated.")

    results = pd.DataFrame(rows)
    summarize(results, out_dir, model_name, args.prompt_type, device_name)


if __name__ == "__main__":
    main()
