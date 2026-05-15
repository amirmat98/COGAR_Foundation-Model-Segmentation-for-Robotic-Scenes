#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


KNOWN_CHALLENGES = [
    "small_parts",
    "partial_occlusion",
    "dynamic_scene",
    "reflective_metal",
    "transparent_glass",
]


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df, candidates):
    norm_to_col = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        k = norm_name(cand)
        if k in norm_to_col:
            return norm_to_col[k]
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
    if value is None or pd.isna(value):
        return None
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


def parse_number_list(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            return [float(x) for x in nums] if nums else None


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
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
    return m


def get_bbox(row, df):
    colmap = {norm_name(c): c for c in df.columns}
    names = ["bboxxmin", "bboxymin", "bboxxmax", "bboxymax"]
    if all(n in colmap for n in names):
        return [
            float(row[colmap["bboxxmin"]]),
            float(row[colmap["bboxymin"]]),
            float(row[colmap["bboxxmax"]]),
            float(row[colmap["bboxymax"]]),
        ]

    names = ["xmin", "ymin", "xmax", "ymax"]
    if all(n in colmap for n in names):
        return [
            float(row[colmap["xmin"]]),
            float(row[colmap["ymin"]]),
            float(row[colmap["xmax"]]),
            float(row[colmap["ymax"]]),
        ]

    raise RuntimeError("No bbox columns found.")


def get_point(row, df):
    px_col = find_col(df, ["point_x", "prompt_x", "center_x", "cx"])
    py_col = find_col(df, ["point_y", "prompt_y", "center_y", "cy"])
    if px_col is not None and py_col is not None:
        return [float(row[px_col]), float(row[py_col])]

    x1, y1, x2, y2 = get_bbox(row, df)
    return [(x1 + x2) / 2, (y1 + y2) / 2]


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


def extract_masks_from_result(result):
    if result is None:
        return []

    if hasattr(result, "masks") and result.masks is not None:
        data = result.masks.data
        if hasattr(data, "detach"):
            arr = data.detach().cpu().numpy()
        else:
            arr = np.array(data)
        return [(m > 0.5) for m in arr]

    return []


def best_mask_by_iou(candidate_masks, gt_mask):
    if not candidate_masks:
        return None, 0.0

    best = None
    best_iou = -1.0
    for m in candidate_masks:
        if m.shape[:2] != gt_mask.shape[:2]:
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
            )) > 0
        iou = mask_iou(m, gt_mask)
        if iou > best_iou:
            best_iou = iou
            best = m

    return best, float(best_iou)


def summarize(results, out_dir, model_name, prompt_type, device_name):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance = out_dir / f"{model_name.lower().replace('-', '_')}_{prompt_type}_per_instance.csv"
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
    ap.add_argument("--model", default="FastSAM-s.pt")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt-type", choices=["point", "auto"], required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--boundary-tolerance-px", type=int, default=2)
    args = ap.parse_args()

    from ultralytics import FastSAM

    project_root = Path(args.project_root).expanduser().resolve()
    index_csv = Path(args.index_csv)
    if not index_csv.is_absolute():
        index_csv = project_root / index_csv

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    df = pd.read_csv(index_csv)
    print("CSV columns:", list(df.columns))

    if args.limit is not None:
        df = df.head(args.limit).copy()

    image_col = find_col(df, ["image_path", "rgb_path", "file_name", "filename"])
    if image_col is None:
        raise RuntimeError("No image path column found.")

    model = FastSAM(args.model)
    model_name = "FastSAM-S" if "s" in Path(args.model).stem.lower() else "FastSAM"

    rows = []

    for image_value, g in tqdm(df.groupby(image_col, sort=False), desc=f"{model_name} {args.prompt_type}"):
        image_path = resolve_path(image_value, project_root)
        if image_path is None or not image_path.exists():
            print(f"[WARN] missing image: {image_value}")
            continue

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # FastSAM first stage: generate candidate masks.
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = model(
            str(image_path),
            device=args.device,
            retina_masks=True,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_image = time.perf_counter() - t0

        if len(results) == 0:
            candidate_masks = []
        else:
            candidate_masks = extract_masks_from_result(results[0])

        per_obj_time = elapsed_image / max(len(g), 1)

        for idx, row in g.iterrows():
            gt_mask = load_gt_mask(row, df, project_root, (h, w))

            if args.prompt_type == "auto":
                pred_mask, iou_val = best_mask_by_iou(candidate_masks, gt_mask)
            else:
                # Approximate point-prompt behavior by selecting the generated mask that contains
                # the positive prompt point. If multiple masks contain it, select the one with best
                # IoU for benchmark scoring.
                px, py = get_point(row, df)
                px_i = int(np.clip(round(px), 0, w - 1))
                py_i = int(np.clip(round(py), 0, h - 1))

                point_candidates = []
                for m in candidate_masks:
                    mm = m
                    if mm.shape[:2] != gt_mask.shape[:2]:
                        mm = np.array(Image.fromarray(mm.astype(np.uint8) * 255).resize(
                            (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                        )) > 0
                    if bool(mm[py_i, px_i]):
                        point_candidates.append(mm)

                pred_mask, iou_val = best_mask_by_iou(point_candidates, gt_mask)

            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask, dtype=bool)
                iou_val = 0.0

            bf1 = boundary_f1(pred_mask, gt_mask, tolerance_px=args.boundary_tolerance_px)

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
                "elapsed_s": per_obj_time,
                "fps": 1.0 / per_obj_time if per_obj_time > 0 else 0.0,
                "device": args.device,
            })

        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No results generated.")

    results_df = pd.DataFrame(rows)
    summarize(results_df, out_dir, model_name, args.prompt_type, args.device)


if __name__ == "__main__":
    main()
