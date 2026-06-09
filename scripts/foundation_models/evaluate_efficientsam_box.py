#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from cogar_seg.metrics import compute_boundary_f1, compute_iou
from PIL import Image
from tqdm import tqdm


KNOWN_CHALLENGES = [
    "small_parts",
    "partial_occlusion",
    "dynamic_scene",
    "reflective_metal",
    "transparent_glass",
]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates):
    norm_to_col = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        key = norm_name(cand)
        if key in norm_to_col:
            return norm_to_col[key]
    return None


def truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v) != 0.0
    return str(v).strip().lower() in {"1", "true", "yes", "y", "present"}


def resolve_path(value, project_root: Path):
    if value is None or pd.isna(value):
        return None

    raw = str(value).strip()
    if raw == "":
        return None

    raw = os.path.expanduser(raw)
    p = Path(raw)

    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            project_root / p,
            project_root / "data" / "cogar_sim_500_final" / p,
            project_root / "data" / p,
        ])

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
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+", s)
            return [float(x) for x in nums] if nums else None

    if isinstance(obj, dict):
        return obj
    return obj


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def get_bbox(row, df: pd.DataFrame, mask=None, bbox_format="auto"):
    colmap = {norm_name(c): c for c in df.columns}

    xyxy_groups = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("bboxx1", "bboxy1", "bboxx2", "bboxy2"),
        ("bboxxmin", "bboxymin", "bboxxmax", "bboxymax"),
        ("left", "top", "right", "bottom"),
    ]

    for group in xyxy_groups:
        if all(g in colmap for g in group):
            vals = [float(row[colmap[g]]) for g in group]
            return vals

    xywh_groups = [
        ("x", "y", "w", "h"),
        ("bboxx", "bboxy", "bboxw", "bboxh"),
        ("bboxx", "bboxy", "bboxwidth", "bboxheight"),
        ("xmin", "ymin", "width", "height"),
    ]

    for group in xywh_groups:
        if all(g in colmap for g in group):
            x, y, w, h = [float(row[colmap[g]]) for g in group]
            return [x, y, x + w, y + h]

    bbox_col = find_col(df, ["bbox", "box", "bounding_box", "bbox_xywh", "bbox_xyxy"])
    if bbox_col is not None and not pd.isna(row[bbox_col]):
        obj = parse_number_list(row[bbox_col])
        if isinstance(obj, dict):
            if all(k in obj for k in ["x1", "y1", "x2", "y2"]):
                return [float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])]
            if all(k in obj for k in ["x", "y", "w", "h"]):
                x, y, w, h = map(float, [obj["x"], obj["y"], obj["w"], obj["h"]])
                return [x, y, x + w, y + h]
        elif isinstance(obj, (list, tuple)) and len(obj) >= 4:
            vals = [float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])]
            if bbox_format == "xyxy":
                return vals
            if bbox_format == "xywh":
                x, y, w, h = vals
                return [x, y, x + w, y + h]
            # auto: assume xywh when third/fourth look like width/height
            x, y, a, b = vals
            if a <= x or b <= y:
                return [x, y, x + a, y + b]
            return vals

    if mask is not None:
        return bbox_from_mask(mask)

    return None


def load_mask_from_segmentation(seg_value, image_hw):
    h, w = image_hw
    obj = parse_number_list(seg_value)
    if obj is None:
        return None

    # Optional COCO RLE support.
    if isinstance(obj, dict) and "counts" in obj and "size" in obj:
        try:
            from pycocotools import mask as mask_utils
            m = mask_utils.decode(obj)
            return (m > 0).astype(np.uint8)
        except Exception:
            return None

    mask = np.zeros((h, w), dtype=np.uint8)

    def fill_one(poly):
        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] >= 3:
            cv2.fillPoly(mask, [np.round(arr).astype(np.int32)], 1)

    if isinstance(obj, list):
        if len(obj) == 0:
            return None
        if all(isinstance(x, (int, float)) for x in obj):
            fill_one(obj)
        else:
            for poly in obj:
                if isinstance(poly, list):
                    fill_one(poly)

    return mask.astype(bool)


def load_gt_mask(row, df: pd.DataFrame, project_root: Path, image_hw):
    mask_col = find_col(df, [
        "mask_path",
        "gt_mask_path",
        "binary_mask_path",
        "segmentation_mask_path",
        "mask_file",
        "mask_filename",
        "annotation_path",
    ])

    if mask_col is not None:
        p = resolve_path(row[mask_col], project_root)
        if p is not None and p.exists():
            m = np.array(Image.open(p).convert("L"))
            m = m > 0
            h, w = image_hw
            if m.shape[:2] != (h, w):
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
            return m

    seg_col = find_col(df, ["segmentation", "polygon", "polygons"])
    if seg_col is not None and not pd.isna(row[seg_col]):
        return load_mask_from_segmentation(row[seg_col], image_hw)

    return None


def get_category(row, df):
    col = find_col(df, [
        "category_name",
        "category",
        "class_name",
        "class",
        "label",
        "object_category",
        "name",
    ])
    return str(row[col]) if col is not None else "unknown"


def get_split(row, df):
    col = find_col(df, ["split", "dataset_split"])
    return str(row[col]) if col is not None else "all"


def get_instance_id(row, df, fallback):
    col = find_col(df, ["instance_id", "object_id", "annotation_id", "id"])
    return str(row[col]) if col is not None else str(fallback)


def get_challenge(row, df):
    found = []

    primary_col = find_col(df, ["challenge_primary", "primary_challenge", "challenge", "challenge_type", "condition"])
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

    # Remove duplicates while preserving order.
    deduped = []
    for ch in found:
        if ch not in deduped:
            deduped.append(ch)

    return ";".join(deduped) if deduped else "none"


def build_model(efficientsam_repo: Path, model_size: str, device):
    sys.path.insert(0, str(efficientsam_repo))

    if model_size == "ti":
        ckpt = efficientsam_repo / "weights" / "efficient_sam_vitt.pt"
    else:
        ckpt = efficientsam_repo / "weights" / "efficient_sam_vits.pt"
        z = efficientsam_repo / "weights" / "efficient_sam_vits.pt.zip"
        if not ckpt.exists() and z.exists():
            with zipfile.ZipFile(z, "r") as zip_ref:
                zip_ref.extractall(efficientsam_repo / "weights")

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    old_cwd = os.getcwd()
    os.chdir(efficientsam_repo)
    try:
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
        model = build_efficient_sam_vitt() if model_size == "ti" else build_efficient_sam_vits()
    finally:
        os.chdir(old_cwd)

    model = model.to(device)
    model.eval()
    return model


def image_to_tensor(image: Image.Image, device):
    arr = np.array(image.convert("RGB"))
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def summarize(results: pd.DataFrame, out_dir: Path, model_name: str, device_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    per_instance_path = out_dir / f"{model_name.lower().replace(' ', '_')}_box_per_instance.csv"
    results.to_csv(per_instance_path, index=False)

    total_time = float(results["elapsed_s"].sum())
    n = int(len(results))
    mean_fps = float(n / total_time) if total_time > 0 else 0.0

    overall = pd.DataFrame([{
        "model": model_name,
        "prompt_type": "box",
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
    overall_path = out_dir / "overall_summary.csv"
    overall.to_csv(overall_path, index=False)

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
    cat_path = out_dir / "mean_iou_by_category.csv"
    cat.to_csv(cat_path, index=False)

    expanded = []
    for _, r in results.iterrows():
        challenges = str(r["challenge"]).split(";") if str(r["challenge"]) else ["none"]
        for ch in challenges:
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
    chal_path = out_dir / "mean_iou_by_challenge.csv"
    chal.to_csv(chal_path, index=False)

    print("\nSaved:")
    print(" ", per_instance_path)
    print(" ", overall_path)
    print(" ", cat_path)
    print(" ", chal_path)
    print("\nOverall:")
    print(overall.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--efficientsam-repo", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--output-dir", default="outputs/tables/efficientsam")
    ap.add_argument("--model-size", choices=["ti", "s"], default="ti")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--split", default="all", help="all, train, val, or test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-queries-per-image", type=int, default=8)
    ap.add_argument("--bbox-format", choices=["auto", "xyxy", "xywh"], default="auto")
    ap.add_argument("--boundary-tolerance-px", type=int, default=2)
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    index_csv = Path(args.index_csv).expanduser()
    if not index_csv.is_absolute():
        index_csv = project_root / index_csv

    efficientsam_repo = Path(args.efficientsam_repo).expanduser().resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device

    device = torch.device(device_name)
    print(f"Device: {device}")
    print(f"EfficientSAM repo: {efficientsam_repo}")
    print(f"Index CSV: {index_csv}")

    df = pd.read_csv(index_csv)
    print("CSV columns:", list(df.columns))

    split_col = find_col(df, ["split", "dataset_split"])
    if args.split != "all":
        if split_col is None:
            raise RuntimeError("--split was requested but no split column was found.")
        df = df[df[split_col].astype(str).str.lower() == args.split.lower()].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    image_col = find_col(df, ["image_path", "rgb_path", "file_name", "filename", "image", "image_file", "path"])
    if image_col is None:
        raise RuntimeError("Could not find an image path column. Add/rename image_path in the index CSV.")

    model = build_model(efficientsam_repo, args.model_size, device)
    model_name = "EfficientSAM-Ti" if args.model_size == "ti" else "EfficientSAM-S"

    rows = []
    grouped = df.groupby(image_col, sort=False)

    with torch.no_grad():
        for image_value, g in tqdm(grouped, total=len(grouped), desc=f"{model_name} box"):
            image_path = resolve_path(image_value, project_root)
            if image_path is None or not image_path.exists():
                print(f"[WARN] missing image: {image_value}")
                continue

            image = Image.open(image_path).convert("RGB")
            w, h = image.size
            image_tensor = image_to_tensor(image, device)

            prepared = []
            for idx, row in g.iterrows():
                gt_mask = load_gt_mask(row, df, project_root, (h, w))
                bbox = get_bbox(row, df, mask=gt_mask, bbox_format=args.bbox_format)

                if gt_mask is None:
                    print(f"[WARN] missing GT mask for row {idx}; skipping")
                    continue
                if bbox is None:
                    print(f"[WARN] missing bbox for row {idx}; skipping")
                    continue

                x1, y1, x2, y2 = bbox
                x1 = float(np.clip(x1, 0, w - 1))
                x2 = float(np.clip(x2, 0, w - 1))
                y1 = float(np.clip(y1, 0, h - 1))
                y2 = float(np.clip(y2, 0, h - 1))

                if x2 <= x1 or y2 <= y1:
                    print(f"[WARN] invalid bbox for row {idx}: {bbox}; skipping")
                    continue

                prepared.append((idx, row, gt_mask, [x1, y1, x2, y2]))

            for start in range(0, len(prepared), args.max_queries_per_image):
                chunk = prepared[start:start + args.max_queries_per_image]
                if not chunk:
                    continue

                pts = []
                labels = []
                for _, _, _, bbox in chunk:
                    x1, y1, x2, y2 = bbox
                    pts.append([[x1, y1], [x2, y2]])
                    labels.append([2, 3])

                input_points = torch.tensor([pts], dtype=torch.float32, device=device)
                input_labels = torch.tensor([labels], dtype=torch.int64, device=device)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                predicted_logits, predicted_iou = model(
                    image_tensor,
                    input_points,
                    input_labels,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                per_obj_elapsed = elapsed / max(len(chunk), 1)

                best_ids = torch.argmax(predicted_iou[0], dim=-1).detach().cpu().numpy()

                for j, (idx, row, gt_mask, bbox) in enumerate(chunk):
                    best = int(best_ids[j])
                    pred_mask = (predicted_logits[0, j, best] >= 0).detach().cpu().numpy()

                    iou = compute_iou(pred_mask, gt_mask, empty_value=1.0)
                    bf1 = compute_boundary_f1(
                        pred_mask, gt_mask, bound_thresh=args.boundary_tolerance_px
                    )
                    pred_score = float(predicted_iou[0, j, best].detach().cpu().item())

                    rows.append({
                        "model": model_name,
                        "prompt_type": "box",
                        "split": get_split(row, df),
                        "image_path": str(image_path.relative_to(project_root)) if str(image_path).startswith(str(project_root)) else str(image_path),
                        "instance_id": get_instance_id(row, df, idx),
                        "category": get_category(row, df),
                        "challenge": get_challenge(row, df),
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "iou": iou,
                        "boundary_f1": bf1,
                        "predicted_iou": pred_score,
                        "elapsed_s": per_obj_elapsed,
                        "fps": 1.0 / per_obj_elapsed if per_obj_elapsed > 0 else 0.0,
                        "device": device_name,
                    })

                del predicted_logits, predicted_iou, input_points, input_labels
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No rows evaluated. Check image/mask/bbox columns printed above.")

    results = pd.DataFrame(rows)
    summarize(results, out_dir, model_name, device_name)


if __name__ == "__main__":
    main()
