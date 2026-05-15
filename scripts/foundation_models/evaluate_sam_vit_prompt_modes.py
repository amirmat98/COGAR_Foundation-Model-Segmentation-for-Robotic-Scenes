#!/usr/bin/env python3
import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


IMAGE_COLS = ["image_path", "rgb_path", "image", "img_path", "file_name", "filename"]
MASK_COLS = ["mask_path", "instance_mask_path", "binary_mask_path", "mask", "segmentation_path"]
ID_COLS = ["instance_id", "object_id", "mask_id", "id"]



def precision_context(args):
    if args.device == "cuda" and args.precision == "amp":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}. Available columns: {list(df.columns)}")
    return None


def resolve_path(p, root):
    p = Path(str(p))
    if p.is_absolute() and p.exists():
        return p
    q = Path(root) / p
    if q.exists():
        return q
    raise FileNotFoundError(f"Could not resolve path: {p}")


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_raw_mask(path):
    if str(path).endswith(".npy"):
        return np.load(path)
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def row_gt_mask(row, mask_col, root):
    mask_path = resolve_path(row[mask_col], root)
    raw = read_raw_mask(mask_path)

    inst_id = None
    for c in ID_COLS:
        if c in row.index:
            try:
                val = int(row[c])
                if val > 0:
                    inst_id = val
                    break
            except Exception:
                pass

    if inst_id is not None and np.any(raw == inst_id):
        return raw == inst_id

    return raw > 0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def bbox_from_row_or_mask(row, mask):
    cols = row.index

    xyxy_sets = [
        ("x_min", "y_min", "x_max", "y_max"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("x1", "y1", "x2", "y2"),
    ]
    for names in xyxy_sets:
        if all(c in cols for c in names):
            return np.array([float(row[names[0]]), float(row[names[1]]), float(row[names[2]]), float(row[names[3]])], dtype=np.float32)

    xywh_sets = [
        ("bbox_x", "bbox_y", "bbox_w", "bbox_h"),
        ("x", "y", "w", "h"),
        ("left", "top", "width", "height"),
    ]
    for names in xywh_sets:
        if all(c in cols for c in names):
            x, y, w, h = [float(row[n]) for n in names]
            return np.array([x, y, x + w, y + h], dtype=np.float32)

    return bbox_from_mask(mask)


def point_from_mask(mask):
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist)
    x, y = max_loc
    return np.array([[float(x), float(y)]], dtype=np.float32)


def iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(inter / union)


def boundary_f1(pred, gt, tolerance=2):
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = pred - cv2.erode(pred, kernel, iterations=1)
    gt_boundary = gt - cv2.erode(gt, kernel, iterations=1)

    dil_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    pred_dil = cv2.dilate(pred_boundary, dil_kernel, iterations=1)
    gt_dil = cv2.dilate(gt_boundary, dil_kernel, iterations=1)

    pred_count = pred_boundary.sum()
    gt_count = gt_boundary.sum()
    if pred_count == 0 or gt_count == 0:
        return 0.0

    precision = (pred_boundary & gt_dil).sum() / pred_count
    recall = (gt_boundary & pred_dil).sum() / gt_count
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def summarize(results, total_time, args):
    df = pd.DataFrame(results)
    valid = df.dropna(subset=["iou"])

    summary = {
        "model_type": args.model_type,
        "prompt_mode": args.prompt_mode,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "num_objects": int(len(valid)),
        "mean_iou": float(valid["iou"].mean()),
        "median_iou": float(valid["iou"].median()),
        "mean_boundary_f1": float(valid["boundary_f1"].mean()),
        "iou_ge_090": float((valid["iou"] >= 0.90).mean()),
        "iou_ge_075": float((valid["iou"] >= 0.75).mean()),
        "iou_ge_050": float((valid["iou"] >= 0.50).mean()),
        "iou_lt_010": float((valid["iou"] < 0.10).mean()),
        "total_model_time_s": float(total_time),
        "mean_fps": float(len(valid) / total_time) if total_time > 0 else None,
    }

    if "predicted_iou" in valid.columns and valid["predicted_iou"].notna().any():
        summary["mean_predicted_iou"] = float(valid["predicted_iou"].dropna().mean())

    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_b", "vit_l", "vit_h", "default"])
    ap.add_argument("--prompt-mode", required=True, choices=["box", "point", "auto"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-objects", type=int, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--output-prefix", required=True)
    ap.add_argument("--points-per-side", type=int, default=8)
    ap.add_argument("--points-per-batch", type=int, default=8)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.80)
    ap.add_argument("--stability-score-thresh", type=float, default=0.90)
    ap.add_argument("--crop-n-layers", type=int, default=0)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "amp"])
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_results = root / "outputs" / "results"
    out_tables = root / "outputs" / "tables"
    out_results.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index_csv)

    if args.split is not None:
        split_cols = [c for c in ["split", "dataset_split"] if c in df.columns]
        if not split_cols:
            raise ValueError(f"--split was given but no split column exists. Columns: {list(df.columns)}")
        df = df[df[split_cols[0]].astype(str) == str(args.split)].copy()

    image_col = find_col(df, IMAGE_COLS)
    mask_col = find_col(df, MASK_COLS)

    if args.max_images is not None:
        keep_images = list(df[image_col].drop_duplicates().head(args.max_images))
        df = df[df[image_col].isin(keep_images)].copy()

    if args.max_objects is not None:
        df = df.head(args.max_objects).copy()

    print(f"Loaded {len(df)} objects")
    print(f"image_col={image_col}, mask_col={mask_col}")
    print(f"Loading SAM {args.model_type} from {args.checkpoint} on {args.device}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    if args.precision in ["fp16", "amp"] and device != "cuda":
        raise ValueError("--precision fp16/amp is only supported with --device cuda")
    if args.precision == "fp16":
        # Full FP16 is experimental and may fail for SAM ViT-H prompt encoding.
        sam.half()
    sam.eval()

    predictor = None
    mask_generator = None
    if args.prompt_mode in ["box", "point"]:
        predictor = SamPredictor(sam)
    else:
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            crop_n_layers=args.crop_n_layers,
        )

    results = []
    total_model_time = 0.0

    grouped = df.groupby(image_col, sort=False)
    n_images = len(grouped)

    with torch.inference_mode():
        for img_i, (img_rel, g) in enumerate(grouped, start=1):
            img_path = resolve_path(img_rel, root)
            image = read_image(img_path)

            gt_masks = []
            boxes = []
            points = []

            for idx, row in g.iterrows():
                gt = row_gt_mask(row, mask_col, root)
                box = bbox_from_row_or_mask(row, gt)
                point = point_from_mask(gt)
                gt_masks.append(gt)
                boxes.append(box)
                points.append(point)

            if args.prompt_mode in ["box", "point"]:
                t0 = time.perf_counter()
                with precision_context(args):
                    predictor.set_image(image)
                set_image_time = time.perf_counter() - t0
                total_model_time += set_image_time

                for local_i, (idx, row) in enumerate(g.iterrows()):
                    gt = gt_masks[local_i]
                    pred = None
                    score = float("nan")

                    if args.prompt_mode == "box":
                        box = boxes[local_i]
                        if box is not None:
                            t0 = time.perf_counter()
                            with precision_context(args):
                                masks, scores, _ = predictor.predict(
                                    box=box,
                                    multimask_output=False,
                                )
                            total_model_time += time.perf_counter() - t0
                            pred = masks[0]
                            score = float(scores[0])
                    else:
                        point = points[local_i]
                        if point is not None:
                            labels = np.array([1], dtype=np.int32)
                            t0 = time.perf_counter()
                            with precision_context(args):
                                masks, scores, _ = predictor.predict(
                                    point_coords=point,
                                    point_labels=labels,
                                    multimask_output=False,
                                )
                            total_model_time += time.perf_counter() - t0
                            pred = masks[0]
                            score = float(scores[0])

                    if pred is None:
                        miou = float("nan")
                        bf1 = float("nan")
                    else:
                        miou = iou(pred, gt)
                        bf1 = boundary_f1(pred, gt)

                    rec = row.to_dict()
                    rec.update({
                        "source_index": int(idx),
                        "iou": miou,
                        "boundary_f1": bf1,
                        "predicted_iou": score,
                    })
                    results.append(rec)

            else:
                t0 = time.perf_counter()
                with precision_context(args):
                    candidates = mask_generator.generate(image)
                elapsed = time.perf_counter() - t0
                total_model_time += elapsed

                cand_masks = [c["segmentation"].astype(bool) for c in candidates]
                cand_scores = [float(c.get("predicted_iou", np.nan)) for c in candidates]

                for local_i, (idx, row) in enumerate(g.iterrows()):
                    gt = gt_masks[local_i]
                    best_iou = -1.0
                    best_mask = None
                    best_score = float("nan")

                    for cm, cs in zip(cand_masks, cand_scores):
                        val = iou(cm, gt)
                        if val > best_iou:
                            best_iou = val
                            best_mask = cm
                            best_score = cs

                    if best_mask is None:
                        miou = float("nan")
                        bf1 = float("nan")
                    else:
                        miou = best_iou
                        bf1 = boundary_f1(best_mask, gt)

                    rec = row.to_dict()
                    rec.update({
                        "source_index": int(idx),
                        "iou": miou,
                        "boundary_f1": bf1,
                        "predicted_iou": best_score,
                        "num_auto_candidates": len(candidates),
                    })
                    results.append(rec)

            if device == "cuda":
                torch.cuda.empty_cache()

            if img_i % 10 == 0 or img_i == n_images:
                print(f"[{img_i}/{n_images}] objects_done={len(results)}")

    res_df, summary = summarize(results, total_model_time, args)

    result_csv = out_results / f"{args.output_prefix}.csv"
    summary_json = out_tables / f"{args.output_prefix}_summary.json"
    summary_csv = out_tables / f"{args.output_prefix}_summary.csv"

    res_df.to_csv(result_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved results: {result_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
