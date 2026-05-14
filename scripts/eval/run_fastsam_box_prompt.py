import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import FastSAM


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] == (h, w):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else 0.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    h, w = gt.shape
    diag = (h * h + w * w) ** 0.5
    dilation = max(1, int(round(dilation_ratio * diag)))
    kernel = np.ones((3, 3), np.uint8)

    pred_boundary = pred - cv2.erode(pred, kernel, iterations=1)
    gt_boundary = gt - cv2.erode(gt, kernel, iterations=1)

    pred_dil = cv2.dilate(pred_boundary, kernel, iterations=dilation)
    gt_dil = cv2.dilate(gt_boundary, kernel, iterations=dilation)

    pred_match = (pred_boundary > 0) & (gt_dil > 0)
    gt_match = (gt_boundary > 0) & (pred_dil > 0)

    precision = pred_match.sum() / max((pred_boundary > 0).sum(), 1)
    recall = gt_match.sum() / max((gt_boundary > 0).sum(), 1)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def select_mask_by_box_iou(result, target_box_xyxy: np.ndarray, image_shape: tuple[int, int]):
    if result.masks is None or result.boxes is None:
        return None, 0.0, 0

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))

    if len(masks) == 0:
        return None, 0.0, 0

    tx1, ty1, tx2, ty2 = target_box_xyxy.astype(float)
    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)

    best_idx = 0
    best_score = -1.0

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)
        inter_x1 = max(tx1, x1)
        inter_y1 = max(ty1, y1)
        inter_x2 = min(tx2, x2)
        inter_y2 = min(ty2, y2)

        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = target_area + area - inter
        box_iou = inter / union if union > 0 else 0.0

        combined = box_iou + 0.001 * float(scores[idx])
        if combined > best_score:
            best_score = combined
            best_idx = idx

    mask = resize_mask_to(masks[best_idx] > 0.5, image_shape)
    return mask, float(scores[best_idx]), len(masks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/FastSAM-s.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--output-dir", default="outputs/sim_fastsam_s_box_final")
    parser.add_argument("--results-csv", default="outputs/results/sim_fastsam_s_box_final.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    results_csv = Path(args.results_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.index)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    df = df.iloc[args.start_row:].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Rows to evaluate: {len(df)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model = FastSAM(args.checkpoint)

    rows = []
    cached_image_path = None
    cached_result = None
    cached_latency = 0.0
    cached_shape = None

    for i, row in df.iterrows():
        image_path = str(row["image_path"])
        gt_mask_path = str(row["binary_mask_path"])

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image_shape = image_bgr.shape[:2]

        if image_path != cached_image_path:
            t0 = time.perf_counter()
            results = model(
                image_path,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )
            cached_latency = time.perf_counter() - t0
            cached_result = results[0]
            cached_image_path = image_path
            cached_shape = image_shape
        else:
            cached_latency = 0.0

        box = np.array([
            float(row["bbox_xmin"]),
            float(row["bbox_ymin"]),
            float(row["bbox_xmax"]),
            float(row["bbox_ymax"]),
        ], dtype=np.float32)

        t1 = time.perf_counter()
        pred_mask, score, num_masks = select_mask_by_box_iou(cached_result, box, cached_shape)
        selection_latency = time.perf_counter() - t1

        if pred_mask is None:
            pred_mask = np.zeros(cached_shape, dtype=bool)
            score = 0.0
            num_masks = 0

        gt_mask = read_mask(gt_mask_path)
        pred_mask = resize_mask_to(pred_mask, gt_mask.shape)

        iou_value = mask_iou(pred_mask, gt_mask)
        bf1 = boundary_f1(pred_mask, gt_mask)

        out_mask_path = mask_dir / f"row_{i:05d}_object_{int(row['object_id'])}_fastsam_box_mask.png"
        cv2.imwrite(str(out_mask_path), pred_mask.astype(np.uint8) * 255)

        total_latency = cached_latency + selection_latency

        rows.append({
            "row_index": i,
            "image_id": row.get("image_id"),
            "file_name": row.get("file_name"),
            "object_id": row.get("object_id"),
            "category_name": row.get("category_name"),
            "challenge_primary": row.get("challenge_primary"),
            "image_path": image_path,
            "gt_mask_path": gt_mask_path,
            "mask_output_path": str(out_mask_path),
            "iou": iou_value,
            "boundary_f1": bf1,
            "fastsam_score": score,
            "num_candidate_masks": num_masks,
            "image_latency_sec": cached_latency,
            "selection_latency_sec": selection_latency,
            "total_latency_sec": total_latency,
            "fps": 1.0 / max(total_latency, 1e-9),
        })

        print(f"[{i+1:04d}/{len(df):04d}] obj={row['object_id']} masks={num_masks} score={score:.4f} IoU={iou_value:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(results_csv, index=False)

    print(f"Saved results CSV: {results_csv}")
    print(f"Rows evaluated: {len(out)}")
    print(f"Mean IoU: {out['iou'].mean():.4f}")
    print(f"Median IoU: {out['iou'].median():.4f}")
    print(f"Mean Boundary F1: {out['boundary_f1'].mean():.4f}")
    print(f"Mean FPS: {out['fps'].mean():.4f}")


if __name__ == "__main__":
    main()
