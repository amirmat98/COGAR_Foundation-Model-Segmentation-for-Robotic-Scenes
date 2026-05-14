import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from mobile_sam import SamPredictor, sam_model_registry


def read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/mobile_sam.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="outputs/sim_mobilesam_box_final")
    parser.add_argument("--results-csv", default="outputs/results/sim_mobilesam_box_final.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()

    device = resolve_device(args.device)
    index_path = Path(args.index)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    results_csv = Path(args.results_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(index_path)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    df = df.iloc[args.start_row:].reset_index(drop=True)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Rows to evaluate: {len(df)}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)

    rows = []
    current_image_path = None
    image_rgb = None
    image_start = None

    for i, row in df.iterrows():
        image_path = str(row["image_path"])
        gt_mask_path = str(row["binary_mask_path"])

        if image_path != current_image_path:
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image_start = time.perf_counter()
            predictor.set_image(image_rgb)
            image_latency_sec = time.perf_counter() - image_start
            current_image_path = image_path
        else:
            image_latency_sec = 0.0

        box = np.array([
            float(row["bbox_xmin"]),
            float(row["bbox_ymin"]),
            float(row["bbox_xmax"]),
            float(row["bbox_ymax"]),
        ], dtype=np.float32)

        t0 = time.perf_counter()
        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=True,
        )
        pred_latency_sec = time.perf_counter() - t0

        best_idx = int(np.argmax(scores))
        pred_mask = masks[best_idx].astype(bool)
        sam_score = float(scores[best_idx])

        gt_mask = read_mask(gt_mask_path)
        iou = mask_iou(pred_mask, gt_mask)
        bf1 = boundary_f1(pred_mask, gt_mask)

        out_mask_path = mask_dir / f"row_{i:05d}_object_{int(row['object_id'])}_mobilesam_box_mask.png"
        cv2.imwrite(str(out_mask_path), (pred_mask.astype(np.uint8) * 255))

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
            "iou": iou,
            "boundary_f1": bf1,
            "sam_score": sam_score,
            "image_latency_sec": image_latency_sec,
            "prediction_latency_sec": pred_latency_sec,
            "total_latency_sec": image_latency_sec + pred_latency_sec,
            "fps": 1.0 / max(image_latency_sec + pred_latency_sec, 1e-9),
        })

        print(f"[{i+1:04d}/{len(df):04d}] obj={row['object_id']} score={sam_score:.4f} IoU={iou:.4f}")

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
