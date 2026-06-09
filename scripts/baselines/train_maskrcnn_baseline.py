#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from cogar_seg.metrics import compute_boundary_f1, compute_iou
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


IMAGE_COLS = ["image_path", "rgb_path", "image", "img_path", "file_name", "filename"]
MASK_COLS = ["instance_mask_path", "mask_path", "binary_mask_path", "mask", "segmentation_path"]
CATEGORY_COLS = ["category_name", "category", "class", "class_name", "label", "object_category", "category_id"]
ID_COLS = ["instance_id", "object_id", "mask_id", "id"]
SPLIT_COLS = ["split", "dataset_split"]

CATEGORIES = [
    "box",
    "cable",
    "connector",
    "glass_object",
    "metal_part",
    "plastic_object",
    "robot_gripper",
    "screw",
    "tool",
]
CAT_TO_ID = {c: i + 1 for i, c in enumerate(CATEGORIES)}
ID_TO_CAT = {v: k for k, v in CAT_TO_ID.items()}


def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Missing column. Tried {candidates}. Available: {list(df.columns)}")
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_raw_mask(path):
    if str(path).endswith(".npy"):
        return np.load(path)
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def mask_from_row(row, mask_col, root):
    mask_path = resolve_path(row[mask_col], root)
    raw = read_raw_mask(mask_path)

    inst_id = None
    for c in ID_COLS:
        if c in row.index:
            try:
                v = int(row[c])
                if v > 0:
                    inst_id = v
                    break
            except Exception:
                pass

    if inst_id is not None and np.any(raw == inst_id):
        return raw == inst_id
    return raw > 0


def box_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


class CogarInstanceDataset(Dataset):
    def __init__(self, csv_path, root, split, max_images=None):
        self.root = Path(root).resolve()
        df = pd.read_csv(csv_path)

        self.image_col = find_col(df, IMAGE_COLS)
        self.mask_col = find_col(df, MASK_COLS)
        self.category_col = find_col(df, CATEGORY_COLS)
        self.split_col = find_col(df, SPLIT_COLS, required=False)

        if self.split_col is not None:
            df = df[df[self.split_col].astype(str) == str(split)].copy()
        else:
            print("WARNING: no split column found; using full CSV")

        image_values = list(df[self.image_col].drop_duplicates())
        if max_images is not None:
            image_values = image_values[:max_images]
            df = df[df[self.image_col].isin(image_values)].copy()

        self.df = df.reset_index(drop=True)
        self.images = list(self.df[self.image_col].drop_duplicates())

        print(
            f"{split}: images={len(self.images)}, objects={len(self.df)}, "
            f"image_col={self.image_col}, mask_col={self.mask_col}, category_col={self.category_col}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_rel = self.images[idx]
        g = self.df[self.df[self.image_col] == img_rel]

        img_path = resolve_path(img_rel, self.root)
        img = read_image(img_path)
        h, w = img.shape[:2]

        boxes, masks, labels, areas = [], [], [], []

        for _, row in g.iterrows():
            cat_raw = row[self.category_col]

            if self.category_col == "category_id":
                try:
                    label_id = int(cat_raw)
                except Exception:
                    continue
                if label_id < 1 or label_id > len(CATEGORIES):
                    continue
            else:
                cat = str(cat_raw)
                if cat not in CAT_TO_ID:
                    continue
                label_id = CAT_TO_ID[cat]

            m = mask_from_row(row, self.mask_col, self.root)
            if m.shape[:2] != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

            box = box_from_mask(m)
            if box is None:
                continue

            boxes.append(box)
            masks.append(m.astype(np.uint8))
            labels.append(label_id)
            areas.append(float(m.sum()))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)

        image_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "image_path": str(img_rel),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes, weights_name, min_size, max_size):
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

        weights = None
        if weights_name == "coco":
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT

        model = maskrcnn_resnet50_fpn(
            weights=weights,
            min_size=min_size,
            max_size=max_size,
            box_detections_per_img=100,
        )
    except Exception as e:
        print(f"Modern torchvision constructor failed: {e}")
        print("Trying legacy constructor...")
        pretrained = weights_name == "coco"
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            min_size=min_size,
            max_size=max_size,
            box_detections_per_img=100,
        )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)

    return model


def train_one_epoch(model, loader, optimizer, device, epoch, amp=False, print_freq=10):
    model.train()
    total_loss = 0.0
    n = 0

    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    for step, (images, targets) in enumerate(loader, start=1):
        images = [img.to(device) for img in images]
        targets_gpu = []
        for t in targets:
            tg = {}
            for k, v in t.items():
                if torch.is_tensor(v):
                    tg[k] = v.to(device)
                elif k != "image_path":
                    tg[k] = v
            targets_gpu.append(tg)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp):
            loss_dict = model(images, targets_gpu)
            loss = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(loss):
            print("Non-finite loss:", loss.item(), loss_dict)
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        val = float(loss.item())
        total_loss += val
        n += 1

        if step % print_freq == 0 or step == len(loader):
            pieces = ", ".join(f"{k}={float(v.item()):.4f}" for k, v in loss_dict.items())
            print(f"epoch={epoch} step={step}/{len(loader)} loss={val:.4f} {pieces}")

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, score_thresh, output_prefix, root):
    model.eval()

    records = []
    total_inf_time = 0.0
    total_images = 0
    total_gt = 0

    for images, targets in loader:
        image = images[0].to(device)
        target = targets[0]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model([image])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_inf_time += time.perf_counter() - t0
        total_images += 1

        gt_masks = target["masks"].cpu().numpy().astype(bool)
        gt_labels = target["labels"].cpu().numpy().astype(int)
        total_gt += len(gt_labels)

        keep = pred["scores"].detach().cpu().numpy() >= score_thresh
        pred_scores = pred["scores"].detach().cpu().numpy()[keep]
        pred_labels = pred["labels"].detach().cpu().numpy().astype(int)[keep]
        pred_masks_raw = pred["masks"].detach().cpu().numpy()[keep, 0]
        pred_masks = pred_masks_raw >= 0.5

        for gi, (gm, gl) in enumerate(zip(gt_masks, gt_labels)):
            best_iou = 0.0
            best_bf1 = 0.0
            best_score = 0.0
            best_label = -1

            for pm, pl, ps in zip(pred_masks, pred_labels, pred_scores):
                if int(pl) != int(gl):
                    continue
                miou = compute_iou(pm, gm, empty_value=float("nan"))
                if miou > best_iou:
                    best_iou = miou
                    best_bf1 = compute_boundary_f1(pm, gm)
                    best_score = float(ps)
                    best_label = int(pl)

            records.append({
                "image_path": target["image_path"],
                "gt_label_id": int(gl),
                "gt_category": ID_TO_CAT.get(int(gl), str(gl)),
                "best_pred_label_id": int(best_label),
                "best_score": float(best_score),
                "iou": float(best_iou),
                "boundary_f1": float(best_bf1),
                "matched": bool(best_iou > 0),
            })

        print(f"eval image {total_images}/{len(loader)} gt_done={len(records)} preds_kept={len(pred_scores)}")

    df = pd.DataFrame(records)
    valid = df.dropna(subset=["iou"])

    summary = {
        "model": "maskrcnn_resnet50_fpn",
        "num_test_images": int(total_images),
        "num_test_objects": int(total_gt),
        "score_thresh": float(score_thresh),
        "mean_iou": float(valid["iou"].mean()) if len(valid) else 0.0,
        "median_iou": float(valid["iou"].median()) if len(valid) else 0.0,
        "mean_boundary_f1": float(valid["boundary_f1"].mean()) if len(valid) else 0.0,
        "iou_ge_090": float((valid["iou"] >= 0.90).mean()) if len(valid) else 0.0,
        "iou_ge_075": float((valid["iou"] >= 0.75).mean()) if len(valid) else 0.0,
        "iou_ge_050": float((valid["iou"] >= 0.50).mean()) if len(valid) else 0.0,
        "iou_lt_010": float((valid["iou"] < 0.10).mean()) if len(valid) else 0.0,
        "total_inference_time_s": float(total_inf_time),
        "mean_fps_images": float(total_images / total_inf_time) if total_inf_time > 0 else 0.0,
        "mean_fps_objects": float(total_gt / total_inf_time) if total_inf_time > 0 else 0.0,
    }

    per_class = (
        df.groupby("gt_category")
        .agg(
            n=("iou", "size"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            iou_ge_050=("iou", lambda x: float((x >= 0.50).mean())),
            iou_lt_010=("iou", lambda x: float((x < 0.10).mean())),
        )
        .reset_index()
    )

    out_results = Path(root) / "outputs" / "results"
    out_tables = Path(root) / "outputs" / "tables"
    out_results.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_results / f"{output_prefix}.csv", index=False)
    per_class.to_csv(out_tables / f"{output_prefix}_per_class.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_tables / f"{output_prefix}_summary.csv", index=False)
    with open(out_tables / f"{output_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved:", out_results / f"{output_prefix}.csv")
    print("Saved:", out_tables / f"{output_prefix}_summary.json")
    print("Saved:", out_tables / f"{output_prefix}_per_class.csv")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--output-prefix", default="maskrcnn_resnet50_fpn_cogar_small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0025)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--weights", default="coco", choices=["coco", "none"])
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--max-size", type=int, default=768)
    ap.add_argument("--train-max-images", type=int, default=None)
    ap.add_argument("--val-max-images", type=int, default=None)
    ap.add_argument("--test-max-images", type=int, default=None)
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = CogarInstanceDataset(args.index_csv, root, "train", args.train_max_images)
    val_ds = CogarInstanceDataset(args.index_csv, root, "val", args.val_max_images)
    test_ds = CogarInstanceDataset(args.index_csv, root, "test", args.test_max_images)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    num_classes = len(CATEGORIES) + 1
    model = build_model(num_classes, args.weights, args.min_size, args.max_size)

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("Backbone frozen")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 2, 1), gamma=0.1)

    ckpt_dir = root / "outputs" / "baselines" / "maskrcnn" / args.output_prefix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            amp=(args.amp and device.type == "cuda"),
            print_freq=10,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch={epoch} avg_train_loss={avg_loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "categories": CATEGORIES,
                "args": vars(args),
                "history": history,
            },
            ckpt_dir / "last.pt",
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "categories": CATEGORIES,
                    "args": vars(args),
                    "history": history,
                },
                ckpt_dir / "best.pt",
            )

    pd.DataFrame(history).to_csv(ckpt_dir / "training_history.csv", index=False)

    summary = evaluate(
        model,
        test_loader,
        device,
        args.score_thresh,
        args.output_prefix,
        root,
    )

    with open(ckpt_dir / "final_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Checkpoint dir:", ckpt_dir)


if __name__ == "__main__":
    main()
