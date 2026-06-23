"""Train Mask R-CNN baselines on Task 5A COCO subsets."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

COCO_STATS = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)

from cogar_seg.config import load_config as load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task5_mask_rcnn.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for setup validation.")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def relative_to_repo(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    return load_project_config(resolve_repo_path(path))


def write_json(path: str | Path, data: Any) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "status",
        "epochs",
        "best_epoch",
        "best_segm_AP",
        "best_segm_AP50",
        "best_bbox_AP",
        "best_bbox_AP50",
        "elapsed_s",
        "best_checkpoint",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            best_metrics = record.get("best_metrics") or {}
            segm = best_metrics.get("segm") or {}
            bbox = best_metrics.get("bbox") or {}
            writer.writerow(
                {
                    "dataset": record.get("dataset"),
                    "status": record.get("status"),
                    "epochs": record.get("epochs"),
                    "best_epoch": record.get("best_epoch"),
                    "best_segm_AP": segm.get("AP"),
                    "best_segm_AP50": segm.get("AP50"),
                    "best_bbox_AP": bbox.get("AP"),
                    "best_bbox_AP50": bbox.get("AP50"),
                    "elapsed_s": record.get("elapsed_s"),
                    "best_checkpoint": record.get("best_checkpoint"),
                }
            )


def resolve_image_path(dataset_config: dict[str, Any], file_name: str) -> Path:
    root = Path(dataset_config["root"])
    image_path_config = dataset_config.get("image_path", {})
    mode = image_path_config.get("mode", "coco_file_name_relative_to_root")

    if mode == "coco_file_name_relative_to_root":
        return root / file_name

    if mode == "basename_in_image_dir":
        return Path(image_path_config["image_dir"]) / Path(file_name).name

    raise ValueError(f"Unsupported image path mode: {mode}")


def decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(np.uint8)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rles)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        return decoded.astype(np.uint8)

    raise ValueError(f"Unsupported segmentation type: {type(segmentation)!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class CocoInstanceDataset:
    def __init__(
        self,
        annotation_file: str | Path,
        source_dataset_config: dict[str, Any],
        category_id_to_label: dict[int, int] | None = None,
    ) -> None:
        self.annotation_file = resolve_repo_path(annotation_file)
        self.source_dataset_config = source_dataset_config
        self.coco = load_json(self.annotation_file)
        self.images = sorted(self.coco["images"], key=lambda item: int(item["id"]))
        self.image_by_id = {int(image["id"]): image for image in self.images}

        categories = sorted(self.coco["categories"], key=lambda item: int(item["id"]))
        if category_id_to_label is None:
            category_id_to_label = {
                int(category["id"]): idx + 1 for idx, category in enumerate(categories)
            }
        self.category_id_to_label = category_id_to_label
        self.label_to_category_id = {
            label: category_id for category_id, label in category_id_to_label.items()
        }
        self.categories = categories

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco["annotations"]:
            if int(annotation.get("iscrowd", 0)):
                continue
            if float(annotation.get("area", 0.0)) <= 0:
                continue
            image_id = int(annotation["image_id"])
            annotations_by_image[image_id].append(annotation)
        self.annotations_by_image = annotations_by_image
        self.image_ids = [
            int(image["id"])
            for image in self.images
            if annotations_by_image.get(int(image["id"]))
        ]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch

        image_id = self.image_ids[index]
        image_info = self.image_by_id[image_id]
        image_path = resolve_image_path(self.source_dataset_config, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        height = int(image_info["height"])
        width = int(image_info["width"])
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for annotation in self.annotations_by_image[image_id]:
            x, y, w, h = [float(value) for value in annotation["bbox"]]
            if w <= 0 or h <= 0:
                continue
            mask = decode_mask(annotation["segmentation"], height, width)
            if int(mask.sum()) == 0:
                continue
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_label[category_id])
            masks.append(mask)
            areas.append(float(annotation.get("area", mask.sum())))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        if not boxes:
            raise ValueError(f"No valid annotations after filtering for image_id={image_id}")

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def selected_datasets(
    config: dict[str, Any],
    source_config: dict[str, Any],
    selected_names: list[str] | None,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    datasets = []
    for name, dataset_config in config["datasets"].items():
        if selected_names is not None and name not in selected_names:
            continue
        if not dataset_config.get("enabled", False):
            continue
        if name not in source_config["datasets"]:
            raise KeyError(f"{name} missing from source dataset config")
        datasets.append((name, dataset_config, source_config["datasets"][name]))
    return datasets


def build_model(num_classes: int, config: dict[str, Any]) -> Any:
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model_config = config["model"]
    weights = model_config.get("weights", "DEFAULT")
    if isinstance(weights, str) and weights.upper() == "DEFAULT":
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    elif isinstance(weights, str) and weights.lower() in {"none", "null", "false"}:
        weights = None
    weights_backbone = model_config.get("weights_backbone", "DEFAULT")
    if isinstance(weights_backbone, str) and weights_backbone.upper() == "DEFAULT":
        weights_backbone = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    elif isinstance(weights_backbone, str) and weights_backbone.lower() in {"none", "null", "false"}:
        weights_backbone = None
    trainable_backbone_layers = int(model_config.get("trainable_backbone_layers", 3))

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=int(model_config.get("min_size", 800)),
        max_size=int(model_config.get("max_size", 1333)),
        box_score_thresh=float(model_config.get("box_score_thresh", 0.05)),
    )

    box_in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in_features, num_classes)

    mask_in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        mask_in_features,
        hidden_layer,
        num_classes,
    )
    return model


def build_device(device_config: str | int) -> Any:
    import torch

    if str(device_config).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_config}")
    return torch.device("cpu")


def move_targets_to_device(targets: list[dict[str, Any]], device: Any) -> list[dict[str, Any]]:
    return [
        {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
        for target in targets
    ]


def coco_stats_to_dict(stats: Any) -> dict[str, float]:
    return {name: float(stats[idx]) for idx, name in enumerate(COCO_STATS)}


def run_coco_eval(coco_gt: Any, predictions: list[dict[str, Any]], image_ids: list[int], iou_type: str) -> dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        return {name: 0.0 for name in COCO_STATS}

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = image_ids
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return coco_stats_to_dict(coco_eval.stats)


def evaluate(
    model: Any,
    data_loader: Any,
    dataset: CocoInstanceDataset,
    device: Any,
    score_threshold: float,
    mask_threshold: float,
    predictions_path: Path,
) -> dict[str, Any]:
    import torch
    from pycocotools.coco import COCO

    model.eval()
    predictions = []
    image_ids = []
    started_at = time.perf_counter()

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())
                image_ids.append(image_id)
                boxes = output["boxes"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                masks = output["masks"].detach().cpu().numpy()

                for idx, score in enumerate(scores):
                    if float(score) < score_threshold:
                        continue
                    label = int(labels[idx])
                    category_id = dataset.label_to_category_id.get(label)
                    if category_id is None:
                        continue

                    x1, y1, x2, y2 = [float(value) for value in boxes[idx]]
                    width = max(0.0, x2 - x1)
                    height = max(0.0, y2 - y1)
                    if width <= 0 or height <= 0:
                        continue

                    mask = (masks[idx, 0] >= mask_threshold).astype(np.uint8)
                    if int(mask.sum()) == 0:
                        continue
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("ascii")

                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(category_id),
                            "bbox": [x1, y1, width, height],
                            "score": float(score),
                            "segmentation": rle,
                        }
                    )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")

    coco_gt = COCO(str(dataset.annotation_file))
    unique_image_ids = sorted(set(image_ids))
    bbox_stats = run_coco_eval(coco_gt, predictions, unique_image_ids, "bbox")
    segm_stats = run_coco_eval(coco_gt, predictions, unique_image_ids, "segm")
    return {
        "bbox": bbox_stats,
        "segm": segm_stats,
        "predictions": relative_to_repo(predictions_path),
        "prediction_count": len(predictions),
        "elapsed_s": time.perf_counter() - started_at,
    }


def run_name(dataset_name: str, config: dict[str, Any], epochs: int, smoke: bool) -> str:
    suffix = config["model"].get("run_suffix", "mask_rcnn")
    if smoke:
        return f"{dataset_name}_{suffix}_smoke"
    return f"{dataset_name}_{suffix}_e{epochs}"


def train_one_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
    source_dataset_config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    epochs = 1 if args.smoke else int(args.epochs or training["epochs"])
    batch_size = int(args.batch_size or training["batch_size"])
    workers = int(args.workers if args.workers is not None else training["workers"])
    log_every = int(args.log_every if args.log_every is not None else training["log_every"])
    device_config = args.device if args.device is not None else training["device"]
    output_root = resolve_repo_path(config["task"]["output_root"])
    results_root = resolve_repo_path(config["task"]["results_root"])
    name = run_name(dataset_name, config, epochs, args.smoke)
    run_dir = results_root / name
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    last_checkpoint = run_dir / "checkpoints" / "last.pt"

    train_dataset = CocoInstanceDataset(
        annotation_file=dataset_config["train_annotations"],
        source_dataset_config=source_dataset_config,
    )
    val_dataset = CocoInstanceDataset(
        annotation_file=dataset_config["val_annotations"],
        source_dataset_config=source_dataset_config,
        category_id_to_label=train_dataset.category_id_to_label,
    )
    num_classes = len(train_dataset.category_id_to_label) + 1

    print(
        f"[START] Mask R-CNN dataset={dataset_name} epochs={epochs} batch={batch_size} "
        f"train_images={len(train_dataset)} val_images={len(val_dataset)} "
        f"classes={num_classes} device={device_config} output={relative_to_repo(run_dir)}",
        flush=True,
    )

    if args.dry_run:
        return {
            "dataset": dataset_name,
            "status": "dry_run",
            "epochs": epochs,
            "num_classes": num_classes,
            "train_images": len(train_dataset),
            "val_images": len(val_dataset),
            "run_dir": relative_to_repo(run_dir),
        }

    import torch
    from torch.utils.data import DataLoader

    device = build_device(device_config)

    if best_checkpoint.exists() and not args.rerun_complete:
        print(f"[SKIP] {dataset_name}: existing best checkpoint {best_checkpoint}", flush=True)
        summary_path = output_root / f"{dataset_name}_train_summary.json"
        if summary_path.exists():
            return load_json(summary_path)
        return {
            "dataset": dataset_name,
            "status": "skipped_existing",
            "best_checkpoint": relative_to_repo(best_checkpoint),
        }

    generator = torch.Generator()
    generator.manual_seed(int(config["task"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
    )

    model = build_model(num_classes=num_classes, config=config)
    model.to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training["step_size"]),
        gamma=float(training["gamma"]),
    )
    use_amp = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    eval_every = max(1, int(training["eval_every"]))
    score_threshold = float(training["score_threshold"])
    mask_threshold = float(training["mask_threshold"])

    best_epoch = None
    best_metric = float("-inf")
    best_metrics: dict[str, Any] | None = None
    epoch_summaries = []
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_started_at = time.perf_counter()
        running_loss = 0.0
        running_batches = 0

        for batch_index, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(losses.detach().cpu())
            running_loss += loss_value
            running_batches += 1
            if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                print(
                    f"[PROGRESS] {dataset_name} epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{len(train_loader)} loss={loss_value:.4f}",
                    flush=True,
                )

        lr_scheduler.step()
        average_loss = running_loss / max(running_batches, 1)
        epoch_summary: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": average_loss,
            "elapsed_s": time.perf_counter() - epoch_started_at,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        should_eval = epoch == epochs or epoch % eval_every == 0 or epoch == 1
        if should_eval:
            predictions_path = run_dir / "predictions" / f"val_epoch_{epoch:03d}.json"
            metrics = evaluate(
                model=model,
                data_loader=val_loader,
                dataset=val_dataset,
                device=device,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                predictions_path=predictions_path,
            )
            epoch_summary["metrics"] = metrics
            current_metric = float(metrics["segm"]["AP"])
            print(
                f"[EVAL] {dataset_name} epoch={epoch} "
                f"segm_AP={metrics['segm']['AP']:.4f} segm_AP50={metrics['segm']['AP50']:.4f} "
                f"bbox_AP={metrics['bbox']['AP']:.4f}",
                flush=True,
            )
            if current_metric >= best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_metrics = metrics
                best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "category_id_to_label": train_dataset.category_id_to_label,
                        "label_to_category_id": train_dataset.label_to_category_id,
                        "metrics": metrics,
                        "config": config,
                    },
                    best_checkpoint,
                )
                print(f"[CHECKPOINT] best={relative_to_repo(best_checkpoint)}", flush=True)

        epoch_summaries.append(epoch_summary)

    last_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "category_id_to_label": train_dataset.category_id_to_label,
            "label_to_category_id": train_dataset.label_to_category_id,
            "config": config,
        },
        last_checkpoint,
    )

    status = "ok" if best_checkpoint.exists() else "missing_best_checkpoint"
    summary = {
        "dataset": dataset_name,
        "status": status,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_metric": "segm.AP",
        "best_metrics": best_metrics,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "num_classes": num_classes,
        "category_id_to_label": train_dataset.category_id_to_label,
        "elapsed_s": time.perf_counter() - started_at,
        "run_dir": relative_to_repo(run_dir),
        "best_checkpoint": relative_to_repo(best_checkpoint),
        "last_checkpoint": relative_to_repo(last_checkpoint),
        "epochs_summary": epoch_summaries,
    }
    print(
        f"[DONE] {dataset_name}: status={status} best_epoch={best_epoch} "
        f"best_segm_AP={best_metric:.4f} elapsed={summary['elapsed_s'] / 60.0:.1f}min",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    source_config = load_yaml(config["task"]["source_dataset_config"])
    set_seed(int(config["task"]["seed"]))

    summaries = []
    output_root = resolve_repo_path(config["task"]["output_root"])
    for dataset_name, dataset_config, source_dataset_config in selected_datasets(
        config,
        source_config,
        args.datasets,
    ):
        summary = train_one_dataset(
            args=args,
            config=config,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            source_dataset_config=source_dataset_config,
        )
        summaries.append(summary)
        if not args.dry_run:
            write_json(output_root / f"{dataset_name}_train_summary.json", summary)

    if not args.dry_run:
        write_json(output_root / "summary.json", summaries)
        write_csv(output_root / "metrics_summary.csv", summaries)
    print("[DONE] Mask R-CNN training wrapper", flush=True)


if __name__ == "__main__":
    main()
