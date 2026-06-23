"""Train DeepLabV3+ semantic segmentation baselines on Task 5A masks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

from cogar_seg.config import load_config as load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task5_deeplabv3plus.yaml")
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


def load_yaml(path: str | Path) -> dict[str, Any]:
    return load_project_config(resolve_repo_path(path))


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


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
        "best_miou",
        "best_foreground_miou",
        "best_pixel_accuracy",
        "best_mean_accuracy",
        "elapsed_s",
        "best_checkpoint",
    ]
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            best_metrics = record.get("best_metrics") or {}
            writer.writerow(
                {
                    "dataset": record.get("dataset"),
                    "status": record.get("status"),
                    "epochs": record.get("epochs"),
                    "best_epoch": record.get("best_epoch"),
                    "best_miou": best_metrics.get("miou"),
                    "best_foreground_miou": best_metrics.get("foreground_miou"),
                    "best_pixel_accuracy": best_metrics.get("pixel_accuracy"),
                    "best_mean_accuracy": best_metrics.get("mean_accuracy"),
                    "elapsed_s": record.get("elapsed_s"),
                    "best_checkpoint": record.get("best_checkpoint"),
                }
            )


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


def selected_datasets(
    config: dict[str, Any],
    selected_names: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    datasets = []
    for name, dataset_config in config["datasets"].items():
        if selected_names is not None and name not in selected_names:
            continue
        if dataset_config.get("enabled", False):
            datasets.append((name, dataset_config))
    return datasets


def resolve_dataset_root(dataset_name: str, dataset_yaml: dict[str, Any]) -> Path:
    root = Path(dataset_yaml["root"])
    if root.exists():
        return root
    fallback = REPO_ROOT / "data" / "task5_baselines" / "deeplabv3plus" / dataset_name
    if fallback.exists():
        return fallback
    return root


def image_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    image_size = config["training"]["image_size"]
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("training.image_size must be an int or [height, width]")
    return int(image_size[0]), int(image_size[1])


def class_names_from_map(path: str | Path, num_classes: int) -> list[str]:
    if not str(path):
        return ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]
    resolved = Path(path)
    if not resolved.exists():
        fallback = REPO_ROOT / "outputs" / "task5_baselines" / "deeplabv3plus" / resolved.name
        resolved = fallback if fallback.exists() else resolved
    names = ["background"] + [f"class_{idx}" for idx in range(1, num_classes)]
    if not resolved.exists() or not resolved.is_file():
        return names
    class_map = load_json(resolved)
    names[0] = class_map.get("background", {}).get("name", "background")
    for category in class_map.get("categories", []):
        semantic_id = int(category["semantic_id"])
        if 0 <= semantic_id < len(names):
            names[semantic_id] = category["name"]
    return names


def dry_run_dataset_info(dataset_name: str, dataset_config: dict[str, Any]) -> dict[str, Any]:
    dataset_yaml = load_yaml(dataset_config["data_yaml"])
    num_classes = int(dataset_yaml["num_classes"])
    summary_path = REPO_ROOT / "outputs" / "task5_baselines" / "summaries" / f"{dataset_name}_summary.json"
    train_images = None
    val_images = None
    if summary_path.exists():
        summary = load_json(summary_path)
        train_images = summary.get("train_images")
        val_images = summary.get("val_images")
    return {
        "num_classes": num_classes,
        "class_names": class_names_from_map(dataset_yaml.get("class_map", ""), num_classes),
        "train_images": train_images,
        "val_images": val_images,
    }


class SemanticSegmentationDataset:
    def __init__(
        self,
        dataset_name: str,
        dataset_yaml_path: str | Path,
        split: str,
        image_size: tuple[int, int],
        augment: bool,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_yaml_path = resolve_repo_path(dataset_yaml_path)
        self.dataset_yaml = load_yaml(self.dataset_yaml_path)
        self.root = resolve_dataset_root(dataset_name, self.dataset_yaml)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.num_classes = int(self.dataset_yaml["num_classes"])
        self.class_names = class_names_from_map(
            self.dataset_yaml.get("class_map", ""),
            self.num_classes,
        )

        image_key = f"{split}_images"
        mask_key = f"{split}_masks"
        self.image_dir = self.root / self.dataset_yaml[image_key]
        self.mask_dir = self.root / self.dataset_yaml[mask_key]
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {self.mask_dir}")

        self.samples = []
        for image_path in sorted(self.image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self.mask_dir / f"{image_path.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            self.samples.append((image_path, mask_path))
        if not self.samples:
            raise ValueError(f"No samples found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, str]:
        import torch

        image_path, mask_path = self.samples[index]
        target_height, target_width = self.image_size

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.BILINEAR)
        if mask.size != (target_width, target_height):
            mask = mask.resize((target_width, target_height), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)
        if self.augment and random.random() < 0.5:
            image_array = np.ascontiguousarray(image_array[:, ::-1, :])
            mask_array = np.ascontiguousarray(mask_array[:, ::-1])

        image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_array)).long()
        return image_tensor, mask_tensor, image_path.stem


def build_device(device_config: str | int) -> Any:
    import torch

    if str(device_config).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_config}")
    return torch.device("cpu")


def build_model(num_classes: int, config: dict[str, Any]) -> Any:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "Missing segmentation-models-pytorch. Install requirements.txt "
            "before training DeepLabV3+."
        ) from exc

    model_config = config["model"]
    encoder_weights = model_config.get("encoder_weights", "imagenet")
    if isinstance(encoder_weights, str) and encoder_weights.lower() in {"none", "null", "false"}:
        encoder_weights = None
    return smp.DeepLabV3Plus(
        encoder_name=model_config["encoder_name"],
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )


def confusion_from_batch(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    true_positive = np.diag(confusion).astype(np.float64)
    row_sum = confusion.sum(axis=1).astype(np.float64)
    col_sum = confusion.sum(axis=0).astype(np.float64)
    union = row_sum + col_sum - true_positive

    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=union > 0,
    )
    accuracy = np.divide(
        true_positive,
        row_sum,
        out=np.full_like(true_positive, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )

    valid_iou = iou[~np.isnan(iou)]
    foreground_iou = iou[1:][~np.isnan(iou[1:])]
    valid_accuracy = accuracy[~np.isnan(accuracy)]
    pixel_accuracy = float(true_positive.sum() / max(confusion.sum(), 1.0))

    per_class_iou = {
        class_names[idx] if idx < len(class_names) else f"class_{idx}": None
        if np.isnan(value)
        else float(value)
        for idx, value in enumerate(iou)
    }
    return {
        "miou": float(valid_iou.mean()) if valid_iou.size else 0.0,
        "foreground_miou": float(foreground_iou.mean()) if foreground_iou.size else 0.0,
        "pixel_accuracy": pixel_accuracy,
        "mean_accuracy": float(valid_accuracy.mean()) if valid_accuracy.size else 0.0,
        "per_class_iou": per_class_iou,
    }


def evaluate(
    model: Any,
    data_loader: Any,
    dataset: SemanticSegmentationDataset,
    device: Any,
    predictions_dir: Path,
    save_predictions: bool,
) -> dict[str, Any]:
    import torch

    model.eval()
    confusion = np.zeros((dataset.num_classes, dataset.num_classes), dtype=np.int64)
    started_at = time.perf_counter()
    prediction_count = 0

    if save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, masks, stems in data_loader:
            images = images.to(device)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()

            for idx, stem in enumerate(stems):
                prediction = predictions[idx]
                target = targets[idx]
                confusion += confusion_from_batch(prediction, target, dataset.num_classes)
                prediction_count += 1
                if save_predictions:
                    output = Image.fromarray(prediction.astype(np.uint16), mode="I;16")
                    output.save(predictions_dir / f"{stem}.png")

    metrics = metrics_from_confusion(confusion, dataset.class_names)
    metrics.update(
        {
            "prediction_count": prediction_count,
            "predictions_dir": relative_to_repo(predictions_dir) if save_predictions else None,
            "elapsed_s": time.perf_counter() - started_at,
        }
    )
    return metrics


def run_name(dataset_name: str, config: dict[str, Any], epochs: int, smoke: bool) -> str:
    suffix = config["model"].get("run_suffix", "deeplabv3plus")
    if smoke:
        return f"{dataset_name}_{suffix}_smoke"
    return f"{dataset_name}_{suffix}_e{epochs}"


def train_one_dataset(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset_name: str,
    dataset_config: dict[str, Any],
) -> dict[str, Any]:
    training = config["training"]
    epochs = 1 if args.smoke else int(args.epochs or training["epochs"])
    batch_size = int(args.batch_size or training["batch_size"])
    workers = int(args.workers if args.workers is not None else training["workers"])
    log_every = int(args.log_every if args.log_every is not None else training["log_every"])
    image_size = image_size_from_config(config)
    device_config = args.device if args.device is not None else training["device"]
    output_root = resolve_repo_path(config["task"]["output_root"])
    results_root = resolve_repo_path(config["task"]["results_root"])
    name = run_name(dataset_name, config, epochs, args.smoke)
    run_dir = results_root / name
    best_checkpoint = run_dir / "checkpoints" / "best.pt"
    last_checkpoint = run_dir / "checkpoints" / "last.pt"

    if args.dry_run:
        info = dry_run_dataset_info(dataset_name, dataset_config)
        print(
            f"[START] DeepLabV3+ dataset={dataset_name} epochs={epochs} batch={batch_size} "
            f"train_images={info['train_images']} val_images={info['val_images']} "
            f"classes={info['num_classes']} image_size={image_size[0]}x{image_size[1]} "
            f"device={device_config} output={relative_to_repo(run_dir)}",
            flush=True,
        )
        return {
            "dataset": dataset_name,
            "status": "dry_run",
            "epochs": epochs,
            "num_classes": info["num_classes"],
            "class_names": info["class_names"],
            "train_images": info["train_images"],
            "val_images": info["val_images"],
            "run_dir": relative_to_repo(run_dir),
        }

    train_dataset = SemanticSegmentationDataset(
        dataset_name=dataset_name,
        dataset_yaml_path=dataset_config["data_yaml"],
        split="train",
        image_size=image_size,
        augment=True,
    )
    val_dataset = SemanticSegmentationDataset(
        dataset_name=dataset_name,
        dataset_yaml_path=dataset_config["data_yaml"],
        split="val",
        image_size=image_size,
        augment=False,
    )

    print(
        f"[START] DeepLabV3+ dataset={dataset_name} epochs={epochs} batch={batch_size} "
        f"train_images={len(train_dataset)} val_images={len(val_dataset)} "
        f"classes={train_dataset.num_classes} image_size={image_size[0]}x{image_size[1]} "
        f"device={device_config} output={relative_to_repo(run_dir)}",
        flush=True,
    )

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

    import torch
    from torch.utils.data import DataLoader

    device = build_device(device_config)
    generator = torch.Generator()
    generator.manual_seed(int(config["task"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = build_model(num_classes=train_dataset.num_classes, config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    eval_every = max(1, int(training["eval_every"]))

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

        for batch_index, (images, masks, _) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            running_batches += 1
            if batch_index == 1 or batch_index % log_every == 0 or batch_index == len(train_loader):
                print(
                    f"[PROGRESS] {dataset_name} epoch={epoch}/{epochs} "
                    f"batch={batch_index}/{len(train_loader)} loss={loss_value:.4f}",
                    flush=True,
                )

        average_loss = running_loss / max(running_batches, 1)
        epoch_summary: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": average_loss,
            "elapsed_s": time.perf_counter() - epoch_started_at,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        should_eval = epoch == epochs or epoch % eval_every == 0 or epoch == 1
        if should_eval:
            predictions_dir = run_dir / "predictions" / f"val_epoch_{epoch:03d}"
            metrics = evaluate(
                model=model,
                data_loader=val_loader,
                dataset=val_dataset,
                device=device,
                predictions_dir=predictions_dir,
                save_predictions=should_eval,
            )
            epoch_summary["metrics"] = metrics
            current_metric = float(metrics["foreground_miou"])
            print(
                f"[EVAL] {dataset_name} epoch={epoch} "
                f"miou={metrics['miou']:.4f} foreground_miou={metrics['foreground_miou']:.4f} "
                f"pixel_acc={metrics['pixel_accuracy']:.4f}",
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
                        "class_names": train_dataset.class_names,
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
            "class_names": train_dataset.class_names,
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
        "best_metric": "foreground_miou",
        "best_metrics": best_metrics,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "num_classes": train_dataset.num_classes,
        "class_names": train_dataset.class_names,
        "elapsed_s": time.perf_counter() - started_at,
        "run_dir": relative_to_repo(run_dir),
        "best_checkpoint": relative_to_repo(best_checkpoint),
        "last_checkpoint": relative_to_repo(last_checkpoint),
        "epochs_summary": epoch_summaries,
    }
    print(
        f"[DONE] {dataset_name}: status={status} best_epoch={best_epoch} "
        f"best_foreground_miou={best_metric:.4f} elapsed={summary['elapsed_s'] / 60.0:.1f}min",
        flush=True,
    )
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config["task"]["seed"]))

    summaries = []
    output_root = resolve_repo_path(config["task"]["output_root"])
    for dataset_name, dataset_config in selected_datasets(config, args.datasets):
        summary = train_one_dataset(
            args=args,
            config=config,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )
        summaries.append(summary)
        if not args.dry_run:
            write_json(output_root / f"{dataset_name}_train_summary.json", summary)

    if not args.dry_run:
        write_json(output_root / "summary.json", summaries)
        write_csv(output_root / "metrics_summary.csv", summaries)
    print("[DONE] DeepLabV3+ training wrapper", flush=True)


if __name__ == "__main__":
    main()
