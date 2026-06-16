"""Run zero-shot SAM-family inference from Task 4 prompt manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as mask_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
FASTSAM_SRC_DIR = REPO_ROOT / "external" / "FastSAM"
if FASTSAM_SRC_DIR.exists() and str(FASTSAM_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(FASTSAM_SRC_DIR))
MOBILE_SAM_SRC_DIR = REPO_ROOT / "external" / "MobileSAM"
if MOBILE_SAM_SRC_DIR.exists() and str(MOBILE_SAM_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MOBILE_SAM_SRC_DIR))
EFFICIENT_SAM_SRC_DIR = REPO_ROOT / "external" / "EfficientSAM"
if EFFICIENT_SAM_SRC_DIR.exists() and str(EFFICIENT_SAM_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(EFFICIENT_SAM_SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_zero_shot_sam.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-mode",
        required=True,
        choices=["point", "box", "automatic"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress after this many records/images during inference.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    written = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
                written += 1
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(resolved)
    except Exception:
        print(
            f"[ERROR] incomplete output kept at {tmp_path}",
            flush=True,
        )
        raise

    print(f"[WRITE] wrote {written} records to {resolved}", flush=True)


def load_image_rgb(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def encode_mask(mask: Any) -> dict[str, Any]:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = np.squeeze(array)
    binary = (array > 0).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(binary))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "segmentation": {"size": [int(v) for v in rle["size"]], "counts": counts},
        "area": float(mask_utils.area(rle)),
        "bbox": [float(v) for v in mask_utils.toBbox(rle)],
    }


def select_limited_records(
    records: list[dict[str, Any]],
    max_images: int | None,
    max_instances: int | None,
) -> list[dict[str, Any]]:
    if max_images is None and max_instances is None:
        return records

    selected = []
    seen_images: OrderedDict[Any, None] = OrderedDict()
    for record in records:
        image_id = record["image_id"]
        if image_id not in seen_images:
            if max_images is not None and len(seen_images) >= max_images:
                continue
            seen_images[image_id] = None
        selected.append(record)
        if max_instances is not None and len(selected) >= max_instances:
            break
    return selected


def unique_image_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for record in records:
        images.setdefault(record["image_id"], record)
    return list(images.values())


class SamAdapter:
    """Adapter for original SAM from facebookresearch/segment-anything."""

    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        from segment_anything import (  # type: ignore[import-not-found]
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        checkpoint = model_config["checkpoint"]
        model_type = model_config["model_type"]
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.image_key: str | None = None

    def set_image(self, image: np.ndarray, image_path: str) -> None:
        if self.image_key != image_path:
            self.predictor.set_image(image)
            self.image_key = image_path

    def predict_point(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(record["point_prompt"]["points"], dtype=np.float32),
            point_labels=np.asarray(record["point_prompt"]["labels"], dtype=np.int32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_box(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(record["box_prompt"]["box_xyxy"], dtype=np.float32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_automatic(self, image: np.ndarray) -> list[dict[str, Any]]:
        return automatic_masks_to_predictions(self.mask_generator.generate(image))


class Sam2Adapter:
    """Adapter for Meta SAM2 image prediction."""

    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        from sam2.automatic_mask_generator import (  # type: ignore[import-not-found]
            SAM2AutomaticMaskGenerator,
        )
        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
        from sam2.sam2_image_predictor import (  # type: ignore[import-not-found]
            SAM2ImagePredictor,
        )

        model = build_sam2(
            model_config["model_cfg"],
            model_config["checkpoint"],
            device=device,
        )
        self.predictor = SAM2ImagePredictor(model)
        self.mask_generator = SAM2AutomaticMaskGenerator(model)
        self.image_key: str | None = None

    def set_image(self, image: np.ndarray, image_path: str) -> None:
        if self.image_key != image_path:
            self.predictor.set_image(image)
            self.image_key = image_path

    def predict_point(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(record["point_prompt"]["points"], dtype=np.float32),
            point_labels=np.asarray(record["point_prompt"]["labels"], dtype=np.int32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_box(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(record["box_prompt"]["box_xyxy"], dtype=np.float32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_automatic(self, image: np.ndarray) -> list[dict[str, Any]]:
        return automatic_masks_to_predictions(self.mask_generator.generate(image))


class FastSamAdapter:
    """Adapter for CASIA-LMC-Lab FastSAM."""

    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        from fastsam import FastSAM, FastSAMPrompt  # type: ignore[import-not-found]

        self.model = FastSAM(model_config["checkpoint"])
        self.prompt_cls = FastSAMPrompt
        self.device = device
        self.imgsz = int(model_config.get("imgsz", 1024))
        self.conf = float(model_config.get("conf", 0.4))
        self.iou = float(model_config.get("iou", 0.9))
        self.image_key: str | None = None
        self.prompt_process: Any | None = None

    def set_image(self, image: np.ndarray, image_path: str) -> None:
        if self.image_key == image_path:
            return
        everything_results = self.model(
            image_path,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
        )
        self.prompt_process = self.prompt_cls(
            image_path,
            everything_results,
            device=self.device,
        )
        self.image_key = image_path

    def predict_point(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.prompt_process is not None
        ann = self.prompt_process.point_prompt(
            points=record["point_prompt"]["points"],
            pointlabel=record["point_prompt"]["labels"],
        )
        return fastsam_annotations_to_predictions(ann)

    def predict_box(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        assert self.prompt_process is not None
        ann = self.prompt_process.box_prompt(
            bboxes=[record["box_prompt"]["box_xyxy"]],
        )
        return fastsam_annotations_to_predictions(ann)

    def predict_automatic(self, image: np.ndarray) -> list[dict[str, Any]]:
        assert self.prompt_process is not None
        ann = self.prompt_process.everything_prompt()
        return fastsam_annotations_to_predictions(ann)


class MobileSamAdapter:
    """Adapter for ChaoningZhang/MobileSAM."""

    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        from mobile_sam import (  # type: ignore[import-not-found]
            SamAutomaticMaskGenerator,
            SamPredictor,
            sam_model_registry,
        )

        checkpoint = model_config["checkpoint"]
        model_type = model_config.get("model_type", "vit_t")
        mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.predictor = SamPredictor(mobile_sam)
        self.mask_generator = SamAutomaticMaskGenerator(mobile_sam)
        self.image_key: str | None = None

    def set_image(self, image: np.ndarray, image_path: str) -> None:
        if self.image_key != image_path:
            self.predictor.set_image(image)
            self.image_key = image_path

    def predict_point(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            point_coords=np.asarray(record["point_prompt"]["points"], dtype=np.float32),
            point_labels=np.asarray(record["point_prompt"]["labels"], dtype=np.int32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_box(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(record["box_prompt"]["box_xyxy"], dtype=np.float32),
            multimask_output=True,
        )
        return masks_to_predictions(masks, scores)

    def predict_automatic(self, image: np.ndarray) -> list[dict[str, Any]]:
        return automatic_masks_to_predictions(self.mask_generator.generate(image))


class EfficientSamAdapter:
    """Adapter for yformer/EfficientSAM point, box, and grid automatic prompts."""

    VARIANTS = {
        "vitt": {"encoder_patch_embed_dim": 192, "encoder_num_heads": 3},
        "vits": {"encoder_patch_embed_dim": 384, "encoder_num_heads": 6},
    }

    def __init__(self, model_config: dict[str, Any], device: str) -> None:
        import torch
        from efficient_sam.efficient_sam import (  # type: ignore[import-not-found]
            build_efficient_sam,
        )

        variant = str(model_config.get("variant", "vitt"))
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported EfficientSAM variant: {variant}")

        self.device = torch.device(device)
        self.model = build_efficient_sam(
            checkpoint=model_config["checkpoint"],
            **self.VARIANTS[variant],
        )
        self.model.to(self.device)
        self.model.eval()
        self.image_key: str | None = None
        self.image_embeddings: Any | None = None
        self.input_h = 0
        self.input_w = 0
        self.automatic_points_per_side = int(model_config.get("automatic_points_per_side", 8))
        self.automatic_batch_queries = int(model_config.get("automatic_batch_queries", 64))
        self.automatic_multimask_output = bool(model_config.get("automatic_multimask_output", False))

    def set_image(self, image: np.ndarray, image_path: str) -> None:
        if self.image_key == image_path:
            return
        import torch

        image_array = np.ascontiguousarray(image)
        self.input_h, self.input_w = int(image_array.shape[0]), int(image_array.shape[1])
        tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.float32)
            / 255.0
        )
        self.image_embeddings = self.model.get_image_embeddings(tensor)
        self.image_key = image_path

    def _predict_queries(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool,
    ) -> list[dict[str, Any]]:
        import torch

        if self.image_embeddings is None:
            raise RuntimeError("set_image must be called before EfficientSAM prediction.")

        point_tensor = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        masks, scores = self.model.predict_masks(
            self.image_embeddings,
            point_tensor.unsqueeze(0),
            label_tensor.unsqueeze(0),
            multimask_output=multimask_output,
            input_h=self.input_h,
            input_w=self.input_w,
            output_h=self.input_h,
            output_w=self.input_w,
        )
        sorted_ids = torch.argsort(scores, dim=-1, descending=True)
        scores = torch.take_along_dim(scores, sorted_ids, dim=2)
        masks = torch.take_along_dim(masks, sorted_ids[..., None, None], dim=2)

        masks_np = (masks[0] >= 0).detach().cpu().numpy()
        scores_np = scores[0].detach().cpu().numpy()
        predictions: list[dict[str, Any]] = []
        for query_index in range(masks_np.shape[0]):
            for candidate_index in range(masks_np.shape[1]):
                encoded = encode_mask(masks_np[query_index, candidate_index])
                encoded["score"] = float(scores_np[query_index, candidate_index])
                encoded["query_index"] = query_index
                encoded["candidate_index"] = candidate_index
                predictions.append(encoded)
        return predictions

    def predict_point(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        points = np.asarray(record["point_prompt"]["points"], dtype=np.float32)[None, :, :]
        labels = np.asarray(record["point_prompt"]["labels"], dtype=np.float32)[None, :]
        return self._predict_queries(points, labels, multimask_output=True)

    def predict_box(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        x1, y1, x2, y2 = record["box_prompt"]["box_xyxy"]
        points = np.asarray([[[x1, y1], [x2, y2]]], dtype=np.float32)
        labels = np.asarray([[2, 3]], dtype=np.float32)
        return self._predict_queries(points, labels, multimask_output=True)

    def predict_automatic(self, image: np.ndarray) -> list[dict[str, Any]]:
        del image
        grid_points, grid_labels = self._automatic_grid_prompts()
        predictions: list[dict[str, Any]] = []
        for start in range(0, len(grid_points), self.automatic_batch_queries):
            stop = start + self.automatic_batch_queries
            batch_predictions = self._predict_queries(
                grid_points[start:stop],
                grid_labels[start:stop],
                multimask_output=self.automatic_multimask_output,
            )
            for prediction in batch_predictions:
                prediction["query_index"] = int(prediction["query_index"]) + start
            predictions.extend(batch_predictions)
        return predictions

    def _automatic_grid_prompts(self) -> tuple[np.ndarray, np.ndarray]:
        points_per_side = max(1, self.automatic_points_per_side)
        xs = np.linspace(
            0.5 * self.input_w / points_per_side,
            self.input_w - 0.5 * self.input_w / points_per_side,
            points_per_side,
            dtype=np.float32,
        )
        ys = np.linspace(
            0.5 * self.input_h / points_per_side,
            self.input_h - 0.5 * self.input_h / points_per_side,
            points_per_side,
            dtype=np.float32,
        )
        query_points = np.asarray([[[x, y]] for y in ys for x in xs], dtype=np.float32)
        query_labels = np.ones((query_points.shape[0], 1), dtype=np.float32)
        return query_points, query_labels


def masks_to_predictions(masks: Any, scores: Any) -> list[dict[str, Any]]:
    masks_np = np.asarray(masks)
    scores_np = np.asarray(scores).reshape(-1)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]

    predictions = []
    for idx, mask in enumerate(masks_np):
        encoded = encode_mask(mask)
        encoded["score"] = float(scores_np[idx]) if idx < len(scores_np) else None
        encoded["candidate_index"] = idx
        predictions.append(encoded)
    return predictions


def automatic_masks_to_predictions(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for idx, ann in enumerate(annotations):
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, dict):
            pred = {
                "segmentation": segmentation,
                "area": float(ann.get("area", 0.0)),
                "bbox": [float(v) for v in ann.get("bbox", [])],
            }
            counts = pred["segmentation"].get("counts")
            if isinstance(counts, bytes):
                pred["segmentation"]["counts"] = counts.decode("ascii")
        else:
            pred = encode_mask(segmentation)
        pred["score"] = ann.get("predicted_iou", ann.get("score"))
        pred["stability_score"] = ann.get("stability_score")
        pred["candidate_index"] = idx
        predictions.append(pred)
    return predictions


def fastsam_annotations_to_predictions(annotations: Any) -> list[dict[str, Any]]:
    if hasattr(annotations, "cpu"):
        annotations = annotations.cpu().numpy()
    if isinstance(annotations, list):
        masks = annotations
    else:
        masks = np.asarray(annotations)
        if masks.ndim == 2:
            masks = masks[None, :, :]
    return masks_to_predictions(masks, np.ones(len(masks), dtype=np.float32))


def build_adapter(model_config: dict[str, Any], device: str) -> Any:
    family = model_config["family"]
    if family == "sam":
        return SamAdapter(model_config, device)
    if family == "sam2":
        return Sam2Adapter(model_config, device)
    if family == "fastsam":
        return FastSamAdapter(model_config, device)
    if family == "mobile_sam":
        return MobileSamAdapter(model_config, device)
    if family == "efficient_sam":
        return EfficientSamAdapter(model_config, device)
    raise ValueError(f"Unsupported model family: {family}")


def output_path_for(
    config: dict[str, Any],
    dataset: str,
    model: str,
    prompt_mode: str,
) -> Path:
    return (
        Path(config["task"]["output_root"])
        / dataset
        / model
        / f"{prompt_mode}_predictions.jsonl"
    )


def prompt_manifest_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["task"]["prompt_manifest_dir"]) / f"{dataset}_instances.jsonl"


def dry_run(
    config: dict[str, Any],
    dataset: str,
    model: str,
    prompt_mode: str,
    records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    model_config = config["models"][model]
    dataset_config = config["datasets"][dataset]
    print("[DRY RUN]")
    print(f"dataset: {dataset}")
    print(f"dataset_root: {dataset_config.get('root')}")
    print(f"model: {model} ({model_config['family']})")
    print(f"prompt_mode: {prompt_mode}")
    print(f"records: {len(records)}")
    print(f"output: {out_path}")
    if records:
        sample = records[0]
        print(f"sample_image: {sample['image_path']}")
        print(f"sample_annotation_id: {sample['annotation_id']}")


def run_prompted(
    adapter: Any,
    records: list[dict[str, Any]],
    prompt_mode: str,
    run_metadata: dict[str, Any],
    log_every: int,
) -> Iterable[dict[str, Any]]:
    import torch

    total = len(records)
    started_at = time.perf_counter()
    for index, record in enumerate(records, start=1):
        image = load_image_rgb(record["image_path"])
        start = time.perf_counter()
        with torch.inference_mode():
            adapter.set_image(image, record["image_path"])
            if prompt_mode == "point":
                predictions = adapter.predict_point(record)
            elif prompt_mode == "box":
                predictions = adapter.predict_box(record)
            else:
                raise ValueError(f"Prompted run received {prompt_mode}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if (
            index == 1
            or index == total
            or (log_every > 0 and index % log_every == 0)
        ):
            total_elapsed = time.perf_counter() - started_at
            print(
                f"[PROGRESS] {run_metadata['dataset']} "
                f"{run_metadata['model']} {prompt_mode}: "
                f"{index}/{total} records, last={elapsed_ms:.1f}ms, "
                f"elapsed={total_elapsed:.1f}s",
                flush=True,
            )
        yield {
            **run_metadata,
            "record_index": index,
            "image_id": record["image_id"],
            "annotation_id": record["annotation_id"],
            "category_id": record["category_id"],
            "category_name": record["category_name"],
            "file_name": record["file_name"],
            "image_path": record["image_path"],
            "prompt": record[f"{prompt_mode}_prompt"],
            "elapsed_ms": elapsed_ms,
            "predictions": predictions,
        }


def run_automatic(
    adapter: Any,
    records: list[dict[str, Any]],
    run_metadata: dict[str, Any],
    log_every: int,
) -> Iterable[dict[str, Any]]:
    import torch

    image_records = unique_image_records(records)
    total = len(image_records)
    started_at = time.perf_counter()
    for index, record in enumerate(image_records, start=1):
        image = load_image_rgb(record["image_path"])
        start = time.perf_counter()
        with torch.inference_mode():
            adapter.set_image(image, record["image_path"])
            predictions = adapter.predict_automatic(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if (
            index == 1
            or index == total
            or (log_every > 0 and index % log_every == 0)
        ):
            total_elapsed = time.perf_counter() - started_at
            print(
                f"[PROGRESS] {run_metadata['dataset']} "
                f"{run_metadata['model']} automatic: "
                f"{index}/{total} images, last={elapsed_ms:.1f}ms, "
                f"elapsed={total_elapsed:.1f}s",
                flush=True,
            )
        yield {
            **run_metadata,
            "record_index": index,
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "image_path": record["image_path"],
            "elapsed_ms": elapsed_ms,
            "predictions": predictions,
        }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = args.device or config["runtime"].get("device", "cuda")

    if args.dataset not in config["datasets"]:
        raise KeyError(f"Unknown dataset: {args.dataset}")
    if args.model not in config["models"]:
        raise KeyError(f"Unknown model: {args.model}")
    if not config["models"][args.model].get("enabled", False):
        raise ValueError(f"Model is disabled in config: {args.model}")

    manifest_path = prompt_manifest_path(config, args.dataset)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing prompt manifest: {manifest_path}. "
            "Run scripts/benchmarks/build_prompt_manifest.py first."
        )

    records = load_jsonl(manifest_path)
    records = select_limited_records(records, args.max_images, args.max_instances)
    out_path = output_path_for(config, args.dataset, args.model, args.prompt_mode)
    total_units = (
        len(unique_image_records(records))
        if args.prompt_mode == "automatic"
        else len(records)
    )

    if args.dry_run:
        dry_run(config, args.dataset, args.model, args.prompt_mode, records, out_path)
        return

    run_metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "model_family": config["models"][args.model]["family"],
        "prompt_mode": args.prompt_mode,
        "device": device,
        "checkpoint": config["models"][args.model].get("checkpoint"),
    }
    print(
        f"[START] dataset={args.dataset} model={args.model} "
        f"prompt_mode={args.prompt_mode} device={device} records={len(records)} "
        f"run_units={total_units} output={out_path}",
        flush=True,
    )
    load_started_at = time.perf_counter()
    print(f"[LOAD] {args.model}: loading checkpoint/model", flush=True)
    adapter = build_adapter(config["models"][args.model], device)
    print(
        f"[LOAD] {args.model}: ready in {time.perf_counter() - load_started_at:.1f}s",
        flush=True,
    )
    if args.prompt_mode == "automatic":
        output_records = run_automatic(adapter, records, run_metadata, args.log_every)
    else:
        output_records = run_prompted(
            adapter,
            records,
            args.prompt_mode,
            run_metadata,
            args.log_every,
        )

    write_jsonl(out_path, output_records)
    print(f"[OK] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
