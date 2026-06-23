"""Measure Task 7 segmentation inference speed on GPU and CPU.

The benchmark loads the same dataset/model metadata used by Tasks 4-6 and
reports compact timing summaries. It does not write prediction masks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import statistics
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = REPO_ROOT / "scripts" / "baselines"
FASTSAM_SRC_DIR = REPO_ROOT / "external" / "FastSAM"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

from cogar_seg.config import load_config as load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task7_inference_speed.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompt-modes", nargs="*", default=None)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--automatic-max-images", type=int, default=None)
    parser.add_argument("--cpu-max-images", type=int, default=None)
    parser.add_argument("--cpu-automatic-max-images", type=int, default=None)
    parser.add_argument("--warmup-images", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
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


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, data: Any) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def selected_names(all_names: list[str], selected: list[str] | None) -> list[str]:
    if selected is None:
        return all_names
    selected_set = set(selected)
    return [name for name in all_names if name in selected_set]


def sanitize_device(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


def output_path_for(
    output_root: Path,
    device: str,
    dataset: str,
    model: str,
    prompt_mode: str,
) -> Path:
    return (
        output_root
        / sanitize_device(device)
        / dataset
        / model
        / f"{prompt_mode}_speed.json"
    )


def prompt_manifest_path(task4_config: dict[str, Any], dataset: str) -> Path:
    return (
        resolve_repo_path(task4_config["task"]["prompt_manifest_dir"])
        / f"{dataset}_instances.jsonl"
    )


def load_dataset_records(task4_config: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    manifest_path = prompt_manifest_path(task4_config, dataset)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing prompt manifest: {manifest_path}. "
            "Run scripts/benchmarks/build_prompt_manifest.py first."
        )
    return load_jsonl(manifest_path)


def select_one_prompt_per_image(records: list[dict[str, Any]], max_images: int) -> list[dict[str, Any]]:
    return unique_image_records(records)[:max_images]


def unique_image_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images: OrderedDict[Any, dict[str, Any]] = OrderedDict()
    for record in records:
        images.setdefault(record["image_id"], record)
    return list(images.values())


def purge_modules(module_roots: list[str]) -> None:
    for module_name in list(sys.modules):
        if any(module_name == root or module_name.startswith(f"{root}.") for root in module_roots):
            del sys.modules[module_name]


def module_file(module_name: str) -> str:
    module = sys.modules.get(module_name)
    if module is None:
        return ""
    return str(getattr(module, "__file__", "") or "")


def path_contains(path: str, root: Path) -> bool:
    if not path:
        return False
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def remove_path_from_sys_path(path: Path) -> None:
    resolved = path.resolve()
    sys.path[:] = [
        item
        for item in sys.path
        if not item or Path(item).resolve() != resolved
    ]


def prepare_fastsam_imports() -> None:
    if not FASTSAM_SRC_DIR.exists():
        return
    fastsam_path = str(FASTSAM_SRC_DIR.resolve())
    if fastsam_path not in [str(Path(item).resolve()) for item in sys.path if item]:
        sys.path.insert(0, fastsam_path)
    ultralytics_file = module_file("ultralytics")
    if ultralytics_file and not path_contains(ultralytics_file, FASTSAM_SRC_DIR):
        purge_modules(["ultralytics"])


def prepare_official_ultralytics_import() -> None:
    remove_path_from_sys_path(FASTSAM_SRC_DIR)
    ultralytics_file = module_file("ultralytics")
    if ultralytics_file and path_contains(ultralytics_file, FASTSAM_SRC_DIR):
        purge_modules(["ultralytics"])


def selected_image_count(
    device: str,
    prompt_mode: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    sampling = config["sampling"]
    is_cpu = device.lower() == "cpu"
    if prompt_mode == "automatic":
        if is_cpu:
            return int(
                args.cpu_automatic_max_images
                or sampling.get("cpu_automatic_max_images_per_dataset")
                or 1
            )
        return int(
            args.automatic_max_images
            or sampling.get("automatic_max_images_per_dataset")
            or sampling["max_images_per_dataset"]
        )
    if is_cpu:
        return int(args.cpu_max_images or sampling.get("cpu_max_images_per_dataset") or 10)
    return int(args.max_images or sampling["max_images_per_dataset"])


def load_rgb_array(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def preload_samples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for record in records:
        samples.append(
            {
                "record": record,
                "image": load_rgb_array(record["image_path"]),
                "image_path": record["image_path"],
                "image_id": int(record["image_id"]),
            }
        )
    return samples


def get_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Task 7 speed benchmarking requires PyTorch.") from exc
    return torch


def normalize_device(device: str) -> str:
    if device.lower() == "cpu":
        return "cpu"
    if device in {"cuda", "cuda:0", "0"}:
        return "cuda:0"
    if device.startswith("cuda:"):
        return device
    return device


def torch_device(device: str) -> Any:
    torch = get_torch()
    normalized = normalize_device(device)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(normalized)


def synchronize(device: str) -> None:
    torch = get_torch()
    normalized = normalize_device(device)
    if normalized.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(normalized))


def empty_cuda_cache() -> None:
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_summary(latencies_ms: list[float]) -> dict[str, float]:
    total_s = float(sum(latencies_ms) / 1000.0)
    count = len(latencies_ms)
    return {
        "timed_units": count,
        "total_timed_s": total_s,
        "fps": float(count / total_s) if total_s > 0 else 0.0,
        "latency_mean_ms": float(statistics.mean(latencies_ms)) if latencies_ms else 0.0,
        "latency_median_ms": float(statistics.median(latencies_ms)) if latencies_ms else 0.0,
        "latency_p95_ms": percentile(latencies_ms, 95),
        "latency_min_ms": float(min(latencies_ms)) if latencies_ms else 0.0,
        "latency_max_ms": float(max(latencies_ms)) if latencies_ms else 0.0,
    }


def environment_summary(device: str) -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import psutil

        env["cpu_count_logical"] = psutil.cpu_count(logical=True)
        env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        env["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass
    try:
        torch = get_torch()

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if normalize_device(device).startswith("cuda") and torch.cuda.is_available():
            dev = torch.device(normalize_device(device))
            env["cuda_device"] = torch.cuda.get_device_name(dev)
            props = torch.cuda.get_device_properties(dev)
            env["cuda_total_memory_gb"] = round(props.total_memory / 1e9, 2)
    except ImportError:
        pass
    return env


def run_timed_loop(
    runner: Callable[[dict[str, Any]], Any],
    samples: list[dict[str, Any]],
    device: str,
    label: str,
    warmup_count: int,
    log_every: int,
) -> dict[str, Any]:
    torch = get_torch()
    warmup_samples = samples[: min(warmup_count, len(samples))]
    timed_samples = samples[len(warmup_samples) :]
    if not timed_samples:
        raise ValueError(f"{label}: no timed samples remain after warmup")
    print(f"[WARMUP] {label}: {len(warmup_samples)} images", flush=True)
    with torch.inference_mode():
        for sample in warmup_samples:
            runner(sample)
    synchronize(device)

    latencies_ms = []
    output_counts = []
    started_at = time.perf_counter()
    print(f"[TIMING] {label}: {len(timed_samples)} images", flush=True)
    with torch.inference_mode():
        for index, sample in enumerate(timed_samples, start=1):
            synchronize(device)
            start = time.perf_counter()
            output = runner(sample)
            synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            if isinstance(output, (list, tuple)):
                output_counts.append(len(output))
            elif isinstance(output, dict):
                output_counts.append(len(output))
            else:
                output_counts.append(1 if output is not None else 0)
            if (
                index == 1
                or index == len(timed_samples)
                or (log_every > 0 and index % log_every == 0)
            ):
                print(
                    f"[PROGRESS] {label}: {index}/{len(timed_samples)} images, "
                    f"last={elapsed_ms:.1f}ms, elapsed={time.perf_counter() - started_at:.1f}s",
                    flush=True,
                )

    summary = timing_summary(latencies_ms)
    summary["warmup_units"] = len(warmup_samples)
    summary["output_count_mean"] = float(statistics.mean(output_counts)) if output_counts else 0.0
    summary["output_count_min"] = int(min(output_counts)) if output_counts else 0
    summary["output_count_max"] = int(max(output_counts)) if output_counts else 0
    return summary


def build_zero_shot_runner(
    model_name: str,
    prompt_mode: str,
    model_config: dict[str, Any],
    device: str,
) -> Callable[[dict[str, Any]], Any]:
    if model_config["family"] == "fastsam":
        prepare_fastsam_imports()
    from run_zero_shot_sam import build_adapter

    adapter = build_adapter(model_config, normalize_device(device))

    def runner(sample: dict[str, Any]) -> Any:
        record = sample["record"]
        image = sample["image"]
        image_path = sample["image_path"]
        adapter.set_image(image, image_path)
        if prompt_mode == "point":
            return adapter.predict_point(record)
        if prompt_mode == "box":
            return adapter.predict_box(record)
        if prompt_mode == "automatic":
            return adapter.predict_automatic(image)
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    return runner


def resolve_summary_record(summary_path: str | Path, dataset_name: str) -> dict[str, Any]:
    records = load_json(summary_path)
    for record in records:
        if record["dataset"] == dataset_name:
            return record
    raise KeyError(f"Missing {dataset_name} in {summary_path}")


def resolve_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.is_absolute():
        parts = list(candidate.parts)
        if REPO_ROOT.name in parts:
            index = parts.index(REPO_ROOT.name)
            fallback = REPO_ROOT.joinpath(*parts[index + 1 :])
            if fallback.exists():
                return fallback
    fallback = resolve_repo_path(candidate)
    if fallback.exists():
        return fallback
    return candidate


def ultralytics_device_arg(device: str) -> str | int:
    normalized = normalize_device(device)
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda:"):
        return int(normalized.split(":", 1)[1])
    return 0


def build_yolo_runner(
    dataset_name: str,
    config: dict[str, Any],
    device: str,
) -> Callable[[dict[str, Any]], Any]:
    prepare_official_ultralytics_import()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Missing ultralytics. Install requirements.txt.") from exc
    print(f"[LOAD] yolo8_seg/{dataset_name}: ultralytics={module_file('ultralytics')}", flush=True)

    baseline_config = config["baselines"]["yolo8_seg"]
    record = resolve_summary_record(baseline_config["summary"], dataset_name)
    weight = resolve_artifact_path(record["best_weight"])
    if not weight.exists():
        raise FileNotFoundError(f"Missing YOLOv8-seg weight: {weight}")
    model = YOLO(str(weight))
    image_size = int(baseline_config["image_size"])
    confidence = float(baseline_config.get("confidence", 0.001))
    iou = float(baseline_config.get("iou", 0.7))
    yolo_device = ultralytics_device_arg(device)

    def runner(sample: dict[str, Any]) -> Any:
        return model.predict(
            source=sample["image"],
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            device=yolo_device,
            retina_masks=True,
            verbose=False,
        )

    return runner


def build_mask_rcnn_runner(
    dataset_name: str,
    config: dict[str, Any],
    device: str,
) -> Callable[[dict[str, Any]], Any]:
    import torch
    from train_mask_rcnn import build_model

    baseline_config = config["baselines"]["mask_rcnn"]
    task5_config = copy.deepcopy(load_yaml(baseline_config["config"]))
    task5_config["model"]["weights"] = "none"
    task5_config["model"]["weights_backbone"] = "none"
    record = resolve_summary_record(baseline_config["summary"], dataset_name)
    checkpoint_path = resolve_artifact_path(record["best_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Mask R-CNN checkpoint: {checkpoint_path}")

    device_obj = torch_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    category_id_to_label = checkpoint.get("category_id_to_label") or {}
    num_classes = len(category_id_to_label) + 1
    model = build_model(num_classes=num_classes, config=task5_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device_obj)
    model.eval()

    def runner(sample: dict[str, Any]) -> Any:
        image = sample["image"].astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(device_obj)
        return model([tensor])

    return runner


def image_size_from_config(config: dict[str, Any]) -> tuple[int, int]:
    image_size = config["training"]["image_size"]
    if isinstance(image_size, int):
        return image_size, image_size
    return int(image_size[0]), int(image_size[1])


def deeplab_num_classes(dataset_name: str) -> int:
    dataset_yaml = load_yaml(REPO_ROOT / "outputs" / "task5_baselines" / "deeplabv3plus" / f"{dataset_name}.yaml")
    return int(dataset_yaml["num_classes"])


def build_deeplab_runner(
    dataset_name: str,
    config: dict[str, Any],
    device: str,
) -> Callable[[dict[str, Any]], Any]:
    import torch
    from train_deeplabv3plus import build_model

    baseline_config = config["baselines"]["deeplabv3plus"]
    task5_config = copy.deepcopy(load_yaml(baseline_config["config"]))
    task5_config["model"]["encoder_weights"] = "none"
    record = resolve_summary_record(baseline_config["summary"], dataset_name)
    checkpoint_path = resolve_artifact_path(record["best_checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DeepLabV3+ checkpoint: {checkpoint_path}")

    device_obj = torch_device(device)
    num_classes = deeplab_num_classes(dataset_name)
    model = build_model(num_classes=num_classes, config=task5_config)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(checkpoint["model"])
    model.to(device_obj)
    model.eval()
    target_height, target_width = image_size_from_config(task5_config)

    def runner(sample: dict[str, Any]) -> Any:
        image = Image.fromarray(sample["image"])
        image = image.resize((target_width, target_height), Image.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - IMAGENET_MEAN) / IMAGENET_STD
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image_array))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device_obj)
        )
        logits = model(tensor)
        return torch.argmax(logits, dim=1)

    return runner


def build_baseline_runner(
    model_name: str,
    dataset_name: str,
    config: dict[str, Any],
    device: str,
) -> Callable[[dict[str, Any]], Any]:
    if model_name == "yolo8_seg":
        return build_yolo_runner(dataset_name, config, device)
    if model_name == "mask_rcnn":
        return build_mask_rcnn_runner(dataset_name, config, device)
    if model_name == "deeplabv3plus":
        return build_deeplab_runner(dataset_name, config, device)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def model_family(config: dict[str, Any], task4_config: dict[str, Any], model_name: str) -> str:
    if model_name in task4_config["models"]:
        return "zero_shot"
    if model_name in config["baselines"]:
        return "baseline"
    raise ValueError(f"Unsupported model: {model_name}")


def planned_models(config: dict[str, Any], task4_config: dict[str, Any], selected: list[str] | None) -> list[str]:
    zero_shot = [name for name in config["zero_shot"]["models"] if task4_config["models"][name].get("enabled", False)]
    baselines = [name for name, item in config["baselines"].items() if item.get("enabled", False)]
    return selected_names(zero_shot + baselines, selected)


def planned_prompt_modes(
    config: dict[str, Any],
    task4_config: dict[str, Any],
    selected: list[str] | None,
) -> list[str]:
    modes = [
        mode
        for mode in config["zero_shot"]["prompt_modes"]
        if task4_config["prompt_modes"][mode].get("enabled", False)
    ]
    return selected_names(modes, selected)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "model_family",
        "prompt_mode",
        "device",
        "status",
        "sample_images",
        "warmup_units",
        "timed_units",
        "fps",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_min_ms",
        "latency_max_ms",
        "output_count_mean",
        "elapsed_wall_s",
        "python",
        "platform",
        "processor",
        "cpu_count_logical",
        "cpu_count_physical",
        "memory_gb",
        "torch",
        "cuda_device",
        "cuda_total_memory_gb",
        "metrics_file",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def compact_row(summary: dict[str, Any]) -> dict[str, Any]:
    timing = summary.get("timing", {})
    environment = summary.get("environment", {})
    return {
        "dataset": summary.get("dataset"),
        "model": summary.get("model"),
        "model_family": summary.get("model_family"),
        "prompt_mode": summary.get("prompt_mode"),
        "device": summary.get("device"),
        "status": summary.get("status"),
        "sample_images": summary.get("sample_images"),
        "warmup_units": timing.get("warmup_units"),
        "timed_units": timing.get("timed_units"),
        "fps": timing.get("fps"),
        "latency_mean_ms": timing.get("latency_mean_ms"),
        "latency_median_ms": timing.get("latency_median_ms"),
        "latency_p95_ms": timing.get("latency_p95_ms"),
        "latency_min_ms": timing.get("latency_min_ms"),
        "latency_max_ms": timing.get("latency_max_ms"),
        "output_count_mean": timing.get("output_count_mean"),
        "elapsed_wall_s": summary.get("elapsed_wall_s"),
        "python": environment.get("python"),
        "platform": environment.get("platform"),
        "processor": environment.get("processor"),
        "cpu_count_logical": environment.get("cpu_count_logical"),
        "cpu_count_physical": environment.get("cpu_count_physical"),
        "memory_gb": environment.get("memory_gb"),
        "torch": environment.get("torch"),
        "cuda_device": environment.get("cuda_device"),
        "cuda_total_memory_gb": environment.get("cuda_total_memory_gb"),
        "metrics_file": summary.get("metrics_file"),
        "error": summary.get("error"),
    }


def run_one(
    config: dict[str, Any],
    task4_config: dict[str, Any],
    dataset: str,
    model_name: str,
    prompt_mode: str,
    device: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_root = resolve_repo_path(config["task"]["output_root"])
    out_path = output_path_for(output_root, device, dataset, model_name, prompt_mode)
    if out_path.exists() and not args.rerun_complete:
        print(f"[SKIP] existing speed metrics {relative_to_repo(out_path)}", flush=True)
        return load_json(out_path)

    count = selected_image_count(device, prompt_mode, config, args)
    warmup_count = int(args.warmup_images or config["sampling"].get("warmup_images", 0))
    selected_records = select_one_prompt_per_image(records, count + warmup_count)
    label = f"{dataset}/{model_name}/{prompt_mode}/{device}"
    dry_summary = {
        "dataset": dataset,
        "model": model_name,
        "model_family": model_family(config, task4_config, model_name),
        "prompt_mode": prompt_mode,
        "device": device,
        "status": "dry_run",
        "sample_images": max(0, len(selected_records) - min(warmup_count, len(selected_records))),
        "selected_images_including_warmup": len(selected_records),
        "requested_timed_images": count,
        "metrics_file": relative_to_repo(out_path),
    }
    if args.dry_run:
        print(
            f"[DRY RUN] {label}: timed_images={dry_summary['sample_images']} "
            f"warmup_images={min(warmup_count, len(selected_records))} "
            f"selected_images={len(selected_records)} output={relative_to_repo(out_path)}",
            flush=True,
        )
        return dry_summary

    started_at = time.perf_counter()
    print(f"[START] {label}: loading {len(selected_records)} images", flush=True)
    samples = preload_samples(selected_records)
    print(f"[LOAD] {label}: building model", flush=True)
    if model_name in task4_config["models"]:
        runner = build_zero_shot_runner(
            model_name=model_name,
            prompt_mode=prompt_mode,
            model_config=task4_config["models"][model_name],
            device=device,
        )
    else:
        runner = build_baseline_runner(model_name, dataset, config, device)

    log_every = int(args.log_every or config["runtime"].get("log_every", 5))
    timing = run_timed_loop(runner, samples, device, label, warmup_count, log_every)
    elapsed_wall_s = time.perf_counter() - started_at
    summary = {
        "dataset": dataset,
        "model": model_name,
        "model_family": model_family(config, task4_config, model_name),
        "prompt_mode": prompt_mode,
        "device": device,
        "status": "ok",
        "sample_images": timing["timed_units"],
        "selected_images_including_warmup": len(samples),
        "requested_timed_images": count,
        "timing": timing,
        "environment": environment_summary(device),
        "elapsed_wall_s": elapsed_wall_s,
        "metrics_file": relative_to_repo(out_path),
    }
    write_json(out_path, summary)
    print(
        f"[DONE] {label}: fps={timing['fps']:.3f} "
        f"mean={timing['latency_mean_ms']:.1f}ms output={relative_to_repo(out_path)}",
        flush=True,
    )
    empty_cuda_cache()
    return summary


def run_or_error(
    config: dict[str, Any],
    task4_config: dict[str, Any],
    dataset: str,
    model_name: str,
    prompt_mode: str,
    device: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        return run_one(config, task4_config, dataset, model_name, prompt_mode, device, records, args)
    except Exception as exc:
        if args.fail_fast:
            raise
        output_root = resolve_repo_path(config["task"]["output_root"])
        out_path = output_path_for(output_root, device, dataset, model_name, prompt_mode)
        summary = {
            "dataset": dataset,
            "model": model_name,
            "model_family": model_family(config, task4_config, model_name),
            "prompt_mode": prompt_mode,
            "device": device,
            "status": "error",
            "sample_images": 0,
            "error": repr(exc),
            "metrics_file": relative_to_repo(out_path),
        }
        if not args.dry_run:
            write_json(out_path, summary)
        print(f"[ERROR] {dataset}/{model_name}/{prompt_mode}/{device}: {exc!r}", flush=True)
        empty_cuda_cache()
        return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task4_config = load_yaml(config["zero_shot"]["config"])

    datasets = selected_names(
        [
            name
            for name, dataset_config in task4_config["datasets"].items()
            if dataset_config.get("enabled", False)
        ],
        args.datasets,
    )
    models = planned_models(config, task4_config, args.models)
    prompt_modes = planned_prompt_modes(config, task4_config, args.prompt_modes)
    devices = args.devices or list(config["runtime"].get("devices", ["cuda", "cpu"]))

    print(
        f"[PLAN] datasets={datasets} models={models} prompt_modes={prompt_modes} devices={devices}",
        flush=True,
    )

    rows = []
    records_by_dataset = {dataset: load_dataset_records(task4_config, dataset) for dataset in datasets}
    for device in devices:
        for dataset in datasets:
            records = records_by_dataset[dataset]
            for model_name in models:
                if model_name in task4_config["models"]:
                    for prompt_mode in prompt_modes:
                        summary = run_or_error(
                            config,
                            task4_config,
                            dataset,
                            model_name,
                            prompt_mode,
                            device,
                            records,
                            args,
                        )
                        rows.append(compact_row(summary))
                else:
                    summary = run_or_error(
                        config,
                        task4_config,
                        dataset,
                        model_name,
                        "inference",
                        device,
                        records,
                        args,
                    )
                    rows.append(compact_row(summary))

    if not args.dry_run:
        output_root = resolve_repo_path(config["task"]["output_root"])
        write_json(output_root / "summary.json", rows)
        write_summary_csv(output_root / "summary.csv", rows)
        print(f"[DONE] wrote {relative_to_repo(output_root / 'summary.csv')}", flush=True)
    print("[DONE] Task 7 inference-speed benchmark", flush=True)


if __name__ == "__main__":
    main()
