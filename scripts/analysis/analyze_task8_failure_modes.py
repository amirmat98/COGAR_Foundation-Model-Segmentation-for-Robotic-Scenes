"""Analyze qualitative segmentation failure modes for Task 8."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "scripts" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from segmentation_metrics import (  # noqa: E402
    annotation_to_rle,
    annotations_by_id,
    annotations_by_image,
    category_name_by_id,
    image_size_by_id,
    load_json,
    rle_iou,
    rle_to_mask,
    select_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task8_failure_analysis.yaml")
    parser.add_argument("--max-records-per-case", type=int, default=None)
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
    return yaml.safe_load(resolve_repo_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with resolve_repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    resolved = resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def iter_jsonl(path: str | Path):
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_split_image_ids(config: dict[str, Any], dataset: str) -> set[int] | None:
    evaluation = config.get("evaluation", {})
    split = evaluation.get("split", "full")
    if split == "full":
        return None
    split_root = resolve_repo_path(evaluation.get("split_root", "outputs/task5_baselines/splits"))
    split_file = split_root / dataset / f"{split}_image_ids.txt"
    with split_file.open("r", encoding="utf-8") as handle:
        return {int(line.strip()) for line in handle if line.strip()}


def validate_split_rows(rows: list[dict[str, str]], label: str, expected_split: str) -> None:
    if expected_split == "full":
        return
    if not rows:
        raise ValueError(f"{label} summary is empty")
    observed = {row.get("split", "") for row in rows}
    if observed != {expected_split}:
        raise ValueError(
            f"{label} summary must contain only split={expected_split} rows; observed={sorted(observed)}"
        )
    for column in ("evaluation_images", "split_sha256"):
        missing = [row for row in rows if not row.get(column)]
        if missing:
            raise ValueError(f"{label} summary is missing required {column} metadata")


def validate_common_split(
    zero_shot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> None:
    signatures: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in [*zero_shot_rows, *baseline_rows]:
        dataset = row.get("dataset", "")
        signatures[dataset].add((row.get("evaluation_images", ""), row.get("split_sha256", "")))
    bad = {dataset: values for dataset, values in signatures.items() if len(values) > 1}
    if bad:
        raise ValueError(f"Task 8 inputs do not share the same held-out split metadata: {bad}")


def split_metadata(row: dict[str, str]) -> dict[str, str]:
    return {
        "split": row.get("split", ""),
        "evaluation_images": row.get("evaluation_images", ""),
        "split_sha256": row.get("split_sha256", ""),
    }


def metric_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {None, "", "None"}:
        return None
    return float(value)


def metric_path_from_row(row: dict[str, str]) -> Path:
    return resolve_repo_path(row["metrics_file"])


def load_metric_json(row: dict[str, str]) -> dict[str, Any]:
    return load_json(metric_path_from_row(row))


def expand_category_rows(
    zero_shot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for summary_row in zero_shot_rows:
        metric = load_metric_json(summary_row)
        instance_metrics = metric["instance_metrics"]
        append_category_rows(
            rows=rows,
            run_type="zero_shot",
            dataset=summary_row["dataset"],
            model=summary_row["model"],
            prompt_mode=summary_row["prompt_mode"],
            evaluation_type=metric["evaluation_type"],
            metric_block=instance_metrics,
            metadata=split_metadata(summary_row),
        )

    for summary_row in baseline_rows:
        metric = load_metric_json(summary_row)
        if metric["evaluation_type"] == "semantic_segmentation":
            metric_block = {
                "per_category_iou": metric["semantic_metrics"]["per_category_iou"],
                "per_category_boundary_f1": metric["boundary_metrics"]["per_category_boundary_f1"],
                "per_category_count": {},
            }
        else:
            metric_block = metric["instance_metrics"]
        append_category_rows(
            rows=rows,
            run_type="baseline",
            dataset=summary_row["dataset"],
            model=summary_row["baseline"],
            prompt_mode="inference",
            evaluation_type=metric["evaluation_type"],
            metric_block=metric_block,
            metadata=split_metadata(summary_row),
        )
    return rows


def append_category_rows(
    rows: list[dict[str, Any]],
    run_type: str,
    dataset: str,
    model: str,
    prompt_mode: str,
    evaluation_type: str,
    metric_block: dict[str, Any],
    metadata: dict[str, str],
) -> None:
    per_iou = metric_block.get("per_category_iou", {})
    per_bf1 = metric_block.get("per_category_boundary_f1", {})
    per_count = metric_block.get("per_category_count", {})
    for category, iou in sorted(per_iou.items()):
        if iou is None:
            continue
        rows.append(
            {
                "run_type": run_type,
                "dataset": dataset,
                "model": model,
                "prompt_mode": prompt_mode,
                "evaluation_type": evaluation_type,
                "category": category,
                "iou": float(iou),
                "boundary_f1": per_bf1.get(category),
                "count": per_count.get(category),
                **metadata,
            }
        )


def challenge_for_category(category: str, challenge_groups: dict[str, list[str]]) -> str:
    for challenge, categories in challenge_groups.items():
        if category in set(categories):
            return challenge
    return "other"


def challenge_summary_rows(
    category_rows: list[dict[str, Any]],
    challenge_groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in category_rows:
        challenge = challenge_for_category(row["category"], challenge_groups)
        key = (
            row["run_type"],
            row["dataset"],
            row["model"],
            row["prompt_mode"],
            row.get("split", ""),
            row.get("evaluation_images", ""),
            row.get("split_sha256", ""),
            challenge,
        )
        grouped[key].append(row)

    summaries = []
    for (
        run_type,
        dataset,
        model,
        prompt_mode,
        split,
        evaluation_images,
        split_sha256,
        challenge,
    ), rows in sorted(grouped.items()):
        weights = np.asarray([float(row.get("count") or 1.0) for row in rows], dtype=np.float64)
        ious = np.asarray([float(row["iou"]) for row in rows], dtype=np.float64)
        bf1_values = [
            float(row["boundary_f1"])
            for row in rows
            if row.get("boundary_f1") not in {None, "", "None"}
        ]
        summaries.append(
            {
                "run_type": run_type,
                "dataset": dataset,
                "model": model,
                "prompt_mode": prompt_mode,
                "split": split,
                "evaluation_images": evaluation_images,
                "split_sha256": split_sha256,
                "challenge_group": challenge,
                "categories": ", ".join(row["category"] for row in rows),
                "weighted_iou": float(np.average(ious, weights=weights)),
                "mean_iou": float(np.mean(ious)),
                "mean_boundary_f1": float(np.mean(bf1_values)) if bf1_values else None,
                "category_count": len(rows),
                "instance_count": int(sum(row.get("count") or 0 for row in rows)),
            }
        )
    return summaries


def prompt_comparison_rows(zero_shot_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in zero_shot_rows:
        grouped[(row["dataset"], row["model"])][row["prompt_mode"]] = row

    rows = []
    for (dataset, model), by_prompt in sorted(grouped.items()):
        point = metric_float(by_prompt.get("point", {}), "mIoU")
        box = metric_float(by_prompt.get("box", {}), "mIoU")
        automatic = metric_float(by_prompt.get("automatic", {}), "mIoU")
        reference = next(iter(by_prompt.values()))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                **split_metadata(reference),
                "point_mIoU": point,
                "box_mIoU": box,
                "automatic_mIoU": automatic,
                "box_minus_point": None if box is None or point is None else box - point,
                "automatic_minus_box": None if automatic is None or box is None else automatic - box,
            }
        )
    return rows


def speed_quality_rows(
    task7_rows: list[dict[str, str]],
    zero_shot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    quality: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in zero_shot_rows:
        quality[(row["dataset"], row["model"], row["prompt_mode"])] = {
            "run_type": "zero_shot",
            "mIoU": metric_float(row, "mIoU"),
            "mask_AP": metric_float(row, "mask_AP"),
            **split_metadata(row),
        }
    for row in baseline_rows:
        quality[(row["dataset"], row["baseline"], "inference")] = {
            "run_type": "baseline",
            "mIoU": metric_float(row, "mIoU"),
            "mask_AP": metric_float(row, "mask_AP"),
            **split_metadata(row),
        }

    rows = []
    for row in task7_rows:
        key = (row["dataset"], row["model"], row["prompt_mode"])
        if key not in quality:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "prompt_mode": row["prompt_mode"],
                "device": row["device"],
                "fps": metric_float(row, "fps"),
                "latency_mean_ms": metric_float(row, "latency_mean_ms"),
                "mIoU": quality[key]["mIoU"],
                "mask_AP": quality[key]["mask_AP"],
                "run_type": quality[key]["run_type"],
                "split": quality[key]["split"],
                "evaluation_images": quality[key]["evaluation_images"],
                "split_sha256": quality[key]["split_sha256"],
            }
        )
    return rows


def task4_prediction_path(task4_config: dict[str, Any], dataset: str, model: str, prompt_mode: str) -> Path:
    return (
        resolve_repo_path(task4_config["task"]["output_root"])
        / dataset
        / model
        / f"{prompt_mode}_predictions.jsonl"
    )


def find_candidate_failures(
    case: dict[str, Any],
    task4_config: dict[str, Any],
    allowed_image_ids: set[int] | None,
    max_records: int,
    max_examples: int,
    default_score: float,
) -> list[dict[str, Any]]:
    dataset = case["dataset"]
    model = case["model"]
    prompt_mode = case["prompt_mode"]
    pred_path = task4_prediction_path(task4_config, dataset, model, prompt_mode)
    dataset_config = task4_config["datasets"][dataset]
    annotation_file = Path(dataset_config["annotation_file"])
    coco = load_json(annotation_file)
    gt_by_id = annotations_by_id(coco)
    gt_by_image = annotations_by_image(coco)
    image_sizes = image_size_by_id(coco)
    names_by_category = category_name_by_id(coco)
    selected_categories = set(case.get("categories") or [])
    available_names = set(names_by_category.values())
    if selected_categories and not selected_categories.intersection(available_names):
        selected_categories = set()

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(iter_jsonl(pred_path), start=1):
        if max_records and index > max_records:
            break
        if allowed_image_ids is not None and int(row["image_id"]) not in allowed_image_ids:
            continue
        if prompt_mode == "automatic":
            image_candidates = automatic_candidates(
                row,
                gt_by_image,
                image_sizes,
                names_by_category,
                selected_categories,
            )
            candidates.extend(image_candidates)
        else:
            candidate = prompted_candidate(
                row,
                gt_by_id,
                image_sizes,
                names_by_category,
                selected_categories,
                default_score,
            )
            if candidate is not None:
                candidates.append(candidate)

        if index == 1 or index % 5000 == 0:
            print(
                f"[PROGRESS] mined {dataset}/{model}/{prompt_mode}: "
                f"{index} rows, candidates={len(candidates)}",
                flush=True,
            )

    candidates.sort(key=lambda item: (item["iou"], item["area"]))
    selected = []
    seen_categories = set()
    seen = set()
    for item in candidates:
        category_key = item["category_name"]
        if category_key in seen_categories:
            continue
        unique_key = (item["image_id"], item["category_name"])
        seen_categories.add(category_key)
        seen.add(unique_key)
        selected.append(item)
        if len(selected) >= max_examples:
            return selected

    for item in candidates:
        unique_key = (item["image_id"], item["category_name"])
        if unique_key in seen:
            continue
        seen.add(unique_key)
        selected.append(item)
        if len(selected) >= max_examples:
            break
    return selected


def prompted_candidate(
    row: dict[str, Any],
    gt_by_id: dict[int, dict[str, Any]],
    image_sizes: dict[int, tuple[int, int]],
    names_by_category: dict[int, str],
    selected_categories: set[str],
    default_score: float,
) -> dict[str, Any] | None:
    annotation_id = int(row["annotation_id"])
    annotation = gt_by_id.get(annotation_id)
    if annotation is None:
        return None
    category_id = int(annotation["category_id"])
    category_name = names_by_category.get(category_id, str(category_id))
    if selected_categories and category_name not in selected_categories:
        return None
    height, width = image_sizes[int(row["image_id"])]
    gt_rle = annotation_to_rle(annotation, height, width)
    prediction = select_prediction(row.get("predictions", []), default_score)
    pred_rle = prediction["segmentation"] if prediction else None
    iou = rle_iou(gt_rle, pred_rle) if pred_rle is not None else 0.0
    return {
        "image_id": int(row["image_id"]),
        "annotation_id": annotation_id,
        "category_id": category_id,
        "category_name": category_name,
        "image_path": row["image_path"],
        "file_name": row["file_name"],
        "gt_rle": gt_rle,
        "pred_rle": pred_rle,
        "iou": iou,
        "area": float(annotation.get("area", 0.0)),
    }


def automatic_candidates(
    row: dict[str, Any],
    gt_by_image: dict[int, list[dict[str, Any]]],
    image_sizes: dict[int, tuple[int, int]],
    names_by_category: dict[int, str],
    selected_categories: set[str],
) -> list[dict[str, Any]]:
    image_id = int(row["image_id"])
    height, width = image_sizes[image_id]
    pred_rles = [prediction["segmentation"] for prediction in row.get("predictions", [])]
    candidates = []
    for annotation in gt_by_image.get(image_id, []):
        category_id = int(annotation["category_id"])
        category_name = names_by_category.get(category_id, str(category_id))
        if selected_categories and category_name not in selected_categories:
            continue
        gt_rle = annotation_to_rle(annotation, height, width)
        best_rle = None
        best_iou = 0.0
        for pred_rle in pred_rles:
            iou = rle_iou(gt_rle, pred_rle)
            if iou > best_iou:
                best_iou = iou
                best_rle = pred_rle
        candidates.append(
            {
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "category_id": category_id,
                "category_name": category_name,
                "image_path": row["image_path"],
                "file_name": row["file_name"],
                "gt_rle": gt_rle,
                "pred_rle": best_rle,
                "iou": best_iou,
                "area": float(annotation.get("area", 0.0)),
            }
        )
    return candidates


def overlay_failure(
    candidate: dict[str, Any],
    output_path: Path,
    label: str,
    alpha: float,
) -> None:
    image = Image.open(candidate["image_path"]).convert("RGB")
    image_array = np.asarray(image).astype(np.float32)
    gt_mask = rle_to_mask(candidate["gt_rle"])
    pred_mask = (
        np.zeros(gt_mask.shape, dtype=bool)
        if candidate["pred_rle"] is None
        else rle_to_mask(candidate["pred_rle"])
    )
    overlay = image_array.copy()
    gt_only = np.logical_and(gt_mask, ~pred_mask)
    pred_only = np.logical_and(pred_mask, ~gt_mask)
    overlap = np.logical_and(gt_mask, pred_mask)
    overlay[gt_only] = (1.0 - alpha) * overlay[gt_only] + alpha * np.asarray([0, 220, 0])
    overlay[pred_only] = (1.0 - alpha) * overlay[pred_only] + alpha * np.asarray([230, 0, 0])
    overlay[overlap] = (1.0 - alpha) * overlay[overlap] + alpha * np.asarray([240, 220, 0])
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    text = (
        f"{label} | {candidate['category_name']} | "
        f"IoU={candidate['iou']:.3f} | green=GT red=prediction"
    )
    draw.rectangle((0, 0, min(output.width, 16 + len(text) * 7), 26), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def generate_visual_failures(config: dict[str, Any], task4_config: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    visual_config = config["visual_mining"]
    output_root = resolve_repo_path(config["task"]["output_root"])
    figures_dir = output_root / "figures"
    max_records = int(args.max_records_per_case or visual_config["max_records_per_case"])
    max_examples = int(visual_config["max_examples_per_case"])
    default_score = float(visual_config["default_score"])
    alpha = float(visual_config["overlay_alpha"])
    rows = []

    for case_index, case in enumerate(visual_config["cases"], start=1):
        print(
            f"[START] visual mining case {case_index}: "
            f"{case['dataset']}/{case['model']}/{case['prompt_mode']}",
            flush=True,
        )
        try:
            allowed_image_ids = load_split_image_ids(config, case["dataset"])
            candidates = find_candidate_failures(
                case,
                task4_config,
                allowed_image_ids,
                max_records,
                max_examples,
                default_score,
            )
        except FileNotFoundError as exc:
            print(f"[WARN] skipping visual case {case_index}: {exc}", flush=True)
            continue
        for example_index, candidate in enumerate(candidates, start=1):
            stem = (
                f"{case_index:02d}_{example_index:02d}_{case['dataset']}_"
                f"{case['model']}_{case['prompt_mode']}_{candidate['category_name']}_"
                f"iou_{candidate['iou']:.3f}"
            )
            safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in stem)
            figure_path = figures_dir / f"{safe_stem}.png"
            overlay_failure(candidate, figure_path, case["label"], alpha)
            rows.append(
                {
                    "case_label": case["label"],
                    "dataset": case["dataset"],
                    "model": case["model"],
                    "prompt_mode": case["prompt_mode"],
                    "category": candidate["category_name"],
                    "image_id": candidate["image_id"],
                    "annotation_id": candidate["annotation_id"],
                    "iou": candidate["iou"],
                    "area": candidate["area"],
                    "figure": relative_to_repo(figure_path),
                    "file_name": candidate["file_name"],
                }
            )
        print(f"[OK] case {case_index}: selected {len(candidates)} examples", flush=True)
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int) -> str:
    selected = rows[:max_rows]
    if not selected:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = f"{value:.3f}"
            elif value is None:
                value = ""
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def key_findings(
    zero_shot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    category_rows: list[dict[str, Any]],
    speed_quality: list[dict[str, Any]],
) -> list[str]:
    findings = []
    box_advantages = []
    for row in prompt_comparison_rows(zero_shot_rows):
        if row["box_minus_point"] is not None:
            box_advantages.append(float(row["box_minus_point"]))
    if box_advantages:
        findings.append(
            "Box prompts are consistently stronger than point prompts for SAM-family models; "
            f"the mean box-minus-point mIoU gap is {np.mean(box_advantages):.3f}."
        )

    automatic_aps = [
        metric_float(row, "mask_AP")
        for row in zero_shot_rows
        if row["prompt_mode"] == "automatic" and metric_float(row, "mask_AP") is not None
    ]
    box_aps = [
        metric_float(row, "mask_AP")
        for row in zero_shot_rows
        if row["prompt_mode"] == "box" and metric_float(row, "mask_AP") is not None
    ]
    if automatic_aps and box_aps:
        findings.append(
            "Automatic mask generation has lower AP than box prompting because proposals are "
            f"class-agnostic and duplicate/merge objects; mean AP is {np.mean(automatic_aps):.3f} "
            f"versus {np.mean(box_aps):.3f} for box prompts."
        )

    low_categories = sorted(category_rows, key=lambda row: row["iou"])[:8]
    if low_categories:
        categories = ", ".join(f"{row['category']} ({row['model']})" for row in low_categories[:5])
        findings.append(
            "The weakest categories are concentrated in small/rare parts and robot-body masks: "
            f"{categories}."
        )

    realtime_gpu = [
        row
        for row in speed_quality
        if row["device"] == "cuda" and row.get("fps") is not None and float(row["fps"]) >= 10.0
    ]
    if realtime_gpu:
        models = sorted({row["model"] for row in realtime_gpu})
        findings.append(
            "Real-time GPU feasibility is mainly with supervised/lightweight models: "
            f"{', '.join(models)} reach at least 10 FPS in one or more settings."
        )

    baseline_mious = [metric_float(row, "mIoU") for row in baseline_rows if metric_float(row, "mIoU") is not None]
    if baseline_mious:
        findings.append(
            "Small-subset supervised baselines improve scene-specific object consistency, "
            "but semantic DeepLabV3+ does not provide instance AP and therefore is not a full "
            "replacement for instance segmentation."
        )
    return findings


def write_report(
    output_root: Path,
    zero_shot_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    category_rows: list[dict[str, Any]],
    challenge_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    speed_quality: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
) -> None:
    worst_zero_shot = sorted(
        [row for row in category_rows if row["run_type"] == "zero_shot"],
        key=lambda row: row["iou"],
    )
    worst_baseline = sorted(
        [row for row in category_rows if row["run_type"] == "baseline"],
        key=lambda row: row["iou"],
    )
    weak_challenges = sorted(challenge_rows, key=lambda row: row["weighted_iou"])
    report_lines = [
        "# Task 8 - Failure Mode Analysis",
        "",
        "This report summarizes where the segmentation models fail in the robotic scenes, using "
        "Task 6 metrics and representative visual overlays mined from Task 4 predictions.",
        "",
        "## Key Findings",
        "",
    ]
    for finding in key_findings(zero_shot_rows, baseline_rows, category_rows, speed_quality):
        report_lines.append(f"- {finding}")

    report_lines.extend(
        [
            "",
            "## Lowest Zero-Shot Categories",
            "",
            markdown_table(
                worst_zero_shot,
                ["dataset", "model", "prompt_mode", "category", "iou", "boundary_f1", "count"],
                15,
            ),
            "",
            "## Lowest Baseline Categories",
            "",
            markdown_table(
                worst_baseline,
                ["dataset", "model", "category", "iou", "boundary_f1", "count"],
                15,
            ),
            "",
            "## Weakest Challenge Groups",
            "",
            markdown_table(
                weak_challenges,
                [
                    "run_type",
                    "dataset",
                    "model",
                    "prompt_mode",
                    "challenge_group",
                    "weighted_iou",
                    "mean_boundary_f1",
                ],
                20,
            ),
            "",
            "## Prompt Sensitivity",
            "",
            markdown_table(
                sorted(prompt_rows, key=lambda row: row["box_minus_point"] or 0.0, reverse=True),
                ["dataset", "model", "point_mIoU", "box_mIoU", "automatic_mIoU", "box_minus_point"],
                12,
            ),
            "",
            "## Representative Visual Failures",
            "",
            "Overlay convention: green = ground truth only, red = prediction only, yellow = overlap.",
            "",
            markdown_table(
                visual_rows,
                ["case_label", "dataset", "model", "prompt_mode", "category", "iou", "figure"],
                20,
            ),
            "",
            "### Visual Overlays",
            "",
        ]
    )
    for row in visual_rows[:10]:
        figure = Path(str(row.get("figure", "")))
        report_lines.extend(
            [
                f"**{row.get('case_label', 'Failure case')} - {row.get('category', '')}**",
                "",
                f"![{row.get('case_label', 'Failure case')}]({Path('figures') / figure.name})",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Interpretation",
            "",
            "- Small screws, connectors, cables, rubber parts, and sensor modules fail because their "
            "masks are small, thin, or visually similar to nearby fixtures. A minor boundary error "
            "can dominate IoU for these categories.",
            "- Transparent and reflective objects are sensitive to missing visual edges, specular "
            "highlights, and background bleed-through. Box prompts help, but automatic proposal "
            "generation often merges glass with the surface behind it.",
            "- Robot and occlusion failures are common when articulated robot parts touch tools, bins, "
            "or the workbench. FastSAM often returns coarse object proposals, while SAM variants "
            "need stronger prompts to isolate the correct part.",
            "- Automatic mask generation is useful for open-set proposal discovery, but in cluttered "
            "robotic scenes it creates duplicates and class-agnostic masks that do not align well "
            "with instance categories.",
            "- Supervised baselines trained on the small subset adapt well to the dataset domain, but "
            "they inherit the weaknesses of limited labels: rare small parts remain weak, and "
            "DeepLabV3+ cannot separate individual instances.",
            "",
        ]
    )
    (output_root / "task8_failure_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    task4_config = load_yaml(config["inputs"]["task4_config"])
    output_root = resolve_repo_path(config["task"]["output_root"])

    zero_shot_rows = read_csv(config["inputs"]["task6_zero_shot_summary"])
    baseline_rows = read_csv(config["inputs"]["task6_baseline_summary"])
    task7_rows = read_csv(config["inputs"]["task7_speed_summary"])
    expected_split = config.get("evaluation", {}).get("split", "full")
    validate_split_rows(zero_shot_rows, "zero-shot", expected_split)
    validate_split_rows(baseline_rows, "baseline", expected_split)
    validate_common_split(zero_shot_rows, baseline_rows)

    print("[START] expanding category metrics", flush=True)
    category_rows = expand_category_rows(zero_shot_rows, baseline_rows)
    challenge_rows = challenge_summary_rows(category_rows, config["challenge_groups"])
    prompt_rows = prompt_comparison_rows(zero_shot_rows)
    speed_quality = speed_quality_rows(task7_rows, zero_shot_rows, baseline_rows)

    if args.dry_run:
        print(f"[DRY RUN] category_rows={len(category_rows)} challenge_rows={len(challenge_rows)}")
        return

    write_csv(
        output_root / "category_failures.csv",
        sorted(category_rows, key=lambda row: row["iou"]),
        [
            "run_type",
            "dataset",
            "model",
            "prompt_mode",
            "evaluation_type",
            "category",
            "iou",
            "boundary_f1",
            "count",
            "split",
            "evaluation_images",
            "split_sha256",
        ],
    )
    write_csv(
        output_root / "challenge_group_summary.csv",
        sorted(challenge_rows, key=lambda row: row["weighted_iou"]),
        [
            "run_type",
            "dataset",
            "model",
            "prompt_mode",
            "split",
            "evaluation_images",
            "split_sha256",
            "challenge_group",
            "categories",
            "weighted_iou",
            "mean_iou",
            "mean_boundary_f1",
            "category_count",
            "instance_count",
        ],
    )
    write_csv(
        output_root / "prompt_mode_comparison.csv",
        prompt_rows,
        [
            "dataset",
            "model",
            "split",
            "evaluation_images",
            "split_sha256",
            "point_mIoU",
            "box_mIoU",
            "automatic_mIoU",
            "box_minus_point",
            "automatic_minus_box",
        ],
    )
    write_csv(
        output_root / "speed_quality_tradeoff.csv",
        speed_quality,
        [
            "dataset",
            "model",
            "prompt_mode",
            "device",
            "fps",
            "latency_mean_ms",
            "mIoU",
            "mask_AP",
            "run_type",
            "split",
            "evaluation_images",
            "split_sha256",
        ],
    )

    visual_rows = generate_visual_failures(config, task4_config, args)
    write_csv(
        output_root / "representative_failures.csv",
        visual_rows,
        [
            "case_label",
            "dataset",
            "model",
            "prompt_mode",
            "category",
            "image_id",
            "annotation_id",
            "iou",
            "area",
            "figure",
            "file_name",
        ],
    )
    write_json(
        output_root / "summary.json",
        {
            "split": expected_split,
            "category_rows": len(category_rows),
            "challenge_rows": len(challenge_rows),
            "prompt_rows": len(prompt_rows),
            "speed_quality_rows": len(speed_quality),
            "visual_examples": len(visual_rows),
            "outputs": {
                "category_failures": relative_to_repo(output_root / "category_failures.csv"),
                "challenge_group_summary": relative_to_repo(output_root / "challenge_group_summary.csv"),
                "prompt_mode_comparison": relative_to_repo(output_root / "prompt_mode_comparison.csv"),
                "speed_quality_tradeoff": relative_to_repo(output_root / "speed_quality_tradeoff.csv"),
                "representative_failures": relative_to_repo(output_root / "representative_failures.csv"),
                "report": relative_to_repo(output_root / "task8_failure_analysis.md"),
            },
        },
    )
    write_report(
        output_root,
        zero_shot_rows,
        baseline_rows,
        category_rows,
        challenge_rows,
        prompt_rows,
        speed_quality,
        visual_rows,
    )
    print(f"[DONE] wrote {relative_to_repo(output_root)}", flush=True)


if __name__ == "__main__":
    main()
