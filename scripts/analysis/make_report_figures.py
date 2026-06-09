"""Generate lightweight figures for the final COGAR benchmark reports.

The plotted benchmark values are read from existing output CSV/JSON files when
available. A few fallback constants are copied from the final documented result
tables in docs/ so the report figures remain reproducible if optional output
tables are absent.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cogar_matplotlib")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "final_report"
DATASET_OUT = OUT / "dataset"
SAMPLES_OUT = DATASET_OUT / "sample_scenes"
METRICS_OUT = OUT / "metrics"
SPEED_OUT = OUT / "speed"
FAILURE_OUT = OUT / "failure_modes"
EDGE_OUT = OUT / "edge_tradeoff"


def ensure_dirs() -> None:
    for directory in [
        DATASET_OUT,
        SAMPLES_OUT,
        METRICS_OUT,
        SPEED_OUT,
        FAILURE_OUT,
        EDGE_OUT,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def as_float(row: dict[str, str], *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                continue
    return default


def as_int(row: dict[str, str], *names: str, default: int | None = None) -> int | None:
    value = as_float(row, *names, default=None)
    if value is None:
        return default
    return int(value)


def save_barh(labels: list[str], values: list[float], path: Path, title: str, xlabel: str) -> None:
    pairs = sorted(zip(labels, values), key=lambda item: item[1])
    labels_sorted = [item[0] for item in pairs]
    values_sorted = [item[1] for item in pairs]
    height = max(4.2, 0.42 * len(labels_sorted) + 1.4)
    fig, ax = plt.subplots(figsize=(9, height))
    bars = ax.barh(labels_sorted, values_sorted, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values_sorted):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_grouped_bars(
    groups: list[str],
    series: dict[str, list[float]],
    path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = list(range(len(groups)))
    names = list(series)
    width = min(0.8 / max(len(names), 1), 0.24)
    fig_width = max(10, 0.48 * len(groups) + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    for idx, name in enumerate(names):
        offsets = [pos + (idx - (len(names) - 1) / 2) * width for pos in x]
        ax.bar(offsets, series[name], width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_dataset_charts(generated: list[Path]) -> None:
    category_rows = read_csv_rows(ROOT / "outputs/tables/dataset_audit_final_filtered/category_counts.csv")
    if not category_rows:
        category_rows = [
            {"category_name": "robot_gripper", "count": "1042"},
            {"category_name": "plastic_object", "count": "627"},
            {"category_name": "metal_part", "count": "555"},
            {"category_name": "connector", "count": "531"},
            {"category_name": "screw", "count": "427"},
            {"category_name": "glass_object", "count": "360"},
            {"category_name": "box", "count": "352"},
            {"category_name": "tool", "count": "296"},
            {"category_name": "cable", "count": "281"},
        ]
    path = DATASET_OUT / "category_counts.png"
    save_barh(
        [row["category_name"] for row in category_rows],
        [float(row["count"]) for row in category_rows],
        path,
        "COGAR-SimRobotics-500 Object Count by Category",
        "Object instances",
    )
    generated.append(path)

    challenge_rows = read_csv_rows(ROOT / "outputs/tables/dataset_audit_final_filtered/challenge_counts.csv")
    if not challenge_rows:
        challenge_rows = [
            {"challenge_primary": "small_parts", "count": "1269"},
            {"challenge_primary": "partial_occlusion", "count": "920"},
            {"challenge_primary": "dynamic_scene", "count": "797"},
            {"challenge_primary": "reflective_metal", "count": "743"},
            {"challenge_primary": "transparent_glass", "count": "742"},
        ]
    path = DATASET_OUT / "challenge_distribution.png"
    save_barh(
        [row["challenge_primary"] for row in challenge_rows],
        [float(row["count"]) for row in challenge_rows],
        path,
        "COGAR-SimRobotics-500 Challenge Distribution",
        "Object instances",
    )
    generated.append(path)


def documented_model_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    sam_rows = read_csv_rows(ROOT / "outputs/tables/sim_sam_vit_b_final_prompt_summary.csv")
    for row in sam_rows:
        rows.append(
            {
                "label": f"SAM ViT-B {row['prompt']}",
                "model": "SAM ViT-B",
                "prompt": row["prompt"],
                "mean_iou": float(row["mean_iou"]),
                "boundary_f1": float(row["mean_boundary_f1"]),
                "fps": 61.96196669328483 if row["prompt"] == "box" else math.nan,
                "scope": "full",
            }
        )

    for prompt in ["box", "point", "auto"]:
        path = ROOT / f"outputs/tables/sam2/final_{prompt}_cuda/overall_summary.csv"
        for row in read_csv_rows(path):
            rows.append(
                {
                    "label": f"SAM2.1-Tiny {prompt}",
                    "model": "SAM2.1-Tiny",
                    "prompt": prompt,
                    "mean_iou": float(row["mean_iou"]),
                    "boundary_f1": float(row["mean_boundary_f1"]),
                    "fps": float(row["mean_fps"]),
                    "scope": "full",
                }
            )

    fastsam_box = read_csv_rows(ROOT / "outputs/tables/final_box_prompt_model_summary.csv")
    for row in fastsam_box:
        if row["run"] == "fastsam_s_box":
            rows.append(
                {
                    "label": "FastSAM-S box",
                    "model": "FastSAM-S",
                    "prompt": "box",
                    "mean_iou": float(row["mean_iou"]),
                    "boundary_f1": float(row["mean_boundary_f1"]),
                    "fps": float(row["mean_fps"]),
                    "scope": "full",
                }
            )
        if row["run"] == "mobilesam_box":
            rows.append(
                {
                    "label": "MobileSAM box",
                    "model": "MobileSAM",
                    "prompt": "box",
                    "mean_iou": float(row["mean_iou"]),
                    "boundary_f1": float(row["mean_boundary_f1"]),
                    "fps": float(row["mean_fps"]),
                    "scope": "full",
                }
            )
    for prompt in ["point", "auto"]:
        path = ROOT / f"outputs/tables/fastsam/{prompt}_final/overall_summary.csv"
        for row in read_csv_rows(path):
            rows.append(
                {
                    "label": f"FastSAM-S {prompt}",
                    "model": "FastSAM-S",
                    "prompt": prompt,
                    "mean_iou": float(row["mean_iou"]),
                    "boundary_f1": float(row["mean_boundary_f1"]),
                    "fps": float(row["mean_fps"]),
                    "scope": "full",
                }
            )

    for row in read_csv_rows(ROOT / "outputs/tables/efficientsam/final_ti_cuda_fixed/overall_summary.csv"):
        rows.append(
            {
                "label": "EfficientSAM-Ti box",
                "model": "EfficientSAM-Ti",
                "prompt": "box",
                "mean_iou": float(row["mean_iou"]),
                "boundary_f1": float(row["mean_boundary_f1"]),
                "fps": float(row["mean_fps"]),
                "scope": "full",
            }
        )

    for filename in [
        "sim_sam_vit_h_box_cpu_25_retry_summary.csv",
        "sim_sam_vit_h_point_cpu_25_retry_summary.csv",
        "sim_sam_vit_h_auto_cpu_5img_retry_summary.csv",
    ]:
        for row in read_csv_rows(ROOT / "outputs/tables" / filename):
            prompt = row["prompt_mode"]
            rows.append(
                {
                    "label": f"SAM ViT-H {prompt} CPU subset",
                    "model": "SAM ViT-H",
                    "prompt": prompt,
                    "mean_iou": float(row["mean_iou"]),
                    "boundary_f1": float(row["mean_boundary_f1"]),
                    "fps": float(row["mean_fps"]),
                    "scope": "subset",
                }
            )

    return rows


def generate_metric_charts(generated: list[Path]) -> None:
    rows = documented_model_rows()
    labels = [str(row["label"]) for row in rows]

    path = METRICS_OUT / "mean_iou_by_model_prompt.png"
    save_barh(
        labels,
        [float(row["mean_iou"]) for row in rows],
        path,
        "Mean IoU by Model and Prompt Mode",
        "Mean IoU",
    )
    generated.append(path)

    path = METRICS_OUT / "boundary_f1_by_model_prompt.png"
    save_barh(
        labels,
        [float(row["boundary_f1"]) for row in rows],
        path,
        "Boundary F1 by Model and Prompt Mode",
        "Mean boundary F1",
    )
    generated.append(path)

    fps_rows = [row for row in rows if not math.isnan(float(row["fps"]))]
    path = SPEED_OUT / "fps_comparison.png"
    labels_fps = [str(row["label"]) for row in fps_rows]
    fps_values = [float(row["fps"]) for row in fps_rows]
    pairs = sorted(zip(labels_fps, fps_values), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(10, 6.8))
    bars = ax.barh([item[0] for item in pairs], [item[1] for item in pairs], color="#0891b2")
    ax.set_xscale("log")
    ax.set_title("FPS Comparison Across Completed Evaluations")
    ax.set_xlabel("Mean FPS, log scale")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, [item[1] for item in pairs]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path)

    path = EDGE_OUT / "iou_vs_fps_tradeoff.png"
    tradeoff_labels = {
        "SAM ViT-B box",
        "MobileSAM box",
        "EfficientSAM-Ti box",
        "FastSAM-S box",
        "SAM2.1-Tiny box",
    }
    tradeoff_rows = [row for row in rows if row["label"] in tradeoff_labels]
    tradeoff_rows.append(
        {
            "label": "YOLOv8n-seg supervised",
            "mean_iou": 0.601,
            "fps": 37.3,
        }
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for row in tradeoff_rows:
        ax.scatter(float(row["fps"]), float(row["mean_iou"]), s=80)
        ax.text(float(row["fps"]) * 1.04, float(row["mean_iou"]), str(row["label"]), fontsize=8, va="center")
    ax.set_xscale("log")
    ax.set_title("Accuracy and Speed Trade-Off")
    ax.set_xlabel("FPS, log scale")
    ax.set_ylabel("Mean IoU or mask mAP50-95 for YOLOv8n-seg")
    ax.set_ylim(0.55, 0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path)

    path = METRICS_OUT / "supervised_baselines_summary.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    yolo_metrics = {"YOLO mask mAP50": 0.806, "YOLO mask mAP50-95": 0.601}
    axes[0].bar(yolo_metrics.keys(), yolo_metrics.values(), color=["#16a34a", "#22c55e"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("YOLOv8n-seg AP Metrics")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    mask_summary = read_csv_rows(ROOT / "outputs/tables/maskrcnn_resnet50_fpn_cogar_full_summary.csv")
    if mask_summary:
        mask_iou = float(mask_summary[0]["mean_iou"])
        mask_bf1 = float(mask_summary[0]["mean_boundary_f1"])
    else:
        mask_iou = 0.7461524914807578
        mask_bf1 = 0.7218426575393403
    axes[1].bar(["Mask R-CNN mean IoU", "Mask R-CNN BF1"], [mask_iou, mask_bf1], color=["#2563eb", "#60a5fa"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Mask R-CNN IoU Metrics")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Supervised Baseline Summary")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path)

    generate_per_category_chart(generated)


def generate_per_category_chart(generated: list[Path]) -> None:
    sam2_rows = read_csv_rows(ROOT / "outputs/tables/sam2/final_box_cuda/mean_iou_by_category.csv")
    mask_rows = read_csv_rows(ROOT / "outputs/tables/maskrcnn_resnet50_fpn_cogar_full_per_class.csv")
    if not sam2_rows or not mask_rows:
        return
    categories = sorted({row["category"] for row in sam2_rows} & {row["gt_category"] for row in mask_rows})
    sam2_by_cat = {row["category"]: float(row["mean_iou"]) for row in sam2_rows}
    mask_by_cat = {row["gt_category"]: float(row["mean_iou"]) for row in mask_rows}
    path = METRICS_OUT / "per_category_iou.png"
    save_grouped_bars(
        categories,
        {
            "SAM2.1-Tiny box": [sam2_by_cat[category] for category in categories],
            "Mask R-CNN": [mask_by_cat[category] for category in categories],
        },
        path,
        "Per-Category IoU Comparison",
        "Mean IoU",
        (0, 1),
    )
    generated.append(path)


def generate_pipeline_diagram(generated: list[Path]) -> None:
    path = DATASET_OUT / "simulation_pipeline.png"
    steps = [
        "BlenderProc scene setup\nIsaac route documented",
        "RGB/depth/mask\nrendering",
        "COCO-style\nannotations",
        "Object-level\nbenchmark index",
        "Model evaluation\nand reports",
    ]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for idx, (x_pos, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.35", "fc": "#e0f2fe", "ec": "#0369a1"},
        )
        if idx < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.075, 0.55),
                xytext=(x_pos + 0.075, 0.55),
                arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#334155"},
            )
    ax.set_title("Simulation Dataset Generation and Benchmark Pipeline", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path)


def select_sample_images() -> list[tuple[str, Path]]:
    index_path = ROOT / "data/cogar_sim_500_final/annotations/sim_robotic_scenes_index_final_filtered.csv"
    rows = read_csv_rows(index_path)
    wanted = [
        ("Reflective metal", lambda row: row.get("challenge_primary") == "reflective_metal"),
        ("Transparent glass", lambda row: row.get("challenge_primary") == "transparent_glass"),
        ("Small parts", lambda row: row.get("challenge_primary") == "small_parts"),
        ("Robot gripper", lambda row: row.get("category_name") == "robot_gripper"),
        ("Partial occlusion", lambda row: row.get("challenge_primary") == "partial_occlusion"),
        ("Dynamic scene", lambda row: row.get("challenge_primary") == "dynamic_scene"),
    ]
    selected: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for label, predicate in wanted:
        for row in rows:
            if not predicate(row):
                continue
            image_path = ROOT / row["image_path"]
            if image_path.exists() and image_path not in seen:
                selected.append((label, image_path))
                seen.add(image_path)
                break
    return selected


def image_on_axis(ax, image_path: Path, title: str) -> None:
    image = mpimg.imread(image_path)
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def generate_sample_scene_montage(generated: list[Path]) -> None:
    selected = select_sample_images()
    if not selected:
        return
    for label, image_path in selected:
        safe_label = label.lower().replace(" ", "_")
        out_path = SAMPLES_OUT / f"{safe_label}.png"
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        image_on_axis(ax, image_path, label)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        generated.append(out_path)

    cols = 3
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.7))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, (label, image_path) in zip(axes_list, selected):
        image_on_axis(ax, image_path, label)
    for ax in axes_list[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Representative COGAR-SimRobotics-500 Scenes", fontsize=13)
    fig.tight_layout()
    path = DATASET_OUT / "sample_scene_montage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path)


def generate_failure_montage(generated: list[Path]) -> None:
    representative = [
        ROOT / "outputs/figures/failure_modes/sam_vit_b_box/worst_01_iou_0.000_cable_partial_occlusion.png",
        ROOT / "outputs/figures/failure_modes/sam_vit_b_box/worst_04_iou_0.007_robot_gripper_dynamic_scene.png",
        ROOT / "outputs/figures/failure_modes/sam_vit_b_box/worst_12_iou_0.062_box_reflective_metal.png",
        ROOT / "outputs/figures/failure_modes/mobilesam_box/worst_01_iou_0.000_robot_gripper_transparent_glass.png",
        ROOT / "outputs/figures/failure_modes/mobilesam_box/worst_10_iou_0.044_connector_dynamic_scene.png",
        ROOT / "outputs/figures/failure_modes/fastsam_s_box/worst_06_iou_0.000_screw_partial_occlusion.png",
    ]
    labels = [
        "Cable / occlusion",
        "Robot gripper / dynamic",
        "Reflective scene",
        "Transparent gripper case",
        "Connector / dynamic",
        "Screw / occlusion",
    ]
    existing = [(label, path) for label, path in zip(labels, representative) if path.exists()]
    if not existing:
        return
    copied_paths: list[Path] = []
    for label, source in existing:
        target = FAILURE_OUT / source.name
        if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(source, target)
        copied_paths.append(target)
        generated.append(target)

    cols = 2
    rows = math.ceil(len(existing) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 3.8))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, (label, image_path) in zip(axes_list, zip(labels, copied_paths)):
        image_on_axis(ax, image_path, label)
    for ax in axes_list[len(existing) :]:
        ax.axis("off")
    fig.suptitle("Representative Failure Modes", fontsize=13)
    fig.tight_layout()
    path = FAILURE_OUT / "failure_mode_montage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    generated.append(path)


def main() -> None:
    ensure_dirs()
    generated: list[Path] = []
    generate_dataset_charts(generated)
    generate_pipeline_diagram(generated)
    generate_sample_scene_montage(generated)
    generate_metric_charts(generated)
    generate_failure_montage(generated)

    print("Generated report figures:")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
