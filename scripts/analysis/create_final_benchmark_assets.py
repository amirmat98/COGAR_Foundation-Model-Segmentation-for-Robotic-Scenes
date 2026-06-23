"""Create final benchmark plots and recommendation guide.

The script only uses compact CSV/JSON artifacts from Tasks 4-9. It does not
read raw prediction JSONL files or require model checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]


DATASET_LABELS = {
    "isaac_official_unitree_g1": "Isaac G1",
    "blenderproc_cogar_sim": "BlenderProc",
    "ocid": "OCID",
}

MODEL_LABELS = {
    "sam_vit_h": "SAM ViT-H",
    "sam_vit_b": "SAM ViT-B",
    "sam2_hiera_large": "SAM2-L",
    "fastsam_x": "FastSAM-X",
    "mobile_sam_vit_t": "MobileSAM",
    "efficient_sam_ti": "EffSAM-Ti",
    "efficient_sam_s": "EffSAM-S",
    "yolo8_seg": "YOLOv8-seg",
    "mask_rcnn": "Mask R-CNN",
    "deeplabv3plus": "DeepLabV3+",
}

PROMPT_LABELS = {
    "point": "Point",
    "box": "Box",
    "automatic": "Automatic",
    "inference": "Inference",
}

DATASET_EXAMPLE_IMAGES = {
    "Isaac official Unitree G1": Path(
        "/mnt/Info/COGAR_DATASETs/Isacc_dataset/datasets/robotic_sdg_v3_official_g1_1000/isaac/rgb_0000.png"
    ),
    "BlenderProc COGAR-SimRobotics-1000": Path(
        "/mnt/Info/COGAR_DATASETs/BlenderProc_cogar_sim_1000/rgb/000000.png"
    ),
    "OCID": Path(
        "/mnt/Info/COGAR_DATASETs/OCID-dataset/ARID10/floor/bottom/box/seq03/rgb/result_2018-08-27-15-54-17.png"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/final_benchmark_assets")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rel(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(resolve(path))


def label_dataset(value: str) -> str:
    return DATASET_LABELS.get(value, value)


def label_model(value: str) -> str:
    return MODEL_LABELS.get(value, value)


def label_prompt(value: str) -> str:
    return PROMPT_LABELS.get(value, value)


def numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - text_width) / 2, y0 + (y1 - y0 - text_height) / 2),
        text,
        fill=(30, 30, 30),
    )


def create_dataset_montage(output: Path) -> None:
    tile_size = (420, 315)
    label_height = 44
    gap = 18
    margin = 24
    width = margin * 2 + len(DATASET_EXAMPLE_IMAGES) * tile_size[0] + (len(DATASET_EXAMPLE_IMAGES) - 1) * gap
    height = margin * 2 + label_height + tile_size[1]
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    x = margin
    for label, path in DATASET_EXAMPLE_IMAGES.items():
        if path.exists():
            tile = fit_image(Image.open(path), tile_size)
        else:
            tile = Image.new("RGB", tile_size, (235, 235, 235))
            draw_missing = ImageDraw.Draw(tile)
            draw_centered_text(draw_missing, (0, 0, tile_size[0], tile_size[1]), "missing local image")
        canvas.paste(tile, (x, margin + label_height))
        draw.rectangle(
            (x, margin + label_height, x + tile_size[0] - 1, margin + label_height + tile_size[1] - 1),
            outline=(215, 215, 215),
            width=1,
        )
        draw_centered_text(draw, (x, margin, x + tile_size[0], margin + label_height), label)
        x += tile_size[0] + gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            xytext=(0, 2),
            textcoords="offset points",
        )


def plot_zero_shot_heatmap(df: pd.DataFrame, output: Path) -> None:
    data = df.copy()
    data["row"] = data["model"].map(label_model) + " / " + data["prompt_mode"].map(label_prompt)
    data["dataset_label"] = data["dataset"].map(label_dataset)
    pivot = data.pivot_table(index="row", columns="dataset_label", values="mIoU", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index), axis=0)

    fig_h = max(7.0, 0.32 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    image = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Zero-Shot Segmentation mIoU")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="mIoU")
    savefig(output)


def plot_baseline_bars(df: pd.DataFrame, output: Path) -> None:
    data = df.copy()
    data["label"] = data["baseline"].map(label_model)
    data["dataset_label"] = data["dataset"].map(label_dataset)
    datasets = [label_dataset(name) for name in DATASET_LABELS]
    models = [label_model(name) for name in ["yolo8_seg", "mask_rcnn", "deeplabv3plus"]]

    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, model in enumerate(models):
        values = []
        for dataset in datasets:
            subset = data[(data["dataset_label"] == dataset) & (data["label"] == model)]
            values.append(float(subset["mIoU"].iloc[0]) if not subset.empty else np.nan)
        ax.bar(x + (idx - 1) * width, values, width, label=model)
    ax.set_title("Classical Baseline mIoU")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(ncol=3, fontsize=8)
    annotate_bars(ax)
    savefig(output)


def plot_speed_quality(df: pd.DataFrame, output: Path) -> None:
    data = df[(df["device"] == "cuda") & df["mIoU"].notna() & df["fps"].notna()].copy()
    data = data[data["fps"] > 0]
    colors = {
        "heavy_sam": "#4169e1",
        "lightweight_sam": "#2e8b57",
        "baseline": "#b45f06",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for group, subset in data.groupby("model_group"):
        ax.scatter(
            subset["fps"],
            subset["mIoU"],
            s=50,
            alpha=0.78,
            label=group.replace("_", " "),
            color=colors.get(group, None),
        )
    ax.axvline(30, color="black", linestyle="--", linewidth=1, label="30 FPS")
    ax.set_xscale("log")
    ax.set_xlabel("GPU FPS (log scale)")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("GPU Speed-Quality Trade-Off")
    ax.legend(fontsize=8)

    top = data.sort_values("miou_fps_product", ascending=False).head(8)
    for _, row in top.iterrows():
        ax.annotate(
            label_model(row["model"]),
            (row["fps"], row["mIoU"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    savefig(output)


def plot_lightweight_tradeoff(df: pd.DataFrame, output: Path) -> None:
    data = df[
        (df["model_group"] == "lightweight_sam")
        & (df["device"] == "cuda")
        & df["mIoU"].notna()
        & df["fps"].notna()
    ].copy()
    markers = {"point": "o", "box": "s", "automatic": "^"}
    colors = {
        "mobile_sam_vit_t": "#1f77b4",
        "efficient_sam_ti": "#2ca02c",
        "efficient_sam_s": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (model, prompt), subset in data.groupby(["model", "prompt_mode"]):
        ax.scatter(
            subset["fps"],
            subset["mIoU"],
            marker=markers.get(prompt, "o"),
            color=colors.get(model),
            s=65,
            alpha=0.85,
            label=f"{label_model(model)} / {label_prompt(prompt)}",
        )
    ax.set_xlabel("GPU FPS")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Lightweight SAM Quality vs Speed")
    ax.legend(fontsize=7, ncol=2)
    savefig(output)


def plot_challenge_summary(df: pd.DataFrame, output: Path) -> None:
    data = df.copy()
    data = data[data["weighted_iou"].notna()]
    grouped = (
        data.groupby(["challenge_group", "run_type"], as_index=False)["weighted_iou"]
        .mean()
        .sort_values(["challenge_group", "run_type"])
    )
    challenges = list(grouped["challenge_group"].drop_duplicates())
    run_types = list(grouped["run_type"].drop_duplicates())
    x = np.arange(len(challenges))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, run_type in enumerate(run_types):
        values = []
        for challenge in challenges:
            subset = grouped[
                (grouped["challenge_group"] == challenge) & (grouped["run_type"] == run_type)
            ]
            values.append(float(subset["weighted_iou"].iloc[0]) if not subset.empty else np.nan)
        ax.bar(x + (idx - (len(run_types) - 1) / 2) * width, values, width, label=run_type)
    ax.set_ylabel("Mean weighted IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title("Robotic Challenge Group Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([item.replace("_", "\n") for item in challenges], fontsize=8)
    ax.legend(fontsize=8)
    annotate_bars(ax)
    savefig(output)


def plot_dataset_prompt_winners(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    idx = df.groupby(["dataset", "prompt_mode"])["mIoU"].idxmax()
    winners = df.loc[idx].copy().sort_values(["dataset", "prompt_mode"])
    winners["dataset_label"] = winners["dataset"].map(label_dataset)
    winners["model_label"] = winners["model"].map(label_model)
    winners["prompt_label"] = winners["prompt_mode"].map(label_prompt)

    labels = winners["dataset_label"] + "\n" + winners["prompt_label"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(range(len(winners)), winners["mIoU"], color="#4c78a8")
    ax.set_title("Best Zero-Shot Model per Dataset and Prompt")
    ax.set_ylabel("mIoU")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(winners)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    for bar, model in zip(bars, winners["model_label"], strict=True):
        ax.annotate(
            model,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
            xytext=(0, 2),
            textcoords="offset points",
        )
    savefig(output)
    return winners


def best_rows(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    idx = df.groupby(group_cols)[metric].idxmax()
    return df.loc[idx].sort_values(group_cols).reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row.get(column, "")
            if pd.isna(value):
                values.append("N/A")
            elif column in float_cols:
                values.append(f"{float(value):.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_recommendation_guide(
    output: Path,
    tradeoff: pd.DataFrame,
    zero_shot_all: pd.DataFrame,
    baseline: pd.DataFrame,
    winners: pd.DataFrame,
    challenge: pd.DataFrame,
    plots: dict[str, str],
) -> None:
    cuda = tradeoff[(tradeoff["device"] == "cuda") & tradeoff["mIoU"].notna() & tradeoff["fps"].notna()].copy()
    best_quality = best_rows(cuda, ["dataset"], "mIoU")
    best_tradeoff = best_rows(cuda, ["dataset"], "miou_fps_product")
    best_light = best_rows(
        cuda[cuda["model_group"] == "lightweight_sam"],
        ["dataset", "prompt_mode"],
        "miou_fps_product",
    )
    best_baseline = baseline.loc[baseline.groupby("dataset")["mIoU"].idxmax()].copy()
    low_challenges = challenge.sort_values("weighted_iou").head(8).copy()

    for df in [best_quality, best_tradeoff, best_light, best_baseline, winners, low_challenges]:
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].map(label_dataset)
        if "model" in df.columns:
            df["model"] = df["model"].map(label_model)
        if "baseline" in df.columns:
            df["baseline"] = df["baseline"].map(label_model)
        if "prompt_mode" in df.columns:
            df["prompt_mode"] = df["prompt_mode"].map(label_prompt)

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final Recommendation Guide",
        "",
        "This guide summarizes which segmentation model to use under the benchmark conditions. It uses the compact Task 6, Task 7, Task 8, and Task 9 outputs.",
        "",
        "## High-Level Recommendations",
        "",
        "- Use box prompts when a detector, tracker, or robot prior can provide a target region. Box prompting consistently dominates point prompting for quality.",
        "- Use SAM2 or SAM ViT-H/B when segmentation quality is the main goal and real-time operation is not required.",
        "- Use YOLOv8-seg or DeepLabV3+ when GPU real-time throughput is required. They are supervised baselines, so they depend on labeled data from the target domain.",
        "- Use MobileSAM for the best lightweight point/box speed-size trade-off. It is much smaller than SAM ViT-H/B and faster than EfficientSAM-S in this benchmark.",
        "- Use EfficientSAM-S when lightweight box-prompt quality matters more than speed. Its box-prompt mIoU is strong, but it is slower and larger than MobileSAM.",
        "- Avoid automatic mask generation for real-time robot loops in this setup. Automatic modes are much slower, and EfficientSAM grid automatic quality is weak.",
        "- Treat reflective/transparent objects, small/thin parts, and occluded robot-body regions as the highest-risk qualitative failure areas.",
        "",
        "## Plots",
        "",
    ]
    for name, path in plots.items():
        guide_path = Path(path)
        if guide_path.parts[:2] == ("outputs", "final_benchmark_assets"):
            guide_path = Path(*guide_path.parts[2:])
        lines.extend([f"### {name}", "", f"![{name}]({guide_path.as_posix()})", ""])

    lines.extend(
        [
            "",
            "## Best Overall CUDA Quality by Dataset",
            "",
            markdown_table(
                best_quality[["dataset", "model", "prompt_mode", "mIoU", "fps", "mask_AP"]],
                ["dataset", "model", "prompt_mode", "mIoU", "fps", "mask_AP"],
                {"mIoU", "fps", "mask_AP"},
            ),
            "",
            "## Best CUDA Speed-Quality Product by Dataset",
            "",
            markdown_table(
                best_tradeoff[["dataset", "model", "prompt_mode", "mIoU", "fps", "miou_fps_product"]],
                ["dataset", "model", "prompt_mode", "mIoU", "fps", "miou_fps_product"],
                {"mIoU", "fps", "miou_fps_product"},
            ),
            "",
            "## Best Lightweight CUDA Trade-Off by Dataset and Prompt",
            "",
            markdown_table(
                best_light[
                    [
                        "dataset",
                        "prompt_mode",
                        "model",
                        "mIoU",
                        "fps",
                        "checkpoint_size_mb",
                        "miou_fps_product",
                    ]
                ],
                [
                    "dataset",
                    "prompt_mode",
                    "model",
                    "mIoU",
                    "fps",
                    "checkpoint_size_mb",
                    "miou_fps_product",
                ],
                {"mIoU", "fps", "checkpoint_size_mb", "miou_fps_product"},
            ),
            "",
            "## Best Supervised Baseline by Dataset",
            "",
            markdown_table(
                best_baseline[["dataset", "baseline", "evaluation_type", "mIoU", "boundary_f1", "mask_AP"]],
                ["dataset", "baseline", "evaluation_type", "mIoU", "boundary_f1", "mask_AP"],
                {"mIoU", "boundary_f1", "mask_AP"},
            ),
            "",
            "## Best Zero-Shot Prompted Model by Dataset and Prompt",
            "",
            markdown_table(
                winners[["dataset", "prompt_mode", "model", "mIoU", "boundary_f1", "mask_AP"]],
                ["dataset", "prompt_mode", "model", "mIoU", "boundary_f1", "mask_AP"],
                {"mIoU", "boundary_f1", "mask_AP"},
            ),
            "",
            "## Lowest Challenge-Group Rows",
            "",
            markdown_table(
                low_challenges[
                    [
                        "run_type",
                        "dataset",
                        "model",
                        "prompt_mode",
                        "challenge_group",
                        "weighted_iou",
                        "mean_boundary_f1",
                    ]
                ],
                [
                    "run_type",
                    "dataset",
                    "model",
                    "prompt_mode",
                    "challenge_group",
                    "weighted_iou",
                    "mean_boundary_f1",
                ],
                {"weighted_iou", "mean_boundary_f1"},
            ),
            "",
            "## Mask R-CNN Implementation Note",
            "",
            "Mask R-CNN is implemented with TorchVision's `maskrcnn_resnet50_fpn` instead of Detectron2. This keeps the baseline reproducible in the existing PyTorch environment while still evaluating the requested Mask R-CNN baseline family. A Detectron2 run would be a duplicate Mask R-CNN implementation rather than a different baseline category.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root)
    plot_root = output_root / "plots"
    table_root = output_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    table_root.mkdir(parents=True, exist_ok=True)

    zero_shot = numeric_columns(
        read_csv("outputs/task6_evaluation/zero_shot/test/summary.csv"),
        ["mIoU", "boundary_f1", "mask_AP", "mask_AP50", "mask_AP75", "elapsed_s"],
    )
    task9_quality = numeric_columns(
        read_csv("outputs/task9_lightweight_sam/summary/lightweight_quality.csv"),
        ["mIoU", "boundary_f1", "mask_AP", "mask_AP50", "mask_AP75"],
    )
    task9_quality = task9_quality[task9_quality["model_group"] == "lightweight_sam"].copy()
    for label, frame in {
        "heavy zero-shot": zero_shot,
        "lightweight zero-shot": task9_quality,
    }.items():
        if "split" not in frame.columns or set(frame["split"].dropna()) != {"test"}:
            raise ValueError(f"{label} inputs must contain only split=test rows")
    task9_quality = task9_quality[
        [
            "dataset",
            "model",
            "prompt_mode",
            "split",
            "evaluation_images",
            "split_sha256",
            "status",
            "mIoU",
            "boundary_f1",
            "mask_AP",
            "mask_AP50",
            "mask_AP75",
        ]
    ]
    zero_shot_all = pd.concat([zero_shot, task9_quality], ignore_index=True, sort=False)
    baseline = numeric_columns(
        read_csv("outputs/task6_evaluation/baselines/test/summary.csv"),
        ["mIoU", "boundary_f1", "mask_AP", "mask_AP50", "mask_AP75", "elapsed_s"],
    )
    if "split" not in baseline.columns or set(baseline["split"].dropna()) != {"test"}:
        raise ValueError("baseline inputs must contain only split=test rows")
    expected_images = {
        dataset: int(group["evaluation_images"].iloc[0])
        for dataset, group in zero_shot.groupby("dataset")
    }
    expected_split_hashes = {
        dataset: str(group["split_sha256"].iloc[0])
        for dataset, group in zero_shot.groupby("dataset")
    }
    for label, frame in {
        "heavy zero-shot": zero_shot,
        "lightweight zero-shot": task9_quality,
        "baselines": baseline,
    }.items():
        for dataset, group in frame.groupby("dataset"):
            observed = {int(value) for value in group["evaluation_images"].dropna()}
            expected = expected_images.get(dataset)
            if expected is None or observed != {expected}:
                raise ValueError(
                    f"{label}/{dataset} does not use the common test image count: "
                    f"expected={expected}, observed={sorted(observed)}"
                )
            observed_hashes = set(group["split_sha256"].dropna().astype(str))
            expected_hash = expected_split_hashes.get(dataset)
            if expected_hash is None or observed_hashes != {expected_hash}:
                raise ValueError(
                    f"{label}/{dataset} does not use the identical test ID file: "
                    f"expected={expected_hash}, observed={sorted(observed_hashes)}"
                )
    tradeoff = numeric_columns(
        read_csv("outputs/task9_lightweight_sam/summary/speed_quality_tradeoff.csv"),
        [
            "mIoU",
            "boundary_f1",
            "mask_AP",
            "mask_AP50",
            "mask_AP75",
            "fps",
            "latency_mean_ms",
            "latency_p95_ms",
            "checkpoint_size_mb",
            "miou_fps_product",
            "miou_per_ms",
        ],
    )
    challenge = numeric_columns(
        read_csv("outputs/task8_failure_analysis/challenge_group_summary.csv"),
        ["weighted_iou", "mean_iou", "mean_boundary_f1", "category_count", "instance_count"],
    )

    plots = {
        "Dataset examples": rel(plot_root / "dataset_examples.png"),
        "Zero-shot mIoU heatmap": rel(plot_root / "zero_shot_miou_heatmap.png"),
        "Baseline mIoU bars": rel(plot_root / "baseline_miou_bars.png"),
        "CUDA speed-quality scatter": rel(plot_root / "cuda_speed_quality_scatter.png"),
        "Lightweight SAM trade-off": rel(plot_root / "lightweight_sam_tradeoff_cuda.png"),
        "Challenge group summary": rel(plot_root / "challenge_group_weighted_iou.png"),
        "Zero-shot winners": rel(plot_root / "zero_shot_dataset_prompt_winners.png"),
    }

    create_dataset_montage(plot_root / "dataset_examples.png")
    plot_zero_shot_heatmap(zero_shot_all, plot_root / "zero_shot_miou_heatmap.png")
    plot_baseline_bars(baseline, plot_root / "baseline_miou_bars.png")
    plot_speed_quality(tradeoff, plot_root / "cuda_speed_quality_scatter.png")
    plot_lightweight_tradeoff(tradeoff, plot_root / "lightweight_sam_tradeoff_cuda.png")
    plot_challenge_summary(challenge, plot_root / "challenge_group_weighted_iou.png")
    winners = plot_dataset_prompt_winners(zero_shot_all, plot_root / "zero_shot_dataset_prompt_winners.png")

    cuda = tradeoff[(tradeoff["device"] == "cuda") & tradeoff["mIoU"].notna()].copy()
    best_quality = best_rows(cuda, ["dataset"], "mIoU")
    best_tradeoff = best_rows(cuda, ["dataset"], "miou_fps_product")
    best_lightweight = best_rows(
        cuda[cuda["model_group"] == "lightweight_sam"],
        ["dataset", "prompt_mode"],
        "miou_fps_product",
    )
    best_quality.to_csv(table_root / "best_cuda_quality_by_dataset.csv", index=False)
    best_tradeoff.to_csv(table_root / "best_cuda_tradeoff_by_dataset.csv", index=False)
    best_lightweight.to_csv(table_root / "best_lightweight_cuda_tradeoff.csv", index=False)
    winners.to_csv(table_root / "best_zero_shot_by_dataset_prompt.csv", index=False)

    write_recommendation_guide(
        output=output_root / "recommendation_guide.md",
        tradeoff=tradeoff,
        zero_shot_all=zero_shot_all,
        baseline=baseline,
        winners=winners,
        challenge=challenge,
        plots=plots,
    )

    summary: dict[str, Any] = {
        "plots": plots,
        "tables": {
            "best_cuda_quality_by_dataset": rel(table_root / "best_cuda_quality_by_dataset.csv"),
            "best_cuda_tradeoff_by_dataset": rel(table_root / "best_cuda_tradeoff_by_dataset.csv"),
            "best_lightweight_cuda_tradeoff": rel(table_root / "best_lightweight_cuda_tradeoff.csv"),
            "best_zero_shot_by_dataset_prompt": rel(table_root / "best_zero_shot_by_dataset_prompt.csv"),
        },
        "recommendation_guide": rel(output_root / "recommendation_guide.md"),
        "source_rows": {
            "zero_shot_heavy": int(len(zero_shot)),
            "zero_shot_lightweight": int(len(task9_quality)),
            "baselines": int(len(baseline)),
            "speed_quality_tradeoff": int(len(tradeoff)),
            "challenge_groups": int(len(challenge)),
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {rel(output_root)}", flush=True)


if __name__ == "__main__":
    main()
