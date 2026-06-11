#!/usr/bin/env python3
"""Summarize the full-OCID benchmark index and optional model result CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the full OCID benchmark.")
    parser.add_argument(
        "--index",
        default="outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv",
    )
    parser.add_argument(
        "--image-index",
        default="outputs/ocid_full/indexes/ocid_full_images.csv",
    )
    parser.add_argument("--ocid-root", default="/mnt/Info/COGAR_DATASETs/OCID-dataset")
    parser.add_argument("--results", nargs="*", default=[])
    parser.add_argument("--output-md", default="docs/ocid_massive_benchmark_report.md")
    parser.add_argument("--tables-dir", default="outputs/ocid_full/tables")
    parser.add_argument(
        "--strict-results",
        action="store_true",
        help="Fail if any --results CSV is missing, empty, or unreadable.",
    )
    parser.add_argument("--debug", action="store_true", help="Print extra CSV diagnostics.")
    return parser.parse_args()


def maybe_read_csv(path: str | Path) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def read_csv_for_debug(path: str | Path, strict: bool = False) -> pd.DataFrame | None:
    """Read a CSV for reporting, optionally failing on missing/unreadable files."""
    csv_path = Path(path)
    if not csv_path.exists():
        if strict:
            raise FileNotFoundError(f"Required result CSV not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not read result CSV {csv_path}: {exc}") from exc
        print(f"[OCID][warn] Could not read result CSV {csv_path}: {exc}", flush=True)
        return None
    if df.empty and strict:
        raise RuntimeError(f"Required result CSV is empty: {csv_path}")
    return df


def write_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def value_counts_table(df: pd.DataFrame, column: str, count_name: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, count_name])
    return (
        df[column]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name=count_name)
    )


def summarize_results(
    result_paths: list[str],
    tables_dir: Path,
    strict: bool = False,
    debug: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for result_index, result_path_raw in enumerate(result_paths, start=1):
        result_path = Path(result_path_raw)
        print(
            f"[OCID] Reading result CSV {result_index:,}/{len(result_paths):,}: {result_path}",
            flush=True,
        )
        results = read_csv_for_debug(result_path, strict=strict)
        if results is None or results.empty:
            rows.append(
                {
                    "result_csv": str(result_path),
                    "status": "missing_or_empty",
                }
            )
            continue

        if debug:
            print(
                f"[OCID][debug] {result_path}: rows={len(results):,}, "
                f"columns={list(results.columns)}",
                flush=True,
            )

        row: dict[str, object] = {
            "result_csv": str(result_path),
            "status": "available",
            "rows": int(len(results)),
        }
        if "iou" in results.columns:
            iou = pd.to_numeric(results["iou"], errors="coerce")
            row["mean_iou"] = float(iou.mean())
            row["median_iou"] = float(iou.median())
            row["iou_ge_075"] = float((iou >= 0.75).mean())
            row["iou_lt_010"] = float((iou < 0.10).mean())
        if "boundary_f1" in results.columns:
            boundary_f1 = pd.to_numeric(results["boundary_f1"], errors="coerce")
            row["mean_boundary_f1"] = float(boundary_f1.mean())
        fps_cols = [c for c in ["fps", "image_gen_fps", "mean_fps"] if c in results.columns]
        if fps_cols:
            fps = pd.to_numeric(results[fps_cols[0]], errors="coerce")
            row["mean_fps"] = float(fps.mean())

        group_column = next(
            (column for column in ["scene_type", "category", "challenge"] if column in results.columns),
            None,
        )
        if group_column is not None and "iou" in results.columns:
            grouped = results.copy()
            grouped["iou"] = pd.to_numeric(grouped["iou"], errors="coerce")
            by_group = (
                grouped.groupby(group_column, dropna=False)
                .agg(rows=("iou", "size"), mean_iou=("iou", "mean"))
                .reset_index()
            )
            write_table(
                by_group,
                tables_dir / f"{result_path.stem}_by_{group_column}.csv",
            )

        rows.append(row)

    return rows


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    display_df = df.head(max_rows).copy()
    headers = [str(c) for c in display_df.columns]
    rows = [
        [str(value) for value in row]
        for row in display_df.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    image_index_path = Path(args.image_index)
    output_md = Path(args.output_md)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not index_path.exists():
        raise FileNotFoundError(f"OCID object index not found: {index_path}")

    print(f"[OCID] Reading object index: {index_path}", flush=True)
    objects = pd.read_csv(index_path)
    if objects.empty:
        raise RuntimeError(f"OCID object index is empty: {index_path}")
    print(f"[OCID] Object rows: {len(objects):,}", flush=True)
    if args.debug:
        print(f"[OCID][debug] Object index columns: {list(objects.columns)}", flush=True)

    print(f"[OCID] Reading image index: {image_index_path}", flush=True)
    images = maybe_read_csv(image_index_path)
    if images is None:
        print(f"[OCID][warn] Image index is missing: {image_index_path}", flush=True)
    elif args.debug:
        print(f"[OCID][debug] Image rows: {len(images):,}", flush=True)
        print(f"[OCID][debug] Image index columns: {list(images.columns)}", flush=True)

    object_set_counts = value_counts_table(objects, "object_set", "objects")
    surface_counts = value_counts_table(objects, "surface", "objects")
    camera_counts = value_counts_table(objects, "camera_view", "objects")
    scene_counts = value_counts_table(objects, "scene_type", "objects")
    sequence_counts = value_counts_table(objects, "sequence", "objects")

    write_table(object_set_counts, tables_dir / "object_set_counts.csv")
    write_table(surface_counts, tables_dir / "surface_counts.csv")
    write_table(camera_counts, tables_dir / "camera_view_counts.csv")
    write_table(scene_counts, tables_dir / "scene_type_counts.csv")
    write_table(sequence_counts, tables_dir / "sequence_counts.csv")

    result_summaries = summarize_results(
        args.results,
        tables_dir,
        strict=args.strict_results,
        debug=args.debug,
    )
    result_summary_df = pd.DataFrame(result_summaries)
    if not result_summary_df.empty:
        write_table(result_summary_df, tables_dir / "result_summary.csv")

    image_count = int(objects["image_path"].nunique()) if "image_path" in objects.columns else 0
    sequence_count = int(objects["sequence"].nunique()) if "sequence" in objects.columns else 0
    exported_mask_count = int(objects["binary_mask_path"].notna().sum()) if "binary_mask_path" in objects.columns else 0
    source_image_count = int(len(images)) if images is not None else image_count

    lines = [
        "# OCID Massive Benchmark Report",
        "",
        "## Dataset",
        "",
        f"- Source root: `{args.ocid_root}`",
        f"- Image index: `{image_index_path}`",
        f"- Object index: `{index_path}`",
        f"- Source RGB-label pairs: {source_image_count:,}",
        f"- Indexed RGB images after filtering: {image_count:,}",
        f"- Object instances after filtering: {len(objects):,}",
        f"- Exported binary object masks: {exported_mask_count:,}",
        f"- Sequences: {sequence_count:,}",
        "",
        "## Object Set Distribution",
        "",
        markdown_table(object_set_counts),
        "",
        "## Scene Type Distribution",
        "",
        markdown_table(scene_counts),
        "",
        "## Surface Distribution",
        "",
        markdown_table(surface_counts),
        "",
        "## Camera View Distribution",
        "",
        markdown_table(camera_counts),
        "",
        "## Result CSV Summary",
        "",
        markdown_table(result_summary_df) if not result_summary_df.empty else "_No result CSVs were provided._",
        "",
        "## Compatible Evaluation Commands",
        "",
        "SAM ViT-B box prompts:",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/eval/run_sam_box_prompt.py \\",
        "  --config configs/paths.yaml \\",
        "  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \\",
        "  --checkpoint checkpoints/sam_vit_b_01ec64.pth \\",
        "  --model-type vit_b \\",
        "  --device auto \\",
        "  --output-dir outputs/ocid_full/sam_vit_b_box \\",
        "  --results-csv outputs/ocid_full/results/sam_vit_b_box.csv \\",
        "  --no-visualizations \\",
        "  --no-save-masks \\",
        "  --progress-every 500",
        "```",
        "",
        "SAM ViT-B point prompts:",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/eval/run_sam_point_prompt.py \\",
        "  --config configs/paths.yaml \\",
        "  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \\",
        "  --checkpoint checkpoints/sam_vit_b_01ec64.pth \\",
        "  --model-type vit_b \\",
        "  --device auto \\",
        "  --output-dir outputs/ocid_full/sam_vit_b_point \\",
        "  --results-csv outputs/ocid_full/results/sam_vit_b_point.csv \\",
        "  --no-visualizations \\",
        "  --no-save-masks \\",
        "  --progress-every 500",
        "```",
        "",
        "SAM ViT-B automatic masks:",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/eval/run_sam_auto_masks.py \\",
        "  --config configs/paths.yaml \\",
        "  --index outputs/ocid_full/indexes/ocid_full_objects_filtered_with_masks.csv \\",
        "  --checkpoint checkpoints/sam_vit_b_01ec64.pth \\",
        "  --model-type vit_b \\",
        "  --device auto \\",
        "  --output-dir outputs/ocid_full/sam_vit_b_auto_fast16 \\",
        "  --results-csv outputs/ocid_full/results/sam_vit_b_auto_fast16.csv \\",
        "  --points-per-side 16 \\",
        "  --pred-iou-thresh 0.90 \\",
        "  --stability-score-thresh 0.92 \\",
        "  --no-save-masks \\",
        "  --progress-every 500",
        "```",
        "",
        "Regenerate this report after result CSVs exist:",
        "",
        "```bash",
        "PYTHONPATH=src python3 scripts/analysis/summarize_ocid_massive_benchmark.py \\",
        "  --results outputs/ocid_full/results/sam_vit_b_box.csv \\",
        "            outputs/ocid_full/results/sam_vit_b_point.csv \\",
        "            outputs/ocid_full/results/sam_vit_b_auto_fast16.csv \\",
        "            outputs/ocid_full/results/sam_vit_b_auto.csv \\",
        "            outputs/ocid_full/results/fastsam_s_box.csv \\",
        "            outputs/ocid_full/results/mobilesam_box.csv \\",
        "            outputs/ocid_full/fastsam_s_point/fastsam_s_point_per_instance.csv \\",
        "            outputs/ocid_full/fastsam_s_auto/fastsam_s_auto_per_instance.csv \\",
        "            outputs/ocid_full/sam2_tiny_box/sam2_1-tiny_box_per_instance.csv \\",
        "            outputs/ocid_full/sam2_tiny_point/sam2_1-tiny_point_per_instance.csv \\",
        "            outputs/ocid_full/sam2_tiny_auto/sam2_1-tiny_auto_per_instance.csv \\",
        "            outputs/ocid_full/efficientsam_ti_box/efficientsam-ti_box_per_instance.csv",
        "```",
        "",
        "## Notes",
        "",
        "- This is a real-world OCID generalization benchmark, separate from the simulated COGAR-SimRobotics-500 assignment benchmark.",
        "- The index uses OCID instance-label images to derive object masks, boxes, and point prompts.",
        "- Additional FastSAM, MobileSAM, SAM2, and EfficientSAM OCID commands are documented in `docs/ocid_massive_benchmark.md`.",
        "- AWS packaging and execution commands are documented in `docs/aws_ocid_benchmark.md`.",
        "- Full-model runs can be expensive; use `--limit` for smoke tests before launching full OCID runs.",
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote report: {output_md}")
    print(f"Wrote tables: {tables_dir}")


if __name__ == "__main__":
    main()
