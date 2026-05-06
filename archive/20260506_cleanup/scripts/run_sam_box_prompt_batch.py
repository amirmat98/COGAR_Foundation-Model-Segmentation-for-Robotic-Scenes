import argparse
from typing import Any

from cogar_seg.evaluation import run_batch_sam_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM box-prompt inference over all filtered OCID objects."
    )

    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Path to project paths YAML file.",
    )
    parser.add_argument(
        "--index",
        default="outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv",
        help="Path to final object-level CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint.",
    )
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="If CUDA fails, continue on CPU instead of stopping.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, uses sam_outputs_dir from configs/paths.yaml.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path to output results CSV.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of rows to process for debugging.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start processing from this row index.",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable saving visualization PNGs.",
    )

    return parser.parse_args()


def print_progress(result: dict[str, Any], counter: int, total: int) -> None:
    print(
        f"[{counter:03d}/{total:03d}] "
        f"row={int(result['row_index']):04d} "
        f"obj={result['object_id']} "
        f"score={result['sam_score']:.4f} "
        f"IoU={result['iou']:.4f}"
    )


def main() -> None:
    args = parse_args()

    run = run_batch_sam_box(
        config=args.config,
        index=args.index,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        max_rows=args.max_rows,
        start_row=args.start_row,
        save_visualizations=not args.no_visualizations,
        progress_callback=print_progress,
    )

    batch_cfg = run.config
    results_df = run.results

    print("Using config:", batch_cfg.config_path)
    print("Project root:", batch_cfg.project_root)
    print("OCID root:", batch_cfg.ocid_root)
    print("Index path:", batch_cfg.index_path)
    print("Checkpoint path:", batch_cfg.checkpoint_path)
    print("Output dir:", batch_cfg.output_dir)
    print("Results CSV:", batch_cfg.results_csv_path)
    print("Device:", run.device)
    print("Model type:", run.model_type)
    print()
    print("Done.")
    print("Saved results CSV:", batch_cfg.results_csv_path)
    print("Saved masks dir:", batch_cfg.masks_dir)

    if not args.no_visualizations:
        print("Saved visualizations dir:", batch_cfg.visualizations_dir)

    print()
    print("Summary:")
    print(f"Number of evaluated objects: {len(results_df)}")
    print(f"Mean IoU: {results_df['iou'].mean():.4f}")
    print(f"Median IoU: {results_df['iou'].median():.4f}")
    print(f"Min IoU: {results_df['iou'].min():.4f}")
    print(f"Max IoU: {results_df['iou'].max():.4f}")
    print(f"Mean SAM score: {results_df['sam_score'].mean():.4f}")


if __name__ == "__main__":
    main()
