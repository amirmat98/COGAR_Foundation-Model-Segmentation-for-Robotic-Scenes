import argparse
from typing import Any

from cogar_seg.evaluation import run_batch_sam_point, run_single_sam_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM with positive point prompts.")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument(
        "--index",
        default="outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv",
    )
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--no-visualizations", action="store_true")
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

    if args.row is not None:
        result = run_single_sam_point(
            config=args.config,
            index=args.index,
            row_index=args.row,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            output_dir=args.output_dir,
        )
        print("Saved SAM point mask:", result.mask_output_path)
        print("Saved point visualization:", result.visualization_output_path)
        print(f"SAM score: {result.sam_score:.4f}")
        print(f"IoU with GT mask: {result.iou:.4f}")
        return

    run = run_batch_sam_point(
        config=args.config,
        index=args.index,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        output_dir=args.output_dir,
        results_csv=args.results_csv,
        max_rows=args.limit,
        start_row=args.start_row,
        split=args.split,
        save_visualizations=not args.no_visualizations,
        progress_callback=print_progress,
    )

    print("Saved results CSV:", run.results_csv_path)
    print("Saved masks dir:", run.output_dir / "masks")
    if not args.no_visualizations:
        print("Saved visualizations dir:", run.output_dir / "visualizations")
    print(f"Rows evaluated: {run.num_rows}")
    print(f"Mean IoU: {run.mean_iou:.4f}")
    print(f"Median IoU: {run.median_iou:.4f}")
    print(f"Mean SAM score: {run.mean_sam_score:.4f}")


if __name__ == "__main__":
    main()
