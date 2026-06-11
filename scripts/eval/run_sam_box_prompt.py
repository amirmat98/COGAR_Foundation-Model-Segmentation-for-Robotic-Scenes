import argparse
from typing import Any

from cogar_seg.evaluation import run_batch_sam_box, run_single_sam_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM with box prompts.")
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
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Do not write predicted mask PNGs; only write the results CSV.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows. Use 1 for every row, 0 to disable.",
    )
    return parser.parse_args()


def should_print_progress(counter: int, total: int, progress_every: int) -> bool:
    if progress_every <= 0:
        return False
    return counter == 1 or counter == total or counter % progress_every == 0


def print_progress(
    result: dict[str, Any],
    counter: int,
    total: int,
    progress_every: int,
) -> None:
    if not should_print_progress(counter, total, progress_every):
        return
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
        result = run_single_sam_box(
            config=args.config,
            index=args.index,
            row_index=args.row,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            output_dir=args.output_dir,
        )
        print("Saved SAM mask:", result.mask_output_path)
        print("Saved visualization:", result.visualization_output_path)
        print(f"SAM score: {result.sam_score:.4f}")
        print(f"IoU with GT mask: {result.iou:.4f}")
        return

    run = run_batch_sam_box(
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
        save_masks=not args.no_save_masks,
        progress_callback=lambda result, counter, total: print_progress(
            result,
            counter,
            total,
            args.progress_every,
        ),
    )

    results = run.results
    print("Saved results CSV:", run.config.results_csv_path)
    if not args.no_save_masks:
        print("Saved masks dir:", run.config.masks_dir)
    if not args.no_visualizations:
        print("Saved visualizations dir:", run.config.visualizations_dir)
    print(f"Rows evaluated: {len(results)}")
    print(f"Mean IoU: {results['iou'].mean():.4f}")
    print(f"Median IoU: {results['iou'].median():.4f}")
    print(f"Mean SAM score: {results['sam_score'].mean():.4f}")


if __name__ == "__main__":
    main()
