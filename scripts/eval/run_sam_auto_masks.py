import argparse
import time
from typing import Any

from cogar_seg.evaluation import run_batch_sam_automatic_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM automatic mask evaluation.")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument(
        "--index",
        default="outputs/indexes/ocid_debug_seq21_objects_filtered_with_masks.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N rows. Use 1 for every row, 0 to disable.",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="SAM automatic grid density. Default uses SAM's built-in default; 16 is faster than 32.",
    )
    parser.add_argument("--pred-iou-thresh", type=float, default=None)
    parser.add_argument("--stability-score-thresh", type=float, default=None)
    parser.add_argument("--crop-n-layers", type=int, default=None)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=None)
    parser.add_argument("--min-mask-region-area", type=int, default=None)
    parser.add_argument(
        "--no-save-masks",
        action="store_true",
        help="Do not write per-object predicted mask PNGs; only write the CSV metrics.",
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
    start_time: float,
) -> None:
    if not should_print_progress(counter, total, progress_every):
        return
    elapsed = time.perf_counter() - start_time
    rows_per_sec = counter / elapsed if elapsed > 0 else 0.0
    remaining = (total - counter) / rows_per_sec if rows_per_sec > 0 else 0.0
    print(
        f"[{counter:03d}/{total:03d}] "
        f"row={int(result['row_index']):04d} "
        f"obj={result['object_id']} "
        f"masks={result['generated_mask_count']} "
        f"IoU={result['iou']:.4f} "
        f"elapsed={elapsed/60:.1f}m "
        f"eta={remaining/60:.1f}m"
    )


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    run = run_batch_sam_automatic_masks(
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
        progress_callback=lambda result, counter, total: print_progress(
            result,
            counter,
            total,
            args.progress_every,
            start_time,
        ),
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,
        save_masks=not args.no_save_masks,
    )

    print("Saved results CSV:", run.results_csv_path)
    print("Saved masks dir:", run.masks_dir)
    print(f"Rows evaluated: {len(run.results)}")
    print(f"Mean IoU: {run.results['iou'].mean():.4f}")
    print(f"Elapsed minutes: {(time.perf_counter() - start_time) / 60:.2f}")


if __name__ == "__main__":
    main()
