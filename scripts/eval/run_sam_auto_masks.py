import argparse
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
    return parser.parse_args()


def print_progress(result: dict[str, Any], counter: int, total: int) -> None:
    print(
        f"[{counter:03d}/{total:03d}] "
        f"row={int(result['row_index']):04d} "
        f"obj={result['object_id']} "
        f"masks={result['generated_mask_count']} "
        f"IoU={result['iou']:.4f}"
    )


def main() -> None:
    args = parse_args()
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
        progress_callback=print_progress,
    )

    print("Saved results CSV:", run.results_csv_path)
    print("Saved masks dir:", run.masks_dir)
    print(f"Rows evaluated: {len(run.results)}")
    print(f"Mean IoU: {run.results['iou'].mean():.4f}")


if __name__ == "__main__":
    main()
