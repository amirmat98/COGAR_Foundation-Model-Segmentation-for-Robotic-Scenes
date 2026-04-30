import argparse

from cogar_seg.evaluation import run_batch_sam_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch SAM evaluation using positive point prompts."
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
        help="Output directory. If omitted, uses outputs/sam_point_prompt_batch.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_batch_sam_point(
        config=args.config,
        index=args.index,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        allow_cpu_fallback=args.allow_cpu_fallback,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
    )

    print()
    print("Done.")
    print("Config:", result.config_path)
    print("Index:", result.index_path)
    print("Checkpoint:", result.checkpoint_path)
    print("Output dir:", result.output_dir)
    print("Results CSV:", result.results_csv_path)
    print("Rows evaluated:", result.num_rows)
    print("Device:", result.device)
    print(f"Mean IoU: {result.mean_iou:.4f}")
    print(f"Median IoU: {result.median_iou:.4f}")
    print(f"Mean SAM score: {result.mean_sam_score:.4f}")


if __name__ == "__main__":
    main()
