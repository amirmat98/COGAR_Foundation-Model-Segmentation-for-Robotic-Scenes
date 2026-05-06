import argparse

from cogar_seg.evaluation import run_single_sam_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM on one OCID object using a bounding-box prompt."
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
        "--row",
        type=int,
        default=0,
        help="Object row number to test.",
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
        help="Inference device. Use 'cuda' to require GPU, 'auto' to use GPU if compatible.",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    print("Using config:", result.config_path)
    print("Project root:", result.project_root)
    print("OCID root:", result.ocid_root)
    print("Index path:", result.index_path)
    print("Checkpoint path:", result.checkpoint_path)
    print("Output dir:", result.output_dir)
    print()
    print("Selected row:", result.row_index)
    print("Object ID:", result.object_id)
    print("Original CSV image path:", result.original_image_path)
    print("Resolved image path:", result.image_path)
    print("GT mask path:", result.gt_mask_path)
    print("Box prompt XYXY:", result.box_xyxy)
    print("Device:", result.device)
    print()
    print("Done.")
    print("Saved SAM mask:", result.mask_output_path)
    print("Saved visualization:", result.visualization_output_path)
    print(f"SAM score: {result.sam_score:.4f}")
    print(f"IoU with GT mask: {result.iou:.4f}")


if __name__ == "__main__":
    main()
