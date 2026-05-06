import argparse

from cogar_seg.indexing.mask_export import (
    export_binary_masks,
    export_cogar_sim_binary_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export binary ground-truth masks from an object index."
    )

    parser.add_argument(
        "--dataset",
        choices=["ocid_debug", "cogar_sim_500"],
        default="ocid_debug",
        help="Dataset workflow to use.",
    )

    # OCID arguments
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--input-csv", default=None)

    # COGAR-Sim arguments
    parser.add_argument("--coco", default=None)
    parser.add_argument("--object-index", default=None)

    # Shared outputs
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-mask-dir", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "ocid_debug":
        run = export_binary_masks(
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            output_mask_dir=args.output_mask_dir,
            config_path=args.config,
        )
    else:
        if args.coco is None:
            raise ValueError("--coco is required when --dataset cogar_sim_500")
        if args.object_index is None:
            raise ValueError("--object-index is required when --dataset cogar_sim_500")
        if args.output_csv is None:
            raise ValueError("--output-csv is required when --dataset cogar_sim_500")
        if args.output_mask_dir is None:
            raise ValueError("--output-mask-dir is required when --dataset cogar_sim_500")

        run = export_cogar_sim_binary_masks(
            coco_path=args.coco,
            object_index_csv=args.object_index,
            output_csv=args.output_csv,
            output_mask_dir=args.output_mask_dir,
        )

    print("Input index:", run.input_csv)
    print("Output index:", run.output_csv)
    print("Mask dir:", run.output_mask_dir)
    print("Number of binary masks:", run.num_masks)


if __name__ == "__main__":
    main()
