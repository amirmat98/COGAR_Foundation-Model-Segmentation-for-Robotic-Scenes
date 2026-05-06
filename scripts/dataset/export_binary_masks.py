import argparse

from cogar_seg.indexing import export_binary_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export binary ground-truth masks from an object index."
    )
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-mask-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = export_binary_masks(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_mask_dir=args.output_mask_dir,
        config_path=args.config,
    )

    print("Input index:", run.input_csv)
    print("Output index:", run.output_csv)
    print("Mask dir:", run.output_mask_dir)
    print("Number of binary masks:", run.num_masks)


if __name__ == "__main__":
    main()
