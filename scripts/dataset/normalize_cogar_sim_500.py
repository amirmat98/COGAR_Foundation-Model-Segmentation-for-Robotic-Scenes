import argparse

from cogar_seg.datasets import normalize_cogar_sim_500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw BlenderProc COGAR-SimRobotics-500 output."
    )
    parser.add_argument(
        "--raw-coco-dir",
        default="data/cogar_sim_500/raw_blenderproc/pilot_v2_ocid_like/coco_data",
    )
    parser.add_argument(
        "--raw-metadata",
        default="data/cogar_sim_500/metadata/frame_index_pilot_v2.csv",
    )
    parser.add_argument("--output-root", default="data/cogar_sim_500")
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument("--expected-images", type=int, default=500)
    parser.add_argument("--keep-existing-rgb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = normalize_cogar_sim_500(
        raw_coco_dir=args.raw_coco_dir,
        raw_metadata_path=args.raw_metadata,
        output_root=args.output_root,
        config_path=args.config,
        expected_images=args.expected_images,
        clean_rgb_dir=not args.keep_existing_rgb,
    )

    print("[OK] Normalized COGAR-SimRobotics-500 dataset created.")
    print("RGB:", run.rgb_dir)
    print("Annotations:", run.annotations_path)
    print("Metadata:", run.metadata_path)
    print("Categories:", run.categories_path)
    print("Splits:", run.splits_dir)
    print("Images:", run.num_images)
    print("Annotations:", run.num_annotations)
    print("Metadata rows:", run.num_metadata_rows)


if __name__ == "__main__":
    main()
