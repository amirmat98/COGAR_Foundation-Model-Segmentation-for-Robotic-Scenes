import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate randomized BlenderProc scenes for COGAR-SimRobotics-500."
    )
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--raw-dataset-name", default="pilot_v2_ocid_like")
    parser.add_argument("--no-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.blenderproc_scene import generate_cogar_sim_500

    generate_cogar_sim_500(
        config_path=args.config,
        num_images=args.num_images,
        raw_dataset_name=args.raw_dataset_name,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
