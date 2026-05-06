import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed-camera BlenderProc pilot dataset."
    )
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from cogar_seg.generation.blenderproc_pilot import generate_pilot_dataset

    generate_pilot_dataset(config_path=args.config)


if __name__ == "__main__":
    main()
