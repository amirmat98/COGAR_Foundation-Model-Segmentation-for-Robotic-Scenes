import argparse
from pathlib import Path
import tempfile
import yaml

from cogar_seg.config import load_config
from cogar_seg.indexing import (
    create_cogar_sim_object_index,
    create_ocid_object_index,
    prepare_ocid_full_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create object-level dataset indexes.")
    parser.add_argument(
        "--dataset",
        choices=["ocid_debug", "ocid_full", "cogar_sim_500"],
        default="ocid_debug",
    )
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--ocid-root", default=None)
    parser.add_argument("--ocid-min-area", type=int, default=None)
    parser.add_argument("--ocid-max-area-ratio", type=float, default=None)
    parser.add_argument("--ocid-max-bbox-area-ratio", type=float, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress every N processed rows or sequences for OCID workflows.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress output for OCID workflows.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra OCID diagnostics such as sample paths and first rows.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/unreadable OCID labels instead of warning and skipping.",
    )
    parser.add_argument("--coco", default="data/cogar_sim_500/annotations/instances_all.json")
    parser.add_argument("--metadata", default="data/cogar_sim_500/metadata/frame_index.csv")
    parser.add_argument("--rgb-dir", default="data/cogar_sim_500/rgb")
    parser.add_argument("--output", default="outputs/indexes/cogar_sim_500_objects.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_progress = not args.quiet

    if args.dataset == "ocid_debug":
        paths, num_images, num_objects, num_filtered = create_ocid_object_index(
            args.config,
            progress=show_progress,
            progress_every=args.progress_every,
            debug=args.debug,
            strict=args.strict,
        )
        print("Image index:", paths.image_index_csv)
        print("Object index:", paths.object_index_csv)
        print("Filtered object index:", paths.filtered_object_index_csv)
        print("Number of RGB-label pairs:", num_images)
        print("Number of object instances:", num_objects)
        print("Number of filtered object instances:", num_filtered)
        return

    if args.dataset == "ocid_full":
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_config(config_path)
        if args.ocid_root is not None:
            config["ocid_root"] = args.ocid_root
        if args.ocid_min_area is not None:
            config["ocid_min_area"] = args.ocid_min_area
        if args.ocid_max_area_ratio is not None:
            config["ocid_max_area_ratio"] = args.ocid_max_area_ratio
        if args.ocid_max_bbox_area_ratio is not None:
            config["ocid_max_bbox_area_ratio"] = args.ocid_max_bbox_area_ratio

        ocid_root = Path(config["ocid_root"])
        if not ocid_root.exists():
            raise FileNotFoundError(f"OCID root does not exist: {ocid_root}")

        if show_progress:
            print("[OCID] Full benchmark index build", flush=True)
            print(f"[OCID] Config: {config_path}", flush=True)
            print(f"[OCID] Root: {ocid_root}", flush=True)
            print(f"[OCID] Outputs dir: {config.get('outputs_dir', 'outputs')}/ocid_full", flush=True)
            print(f"[OCID] Progress interval: {args.progress_every}", flush=True)
            print(f"[OCID] Strict mode: {args.strict}", flush=True)

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(config, tmp, sort_keys=False)
            tmp_config_path = Path(tmp.name)

        try:
            run = prepare_ocid_full_dataset(
                tmp_config_path,
                progress=show_progress,
                progress_every=args.progress_every,
                debug=args.debug,
                strict=args.strict,
            )
        finally:
            tmp_config_path.unlink(missing_ok=True)

        print("Full OCID image index:", run.paths.image_index_csv)
        print("Full OCID object index:", run.paths.object_index_csv)
        print("Full OCID filtered object index:", run.paths.filtered_object_index_csv)
        print("Full OCID final object index:", run.paths.final_object_index_csv)
        print("Full OCID binary mask dir:", run.paths.mask_dir)
        print("Number of RGB-label pairs:", run.num_images)
        print("Number of object instances:", run.num_objects)
        print("Number of filtered object instances:", run.num_filtered_objects)
        print("Number of exported binary masks:", run.num_masks)
        return

    run = create_cogar_sim_object_index(
        coco_path=args.coco,
        metadata_path=args.metadata,
        rgb_dir=args.rgb_dir,
        output_csv=args.output,
    )
    print("COCO annotations:", run.coco_path)
    print("Frame metadata:", run.metadata_path)
    print("RGB dir:", run.rgb_dir)
    print("Object index:", run.output_csv)
    print("Images:", run.num_images)
    print("Metadata rows:", run.num_metadata_rows)
    print("Annotations:", run.num_annotations)
    print("Rows written:", run.num_rows)


if __name__ == "__main__":
    main()
