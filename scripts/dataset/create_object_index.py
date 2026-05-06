import argparse

from cogar_seg.indexing import create_cogar_sim_object_index, create_ocid_object_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create object-level dataset indexes.")
    parser.add_argument("--dataset", choices=["ocid_debug", "cogar_sim_500"], default="ocid_debug")
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--coco", default="data/cogar_sim_500/annotations/instances_all.json")
    parser.add_argument("--metadata", default="data/cogar_sim_500/metadata/frame_index.csv")
    parser.add_argument("--rgb-dir", default="data/cogar_sim_500/rgb")
    parser.add_argument("--output", default="outputs/indexes/cogar_sim_500_objects.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "ocid_debug":
        paths, num_images, num_objects, num_filtered = create_ocid_object_index(args.config)
        print("Image index:", paths.image_index_csv)
        print("Object index:", paths.object_index_csv)
        print("Filtered object index:", paths.filtered_object_index_csv)
        print("Number of RGB-label pairs:", num_images)
        print("Number of object instances:", num_objects)
        print("Number of filtered object instances:", num_filtered)
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
