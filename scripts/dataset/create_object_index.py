import argparse

from cogar_seg.indexing import create_ocid_object_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create OCID debug image, object, and filtered object indexes."
    )
    parser.add_argument("--config", default="configs/paths.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, num_images, num_objects, num_filtered = create_ocid_object_index(args.config)

    print("Image index:", paths.image_index_csv)
    print("Object index:", paths.object_index_csv)
    print("Filtered object index:", paths.filtered_object_index_csv)
    print("Number of RGB-label pairs:", num_images)
    print("Number of object instances:", num_objects)
    print("Number of filtered object instances:", num_filtered)


if __name__ == "__main__":
    main()
