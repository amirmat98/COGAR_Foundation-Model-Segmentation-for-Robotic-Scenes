from cogar_seg.config import load_config, get_outputs_dir
from cogar_seg.dataset_ocid import create_image_index, create_object_index
from cogar_seg.masks import filter_object_index, export_binary_gt_masks


def main():
    config = load_config()

    outputs_dir = get_outputs_dir(config)
    index_dir = outputs_dir / "indexes"
    mask_dir = outputs_dir / "gt_binary_masks"

    image_index_csv = index_dir / "ocid_debug_seq21.csv"
    object_index_csv = index_dir / "ocid_debug_seq21_objects.csv"
    filtered_object_index_csv = index_dir / "ocid_debug_seq21_objects_filtered.csv"
    final_object_index_csv = index_dir / "ocid_debug_seq21_objects_filtered_with_masks.csv"

    print("Creating image-level index...")
    num_images = create_image_index(config, image_index_csv)
    print(f"Saved: {image_index_csv}")
    print(f"Number of RGB-label pairs: {num_images}")

    print("\nCreating object-level index...")
    num_objects = create_object_index(image_index_csv, object_index_csv)
    print(f"Saved: {object_index_csv}")
    print(f"Number of object instances: {num_objects}")

    print("\nFiltering object-level index...")
    num_filtered = filter_object_index(object_index_csv, filtered_object_index_csv)
    print(f"Saved: {filtered_object_index_csv}")
    print(f"Number of filtered object instances: {num_filtered}")

    print("\nExporting binary ground-truth masks...")
    num_masks = export_binary_gt_masks(
        input_csv=filtered_object_index_csv,
        output_csv=final_object_index_csv,
        output_mask_dir=mask_dir,
    )
    print(f"Saved masks to: {mask_dir}")
    print(f"Saved final index: {final_object_index_csv}")
    print(f"Number of binary masks: {num_masks}")


if __name__ == "__main__":
    main()