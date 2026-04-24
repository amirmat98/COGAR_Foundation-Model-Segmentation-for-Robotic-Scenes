from pathlib import Path
import yaml
import cv2
import numpy as np


CONFIG_PATH = Path("configs/paths.yaml")


def main():
    # Read config file safely
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])

    seq_path = ocid_root / sequence
    rgb_dir = seq_path / config["rgb_folder_name"]
    label_dir = seq_path / config["label_folder_name"]

    print("OCID root:", ocid_root)
    print("Sequence path:", seq_path)
    print("RGB directory:", rgb_dir)
    print("Label directory:", label_dir)

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB folder not found: {rgb_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {label_dir}")

    rgb_files = sorted(rgb_dir.glob("*.png"))
    label_files = sorted(label_dir.glob("*.png"))

    print("Number of RGB images:", len(rgb_files))
    print("Number of label masks:", len(label_files))

    if len(rgb_files) == 0:
        raise RuntimeError("No RGB images found.")

    first_rgb = rgb_files[0]
    first_label = label_dir / first_rgb.name

    if not first_label.exists():
        raise FileNotFoundError(f"Matching label not found for: {first_rgb.name}")

    print("First RGB file:", first_rgb)
    print("Matching label file:", first_label)

    rgb = cv2.imread(str(first_rgb), cv2.IMREAD_COLOR)
    label = cv2.imread(str(first_label), cv2.IMREAD_UNCHANGED)

    if rgb is None:
        raise RuntimeError(f"Could not read RGB image: {first_rgb}")

    if label is None:
        raise RuntimeError(f"Could not read label image: {first_label}")

    print("RGB shape:", rgb.shape)
    print("RGB dtype:", rgb.dtype)

    print("Label shape:", label.shape)
    print("Label dtype:", label.dtype)
    print("Unique label IDs:", np.unique(label))


if __name__ == "__main__":
    main()