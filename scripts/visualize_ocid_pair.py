from pathlib import Path
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt


CONFIG_PATH = Path("configs/paths.yaml")


def colorize_label(label: np.ndarray) -> np.ndarray:
    """
    Convert an integer label mask into a colored RGB image for visualization.
    Each object ID gets a different color.
    """
    colored = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # Fixed random seed gives the same colors every time.
    rng = np.random.default_rng(seed=42)

    for object_id in np.unique(label):
        if object_id == 0:
            continue

        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        colored[label == object_id] = color

    return colored


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])

    seq_path = ocid_root / sequence
    rgb_dir = seq_path / config["rgb_folder_name"]
    label_dir = seq_path / config["label_folder_name"]

    rgb_files = sorted(rgb_dir.glob("*.png"))

    first_rgb = rgb_files[0]
    first_label = label_dir / first_rgb.name

    rgb_bgr = cv2.imread(str(first_rgb), cv2.IMREAD_COLOR)
    label = cv2.imread(str(first_label), cv2.IMREAD_UNCHANGED)

    if rgb_bgr is None:
        raise RuntimeError(f"Could not read RGB image: {first_rgb}")

    if label is None:
        raise RuntimeError(f"Could not read label mask: {first_label}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    colored_label = colorize_label(label)

    print("Visualizing:")
    print("RGB:", first_rgb)
    print("Label:", first_label)
    print("Unique label IDs:", np.unique(label))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("OCID RGB image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(colored_label)
    plt.title("Ground-truth instance label mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()