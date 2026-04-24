from pathlib import Path
import csv
import yaml


CONFIG_PATH = Path("configs/paths.yaml")


def main():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    ocid_root = Path(config["ocid_root"])
    sequence = Path(config["ocid_debug_sequence"])

    seq_path = ocid_root / sequence
    rgb_dir = seq_path / config["rgb_folder_name"]
    label_dir = seq_path / config["label_folder_name"]

    output_dir = Path(config["outputs_dir"]) / "indexes"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "ocid_debug_seq21.csv"

    rgb_files = sorted(rgb_dir.glob("*.png"))

    rows = []

    for rgb_path in rgb_files:
        label_path = label_dir / rgb_path.name

        if not label_path.exists():
            print(f"Skipping {rgb_path.name}: matching label not found")
            continue

        rows.append({
            "image_path": str(rgb_path),
            "label_path": str(label_path),
            "sequence": str(sequence),
            "file_name": rgb_path.name,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label_path", "sequence", "file_name"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved index to: {output_csv}")
    print(f"Number of valid RGB-label pairs: {len(rows)}")


if __name__ == "__main__":
    main()