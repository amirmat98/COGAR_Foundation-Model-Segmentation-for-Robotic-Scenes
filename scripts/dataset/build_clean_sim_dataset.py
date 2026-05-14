import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], env: dict[str, str]) -> None:
    print("\n[RUN]", shlex.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def prepend_pythonpath(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not current else f"{src}:{current}"
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, finalize, validate, audit, and optionally filter COGAR-Sim after BlenderProc generation."
    )
    parser.add_argument("--raw-coco-dir", required=True)
    parser.add_argument("--raw-metadata", required=True)
    parser.add_argument("--output-root", default="data/cogar_sim_500")
    parser.add_argument("--config", default="configs/blenderproc_dataset.yaml")
    parser.add_argument("--expected-images", type=int, default=500)
    parser.add_argument("--index-output", default="outputs/indexes/cogar_sim_500_objects.csv")
    parser.add_argument("--mask-dir", default="data/cogar_sim_500/instance_masks/v1")
    parser.add_argument("--final-index", default="data/cogar_sim_500/annotations/sim_robotic_scenes_index_v1.csv")
    parser.add_argument("--audit-output-dir", default="outputs/tables/dataset_audit_v1")
    parser.add_argument("--filtered-index", default=None)
    parser.add_argument("--exclude-categories", nargs="*", default=["table"])
    parser.add_argument("--min-area", type=float, default=25.0)
    parser.add_argument("--max-objects-per-image", type=int, default=25)
    parser.add_argument("--filter-bad", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    env = prepend_pythonpath(repo_root)

    output_root = Path(args.output_root)
    coco_path = output_root / "annotations" / "instances_all.json"
    metadata_path = output_root / "metadata" / "frame_index.csv"
    rgb_dir = output_root / "rgb"
    index_output = Path(args.index_output)
    mask_index = index_output.with_name(f"{index_output.stem}_with_masks.csv")
    final_index = Path(args.final_index)
    audit_output_dir = Path(args.audit_output_dir)

    python = sys.executable
    run_command(
        [
            python,
            "scripts/dataset/normalize_cogar_sim_500.py",
            "--raw-coco-dir",
            args.raw_coco_dir,
            "--raw-metadata",
            args.raw_metadata,
            "--output-root",
            args.output_root,
            "--config",
            args.config,
            "--expected-images",
            str(args.expected_images),
        ],
        env,
    )
    run_command(
        [
            python,
            "scripts/dataset/create_object_index.py",
            "--dataset",
            "cogar_sim_500",
            "--coco",
            str(coco_path),
            "--metadata",
            str(metadata_path),
            "--rgb-dir",
            str(rgb_dir),
            "--output",
            str(index_output),
        ],
        env,
    )
    run_command(
        [
            python,
            "scripts/dataset/export_binary_masks.py",
            "--dataset",
            "cogar_sim_500",
            "--coco",
            str(coco_path),
            "--object-index",
            str(index_output),
            "--output-csv",
            str(mask_index),
            "--output-mask-dir",
            args.mask_dir,
        ],
        env,
    )

    finalize_cmd = [
        python,
        "scripts/dataset/finalize_cogar_sim_index.py",
        "--input",
        str(mask_index),
        "--metadata",
        str(metadata_path),
        "--output",
        str(final_index),
        "--min-area",
        str(args.min_area),
    ]
    if args.exclude_categories:
        finalize_cmd.append("--exclude-categories")
        finalize_cmd.extend(args.exclude_categories)
    run_command(finalize_cmd, env)

    run_command(
        [python, "scripts/dataset/validate_sim_index.py", "--index", str(final_index)],
        env,
    )
    run_command(
        [
            python,
            "scripts/dataset/audit_sim_dataset.py",
            "--index",
            str(final_index),
            "--output-dir",
            str(audit_output_dir),
            "--max-objects-per-image",
            str(args.max_objects_per_image),
            "--min-mask-area",
            str(args.min_area),
        ],
        env,
    )

    if args.filter_bad:
        if args.filtered_index is None:
            raise ValueError("--filtered-index is required when --filter-bad is set")
        filtered_index = Path(args.filtered_index)
        run_command(
            [
                python,
                "scripts/dataset/filter_sim_index.py",
                "--index",
                str(final_index),
                "--audit",
                str(audit_output_dir / "image_quality_audit.csv"),
                "--output",
                str(filtered_index),
                "--exclude-bad",
                "--max-objects-per-image",
                str(args.max_objects_per_image),
                "--min-objects-per-image",
                "3",
            ],
            env,
        )
        run_command(
            [python, "scripts/dataset/validate_sim_index.py", "--index", str(filtered_index)],
            env,
        )

    print("\n[OK] Clean COGAR-Sim build complete.")
    print("Normalized COCO:", coco_path)
    print("Object index:", index_output)
    print("Mask index:", mask_index)
    print("Mask directory:", args.mask_dir)
    print("Final index:", final_index)
    print("Audit output directory:", audit_output_dir)
    if args.filtered_index:
        print("Filtered index:", args.filtered_index)


if __name__ == "__main__":
    main()
