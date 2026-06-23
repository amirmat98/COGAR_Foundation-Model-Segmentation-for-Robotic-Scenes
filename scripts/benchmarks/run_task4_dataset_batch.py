"""Run Task 4 zero-shot inference jobs for one dataset.

The script runs model/prompt combinations sequentially, skips already complete
outputs by default, and validates each JSONL after the runner exits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
RUNNER = REPO_ROOT / "scripts" / "benchmarks" / "run_zero_shot_sam.py"
PROMPT_MODES = ("point", "box", "automatic")

from cogar_seg.config import load_config as load_project_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/task4_zero_shot_sam.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--prompt-modes", nargs="*", choices=PROMPT_MODES, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun jobs even if the expected output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running inference.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return load_project_config(path)


def selected_models(config: dict[str, Any], selected: list[str] | None) -> list[str]:
    models = []
    for name, model_config in config["models"].items():
        if selected is not None and name not in selected:
            continue
        if model_config.get("enabled", False):
            models.append(name)
    return models


def prompt_manifest_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["task"]["prompt_manifest_dir"]) / f"{dataset}_instances.jsonl"


def output_path(
    config: dict[str, Any],
    dataset: str,
    model: str,
    prompt_mode: str,
) -> Path:
    return (
        Path(config["task"]["output_root"])
        / dataset
        / model
        / f"{prompt_mode}_predictions.jsonl"
    )


def prompt_stats(path: Path) -> dict[str, int]:
    records = 0
    image_ids: OrderedDict[Any, None] = OrderedDict()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            image_ids.setdefault(record["image_id"], None)
    return {"records": records, "images": len(image_ids)}


def expected_count(stats: dict[str, int], prompt_mode: str) -> int:
    if prompt_mode == "automatic":
        return stats["images"]
    return stats["records"]


def valid_jsonl_rows(path: Path) -> tuple[int, int]:
    valid = 0
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception:  # noqa: BLE001
                bad += 1
                continue
            valid += 1
    return valid, bad


def is_complete(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    valid, bad = valid_jsonl_rows(path)
    return valid == expected and bad == 0


def command_for_job(
    args: argparse.Namespace,
    dataset: str,
    model: str,
    prompt_mode: str,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        args.config,
        "--dataset",
        dataset,
        "--model",
        model,
        "--prompt-mode",
        prompt_mode,
        "--log-every",
        str(args.log_every),
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = args.dataset
    if dataset not in config["datasets"]:
        raise KeyError(f"Unknown dataset: {dataset}")

    manifest = prompt_manifest_path(config, dataset)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing prompt manifest: {manifest}. "
            "Run scripts/benchmarks/build_prompt_manifest.py first."
        )

    models = selected_models(config, args.models)
    prompt_modes = args.prompt_modes or list(PROMPT_MODES)
    stats = prompt_stats(manifest)

    print(
        f"[BATCH START] dataset={dataset} records={stats['records']} "
        f"images={stats['images']} models={models} prompt_modes={prompt_modes}",
        flush=True,
    )

    for model in models:
        for prompt_mode in prompt_modes:
            expected = expected_count(stats, prompt_mode)
            out_path = output_path(config, dataset, model, prompt_mode)
            label = f"{dataset}/{model}/{prompt_mode}"

            if not args.rerun_complete and is_complete(out_path, expected):
                print(
                    f"[SKIP] {label}: complete with {expected} rows at {out_path}",
                    flush=True,
                )
                continue

            command = command_for_job(args, dataset, model, prompt_mode)
            print(f"[RUN] {label}: expected_rows={expected}", flush=True)
            print("+ " + " ".join(command), flush=True)
            if args.dry_run:
                continue

            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if result.returncode != 0:
                print(f"[FAILED] {label}: returncode={result.returncode}", flush=True)
                raise SystemExit(result.returncode)

            valid, bad = valid_jsonl_rows(out_path)
            if valid != expected or bad:
                print(
                    f"[BAD] {label}: expected={expected} valid={valid} bad={bad} "
                    f"path={out_path}",
                    flush=True,
                )
                raise SystemExit(1)
            print(f"[OK] {label}: {valid} rows", flush=True)

    print(f"[BATCH OK] dataset={dataset}", flush=True)


if __name__ == "__main__":
    main()
