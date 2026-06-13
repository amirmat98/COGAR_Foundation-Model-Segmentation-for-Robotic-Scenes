"""Sync generated benchmark artifacts from the AWS workspace to this repo.

This is intentionally limited to outputs, results, and log folders. Datasets stay
outside git and should be copied or downloaded separately when needed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HOST = "ubuntu@ec2-3-76-38-19.eu-central-1.compute.amazonaws.com"
DEFAULT_REMOTE_ROOT = "~/COGAR_Foundation-Model-Segmentation-for-Robotic-Scenes"
DEFAULT_PATHS = (
    "outputs/task4_zero_shot_sam",
    "results/task4_zero_shot_sam",
    "logs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("COGAR_AWS_HOST", DEFAULT_HOST),
        help="SSH host, for example ubuntu@example.compute.amazonaws.com.",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("COGAR_AWS_KEY", str(Path.home() / "Downloads" / "Ubu.pem")),
        help="SSH private key path.",
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("COGAR_AWS_REPO", DEFAULT_REMOTE_ROOT),
        help="Remote repository root.",
    )
    parser.add_argument(
        "--local-root",
        default=str(REPO_ROOT),
        help="Local repository root.",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=list(DEFAULT_PATHS),
        help="Repo-relative artifact paths to sync.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete local files that were removed remotely.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without copying files.",
    )
    return parser.parse_args()


def run(
    command: list[str],
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def remote_path_type(host: str, key: str, remote_root: str, rel_path: str) -> str:
    remote_path = f"{remote_root.rstrip('/')}/{rel_path}"
    command = [
        "ssh",
        "-i",
        key,
        host,
        (
            f"if [ -d {remote_path} ]; then echo dir; "
            f"elif [ -f {remote_path} ]; then echo file; "
            "else echo missing; fi"
        ),
    ]
    result = run(command, check=False, capture_output=True)
    if result.returncode != 0:
        return "missing"
    return result.stdout.strip()


def sync_path(args: argparse.Namespace, rel_path: str) -> None:
    rel_path = rel_path.strip("/")
    if not rel_path:
        raise ValueError("Refusing to sync an empty relative path")

    path_type = remote_path_type(args.host, args.key, args.remote_root, rel_path)
    if path_type == "missing":
        print(f"[SKIP] missing remote path: {rel_path}", flush=True)
        return

    local_target = Path(args.local_root) / rel_path
    if path_type == "dir":
        local_target.parent.mkdir(parents=True, exist_ok=True)
        remote_source = f"{args.host}:{args.remote_root.rstrip('/')}/{rel_path}/"
        local_destination = f"{local_target}/"
    elif path_type == "file":
        local_target.parent.mkdir(parents=True, exist_ok=True)
        remote_source = f"{args.host}:{args.remote_root.rstrip('/')}/{rel_path}"
        local_destination = str(local_target)
    else:
        raise RuntimeError(f"Unsupported remote path type for {rel_path}: {path_type}")

    command = [
        "rsync",
        "-az",
        "--partial",
        "--info=progress2",
    ]
    if args.delete:
        command.append("--delete")
    if args.dry_run:
        command.append("--dry-run")

    command.extend(
        [
            "-e",
            f"ssh -i {shlex.quote(args.key)}",
            remote_source,
            local_destination,
        ]
    )
    run(command)
    print(f"[OK] synced {rel_path}", flush=True)


def main() -> None:
    args = parse_args()
    for rel_path in args.paths:
        sync_path(args, rel_path)


if __name__ == "__main__":
    main()
