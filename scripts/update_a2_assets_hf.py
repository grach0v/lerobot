#!/usr/bin/env python3
"""Upload A2 assets to a Hugging Face dataset repo.

Defaults to syncing from the local A2_new assets checkout. The upload
excludes cache artifacts and the stray `...urdf` file unless overridden.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_SOURCE = "/home/denis-office/lerobot_a2/A2_new/assets"
DEFAULT_REPO = "dgrachev/a2_assets"


def _iter_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _filter_paths(paths: list[Path], root: Path, ignore_patterns: list[str]) -> list[Path]:
    filtered: list[Path] = []
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in ignore_patterns):
            continue
        filtered.append(path)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload A2 assets to Hugging Face dataset repo")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Path to assets directory")
    parser.add_argument("--repo-id", default=DEFAULT_REPO, help="Hugging Face dataset repo id")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument(
        "--commit-message",
        default="Update A2 assets",
        help="Commit message for the dataset update",
    )
    parser.add_argument(
        "--keep-ellipsis-urdf",
        action="store_true",
        help="Do not exclude unseen_objects/...urdf",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files that would be uploaded")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source assets path not found: {source}")

    ignore_patterns = [
        "**/.cache/**",
        "**/__pycache__/**",
    ]
    if not args.keep_ellipsis_urdf:
        ignore_patterns.append("**/...urdf")

    if args.dry_run:
        files = _iter_files(source)
        filtered = _filter_paths(files, source, ignore_patterns)
        print(f"Found {len(filtered)} files to upload from {source}")
        for path in filtered[:50]:
            print(path.relative_to(source))
        if len(filtered) > 50:
            print(f"... and {len(filtered) - 50} more")
        return

    if not args.token:
        raise ValueError("HF token required. Pass --token or set HF_TOKEN.")

    api = HfApi(token=args.token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(source),
        commit_message=args.commit_message,
        ignore_patterns=ignore_patterns,
    )

    print(f"Upload complete: {args.repo_id}")


if __name__ == "__main__":
    main()
