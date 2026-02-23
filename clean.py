#!/usr/bin/env python
"""
Clean all generated results
===========================
Removes figures, datasets, nn_datasets, __pycache__ dirs,
.ipynb_checkpoints, and other build artifacts from the project.

Usage
-----
    python clean.py          # dry-run (shows what would be deleted)
    python clean.py --force  # actually delete everything
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Directories to remove entirely
DIRS_TO_REMOVE = [
    "outputs",
    "dist",
    "build",
]

# Glob patterns for directories removed recursively (anywhere in tree)
RECURSIVE_DIR_PATTERNS = [
    "__pycache__",
    ".ipynb_checkpoints",
    "*.egg-info",
]

# File glob patterns to remove from project root
FILE_PATTERNS = [
    "*.log",
    "*.egg",
]


def _collect_targets(root: Path) -> tuple[list[Path], list[Path]]:
    """Return (dirs_to_delete, files_to_delete)."""
    dirs: list[Path] = []
    files: list[Path] = []

    # Top-level output directories
    for name in DIRS_TO_REMOVE:
        d = root / name
        if d.is_dir():
            dirs.append(d)

    # Recursive directory patterns (e.g. __pycache__ everywhere)
    for pattern in RECURSIVE_DIR_PATTERNS:
        for d in root.rglob(pattern):
            if d.is_dir() and d not in dirs:
                dirs.append(d)

    # File patterns at root level
    for pattern in FILE_PATTERNS:
        for f in root.glob(pattern):
            if f.is_file():
                files.append(f)

    return dirs, files


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean all generated results.")
    parser.add_argument(
        "--force", action="store_true",
        help="Actually delete files. Without this flag, only a dry-run is performed.",
    )
    args = parser.parse_args()

    dirs, files = _collect_targets(ROOT)

    if not dirs and not files:
        print("Nothing to clean.")
        return

    total_bytes = 0

    print("Directories:")
    for d in sorted(dirs):
        sz = _dir_size(d)
        total_bytes += sz
        rel = d.relative_to(ROOT)
        print(f"  {'[DEL]' if args.force else '[DRY]'}  {rel}/  ({_human_size(sz)})")

    if files:
        print("Files:")
        for f in sorted(files):
            sz = f.stat().st_size
            total_bytes += sz
            rel = f.relative_to(ROOT)
            print(f"  {'[DEL]' if args.force else '[DRY]'}  {rel}  ({_human_size(sz)})")

    print(f"\nTotal: {_human_size(total_bytes)}")

    if not args.force:
        print("\nDry run — nothing was deleted. Re-run with --force to delete.")
        return

    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    for f in files:
        f.unlink(missing_ok=True)

    print("\nDone. All results cleaned.")


if __name__ == "__main__":
    main()
