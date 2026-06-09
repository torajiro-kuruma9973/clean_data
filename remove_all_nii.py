#!/usr/bin/env python3
"""
Delete all .nii.gz files recursively under a root directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Union


def delete_all_nii_gz(root_dir: Union[str, Path]) -> List[Path]:
    """Recursively delete every .nii.gz file under root_dir.

    Args:
        root_dir: Directory to traverse.

    Returns:
        Paths of deleted .nii.gz files.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    deleted: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower().endswith(".nii.gz"):
            path.unlink()
            deleted.append(path)

    return deleted


def list_nii_gz(root_dir: Union[str, Path]) -> List[Path]:
    """Recursively list every .nii.gz file under root_dir without deleting."""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith(".nii.gz")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete all .nii.gz files under a root directory.")
    parser.add_argument("root_dir", help="Root directory to traverse.")
    parser.add_argument("--dry-run", action="store_true", help="Only print files that would be deleted.")
    args = parser.parse_args()

    if args.dry_run:
        files = list_nii_gz(args.root_dir)
        for path in files:
            print(path)
        print(f"Would delete {len(files)} .nii.gz files.")
        return

    deleted = delete_all_nii_gz(args.root_dir)
    for path in deleted:
        print(path)
    print(f"Deleted {len(deleted)} .nii.gz files.")


if __name__ == "__main__":
    main()