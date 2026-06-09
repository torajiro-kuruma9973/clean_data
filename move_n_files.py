#!/usr/bin/env python3
"""
Move NIfTI files named like n.nii.gz when n >= threshold.

Example:
  source/
    4.nii.gz
    5.nii.gz
    10.nii.gz

With threshold=5, moves 5.nii.gz and 10.nii.gz to target/.

Dependencies:
  Python standard library only.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class MoveIndexRecord:
    source_path: Path
    target_path: Path
    index: int
    status: str


def move_nii_ge_index(
    source_dir: Union[str, Path],
    threshold: int,
    target_dir: Union[str, Path],
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[MoveIndexRecord]:
    """Move files named n.nii.gz from source_dir to target_dir when n >= threshold.

    Args:
        source_dir: Folder containing files named like 5.nii.gz.
        threshold: Move files whose numeric filename prefix is >= threshold.
        target_dir: Destination folder.
        overwrite: If True, replace existing target files.
        dry_run: If True, only print planned moves.

    Returns:
        A list of move records.
    """
    source = Path(source_dir)
    target = Path(target_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got: {threshold}")

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    records: List[MoveIndexRecord] = []
    for path in sorted(source.iterdir(), key=lambda p: _sort_key(p.name)):
        if not path.is_file():
            continue
        file_index = _parse_index_name(path.name)
        if file_index is None or file_index < threshold:
            continue

        target_path = target / path.name
        if target_path.exists() and not overwrite:
            record = MoveIndexRecord(path, target_path, file_index, "skipped_target_exists")
            records.append(record)
            print(f"[skip] {path.name}: target exists: {target_path}")
            continue

        if dry_run:
            record = MoveIndexRecord(path, target_path, file_index, "dry_run")
            records.append(record)
            print(f"[dry-run] {path} -> {target_path}")
            continue

        if overwrite and target_path.exists():
            target_path.unlink()

        shutil.move(str(path), str(target_path))
        record = MoveIndexRecord(path, target_path, file_index, "moved")
        records.append(record)
        print(f"[moved] {path} -> {target_path}")

    moved = sum(1 for record in records if record.status == "moved")
    skipped = sum(1 for record in records if record.status.startswith("skipped"))
    dry = sum(1 for record in records if record.status == "dry_run")
    print(f"Matched files: {len(records)}; moved: {moved}; skipped: {skipped}; dry_run: {dry}")
    return records


def _parse_index_name(filename: str) -> Optional[int]:
    if not filename.endswith(".nii.gz"):
        return None
    prefix = filename[: -len(".nii.gz")]
    if not prefix.isdigit():
        return None
    return int(prefix)


def _sort_key(filename: str) -> tuple[int, Union[int, str]]:
    index = _parse_index_name(filename)
    if index is None:
        return (1, filename)
    return (0, index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move files named n.nii.gz whose n is >= a threshold."
    )
    parser.add_argument("source_dir", help="Source folder containing n.nii.gz files")
    parser.add_argument("threshold", type=int, help="Move files with n >= threshold")
    parser.add_argument("target_dir", help="Destination folder")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing target files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without moving files")
    args = parser.parse_args()

    move_nii_ge_index(
        args.source_dir,
        args.threshold,
        args.target_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()