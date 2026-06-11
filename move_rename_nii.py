#!/usr/bin/env python3
"""
Move preprocessed CT/PET/SEG NIfTI files into data/labels/segs folders.

Expected PSMA-style source structure:
  root/
    PSMA_xxx/
      StudyID/
        xxx-CT-xxxxx/
          ct_norm.nii.gz
        xxx-PET-xxxxx/
          pet_norm.nii.gz
        xxx-Segmentation-xxxxx/
          seg_resampled.nii.gz

Name-rule JSON format:
  {
    "0.nii.gz": "01-01-2002-NA-PETCT whole-body PSMA-21061",
    "1.nii.gz": "01-01-2003-NA-PETCT whole-body PSMA-65095"
  }

The JSON key is the new filename. The JSON value is the StudyID.

Dependencies:
  Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class MoveRecord:
    study_id: str
    new_name: Optional[str]
    status: str
    ct_source: Optional[Path]
    pet_source: Optional[Path]
    seg_source: Optional[Path]
    ct_target: Optional[Path]
    pet_target: Optional[Path]
    seg_target: Optional[Path]


def move_preprocessed_nii_by_rule(
    root_dir: Union[str, Path],
    name_rule_json: Union[str, Path],
    data_dir: Union[str, Path] = "data",
    labels_dir: Union[str, Path] = "labels",
    segs_dir: Union[str, Path] = "segs",
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> List[MoveRecord]:
    """Move ct_norm, pet_norm, and seg_resampled files by a JSON name rule.

    Args:
        root_dir: PSMA root directory containing project folders and StudyID folders.
        name_rule_json: JSON mapping new filename -> StudyID.
        data_dir: Destination directory for CT files. Defaults to relative "data".
        labels_dir: Destination directory for PET files. Defaults to relative "labels".
        segs_dir: Destination directory for SEG files. Defaults to relative "segs".
        overwrite: If True, existing destination files may be replaced.
        dry_run: If True, print what would be moved without moving files.
        verbose: Print extra skip information.

    Returns:
        A list of move records, including skipped studies.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    study_to_name = _load_study_to_new_name(name_rule_json)
    data_path = Path(data_dir)
    labels_path = Path(labels_dir)
    segs_path = Path(segs_dir)

    records: List[MoveRecord] = []
    study_dirs = list(_find_study_dirs(root))
    total = len(study_dirs)

    if not dry_run:
        data_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        segs_path.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped = 0
    failed = 0

    for index, study_dir in enumerate(study_dirs, start=1):
        study_id = study_dir.name
        new_name = study_to_name.get(study_id)

        if new_name is None:
            skipped += 1
            record = MoveRecord(study_id, None, "skipped_not_in_json", None, None, None, None, None, None)
            records.append(record)
            print(f"[{index}/{total}] skipped {study_id}: StudyID not in JSON rule")
            continue

        sources = _find_required_sources(study_dir)
        missing = [name for name, path in sources.items() if path is None]
        targets = {
            "ct": data_path / new_name,
            "pet": labels_path / new_name,
            "seg": segs_path / new_name,
        }

        record = MoveRecord(
            study_id=study_id,
            new_name=new_name,
            status="pending",
            ct_source=sources["ct"],
            pet_source=sources["pet"],
            seg_source=sources["seg"],
            ct_target=targets["ct"],
            pet_target=targets["pet"],
            seg_target=targets["seg"],
        )

        if missing:
            skipped += 1
            record.status = "skipped_missing_" + "_".join(missing)
            records.append(record)
            print(f"[{index}/{total}] skipped {study_id}: missing {', '.join(missing)}")
            continue

        existing_targets = [str(path) for path in targets.values() if path.exists()]
        if existing_targets and not overwrite:
            skipped += 1
            record.status = "skipped_target_exists"
            records.append(record)
            print(
                f"[{index}/{total}] skipped {study_id}: target exists; "
                f"use --overwrite to replace: {existing_targets}"
            )
            continue

        if dry_run:
            skipped += 1
            record.status = "dry_run"
            records.append(record)
            print(
                f"[{index}/{total}] dry-run {study_id}: "
                f"{sources['ct']} -> {targets['ct']}; "
                f"{sources['pet']} -> {targets['pet']}; "
                f"{sources['seg']} -> {targets['seg']}"
            )
            continue

        moved_sources: List[Tuple[Path, Path]] = []
        try:
            if overwrite:
                for target in targets.values():
                    if target.exists():
                        target.unlink()

            _move_one(sources["ct"], targets["ct"])
            moved_sources.append((targets["ct"], sources["ct"]))
            _move_one(sources["pet"], targets["pet"])
            moved_sources.append((targets["pet"], sources["pet"]))
            _move_one(sources["seg"], targets["seg"])
            moved_sources.append((targets["seg"], sources["seg"]))

            moved += 1
            record.status = "moved"
            records.append(record)
            print(f"[{index}/{total}] moved {study_id} -> {new_name}")
        except Exception as exc:
            failed += 1
            record.status = f"failed: {exc}"
            records.append(record)
            _rollback_moves(moved_sources)
            print(f"[{index}/{total}] failed {study_id}: {exc}")

    print(f"Studies checked: {total}; moved: {moved}; skipped: {skipped}; failed: {failed}")
    return records


def _load_study_to_new_name(name_rule_json: Union[str, Path]) -> Dict[str, str]:
    path = Path(name_rule_json)
    with path.open("r", encoding="utf-8") as f:
        name_to_study = json.load(f)

    if not isinstance(name_to_study, dict):
        raise ValueError("name_rule_json must be an object mapping new filename -> StudyID")

    study_to_name: Dict[str, str] = {}
    for new_name, study_id in name_to_study.items():
        if not isinstance(new_name, str) or not isinstance(study_id, str):
            raise ValueError("all JSON keys and values must be strings")
        if study_id in study_to_name:
            raise ValueError(
                f"duplicate StudyID in JSON rule: {study_id!r} maps to both "
                f"{study_to_name[study_id]!r} and {new_name!r}"
            )
        study_to_name[study_id] = new_name
    return study_to_name


def _find_study_dirs(root: Path) -> Iterable[Path]:
    """Yield actual StudyID folders, not project folders.

    StudyID names often contain "PETCT", so a project folder's direct children
    must not be classified as PET/CT series just because their names contain
    that substring. A StudyID folder is identified by direct child series
    folders whose names contain standalone CT/PET tokens or "Segmentation".
    """
    directories = [root]
    directories.extend(sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: str(path)))

    for directory in directories:
        series_dirs = _study_series_dirs(directory)
        if series_dirs:
            yield directory


def _study_series_dirs(study_dir: Path) -> Dict[str, Path]:
    series_dirs: Dict[str, Path] = {}
    try:
        children = list(study_dir.iterdir())
    except OSError:
        return series_dirs

    for child in children:
        if not child.is_dir():
            continue
        series_type = _classify_series_dir(child)
        if series_type is not None and series_type not in series_dirs:
            series_dirs[series_type] = child
    return series_dirs


def _find_required_sources(study_dir: Path) -> Dict[str, Optional[Path]]:
    series_dirs = _study_series_dirs(study_dir)
    expected = {
        "ct": ("ct", "ct_norm.nii.gz"),
        "pet": ("pet", "pet_norm.nii.gz"),
        "seg": ("seg", "seg_resampled.nii.gz"),
    }

    sources: Dict[str, Optional[Path]] = {}
    for key, (series_type, filename) in expected.items():
        series_dir = series_dirs.get(series_type)
        if series_dir is None:
            sources[key] = None
            continue
        path = series_dir / filename
        sources[key] = path if path.is_file() else None
    return sources


def _classify_series_dir(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "segmentation" in name:
        return "seg"
    if _contains_standalone_token(name, "pet"):
        return "pet"
    if _contains_standalone_token(name, "ct"):
        return "ct"
    return None


def _contains_standalone_token(name: str, token: str) -> bool:
    """Match PET/CT as a series token, but not inside StudyID text like PETCT."""
    return re.search(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", name) is not None


def _move_one(source: Optional[Path], target: Path) -> None:
    if source is None:
        raise ValueError(f"missing source for target {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _rollback_moves(moved_pairs: List[Tuple[Path, Path]]) -> None:
    for current_path, original_path in reversed(moved_pairs):
        try:
            if current_path.exists() and not original_path.exists():
                original_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_path), str(original_path))
        except Exception:
            pass


def _warn(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move ct_norm, pet_norm, and seg_resampled NIfTI files by JSON naming rule."
    )
    parser.add_argument("root_dir", help="PSMA root directory")
    parser.add_argument("name_rule_json", help="JSON mapping new filename -> StudyID")
    parser.add_argument("--data-dir", default="..final_data/data", help="Destination directory for CT files")
    parser.add_argument("--labels-dir", default="..final_data/labels", help="Destination directory for PET files")
    parser.add_argument("--segs-dir", default="..final_data/segs", help="Destination directory for SEG files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without moving files")
    parser.add_argument("--verbose", action="store_true", help="Print extra skip information")
    args = parser.parse_args()

    move_preprocessed_nii_by_rule(
        args.root_dir,
        args.name_rule_json,
        data_dir=args.data_dir,
        labels_dir=args.labels_dir,
        segs_dir=args.segs_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()