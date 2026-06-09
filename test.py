#!/usr/bin/env python3
"""
Move ct.nii.gz and pet_resampled.nii.gz by a StudyID naming rule.

Expected PSMA-style source structure:
  root/
    PSMA_xxx/
      StudyID/
        xxx-CT-xxxxx/
          ct.nii.gz
        xxx-PET-xxxxx/
          pet_resampled.nii.gz

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
class MoveCtPetRecord:
    study_id: str
    new_name: Optional[str]
    status: str
    ct_source: Optional[Path]
    pet_source: Optional[Path]
    ct_target: Optional[Path]
    pet_target: Optional[Path]


def move_ct_pet_by_rule(
    root_dir: Union[str, Path],
    name_rule_json: Union[str, Path],
    ct_target_dir: Union[str, Path],
    pet_target_dir: Union[str, Path],
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[MoveCtPetRecord]:
    """Move same-study CT and resampled PET NIfTI files by JSON naming rule.

    Args:
        root_dir: PSMA root directory containing project folders and StudyID folders.
        name_rule_json: JSON mapping new filename -> StudyID.
        ct_target_dir: Destination directory for ct.nii.gz files.
        pet_target_dir: Destination directory for pet_resampled.nii.gz files.
        overwrite: If True, existing destination files may be replaced.
        dry_run: If True, print planned moves without moving files.

    Returns:
        A list of records for moved and skipped studies.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    study_to_name = _load_study_to_new_name(name_rule_json)
    ct_target_root = Path(ct_target_dir)
    pet_target_root = Path(pet_target_dir)

    if not dry_run:
        ct_target_root.mkdir(parents=True, exist_ok=True)
        pet_target_root.mkdir(parents=True, exist_ok=True)

    records: List[MoveCtPetRecord] = []
    study_dirs = list(_find_study_dirs(root))
    total = len(study_dirs)
    moved = 0
    skipped = 0
    failed = 0

    for index, study_dir in enumerate(study_dirs, start=1):
        study_id = study_dir.name
        new_name = study_to_name.get(study_id)

        if new_name is None:
            skipped += 1
            record = MoveCtPetRecord(study_id, None, "skipped_not_in_json", None, None, None, None)
            records.append(record)
            print(f"[{index}/{total}] skipped {study_id}: StudyID not in JSON rule")
            continue

        sources = _find_required_sources(study_dir)
        ct_target = ct_target_root / new_name
        pet_target = pet_target_root / new_name
        record = MoveCtPetRecord(
            study_id=study_id,
            new_name=new_name,
            status="pending",
            ct_source=sources["ct"],
            pet_source=sources["pet"],
            ct_target=ct_target,
            pet_target=pet_target,
        )

        missing = [key for key, path in sources.items() if path is None]
        if missing:
            skipped += 1
            record.status = "skipped_missing_" + "_".join(missing)
            records.append(record)
            print(f"[{index}/{total}] skipped {study_id}: missing {', '.join(missing)}")
            continue

        existing_targets = [str(path) for path in (ct_target, pet_target) if path.exists()]
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
                f"{sources['ct']} -> {ct_target}; "
                f"{sources['pet']} -> {pet_target}"
            )
            continue

        moved_pairs: List[Tuple[Path, Path]] = []
        try:
            if overwrite:
                for target in (ct_target, pet_target):
                    if target.exists():
                        target.unlink()

            _move_one(sources["ct"], ct_target)
            moved_pairs.append((ct_target, sources["ct"]))
            _move_one(sources["pet"], pet_target)
            moved_pairs.append((pet_target, sources["pet"]))

            moved += 1
            record.status = "moved"
            records.append(record)
            print(f"[{index}/{total}] moved {study_id} -> {new_name}")
        except Exception as exc:
            failed += 1
            record.status = f"failed: {exc}"
            records.append(record)
            _rollback_moves(moved_pairs)
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
    """Yield StudyID folders under root/project/study.

    StudyID names often contain "PETCT"; those must not be classified as PET
    folders. Only direct child folders with standalone CT/PET tokens or
    "Segmentation" are treated as series folders.
    """
    directories = [root]
    directories.extend(sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: str(path)))

    for directory in directories:
        series_dirs = _study_series_dirs(directory)
        if "ct" in series_dirs or "pet" in series_dirs:
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
        "ct": ("ct", "ct.nii.gz"),
        "pet": ("pet", "pet_resampled.nii.gz"),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move ct.nii.gz and pet_resampled.nii.gz files by JSON naming rule."
    )
    parser.add_argument("root_dir", help="PSMA root directory")
    parser.add_argument("name_rule_json", help="JSON mapping new filename -> StudyID")
    parser.add_argument("ct_target_dir", help="Destination directory for CT files")
    parser.add_argument("pet_target_dir", help="Destination directory for PET files")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination files")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without moving files")
    args = parser.parse_args()

    move_ct_pet_by_rule(
        args.root_dir,
        args.name_rule_json,
        args.ct_target_dir,
        args.pet_target_dir,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()