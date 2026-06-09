#!/usr/bin/env python3
"""
Verify same-study CT/PET NIfTI z-axis alignment for NBIA/TCIA PSMA-style data.

Expected structure:
  root/
    PSMA_xxx/
      study-id/
        xxx-CT-xxxxx/
          ct.nii.gz
        xxx-PET-xxxxx/
          pet.nii.gz
        xxx-Segmentation-xxxxx/
          seg.nii.gz

What is checked:
  - CT and PET slice-axis z order have the same direction.
  - CT/PET minimum slice-center z coordinates match within tolerance.
  - CT/PET maximum slice-center z coordinates match within tolerance.

Dependencies:
  pip install nibabel numpy
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import nibabel as nib
import numpy as np


@dataclass
class ZAlignmentRecord:
    study_id: str
    ct_path: Path
    pet_path: Path
    ct_slices: int
    pet_slices: int
    ct_z_first: float
    ct_z_last: float
    pet_z_first: float
    pet_z_last: float
    ct_z_min: float
    ct_z_max: float
    pet_z_min: float
    pet_z_max: float
    ct_z_spacing: float
    pet_z_spacing: float
    direction_ok: bool
    min_z_diff_mm: float
    max_z_diff_mm: float
    range_ok: bool
    status: str


def verify_ct_pet_z_alignment(
    root_dir: Union[str, Path],
    tolerance_mm: float = 2.0,
    output_csv: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> List[ZAlignmentRecord]:
    """Verify CT/PET NIfTI z-axis order and z-range for every study.

    Args:
        root_dir: Root directory containing project-id and study-id folders.
        tolerance_mm: Allowed absolute difference for z min and z max.
        output_csv: Optional CSV path for per-study check results.
        verbose: Print skipped study information.

    Returns:
        A list of per-study verification records. Studies missing CT or PET
        NIfTI files are skipped unless verbose=True, in which case they are
        printed to stderr-like console output.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be non-negative")

    records: List[ZAlignmentRecord] = []
    study_dirs = list(_find_study_dirs_with_nifti(root))

    for study_dir in study_dirs:
        study_id = study_dir.name
        ct_path = _find_series_nifti(study_dir, "ct")
        pet_path = _find_series_nifti(study_dir, "pet")

        if ct_path is None or pet_path is None:
            if verbose:
                print(f"[skip] {study_id}: missing ct.nii.gz or pet.nii.gz")
            continue

        record = _check_one_study(study_id, ct_path, pet_path, tolerance_mm)
        records.append(record)

        if record.status != "ok":
            print(
                f"[FAIL] {study_id}: {record.status}; "
                f"CT first/last {record.ct_z_first:.3f}/{record.ct_z_last:.3f}, "
                f"PET first/last {record.pet_z_first:.3f}/{record.pet_z_last:.3f}; "
                f"CT min/max {record.ct_z_min:.3f}/{record.ct_z_max:.3f}, "
                f"PET min/max {record.pet_z_min:.3f}/{record.pet_z_max:.3f}; "
                f"diff min/max {record.min_z_diff_mm:.3f}/{record.max_z_diff_mm:.3f} mm"
            )
            

    if output_csv is not None:
        _write_csv(records, Path(output_csv))

    total = len(records)
    failed = sum(1 for record in records if record.status != "ok")
    print(f"Checked studies: {total}; passed: {total - failed}; failed: {failed}")

    return records


def _find_study_dirs_with_nifti(root: Path) -> Iterable[Path]:
    """Yield directories that have CT/PET NIfTI files in direct child folders."""
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
        has_ct = _find_series_nifti(directory, "ct") is not None
        has_pet = _find_series_nifti(directory, "pet") is not None
        if has_ct or has_pet:
            yield directory


def _find_series_nifti(study_dir: Path, series_type: str) -> Optional[Path]:
    """Find ct.nii.gz or pet.nii.gz inside a direct child series folder."""
    expected_name = f"{series_type}.nii.gz"

    try:
        children = list(study_dir.iterdir())
    except OSError:
        return None

    candidates: List[Path] = []
    for child in children:
        if not child.is_dir():
            continue
        if _classify_series_dir(child) != series_type:
            continue
        nii_path = child / expected_name
        if nii_path.is_file():
            candidates.append(nii_path)

    if not candidates:
        return None
    return sorted(candidates, key=lambda p: str(p))[0]


def _classify_series_dir(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "segmentation" in name:
        return "seg"
    if "pet" in name:
        return "pet"
    if "ct" in name:
        return "ct"
    return None


def _check_one_study(
    study_id: str,
    ct_path: Path,
    pet_path: Path,
    tolerance_mm: float,
) -> ZAlignmentRecord:
    ct_img = nib.load(str(ct_path))
    pet_img = nib.load(str(pet_path))

    ct_z = _slice_axis_z_values(ct_img, ct_path)
    pet_z = _slice_axis_z_values(pet_img, pet_path)

    ct_direction = _direction_sign(ct_z)
    pet_direction = _direction_sign(pet_z)
    direction_ok = ct_direction == pet_direction

    ct_z_min = float(np.min(ct_z))
    ct_z_max = float(np.max(ct_z))
    pet_z_min = float(np.min(pet_z))
    pet_z_max = float(np.max(pet_z))
    min_z_diff = abs(ct_z_min - pet_z_min)
    max_z_diff = abs(ct_z_max - pet_z_max)
    range_ok = min_z_diff <= tolerance_mm and max_z_diff <= tolerance_mm

    status_parts: List[str] = []
    if not direction_ok:
        status_parts.append("z_direction_mismatch")
    if min_z_diff > tolerance_mm:
        status_parts.append(f"min_z_diff>{tolerance_mm:g}mm")
    if max_z_diff > tolerance_mm:
        status_parts.append(f"max_z_diff>{tolerance_mm:g}mm")
    status = "ok" if not status_parts else ";".join(status_parts)

    return ZAlignmentRecord(
        study_id=study_id,
        ct_path=ct_path,
        pet_path=pet_path,
        ct_slices=len(ct_z),
        pet_slices=len(pet_z),
        ct_z_first=float(ct_z[0]),
        ct_z_last=float(ct_z[-1]),
        pet_z_first=float(pet_z[0]),
        pet_z_last=float(pet_z[-1]),
        ct_z_min=ct_z_min,
        ct_z_max=ct_z_max,
        pet_z_min=pet_z_min,
        pet_z_max=pet_z_max,
        ct_z_spacing=_median_abs_spacing(ct_z),
        pet_z_spacing=_median_abs_spacing(pet_z),
        direction_ok=direction_ok,
        min_z_diff_mm=float(min_z_diff),
        max_z_diff_mm=float(max_z_diff),
        range_ok=range_ok,
        status=status,
    )


def _slice_axis_z_values(img: nib.spatialimages.SpatialImage, path: Path) -> np.ndarray:
    """Return physical z coordinates for slice centers along NIfTI axis 2."""
    shape = img.shape
    if len(shape) < 3 or shape[2] < 2:
        raise ValueError(f"NIfTI must be at least 3D with >=2 z slices: {path}")

    affine = np.asarray(img.affine, dtype=np.float64)
    k = np.arange(shape[2], dtype=np.float64)
    return affine[2, 3] + k * affine[2, 2]


def _direction_sign(z_values: np.ndarray) -> int:
    delta = float(z_values[-1] - z_values[0])
    if np.isclose(delta, 0.0):
        raise ValueError("z-axis has near-zero first-to-last physical z difference")
    return 1 if delta > 0 else -1


def _median_abs_spacing(z_values: np.ndarray) -> float:
    if len(z_values) < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(z_values))))


def _write_csv(records: Sequence[ZAlignmentRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id",
        "status",
        "direction_ok",
        "range_ok",
        "min_z_diff_mm",
        "max_z_diff_mm",
        "ct_slices",
        "pet_slices",
        "ct_z_spacing",
        "pet_z_spacing",
        "ct_z_first",
        "ct_z_last",
        "pet_z_first",
        "pet_z_last",
        "ct_z_min",
        "ct_z_max",
        "pet_z_min",
        "pet_z_max",
        "ct_path",
        "pet_path",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "study_id": record.study_id,
                    "status": record.status,
                    "direction_ok": record.direction_ok,
                    "range_ok": record.range_ok,
                    "min_z_diff_mm": f"{record.min_z_diff_mm:.6f}",
                    "max_z_diff_mm": f"{record.max_z_diff_mm:.6f}",
                    "ct_slices": record.ct_slices,
                    "pet_slices": record.pet_slices,
                    "ct_z_spacing": f"{record.ct_z_spacing:.6f}",
                    "pet_z_spacing": f"{record.pet_z_spacing:.6f}",
                    "ct_z_first": f"{record.ct_z_first:.6f}",
                    "ct_z_last": f"{record.ct_z_last:.6f}",
                    "pet_z_first": f"{record.pet_z_first:.6f}",
                    "pet_z_last": f"{record.pet_z_last:.6f}",
                    "ct_z_min": f"{record.ct_z_min:.6f}",
                    "ct_z_max": f"{record.ct_z_max:.6f}",
                    "pet_z_min": f"{record.pet_z_min:.6f}",
                    "pet_z_max": f"{record.pet_z_max:.6f}",
                    "ct_path": str(record.ct_path),
                    "pet_path": str(record.pet_path),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify same-study ct.nii.gz and pet.nii.gz z-axis alignment."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders")
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=2.0,
        help="Allowed z min/max difference in mm (default: 1.5)",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for detailed per-study results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped study information",
    )
    args = parser.parse_args()

    verify_ct_pet_z_alignment(
        args.root_dir,
        tolerance_mm=args.tolerance_mm,
        output_csv=args.output_csv,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
