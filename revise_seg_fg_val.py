#!/usr/bin/env python3
"""
Clean PSMA segmentation masks in-place by removing foreground mask pixels whose
corresponding PET pixels are zero.

Expected directory pattern:
  root/
    PSMA_xxx/
      study-id/
        *PET*/
        *Segmentation*/
        *CT*/

The segmentation DICOM may store binary mask pixels as bit-packed data
(`BitsAllocated == 1`) or byte data (`BitsAllocated == 8`). This script reads
that from the DICOM header and writes PixelData back using the same storage
size.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pydicom


def clean_masks_under_root(
    root_dir: Union[str, Path],
    z_tolerance: float = 1e-3,
    dry_run: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Clean all segmentation masks under root_dir and print each changed study.

    Args:
        root_dir: Root directory containing project-id folders.
        z_tolerance: Max absolute z-coordinate difference for matching PET slice.
        dry_run: If True, report studies that would change but do not save.
        verbose: If True, print skipped/diagnostic messages to stderr.

    Returns:
        Study folder names whose segmentation masks were modified.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    changed_studies: List[str] = []

    for seg_dir in _find_segmentation_dirs(root):
        study_dir = seg_dir.parent
        study_id = study_dir.name
        pet_dir = _find_pet_dir(study_dir)
        if pet_dir is None:
            _log(verbose, f"[skip] PET folder not found for study: {study_dir}")
            continue

        pet_index = _build_pet_index(pet_dir, verbose=verbose)
        if not pet_index:
            _log(verbose, f"[skip] no PET slices with z coordinates: {pet_dir}")
            continue

        study_changed = False
        for mask_path in _iter_dicom_files(seg_dir):
            try:
                if _clean_one_mask(mask_path, pet_index, z_tolerance, dry_run, verbose):
                    study_changed = True
            except Exception as exc:  # Keep walking other studies.
                _log(verbose, f"[skip] failed mask {mask_path}: {exc}")

        if study_changed:
            print(study_id)
            changed_studies.append(study_id)

    return changed_studies


def _find_segmentation_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir() and "segmentation" in path.name.lower():
            yield path


def _find_pet_dir(study_dir: Path) -> Optional[Path]:
    candidates = []
    for child in study_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "segmentation" in name:
            continue
        if "pet" in name:
            candidates.append(child)
    return sorted(candidates)[0] if candidates else None


def _iter_dicom_files(directory: Path) -> list[Path]:
    dcm_files = sorted(p for p in directory.rglob("*.dcm") if p.is_file())
    if dcm_files:
        return dcm_files
    return sorted(p for p in directory.rglob("*") if p.is_file() and not p.name.startswith("."))


def _build_pet_index(pet_dir: Path, verbose: bool) -> List[Tuple[float, Path]]:
    slices: List[Tuple[float, Path]] = []
    for path in _iter_dicom_files(pet_dir):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            z = _dataset_z(ds)
        except Exception as exc:
            _log(verbose, f"[skip] PET header unreadable {path}: {exc}")
            continue
        if z is not None:
            slices.append((z, path))
    slices.sort(key=lambda item: item[0])
    return slices


def _clean_one_mask(
    mask_path: Path,
    pet_index: List[Tuple[float, Path]],
    z_tolerance: float,
    dry_run: bool,
    verbose: bool,
) -> bool:
    ds = pydicom.dcmread(mask_path)
    _ensure_uncompressed(ds, mask_path)

    original_pixel_data = bytes(ds.PixelData)
    mask = _mask_pixel_array_3d(ds)
    frame_count = mask.shape[0]

    if frame_count != _number_of_frames(ds):
        _log(verbose, f"[warn] frame count inferred from pixel_array for {mask_path}")

    changed = False
    pet_cache: Dict[Path, np.ndarray] = {}

    for frame_idx in range(frame_count):
        frame = mask[frame_idx]
        foreground = frame == 1
        if not np.any(foreground):
            continue

        z = _frame_z(ds, frame_idx)
        if z is None:
            _log(verbose, f"[skip] missing z for frame {frame_idx + 1}: {mask_path}")
            continue

        pet_path = _match_pet_slice(z, pet_index, z_tolerance)
        if pet_path is None:
            _log(verbose, f"[skip] no PET slice matched z={z} for frame {frame_idx + 1}: {mask_path}")
            continue

        pet_pixels = pet_cache.get(pet_path)
        if pet_pixels is None:
            pet_pixels = _read_pet_pixels(pet_path)
            pet_cache[pet_path] = pet_pixels

        if pet_pixels.shape != frame.shape:
            raise ValueError(
                f"shape mismatch for z={z}: mask frame {frame.shape}, PET {pet_pixels.shape}"
            )

        remove = foreground & (pet_pixels == 0)
        if np.any(remove):
            frame[remove] = 0
            changed = True

    if not changed:
        return False

    new_pixel_data = _encode_mask_pixel_data(ds, mask, original_len=len(original_pixel_data))
    if len(new_pixel_data) != len(original_pixel_data):
        raise ValueError(
            f"PixelData size changed for {mask_path}: "
            f"{len(original_pixel_data)} -> {len(new_pixel_data)}"
        )

    if not dry_run:
        ds.PixelData = new_pixel_data
        _save_dataset_in_place(ds, mask_path)

    return True


def _ensure_uncompressed(ds: pydicom.dataset.Dataset, path: Path) -> None:
    transfer_syntax = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if transfer_syntax is not None and getattr(transfer_syntax, "is_compressed", False):
        raise ValueError(f"compressed segmentation PixelData is not supported: {path}")


def _mask_pixel_array_3d(ds: pydicom.dataset.Dataset) -> np.ndarray:
    arr = ds.pixel_array
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError(f"expected 2D/3D mask pixel array, got shape {arr.shape}")
    return arr.copy()


def _read_pet_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    arr = np.asarray(ds.pixel_array)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D PET slice, got shape {arr.shape}: {path}")
    return arr


def _encode_mask_pixel_data(
    ds: pydicom.dataset.Dataset,
    mask: np.ndarray,
    original_len: int,
) -> bytes:
    bits_allocated = int(getattr(ds, "BitsAllocated", 0))
    bits_stored = int(getattr(ds, "BitsStored", bits_allocated))

    if bits_allocated == 1 or bits_stored == 1:
        # DICOM bit-packed binary pixel data stores the first pixel in the
        # least-significant bit of the first byte.
        flat = (mask.reshape(-1) > 0).astype(np.uint8)
        packed = np.packbits(flat, bitorder="little").tobytes()
        return _pad_to_original_len(packed, original_len)

    if bits_allocated == 8:
        data = mask.astype(np.uint8, copy=False).tobytes(order="C")
        return _pad_to_original_len(data, original_len)

    if bits_allocated == 16:
        dtype = np.dtype("<u2" if _is_little_endian(ds) else ">u2")
        data = mask.astype(dtype, copy=False).tobytes(order="C")
        return _pad_to_original_len(data, original_len)

    raise ValueError(f"unsupported mask BitsAllocated={bits_allocated}, BitsStored={bits_stored}")


def _pad_to_original_len(data: bytes, original_len: int) -> bytes:
    if len(data) > original_len:
        raise ValueError(f"encoded PixelData is larger than original: {len(data)} > {original_len}")
    if len(data) < original_len:
        data += b"\x00" * (original_len - len(data))
    return data


def _is_little_endian(ds: pydicom.dataset.Dataset) -> bool:
    value = getattr(ds, "is_little_endian", None)
    return True if value is None else bool(value)


def _number_of_frames(ds: pydicom.dataset.Dataset) -> int:
    return int(getattr(ds, "NumberOfFrames", 1) or 1)


def _dataset_z(ds: pydicom.dataset.Dataset) -> Optional[float]:
    image_position = getattr(ds, "ImagePositionPatient", None)
    if image_position is not None and len(image_position) >= 3:
        return float(image_position[2])
    slice_location = getattr(ds, "SliceLocation", None)
    if slice_location is not None:
        return float(slice_location)
    return None


def _frame_z(ds: pydicom.dataset.Dataset, frame_idx: int) -> Optional[float]:
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame is not None and frame_idx < len(per_frame):
        frame_group = per_frame[frame_idx]
        z = _z_from_plane_position(frame_group)
        if z is not None:
            return z

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared is not None and len(shared) > 0:
        z = _z_from_plane_position(shared[0])
        if z is not None:
            return z

    return _dataset_z(ds)


def _z_from_plane_position(group: pydicom.dataset.Dataset) -> Optional[float]:
    for attr in ("PlanePositionSequence", "PlanePositionVolumeSequence"):
        seq = getattr(group, attr, None)
        if seq is not None and len(seq) > 0:
            image_position = getattr(seq[0], "ImagePositionPatient", None)
            if image_position is not None and len(image_position) >= 3:
                return float(image_position[2])
    return None


def _match_pet_slice(
    target_z: float,
    pet_index: List[Tuple[float, Path]],
    z_tolerance: float,
) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_diff = math.inf
    for pet_z, pet_path in pet_index:
        diff = abs(pet_z - target_z)
        if diff < best_diff:
            best_diff = diff
            best_path = pet_path
    if best_path is not None and best_diff <= z_tolerance:
        return best_path
    return None


def _save_dataset_in_place(ds: pydicom.dataset.Dataset, path: Path) -> None:
    try:
        ds.save_as(path, enforce_file_format=False)
    except TypeError:
        ds.save_as(path, write_like_original=True)


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove mask foreground pixels where corresponding PET pixels are zero."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders.")
    parser.add_argument(
        "--z-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance for matching mask frame z to PET slice z. Default: 1e-3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find and print studies that would change, but do not save DICOM files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped files and diagnostic messages to stderr.",
    )
    args = parser.parse_args()

    clean_masks_under_root(
        args.root_dir,
        z_tolerance=args.z_tolerance,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()