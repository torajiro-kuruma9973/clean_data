#!/usr/bin/env python3
"""
Convert NBIA/TCIA PSMA-style DICOM folders to per-series NIfTI files.

Expected structure:
  root/
    PSMA_xxx/
      study-id/
        *PET*/
        *Segmentation*/
        *CT*/

Output naming:
  CT folder           -> ct.nii.gz
  PET folder          -> pet.nii.gz
  Segmentation folder -> seg.nii.gz

Important behavior:
  - DICOM slices are ordered by physical coordinates, not filenames.
  - Segmentation folders are detected before CT, so names containing both
    "Segmentation" and "ct" are still treated as SEG.
  - PET/SEG z-axis direction is flipped when needed so it matches same-study CT.
  - No intensity clipping is applied during conversion.
  - CT is saved as int16 HU when the rescaled HU values are integer-like and
    fit int16; otherwise float32 is used to avoid clipping/truncation.
  - PET is saved as float32 SUVbw. BQML PET values are converted using
    patient weight, injected dose, half-life, and acquisition/injection times.
    PET values that are already encoded as SUV are kept as SUV.
  - SEG is saved as uint8 binary mask.

Dependencies:
  pip install pydicom numpy nibabel
"""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pydicom


@dataclass
class ImagePlane:
    ipp: np.ndarray
    row_dir: np.ndarray
    col_dir: np.ndarray
    normal: np.ndarray
    row_spacing: float
    col_spacing: float


@dataclass
class NiftiVolume:
    data: np.ndarray
    affine_lps: np.ndarray


@dataclass
class ConvertRecord:
    study_id: str
    series_type: str
    series_dir: Path
    output_path: Path
    status: str


def _format_file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.2f} {unit}"


def convert_psma_dicom_to_nifti(
    root_dir: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = False,
) -> List[ConvertRecord]:
    """Convert each CT/PET/Segmentation series under root_dir to .nii.gz.

    Args:
        root_dir: Root directory containing project-id and study-id folders.
        overwrite: If True, rewrite existing nii.gz outputs.
        verbose: Print diagnostic information to stderr.

    Returns:
        A list of conversion records.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    records: List[ConvertRecord] = []
    study_dirs = list(_find_study_dirs(root))
    total_studies = len(study_dirs)

    for study_index, study_dir in enumerate(study_dirs, start=1):
        study_id = study_dir.name
        series_dirs = _study_series_dirs(study_dir)
        ct_dir = series_dirs.get("ct")
        study_written_or_skipped = 0

        ct_reference: Optional[NiftiVolume] = None
        if ct_dir is not None:
            output = ct_dir / "ct.nii.gz"
            if output.exists() and not overwrite:
                records.append(ConvertRecord(study_id, "ct", ct_dir, output, "skipped_exists"))
                study_written_or_skipped += 1
                print(f"[study {study_index}/{total_studies}] {study_id} CT skipped: {output} ({_format_file_size(output)})")
                try:
                    ct_reference = _build_slice_series_reference(ct_dir, expected_modality="CT")
                except Exception as exc:
                    _log(verbose, f"[warn] CT DICOM cannot be used as z-axis reference {ct_dir}: {exc}")
            else:
                try:
                    ct_reference = _convert_slice_series(
                        ct_dir,
                        output,
                        series_type="ct",
                        reference_z_axis_lps=None,
                    )
                    records.append(ConvertRecord(study_id, "ct", ct_dir, output, "written"))
                    study_written_or_skipped += 1
                    print(f"[study {study_index}/{total_studies}] {study_id} CT written: {output} ({_format_file_size(output)})")
                    _log(verbose, f"[ok] CT -> {output}")
                except Exception as exc:
                    records.append(ConvertRecord(study_id, "ct", ct_dir, output, f"failed: {exc}"))
                    _log(verbose, f"[fail] CT {ct_dir}: {exc}")

        for series_type, filename in (("pet", "pet.nii.gz"), ("seg", "seg.nii.gz")):
            series_dir = series_dirs.get(series_type)
            if series_dir is None:
                continue

            output = series_dir / filename
            if output.exists() and not overwrite:
                records.append(ConvertRecord(study_id, series_type, series_dir, output, "skipped_exists"))
                study_written_or_skipped += 1
                print(
                    f"[study {study_index}/{total_studies}] {study_id} "
                    f"{series_type.upper()} skipped: {output} ({_format_file_size(output)})"
                )
                continue

            try:
                reference_z_axis = ct_reference.affine_lps[:3, 2] if ct_reference is not None else None
                if series_type == "pet":
                    _convert_slice_series(
                        series_dir,
                        output,
                        series_type="pet",
                        reference_z_axis_lps=reference_z_axis,
                    )
                else:
                    _convert_seg_series(
                        series_dir,
                        output,
                        reference_z_axis_lps=reference_z_axis,
                    )
                records.append(ConvertRecord(study_id, series_type, series_dir, output, "written"))
                study_written_or_skipped += 1
                print(
                    f"[study {study_index}/{total_studies}] {study_id} "
                    f"{series_type.upper()} written: {output} ({_format_file_size(output)})"
                )
                _log(verbose, f"[ok] {series_type.upper()} -> {output}")
            except Exception as exc:
                records.append(ConvertRecord(study_id, series_type, series_dir, output, f"failed: {exc}"))
                _log(verbose, f"[fail] {series_type.upper()} {series_dir}: {exc}")

        print(
            f"Completed study {study_index}/{total_studies}: {study_id} "
            f"({study_written_or_skipped}/3 NIfTI outputs written or already existed)"
        )

    return records


def _find_study_dirs(root: Path) -> Iterable[Path]:
    for directory in root.rglob("*"):
        if not directory.is_dir():
            continue
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
        if not _has_direct_dicom_like_files(child):
            continue
        series_type = _classify_series_dir(child)
        if series_type is not None and series_type not in series_dirs:
            series_dirs[series_type] = child
    return series_dirs


def _has_direct_dicom_like_files(directory: Path) -> bool:
    """Return True only for actual series folders with DICOM files directly inside.

    Study IDs in this dataset often contain the string "PETCT"; without this
    guard, a project folder can incorrectly classify its study child as a PET
    series and write pet.nii.gz into the study root.
    """
    try:
        children = list(directory.iterdir())
    except OSError:
        return False

    for child in children:
        if not child.is_file() or child.name.startswith("."):
            continue
        suffixes = "".join(child.suffixes).lower()
        if suffixes.endswith(".nii.gz") or child.suffix.lower() == ".nii":
            continue
        if child.suffix.lower() == ".dcm" or child.suffix == "":
            return True
    return False


def _classify_series_dir(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "segmentation" in name:
        return "seg"
    if "pet" in name:
        return "pet"
    if "ct" in name:
        return "ct"
    return None


def _iter_dicom_files(directory: Path) -> List[Path]:
    dcm_files = sorted(p for p in directory.rglob("*.dcm") if p.is_file())
    if dcm_files:
        return dcm_files
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and ".nii" not in "".join(p.suffixes).lower()
    )


def _convert_slice_series(
    series_dir: Path,
    output_path: Path,
    series_type: str,
    reference_z_axis_lps: Optional[np.ndarray],
) -> NiftiVolume:
    slices = _read_slice_headers(series_dir, expected_modality="CT" if series_type == "ct" else "PT")
    if not slices:
        raise RuntimeError(f"no readable {series_type.upper()} DICOM slices found")

    data_slices: List[np.ndarray] = []
    for item in slices:
        ds = pydicom.dcmread(item["path"])
        pixels = np.asarray(ds.pixel_array)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)

        if series_type == "ct":
            hu = pixels.astype(np.float32) * slope + intercept
            if _can_store_as_int16_without_clipping(hu):
                array = np.rint(hu).astype(np.int16)
            else:
                array = hu.astype(np.float32)
        else:
            array = _pet_pixels_to_suvbw(ds, pixels).astype(np.float32)

        if array.ndim != 2:
            raise ValueError(f"expected 2D slice, got shape {array.shape}: {item['path']}")
        data_slices.append(array)

    volume_data = np.stack(data_slices, axis=2)
    first_plane = slices[0]["plane"]
    slice_spacing = _slice_spacing_from_positions([float(item["position"]) for item in slices])
    affine_lps = _affine_lps_from_plane(first_plane, slice_spacing)
    volume = NiftiVolume(data=volume_data, affine_lps=affine_lps)
    volume = _align_z_axis_to_reference(volume, reference_z_axis_lps)

    _save_nifti(volume, output_path)
    return volume


def _pet_pixels_to_suvbw(ds: pydicom.dataset.Dataset, pixels: np.ndarray) -> np.ndarray:
    """Convert a PET slice to SUVbw when DICOM units are BQML.

    If the PET DICOM already stores SUV-like values, keep the rescaled values.
    Unsupported units fail loudly because converting counts to SUV requires
    vendor-specific information that is not reliably encoded by these tags.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    rescaled = pixels.astype(np.float32) * slope + intercept

    units = _upper_text(getattr(ds, "Units", ""))
    rescale_type = _upper_text(getattr(ds, "RescaleType", ""))
    suv_type = _upper_text(getattr(ds, "SUVType", ""))

    if _pet_values_are_already_suv(units, rescale_type, suv_type):
        return rescaled.astype(np.float32, copy=False)

    if units != "BQML":
        raise ValueError(
            "PET SUV conversion requires Units=BQML or an already-SUV PET series; "
            f"got Units={units or '<missing>'}, RescaleType={rescale_type or '<missing>'}, "
            f"SUVType={suv_type or '<missing>'}"
        )

    factor = _suvbw_factor_from_dicom(ds)
    return (rescaled * factor).astype(np.float32, copy=False)


def _pet_values_are_already_suv(units: str, rescale_type: str, suv_type: str) -> bool:
    if units == "GML":
        if suv_type and suv_type not in {"BW", "SUVBW"}:
            raise ValueError(f"PET series is SUV-like but not SUVbw: Units={units}, SUVType={suv_type}")
        return True
    if "SUV" in units or "SUV" in rescale_type:
        if suv_type and suv_type not in {"BW", "SUVBW"}:
            raise ValueError(f"PET series is SUV-like but not SUVbw: Units={units}, SUVType={suv_type}")
        return True
    return False


def _suvbw_factor_from_dicom(ds: pydicom.dataset.Dataset) -> float:
    patient_weight_kg = _required_float(ds, "PatientWeight")
    if patient_weight_kg <= 0:
        raise ValueError(f"invalid PatientWeight for SUVbw: {patient_weight_kg}")

    radio = _radiopharmaceutical_item(ds)
    total_dose_bq = _required_float(radio, "RadionuclideTotalDose")
    if total_dose_bq <= 0:
        raise ValueError(f"invalid RadionuclideTotalDose for SUVbw: {total_dose_bq}")

    effective_dose_bq = _effective_injected_dose_bq(ds, radio, total_dose_bq)
    if effective_dose_bq <= 0:
        raise ValueError(f"invalid effective injected dose for SUVbw: {effective_dose_bq}")

    # SUVbw = activity concentration [Bq/ml] * patient weight [g] / effective dose [Bq].
    return float(patient_weight_kg * 1000.0 / effective_dose_bq)


def _effective_injected_dose_bq(
    ds: pydicom.dataset.Dataset,
    radio: pydicom.dataset.Dataset,
    total_dose_bq: float,
) -> float:
    decay_correction = _upper_text(getattr(ds, "DecayCorrection", ""))
    if decay_correction == "ADMIN":
        return total_dose_bq

    half_life_s = _required_float(radio, "RadionuclideHalfLife")
    if half_life_s <= 0:
        raise ValueError(f"invalid RadionuclideHalfLife for SUVbw: {half_life_s}")

    scan_dt = _scan_datetime(ds)
    injection_dt = _radiopharmaceutical_start_datetime(ds, radio, scan_dt)
    delta_seconds = (scan_dt - injection_dt).total_seconds()
    if delta_seconds < 0:
        raise ValueError(
            "radiopharmaceutical start time is after scan time: "
            f"start={injection_dt}, scan={scan_dt}"
        )

    return float(total_dose_bq * math.exp(-math.log(2.0) * delta_seconds / half_life_s))


def _radiopharmaceutical_item(ds: pydicom.dataset.Dataset) -> pydicom.dataset.Dataset:
    seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if seq is None or len(seq) == 0:
        raise ValueError("RadiopharmaceuticalInformationSequence is required for PET SUVbw conversion")
    return seq[0]


def _scan_datetime(ds: pydicom.dataset.Dataset) -> _dt.datetime:
    for name in ("AcquisitionDateTime", "FrameReferenceDateTime"):
        value = getattr(ds, name, None)
        if value not in (None, ""):
            return _parse_dicom_datetime(value)

    for date_name, time_name in (
        ("SeriesDate", "SeriesTime"),
        ("AcquisitionDate", "AcquisitionTime"),
        ("ContentDate", "ContentTime"),
        ("StudyDate", "StudyTime"),
    ):
        date_value = getattr(ds, date_name, None)
        time_value = getattr(ds, time_name, None)
        if date_value not in (None, "") and time_value not in (None, ""):
            return _parse_dicom_date_time(date_value, time_value)

    raise ValueError("scan date/time is required for PET SUVbw conversion")


def _radiopharmaceutical_start_datetime(
    ds: pydicom.dataset.Dataset,
    radio: pydicom.dataset.Dataset,
    scan_dt: _dt.datetime,
) -> _dt.datetime:
    for source in (radio, ds):
        value = getattr(source, "RadiopharmaceuticalStartDateTime", None)
        if value not in (None, ""):
            return _parse_dicom_datetime(value)

    start_time = None
    for source in (radio, ds):
        value = getattr(source, "RadiopharmaceuticalStartTime", None)
        if value not in (None, ""):
            start_time = value
            break

    if start_time in (None, ""):
        raise ValueError("RadiopharmaceuticalStartTime is required for PET SUVbw conversion")

    parsed_time = _parse_dicom_time(start_time)
    start_dt = _dt.datetime.combine(scan_dt.date(), parsed_time)
    if start_dt > scan_dt:
        start_dt -= _dt.timedelta(days=1)
    return start_dt


def _parse_dicom_datetime(value: object) -> _dt.datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("empty DICOM datetime")
    text = re.sub(r"([+-]\d{4})$", "", text)
    main, frac = _split_fraction(text)
    digits = re.sub(r"\D", "", main)
    if len(digits) < 8:
        raise ValueError(f"invalid DICOM datetime: {value}")

    year = int(digits[0:4])
    month = int(digits[4:6])
    day = int(digits[6:8])
    hour = int(digits[8:10]) if len(digits) >= 10 else 0
    minute = int(digits[10:12]) if len(digits) >= 12 else 0
    second = int(digits[12:14]) if len(digits) >= 14 else 0
    microsecond = _fraction_to_microsecond(frac)
    return _dt.datetime(year, month, day, hour, minute, second, microsecond)


def _parse_dicom_date_time(date_value: object, time_value: object) -> _dt.datetime:
    date_digits = re.sub(r"\D", "", str(date_value).strip())
    if len(date_digits) < 8:
        raise ValueError(f"invalid DICOM date: {date_value}")
    parsed_date = _dt.date(int(date_digits[0:4]), int(date_digits[4:6]), int(date_digits[6:8]))
    parsed_time = _parse_dicom_time(time_value)
    return _dt.datetime.combine(parsed_date, parsed_time)


def _parse_dicom_time(value: object) -> _dt.time:
    text = str(value).strip()
    if not text:
        raise ValueError("empty DICOM time")
    main, frac = _split_fraction(text)
    digits = re.sub(r"\D", "", main)
    if len(digits) < 2:
        raise ValueError(f"invalid DICOM time: {value}")

    hour = int(digits[0:2])
    minute = int(digits[2:4]) if len(digits) >= 4 else 0
    second = int(digits[4:6]) if len(digits) >= 6 else 0
    microsecond = _fraction_to_microsecond(frac)
    return _dt.time(hour, minute, second, microsecond)


def _split_fraction(text: str) -> Tuple[str, str]:
    if "." not in text:
        return text, ""
    main, frac = text.split(".", 1)
    return main, frac


def _fraction_to_microsecond(frac: str) -> int:
    digits = re.sub(r"\D", "", frac)
    if not digits:
        return 0
    return int((digits + "000000")[:6])


def _required_float(ds: pydicom.dataset.Dataset, name: str) -> float:
    value = getattr(ds, name, None)
    if value in (None, ""):
        raise ValueError(f"{name} is required for PET SUVbw conversion")
    return float(value)


def _upper_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _build_slice_series_reference(series_dir: Path, expected_modality: Optional[str]) -> NiftiVolume:
    slices = _read_slice_headers(series_dir, expected_modality=expected_modality)
    if not slices:
        raise RuntimeError(f"no readable DICOM slices found in {series_dir}")
    first_plane = slices[0]["plane"]
    slice_spacing = _slice_spacing_from_positions([float(item["position"]) for item in slices])
    affine_lps = _affine_lps_from_plane(first_plane, slice_spacing)
    return NiftiVolume(data=np.empty((1, 1, len(slices)), dtype=np.uint8), affine_lps=affine_lps)


def _read_slice_headers(series_dir: Path, expected_modality: Optional[str]) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for path in _iter_dicom_files(series_dir):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            modality = str(getattr(ds, "Modality", "")).upper()
            if expected_modality and modality and modality != expected_modality:
                continue
            plane = _root_image_plane(ds)
            position = float(np.dot(plane.ipp, plane.normal))
            items.append({"path": path, "plane": plane, "position": position})
        except Exception:
            continue

    if not items:
        return []
    items.sort(key=lambda item: float(item["position"]))
    return items


def _convert_seg_series(
    seg_dir: Path,
    output_path: Path,
    reference_z_axis_lps: Optional[np.ndarray],
) -> NiftiVolume:
    mask_files = _iter_dicom_files(seg_dir)
    if not mask_files:
        raise RuntimeError("no SEG DICOM files found")

    frame_records: List[Tuple[float, ImagePlane, np.ndarray]] = []
    for path in mask_files:
        ds = pydicom.dcmread(path)
        arr = np.asarray(ds.pixel_array)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        if arr.ndim != 3:
            raise ValueError(f"expected 2D/3D SEG pixel array, got shape {arr.shape}: {path}")

        for frame_idx in range(arr.shape[0]):
            plane = _seg_frame_plane(ds, frame_idx)
            position = float(np.dot(plane.ipp, plane.normal))
            frame_records.append((position, plane, (arr[frame_idx] > 0).astype(np.uint8)))

    if not frame_records:
        raise RuntimeError("SEG contains no frames")

    frame_records.sort(key=lambda item: item[0])
    unique_frames = _merge_duplicate_seg_frames(frame_records)
    first_plane = unique_frames[0][1]
    positions = [record[0] for record in unique_frames]
    slice_spacing = _slice_spacing_from_positions(positions)

    volume_data = np.stack([record[2] for record in unique_frames], axis=2).astype(np.uint8)
    affine_lps = _affine_lps_from_plane(first_plane, slice_spacing)
    volume = NiftiVolume(data=volume_data, affine_lps=affine_lps)
    volume = _align_z_axis_to_reference(volume, reference_z_axis_lps)

    _save_nifti(volume, output_path)
    return volume


def _merge_duplicate_seg_frames(
    frame_records: List[Tuple[float, ImagePlane, np.ndarray]],
    decimals: int = 5,
) -> List[Tuple[float, ImagePlane, np.ndarray]]:
    merged: List[Tuple[float, ImagePlane, np.ndarray]] = []
    current_key: Optional[float] = None
    current_position: Optional[float] = None
    current_plane: Optional[ImagePlane] = None
    current_mask: Optional[np.ndarray] = None

    for position, plane, mask in frame_records:
        key = round(position, decimals)
        if current_key is None or key != current_key:
            if current_mask is not None and current_plane is not None and current_position is not None:
                merged.append((current_position, current_plane, current_mask))
            current_key = key
            current_position = position
            current_plane = plane
            current_mask = mask.copy()
        else:
            if current_mask is None:
                current_mask = mask.copy()
            else:
                current_mask = np.maximum(current_mask, mask)

    if current_mask is not None and current_plane is not None and current_position is not None:
        merged.append((current_position, current_plane, current_mask))
    return merged


def _root_image_plane(ds: pydicom.dataset.Dataset) -> ImagePlane:
    ipp = _vector3(getattr(ds, "ImagePositionPatient"))
    orientation = _float_sequence(getattr(ds, "ImageOrientationPatient"), expected_len=6)
    pixel_spacing = _float_sequence(getattr(ds, "PixelSpacing"), expected_len=2)
    return _make_image_plane(ipp, orientation, pixel_spacing)


def _seg_frame_plane(ds: pydicom.dataset.Dataset, frame_idx: int) -> ImagePlane:
    ipp = _seg_frame_image_position(ds, frame_idx)
    orientation = _seg_frame_image_orientation(ds, frame_idx)
    pixel_spacing = _seg_frame_pixel_spacing(ds, frame_idx)
    return _make_image_plane(ipp, orientation, pixel_spacing)


def _seg_frame_image_position(ds: pydicom.dataset.Dataset, frame_idx: int) -> np.ndarray:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(
            frame_group,
            ("PlanePositionSequence", "PlanePositionPatientSequence", "PlanePositionSlideSequence"),
        )
        if item is not None and hasattr(item, "ImagePositionPatient"):
            return _vector3(getattr(item, "ImagePositionPatient"))

    if hasattr(ds, "ImagePositionPatient"):
        return _vector3(getattr(ds, "ImagePositionPatient"))
    raise ValueError("ImagePositionPatient not found for SEG frame")


def _seg_frame_image_orientation(ds: pydicom.dataset.Dataset, frame_idx: int) -> List[float]:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(
            frame_group,
            ("PlaneOrientationSequence", "PlaneOrientationPatientSequence"),
        )
        if item is not None and hasattr(item, "ImageOrientationPatient"):
            return _float_sequence(getattr(item, "ImageOrientationPatient"), expected_len=6)

    shared_group = _shared_group(ds)
    if shared_group is not None:
        item = _first_sequence_item(
            shared_group,
            ("PlaneOrientationSequence", "PlaneOrientationPatientSequence"),
        )
        if item is not None and hasattr(item, "ImageOrientationPatient"):
            return _float_sequence(getattr(item, "ImageOrientationPatient"), expected_len=6)

    if hasattr(ds, "ImageOrientationPatient"):
        return _float_sequence(getattr(ds, "ImageOrientationPatient"), expected_len=6)
    raise ValueError("ImageOrientationPatient not found for SEG frame")


def _seg_frame_pixel_spacing(ds: pydicom.dataset.Dataset, frame_idx: int) -> List[float]:
    frame_group = _per_frame_group(ds, frame_idx)
    if frame_group is not None:
        item = _first_sequence_item(frame_group, ("PixelMeasuresSequence",))
        if item is not None and hasattr(item, "PixelSpacing"):
            return _float_sequence(getattr(item, "PixelSpacing"), expected_len=2)

    shared_group = _shared_group(ds)
    if shared_group is not None:
        item = _first_sequence_item(shared_group, ("PixelMeasuresSequence",))
        if item is not None and hasattr(item, "PixelSpacing"):
            return _float_sequence(getattr(item, "PixelSpacing"), expected_len=2)

    if hasattr(ds, "PixelSpacing"):
        return _float_sequence(getattr(ds, "PixelSpacing"), expected_len=2)
    raise ValueError("PixelSpacing not found for SEG frame")


def _per_frame_group(ds: pydicom.dataset.Dataset, frame_idx: int) -> Optional[pydicom.dataset.Dataset]:
    seq = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if seq is not None and frame_idx < len(seq):
        return seq[frame_idx]
    return None


def _shared_group(ds: pydicom.dataset.Dataset) -> Optional[pydicom.dataset.Dataset]:
    seq = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if seq is not None and len(seq) > 0:
        return seq[0]
    return None


def _first_sequence_item(
    ds: pydicom.dataset.Dataset,
    names: Sequence[str],
) -> Optional[pydicom.dataset.Dataset]:
    for name in names:
        seq = getattr(ds, name, None)
        if seq is not None and len(seq) > 0:
            return seq[0]
    return None


def _make_image_plane(
    ipp: np.ndarray,
    orientation: Sequence[float],
    pixel_spacing: Sequence[float],
) -> ImagePlane:
    row_dir = _unit(np.asarray(orientation[:3], dtype=np.float64))
    col_dir = _unit(np.asarray(orientation[3:6], dtype=np.float64))
    normal = _unit(np.cross(row_dir, col_dir))
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    if row_spacing <= 0 or col_spacing <= 0:
        raise ValueError(f"invalid PixelSpacing: {pixel_spacing}")
    return ImagePlane(
        ipp=ipp.astype(np.float64, copy=False),
        row_dir=row_dir,
        col_dir=col_dir,
        normal=normal,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
    )


def _affine_lps_from_plane(plane: ImagePlane, slice_spacing: float) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    # Data shape is [row, col, slice]. DICOM row index moves along col_dir;
    # DICOM column index moves along row_dir.
    affine[:3, 0] = plane.col_dir * plane.row_spacing
    affine[:3, 1] = plane.row_dir * plane.col_spacing
    affine[:3, 2] = plane.normal * slice_spacing
    affine[:3, 3] = plane.ipp
    return affine


def _align_z_axis_to_reference(
    volume: NiftiVolume,
    reference_z_axis_lps: Optional[np.ndarray],
) -> NiftiVolume:
    if reference_z_axis_lps is None:
        return volume
    if volume.data.shape[2] < 2:
        return volume

    current = volume.affine_lps[:3, 2]
    if float(np.dot(current, reference_z_axis_lps)) >= 0:
        return volume

    flipped_data = np.flip(volume.data, axis=2).copy()
    flipped_affine = volume.affine_lps.copy()
    flipped_affine[:3, 3] = volume.affine_lps[:3, 3] + volume.affine_lps[:3, 2] * (volume.data.shape[2] - 1)
    flipped_affine[:3, 2] = -volume.affine_lps[:3, 2]
    return NiftiVolume(data=flipped_data, affine_lps=flipped_affine)


def _save_nifti(volume: NiftiVolume, output_path: Path) -> None:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("nibabel is required. Install with: pip install nibabel") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    affine_ras = _lps_affine_to_ras(volume.affine_lps)
    image = nib.Nifti1Image(volume.data, affine_ras)
    image.header.set_sform(affine_ras, code=1)
    image.header.set_qform(affine_ras, code=1)
    image.header.set_data_dtype(volume.data.dtype)
    nib.save(image, str(output_path))


def _lps_affine_to_ras(affine_lps: np.ndarray) -> np.ndarray:
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def _slice_spacing_from_positions(positions: Sequence[float]) -> float:
    if len(positions) < 2:
        return 1.0
    values = np.asarray(sorted(set(round(float(p), 6) for p in positions)), dtype=np.float64)
    diffs = np.diff(values)
    diffs = diffs[diffs > 1e-6]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _can_store_as_int16_without_clipping(values: np.ndarray, atol: float = 1e-3) -> bool:
    if values.size == 0:
        return True

    info = np.iinfo(np.int16)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if min_value < info.min or max_value > info.max:
        return False

    sample = values
    if values.size > 200_000:
        sample = values.reshape(-1)[:: max(1, values.size // 200_000)]
    return bool(np.all(np.abs(sample - np.rint(sample)) <= atol))


def _vector3(value: object) -> np.ndarray:
    values = _float_sequence(value, expected_len=3)
    return np.asarray(values, dtype=np.float64)


def _float_sequence(value: object, expected_len: int) -> List[float]:
    values = [float(item) for item in value]
    if len(values) < expected_len:
        raise ValueError(f"expected at least {expected_len} values, got {values}")
    return values[:expected_len]


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("zero-length direction vector")
    return value / norm


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PSMA DICOM CT/PET/SEG folders to ct.nii.gz, pet.nii.gz, seg.nii.gz."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing nii.gz outputs.")
    parser.add_argument("--verbose", action="store_true", help="Print progress and failures.")
    args = parser.parse_args()

    records = convert_psma_dicom_to_nifti(
        args.root_dir,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )

    written = sum(1 for item in records if item.status == "written")
    skipped = sum(1 for item in records if item.status == "skipped_exists")
    failed = [item for item in records if item.status.startswith("failed")]
    print(f"Written: {written}")
    print(f"Skipped existing: {skipped}")
    print(f"Failed: {len(failed)}")
    for item in failed[:20]:
        print(f"[failed] {item.study_id} {item.series_type}: {item.status}", file=sys.stderr)
    if len(failed) > 20:
        print(f"... {len(failed) - 20} more failures", file=sys.stderr)


if __name__ == "__main__":
    main()
