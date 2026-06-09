#!/usr/bin/env python3
"""
Resample PET and SEG NIfTI files to the same physical grid as same-study CT.

Expected PSMA-style structure:
  root/
    PSMA_xxx/
      study-id/
        xxx-CT-xxxxx/
          ct.nii.gz
        xxx-PET-xxxxx/
          pet.nii.gz
        xxx-Segmentation-xxxxx/
          seg.nii.gz

Outputs:
  xxx-PET-xxxxx/pet_resampled.nii.gz
  xxx-Segmentation-xxxxx/seg_resampled.nii.gz

Resampling is done by physical coordinates:
  target CT voxel index -> CT affine -> world coordinate
  world coordinate -> inverse source affine -> source PET/SEG voxel index

PET uses linear interpolation. SEG uses nearest-neighbor interpolation and
outside-source values are filled with 0, so missing/non-covered SEG slices
become all-background on the CT grid.

Dependencies:
  pip install nibabel numpy scipy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
from scipy import ndimage


@dataclass
class ResampleRecord:
    study_id: str
    input_path: Path
    output_path: Path
    series_type: str
    status: str
    input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    output_size: str
    input_storage: str
    output_storage: str


def resample_pet_seg_to_ct(
    root_dir: Union[str, Path],
    overwrite: bool = False,
    verbose: bool = False,
) -> List[ResampleRecord]:
    """Resample every same-study PET and SEG NIfTI to the CT NIfTI grid.

    Args:
        root_dir: Root directory containing project-id and study-id folders.
        overwrite: If True, rewrite existing pet_resampled/seg_resampled files.
        verbose: Print skipped or failed study details.

    Returns:
        A list of per-output records.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    records: List[ResampleRecord] = []
    study_dirs = list(_find_study_dirs(root))
    total_studies = len(study_dirs)

    for study_index, study_dir in enumerate(study_dirs, start=1):
        study_id = study_dir.name
        paths = _find_study_niftis(study_dir)
        ct_path = paths.get("ct")

        if ct_path is None:
            _log(verbose, f"[skip] {study_id}: missing ct.nii.gz")
            continue

        try:
            ct_img = nib.load(str(ct_path))
            target_shape = _spatial_shape(ct_img, ct_path)
        except Exception as exc:
            _log(verbose, f"[fail] {study_id}: cannot read CT NIfTI {ct_path}: {exc}")
            continue

        completed = 0
        for series_type, order, dtype, output_name in (
            ("pet", 1, np.float32, "pet_resampled.nii.gz"),
            ("seg", 0, np.uint8, "seg_resampled.nii.gz"),
        ):
            input_path = paths.get(series_type)
            if input_path is None:
                _log(verbose, f"[skip] {study_id}: missing {series_type}.nii.gz")
                continue

            output_path = input_path.parent / output_name
            input_storage = _safe_nifti_storage(input_path)
            if output_path.exists() and not overwrite:
                output_storage = _safe_nifti_storage(output_path)
                record = ResampleRecord(
                    study_id=study_id,
                    input_path=input_path,
                    output_path=output_path,
                    series_type=series_type,
                    status="skipped_exists",
                    input_shape=_safe_nifti_shape(input_path),
                    output_shape=target_shape,
                    output_size=_format_file_size(output_path),
                    input_storage=input_storage,
                    output_storage=output_storage,
                )
                records.append(record)
                completed += 1
                print(
                    f"[study {study_index}/{total_studies}] {study_id} "
                    f"{series_type.upper()} skipped: {output_path} ({record.output_size}); "
                    f"storage {record.input_storage} -> {record.output_storage}"
                )
                continue

            try:
                source_img = nib.load(str(input_path))
                input_storage = _nifti_storage_description(source_img)
                input_shape = _spatial_shape(source_img, input_path)
                data = _load_3d_data(source_img, input_path, dtype=np.float32)
                resampled = _resample_to_target_grid(
                    source_data=data,
                    source_affine=np.asarray(source_img.affine, dtype=np.float64),
                    target_affine=np.asarray(ct_img.affine, dtype=np.float64),
                    target_shape=target_shape,
                    order=order,
                )

                if series_type == "seg":
                    output_data = (resampled > 0).astype(np.uint8, copy=False)
                else:
                    output_data = resampled.astype(np.float32, copy=False)

                _save_like_ct(output_data, ct_img, output_path, dtype=dtype)
                output_storage = _safe_nifti_storage(output_path)

                record = ResampleRecord(
                    study_id=study_id,
                    input_path=input_path,
                    output_path=output_path,
                    series_type=series_type,
                    status="written",
                    input_shape=input_shape,
                    output_shape=target_shape,
                    output_size=_format_file_size(output_path),
                    input_storage=input_storage,
                    output_storage=output_storage,
                )
                records.append(record)
                completed += 1
                print(
                    f"[study {study_index}/{total_studies}] {study_id} "
                    f"{series_type.upper()} written: {output_path} "
                    f"shape {input_shape} -> {target_shape}, size {record.output_size}; "
                    f"storage {record.input_storage} -> {record.output_storage}"
                )
            except Exception as exc:
                records.append(
                    ResampleRecord(
                        study_id=study_id,
                        input_path=input_path,
                        output_path=output_path,
                        series_type=series_type,
                        status=f"failed: {exc}",
                        input_shape=_safe_nifti_shape(input_path),
                        output_shape=target_shape,
                        output_size="missing",
                        input_storage=input_storage,
                        output_storage="missing",
                    )
                )
                _log(verbose, f"[fail] {study_id} {series_type.upper()}: {exc}")

        print(f"Completed study {study_index}/{total_studies}: {study_id} ({completed}/2 outputs)")

    return records


def _find_study_dirs(root: Path) -> Iterable[Path]:
    """Yield directories that directly contain CT/PET/SEG NIfTI series folders."""
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())

    for directory in directories:
        if not directory.is_dir():
            continue
        paths = _find_study_niftis(directory)
        if paths["ct"] is not None and (paths["pet"] is not None or paths["seg"] is not None):
            yield directory


def _find_study_niftis(study_dir: Path) -> Dict[str, Optional[Path]]:
    paths: Dict[str, Optional[Path]] = {"ct": None, "pet": None, "seg": None}

    try:
        children = list(study_dir.iterdir())
    except OSError:
        return paths

    for child in children:
        if not child.is_dir():
            continue
        series_type = _classify_series_dir(child)
        if series_type is None or paths[series_type] is not None:
            continue

        filename = {
            "ct": "ct.nii.gz",
            "pet": "pet.nii.gz",
            "seg": "seg.nii.gz",
        }[series_type]
        candidate = child / filename
        if candidate.is_file():
            paths[series_type] = candidate

    return paths


def _classify_series_dir(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "segmentation" in name:
        return "seg"
    if "pet" in name:
        return "pet"
    if "ct" in name:
        return "ct"
    return None


def _spatial_shape(img: nib.spatialimages.SpatialImage, path: Path) -> Tuple[int, int, int]:
    shape = img.shape
    if len(shape) < 3:
        raise ValueError(f"NIfTI must be at least 3D: {path}")
    if len(shape) > 3 and any(dim != 1 for dim in shape[3:]):
        raise ValueError(f"only 3D NIfTI or singleton extra dimensions are supported: {path}, shape={shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _load_3d_data(
    img: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> np.ndarray:
    _spatial_shape(img, path)
    data = img.get_fdata(dtype=dtype)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected 3D data after squeeze: {path}, shape={data.shape}")
    return np.asarray(data, dtype=dtype)


def _resample_to_target_grid(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int,
) -> np.ndarray:
    if source_affine.shape != (4, 4) or target_affine.shape != (4, 4):
        raise ValueError("source_affine and target_affine must be 4x4 matrices")
    if not np.all(np.isfinite(source_affine)) or not np.all(np.isfinite(target_affine)):
        raise ValueError("source_affine and target_affine must contain finite values")

    transform = np.linalg.inv(source_affine) @ target_affine
    matrix = transform[:3, :3]
    offset = transform[:3, 3]
    output = np.empty(target_shape, dtype=np.float32)

    ndimage.affine_transform(
        source_data,
        matrix=matrix,
        offset=offset,
        output_shape=target_shape,
        output=output,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return output


def _save_like_ct(
    data: np.ndarray,
    ct_img: nib.spatialimages.SpatialImage,
    output_path: Path,
    dtype: np.dtype,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = np.asarray(data, dtype=dtype)
    header = ct_img.header.copy()
    header.set_data_dtype(output_data.dtype)
    header.set_slope_inter(None, None)

    image = nib.Nifti1Image(output_data, ct_img.affine, header=header)
    sform_code = int(ct_img.header["sform_code"]) or 1
    qform_code = int(ct_img.header["qform_code"]) or 1
    image.set_sform(ct_img.affine, code=sform_code)
    image.set_qform(ct_img.affine, code=qform_code)
    nib.save(image, str(output_path))


def _safe_nifti_shape(path: Path) -> Tuple[int, int, int]:
    try:
        return _spatial_shape(nib.load(str(path)), path)
    except Exception:
        return (0, 0, 0)


def _safe_nifti_storage(path: Path) -> str:
    try:
        return _nifti_storage_description(nib.load(str(path)))
    except Exception as exc:
        return f"unreadable_storage({exc})"


def _nifti_storage_description(img: nib.spatialimages.SpatialImage) -> str:
    header = img.header
    datatype_code = int(np.asarray(header["datatype"]).item())
    bitpix = int(np.asarray(header["bitpix"]).item())
    try:
        dtype = str(np.dtype(header.get_data_dtype()))
    except Exception:
        dtype = f"unresolved_datatype_code_{datatype_code}"
    slope, intercept = header.get_slope_inter()
    if slope is None and intercept is None:
        scaling = "scl=none"
    else:
        scaling = f"scl_slope={slope},scl_inter={intercept}"
    return f"dtype={dtype},datatype={datatype_code},bitpix={bitpix},{scaling}"


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


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resample same-study PET and SEG NIfTI files to CT NIfTI grid."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing pet_resampled.nii.gz and seg_resampled.nii.gz",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print skipped and failed study details",
    )
    args = parser.parse_args()

    resample_pet_seg_to_ct(
        args.root_dir,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()