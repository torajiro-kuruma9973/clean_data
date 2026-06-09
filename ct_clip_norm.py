#!/usr/bin/env python3
"""
Preprocess CT NIfTI files for model input.

Expected PSMA-style structure:
  root/
    PSMA_xxx/
      study-id/
        xxx-CT-xxxxx/
          ct.nii.gz

For every ct.nii.gz:
  1. Clip HU values to the user-provided window.
  2. Normalize clipped HU values to [0, 1].
  3. Save ct_norm.nii.gz in the same CT folder.

No resampling is performed. Shape, affine, sform, and qform are preserved.

Dependencies:
  pip install nibabel numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np


@dataclass
class CtPreprocessRecord:
    study_id: str
    input_path: Path
    output_path: Path
    status: str
    shape: Tuple[int, int, int]
    clip_window: Tuple[float, float]
    dtype: str
    output_size: str


def preprocess_ct_nii(
    root_dir: Union[str, Path],
    clip_window: Tuple[float, float],
    dtype: Union[str, np.dtype] = "float32",
    output_name: str = "ct_norm.nii.gz",
    overwrite: bool = False,
    verbose: bool = False,
) -> List[CtPreprocessRecord]:
    """Clip and normalize all ct.nii.gz files under a PSMA-style root.

    Args:
        root_dir: Root directory containing project-id and study-id folders.
        clip_window: Two values (lower, upper), for example (-1000, 2000).
        dtype: Output NIfTI storage dtype. Use float32 unless you have a
            specific reason to use float64; NIfTI does not define float16.
        output_name: Output filename inside each CT folder.
        overwrite: If True, rewrite existing outputs.
        verbose: Print skipped or failed study details.

    Returns:
        A list of preprocessing records.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    lower, upper = _validate_clip_window(clip_window)
    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(
            f"dtype must be float32 or float64 for NIfTI output; got {out_dtype}. "
            "NIfTI does not define a float16 storage datatype."
        )

    records: List[CtPreprocessRecord] = []
    ct_items = list(_find_ct_nii_files(root))
    total = len(ct_items)

    for index, (study_dir, ct_path) in enumerate(ct_items, start=1):
        study_id = study_dir.name
        output_path = ct_path.parent / output_name

        if output_path.exists() and not overwrite:
            record = CtPreprocessRecord(
                study_id=study_id,
                input_path=ct_path,
                output_path=output_path,
                status="skipped_exists",
                shape=_safe_nifti_shape(ct_path),
                clip_window=(lower, upper),
                dtype=_safe_nifti_dtype(output_path),
                output_size=_format_file_size(output_path),
            )
            records.append(record)
            print(
                f"[{index}/{total}] {study_id} skipped: {output_path} "
                f"shape={record.shape}, dtype={record.dtype}, size={record.output_size}"
            )
            continue

        try:
            img = nib.load(str(ct_path))
            shape = _spatial_shape(img, ct_path)
            ct_hu = _load_3d_data(img, ct_path)
            ct_norm = _clip_and_normalize(ct_hu, lower, upper).astype(out_dtype, copy=False)

            _save_with_original_grid(ct_norm, img, output_path, out_dtype)

            record = CtPreprocessRecord(
                study_id=study_id,
                input_path=ct_path,
                output_path=output_path,
                status="written",
                shape=shape,
                clip_window=(lower, upper),
                dtype=str(out_dtype),
                output_size=_format_file_size(output_path),
            )
            records.append(record)
            print(
                f"[{index}/{total}] {study_id} written: {output_path} "
                f"shape={shape}, clip=({lower:g},{upper:g}), dtype={out_dtype}, "
                f"size={record.output_size}"
            )
        except Exception as exc:
            records.append(
                CtPreprocessRecord(
                    study_id=study_id,
                    input_path=ct_path,
                    output_path=output_path,
                    status=f"failed: {exc}",
                    shape=_safe_nifti_shape(ct_path),
                    clip_window=(lower, upper),
                    dtype=str(out_dtype),
                    output_size="missing",
                )
            )
            _log(verbose, f"[fail] {study_id} CT preprocessing failed: {ct_path}: {exc}")

    print(f"CT files checked: {total}; outputs written/skipped: {len(records)}")
    return records


def _find_ct_nii_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (study_dir, ct.nii.gz) for direct CT child folders."""
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())

    for study_dir in directories:
        try:
            children = list(study_dir.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue
            if _classify_series_dir(child) != "ct":
                continue
            ct_path = child / "ct.nii.gz"
            if ct_path.is_file():
                yield study_dir, ct_path


def _classify_series_dir(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "segmentation" in name:
        return "seg"
    if "pet" in name:
        return "pet"
    if "ct" in name:
        return "ct"
    return None


def _validate_clip_window(clip_window: Sequence[float]) -> Tuple[float, float]:
    if len(clip_window) != 2:
        raise ValueError(f"clip_window must contain exactly two values, got: {clip_window}")
    lower = float(clip_window[0])
    upper = float(clip_window[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"clip_window values must be finite, got: {clip_window}")
    if upper <= lower:
        raise ValueError(f"clip_window upper must be > lower, got: {clip_window}")
    return lower, upper


def _spatial_shape(img: nib.spatialimages.SpatialImage, path: Path) -> Tuple[int, int, int]:
    shape = img.shape
    if len(shape) < 3:
        raise ValueError(f"NIfTI must be at least 3D: {path}")
    if len(shape) > 3 and any(dim != 1 for dim in shape[3:]):
        raise ValueError(f"only 3D NIfTI or singleton extra dimensions are supported: {path}, shape={shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _load_3d_data(img: nib.spatialimages.SpatialImage, path: Path) -> np.ndarray:
    _spatial_shape(img, path)
    data = img.get_fdata(dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected 3D data after squeeze: {path}, shape={data.shape}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"CT data contains NaN or inf: {path}")
    return data


def _clip_and_normalize(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.clip(data, lower, upper)
    normalized = (clipped - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0)


def _save_with_original_grid(
    data: np.ndarray,
    source_img: nib.spatialimages.SpatialImage,
    output_path: Path,
    dtype: np.dtype,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = np.asarray(data, dtype=dtype)

    header = source_img.header.copy()
    header.set_data_dtype(output_data.dtype)
    header.set_slope_inter(None, None)

    image = nib.Nifti1Image(output_data, source_img.affine, header=header)
    sform_code = int(source_img.header["sform_code"]) or 1
    qform_code = int(source_img.header["qform_code"]) or 1
    image.set_sform(source_img.affine, code=sform_code)
    image.set_qform(source_img.affine, code=qform_code)
    nib.save(image, str(output_path))


def _safe_nifti_shape(path: Path) -> Tuple[int, int, int]:
    try:
        return _spatial_shape(nib.load(str(path)), path)
    except Exception:
        return (0, 0, 0)


def _safe_nifti_dtype(path: Path) -> str:
    try:
        return str(np.dtype(nib.load(str(path)).header.get_data_dtype()))
    except Exception:
        return "unknown"


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
        description="Clip HU CT NIfTI files and normalize them to [0, 1]."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders")
    parser.add_argument(
        "--clip-window",
        nargs=2,
        type=float,
        required=True,
        metavar=("LOWER", "UPPER"),
        help="HU clipping window, for example: --clip-window -1000 2000",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Output dtype for ct_norm.nii.gz (default: float32)",
    )
    parser.add_argument(
        "--output-name",
        default="ct_norm.nii.gz",
        help="Output filename inside each CT folder (default: ct_norm.nii.gz)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing normalized CT files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print failed study details",
    )
    args = parser.parse_args()

    preprocess_ct_nii(
        args.root_dir,
        clip_window=(args.clip_window[0], args.clip_window[1]),
        dtype=args.dtype,
        output_name=args.output_name,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()