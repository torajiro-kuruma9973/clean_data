#!/usr/bin/env python3
"""
Preprocess resampled PET NIfTI files for model labels.


Expected PSMA-style structure:
  root/
    PSMA_xxx/
      study-id/
        xxx-PET-xxxxx/
          pet_resampled.nii.gz


For every pet_resampled.nii.gz:
  1. Clip SUV values to the user-provided window.
  2. Apply the selected intensity scaling.
     Default: log1p scaling, because PET SUV values are usually long-tailed.
  3. Normalize scaled SUV values to [0, 1].
  4. Save pet_norm.nii.gz in the same PET folder.


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



PET_RESAMPLED_NAME = "pet_resampled.nii.gz"
PET_ORIGINAL_NAME = "pet.nii.gz"



@dataclass
class PetPreprocessRecord:
    study_id: str
    input_path: Path
    output_path: Path
    status: str
    shape: Tuple[int, int, int]
    clip_window: Tuple[float, float]
    scale_mode: str
    gamma: Optional[float]
    dtype: str
    output_size: str



def preprocess_pet_nii(
    root_dir: Union[str, Path],
    clip_window: Tuple[float, float] = (0.0, 50.0),
    output_name: str = "pet_norm.nii.gz",
    dtype: Union[str, np.dtype] = "float32",
    scale_mode: str = "log",
    gamma: float = 0.5,
    overwrite: bool = False,
    verbose: bool = False,
) -> List[PetPreprocessRecord]:
    """Clip, scale, and normalize all resampled PET SUV NIfTI files under root_dir.


    Args:
        root_dir: Root directory containing project-id and study-id folders.
        clip_window: Two values (lower, upper), for example (0, 50).
        output_name: Output filename inside each PET folder.
        dtype: Output NIfTI storage dtype. Use float32 unless you have a
            specific reason to use float64; NIfTI does not define float16.
        scale_mode: Intensity scaling before final [0, 1] normalization.
            Use "log" by default for long-tailed PET SUV values. Use "gamma"
            for gamma scaling or "linear" to keep the old behavior.
        gamma: Gamma value used only when scale_mode="gamma". Values below 1
            expand low/mid SUV values; values above 1 emphasize high SUV values.
        overwrite: If True, rewrite existing outputs.
        verbose: Print skipped or failed study details.


    Returns:
        A list of preprocessing records.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")
    if output_name in {PET_ORIGINAL_NAME, PET_RESAMPLED_NAME}:
        raise ValueError(
            f"output_name must not be {PET_ORIGINAL_NAME!r} or {PET_RESAMPLED_NAME!r}; "
            "this script never modifies source PET files"
        )


    lower, upper = _validate_clip_window(clip_window)
    scale_mode = _validate_scale_mode(scale_mode)
    gamma_value = _validate_gamma(gamma) if scale_mode == "gamma" else None
    out_dtype = np.dtype(dtype)
    if out_dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(
            f"dtype must be float32 or float64 for NIfTI output; got {out_dtype}. "
            "NIfTI does not define a float16 storage datatype."
        )


    records: List[PetPreprocessRecord] = []
    pet_items = list(_find_pet_resampled_nii_files(root))
    total = len(pet_items)


    for index, (study_dir, pet_path) in enumerate(pet_items, start=1):
        study_id = study_dir.name
        output_path = pet_path.parent / output_name


        if output_path.exists() and not overwrite:
            record = PetPreprocessRecord(
                study_id=study_id,
                input_path=pet_path,
                output_path=output_path,
                status="skipped_exists",
                shape=_safe_nifti_shape(pet_path),
                clip_window=(lower, upper),
                scale_mode=scale_mode,
                gamma=gamma_value,
                dtype=_safe_nifti_dtype(output_path),
                output_size=_format_file_size(output_path),
            )
            records.append(record)
            print(
                f"[{index}/{total}] {study_id} skipped: {output_path} "
                f"shape={record.shape}, scale={_format_scale(scale_mode, gamma_value)}, "
                f"dtype={record.dtype}, size={record.output_size}"
            )
            continue


        try:
            img = nib.load(str(pet_path))
            shape = _spatial_shape(img, pet_path)
            pet_suv = _load_3d_data(img, pet_path)
            pet_norm = _clip_scale_and_normalize(
                pet_suv,
                lower,
                upper,
                scale_mode=scale_mode,
                gamma=gamma_value,
            ).astype(out_dtype, copy=False)


            _save_with_original_grid(pet_norm, img, output_path, out_dtype)


            record = PetPreprocessRecord(
                study_id=study_id,
                input_path=pet_path,
                output_path=output_path,
                status="written",
                shape=shape,
                clip_window=(lower, upper),
                scale_mode=scale_mode,
                gamma=gamma_value,
                dtype=str(out_dtype),
                output_size=_format_file_size(output_path),
            )
            records.append(record)
            print(
                f"[{index}/{total}] {study_id} written: {output_path} "
                f"shape={shape}, clip=({lower:g},{upper:g}), "
                f"scale={_format_scale(scale_mode, gamma_value)}, dtype={out_dtype}, "
                f"size={record.output_size}"
            )
        except Exception as exc:
            records.append(
                PetPreprocessRecord(
                    study_id=study_id,
                    input_path=pet_path,
                    output_path=output_path,
                    status=f"failed: {exc}",
                    shape=_safe_nifti_shape(pet_path),
                    clip_window=(lower, upper),
                    scale_mode=scale_mode,
                    gamma=gamma_value,
                    dtype=str(out_dtype),
                    output_size="missing",
                )
            )
            _log(verbose, f"[fail] {study_id} PET preprocessing failed: {pet_path}: {exc}")


    print(f"PET files checked: {total}; outputs written/skipped: {len(records)}")
    return records



def _find_pet_resampled_nii_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (study_dir, pet_resampled.nii.gz) for direct PET child folders."""
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
            if _classify_series_dir(child) != "pet":
                continue
            pet_path = child / PET_RESAMPLED_NAME
            if pet_path.is_file():
                yield study_dir, pet_path



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



def _validate_scale_mode(scale_mode: str) -> str:
    mode = scale_mode.strip().lower()
    allowed = {"linear", "log", "gamma"}
    if mode not in allowed:
        raise ValueError(f"scale_mode must be one of {sorted(allowed)}, got: {scale_mode!r}")
    return mode



def _validate_gamma(gamma: float) -> float:
    value = float(gamma)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"gamma must be a finite positive value, got: {gamma}")
    return value



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
        raise ValueError(f"PET data contains NaN or inf: {path}")
    return data



def _clip_scale_and_normalize(
    data: np.ndarray,
    lower: float,
    upper: float,
    scale_mode: str,
    gamma: Optional[float],
) -> np.ndarray:
    clipped = np.clip(data, lower, upper)
    shifted = clipped - lower
    width = upper - lower


    if scale_mode == "linear":
        normalized = shifted / width
    elif scale_mode == "log":
        normalized = np.log1p(shifted) / np.log1p(width)
    elif scale_mode == "gamma":
        print(f"GAMMA scaling is atopted: gamma = {gamma}")
        if gamma is None:
            raise ValueError("gamma must be provided when scale_mode='gamma'")
        normalized = np.power(shifted / width, gamma)
    else:
        raise ValueError(f"unsupported scale_mode: {scale_mode}")


    return np.clip(normalized, 0.0, 1.0)



def _format_scale(scale_mode: str, gamma: Optional[float]) -> str:
    if scale_mode == "log":
        return "log1p"
    if scale_mode == "gamma":
        return f"gamma({gamma:g})"
    return "linear"



def invert_pet_normalization(
    normalized: np.ndarray,
    clip_window: Tuple[float, float] = (0.0, 50.0),
    scale_mode: str = "log",
    gamma: float = 0.5,
) -> np.ndarray:
    """Invert this script's normalization back to clipped SUV values."""
    lower, upper = _validate_clip_window(clip_window)
    scale_mode = _validate_scale_mode(scale_mode)
    gamma_value = _validate_gamma(gamma) if scale_mode == "gamma" else None


    normalized = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    width = upper - lower


    if scale_mode == "linear":
        shifted = normalized * width
    elif scale_mode == "log":
        shifted = np.expm1(normalized * np.log1p(width))
    elif scale_mode == "gamma":
        shifted = np.power(normalized, 1.0 / gamma_value) * width
    else:
        raise ValueError(f"unsupported scale_mode: {scale_mode}")


    return np.clip(shifted + lower, lower, upper)



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
        description="Clip, scale, and normalize SUV PET NIfTI files to [0, 1]."
    )
    parser.add_argument("root_dir", help="Root directory containing PSMA project folders")
    parser.add_argument(
        "--clip-window",
        nargs=2,
        type=float,
        default=(0.0, 50.0),
        metavar=("LOWER", "UPPER"),
        help="SUV clipping window (default: 0 50)",
    )
    parser.add_argument(
        "--output-name",
        default="pet_norm.nii.gz",
        help="Output filename inside each PET folder (default: pet_norm.nii.gz)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float64"),
        help="Output dtype for pet_norm.nii.gz (default: float32)",
    )
    parser.add_argument(
        "--scale-mode",
        default="gamma",
        choices=("log", "gamma", "linear"),
        help=(
            "Intensity scaling after clipping and before final [0, 1] normalization. "
            "Default: log, implemented as log1p, which is suitable for long-tailed SUV values."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value when --scale-mode gamma is used (default: 0.5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing normalized PET files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print failed study details",
    )
    args = parser.parse_args()


    preprocess_pet_nii(
        args.root_dir,
        clip_window=(args.clip_window[0], args.clip_window[1]),
        output_name=args.output_name,
        dtype=args.dtype,
        scale_mode=args.scale_mode,
        gamma=args.gamma,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )



if __name__ == "__main__":
    main()