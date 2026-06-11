#!/usr/bin/env python3
"""
Compare two NIfTI files and print whether they are equivalent.


Dependencies:
  pip install nibabel numpy
"""


from __future__ import annotations


import argparse
from pathlib import Path
from typing import Dict, Tuple, Union


import nibabel as nib
import numpy as np



def compare_nii_files(
    nii_path_1: Union[str, Path],
    nii_path_2: Union[str, Path],
    atol: float = 1e-6,
    rtol: float = 1e-5,
    strict_header: bool = False,
) -> bool:
    """Compare two NIfTI files.


    Args:
        nii_path_1: First .nii or .nii.gz path.
        nii_path_2: Second .nii or .nii.gz path.
        atol: Absolute tolerance for floating-point data and affine comparison.
        rtol: Relative tolerance for floating-point data and affine comparison.
        strict_header: If True, compare the full binary NIfTI header. If False,
            compare the main fields that affect image interpretation.


    Returns:
        True if the two images are equivalent under the selected checks.
    """
    path_1 = Path(nii_path_1)
    path_2 = Path(nii_path_2)
    img_1 = nib.load(str(path_1))
    img_2 = nib.load(str(path_2))


    same = True
    print(f"file 1: {path_1}")
    print(f"file 2: {path_2}")


    if img_1.shape != img_2.shape:
        same = False
        print(f"DIFF shape: {img_1.shape} vs {img_2.shape}")
    else:
        print(f"OK shape: {img_1.shape}")


    dtype_1 = np.dtype(img_1.header.get_data_dtype())
    dtype_2 = np.dtype(img_2.header.get_data_dtype())
    if dtype_1 != dtype_2:
        same = False
        print(f"DIFF storage dtype: {dtype_1} vs {dtype_2}")
    else:
        print(f"OK storage dtype: {dtype_1}")


    if not np.allclose(img_1.affine, img_2.affine, atol=atol, rtol=rtol):
        same = False
        max_diff = float(np.max(np.abs(img_1.affine - img_2.affine)))
        print(f"DIFF affine: max_abs_diff={max_diff:g}")
        print(f"affine 1:\n{img_1.affine}")
        print(f"affine 2:\n{img_2.affine}")
    else:
        print("OK affine")


    header_same = _compare_headers(img_1, img_2, strict_header=strict_header, atol=atol, rtol=rtol)
    same = same and header_same


    data_same = _compare_data(img_1, img_2, atol=atol, rtol=rtol)
    same = same and data_same


    print(f"RESULT: {'same' if same else 'different'}")
    return same



def _compare_headers(
    img_1: nib.spatialimages.SpatialImage,
    img_2: nib.spatialimages.SpatialImage,
    strict_header: bool,
    atol: float,
    rtol: float,
) -> bool:
    if strict_header:
        raw_1 = img_1.header.binaryblock
        raw_2 = img_2.header.binaryblock
        if raw_1 == raw_2:
            print("OK full header binaryblock")
            return True
        print("DIFF full header binaryblock")
        return False


    fields_1 = _important_header_fields(img_1)
    fields_2 = _important_header_fields(img_2)
    same = True
    for key in fields_1:
        value_1 = fields_1[key]
        value_2 = fields_2[key]
        if isinstance(value_1, np.ndarray):
            equal = np.allclose(value_1, value_2, atol=atol, rtol=rtol)
        else:
            equal = value_1 == value_2
        if not equal:
            same = False
            print(f"DIFF header {key}: {value_1} vs {value_2}")


    if same:
        print("OK important header fields")
    return same



def _important_header_fields(img: nib.spatialimages.SpatialImage) -> Dict[str, object]:
    header = img.header
    return {
        "zooms": np.asarray(header.get_zooms()[: len(img.shape)], dtype=np.float64),
        "xyzt_units": tuple(header.get_xyzt_units()),
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
        "datatype": int(header["datatype"]),
        "bitpix": int(header["bitpix"]),
        "scl_slope": _finite_or_nan(header["scl_slope"]),
        "scl_inter": _finite_or_nan(header["scl_inter"]),
        "qform": img.get_qform(),
        "sform": img.get_sform(),
    }



def _finite_or_nan(value: object) -> float:
    result = float(np.asarray(value).item())
    return result if np.isfinite(result) else float("nan")



def _compare_data(
    img_1: nib.spatialimages.SpatialImage,
    img_2: nib.spatialimages.SpatialImage,
    atol: float,
    rtol: float,
) -> bool:
    if img_1.shape != img_2.shape:
        print("SKIP data comparison because shapes differ")
        return False


    data_1 = img_1.get_fdata(dtype=np.float64)
    data_2 = img_2.get_fdata(dtype=np.float64)


    finite_1 = np.isfinite(data_1)
    finite_2 = np.isfinite(data_2)
    if not np.array_equal(finite_1, finite_2):
        print("DIFF data finite/NaN/inf mask")
        return False


    if np.allclose(data_1, data_2, atol=atol, rtol=rtol, equal_nan=True):
        print("OK data values")
        return True


    diff = np.abs(data_1 - data_2)
    max_index = np.unravel_index(int(np.nanargmax(diff)), diff.shape)
    max_diff = float(diff[max_index])
    mean_diff = float(np.nanmean(diff))
    print(
        f"DIFF data values: max_abs_diff={max_diff:g} at index={max_index}, "
        f"mean_abs_diff={mean_diff:g}, value1={data_1[max_index]:g}, value2={data_2[max_index]:g}"
    )
    return False



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two NIfTI files.")
    parser.add_argument("nii_path_1", help="First .nii or .nii.gz file")
    parser.add_argument("nii_path_2", help="Second .nii or .nii.gz file")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance")
    parser.add_argument(
        "--strict-header",
        action="store_true",
        help="Compare full binary NIfTI header instead of important interpretation fields",
    )
    args = parser.parse_args()


    same = compare_nii_files(
        args.nii_path_1,
        args.nii_path_2,
        atol=args.atol,
        rtol=args.rtol,
        strict_header=args.strict_header,
    )
    raise SystemExit(0 if same else 1)



if __name__ == "__main__":
    main()