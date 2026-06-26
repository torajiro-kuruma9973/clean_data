#!/usr/bin/env python3
"""
Create left-right and up-down flip augmentations for paired CT/PET/SEG NIfTI files.

Expected folders:
  data/
    n.nii.gz
  label/
    n.nii.gz
  seg/
    n.nii.gz

For every original n.nii.gz found in data/:
  - label/n.nii.gz and seg/n.nii.gz must also exist.
  - All three files must have the same 3D shape.
  - Left-right flip is saved as (1000 + n).nii.gz in all three folders.
  - Up-down flip is saved as (3000 + n).nii.gz in all three folders.

Important:
  - The original file list is collected once before any output is written.
  - By default, only n < 1000 is treated as original data, so rerunning this
    script will not augment already augmented files.
  - Affine/header are kept unchanged because these files are intended for
    model training, not physical-space medical use.
  - Each output file keeps the dtype of its own source file.

Dependencies:
  pip install nibabel numpy
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np


@dataclass
class FlipAugmentRecord:
    source_index: int
    source_name: str
    left_right_name: str
    up_down_name: str
    status: str
    shape: Tuple[int, int, int]


def augment_nii_flips(
    data_dir: Union[str, Path],
    label_dir: Union[str, Path],
    seg_dir: Union[str, Path],
    left_right_offset: int = 20000,
    up_down_offset: int = 30000,
    original_max_index: int = 5000,
    left_right_axis: int = 1,
    up_down_axis: int = 0,
    overwrite: bool = False,
) -> List[FlipAugmentRecord]:
    """Create synchronized left-right and up-down flips for n.nii.gz triples.

    Args:
        data_dir: CT NIfTI folder.
        label_dir: PET NIfTI folder.
        seg_dir: SEG NIfTI folder.
        left_right_offset: New index offset for left-right flips. Default 1000.
        up_down_offset: New index offset for up-down flips. Default 3000.
        original_max_index: Only files with n < this value are processed.
        left_right_axis: Pixel axis for left-right flip. Default 1 for (H,W,Z).
        up_down_axis: Pixel axis for up-down flip. Default 0 for (H,W,Z).
        overwrite: If True, replace existing augmented files.

    Returns:
        A list of augmentation records.
    """
    data_path = Path(data_dir)
    label_path = Path(label_dir)
    seg_path = Path(seg_dir)

    for directory, name in ((data_path, "data_dir"), (label_path, "label_dir"), (seg_path, "seg_dir")):
        if not directory.is_dir():
            raise NotADirectoryError(f"{name} is not a directory: {directory}")

    if left_right_offset <= 0 or up_down_offset <= 0:
        raise ValueError("offsets must be positive")
    if left_right_offset == up_down_offset:
        raise ValueError("left_right_offset and up_down_offset must be different")
    if original_max_index <= 0:
        raise ValueError("original_max_index must be positive")

    original_items = _collect_original_items(data_path, original_max_index)
    records: List[FlipAugmentRecord] = []

    for count, (index, filename) in enumerate(original_items, start=1):
        data_file = data_path / filename
        label_file = label_path / filename
        seg_file = seg_path / filename

        lr_name = f"{index + left_right_offset}.nii.gz"
        ud_name = f"{index + up_down_offset}.nii.gz"
        lr_targets = {
            "data": data_path / lr_name,
            "label": label_path / lr_name,
            "seg": seg_path / lr_name,
        }
        ud_targets = {
            "data": data_path / ud_name,
            "label": label_path / ud_name,
            "seg": seg_path / ud_name,
        }

        missing = [str(path) for path in (label_file, seg_file) if not path.is_file()]
        if missing:
            record = FlipAugmentRecord(index, filename, lr_name, ud_name, "skipped_missing_pair", (0, 0, 0))
            records.append(record)
            print(f"[{count}/{len(original_items)}] skipped {filename}: missing {missing}")
            continue

        target_conflicts = [
            str(path)
            for path in list(lr_targets.values()) + list(ud_targets.values())
            if path.exists()
        ]
        if target_conflicts and not overwrite:
            record = FlipAugmentRecord(index, filename, lr_name, ud_name, "skipped_target_exists", (0, 0, 0))
            records.append(record)
            print(f"[{count}/{len(original_items)}] skipped {filename}: target exists; use --overwrite")
            continue

        try:
            data_img = nib.load(str(data_file))
            label_img = nib.load(str(label_file))
            seg_img = nib.load(str(seg_file))

            shape = _require_same_3d_shape(
                {
                    "data": (data_img, data_file),
                    "label": (label_img, label_file),
                    "seg": (seg_img, seg_file),
                }
            )
            _validate_axis(left_right_axis, shape, "left_right_axis")
            _validate_axis(up_down_axis, shape, "up_down_axis")

            if overwrite:
                for target in list(lr_targets.values()) + list(ud_targets.values()):
                    if target.exists():
                        target.unlink()

            _save_flipped_pair(data_img, lr_targets["data"], axis=left_right_axis)
            _save_flipped_pair(label_img, lr_targets["label"], axis=left_right_axis)
            _save_flipped_pair(seg_img, lr_targets["seg"], axis=left_right_axis)

            _save_flipped_pair(data_img, ud_targets["data"], axis=up_down_axis)
            _save_flipped_pair(label_img, ud_targets["label"], axis=up_down_axis)
            _save_flipped_pair(seg_img, ud_targets["seg"], axis=up_down_axis)

            record = FlipAugmentRecord(index, filename, lr_name, ud_name, "written", shape)
            records.append(record)
            print(
                f"[{count}/{len(original_items)}] written {filename}: "
                f"LR={lr_name}, UD={ud_name}, shape={shape}"
            )
        except Exception as exc:
            record = FlipAugmentRecord(index, filename, lr_name, ud_name, f"failed: {exc}", (0, 0, 0))
            records.append(record)
            print(f"[{count}/{len(original_items)}] failed {filename}: {exc}")

    written = sum(1 for record in records if record.status == "written")
    skipped = sum(1 for record in records if record.status.startswith("skipped"))
    failed = sum(1 for record in records if record.status.startswith("failed"))
    print(f"Original files checked: {len(original_items)}; written: {written}; skipped: {skipped}; failed: {failed}")
    return records


def _collect_original_items(data_dir: Path, original_max_index: int) -> List[Tuple[int, str]]:
    """Collect original n.nii.gz names once before writing augmented files."""
    items: List[Tuple[int, str]] = []
    for path in data_dir.iterdir():
        if not path.is_file():
            continue
        index = _parse_index_name(path.name)
        if index is None:
            continue
        if index >= original_max_index:
            continue
        items.append((index, path.name))
    return sorted(items, key=lambda item: item[0])


def _parse_index_name(filename: str) -> Optional[int]:
    match = re.fullmatch(r"(\d+)\.nii\.gz", filename)
    if match is None:
        return None
    return int(match.group(1))


def _require_same_3d_shape(
    images: Dict[str, Tuple[nib.spatialimages.SpatialImage, Path]],
) -> Tuple[int, int, int]:
    shapes: Dict[str, Tuple[int, int, int]] = {}
    for key, (img, path) in images.items():
        shape = img.shape
        if len(shape) != 3:
            raise ValueError(f"{key} must be exactly 3D, got shape={shape}: {path}")
        shapes[key] = (int(shape[0]), int(shape[1]), int(shape[2]))

    first_shape = next(iter(shapes.values()))
    mismatched = {key: shape for key, shape in shapes.items() if shape != first_shape}
    if mismatched:
        raise ValueError(f"shape mismatch: {shapes}")
    return first_shape


def _validate_axis(axis: int, shape: Tuple[int, int, int], name: str) -> None:
    if axis not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1 for in-plane flips, got: {axis}")
    if axis >= len(shape):
        raise ValueError(f"{name}={axis} is out of bounds for shape={shape}")


def _save_flipped_pair(
    img: nib.spatialimages.SpatialImage,
    output_path: Path,
    axis: int,
) -> None:
    original_dtype = np.dtype(img.header.get_data_dtype())
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"expected exactly 3D data, got shape={data.shape}")

    flipped = np.flip(data, axis=axis).astype(original_dtype, copy=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = img.header.copy()
    header.set_data_dtype(original_dtype)
    image = nib.Nifti1Image(flipped, img.affine, header=header)
    sform_code = int(img.header["sform_code"]) or 1
    qform_code = int(img.header["qform_code"]) or 1
    image.set_sform(img.affine, code=sform_code)
    image.set_qform(img.affine, code=qform_code)
    nib.save(image, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create synchronized left-right and up-down NIfTI flip augmentations."
    )
    parser.add_argument("data_dir", help="CT NIfTI folder")
    parser.add_argument("label_dir", help="PET NIfTI folder")
    parser.add_argument("seg_dir", help="SEG NIfTI folder")
    parser.add_argument("--left-right-offset", type=int, default=20000, help="LR filename offset (default: 1000)")
    parser.add_argument("--up-down-offset", type=int, default=30000, help="UD filename offset (default: 3000)")
    parser.add_argument(
        "--original-max-index",
        type=int,
        default=5000,
        help="Only n < this value is treated as original data (default: 1000)",
    )
    parser.add_argument("--left-right-axis", type=int, default=1, help="LR flip axis for (H,W,Z), default: 1")
    parser.add_argument("--up-down-axis", type=int, default=0, help="UD flip axis for (H,W,Z), default: 0")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing augmented files")
    args = parser.parse_args()

    augment_nii_flips(
        args.data_dir,
        args.label_dir,
        args.seg_dir,
        left_right_offset=args.left_right_offset,
        up_down_offset=args.up_down_offset,
        original_max_index=args.original_max_index,
        left_right_axis=args.left_right_axis,
        up_down_axis=args.up_down_axis,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()