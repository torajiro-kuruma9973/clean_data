#!/usr/bin/env python3
"""
Export foreground frame indices from SEG NIfTI masks to JSON.

Input folder:
  segs/
    0.nii.gz
    1.nii.gz
    99.nii.gz

Output JSON:
  {
    "0": {
      "147": 147,
      "148": 148
    },
    "99": {
      "0": 0
    }
  }

Frame index is 0-based and follows the last NIfTI axis: mask[:, :, frame_idx].

Dependencies:
  pip install nibabel numpy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Union

import nibabel as nib
import numpy as np


def export_seg_foreground_frames(
    seg_dir: Union[str, Path],
    output_json: Union[str, Path],
    include_empty: bool = True,
) -> Dict[str, Dict[str, int]]:
    """Export 0-based frame indices that contain foreground voxels.

    Args:
        seg_dir: Folder containing SEG mask .nii.gz files.
        output_json: Path to the output JSON file.
        include_empty: If True, include files with no foreground as empty dicts.

    Returns:
        The JSON-serializable result dictionary.
    """
    seg_path = Path(seg_dir)
    if not seg_path.is_dir():
        raise NotADirectoryError(f"seg_dir is not a directory: {seg_path}")

    result: Dict[str, Dict[str, int]] = {}
    seg_files = sorted(path for path in seg_path.iterdir() if path.is_file() and path.name.endswith(".nii.gz"))

    for index, path in enumerate(seg_files, start=1):
        key = _nii_gz_stem(path)
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) != 3:
            raise ValueError(f"expected exactly 3D NIfTI, got shape={shape}: {path}")

        frames: Dict[str, int] = {}
        for frame_idx in range(shape[2]):
            frame = np.asanyarray(img.dataobj[:, :, frame_idx])
            if np.any(frame > 0):
                frames[str(frame_idx)] = frame_idx

        if frames or include_empty:
            result[key] = frames

        print(f"[{index}/{len(seg_files)}] {path.name}: foreground_frames={len(frames)}")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {output_path}")
    return result


def _nii_gz_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export foreground frame indices from 3D SEG .nii.gz masks."
    )
    parser.add_argument("seg_dir", help="Folder containing SEG .nii.gz files")
    parser.add_argument("output_json", help="Output JSON path")
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Do not include files that have no foreground frames",
    )
    args = parser.parse_args()

    export_seg_foreground_frames(
        args.seg_dir,
        args.output_json,
        include_empty=not args.skip_empty,
    )


if __name__ == "__main__":
    main()
