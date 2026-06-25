#!/usr/bin/env python3
"""
遍历目录下所有 seg 的 .nii.gz 文件，把所有非 0 前景点统一置为 1，原地修改。

输入:
  seg_dir : 目录，内含若干 seg .nii.gz 文件

行为:
  - 对每个文件，data != 0 的体素一律置 1，其余保持 0。
  - 原地覆盖保存，affine/header 沿用原文件，dtype 存为 uint8。

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _fatal(msg: str):
    """打印详细信息并终止程序。"""
    print("\n" + "=" * 80, file=sys.stderr)
    print("❌ 致命错误，程序终止：", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    raise RuntimeError(msg)


def binarize_segs_inplace(seg_dir) -> None:
    """
    遍历 seg_dir 下所有 .nii.gz，把非 0 前景统一置 1，原地保存。
    """
    root = Path(seg_dir)
    if not root.is_dir():
        _fatal(f"目录不存在或不是目录: {root}")

    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.name.endswith(".nii.gz")
    )
    if not files:
        _fatal(f"目录中未找到任何 .nii.gz 文件: {root}")

    processed = 0
    for path in files:
        img = nib.load(str(path))
        data = np.asarray(img.get_fdata(dtype=np.float32))

        binarized = (data != 0).astype(np.uint8)

        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        header.set_slope_inter(None, None)
        out_img = nib.Nifti1Image(binarized, img.affine, header=header)
        sform_code = int(img.header["sform_code"]) or 1
        qform_code = int(img.header["qform_code"]) or 1
        out_img.set_sform(img.affine, code=sform_code)
        out_img.set_qform(img.affine, code=qform_code)
        nib.save(out_img, str(path))  # 原地覆盖

        processed += 1
        print(f"  ✅ {path.name}: 前景体素数={int(binarized.sum())}")

    print(f"\n🎉 完成，共处理 {processed} 个文件。")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python binarize_segs.py <seg_dir>")
        sys.exit(1)
    binarize_segs_inplace(sys.argv[1])