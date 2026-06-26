#!/usr/bin/env python3
"""
对三个目录(CT / PET / 掩码)中同名 <n>.nii.gz 成组做面内(轴位 X-Y 平面)顺时针 45° 旋转增强，
按模态选择正确的插值方式与填充值，输出文件名为 <n+2000>.nii.gz，放回各自原目录。

输入:
  ct_dir   : CT，已 HU 截断 [-1000, 2000] 且归一化到 [0,1]
  pet_dir  : PET，已 SUV 截断 [0,50] + gamma + 归一化到 [0,1]
  mask_dir : 掩码，值只有 0/1

配对:
  三目录中同名 <n>.nii.gz 为一组(同 n 必三处都有)。只处理 n < 10000 的文件。

变换(三者共用同一角度/平面/reshape):
  scipy.ndimage.rotate, axes=(0,1)(逐 z 切片在面内旋转), angle=-45(顺时针), reshape=False。
  - CT  : order=1, cval=0.0   (归一化后空气 -1000 -> 0)
  - PET : order=1, cval=0.0   (SUV=0 归一化后 -> 0)
  - 掩码: order=0, cval=0      (最近邻保持 0/1，背景 0)

输出:
  <n+2000>.nii.gz，写回同一目录；affine/header 沿用原文件，dtype 沿用原文件。

约定/终止条件(打印详情并终止):
  - 三目录任一缺少同名 <n>.nii.gz。
  - 输出文件 <n+10000>.nii.gz 已存在(命名冲突)。

依赖:
  pip install nibabel numpy scipy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate


NII_SUFFIX = ".nii.gz"
ANGLE = -45        # scipy 正角度为逆时针；顺时针 45° 用 -45
NAME_OFFSET = 10000
PROCESS_MAX_N = NAME_OFFSET  # 只处理 n < 2000


def _fatal(msg: str):
    """打印详细信息并终止程序。"""
    print("\n" + "=" * 80, file=sys.stderr)
    print("❌ 致命错误，程序终止：", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    raise RuntimeError(msg)


def _int_stem(name: str) -> Optional[int]:
    """若 name 形如 '<整数>.nii.gz' 则返回该整数，否则 None。"""
    if not name.endswith(NII_SUFFIX):
        return None
    stem = name[: -len(NII_SUFFIX)]
    try:
        return int(stem)
    except ValueError:
        return None


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header"]:
    """读取 3D NIfTI，返回 (按存储 dtype 的数组, affine, header)。"""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        _fatal(f"期望 3D 数据: {path}  shape={arr.shape}")
    return arr, img.affine, img.header.copy()


def _save(arr: np.ndarray, affine: np.ndarray, header, out_path: Path):
    """以原 affine/header 保存；dtype 沿用 arr。"""
    hdr = header.copy()
    hdr.set_data_dtype(arr.dtype)
    hdr.set_slope_inter(None, None)
    out_img = nib.Nifti1Image(np.ascontiguousarray(arr), affine, header=hdr)
    nib.save(out_img, str(out_path))


def _rotate_one(path: Path, out_path: Path, order: int, cval: float):
    """对单个文件做面内 45° 旋转并保存。保持原 dtype。"""
    arr, affine, header = _load(path)
    orig_dtype = arr.dtype

    # 旋转在浮点上进行
    rotated = rotate(
        arr.astype(np.float32),
        angle=ANGLE,
        axes=(0, 1),       # 面内(X-Y)逐 z 切片旋转
        reshape=False,     # 输出 shape 与原图一致
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )

    # 恢复 dtype：掩码(order=0)保持整型 0/1；CT/PET 保持原浮点
    if np.issubdtype(orig_dtype, np.integer):
        out_arr = np.rint(rotated).astype(orig_dtype)
    else:
        out_arr = rotated.astype(orig_dtype)

    _save(out_arr, affine, header, out_path)


def augment_rotate45(ct_dir, pet_dir, mask_dir) -> None:
    """
    对三目录同名 <n>.nii.gz(n<2000) 成组做面内 45° 旋转增强，
    按模态选 order/cval，输出 <n+2000>.nii.gz 到各自原目录。
    """
    # (目录, order, cval)
    specs = [
        (Path(ct_dir),   1, 0.0),   # CT  线性, 空气归一化后=0
        (Path(pet_dir),  1, 0.0),   # PET 线性, SUV=0 归一化后=0
        (Path(mask_dir), 0, 0.0),   # 掩码 最近邻, 背景=0
    ]
    names = ["CT", "PET", "MASK"]

    for (d, _o, _c) in specs:
        if not d.is_dir():
            _fatal(f"目录不存在或不是目录: {d}")

    ct_root = specs[0][0]

    # 以 CT 目录为驱动，收集 n < 2000 的整数命名文件
    items = []  # (n, filename)
    for fname in sorted(os.listdir(ct_root)):
        n = _int_stem(fname)
        if n is None or not (ct_root / fname).is_file():
            continue
        if n >= PROCESS_MAX_N:
            continue   # 只处理 n < 2000(跳过旋转产物等)
        items.append((n, fname))
    items.sort(key=lambda t: t[0])

    if not items:
        _fatal(f"CT 目录中未找到 n<{PROCESS_MAX_N} 的 <整数>.nii.gz 文件: {ct_root}")

    processed = 0
    for n, fname in items:
        out_name = f"{n + NAME_OFFSET}.nii.gz"

        # 校验三目录同名输入都在、且三处输出都不冲突
        for (d, _o, _c), label in zip(specs, names):
            in_path = d / fname
            if not in_path.is_file():
                _fatal(
                    f"{label} 目录缺少同名文件。\n"
                    f"  期望: {in_path}\n  组 n = {n}"
                )
            out_path = d / out_name
            if out_path.exists():
                _fatal(
                    f"{label} 输出文件已存在(命名冲突)。\n"
                    f"  目标: {out_path}\n  组 n = {n}"
                )

        # 三模态各自旋转
        for (d, order, cval) in specs:
            _rotate_one(d / fname, d / out_name, order=order, cval=cval)

        processed += 1
        print(f"  ✅ n={n} -> {out_name}  (CT/PET order=1 cval=0, MASK order=0 cval=0)")

    print(f"\n🎉 完成：旋转增强 {processed} 组，输出文件名 = 原 n + {NAME_OFFSET}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python augment_rotate45.py <ct_dir> <pet_dir> <mask_dir>")
        sys.exit(1)
    augment_rotate45(sys.argv[1], sys.argv[2], sys.argv[3])