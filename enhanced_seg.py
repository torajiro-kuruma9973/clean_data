#!/usr/bin/env python3
"""
两目录同名配对，在 seg 的全黑帧上依据 pet 同帧高强度像素播种前景点(值=2)，
结果以同名写到指定输出目录。

输入:
  pet_dir : 目录，内含 <n>.nii.gz (n 为整数) 的 PET 文件
  seg_dir : 目录，内含 <n>.nii.gz 的分割文件
  out_dir : 输出目录，处理后的 seg 以原名 <n>.nii.gz 写入这里
  a       : 阈值，pet 帧中被选点的像素值必须 >= a

配对:
  两目录中“文件名完全相同”的一对 <n>.nii.gz 为一组；以 seg 为驱动遍历。
  (按约定必然成对存在；若缺失或同组 shape 不一致 -> 打印详情并终止)

逐帧处理(沿第 3 维的每个切片 k):
  - seg 第 k 帧不是全黑(含非 0 点) -> 跳过，保持原样。
  - seg 第 k 帧全黑 -> 在 pet 第 k 帧取像素值最大、且 >= a 的至多 4 个点：
      * 取到的 numpy 坐标直接用于 seg 同帧同坐标，置为 2(无任何坐标转换)。
      * 边界并列时随机选取以补足名额。
      * 合法点不足 4 个但 >0 -> 只标这几个。
      * 合法点为 0(含 pet 该帧无 >= a 的值) -> 该帧保持全黑，处理下一帧。

输出 seg 继承原 seg 的 affine/header，以 uint8 保存。

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


NII_SUFFIX = ".nii.gz"
FOREGROUND_VALUE = 2
TOP_K = 4


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


def _load_3d(path: Path) -> "tuple[nib.Nifti1Image, np.ndarray]":
    """读取 3D NIfTI，返回 (img, float32 数据)。"""
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        _fatal(f"期望 3D 数据: {path}  shape={data.shape}")
    return img, data


def _seed_one_pair(name: str, pet_path: Path, seg_path: Path, out_path: Path, threshold: float):
    pet_img, pet_data = _load_3d(pet_path)
    seg_img, seg_data = _load_3d(seg_path)

    # shape 必须一致
    if pet_data.shape != seg_data.shape:
        _fatal(
            f"同组 PET 与 SEG 的 shape 不一致。\n"
            f"  文件: {name}\n"
            f"  PET: {pet_path}  shape={pet_data.shape}\n"
            f"  SEG: {seg_path}  shape={seg_data.shape}"
        )

    out_data = seg_data.copy()
    z_count = seg_data.shape[2]

    seeded_frames = 0
    seeded_points = 0
    skipped_blank = 0   # 全黑但 pet 无合格点
    kept_fg = 0         # 本身有前景、未改动

    for k in range(z_count):
        seg_frame = seg_data[:, :, k]

        # 仅处理全黑帧
        if np.any(seg_frame != 0):
            kept_fg += 1
            continue

        pet_frame = pet_data[:, :, k]
        flat = pet_frame.ravel()

        # 取 >= 阈值 a 的候选点
        qualifying = np.flatnonzero(flat >= threshold)
        if qualifying.size == 0:
            skipped_blank += 1
            continue

        vals = flat[qualifying]

        # 随机打乱以对边界并列值做随机取舍，再按值稳定降序排序取前 TOP_K
        perm = np.random.permutation(qualifying.size)
        qualifying = qualifying[perm]
        vals = vals[perm]
        order = np.argsort(-vals, kind="stable")
        chosen_flat = qualifying[order[:TOP_K]]

        # flat 索引 -> 2D (i, j)；直接用于 seg 同帧同坐标，无任何转换
        iis, jjs = np.unravel_index(chosen_flat, pet_frame.shape)
        out_data[iis, jjs, k] = FOREGROUND_VALUE

        seeded_frames += 1
        seeded_points += chosen_flat.size

    # 输出：继承原 seg 的 affine/header，存 uint8
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_arr = out_data.astype(np.uint8)
    header = seg_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(None, None)
    out_img = nib.Nifti1Image(out_arr, seg_img.affine, header=header)
    sform_code = int(seg_img.header["sform_code"]) or 1
    qform_code = int(seg_img.header["qform_code"]) or 1
    out_img.set_sform(seg_img.affine, code=sform_code)
    out_img.set_qform(seg_img.affine, code=qform_code)
    nib.save(out_img, str(out_path))

    print(
        f"  ✅ {name}: 总帧={z_count}  已有前景帧={kept_fg}  "
        f"播种帧={seeded_frames}(共{seeded_points}点)  无合格点跳过={skipped_blank}"
    )


def seed_seg_from_pet(pet_dir, seg_dir, out_dir, threshold: float, seed: int = None) -> None:
    """
    两目录同名配对，对每组在 seg 全黑帧上依据 pet 同帧高强度像素播种前景点(值=2)，
    结果以原名写入 out_dir。

    threshold: 阈值 a，pet 帧中被选点的像素值必须 >= a。
    seed:      可选随机种子，固定后边界并列点的随机选取可复现。

    以 seg 为驱动；同名缺失或同组 shape 不一致 -> 打印详情并终止。
    """
    if seed is not None:
        np.random.seed(seed)

    pet_root = Path(pet_dir)
    seg_root = Path(seg_dir)
    out_root = Path(out_dir)

    if not pet_root.is_dir():
        _fatal(f"PET 目录不存在或不是目录: {pet_root}")
    if not seg_root.is_dir():
        _fatal(f"SEG 目录不存在或不是目录: {seg_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # seg 中整数命名文件，按 n 排序
    items = []  # (n, name)
    for name in sorted(os.listdir(seg_root)):
        n = _int_stem(name)
        if n is not None and (seg_root / name).is_file():
            items.append((n, name))
    items.sort(key=lambda t: t[0])

    if not items:
        _fatal(f"SEG 目录中未找到任何 <整数>.nii.gz 文件: {seg_root}")

    processed = 0
    for n, name in items:
        seg_path = seg_root / name
        pet_path = pet_root / name  # 完全同名配对

        if not pet_path.is_file():
            _fatal(
                f"PET 目录缺少与 SEG 同名的文件。\n"
                f"  组号 n = {n}\n"
                f"  SEG: {seg_path}\n"
                f"  期望 PET: {pet_path}"
            )

        out_path = out_root / name  # 名字不变
        _seed_one_pair(name, pet_path, seg_path, out_path, threshold)
        processed += 1

    print(f"\n🎉 完成，共处理 {processed} 组，输出到 {out_root}")


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        print("用法: python seed_seg_from_pet.py <pet_dir> <seg_dir> <out_dir> <threshold_a> [random_seed]")
        sys.exit(1)
    pet_d, seg_d, out_d = sys.argv[1], sys.argv[2], sys.argv[3]
    a = float(sys.argv[4]) # a = 0.1, seed = 42
    rseed = int(sys.argv[5]) if len(sys.argv) == 6 else None
    seed_seg_from_pet(pet_d, seg_d, out_d, threshold=a, seed=rseed)