#!/usr/bin/env python3
"""
对三个目录的同名 <n>.nii.gz 分组缝合(沿 z 轴拼接)，输出到各自 <目录名>_baches，
并生成两个记录 JSON。

输入:
  dir1, dir2, dir3 : 三个目录(如 ct/pet/seg)，各含 <n>.nii.gz；同 n 为一组
  a                : 整数，每 a 个文件缝合成一个(末组允许不足 a 个)
  out_json_members : JSON1 路径，记录 {k: [n, ...]}(子文件的整数 n 值)
  out_json_ranges  : JSON2 路径，记录 {k: {"starters": [...], "ends": [...]}}

顺序基准:
  以 dir1 的文件名为准：先按 n 升序确定初始顺序，再 random.seed(42) 打乱，
  得到唯一的文件名顺序 list；三个目录的所有操作都严格遵从该顺序。

缝合:
  按 list 顺序每 a 个一组，全局 k(0-based) 作为输出文件名整数。
  每组内 a 个文件沿 z 轴(第 3 维)首尾拼接：np.concatenate(arrs, axis=2)。
  三个目录各自独立拼接，结果写入 <目录名>_baches/k.nii.gz。
  affine/header 沿用该目录该组“第一个文件”的。

约定/终止条件(打印详情并终止):
  - 每个 <目录名>_baches 文件夹必须已存在(assert)。
  - dir1 list 中的文件名在 dir2、dir3 必须都存在(assert)。
  - 同组内 a 个文件的前两维 (X, Y) 必须一致(assert，按每个目录各自检查)。
  - 三目录同名文件的 z 长度必须一致(assert)。
  - a <= 0。

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np


NII_SUFFIX = ".nii.gz"
SHUFFLE_SEED = 42


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


def _baches_dir(d: Path) -> Path:
    """<目录名>_baches 同级文件夹；必须已存在。"""
    bd = d.parent / (d.name + "_baches")
    if not bd.is_dir():
        _fatal(f"缝合输出文件夹不存在(需预先创建): {bd}")
    return bd


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header"]:
    """读取 3D NIfTI，返回 (按存储 dtype 的数组, affine, header)。"""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)   # 保持原始存储数值/类型
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        _fatal(f"期望 3D 数据: {path}  shape={arr.shape}")
    return arr, img.affine, img.header.copy()


def _save(arr: np.ndarray, affine: np.ndarray, header, out_path: Path):
    """以给定 affine/header 保存数组；dtype 沿用 arr。"""
    hdr = header.copy()
    hdr.set_data_dtype(arr.dtype)
    hdr.set_slope_inter(None, None)
    out_img = nib.Nifti1Image(np.ascontiguousarray(arr), affine, header=hdr)
    nib.save(out_img, str(out_path))


def stitch_groups(dir1, dir2, dir3, a: int,
                  out_json_members, out_json_ranges) -> Tuple[dict, dict]:
    """
    以 dir1 打乱后的文件名顺序为准，对三目录同名文件每 a 个沿 z 拼接，
    写出 <目录名>_baches/k.nii.gz，并返回/写出两个记录 JSON。
    """
    if a <= 0:
        _fatal(f"a 必须为正整数，当前 a={a}")

    dirs = [Path(dir1), Path(dir2), Path(dir3)]
    for d in dirs:
        if not d.is_dir():
            _fatal(f"目录不存在或不是目录: {d}")
    baches_dirs = [_baches_dir(d) for d in dirs]   # 同时 assert 存在

    # dir1 文件名：先按 n 升序，再 seed=42 打乱
    names = []
    for name in sorted(os.listdir(dirs[0])):
        if (dirs[0] / name).is_file() and _int_stem(name) is not None:
            names.append(name)
    if not names:
        _fatal(f"dir1 中未找到任何 <整数>.nii.gz 文件: {dirs[0]}")
    names.sort(key=lambda s: _int_stem(s))         # 确定性初始顺序
    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(names)                              # 唯一基准顺序

    print(f"打乱后的顺序(共 {len(names)} 个): {names}")

    # dir2、dir3 必须包含每个名字
    for name in names:
        for d in dirs[1:]:
            if not (d / name).is_file():
                _fatal(f"目录缺少与 dir1 同名的文件。\n  期望: {d / name}")

    members = {}   # JSON1: {k: [n, ...]}
    ranges = {}    # JSON2: {k: {"starters": [...], "ends": [...]}}
    k = 0

    # 按 a 分块
    for start_i in range(0, len(names), a):
        group = names[start_i:start_i + a]

        # 三目录同名文件 z 长度一致校验(逐文件)，用 dir1 的 z 记录 starters/ends
        z_per_file = []
        for name in group:
            zs = []
            for d in dirs:
                arr, _aff, _hdr = _load(d / name)
                zs.append(arr.shape[2])
            if not (zs[0] == zs[1] == zs[2]):
                _fatal(
                    f"三目录同名文件 z 长度不一致。\n"
                    f"  文件: {name}\n"
                    f"  z(dir1,dir2,dir3) = {zs}"
                )
            z_per_file.append(zs[0])

        # 每个目录各自拼接
        for d, bd in zip(dirs, baches_dirs):
            arrs = []
            base_affine = None
            base_header = None
            xy = None
            for j, name in enumerate(group):
                arr, aff, hdr = _load(d / name)
                if j == 0:
                    base_affine, base_header = aff, hdr   # 该组第一个文件
                    xy = arr.shape[:2]
                elif arr.shape[:2] != xy:
                    _fatal(
                        f"同组内前两维 (X,Y) 不一致，无法沿 z 拼接。\n"
                        f"  目录: {d}\n  组 k={k}\n"
                        f"  首文件 {group[0]} XY={xy}，{name} XY={arr.shape[:2]}"
                    )
                arrs.append(arr)
            stitched = np.concatenate(arrs, axis=2)
            _save(stitched, base_affine, base_header, bd / f"{k}.nii.gz")

        # 记录(基于 dir1 的 z)
        members[str(k)] = [int(_int_stem(name)) for name in group]
        starters, ends, cur = [], [], 0
        for z in z_per_file:
            starters.append(cur)
            cur += z
            ends.append(cur)
        ranges[str(k)] = {"starters": starters, "ends": ends}

        print(f"  ✅ k={k}: 子文件={members[str(k)]}  z={z_per_file}  总 z={ends[-1]}")
        k += 1

    # 写出两个 JSON
    for path_str, obj in ((out_json_members, members), (out_json_ranges, ranges)):
        op = Path(path_str)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    print(
        f"\n🎉 完成：{len(names)} 个文件 -> {k} 个缝合文件(每目录各 {k} 个)。\n"
        f"   成员记录 -> {out_json_members}\n   区间记录 -> {out_json_ranges}"
    )
    return members, ranges


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("用法: python stitch_groups.py <dir1> <dir2> <dir3> <a> <out_json_members> <out_json_ranges>")
        sys.exit(1)
    stitch_groups(
        sys.argv[1], sys.argv[2], sys.argv[3],
        int(sys.argv[4]), sys.argv[5], sys.argv[6],
    )