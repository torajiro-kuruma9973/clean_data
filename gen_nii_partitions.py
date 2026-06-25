#!/usr/bin/env python3
"""
按划分 JSON 把三个目录中同名 <n>.nii.gz 沿第 3 维切片，三目录同一区间共用同一个全局 k，
切片写入各自的 <目录名>_slices 文件夹并命名 k.nii.gz，同时输出 {n: [k,...]} 记录。

输入:
  dir1, dir2, dir3 : 三个目录(如 ct/pet/seg)，各含 <n>.nii.gz；同 n 为一组
  split_json       : 划分 JSON，结构 {"n": {"starters": [...], "ends": [...]}}
                     同 n 下 starters 与 ends 等长，同 idx 构成左闭右开区间 [start, end)
  out_json         : 输出记录 JSON 路径，内容 {n: [该 n 产生的所有 k]}

行为:
  - 维护全局 k=0；按 n 数值升序遍历。
  - 每个区间 [start, end) 对三个目录的同名文件分别切出 data[:, :, start:end]，
    三者用同一个 k，写入 <目录名>_slices/k.nii.gz；每处理完一个区间 k += 1。
  - 切片沿用原文件 affine/header(dtype 保持原样)。

约定/终止条件(打印详情并终止):
  - 每个 <目录名>_slices 文件夹必须已存在(assert)。
  - 同 n 的 starters 与 ends 必须等长。
  - 三目录中任一缺少 <n>.nii.gz。
  - 区间非法：需满足 0 <= start < end <= z(该文件第 3 维长度)。

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np


NII_SUFFIX = ".nii.gz"


def _fatal(msg: str):
    """打印详细信息并终止程序。"""
    print("\n" + "=" * 80, file=sys.stderr)
    print("❌ 致命错误，程序终止：", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    raise RuntimeError(msg)


def _slices_dir(d: Path) -> Path:
    """<目录名>_intvs 同级文件夹；必须已存在。"""
    sd = d.parent / (d.name + "_slices")
    if not sd.is_dir():
        _fatal(f"切片输出文件夹不存在(需预先创建): {sd}")
    return sd


def _load(path: Path) -> Tuple[np.ndarray, np.ndarray, "nib.Nifti1Header", int]:
    """读取 3D NIfTI，返回 (按存储 dtype 的数组, affine, header, z 长度)。"""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)   # 保持原始存储数值/类型
    while arr.ndim > 3:
        arr = arr[..., 0]
    if arr.ndim != 3:
        _fatal(f"期望 3D 数据: {path}  shape={arr.shape}")
    return arr, img.affine, img.header.copy(), int(arr.shape[2])


def _save_slice(sub: np.ndarray, affine: np.ndarray, header, out_path: Path):
    """以原 affine/header 保存切片子块；dtype 沿用 sub。"""
    hdr = header.copy()
    hdr.set_data_dtype(sub.dtype)
    hdr.set_slope_inter(None, None)
    out_img = nib.Nifti1Image(np.ascontiguousarray(sub), affine, header=hdr)
    nib.save(out_img, str(out_path))


def slice_groups_by_json(dir1, dir2, dir3, split_json, out_json) -> dict:
    """
    按 split_json 的区间把三个目录的同名文件切片，三目录同区间共用同一全局 k，
    切片写入各自 <目录名>_slices/k.nii.gz，返回并写出 {n: [k,...]} 记录到 out_json。
    """
    dirs = [Path(dir1), Path(dir2), Path(dir3)]
    for d in dirs:
        if not d.is_dir():
            _fatal(f"目录不存在或不是目录: {d}")

    slices_dirs = [_slices_dir(d) for d in dirs]   # 同时 assert 存在

    sp = Path(split_json)
    if not sp.is_file():
        _fatal(f"划分 JSON 不存在: {sp}")
    try:
        with open(sp, "r", encoding="utf-8") as f:
            split = json.load(f)
    except Exception as exc:
        _fatal(f"无法解析划分 JSON: {sp}\n  原因: {exc}")

    # n 按数值升序
    try:
        ns = sorted(split.keys(), key=lambda s: int(s))
    except (ValueError, TypeError):
        _fatal("划分 JSON 的外层 key 存在非整数。")

    record = {}
    k = 0

    for n in ns:
        entry = split[n]
        starters = entry.get("starters")
        ends = entry.get("ends")
        if starters is None or ends is None:
            _fatal(f"n={n} 缺少 starters 或 ends 字段。")
        if len(starters) != len(ends):
            _fatal(
                f"n={n} 的 starters 与 ends 长度不一致。\n"
                f"  len(starters)={len(starters)}, len(ends)={len(ends)}"
            )

        fname = f"{n}.nii.gz"

        # 加载三个同名文件(缺失即终止)
        loaded = []
        for d in dirs:
            p = d / fname
            if not p.is_file():
                #_fatal(f"目录缺少同名文件。\n  期望: {p}\n  组 n = {n}")
                print(f"目录缺少同名文件。\n  期望: {p}\n  组 n = {n}")
                continue
            loaded.append(_load(p))

        # 区间合法性校验：0 <= start < end <= z(对每个文件各自的 z)
        for start, end in zip(starters, ends):
            s, e = int(start), int(end)
            for (arr, _aff, _hdr, z), d in zip(loaded, dirs):
                if not (0 <= s < e <= z):
                    _fatal(
                        f"非法区间 [{s}, {e})，需满足 0 <= start < end <= z。\n"
                        f"  文件: {d / fname}  z={z}\n  组 n = {n}"
                    )

        # 逐区间切片，三目录共用同一个 k
        ks_for_n = []
        for start, end in zip(starters, ends):
            s, e = int(start), int(end)
            for (arr, aff, hdr, _z), sd in zip(loaded, slices_dirs):
                sub = arr[:, :, s:e]
                _save_slice(sub, aff, hdr, sd / f"{k}.nii.gz")
            ks_for_n.append(k)
            k += 1

        record[str(int(n))] = ks_for_n
        print(f"  ✅ n={n}: 区间数={len(starters)}  k={ks_for_n[0]}..{ks_for_n[-1]}")

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成：共 {len(record)} 组，生成 {k} 个切片(每目录各 {k} 个)，记录 -> {out_path}")
    return record


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("用法: python slice_groups_by_json.py <dir1> <dir2> <dir3> <split_json> <out_json>")
        sys.exit(1)
    slice_groups_by_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])