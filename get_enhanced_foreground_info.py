#!/usr/bin/env python3
"""
遍历目录下所有 seg 的 <n>.nii.gz 文件，沿第 3 维逐帧(0-based)记录“非全黑”的帧索引，
汇总写出一个 JSON 文件。

输入:
  seg_dir  : 目录，内含 <n>.nii.gz (n 为整数) 的 seg 文件
  out_json : 输出 JSON 路径

输出格式(文件 key 与帧 key 均按数值从小到大):
  {
    "0":  {"147": 147, "148": 148, ...},
    "1":  {"37": 37, "38": 38, ...},
    ...
  }
  外层 key 为文件名去后缀(整数 n 的字符串)，内层 {"帧索引": 帧索引}(key=字符串, value=整数)。

规则:
  - 帧维度为第 3 维 data[:, :, k]；非全黑 = 该帧含任意非 0 像素。
  - 文件名必须为 <整数>.nii.gz (assert)；否则终止。
  - 某文件所有帧均全黑 -> 终止并打印详情。

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

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


def _int_stem(name: str) -> Optional[int]:
    """若 name 形如 '<整数>.nii.gz' 则返回该整数，否则 None。"""
    if not name.endswith(NII_SUFFIX):
        return None
    stem = name[: -len(NII_SUFFIX)]
    try:
        return int(stem)
    except ValueError:
        return None


def _load_3d(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        _fatal(f"期望 3D 数据: {path}  shape={data.shape}")
    return data


def record_nonblank_frames(seg_dir, out_json) -> dict:
    """
    遍历 seg_dir 下所有 <n>.nii.gz，记录每个文件中非全黑的帧索引(0-based)，
    汇总写出 out_json，并返回该字典。
    """
    root = Path(seg_dir)
    if not root.is_dir():
        _fatal(f"目录不存在或不是目录: {root}")

    # 收集并 assert 文件名为整数
    items = []  # (n, name)
    for name in sorted(os.listdir(root)):
        full = root / name
        if not (full.is_file() and name.endswith(NII_SUFFIX)):
            continue
        n = _int_stem(name)
        if n is None:
            _fatal(f"文件名不是 <整数>.nii.gz，违反约定: {full}")
        items.append((n, name))

    if not items:
        _fatal(f"目录中未找到任何 <整数>.nii.gz 文件: {root}")

    items.sort(key=lambda t: t[0])  # 文件 key 按数值升序

    result = {}
    for n, name in items:
        data = _load_3d(root / name)
        z_count = data.shape[2]

        frame_map = {}
        for k in range(z_count):
            if np.any(data[:, :, k] != 0):
                frame_map[str(k)] = k  # 帧 key=字符串, value=整数, 自然升序

        if not frame_map:
            _fatal(
                f"该文件所有帧均为全黑(无任何非 0 像素)。\n"
                f"  文件: {root / name}\n"
                f"  shape: {data.shape}"
            )

        result[str(n)] = frame_map
        print(f"  ✅ {name}: 非全黑帧数={len(frame_map)} / 总帧={z_count}")

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成，共 {len(result)} 个文件，结果写入 {out_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python record_nonblank_frames.py <seg_dir> <out_json>")
        sys.exit(1)
    record_nonblank_frames(sys.argv[1], sys.argv[2])