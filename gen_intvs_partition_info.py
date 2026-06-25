#!/usr/bin/env python3
"""
根据输入 JSON 中每个 n 的帧索引，采样得到 starters，并构建 ends(= start + a，且不超过
该 n.nii.gz 的 z 轴长度)，汇总写出 JSON。

输入:
  in_json  : JSON 文件，结构 {"n": {"k": k, ...}}，外层 key n 对应 n.nii.gz
  nii_dir  : 目录，内含 <n>.nii.gz (n 为整数)
  a        : 整数，ends 元素 = 对应 starter + a (上限为 z 长度)
  out_json : 输出 JSON 路径

每个 n 的处理:
  1. 收集内层 value 为升序 list (如 [147,148,149,150])。
  2. starters = random_sample_with_min_gap(list)   # 由使用者提供实现
  3. 读取 n.nii.gz 的 z 长度 z = shape[2]。
  4. ends = [min(start + a, z) for start in starters]   # [start, end) 左闭右开
  5. 记录 {"starters": starters, "ends": ends}

缺失/多余处理:
  - JSON 有 n 但目录无 n.nii.gz(读不到 z) -> 打印信息并跳过，不终止。
  - 目录有 n.nii.gz 但 JSON 无该 n -> 终止并打印(不应出现)。

输出格式:
  { "0": {"starters": [18,33,190], "ends": [21,36,192]}, ... }   外层 key 按数值升序

依赖:
  pip install nibabel numpy
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np


NII_SUFFIX = ".nii.gz"


# ===========================================================================
# 由使用者提供实现：替换下面的函数体即可。
# 约定：输入一个升序整数 list，返回一个整数 list(starters)。
# ===========================================================================
def random_sample_with_min_gap(points, n, b):
    if n <= 0:
        return []
    candidates = list(points)
    selected = []
    for p in candidates:
        if all(abs(p - q) > b for q in selected):
            selected.append(p)
            if len(selected) >= n:
                break
    return selected
# ===========================================================================


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


def _z_length(path: Path) -> int:
    """读取 NIfTI 的 z 轴长度(第 3 维)。"""
    img = nib.load(str(path))
    shape = img.shape
    if len(shape) < 3:
        _fatal(f"NIfTI 必须至少为 3D: {path}  shape={shape}")
    return int(shape[2])


def build_starters_ends(in_json, nii_dir, a: int, out_json) -> dict:
    """
    读取 in_json，对每个 n 采样 starters 并构建 ends(clamp 到 z 长度)，写出 out_json。
    """
    in_path = Path(in_json)
    nii_root = Path(nii_dir)

    if not in_path.is_file():
        _fatal(f"输入 JSON 不存在: {in_path}")
    if not nii_root.is_dir():
        _fatal(f"目录不存在或不是目录: {nii_root}")

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        _fatal(f"无法解析输入 JSON: {in_path}\n  原因: {exc}")

    # JSON 中的 n(整数集合)
    json_ns = set()
    for key in data.keys():
        try:
            json_ns.add(int(key))
        except (ValueError, TypeError):
            _fatal(f"输入 JSON 的外层 key 不是整数: {key!r}")

    # 目录中整数命名文件 -> 必须都在 JSON 中，否则终止
    for name in sorted(os.listdir(nii_root)):
        if not (nii_root / name).is_file():
            continue
        n = _int_stem(name)
        if n is None:
            continue
        if n not in json_ns:
            _fatal(
                f"目录中存在 JSON 未涵盖的文件(不应出现)。\n"
                f"  文件: {nii_root / name}\n"
                f"  其 n = {n} 不在输入 JSON 的 key 中。"
            )

    result = {}
    skipped = 0
    for n in sorted(json_ns):
        inner = data[str(n)]

        # 1) 收集内层 value 为升序 list
        values = sorted(int(v) for v in inner.values())

        nii_path = nii_root / f"{n}.nii.gz"
        if not nii_path.is_file():
            # JSON 有 n 但目录无文件 -> 打印并跳过
            print(f"  ⚠️ 跳过 n={n}: 目录下不存在 {nii_path.name}(读不到 z 长度)")
            skipped += 1
            continue

        z = _z_length(nii_path)

        # 2) 采样 starters(实现由使用者提供)
        starters = list(random_sample_with_min_gap(values, n=10000, b=7))

        # 4) ends = start + a，clamp 到 z(左闭右开 [start, end))
        ends = [min(int(s) + a, z) for s in starters]

        result[str(n)] = {
            "starters": [int(s) for s in starters],
            "ends": ends,
        }
        print(f"  ✅ n={n}: z={z}  starters={result[str(n)]['starters']}  ends={ends}")

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 完成：写出 {len(result)} 个条目，跳过 {skipped} 个，结果 -> {out_path}")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("用法: python build_starters_ends.py <in_json> <nii_dir> <a> <out_json>")
        sys.exit(1)
    build_starters_ends(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])