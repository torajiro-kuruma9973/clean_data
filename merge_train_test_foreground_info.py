#!/usr/bin/env python3
"""
合并两个 JSON 文件，外层 key 按数值升序排列，写出到指定路径。

输入:
  json1    : 第一个 JSON 文件
  json2    : 第二个 JSON 文件
  out_json : 输出 JSON 路径

约定:
  - 两个 JSON 的外层 key 不会重复(assert)；若有交集则打印详情并终止。
  - 外层 key 为整数字符串，输出按数值从小到大排序。

依赖: 仅标准库。
"""

import sys
import json
from pathlib import Path


def _fatal(msg: str):
    """打印详细信息并终止程序。"""
    print("\n" + "=" * 80, file=sys.stderr)
    print("❌ 致命错误，程序终止：", file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    raise RuntimeError(msg)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        _fatal(f"JSON 文件不存在: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        _fatal(f"无法解析 JSON: {path}\n  原因: {exc}")
    if not isinstance(data, dict):
        _fatal(f"JSON 顶层不是对象(dict): {path}")
    return data


def merge_two_jsons(json1, json2, out_json) -> dict:
    """
    合并 json1 与 json2，外层 key 按数值升序写出 out_json。
    两者 key 不应有交集，否则终止。
    """
    p1, p2 = Path(json1), Path(json2)
    data1 = _load_json(p1)
    data2 = _load_json(p2)

    # assert 无重复 key
    overlap = set(data1.keys()) & set(data2.keys())
    if overlap:
        _fatal(
            f"两个 JSON 存在相同的外层 key(不应出现)。\n"
            f"  重复 key: {sorted(overlap)}\n"
            f"  json1: {p1}\n  json2: {p2}"
        )

    combined = {**data1, **data2}

    # 外层 key 按数值升序
    try:
        ordered_keys = sorted(combined.keys(), key=lambda k: int(k))
    except (ValueError, TypeError):
        _fatal("外层 key 存在非整数，无法按数值排序。")

    result = {k: combined[k] for k in ordered_keys}

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"🎉 合并完成：json1 {len(data1)} 项 + json2 {len(data2)} 项 "
        f"= {len(result)} 项，结果 -> {out_path}"
    )
    return result


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python merge_two_jsons.py <json1> <json2> <out_json>")
        sys.exit(1)
    merge_two_jsons(sys.argv[1], sys.argv[2], sys.argv[3])