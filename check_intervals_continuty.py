import json
from pathlib import Path
from typing import Any


def check_json_intervals(
    json_path: str,
    n: int,
    output_txt_path: str,
) -> None:
    """
    检查 JSON 中所有 key >= n 的条目。

    对每个符合条件的 key，检查：

    1. starters 的第一个元素 starters[0] 是否为 0；
    2. ends 中任意相邻元素之差是否为 8，即：
           ends[k + 1] - ends[k] == 8

    所有异常信息写入 output_txt_path。

    参数
    ----------
    json_path:
        输入 JSON 文件路径。

    n:
        只检查数值 key >= n 的条目。

    output_txt_path:
        检查结果 TXT 文件路径。
    """
    json_file = Path(json_path)
    output_file = Path(output_txt_path)

    if not json_file.is_file():
        raise FileNotFoundError(f"JSON 文件不存在：{json_file}")

    if not isinstance(n, int):
        raise TypeError(f"n 必须是整数，当前类型为：{type(n).__name__}")

    try:
        with json_file.open("r", encoding="utf-8") as file:
            data: Any = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 文件格式无效：{json_file}\n"
            f"错误位置：第 {exc.lineno} 行，第 {exc.colno} 列"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"JSON 顶层必须是字典对象，但得到：{type(data).__name__}"
        )

    # 保存：
    # (数值 key, 原始字符串 key, 该 key 对应的数据)
    entries_to_check: list[tuple[int, str, Any]] = []

    for raw_key, value in data.items():
        try:
            numeric_key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "JSON 顶层 key 必须能够转换为整数。\n"
                f"无法转换的 key：{raw_key!r}"
            ) from exc

        if numeric_key >= n:
            entries_to_check.append(
                (numeric_key, str(raw_key), value)
            )

    # 按 key 的数值大小排序，而不是按字符串排序。
    entries_to_check.sort(key=lambda item: item[0])

    records: list[str] = []
    checked_key_count = 0

    for numeric_key, raw_key, entry in entries_to_check:
        checked_key_count += 1

        if not isinstance(entry, dict):
            records.append(
                f"Key {raw_key}: 对应值不是字典，"
                f"实际类型为 {type(entry).__name__}"
            )
            continue

        # -----------------------------------------------------
        # 检查 starters
        # -----------------------------------------------------
        if "starters" not in entry:
            records.append(
                f"Key {raw_key}: 缺少 starters 字段"
            )
        else:
            starters = entry["starters"]

            if not isinstance(starters, list):
                records.append(
                    f"Key {raw_key}: starters 不是列表，"
                    f"实际类型为 {type(starters).__name__}"
                )
            elif len(starters) == 0:
                records.append(
                    f"Key {raw_key}: starters 是空列表，无法检查 starters[0]"
                )
            elif starters[0] != 0:
                records.append(
                    f"Key {raw_key}: starters[0] != 0，"
                    f"实际值为 {starters[0]!r}"
                )

        # -----------------------------------------------------
        # 检查 ends
        # -----------------------------------------------------
        if "ends" not in entry:
            records.append(
                f"Key {raw_key}: 缺少 ends 字段"
            )
        else:
            ends = entry["ends"]

            if not isinstance(ends, list):
                records.append(
                    f"Key {raw_key}: ends 不是列表，"
                    f"实际类型为 {type(ends).__name__}"
                )
            else:
                for k in range(len(ends) - 1):
                    current_value = ends[k]
                    next_value = ends[k + 1]

                    try:
                        difference = next_value - current_value
                    except TypeError:
                        records.append(
                            f"Key {raw_key}: 无法计算 "
                            f"ends[{k + 1}] - ends[{k}]，"
                            f"对应值为 {next_value!r} 和 {current_value!r}"
                        )
                        continue

                    if difference != 8:
                        records.append(
                            f"Key {raw_key}: "
                            f"ends[{k + 1}] - ends[{k}] != 8，"
                            f"实际为 {next_value!r} - "
                            f"{current_value!r} = {difference!r}"
                        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        file.write("JSON interval check results\n")
        file.write("=" * 80 + "\n")
        file.write(f"JSON file       : {json_file}\n")
        file.write(f"Minimum key     : {n}\n")
        file.write(f"Checked key count: {checked_key_count}\n")
        file.write(f"Issue count     : {len(records)}\n")
        file.write("\n")

        if records:
            for index, record in enumerate(records, start=1):
                file.write(f"{index}. {record}\n")
        else:
            file.write("未发现异常。\n")

    print(f"检查完成：共检查 {checked_key_count} 个 key。")
    print(f"发现异常：{len(records)} 条。")
    print(f"结果已保存到：{output_file}")


if __name__ == "__main__":
    check_json_intervals(
    json_path="enhanced_orinal_intv_info.json",
    n=501,
    output_txt_path="check_test_intervals_continuty_results.txt",
)