import os
import re
import json
import sys
import numpy as np
import nibabel as nib


def split_nii_files(input_dir: str, json_path: str, output_dir: str) -> None:
    """
    按照 json_path 指定的分段方式，把 input_dir 下所有 n.nii.gz 文件
    沿第三个维度切成若干段，切片保存到 output_dir。

    参数:
        input_dir:  存放 n.nii.gz 文件的文件夹（n 为整数）
        json_path:  记录切割方案的 json 文件路径，格式:
                    {
                      "0": {"starters": [0, 8, 16], "ends": [8, 16, 24]},
                      "1": {...},
                      ...
                    }
        output_dir: 切片结果存放的文件夹，必须已存在，否则终止程序

    规则（严格模式，任何异常都直接终止程序并打印错误信息）:
        1. output_dir 必须已存在，否则报错终止。
        2. 每个片段若超出原数据在第三维上的实际范围，报错终止。
        3. 某个 key 对应的 starters/ends 长度不一致，报错终止。
        4. input_dir 下文件名不是"整数.nii.gz"格式，报错终止。
        5. 输出文件已存在，报错终止（不覆盖）。
        6. 对每个区间 assert starters[i] < ends[i]。
    """

    # ---- 0. 校验 input_dir / output_dir ----
    if not os.path.isdir(input_dir):
        print(f"[错误] 输入文件夹不存在: {input_dir}")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"[错误] 输出文件夹不存在，请先创建: {output_dir}")
        sys.exit(1)

    # ---- 1. 读取 json ----
    if not os.path.isfile(json_path):
        print(f"[错误] json 文件不存在: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    # ---- 2. 遍历 input_dir 下的 nii.gz 文件 ----
    filename_pattern = re.compile(r"^(\d+)\.nii\.gz$")

    nii_files = sorted(
        fn for fn in os.listdir(input_dir)
        if fn.endswith(".nii.gz")
    )

    if not nii_files:
        print(f"[错误] 输入文件夹下没有找到任何 .nii.gz 文件: {input_dir}")
        sys.exit(1)

    for filename in nii_files:
        match = filename_pattern.match(filename)
        if not match:
            print(f"[错误] 文件名不符合 'n.nii.gz'（n为整数）格式: {filename}")
            sys.exit(1)

        n_str = match.group(1)  # 保留字符串形式，用于查 json key
        n = int(n_str)

        # ---- 3. 在 json 中查找对应 key ----
        if n_str not in split_config:
            print(f"[错误] 文件 {filename} 对应的 key '{n_str}' 在 json 中找不到")
            sys.exit(1)

        seg = split_config[n_str]

        if "starters" not in seg or "ends" not in seg:
            print(f"[错误] json 中 key '{n_str}' 缺少 'starters' 或 'ends' 字段")
            sys.exit(1)

        starters = seg["starters"]
        ends = seg["ends"]

        if len(starters) != len(ends):
            print(
                f"[错误] key '{n_str}' 的 starters(len={len(starters)}) "
                f"与 ends(len={len(ends)}) 长度不一致"
            )
            sys.exit(1)

        # ---- 4. 读取 nii 文件 ----
        input_path = os.path.join(input_dir, filename)
        img = nib.load(input_path)
        # 使用 dataobj 读取，保持原始 dtype，不做隐式类型转换（例如转成 float64）
        orig_dtype = img.get_data_dtype()
        data = np.asanyarray(img.dataobj).astype(orig_dtype, copy=False)

        if data.ndim < 3:
            print(f"[错误] 文件 {filename} 的数据维度小于3，无法沿第三维切片")
            sys.exit(1)

        depth = data.shape[2]

        # ---- 5. 逐段切割并保存 ----
        for i, (start, end) in enumerate(zip(starters, ends)):
            assert start < end, (
                f"key '{n_str}' 第 {i} 段 start({start}) 必须小于 end({end})"
            )

            if end > depth:
                print(
                    f"[错误] 文件 {filename} 第 {i} 段区间 [{start}, {end}) "
                    f"超出数据第三维实际大小 {depth}"
                )
                sys.exit(1)

            out_filename = f"{n}_{i}.nii.gz"
            out_path = os.path.join(output_dir, out_filename)

            if os.path.exists(out_path):
                print(f"[错误] 输出文件已存在，拒绝覆盖: {out_path}")
                sys.exit(1)

            chunk = data[:, :, start:end]

            new_img = nib.Nifti1Image(chunk, affine=img.affine, header=img.header)
            new_img.set_data_dtype(orig_dtype)

            nib.save(new_img, out_path)
            print(f"已保存: {out_path}  (来自 {filename}[:, :, {start}:{end}])")

    print("全部处理完成。")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python split_nii.py <input_dir> <json_path> <output_dir>")
        sys.exit(1)

    split_nii_files(sys.argv[1], sys.argv[2], sys.argv[3])