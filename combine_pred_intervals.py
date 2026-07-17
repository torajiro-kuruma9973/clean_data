import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np


def merge_segmented_nii_files(
    json_path: str,
    segment_folder: str,
    output_folder: str,
) -> None:
    """
    将分段的 3D NIfTI 文件按病例和分段号合并。

    文件名示例
    ----------
    501_0.nii.gz
    501_1.nii.gz
    501_2.nii.gz

    其中：
    - 501 是病例名，同时对应 JSON 顶层 key "501"；
    - 0、1、2 是分段号；
    - 分段号必须从 0 开始且连续。

    合并规则
    --------
    1. 同一病例的分段按分段号升序排列；
    2. 沿第三个维度，即 axis=2 拼接；
    3. 如果 JSON 中 starters[0] == 1，
       则在合并结果最前面增加一个全零切片；
    4. 如果 starters[0] == 0，则不增加零切片；
    5. starters[0] 只允许为 0 或 1；
    6. 输出文件名为病例名加 .nii.gz，例如 501.nii.gz；
    7. 输出使用第0段的 affine、header、qform 和 sform。

    参数
    ----------
    json_path:
        JSON 文件路径。

    segment_folder:
        分段 NIfTI 文件所在文件夹。
        只处理当前目录下的 *.nii.gz，不递归子目录。

    output_folder:
        合并后 NIfTI 文件的输出文件夹。
        文件夹不存在时自动创建。
    """
    json_file = Path(json_path)
    segment_path = Path(segment_folder)
    output_path = Path(output_folder)

    if not json_file.is_file():
        raise FileNotFoundError(f"JSON 文件不存在：{json_file}")

    if not segment_path.is_dir():
        raise NotADirectoryError(
            f"分段文件夹不存在或不是目录：{segment_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with json_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 文件格式无效：{json_file}\n"
            f"错误位置：第 {exc.lineno} 行，第 {exc.colno} 列"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            f"JSON 顶层必须是字典，实际类型为："
            f"{type(metadata).__name__}"
        )

    nii_files = sorted(segment_path.glob("*.nii.gz"))

    if not nii_files:
        raise FileNotFoundError(
            f"分段文件夹中没有找到 .nii.gz 文件：{segment_path}"
        )

    # 匹配：
    #   病例名_分段号.nii.gz
    #
    # 使用贪婪匹配，使最后一个下划线之后的整数被解释为分段号。
    # 例如：
    #   patient_501_0.nii.gz
    #   病例名为 patient_501，分段号为 0。
    filename_pattern = re.compile(
        r"^(?P<case_name>.+)_(?P<segment_index>\d+)\.nii\.gz$"
    )

    # groups:
    # {
    #     "501": {
    #         0: Path("501_0.nii.gz"),
    #         1: Path("501_1.nii.gz")
    #     }
    # }
    groups: dict[str, dict[int, Path]] = {}

    for nii_file in nii_files:
        match = filename_pattern.match(nii_file.name)

        if match is None:
            raise ValueError(
                "文件名不符合“病例名_分段号.nii.gz”的格式。\n"
                f"文件：{nii_file.name}"
            )

        case_name = match.group("case_name")
        segment_index = int(match.group("segment_index"))

        if case_name not in groups:
            groups[case_name] = {}

        if segment_index in groups[case_name]:
            raise ValueError(
                "同一病例出现重复的分段号。\n"
                f"病例：{case_name}\n"
                f"分段号：{segment_index}\n"
                f"文件1：{groups[case_name][segment_index]}\n"
                f"文件2：{nii_file}"
            )

        groups[case_name][segment_index] = nii_file

    # ---------------------------------------------------------
    # 第一阶段：检查所有病例。
    #
    # 全部检查成功后才开始写入输出文件，避免处理一半后失败。
    # ---------------------------------------------------------
    checked_groups: list[
        tuple[str, list[Path], int, tuple[int, int]]
    ] = []

    for case_name in sorted(groups):
        segment_map = groups[case_name]
        segment_indices = sorted(segment_map)

        expected_indices = list(range(len(segment_indices)))

        if segment_indices != expected_indices:
            missing_indices = sorted(
                set(expected_indices) - set(segment_indices)
            )

            raise ValueError(
                "分段号必须从 0 开始并且连续。\n"
                f"病例：{case_name}\n"
                f"实际分段号：{segment_indices}\n"
                f"期望分段号：{expected_indices}\n"
                f"缺少分段号：{missing_indices}"
            )

        if case_name not in metadata:
            raise KeyError(
                "JSON 中找不到与病例名对应的 key。\n"
                f"病例名：{case_name}\n"
                f"期望 JSON key：{case_name!r}"
            )

        case_metadata = metadata[case_name]

        if not isinstance(case_metadata, dict):
            raise ValueError(
                f"JSON key {case_name!r} 对应的值不是字典"
            )

        if "starters" not in case_metadata:
            raise KeyError(
                f"JSON key {case_name!r} 缺少 starters 字段"
            )

        starters = case_metadata["starters"]

        if not isinstance(starters, list):
            raise TypeError(
                f"JSON key {case_name!r} 的 starters 不是列表"
            )

        if len(starters) == 0:
            raise ValueError(
                f"JSON key {case_name!r} 的 starters 是空列表"
            )

        first_starter = starters[0]

        assert first_starter in (0, 1), (
            f"JSON key {case_name!r} 的 starters[0] "
            f"必须是 0 或 1，实际值为 {first_starter!r}"
        )

        ordered_files = [
            segment_map[index]
            for index in segment_indices
        ]

        first_nii = nib.load(str(ordered_files[0]))

        if len(first_nii.shape) != 3:
            raise ValueError(
                "分段文件必须是 3D NIfTI。\n"
                f"文件：{ordered_files[0]}\n"
                f"shape：{first_nii.shape}"
            )

        reference_xy_shape = (
            first_nii.shape[0],
            first_nii.shape[1],
        )

        # 检查每个分段都是 3D，并且前两个维度完全一致。
        for segment_file in ordered_files:
            try:
                segment_nii = nib.load(str(segment_file))
            except Exception as exc:
                raise RuntimeError(
                    f"无法读取 NIfTI 文件：{segment_file}"
                ) from exc

            if len(segment_nii.shape) != 3:
                raise ValueError(
                    "分段文件必须是 3D NIfTI。\n"
                    f"文件：{segment_file}\n"
                    f"shape：{segment_nii.shape}"
                )

            current_xy_shape = (
                segment_nii.shape[0],
                segment_nii.shape[1],
            )

            if current_xy_shape != reference_xy_shape:
                raise ValueError(
                    "同一病例各分段的前两个维度不一致，"
                    "无法沿第三维拼接。\n"
                    f"病例：{case_name}\n"
                    f"参考尺寸：{reference_xy_shape}\n"
                    f"异常文件：{segment_file.name}\n"
                    f"异常尺寸：{current_xy_shape}"
                )

        checked_groups.append(
            (
                case_name,
                ordered_files,
                first_starter,
                reference_xy_shape,
            )
        )

    print(
        f"检查完成：共找到 {len(checked_groups)} 个病例，"
        "所有分段号、JSON 信息和图像尺寸均有效。"
    )

    # ---------------------------------------------------------
    # 第二阶段：加载、拼接并保存。
    # ---------------------------------------------------------
    for case_index, (
        case_name,
        ordered_files,
        first_starter,
        reference_xy_shape,
    ) in enumerate(checked_groups, start=1):
        segment_arrays: list[np.ndarray] = []

        for segment_file in ordered_files:
            segment_nii = nib.load(str(segment_file))
            segment_array = segment_nii.get_fdata(
                dtype=np.float32
            )

            if not np.isfinite(segment_array).all():
                raise ValueError(
                    "分段文件中存在 NaN 或 Inf。\n"
                    f"病例：{case_name}\n"
                    f"文件：{segment_file}"
                )

            segment_arrays.append(segment_array)

        # 沿第三个维度，即 z 轴拼接。
        merged_data = np.concatenate(
            segment_arrays,
            axis=2,
        )

        zero_frame_added = False

        if first_starter == 1:
            zero_frame = np.zeros(
                (
                    reference_xy_shape[0],
                    reference_xy_shape[1],
                    1,
                ),
                dtype=merged_data.dtype,
            )

            merged_data = np.concatenate(
                [zero_frame, merged_data],
                axis=2,
            )
            zero_frame_added = True

        # 复制第一段的空间信息。
        first_nii = nib.load(str(ordered_files[0]))
        output_affine = first_nii.affine.copy()
        output_header = first_nii.header.copy()

        # 输出数据是 float32。
        output_header.set_data_dtype(np.float32)

        output_nii = nib.Nifti1Image(
            merged_data.astype(np.float32, copy=False),
            affine=output_affine,
            header=output_header,
        )

        # 显式复制第一段的 qform、sform 和 code。
        first_qform, first_qform_code = first_nii.get_qform(
            coded=True
        )
        first_sform, first_sform_code = first_nii.get_sform(
            coded=True
        )

        if first_qform is not None:
            output_nii.set_qform(
                first_qform,
                code=int(first_qform_code),
            )

        if first_sform is not None:
            output_nii.set_sform(
                first_sform,
                code=int(first_sform_code),
            )

        output_file = output_path / f"{case_name}.nii.gz"
        nib.save(output_nii, str(output_file))

        segment_description = ", ".join(
            file.name for file in ordered_files
        )

        print(
            f"[{case_index}/{len(checked_groups)}] "
            f"已合并：{case_name}\n"
            f"  分段：{segment_description}\n"
            f"  starters[0]：{first_starter}\n"
            f"  添加零帧：{'是' if zero_frame_added else '否'}\n"
            f"  输出 shape：{merged_data.shape}\n"
            f"  输出文件：{output_file}"
        )

    print(
        f"\n处理完成：共生成 {len(checked_groups)} 个 NIfTI 文件。\n"
        f"输出文件夹：{output_path}"
    )

if __name__ == "__main__":
    merge_segmented_nii_files(
    json_path="enhanced_orinal_intv_info.json",
    segment_folder="../pth2nii/pth2suv_nii_results",
    output_folder="../pth2nii/combined_nii",
)