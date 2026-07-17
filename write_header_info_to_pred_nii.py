from pathlib import Path
import os
import tempfile

import nibabel as nib
import numpy as np


def copy_gt_metadata_to_pred(
    gt_folder: str,
    pred_folder: str,
) -> None:
    """
    将 GT 文件的 affine、header、qform 和 sform
    复制到对应的 pred NIfTI 文件中。

    文件名匹配规则：
        pred/501.nii.gz  ->  gt/501.nii.gz

    即 GT 和 pred 文件夹中的对应文件名必须完全一致。

    注意：
    - GT 文件夹只读，不会修改其中的任何文件。
    - pred 的体素数据保持不变。
    - pred 文件会被原地覆盖。
    - 只处理 pred_folder 当前目录下的 *.nii.gz 文件。
    - 所有文件检查通过后才开始写入。
    """
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)

    if not gt_path.is_dir():
        raise NotADirectoryError(
            f"GT 文件夹不存在或不是目录：{gt_path}"
        )

    if not pred_path.is_dir():
        raise NotADirectoryError(
            f"pred 文件夹不存在或不是目录：{pred_path}"
        )

    pred_files = sorted(pred_path.glob("*.nii.gz"))

    if not pred_files:
        raise FileNotFoundError(
            f"pred 文件夹当前目录下没有 .nii.gz 文件：{pred_path}"
        )

    file_pairs: list[tuple[Path, Path]] = []

    # 第一阶段：检查所有文件，不做任何修改。
    for pred_file in pred_files:
        # GT 和 pred 使用完全相同的文件名。
        gt_file = gt_path / pred_file.name

        if not gt_file.is_file():
            raise FileNotFoundError(
                "处理终止：在 GT 文件夹中找不到同名文件。\n"
                f"pred 文件：{pred_file}\n"
                f"期望的 GT 文件：{gt_file}"
            )

        try:
            gt_nii = nib.load(str(gt_file))
        except Exception as exc:
            raise RuntimeError(
                f"处理终止：无法读取 GT 文件：{gt_file}"
            ) from exc

        try:
            pred_nii = nib.load(str(pred_file))
        except Exception as exc:
            raise RuntimeError(
                f"处理终止：无法读取 pred 文件：{pred_file}"
            ) from exc

        if gt_nii.shape != pred_nii.shape:
            raise ValueError(
                "处理终止：GT 和 pred 的体素形状不一致。\n"
                f"文件名：{pred_file.name}\n"
                f"GT shape：{gt_nii.shape}\n"
                f"pred shape：{pred_nii.shape}"
            )

        file_pairs.append((gt_file, pred_file))

    print(
        f"检查完成：共找到 {len(file_pairs)} 对同名文件，"
        "文件及形状均有效。"
    )

    # 第二阶段：写入 pred 文件。
    for gt_file, pred_file in file_pairs:
        temp_name = None

        try:
            gt_nii = nib.load(str(gt_file))
            pred_nii = nib.load(str(pred_file))

            # 只保留 pred 的体素数据。
            pred_data = pred_nii.get_fdata(dtype=np.float32)

            # 使用副本，避免修改 GT 对象。
            gt_affine = gt_nii.affine.copy()
            gt_header = gt_nii.header.copy()

            # 保证输出保存为 float32，避免 GT header 的数据类型
            # 将 pred 的浮点数据转换成其他类型。
            gt_header.set_data_dtype(np.float32)

            output_nii = nib.Nifti1Image(
                pred_data,
                affine=gt_affine,
                header=gt_header,
            )

            # 显式复制 qform、sform 及对应 code。
            gt_qform, gt_qform_code = gt_nii.get_qform(coded=True)
            gt_sform, gt_sform_code = gt_nii.get_sform(coded=True)

            if gt_qform is not None:
                output_nii.set_qform(
                    gt_qform,
                    code=int(gt_qform_code),
                )

            if gt_sform is not None:
                output_nii.set_sform(
                    gt_sform,
                    code=int(gt_sform_code),
                )

            # 先保存到同目录临时文件，成功后再替换原文件。
            temp_fd, temp_name = tempfile.mkstemp(
                prefix=f".{pred_file.name}_",
                suffix=".nii.gz",
                dir=str(pred_path),
            )
            os.close(temp_fd)

            nib.save(output_nii, temp_name)
            os.replace(temp_name, pred_file)

            print(
                f"已更新：{pred_file.name} "
                f"<- 元信息来自 {gt_file.name}"
            )

        except Exception as exc:
            if temp_name is not None and os.path.exists(temp_name):
                os.remove(temp_name)

            raise RuntimeError(
                "处理过程中发生错误，程序已终止。\n"
                f"GT 文件：{gt_file}\n"
                f"pred 文件：{pred_file}\n"
                f"错误信息：{exc}"
            ) from exc

    print(f"处理完成：共更新 {len(file_pairs)} 个 pred 文件。")


if __name__ == "__main__":
    copy_gt_metadata_to_pred(
        gt_folder=r"D:\D_Work\Samsyn_data\test_pet_not_normed",
        pred_folder=r"E:\datasets\psma_dcm_and_nii\pth2nii\combnined_nii_with_complement_with_0_frames",
    )
