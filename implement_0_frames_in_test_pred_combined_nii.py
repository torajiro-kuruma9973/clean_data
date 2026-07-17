from pathlib import Path

import nibabel as nib
import numpy as np


def pad_pred_z_to_gt(
    gt_folder: str,
    pred_folder: str,
    output_folder: str,
) -> None:
    """
    将 pred 文件的 z 轴长度补齐到同名 GT 文件的 z 轴长度。

    文件匹配规则
    ------------
    pred_folder/502.nii.gz 对应 gt_folder/502.nii.gz。

    处理规则
    --------
    1. GT 和 pred 必须都是 3D NIfTI 文件；
    2. GT 和 pred 的前两个维度必须一致；
    3. 必须满足：

           pred.shape[2] <= gt.shape[2]

    4. 如果 z 轴长度相等，直接将 pred 保存到输出文件夹；
    5. 如果 pred 的 z 轴较短，则在 z 轴末尾补全零帧；
    6. 输出文件保持与 pred 相同的文件名；
    7. 输出复制 pred 的 affine、header、qform 和 sform；
    8. 不修改 GT 文件和原始 pred 文件。

    参数
    ----------
    gt_folder:
        GT NIfTI 文件夹，只读。

    pred_folder:
        pred NIfTI 文件夹，只读。

    output_folder:
        补齐后文件的输出文件夹。
        如果不存在，会自动创建。
    """
    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    output_path = Path(output_folder)

    if not gt_path.is_dir():
        raise NotADirectoryError(
            f"GT 文件夹不存在或不是目录：{gt_path}"
        )

    if not pred_path.is_dir():
        raise NotADirectoryError(
            f"pred 文件夹不存在或不是目录：{pred_path}"
        )

    output_path.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_path.glob("*.nii.gz"))

    if not pred_files:
        raise FileNotFoundError(
            f"pred 文件夹当前目录下没有找到 .nii.gz 文件：{pred_path}"
        )

    # 先检查全部文件，避免处理到一半才发现错误。
    file_pairs: list[tuple[Path, Path]] = []

    for pred_file in pred_files:
        gt_file = gt_path / pred_file.name

        if not gt_file.is_file():
            raise FileNotFoundError(
                "处理终止：GT 文件夹中找不到同名文件。\n"
                f"pred 文件：{pred_file}\n"
                f"期望的 GT 文件：{gt_file}"
            )

        try:
            pred_nii = nib.load(str(pred_file))
        except Exception as exc:
            raise RuntimeError(
                f"处理终止：无法读取 pred 文件：{pred_file}"
            ) from exc

        try:
            gt_nii = nib.load(str(gt_file))
        except Exception as exc:
            raise RuntimeError(
                f"处理终止：无法读取 GT 文件：{gt_file}"
            ) from exc

        if len(pred_nii.shape) != 3:
            raise ValueError(
                "处理终止：pred 文件不是 3D 图像。\n"
                f"文件：{pred_file}\n"
                f"shape：{pred_nii.shape}"
            )

        if len(gt_nii.shape) != 3:
            raise ValueError(
                "处理终止：GT 文件不是 3D 图像。\n"
                f"文件：{gt_file}\n"
                f"shape：{gt_nii.shape}"
            )

        if pred_nii.shape[:2] != gt_nii.shape[:2]:
            raise ValueError(
                "处理终止：GT 和 pred 的前两个维度不一致。\n"
                f"文件名：{pred_file.name}\n"
                f"GT shape：{gt_nii.shape}\n"
                f"pred shape：{pred_nii.shape}"
            )

        assert pred_nii.shape[2] <= gt_nii.shape[2], (
            "处理终止：pred 的 z 轴长度大于 GT。\n"
            f"文件名：{pred_file.name}\n"
            f"GT shape：{gt_nii.shape}\n"
            f"pred shape：{pred_nii.shape}"
        )

        file_pairs.append((gt_file, pred_file))

    print(
        f"检查完成：共找到 {len(file_pairs)} 对有效文件。"
    )

    padded_count = 0
    unchanged_count = 0

    for index, (gt_file, pred_file) in enumerate(
        file_pairs,
        start=1,
    ):
        gt_nii = nib.load(str(gt_file))
        pred_nii = nib.load(str(pred_file))

        pred_data = pred_nii.get_fdata(dtype=np.float32)

        if not np.isfinite(pred_data).all():
            raise ValueError(
                "pred 文件中存在 NaN 或 Inf。\n"
                f"文件：{pred_file}"
            )

        gt_z = gt_nii.shape[2]
        pred_z = pred_nii.shape[2]
        missing_frames = gt_z - pred_z

        if missing_frames == 0:
            output_data = pred_data
            unchanged_count += 1

            print(
                f"[{index}/{len(file_pairs)}] 跳过补零："
                f"{pred_file.name} | "
                f"pred z={pred_z}, GT z={gt_z}"
            )

        else:
            zero_frames = np.zeros(
                (
                    pred_data.shape[0],
                    pred_data.shape[1],
                    missing_frames,
                ),
                dtype=np.float32,
            )

            output_data = np.concatenate(
                [pred_data, zero_frames],
                axis=2,
            )

            padded_count += 1

            print(
                f"[{index}/{len(file_pairs)}] 已补齐："
                f"{pred_file.name} | "
                f"pred z={pred_z}, GT z={gt_z}, "
                f"补充零帧={missing_frames}"
            )

        if output_data.shape != gt_nii.shape:
            raise RuntimeError(
                "补齐后的 shape 与 GT 不一致。\n"
                f"文件名：{pred_file.name}\n"
                f"输出 shape：{output_data.shape}\n"
                f"GT shape：{gt_nii.shape}"
            )

        # 复制 pred 的空间信息。
        output_affine = pred_nii.affine.copy()
        output_header = pred_nii.header.copy()
        output_header.set_data_dtype(np.float32)

        output_nii = nib.Nifti1Image(
            output_data.astype(np.float32, copy=False),
            affine=output_affine,
            header=output_header,
        )

        # 显式复制 pred 的 qform、sform 以及对应 code。
        pred_qform, pred_qform_code = pred_nii.get_qform(
            coded=True
        )
        pred_sform, pred_sform_code = pred_nii.get_sform(
            coded=True
        )

        if pred_qform is not None:
            output_nii.set_qform(
                pred_qform,
                code=int(pred_qform_code),
            )

        if pred_sform is not None:
            output_nii.set_sform(
                pred_sform,
                code=int(pred_sform_code),
            )

        output_file = output_path / pred_file.name
        nib.save(output_nii, str(output_file))

    print("\n处理完成。")
    print(f"总文件数：{len(file_pairs)}")
    print(f"补零文件数：{padded_count}")
    print(f"无需补零文件数：{unchanged_count}")
    print(f"输出文件夹：{output_path}")

if __name__ == "__main__":
    pad_pred_z_to_gt(
    gt_folder=r"D:\D_Work\Samsyn_data\test_pet_not_normed",
    pred_folder=r"E:\datasets\psma_dcm_and_nii\pth2nii\combined_nii",
    output_folder=r"E:\datasets\psma_dcm_and_nii\pth2nii\combnined_nii_with_complement_with_0_frames",
)