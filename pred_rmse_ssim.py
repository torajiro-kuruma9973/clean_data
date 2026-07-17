from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_pet_rmse_ssim(
    gt_folder: str,
    pred_folder: str,
    output_txt_path: str,
    data_range: float,
) -> None:
    """
    计算 pred 文件夹中每个 3D NIfTI 文件与 GT 文件夹中
    同名文件的 RMSE 和 SSIM，并将结果写入 txt。

    匹配方式：
        pred 文件和 GT 文件的完整文件名必须相同。

    特殊处理：
        如果 NIfTI 图像的第三个维度 shape[2] < 3，
        则跳过该文件，不计算指标，也不计入最终平均值。
    """
    if data_range <= 0:
        raise ValueError(
            f"data_range 必须大于 0，当前值为 {data_range}"
        )

    gt_path = Path(gt_folder)
    pred_path = Path(pred_folder)
    txt_path = Path(output_txt_path)

    if not gt_path.is_dir():
        raise NotADirectoryError(
            f"GT 文件夹不存在或不是目录：{gt_path}"
        )

    if not pred_path.is_dir():
        raise NotADirectoryError(
            f"pred 文件夹不存在或不是目录：{pred_path}"
        )

    # 只遍历当前目录。
    pred_files = sorted(pred_path.glob("*.nii.gz"))

    if not pred_files:
        raise FileNotFoundError(
            f"pred 文件夹中没有找到 .nii.gz 文件：{pred_path}"
        )

    file_pairs: list[tuple[Path, Path]] = []
    skipped_files: list[tuple[str, tuple[int, ...], str]] = []

    # ---------------------------------------------------------
    # 第一阶段：检查、匹配文件。
    # ---------------------------------------------------------
    for pred_file in pred_files:
        # 直接在 GT 文件夹中查找完全同名的文件。
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

        if len(gt_nii.shape) != 3:
            raise ValueError(
                "处理终止：GT 文件不是 3D 图像。\n"
                f"文件：{gt_file}\n"
                f"shape：{gt_nii.shape}"
            )

        if len(pred_nii.shape) != 3:
            raise ValueError(
                "处理终止：pred 文件不是 3D 图像。\n"
                f"文件：{pred_file}\n"
                f"shape：{pred_nii.shape}"
            )

        if gt_nii.shape != pred_nii.shape:
            raise ValueError(
                "处理终止：GT 和 pred 的体素形状不一致。\n"
                f"文件名：{pred_file.name}\n"
                f"GT shape：{gt_nii.shape}\n"
                f"pred shape：{pred_nii.shape}"
            )

        # 第三个维度是 shape[2]。
        if gt_nii.shape[2] < 3:
            message = (
                f"第三个维度小于 3：shape={gt_nii.shape}"
            )

            skipped_files.append(
                (
                    pred_file.name,
                    tuple(gt_nii.shape),
                    message,
                )
            )

            print(
                f"跳过：{pred_file.name} | "
                f"shape={gt_nii.shape} | "
                f"第三个维度={gt_nii.shape[2]} < 3"
            )
            continue

        file_pairs.append((gt_file, pred_file))

    print(
        f"\n检查完成：总文件数={len(pred_files)}，"
        f"待计算={len(file_pairs)}，"
        f"跳过={len(skipped_files)}。"
    )

    if not file_pairs:
        raise RuntimeError(
            "没有可用于计算的文件：所有文件的第三个维度都小于 3。"
        )

    results: list[tuple[str, str, float, float]] = []

    default_win_size = 11
    win_sigma = 1.5

    # ---------------------------------------------------------
    # 第二阶段：计算 RMSE 和 SSIM。
    # ---------------------------------------------------------
    with torch.no_grad():
        for index, (gt_file, pred_file) in enumerate(
            file_pairs,
            start=1,
        ):
            try:
                gt_array = nib.load(str(gt_file)).get_fdata(
                    dtype=np.float32
                )
                pred_array = nib.load(str(pred_file)).get_fdata(
                    dtype=np.float32
                )

                if not np.isfinite(gt_array).all():
                    raise ValueError(
                        f"GT 文件中存在 NaN 或 Inf：{gt_file}"
                    )

                if not np.isfinite(pred_array).all():
                    raise ValueError(
                        f"pred 文件中存在 NaN 或 Inf：{pred_file}"
                    )

                # [H, W, D] -> [N, C, H, W, D]
                gt_tensor = torch.from_numpy(
                    np.ascontiguousarray(gt_array)
                ).unsqueeze(0).unsqueeze(0)

                pred_tensor = torch.from_numpy(
                    np.ascontiguousarray(pred_array)
                ).unsqueeze(0).unsqueeze(0)

                # RMSE = sqrt(MSE)
                rmse_tensor = torch.sqrt(
                    F.mse_loss(
                        pred_tensor,
                        gt_tensor,
                        reduction="mean",
                    )
                )

                # pytorch_msssim 的 3D SSIM 要求窗口不能超过
                # 任何一个空间维度，并且窗口大小必须为奇数。
                min_spatial_size = min(gt_array.shape)

                win_size = min(
                    default_win_size,
                    min_spatial_size,
                )

                if win_size % 2 == 0:
                    win_size -= 1

                ssim_tensor = ssim(
                    pred_tensor,
                    gt_tensor,
                    data_range=float(data_range),
                    size_average=True,
                    win_size=win_size,
                    win_sigma=win_sigma,
                    nonnegative_ssim=False,
                )

                rmse_value = float(rmse_tensor.item())
                ssim_value = float(ssim_tensor.item())

                results.append(
                    (
                        pred_file.name,
                        gt_file.name,
                        rmse_value,
                        ssim_value,
                    )
                )

                print(
                    f"[{index}/{len(file_pairs)}] "
                    f"{pred_file.name} | "
                    f"RMSE={rmse_value:.8f} | "
                    f"SSIM={ssim_value:.8f}"
                )

            except Exception as exc:
                raise RuntimeError(
                    "计算过程中发生错误，程序已终止。\n"
                    f"GT 文件：{gt_file}\n"
                    f"pred 文件：{pred_file}\n"
                    f"错误信息：{exc}"
                ) from exc

    # 只对成功计算的文件求平均。
    mean_rmse = sum(item[2] for item in results) / len(results)
    mean_ssim = sum(item[3] for item in results) / len(results)

    txt_path.parent.mkdir(parents=True, exist_ok=True)

    with txt_path.open("w", encoding="utf-8") as file:
        file.write("PET 3D Image Evaluation Results\n")
        file.write("=" * 88 + "\n")
        file.write(f"GT folder       : {gt_path}\n")
        file.write(f"Pred folder     : {pred_path}\n")
        file.write(f"Total files     : {len(pred_files)}\n")
        file.write(f"Evaluated files : {len(results)}\n")
        file.write(f"Skipped files   : {len(skipped_files)}\n")
        file.write(f"SSIM data_range : {data_range}\n\n")

        file.write(
            f"{'Pred file':<32}"
            f"{'GT file':<28}"
            f"{'RMSE':>14}"
            f"{'SSIM':>14}\n"
        )
        file.write("-" * 88 + "\n")

        for pred_name, gt_name, rmse_value, ssim_value in results:
            file.write(
                f"{pred_name:<32}"
                f"{gt_name:<28}"
                f"{rmse_value:>14.8f}"
                f"{ssim_value:>14.8f}\n"
            )

        file.write("-" * 88 + "\n")
        file.write(
            f"{'MEAN':<60}"
            f"{mean_rmse:>14.8f}"
            f"{mean_ssim:>14.8f}\n"
        )

        if skipped_files:
            file.write("\nSkipped files\n")
            file.write("-" * 88 + "\n")

            for filename, shape, reason in skipped_files:
                file.write(
                    f"{filename} | shape={shape} | {reason}\n"
                )

    print("\n计算完成。")
    print(f"成功计算：{len(results)} 个文件")
    print(f"跳过文件：{len(skipped_files)} 个")
    print(f"平均 RMSE：{mean_rmse:.8f}")
    print(f"平均 SSIM：{mean_ssim:.8f}")
    print(f"结果保存到：{txt_path}")


if __name__ == "__main__":
    calculate_pet_rmse_ssim(
    gt_folder="../test_data_normed_intervals",
    pred_folder="../pth2nii/pth2normed_nii_results",
    output_txt_path="./new_version_suv_normed_results.txt",
    data_range=1.0,
)