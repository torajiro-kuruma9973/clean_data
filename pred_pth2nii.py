import os
import glob
import numpy as np
import torch
import nibabel as nib


import os
import glob
import numpy as np
import torch
import nibabel as nib


import os
import glob
import numpy as np
import torch
import nibabel as nib


import os
import glob
import numpy as np
import torch
import nibabel as nib


def convert_pth_to_nii(input_dir, output_dir_raw, output_dir_suv, ref_nii_dir):
    """
    参数:
        input_dir: 存放 *_tensor.pth 文件的文件夹
        output_dir_raw: 原始数值(不做变换)nii.gz 输出文件夹
        output_dir_suv: SUV还原后nii.gz 输出文件夹
        ref_nii_dir: 参考nii文件夹(只读),用于获取affine/header/dtype
    """
    os.makedirs(output_dir_raw, exist_ok=True)
    os.makedirs(output_dir_suv, exist_ok=True)

    pth_files = sorted(glob.glob(os.path.join(input_dir, "*.pth")))

    for pth_path in pth_files:
        pth_name = os.path.basename(pth_path)

        # 生成输出文件名: 去掉 "_tensor.pth" 后缀，加上 ".nii.gz"
        if pth_name.endswith("_tensor.pth"):
            base_name = pth_name[: -len("_tensor.pth")]
        else:
            base_name = os.path.splitext(os.path.splitext(pth_name)[0])[0]
        out_name = base_name + ".nii.gz"

        # 参考nii,获取affine/header/dtype
        ref_nii_path = os.path.join(ref_nii_dir, out_name)
        if not os.path.exists(ref_nii_path):
            raise FileNotFoundError(f"参考nii文件不存在: {ref_nii_path}")

        ref_img = nib.load(ref_nii_path)
        affine = ref_img.affine
        ref_dtype = ref_img.get_data_dtype()

        # 加载pth文件
        data_dict = torch.load(pth_path, map_location="cpu")
        tensor = data_dict["tensor"]

        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = np.asarray(tensor)

        # 形状校验: 第一维(slice数)在[1,8]范围内均可，其余维度固定为(1,512,512)
        shape = tuple(tensor.shape)
        assert (
            len(shape) == 4
            and 1 <= shape[0] <= 8
            and shape[1] == 1
            and shape[2] == 512
            and shape[3] == 512
        ), (
            f"tensor形状不符合预期 [N,1,512,512] (N在1~8之间), 实际为: {shape} "
            f"(文件: {pth_name})"
        )

        # [N,1,512,512] -> [N,512,512]
        tensor = np.squeeze(tensor, axis=1)

        # 交换空间轴 (X/Y互换)
        tensor = np.transpose(tensor, (0, 2, 1))

        # 空间方向修正 (RAS -> LPS/HFS)
        # tensor原始方向为 RAS (X=右, Y=前)
        # DICOM的HFS物理坐标为 LPS (X=左, Y=后)
        # 因此在 Y轴(axis=1) 和 X轴(axis=2) 上做镜像翻转
        tensor = np.flip(tensor, axis=(1, 2))

        # [N,512,512] -> [512,512,N] 用于保存为nii
        tensor = np.transpose(tensor, (1, 2, 0))
        tensor = np.ascontiguousarray(tensor)

        # ---------- 1. 原始数值版本 ----------
        header_raw = ref_img.header.copy()
        header_raw.set_data_dtype(ref_dtype)
        raw_data = tensor.astype(ref_dtype)
        raw_img = nib.Nifti1Image(raw_data, affine, header_raw)
        nib.save(raw_img, os.path.join(output_dir_raw, out_name))

        # ---------- 2. SUV还原版本 ----------
        # suv = clip(normed ** 2 * 50, 0, 50)
        suv_data = np.clip((tensor.astype(np.float64)) ** 2 * 50.0, 0, 50)
        header_suv = ref_img.header.copy()
        header_suv.set_data_dtype(ref_dtype)
        suv_data = suv_data.astype(ref_dtype)
        suv_img = nib.Nifti1Image(suv_data, affine, header_suv)
        nib.save(suv_img, os.path.join(output_dir_suv, out_name))

        print(f"处理完成: {pth_name} -> {out_name}")


if __name__ == "__main__":
    convert_pth_to_nii(
        input_dir="../samsyn_test_predict_results",
        output_dir_raw="../pth2nii/pth2normed_nii_results",
        output_dir_suv="../pth2nii/pth2suv_nii_results",
        ref_nii_dir="../test_data_intervals",
    )