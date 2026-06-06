import os
import dicom2nifti
from pathlib import Path

def convert_psma_to_nifti(root_path):
    """
    遍历指定的 PSMA 数据集目录，将每个 Study 下的 CT 和 PET DICOM 序列合并为 3D NIfTI。
    """
    root_dir = Path(root_path)
    
    if not root_dir.exists():
        print(f"❌ 找不到路径: {root_path}")
        return

    print(f"🚀 开始扫描目录: {root_path}\n" + "="*50)

    # 1. 遍历 Project ID 层级 (例如 PSMA_0179419e313f7d8c)
    for project_dir in root_dir.iterdir():
        if not project_dir.is_dir():
            continue
            
        # 2. 遍历 Study ID 层级 (例如 05-03-2002-NA-PETCT...)
        for study_dir in project_dir.iterdir():
            if not study_dir.is_dir():
                continue
                
            print(f"📂 正在处理 Study: {project_dir.name} / {study_dir.name}")
            
            # 3. 遍历具体的序列文件夹 (CT, PET, Seg)
            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue
                    
                folder_name_lower = series_dir.name.lower()
                
                # 🛑 核心过滤逻辑：首先排除所有包含 segmentation 的文件夹！
                # 即使它名字里有 'ct'，只要有 'segmentation' 就直接跳过
                if 'segmentation' in folder_name_lower:
                    continue
                    
                # 🎯 匹配 CT 文件夹
                elif 'ct' in folder_name_lower:
                    output_nii = series_dir / "ct.nii.gz"
                    _do_conversion(series_dir, output_nii, "CT")
                    
                # 🎯 匹配 PET 文件夹
                elif 'pet' in folder_name_lower:
                    output_nii = series_dir / "pet.nii.gz"
                    _do_conversion(series_dir, output_nii, "PET")

    print("="*50 + "\n✅ 所有转换任务执行完毕！")

def _do_conversion(input_dcm_dir, output_nii_path, modality_name):
    """
    执行具体的转换逻辑，包含跳过已存在文件和异常处理
    """
    # 如果该文件夹下已经存在生成的 nii.gz，则跳过（方便中断后继续运行）
    if output_nii_path.exists():
        print(f"   ⏩ {modality_name} 已存在，跳过: {output_nii_path.name}")
        return
        
    print(f"   ⏳ 正在合成 {modality_name} -> {output_nii_path.name} ...")
    
    try:
        # reorient_nifti=True 极其重要，标准化坐标轴方向 (RAS)
        dicom2nifti.dicom_series_to_nifti(
            str(input_dcm_dir), 
            str(output_nii_path), 
            reorient_nifti=True
        )
        print(f"   ✅ {modality_name} 成功!")
    except Exception as e:
        print(f"   ❌ {modality_name} 转换失败: {e}")

# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    # 替换为你实际的文件夹根目录路径
    # 注意 Windows 路径建议前面加 r，例如 r"D:\Datasets\PSMA_Data"
    target_path = "./PSMA-PET-CT-Lesions"
    convert_psma_to_nifti(target_path)