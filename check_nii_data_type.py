import SimpleITK as sitk
import os

def check_nifti_dtype(file_path):
    """
    极速读取 NIfTI 文件头，判断数据类型。
    绝对不会把完整图像载入内存。
    """
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return None
        
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(file_path)
        
        # 核心操作：只读文件头 (Header)，不读数据体
        reader.ReadImageInformation()
        
        # 【修复点】：先获取 ID，再用全局函数转成人类可读的字符串
        pixel_id = reader.GetPixelID()
        pixel_type = sitk.GetPixelIDValueAsString(pixel_id)
        
        print(f"📄 文件: {os.path.basename(file_path)}")
        print(f"📊 类型: {pixel_type}")
        print("-" * 30)
        
        return pixel_type
        
    except Exception as e:
        print(f"❌ 读取 {os.path.basename(file_path)} 报错: {e}")
        return None

# ===============================
# 使用示例
# ===============================
if __name__ == "__main__":
    check_nifti_dtype("0.nii.gz")