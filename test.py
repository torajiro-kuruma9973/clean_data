import os
import pydicom
import warnings

def scan_for_pixel_warnings(root_dir):
    """
    遍历根目录，寻找 Segmentation 文件夹并读取 DICOM，
    拦截并检测 ds.pixel_array 是否触发了 UserWarning。
    """
    print(f"🚀 开始全盘扫描根目录: {root_dir}")
    print(f"🔎 目标: 寻找触发 'UserWarning' (帧数/体积不匹配) 的 Study ID\n")
    print("-" * 60)
    
    warning_count = 0

    # os.walk 递归遍历所有层级
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 获取当前文件夹名并转小写
        folder_lower = os.path.basename(dirpath).lower()
        
        # 锁定 Segmentation 文件夹
        if 'segmentation' in folder_lower:
            dcm_files = [f for f in filenames if f.endswith('.dcm')]
            if not dcm_files:
                continue
            
            # 定位目标 dcm 文件 (优先找 1-1.dcm)
            target_file = '1-1.dcm' if '1-1.dcm' in dcm_files else dcm_files[0]
            file_path = os.path.join(dirpath, target_file)
            
            # 反推 Study ID
            study_id = os.path.basename(os.path.dirname(dirpath))
            
            try:
                # ==========================================
                # 🌟 核心拦截逻辑：开启“警告录音机”
                # ==========================================
                with warnings.catch_warnings(record=True) as w:
                    # 强制让所有警告都触发，不要被系统默认过滤掉
                    warnings.simplefilter("always")
                    
                    # 1. 读取文件头 (这一步通常不报错)
                    ds = pydicom.dcmread(file_path)
                    
                    # 2. 获取像素矩阵 (⚠️ 触发警告的高发地带)
                    pixels = ds.pixel_array
                    
                    # 3. 检查刚刚的录音机里有没有录下警告
                    has_target_warning = False
                    for warn in w:
                        # 检查是否是 UserWarning 类型
                        if issubclass(warn.category, UserWarning):
                            warn_msg = str(warn.message)
                            # 进一步匹配你的特征词，或者匹配底层源码特征 "The number of bytes"
                            if "The number of bytes of pixel data" in warn_msg or "base.py" in str(warn.filename):
                                has_target_warning = True
                                break
                    
                    # 4. 如果发现目标警告，打印 Study ID
                    if has_target_warning:
                        print(f"🚨 发现异常警告 -> Study ID: {study_id}")
                        # print(f"   [文件路径]: {file_path}") # 如果你需要看具体路径，可以取消注释这行
                        warning_count += 1
                        
            except Exception as e:
                # 捕获其他真实的 Crash 错误（比如文件彻底损坏无法读取）
                print(f"❌ 发生致命错误 [{study_id}]: {e}")

    print("-" * 60)
    if warning_count > 0:
        print(f"🏁 扫描完毕！共揪出 {warning_count} 个潜藏警告的异常 Study。")
    else:
        print("✨ 扫描完毕！所有数据均健康，未触发任何像素读取警告。")

# ===============================
# 运行测试
# ===============================
if __name__ == "__main__":
    # 请替换为你的真实根目录路径
    ROOT_DATA_DIR = "./PSMA-PET-CT-Lesions" 
    
    scan_for_pixel_warnings(ROOT_DATA_DIR)