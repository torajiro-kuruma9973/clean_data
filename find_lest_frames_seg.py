import os
import glob
import pydicom

def find_study_with_min_seg_frames(root_dir):
    """
    递归遍历根目录，找到所有包含 'segmentation' 的文件夹，
    读取其中的 DICOM SEG 文件，并返回帧数最少的 Study ID。
    """
    min_frames = float('inf')
    min_study_id = None
    min_seg_file_path = None

    print(f"🚀 开始扫描目录: {root_dir}")

    # os.walk 递归遍历所有层级
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 将当前文件夹名转为小写以进行无大小写敏感匹配
        folder_lower = os.path.basename(dirpath).lower()
        
        # 精准锁定 segmentation 文件夹（无视是否带有 ct 字样）
        if 'segmentation' in folder_lower:
            # 获取该文件夹下的所有 dcm 文件
            dcm_files = glob.glob(os.path.join(dirpath, "*.dcm"))
            
            if not dcm_files:
                continue
                
            # 按照你的目录结构说明，SEG 文件夹里只有一个多帧 3D dcm 文件
            seg_file = dcm_files[0]
            
            # Study ID 是 SEG 文件夹的上一级父目录
            study_id = os.path.basename(os.path.dirname(dirpath))
            
            try:
                # 极速读取模式：只读文件头，不加载图像数据本身
                ds = pydicom.dcmread(seg_file, stop_before_pixels=True)
                
                # 获取帧数，标准多帧 DICOM 都会有 NumberOfFrames 属性
                if 'NumberOfFrames' in ds:
                    num_frames = int(ds.NumberOfFrames)
                else:
                    # 极小概率的容错机制：如果没写在头文件里，则强制读取像素阵列看深度
                    ds_full = pydicom.dcmread(seg_file)
                    num_frames = ds_full.pixel_array.shape[0]

                # 更新最小值记录
                if num_frames < min_frames:
                    min_frames = num_frames
                    min_study_id = study_id
                    min_seg_file_path = seg_file
                    
            except Exception as e:
                print(f"⚠️ 读取文件出错 [{seg_file}]: {e}")

    # --- 扫描结束，输出结果 ---
    if min_study_id:
        print("\n" + "="*40)
        print("🎯 扫描完成！找到帧数最少的 SEG 文件：")
        print("="*40)
        print(f"📍 Study ID  : {min_study_id}")
        print(f"🎞️ 最少帧数  : {min_frames} 帧")
        print(f"📁 文件路径  : {min_seg_file_path}")
        print("="*40)
    else:
        print("\n❌ 未找到任何有效的 SEG DICOM 文件。")

    return min_study_id, min_frames

# ==========================================
# 运行测试
# ==========================================
if __name__ == "__main__":
    # 请将此处替换为你的真实根目录路径，例如 "/data/PSMA_Dataset"
    ROOT_DATA_DIR = "./PSMA-PET-CT-Lesions" 
    find_study_with_min_seg_frames(ROOT_DATA_DIR)