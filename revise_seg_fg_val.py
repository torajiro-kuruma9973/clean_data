import os
import pydicom
import numpy as np
import re

def parse_txt_report(txt_path):
    """
    解析你的 TXT 报告，提取 Study_ID 和异常点的详细信息。
    """
    abnormal_records = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 使用正则提取每一条记录
        blocks = re.split(r'▶ \[Excel 行号: \d+\]', content)
        for block in blocks[1:]:
            record = {}
            lines = block.strip().split('\n')
            for line in lines:
                if 'Study_ID:' in line: record['study_id'] = line.split(': ')[1].strip()
                if 'col_x:' in line: record['col_x'] = int(line.split(': ')[1].strip())
                if 'row_y:' in line: record['row_y'] = int(line.split(': ')[1].strip())
                if 'space_z_mm:' in line: record['z_mm'] = float(line.split(': ')[1].strip())
            abnormal_records.append(record)
    return abnormal_records

def modify_seg_mask(root_dir, txt_path):
    """
    原地修改 SEG DICOM 文件。
    """
    records = parse_txt_report(txt_path)
    
    # 按照 Study ID 分组，减少遍历开销
    from collections import defaultdict
    study_map = defaultdict(list)
    for r in records:
        study_map[r['study_id']].append(r)

    # 递归遍历根目录
    for root, dirs, files in os.walk(root_dir):
        folder_name = os.path.basename(root)
        if folder_name in study_map:
            print(f"🔍 找到 Study 文件夹: {folder_name}")
            
            # 找到下面的 Segmentation 文件夹
            seg_dir = None
            for d in os.listdir(root):
                if 'segmentation' in d.lower():
                    seg_dir = os.path.join(root, d)
            
            if not seg_dir:
                print(f"⚠️ 找不到 Segmentation 文件夹: {folder_name}")
                continue
                
            # 读取 SEG DICOM
            seg_files = [f for f in os.listdir(seg_dir) if f.endswith('.dcm')]
            seg_path = os.path.join(seg_dir, seg_files[0])
            ds = pydicom.dcmread(seg_path)
            pixels = ds.pixel_array
            
            # 建立 Z 轴映射
            z_map = {}
            for i, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                z = round(float(frame.PlanePositionSequence[0].ImagePositionPatient[2]), 3)
                z_map[z] = i
            
            # 处理异常点
            for rec in study_map[folder_name]:
                z_val = round(rec['z_mm'], 3)
                if z_val in z_map:
                    frame_idx = z_map[z_val]
                    x, y = rec['col_x'], rec['row_y']
                    
                    # Assert 检查
                    current_val = pixels[frame_idx, y, x]
                    assert current_val == 1, f"❌ Assert 失败！Study {folder_name} 在 ({x},{y}) 帧 {frame_idx} 处不是前景！当前值为 {current_val}"
                    
                    # 原地修改
                    print(f"✅ 将 {folder_name} 的 ({x},{y}) 改为 0")
                    pixels[frame_idx, y, x] = 0
            
            # 写回像素并覆盖保存
            ds.PixelData = pixels.tobytes()
            ds.save_as(seg_path)
            print(f"💾 {folder_name} 已保存更新。")

if __name__ == "__main__":
    # 配置你的路径
    ROOT_DATA_DIR = "./PSMA-PET-CT-Lesions" 
    REPORT_TXT = "./zero_value_records_details.txt"
    
    modify_seg_mask(ROOT_DATA_DIR, REPORT_TXT)