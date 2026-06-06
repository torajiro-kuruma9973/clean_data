import os
import glob
import pydicom
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_suv_factor(ds):
    """
    解析 DICOM 头部计算 SUVbw 转换系数 (包含放射性衰减校正)。
    """
    if 'Units' in ds and ds.Units == 'SUVbw':
        return ds.RescaleSlope
    try:
        weight_g = float(ds.PatientWeight) * 1000
        radio_seq = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose_bq = float(radio_seq.RadionuclideTotalDose)
        half_life_sec = float(radio_seq.RadionuclideHalfLife)
        
        scan_time_str = ds.get('AcquisitionTime', ds.get('SeriesTime'))
        inject_time_str = radio_seq.RadiopharmaceuticalStartTime
        
        if not scan_time_str or not inject_time_str:
            return getattr(ds, 'RescaleSlope', 1.0)

        scan_time = datetime.strptime(scan_time_str[:6], "%H%M%S")
        inject_time = datetime.strptime(inject_time_str[:6], "%H%M%S")
        delta_time_sec = (scan_time - inject_time).total_seconds()
        if delta_time_sec < 0:
            delta_time_sec += 24 * 3600

        decay_factor = np.exp(-np.log(2) * (delta_time_sec / half_life_sec))
        actual_dose = injected_dose_bq * decay_factor
        return (ds.RescaleSlope * weight_g) / actual_dose
    except Exception:
        return getattr(ds, 'RescaleSlope', 1.0)


def extract_lesion_points(pet_dir, seg_path, study_id):
    """
    根据 3D 掩码提取病灶内每一个点的详细空间和 SUV 信息。
    """
    print(f"🚀 [处理中] Study ID: {study_id}")
    
    # 1. 扫描并映射 PET 切片的 Z 轴坐标
    pet_slices = {}
    for f in glob.glob(os.path.join(pet_dir, "*.dcm")):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            z_coord = round(float(ds.ImagePositionPatient[2]), 3)
            pet_slices[z_coord] = f
        except Exception:
            continue
            
    # 2. 加载 SEG 数据
    seg_ds = pydicom.dcmread(seg_path)
    seg_pixels = seg_ds.pixel_array
    
    results = []
    
    # 3. 遍历 SEG 的每一帧
    if hasattr(seg_ds, 'PerFrameFunctionalGroupsSequence'):
        for i, frame in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
            try:
                frame_z = round(float(frame.PlanePositionSequence[0].ImagePositionPatient[2]), 3)
            except AttributeError:
                continue
                
            mask_slice = seg_pixels[i]
            
            # 如果这一帧没有病灶，直接跳过
            if np.max(mask_slice) == 0:
                continue
                
            # 找到对应的 PET 切片
            if frame_z in pet_slices:
                pet_file = pet_slices[frame_z]
                pet_ds = pydicom.dcmread(pet_file)
                pet_img = pet_ds.pixel_array.astype(np.float64)
                
                suv_factor = calculate_suv_factor(pet_ds)
                intercept = getattr(pet_ds, 'RescaleIntercept', 0.0)
                
                # 提取空间几何参数
                ipp = [float(x) for x in pet_ds.ImagePositionPatient]       # [X0, Y0, Z0]
                spacing = [float(x) for x in pet_ds.PixelSpacing]           # [dy(row), dx(col)]
                iop = [float(x) for x in pet_ds.ImageOrientationPatient]    # [Xx, Xy, Xz, Yx, Yy, Yz]
                
                # 找到掩码为 1 的所有像素的坐标 (row, col)
                rows, cols = np.where(mask_slice > 0)
                
                filename = os.path.basename(pet_file)
                
                # 遍历这一个切片上的每一个病灶像素点
                for r, c in zip(rows, cols):
                    raw_val = pet_img[r, c]
                    suv_val = (raw_val + intercept) * suv_factor
                    
                    # 几何变换：2D 像素 -> 3D 物理坐标
                    px = ipp[0] + c * spacing[1] * iop[0] + r * spacing[0] * iop[3]
                    py = ipp[1] + c * spacing[1] * iop[1] + r * spacing[0] * iop[4]
                    pz = ipp[2] + c * spacing[1] * iop[2] + r * spacing[0] * iop[5]
                    
                    # 保存到字典
                    results.append({
                        "Study_ID": study_id,
                        "PET_Filename": filename,
                        "col_x": c, 
                        "row_y": r,
                        "space_x_mm": px, 
                        "space_y_mm": py, 
                        "space_z_mm": pz,
                        "raw_value": raw_val,
                        "suv_value": suv_val
                    })
            else:
                print(f"⚠️ 警告: 找不到 Z={frame_z} 的 PET 切片进行匹配。")
                
    print(f"✅ 完成提取！该 Study 共找到 {len(results)} 个病灶像素点。\n")
    return results


def process_all_studies(root_dir, output_csv):
    """
    全局递归调度器：寻找所有合法的 Study，调用提取函数，并统一保存到 CSV。
    """
    all_lesion_points = []
    
    print(f"🔍 开始全局扫描根目录: {root_dir}")
    print("-" * 50)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        seg_dir = None
        pet_dir = None
        
        # 寻找当前层级下的 SEG 和 PET 文件夹
        for d in dirnames:
            d_lower = d.lower()
            if 'segmentation' in d_lower:
                seg_dir = os.path.join(dirpath, d)
            elif 'pet' in d_lower and 'segmentation' not in d_lower:
                pet_dir = os.path.join(dirpath, d)
                
        # 如果两者同时存在，说明定准了一个合法的 Study 目录
        if seg_dir and pet_dir:
            study_id = os.path.basename(dirpath)
            
            # 找到 SEG 文件夹下唯一的 3D dcm 文件
            seg_files = glob.glob(os.path.join(seg_dir, "*.dcm"))
            if not seg_files:
                print(f"⚠️ 警告: Study [{study_id}] 的 SEG 文件夹为空，已跳过。")
                continue
            seg_path = seg_files[0]
            
            # 调用提取核心函数
            study_results = extract_lesion_points(pet_dir, seg_path, study_id)
            all_lesion_points.extend(study_results)
            
    # 遍历结束，开始保存 CSV
    print("=" * 50)
    if not all_lesion_points:
        print("❌ 未在提供的目录中提取到任何有效数据，未生成 CSV。")
        return

    print(f"💾 正在将总计 {len(all_lesion_points)} 条病灶点数据写入 CSV ...")
    df = pd.DataFrame(all_lesion_points)
    
    # 格式化一下浮点数，保留合理的小数位数，减小文件体积
    df = df.round({'space_x_mm': 2, 'space_y_mm': 2, 'space_z_mm': 2, 'suv_value': 4})
    
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"🎉 大功告成！全量数据已成功保存至: {output_csv}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

def plot_suv_statistics_from_csv(csv_path, output_png_path="suv_distribution.png"):
    """
    读取 CSV 文件，统计 suv_value 并绘制高颜值的核密度分布图。
    """
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        return

    print(f"🚀 正在读取数据: {csv_path}")
    
    # 1. 读取 CSV 数据
    try:
        df = pd.read_csv(csv_path)
        if 'suv_value' not in df.columns:
            print("❌ CSV 文件中没有找到 'suv_value' 列！")
            return
            
        data = df['suv_value'].dropna().values
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    if len(data) == 0:
        print("❌ 'suv_value' 列没有有效数据。")
        return

    # 2. 计算统计量
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    median_val = np.median(data)

    # 3. 计算 KDE 曲线以获取峰值 (Mode / Peak)
    kde = gaussian_kde(data)
    x_range = np.linspace(min_val, max_val, 1000)
    kde_values = kde(x_range)
    peak_x = x_range[np.argmax(kde_values)]
    peak_y = np.max(kde_values)

    # 4. 在控制台打印统计结果
    print("\n" + "="*45)
    print("📊 SUV_value 统计汇总")
    print("="*45)
    print(f"有效数据点总数: {len(data)}")
    print(f"最小值 (Min)    : {min_val:.4f}")
    print(f"最大值 (Max)    : {max_val:.4f}")
    print(f"平均值 (Mean)   : {mean_val:.4f}")
    print(f"中位值 (Median) : {median_val:.4f}")
    print(f"峰值对应SUV     : {peak_x:.4f}")
    print("="*45)

    # 5. 开始复刻绘图风格
    plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    ax.grid(color='#E9ECEF', linestyle='-', linewidth=1, alpha=0.8)

    # 绘制带填充的 KDE 密度图
    sns.kdeplot(data, color='#3B5B88', linewidth=2, fill=True, alpha=0.8, bw_adjust=0.5, ax=ax)

    # 设置标题和坐标轴
    plt.title("lesion points SUV statistics in PET Image", fontsize=16, fontweight='bold', pad=20, color='#2B3A42')
    plt.xlabel('SUV value', fontsize=12, labelpad=10, color='#2B3A42')
    plt.ylabel('Probability Density', fontsize=12, labelpad=10, color='#2B3A42')
    
    # 动态调整 X 轴范围，留出一点边距
    x_margin = (max_val - min_val) * 0.05
    if x_margin == 0: x_margin = 1.0 # 防止全为单一值导致除零
    plt.xlim(min_val - x_margin, max_val + x_margin)
    
    # 动态调整 Y 轴范围，给顶部的 Peak 标签留出空间
    plt.ylim(0, peak_y * 1.25)

    # 辅助注释函数 (完全复刻例图风格)
    def add_annotation(text, xy, xytext, edge_color, text_color):
        ax.annotate(text, xy=xy, xytext=xytext,
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=edge_color, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9FA", ec=edge_color, lw=1.5, alpha=0.9),
                    fontsize=10, color=text_color, fontweight='bold', ha='center', zorder=5)

    # 动态计算偏移量，防止标签重叠
    x_span = max_val - min_val
    
    # [1] 峰值注释 (红色) - 放在峰值正上方
    add_annotation(f"Peak Density\nSUV: {peak_x:.2f}\nDensity: {peak_y:.4f}", 
                   xy=(peak_x, peak_y), 
                   xytext=(peak_x, peak_y * 1.1), 
                   edge_color="#C0504D", text_color="#C0504D")

    # [2] 最小值注释 (灰色) - 偏右上
    add_annotation(f"Min\nSUV: {min_val:.2f}", 
                   xy=(min_val, 0), 
                   xytext=(min_val + (x_span * 0.05), peak_y * 0.1), 
                   edge_color="#7F7F7F", text_color="#595959")

    # [3] 平均值注释 (橙色) - 偏右上，更高一点
    add_annotation(f"Mean\nSUV: {mean_val:.2f}", 
                   xy=(mean_val, 0), 
                   xytext=(mean_val + (x_span * 0.08), peak_y * 0.05), 
                   edge_color="#E36C0A", text_color="#E36C0A")

    # [4] 中位值注释 (绿色) - 偏右上，再高一点错开
    add_annotation(f"Median\nSUV: {median_val:.2f}", 
                   xy=(median_val, 0), 
                   xytext=(median_val + (x_span * 0.06), peak_y * 0.15), 
                   edge_color="#9BBB59", text_color="#76923C")

    # [5] 最大值注释 (深灰色) - 偏左上
    add_annotation(f"Max\nSUV: {max_val:.2f}", 
                   xy=(max_val, 0), 
                   xytext=(max_val - (x_span * 0.1), peak_y * 0.1), 
                   edge_color="#4F6272", text_color="#3A4B56")

    # 美化边框：隐藏上方、右方和左方的黑色实线框，只留底部的灰色线
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    # 紧凑布局并保存
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    print(f"🎉 绘图成功！图表已保存至: {output_png_path}")
    plt.close()

# ===============================
# 运行入口
# ===============================
if __name__ == "__main__":
    # 配置参数
    ROOT_DIRECTORY = "./PSMA-PET-CT-Lesions"  # 替换为你的真实根目录
    OUTPUT_CSV_FILE = "./all_psma_lesion_points.csv"  # 你想要的 CSV 名字和路径
    
    process_all_studies(ROOT_DIRECTORY, OUTPUT_CSV_FILE)

    OUTPUT_IMAGE = "lesion_points_suv_statistics.png"
    plot_suv_statistics_from_csv(OUTPUT_CSV_FILE, OUTPUT_IMAGE)