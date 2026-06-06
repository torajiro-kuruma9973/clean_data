import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def get_suv_factor(dcm):
    """
    基于 DICOM 头文件动态计算 SUVbw 的转换系数。
    """
    try:
        units = dcm.get((0x0054, 0x1001), None)
        if units and 'SUV' in str(units.value).upper():
            return 1.0

        weight = float(dcm.PatientWeight) * 1000.0
        radio_seq = dcm.RadiopharmaceuticalInformationSequence[0]
        dose = float(radio_seq.RadionuclideTotalDose)      
        half_life = float(radio_seq.RadionuclideHalfLife)  

        def parse_time(tm_str):
            if '.' in tm_str:
                tm_str = tm_str.split('.')[0]
            tm_str = tm_str.ljust(6, '0') 
            return datetime.strptime(tm_str, "%H%M%S")

        inj_time = parse_time(radio_seq.RadiopharmaceuticalStartTime)
        acq_time = parse_time(dcm.AcquisitionTime)

        if inj_time > acq_time:
            acq_time += timedelta(days=1)
        decay_time = (acq_time - inj_time).total_seconds()

        decayed_dose = dose * (2 ** (-decay_time / half_life))
        return weight / decayed_dose
    except Exception:
        return None

def analyze_and_plot_slice_max_suv(root_dir: str, output_png: str):
    if not os.path.exists(root_dir):
        print(f"❌ 找不到根目录: {root_dir}")
        return

    print(f"🚀 开始深度扫描目录并提取切片最大 SUV: {root_dir} ...")
    
    max_suv_list = []
    processed_slices = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_name = os.path.basename(dirpath).lower()
        if "pet" in folder_name and "segmentation" not in folder_name:
            for filename in filenames:
                if not filename.endswith('.dcm') and not filename.isnumeric():
                    continue
                file_path = os.path.join(dirpath, filename)
                try:
                    dcm = pydicom.dcmread(file_path, stop_before_pixels=False)
                    suv_factor = get_suv_factor(dcm)
                    
                    if suv_factor is not None:
                        slope = getattr(dcm, 'RescaleSlope', 1.0)
                        intercept = getattr(dcm, 'RescaleIntercept', 0.0)
                        
                        pixel_array = dcm.pixel_array.astype(np.float32)
                        suv_matrix = (pixel_array * slope + intercept) * suv_factor
                        slice_max_suv = float(np.max(suv_matrix))
                        
                        if slice_max_suv > 0.1:
                            max_suv_list.append(slice_max_suv)
                            processed_slices += 1
                except Exception:
                    continue

    if not max_suv_list:
        print("❌ 未能提取到任何有效的 SUV 数据。")
        return
        
    print(f"✅ 提取完毕！共计算了 {processed_slices} 张 PET 切片的 SUVmax。")
    plot_data = np.array(max_suv_list)

    suv_max = np.max(plot_data)
    suv_min = np.min(plot_data)
    suv_mean = np.mean(plot_data)
    suv_median = np.median(plot_data)

    print(f"📊 统计结果 -> Max: {suv_max:.2f}, Min: {suv_min:.2f}, Mean: {suv_mean:.2f}, Median: {suv_median:.2f}")

    # ==========================================
    # 开始绘图
    # ==========================================
    plt.figure(figsize=(15, 8), facecolor='#F5F7FA')
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FFFFFF", "grid.color": "#E5E9F0", "axes.edgecolor": "#D8DEE9"})
    
    ax = sns.kdeplot(plot_data, fill=True, color="#4C566A", facecolor="#5E81AC", alpha=0.6, linewidth=3)
    
    plt.title('Global Distribution of $SUV_{max}$ per PET Slice', fontsize=18, fontweight='bold', color='#2E3440', pad=20)
    plt.xlabel('$SUV_{max}$ (Maximum SUV in a single slice)', fontsize=14, color='#3B4252', labelpad=15)
    plt.ylabel('Probability Density', fontsize=14, color='#3B4252', labelpad=15)

    # 🌟 修复点：鲁棒地提取核密度曲线坐标数据 (兼容不同版本的 Seaborn)
    x_data, y_data = None, None
    if ax.get_lines():
        x_data = ax.get_lines()[0].get_xdata()
        y_data = ax.get_lines()[0].get_ydata()
    elif ax.collections:
        # 当 fill=True 时，Seaborn 会生成 PolyCollection
        path = ax.collections[0].get_paths()[0]
        vertices = path.vertices
        x_data = vertices[:, 0]
        y_data = vertices[:, 1]

    # 确保成功提取到了数据才进行标注
    if x_data is not None and y_data is not None and len(x_data) > 0:
        def get_curve_y(x_val):
            idx = (np.abs(x_data - x_val)).argmin()
            return y_data[idx]
            
        peak_idx = np.argmax(y_data)
        peak_x = x_data[peak_idx]
        peak_y = y_data[peak_idx]
        
        y_mean = get_curve_y(suv_mean)
        y_median = get_curve_y(suv_median)
        y_min = get_curve_y(suv_min)
        y_max = get_curve_y(suv_max)

        def create_bbox(color, alpha=0.9):
            return dict(boxstyle="round,pad=0.5", fc="#FFFFFF", ec=color, lw=1.5, alpha=alpha)
            
        def create_arrow(color):
            return dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color, lw=2)

        peak_color = "#BF616A"
        plt.vlines(peak_x, ymin=0, ymax=peak_y, color=peak_color, linestyle='-', linewidth=2, alpha=0.8)
        ax.annotate(f"★ Peak Density ★\nSUV: {peak_x:.2f}\nDensity: {peak_y:.4f}",
                    xy=(peak_x, peak_y), 
                    xytext=(peak_x, peak_y + (np.max(y_data) * 0.15)),
                    bbox=dict(boxstyle="round4,pad=0.6", fc="#FFF0F2", ec=peak_color, lw=2, alpha=0.95),
                    arrowprops=create_arrow(peak_color), 
                    fontsize=12, color=peak_color, fontweight='bold', ha='center')

        mean_color = "#D08770"
        plt.vlines(suv_mean, ymin=0, ymax=y_mean, color=mean_color, linestyle='--', linewidth=2, alpha=0.7)
        ax.annotate(f"Mean\nSUV: {suv_mean:.2f}\nDen: {y_mean:.4f}",
                    xy=(suv_mean, y_mean), 
                    xytext=(suv_mean + (suv_max*0.06), y_mean + (np.max(y_data)*0.05)),
                    bbox=create_bbox(mean_color), arrowprops=create_arrow(mean_color), 
                    fontsize=10, color=mean_color, fontweight='bold')

        median_color = "#A3BE8C"
        plt.vlines(suv_median, ymin=0, ymax=y_median, color=median_color, linestyle=':', linewidth=2, alpha=0.8)
        ax.annotate(f"Median\nSUV: {suv_median:.2f}\nDen: {y_median:.4f}",
                    xy=(suv_median, y_median), 
                    xytext=(suv_median + (suv_max*0.12), y_median + (np.max(y_data)*0.15)),
                    bbox=create_bbox(median_color), arrowprops=create_arrow(median_color), 
                    fontsize=10, color="#8FBCBB", fontweight='bold')

        min_color = "#4C566A"
        ax.annotate(f"Min\nSUV: {suv_min:.2f}",
                    xy=(suv_min, y_min), 
                    xytext=(suv_min + (suv_max*0.02), y_min + (np.max(y_data)*0.08)),
                    bbox=create_bbox(min_color), arrowprops=create_arrow(min_color), 
                    fontsize=10, color=min_color)

        ax.annotate(f"Max\nSUV: {suv_max:.2f}",
                    xy=(suv_max, y_max), 
                    xytext=(suv_max - (suv_max*0.15), y_max + (np.max(y_data)*0.08)),
                    bbox=create_bbox(min_color), arrowprops=create_arrow(min_color), 
                    fontsize=10, color=min_color)

        # 🌟 修复点：将拉高 Y 轴的操作放进 if 里面，确保 y_data 存在才执行
        plt.ylim(0, np.max(y_data) * 1.3)
    else:
        print("⚠️ 警告：无法从图像中提取 KDE 曲线数据，已跳过高级坐标标注。")

    plt.xticks(fontsize=12, color='#4C566A')
    plt.yticks(fontsize=12, color='#4C566A')
    
    plt.tight_layout()

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
        plt.savefig(output_png, dpi=300, format='png', bbox_inches='tight')
        print(f"💾 成功！极具视觉质感的 SUVmax 分布图已保存至: {output_png}")
    except Exception as e:
        print(f"❌ 图像保存失败: {e}")
        
    plt.close()


# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    # 参数 1: 数据集根目录
    root_dataset = r"D:\D_Work\Datasets\NBIA-PSMA-manifest-1772126181965-backup"
    
    # 参数 2: 导出的 PNG 文件路径
    output_image_path = "./suv_global_distribution.png"
    
    # 执行函数
    analyze_and_plot_slice_max_suv(root_dataset, output_image_path)