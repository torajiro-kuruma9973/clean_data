import os
import glob
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

def get_ct_hu_pixels(root_dir, sample_per_slice=2000):
    """
    遍历根目录寻找 CT 文件夹，读取 DICOM，转换为真实 HU 值，
    并对每张切片进行随机下采样以防内存溢出。
    """
    hu_samples = []
    
    print(f"🚀 开始扫描根目录: {root_dir}")
    print("-" * 50)
    
    # 递归遍历所有层级
    for dirpath, dirnames, filenames in os.walk(root_dir):
        folder_lower = os.path.basename(dirpath).lower()
        
        # 精准锁定 CT 文件夹：包含 ct，但绝不能包含 segmentation 或 pet
        if 'ct' in folder_lower and 'segmentation' not in folder_lower and 'pet' not in folder_lower:
            # 找到所有的 dcm 文件
            dcm_files = glob.glob(os.path.join(dirpath, "*.dcm"))
            if not dcm_files:
                continue
                
            study_id = os.path.basename(os.path.dirname(dirpath))
            print(f"🔄 正在处理 CT 序列 -> Study: {study_id} (共 {len(dcm_files)} 张切片)")
            
            for f in dcm_files:
                try:
                    ds = pydicom.dcmread(f)
                    
                    # 确保是图像文件
                    if not hasattr(ds, 'pixel_array'):
                        continue
                        
                    img = ds.pixel_array.astype(np.float32)
                    
                    # 获取转换 HU 的核心参数
                    slope = getattr(ds, 'RescaleSlope', 1.0)
                    intercept = getattr(ds, 'RescaleIntercept', 0.0)
                    
                    # 转换为真实 HU 值: HU = pixel * slope + intercept
                    hu_img = img * slope + intercept
                    
                    # ----------------------------------------------------
                    # 💡 核心清洗与优化：
                    # 1. 过滤扫描仪外部的空气填充伪影 (通常是 -2000 或 -3024)
                    # 真实的空气是 -1000 HU，低于 -1024 的毫无生理意义
                    # 2. 压平并随机采样
                    # ----------------------------------------------------
                    valid_pixels = hu_img[hu_img >= -1024].flatten()
                    
                    if len(valid_pixels) > sample_per_slice:
                        # 从这张切片中随机抽取代表性像素
                        sampled = np.random.choice(valid_pixels, size=sample_per_slice, replace=False)
                        hu_samples.append(sampled)
                    else:
                        hu_samples.append(valid_pixels)
                        
                except Exception as e:
                    continue

    if not hu_samples:
        print("❌ 未提取到任何有效的 CT 像素数据。")
        return np.array([])
        
    # 合并所有的采样点
    all_hu_data = np.concatenate(hu_samples)
    print("\n✅ 数据提取完毕！")
    print(f"📊 参与统计的有效随机采样像素点总数: {len(all_hu_data):,}")
    return all_hu_data


def plot_hu_distribution(data, output_png_path):
    """
    绘制极具科研美感的 HU 概率密度分布图。
    """
    if len(data) == 0:
        return

    print("⏳ 正在计算统计量与核密度 (KDE)，请稍候...")
    
    # 1. 计算基础统计量
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    median_val = np.median(data)

    # 2. KDE 峰值计算优化 (为了防止千万级数据卡死，专门切一块子集算峰值)
    kde_sample_size = min(len(data), 50000)
    kde_data = np.random.choice(data, size=kde_sample_size, replace=False)
    kde = gaussian_kde(kde_data)
    
    # 构建 X 轴区间来寻找最高点
    x_range = np.linspace(min_val, max_val, 1000)
    kde_values = kde(x_range)
    peak_x = x_range[np.argmax(kde_values)]
    peak_y = np.max(kde_values)

    # 打印终端报告
    print("\n" + "="*45)
    print("📊 CT HU 值统计汇总")
    print("="*45)
    print(f"最小值 (Min)    : {min_val:.2f} HU")
    print(f"最大值 (Max)    : {max_val:.2f} HU")
    print(f"平均值 (Mean)   : {mean_val:.2f} HU")
    print(f"中位值 (Median) : {median_val:.2f} HU")
    print(f"峰值对应 HU     : {peak_x:.2f} HU")
    print("="*45)

    # 3. 开始复刻高颜值绘图
    plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
    ax = plt.gca()
    ax.set_facecolor('#FFFFFF')
    ax.grid(color='#E9ECEF', linestyle='-', linewidth=1, alpha=0.8)

    # 为了画图丝滑，限制传给 seaborn 的最大数据量
    plot_data = data if len(data) < 1000000 else np.random.choice(data, 1000000, replace=False)
    
    # 绘制带填充的 KDE 密度图
    sns.kdeplot(plot_data, color='#3B5B88', linewidth=2, fill=True, alpha=0.8, bw_adjust=0.5, ax=ax)

    # 设置标题和坐标轴
    plt.title("Global Distribution of HU in CT Images", fontsize=16, fontweight='bold', pad=20, color='#2B3A42')
    plt.xlabel('HU value (Hounsfield Unit)', fontsize=12, labelpad=10, color='#2B3A42')
    plt.ylabel('Probability Density', fontsize=12, labelpad=10, color='#2B3A42')
    
    # 动态调整 X 轴和 Y 轴范围
    plt.xlim(min_val - 100, max_val + 100)
    plt.ylim(0, peak_y * 1.25)

    # 辅助注释函数 (完全复刻例图的圆角气泡风格)
    def add_annotation(text, xy, xytext, edge_color, text_color):
        ax.annotate(text, xy=xy, xytext=xytext,
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=edge_color, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="#F8F9FA", ec=edge_color, lw=1.5, alpha=0.9),
                    fontsize=10, color=text_color, fontweight='bold', ha='center', zorder=5)

    x_span = max_val - min_val
    
    # [1] 峰值注释 (红色) - 绝大概率出现在 -1000 (空气) 或 0 (水) 附近
    add_annotation(f"Peak Density\nHU: {peak_x:.1f}\nDensity: {peak_y:.4f}", 
                   xy=(peak_x, peak_y), xytext=(peak_x, peak_y * 1.1), 
                   edge_color="#C0504D", text_color="#C0504D")

    # [2] 最小值注释 (灰色)
    add_annotation(f"Min\nHU: {min_val:.1f}", 
                   xy=(min_val, 0), xytext=(min_val + (x_span * 0.05), peak_y * 0.1), 
                   edge_color="#7F7F7F", text_color="#595959")

    # [3] 平均值注释 (橙色)
    add_annotation(f"Mean\nHU: {mean_val:.1f}", 
                   xy=(mean_val, 0), xytext=(mean_val + (x_span * 0.08), peak_y * 0.05), 
                   edge_color="#E36C0A", text_color="#E36C0A")

    # [4] 中位值注释 (绿色)
    add_annotation(f"Median\nHU: {median_val:.1f}", 
                   xy=(median_val, 0), xytext=(median_val + (x_span * 0.06), peak_y * 0.15), 
                   edge_color="#9BBB59", text_color="#76923C")

    # [5] 最大值注释 (深灰色) -> 代表最致密的骨骼或金属伪影
    add_annotation(f"Max\nHU: {max_val:.1f}", 
                   xy=(max_val, 0), xytext=(max_val - (x_span * 0.1), peak_y * 0.1), 
                   edge_color="#4F6272", text_color="#3A4B56")

    # 美化边框：隐藏上方、右方和左方的黑色实线框
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    print(f"🎉 绘图成功！高颜值分布图已保存至: {output_png_path}")
    plt.close()

# ===============================
# 运行调度
# ===============================
if __name__ == "__main__":
    # 配置参数
    ROOT_DATA_DIR = "../PSMA-PET-CT-Lesions"  # 替换为你的真实根目录
    OUTPUT_IMAGE = "./ct_hu_global_distribution.png"  # 你想要的图片输出路径
    
    # 1. 提取所有 CT 的 HU 值 (自动下采样防 OOM)
    hu_data = get_ct_hu_pixels(ROOT_DATA_DIR, sample_per_slice=2000)
    
    # 2. 统计并画图
    plot_hu_distribution(hu_data, OUTPUT_IMAGE)