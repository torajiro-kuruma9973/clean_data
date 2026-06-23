import json
import matplotlib.pyplot as plt
from collections import Counter


def random_sample_with_min_gap(points, n, b):
    if n <= 0:
        return []
    candidates = list(points)
    selected = []
    for p in candidates:
        if all(abs(p - q) > b for q in selected):
            selected.append(p)
            if len(selected) >= n:
                break
    return selected


def analyze_sample_distribution(json_file, n, output_png):
    with open(json_file, 'r') as f:
        data = json.load(f)

    lengths = []
    for outer_key, inner_dict in data.items():
        points = list(inner_dict.values())
        result = random_sample_with_min_gap(points, n, b=7)
        lengths.append(len(result))

    length_counts = Counter(lengths)
    x = sorted(length_counts.keys())
    y = [length_counts[k] for k in x]

    le10 = sum(count for length, count in length_counts.items() if length <= 10)
    gt10 = sum(count for length, count in length_counts.items() if length > 10)
    print(f"y轴（count）中，length <= 10 的点的数量：{le10}")
    print(f"y轴（count）中，length >  10 的点的数量：{gt10}")

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color='steelblue', edgecolor='black')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.title('Distribution of Sample Lengths')
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"图已保存至 {output_png}")

# ==========================================
# 🚀 使用示例
# ==========================================
if __name__ == "__main__":
    
    analyze_sample_distribution("foreground_points_frame_idx_dict.json", n=10000, output_png="itv_dist.png")
