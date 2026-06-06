import pandas as pd
import os

def export_zero_value_records(csv_path, txt_output_path):
    """
    读取 CSV 文件，找出 raw_value=0 且 suv_value=0 的所有行。
    打印这些记录所在的 Excel 行号，并将完整记录写入指定的 txt 文档。
    
    参数:
    - csv_path: 输入的 CSV 文件路径
    - txt_output_path: 输出的 TXT 文档路径
    """
    try:
        # 1. 读取 CSV 文件
        print(f"🚀 正在读取数据: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 2. 检查必要的字段是否存在
        if 'raw_value' not in df.columns or 'suv_value' not in df.columns:
            return "❌ 错误：CSV 文件中找不到 'raw_value' 或 'suv_value' 字段。"
            
        # 3. 筛选 raw_value == 0 且 suv_value == 0 的数据
        # 使用 0.0 防止浮点数匹配问题
        zero_mask = (df['raw_value'] == 0.0) & (df['suv_value'] == 0.0)
        anomaly_df = df[zero_mask]
        
        # 4. 获取行号 (Excel真实行号 = 原始索引 + 2)
        # 因为 Python 索引从 0 开始，且 CSV 第一行是表头
        indices = anomaly_df.index.tolist()
        excel_rows = [i + 2 for i in indices]
        
        # 5. 在控制台直观地打印行号结果
        print("-" * 50)
        print("🕵️ 零值异常数据筛查结果 (raw=0 且 suv=0)")
        print("-" * 50)
        
        if not excel_rows:
            print("✨ 完美！没有找到符合条件的双零值记录，数据很干净。")
            print("-" * 50)
            return []
            
        print(f"📍 共找到 {len(excel_rows)} 条异常记录！")
        print(f"📍 它们分别位于 Excel 的以下行数：")
        print(f"   {excel_rows}")
        print("-" * 50)
        
        # 6. 将完整的记录写入 TXT 文档
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(f"==================================================\n")
            f.write(f" 异常数据报告：包含 raw_value=0 且 suv_value=0 的记录\n")
            f.write(f" 发现异常记录总数: {len(anomaly_df)} 条\n")
            f.write(f"==================================================\n\n")
            
            # 遍历每一行异常数据，格式化写入 TXT
            for original_index, row in anomaly_df.iterrows():
                excel_row_num = original_index + 2
                f.write(f"▶ [Excel 行号: {excel_row_num}]\n")
                
                # 将该行的所有字段转为字典并逐一打印，方便阅读
                for col_name, val in row.items():
                    f.write(f"    - {col_name}: {val}\n")
                f.write("\n")  # 每条记录之间留空行
                
        print(f"💾 完整的异常记录详情已成功导出至: {txt_output_path}\n")
        
        return excel_rows
        
    except FileNotFoundError:
        return f"❌ 错误：找不到路径为 '{csv_path}' 的文件，请检查路径是否正确。"
    except Exception as e:
        return f"❌ 发生未知错误：{e}"

# ===============================
# 使用示例
# ===============================
if __name__ == "__main__":
    # 请替换为你的实际文件路径
    INPUT_CSV = "all_psma_lesion_points.csv" 
    OUTPUT_TXT = "zero_value_records_details.txt"
    
    # 运行函数
    export_zero_value_records(INPUT_CSV, OUTPUT_TXT)