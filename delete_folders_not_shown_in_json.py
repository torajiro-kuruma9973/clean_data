import os
import json
import shutil

def move_unmatched_studies(root_dir, json_path, target_dir):
    """
    遍历根目录下的 Study ID 文件夹，如果其名称没有出现在 JSON 文件的 values 中，
    则打印该 Study ID 并将其移动到目标路径。
    
    参数:
        root_dir (str): 原始 DICOM 数据集的根目录 (包含 Project ID 文件夹)
        json_path (str): 包含 Study ID 映射的 JSON 文件路径
        target_dir (str): 未匹配 Study 的目标存放路径
    """
    print(f"⏳ 步骤 1: 正在读取 JSON 映射表...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 文件失败: {e}")
        return

    # 🌟 核心优化：提取所有的 value 并转为集合 (Set)
    # 这样后续在比对 Study ID 时，查找速度会从 O(N) 提升到 O(1)
    valid_studies = set(mapping_dict.values())
    print(f"   ➤ 成功加载了 {len(valid_studies)} 个合法的 Study ID。")

    print(f"\n⏳ 步骤 2: 准备目标文件夹...")
    os.makedirs(target_dir, exist_ok=True)
    print(f"   ➤ 目标路径已就绪: {target_dir}")

    print(f"\n🚀 步骤 3: 开始遍历目录树进行比对...\n")
    moved_count = 0
    
    # 遍历第一层：Project ID 文件夹 (例如 PSMA_0179419e...)
    for project_folder in os.listdir(root_dir):
        project_path = os.path.join(root_dir, project_folder)
        
        # 忽略文件（如 .DS_Store），只处理文件夹
        if not os.path.isdir(project_path):
            continue
            
        # 遍历第二层：Study ID 文件夹 (例如 05-03-2002-NA-PETCT...)
        for study_folder in os.listdir(project_path):
            study_path = os.path.join(project_path, study_folder)
            
            if not os.path.isdir(study_path):
                continue
                
            # 💡 判断逻辑：如果在 JSON 的 values 中找到了，就 continue 跳过
            if study_folder in valid_studies:
                continue
                
            # 如果没有找到，说明这是多余的/未匹配的数据
            print(f"⚠️ 发现未匹配的 Study ID: {study_folder}")
            
            # 构造移动的目标绝对路径
            target_study_path = os.path.join(target_dir, study_folder)
            
            try:
                # 将整个 Study 文件夹物理移动到目标路径
                shutil.move(study_path, target_study_path)
                moved_count += 1
                print(f"   ✅ 已成功移动至目标文件夹。")
            except Exception as e:
                print(f"   ❌ 移动失败: {str(e)}")

    print(f"\n🎉 任务圆满完成！")
    print(f"📊 统计结果: 共找出并移动了 {moved_count} 个未匹配的 Study 文件夹。")

def count_study_id_folders(root_dir): # count the num of studyID folders
    """
    遍历 PSMA 数据集根目录，严格按照两层结构向下寻找，
    统计并打印 Study ID 文件夹的总个数。
    
    参数:
        root_dir (str): 数据集的根目录
    """
    if not os.path.exists(root_dir):
        print(f"❌ 错误：找不到指定的根目录 '{root_dir}'")
        return 0

    study_count = 0
    
    print(f"🚀 开始扫描根目录: {root_dir} ...")
    
    # 遍历第 1 层：Project ID 文件夹 (例如 PSMA_0179419e...)
    for project_folder in os.listdir(root_dir):
        project_path = os.path.join(root_dir, project_folder)
        
        # 忽略文件（如 .DS_Store 或 .txt），只进入文件夹
        if os.path.isdir(project_path):
            
            # 遍历第 2 层：Study ID 文件夹 (例如 05-03-2002-NA-PETCT...)
            for study_folder in os.listdir(project_path):
                study_path = os.path.join(project_path, study_folder)
                
                # 确认这是一个文件夹后，计数器加 1
                if os.path.isdir(study_path):
                    study_count += 1

    print(f"🎉 扫描完毕！")
    print(f"📊 总计发现 Study ID 文件夹个数: {study_count}")
    
    return study_count


# ===============================
# 运行示例
# ===============================
if __name__ == "__main__":
    # 参数 1: 数据集根目录
    my_root_dir = "../PSMA-PET-CT-Lesions"  
    
    # 参数 2: 你的 JSON 映射文件
    my_json_path = "./name_mapping.json"  
    
    # 参数 3: 打算用来存放废弃/多余 Study 的文件夹
    my_target_dir = "../Unmatched_Studies_Archive"  
    
    # 执行函数
    move_unmatched_studies(my_root_dir, my_json_path, my_target_dir)

    n = count_study_id_folders(my_root_dir) # 537 clean cases, correct!
    print(n)