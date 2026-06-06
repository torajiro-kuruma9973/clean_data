import os
import json

def generate_dataset_summary(root_path, output_json_path):
    """
    遍历 PSMA-PET 根目录，统计 project 数量、study 数量，
    并提取切片数量、病灶存在性以及异常文件前缀。
    """
    if not os.path.exists(root_path):
        print(f"❌ 根目录不存在: {root_path}")
        return

    summary_data = {
        "total_projects_num": 0,
        "total_study_num": 0,
        "study_info": {}
    }

    # 1. 获取所有的 Project 文件夹
    projects = [p for p in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, p))]
    summary_data["total_projects_num"] = len(projects)
    print(f"🔍 发现 {len(projects)} 个 Project 文件夹，开始深度扫描...")

    # 2. 遍历每一个 Project
    for project_id in projects:
        project_dir = os.path.join(root_path, project_id)
        studies = [s for s in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, s))]
        
        summary_data["total_study_num"] += len(studies)

        # 3. 遍历每一个 Study
        for study_id in studies:
            study_dir = os.path.join(project_dir, study_id)
            
            ct_slice_num = 0
            pet_slice_num = 0
            lasions_founded = True  # 默认认为有病灶
            special_prefix = False  # 默认没有特殊前缀

            # 4. 遍历 Study 下的模态子文件夹 (CT, PET, Segmentation)
            for leaf_folder in os.listdir(study_dir):
                leaf_path = os.path.join(study_dir, leaf_folder)
                if not os.path.isdir(leaf_path):
                    continue
                
                leaf_upper = leaf_folder.upper()
                
                # 提取当前文件夹下的所有 dcm 文件
                dcm_files = [f for f in os.listdir(leaf_path) if f.lower().endswith('.dcm')]
                
                # 检查是否包含特殊前缀 (只要发现一个不是 '1-' 开头，就标记为 True)
                if not special_prefix:
                    for dcm_file in dcm_files:
                        if not dcm_file.startswith("1-"):
                            special_prefix = True
                            break  # 发现一个异常即可跳出当前文件检查循环
                
                # 归类统计
                if "SEGMENTATION" in leaf_upper:
                    # 如果文件夹名字里明确写了 "NO TUMOR"，则标记为无病灶
                    if "NO TUMOR" in leaf_upper:
                        lasions_founded = False
                        
                elif "PET" in leaf_upper:
                    pet_slice_num = len(dcm_files)
                    
                elif "CT" in leaf_upper:
                    ct_slice_num = len(dcm_files)

            # 5. 记录当前 Study 的统计信息
            # 严格按照你的需求命名键值
            summary_data["study_info"][study_id] = {
                "CT_slice_num": ct_slice_num,
                "PET_slice_num": pet_slice_num,
                "lasions_founded": lasions_founded,
                "special_prefix": special_prefix
            }

    # 6. 保存为 JSON 文件
    try:
        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
            
        print(f"✅ 扫描完成！\n总 Projects: {summary_data['total_projects_num']}")
        print(f"总 Studies: {summary_data['total_study_num']}")
        print(f"统计文件已保存至: {output_json_path}")
        
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")


if __name__ == "__main__":
    my_dataset_root = "PSMA-PET-CT-Lesions"
    my_output_json = "dataset_basic_info.json"
    generate_dataset_summary(my_dataset_root, my_output_json)