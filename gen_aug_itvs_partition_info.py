import json
import copy

def expand_json_with_offsets(input_path, output_path):
    offsets = [2000, 4000, 6000, 8000]
    
    with open(input_path, "r") as f:
        data = json.load(f)
    
    new_data = copy.deepcopy(data)  # 保留原始内容
    
    for key, value in data.items():
        for offset in offsets:
            new_key = str(int(key) + offset)
            new_data[new_key] = copy.deepcopy(value)
    
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2)
    
    print(f"原始 item 数：{len(data)}")
    print(f"新增 item 数：{len(data) * len(offsets)}")
    print(f"共计 item 数：{len(new_data)}")
    print(f"已保存到：{output_path}")

if __name__ == "__main__":
    source = "new_version_starters_ends_info.json"
    dest = "aug_new_version_starters_ends_info.json"
    expand_json_with_offsets(source, dest)