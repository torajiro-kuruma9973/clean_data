import os
import json
import pydicom
from pydicom.multival import MultiValue
from pydicom.sequence import Sequence
from pydicom.valuerep import PersonName

def sanitize_dicom_value(val):
    """
    递归清洗器：将 pydicom 特殊的数据类型彻底转换为 Python 原生类型，
    确保 json.dump 不会崩溃。
    """
    if isinstance(val, (int, float, str)):
        return val
    elif isinstance(val, MultiValue):
        # 处理类似 [0.5, 0.5] 的坐标或间距数组
        return [sanitize_dicom_value(v) for v in val]
    elif isinstance(val, Sequence):
        # 处理 DICOM 内部的嵌套序列 (例如 SQ 类型的 Tag)
        return [dicom_to_dict(item) for item in val]
    elif isinstance(val, PersonName):
        # 处理病人名字，转为标准字符串
        return str(val)
    elif isinstance(val, bytes):
        # 遇到二进制数据（非图像的私有二进制 tag），直接跳过或替换为提示语
        return "<Binary Data>"
    elif val is None:
        return None
    else:
        # 万能兜底：其余的奇葩类型全部强转为字符串
        return str(val)

def dicom_to_dict(ds):
    """
    将 pydicom 的 Dataset 对象转换为纯净的 Python 字典。
    """
    dcm_dict = {}
    for elem in ds:
        # 坚决跳过 PixelData，防止巨大的二进制撑爆 JSON
        if elem.keyword == 'PixelData':
            continue
            
        # 如果这个 Tag 有官方英文名 (如 PatientName)，就用英文名
        # 如果是私有 Tag 没有名字，就使用它的 Hex 编码 (如 (0008, 0010))
        key = elem.keyword if elem.keyword else str(elem.tag)
        
        # 清洗 value
        dcm_dict[key] = sanitize_dicom_value(elem.value)
        
    return dcm_dict

def extract_dcm_header_to_json(dcm_path, output_json_path):
    """
    读取单个 DCM 文件的 Header 信息，并以 JSON 形式保存。
    """
    if not os.path.exists(dcm_path):
        print(f"❌ 找不到 DCM 文件: {dcm_path}")
        return {}

    try:
        # 🛡️ 核心性能优化：stop_before_pixels=True
        # 这个参数告诉 pydicom 读完 Header 马上停下，绝对不要去解析庞大的图像像素
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        
        # 转换为纯净的字典
        header_dict = dicom_to_dict(ds)

    except Exception as e:
        print(f"❌ 解析 DICOM 失败 [{dcm_path}]: {e}")
        return {}

    # 保存为 JSON
    try:
        out_dir = os.path.dirname(output_json_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 使用我们之前说过的标准参数
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(header_dict, f, indent=4, ensure_ascii=False)
            
        print(f"✅ Header 已成功导出至: {output_json_path}")
        
    except Exception as e:
        print(f"❌ 保存 JSON 失败: {e}")

    return header_dict

import os
import re
import pydicom

def get_dicom_timestamp(ds):
    """
    从 DICOM Header 中提取最精确的物理扫描时间，转换为便于比较的浮点数。
    """
    # 优先级：Acquisition (采集时间) > Content (内容生成时间) > InstanceCreation (文件创建时间)
    date = ds.get('AcquisitionDate', ds.get('ContentDate', ds.get('InstanceCreationDate', '19700101')))
    time = ds.get('AcquisitionTime', ds.get('ContentTime', ds.get('InstanceCreationTime', '000000.0')))
    
    # 清洗数据，防止出现空字符串或多余空格
    date = str(date).strip() if date else '19700101'
    time = str(time).strip() if time else '000000.0'
    
    # 拼装成 YYYYMMDDHHMMSS.FFFFFF 格式的浮点数
    try:
        return float(f"{date}{time}")
    except ValueError:
        return 0.0

# make sure the dcm files' are timing-ordered.
# e.g. 1-100.dcm must be prior to 2-3.dcm
def verify_dcm_chronology(root_dir):
    """
    遍历目录下的所有 dcm 文件，按照文件名数字排序，并验证时间戳是否严格递增。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    # 1. 收集并解析文件
    dcm_files_info = []
    
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith('.dcm'):
                continue
                
            # 提取文件名中的所有数字，比如 "1-10.dcm" -> [1, 10]
            # "PET_2-3.dcm" -> [2, 3]
            numbers = re.findall(r'\d+', filename)
            
            # 过滤掉没有数字的文件 (比如纯英文的 Segmentation.dcm)
            if not numbers:
                continue
                
            # 转换为整数元组作为排序键 (1, 100) > (1, 2) ; (2, 3) > (1, 100)
            sort_key = tuple(int(n) for n in numbers)
            filepath = os.path.join(root, filename)
            
            dcm_files_info.append({
                'filepath': filepath,
                'filename': filename,
                'sort_key': sort_key
            })

    if not dcm_files_info:
        print("⚠️ 未找到包含数字排序命名的 DCM 文件。")
        return

    # 2. 按照解析出的数字键进行自然排序
    dcm_files_info.sort(key=lambda x: x['sort_key'])
    print(f"🔍 共找到 {len(dcm_files_info)} 个有效 DCM 文件，开始时间戳校验...")

    # 3. 遍历校验时间顺序
    violations = []
    previous_time = -1.0
    previous_name = ""

    for i, file_info in enumerate(dcm_files_info):
        filepath = file_info['filepath']
        current_name = file_info['filename']
        
        try:
            # 极速读取模式
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            current_time = get_dicom_timestamp(ds)
            
            # 校验逻辑：当前文件的时间必须 >= 上一个文件的时间
            # 注意：同一个床位/同一秒采集的切片时间戳可能完全相同，所以允许等于
            if i > 0 and current_time < previous_time:
                violations.append({
                    'prev_file': previous_name,
                    'prev_time': previous_time,
                    'curr_file': current_name,
                    'curr_time': current_time
                })
                
            previous_time = current_time
            previous_name = current_name
            
        except Exception as e:
            print(f"⚠️ 读取文件失败 [{current_name}]: {e}")

    # 4. 打印最终报告
    print("\n" + "="*40)
    if not violations:
        print("✅ 校验通过！所有文件严格按照物理时间顺序命名排列。")
    else:
        print(f"❌ 警告：发现 {len(violations)} 处时间顺序倒挂！")
        for v in violations[:10]: # 最多打印前 10 个违规项防止刷屏
            print(f"  -> 冲突: {v['curr_file']} (时间: {v['curr_time']}) 竟然比前面的 {v['prev_file']} (时间: {v['prev_time']}) 还要早！")
        
        if len(violations) > 10:
            print(f"  ... 还有 {len(violations) - 10} 处错误未显示。")
    print("="*40)


def get_z_location(ds):
    """
    优先使用 ImagePositionPatient[2] 获取绝对空间 Z 坐标。
    它是世界坐标系下切片左上角像素的 Z 轴位置，比 SliceLocation 更可靠。
    """
    if 'ImagePositionPatient' in ds and len(ds.ImagePositionPatient) >= 3:
        return float(ds.ImagePositionPatient[2])
    elif 'SliceLocation' in ds:
        return float(ds.SliceLocation)
    else:
        raise ValueError("缺失空间 Z 轴信息")

def verify_spatial_sorting(root_dir):
    """
    遍历目录，提取所有 dcm 文件的 Z 轴坐标，并验证：
    1. 同一前缀内（如 1-x.dcm），随着 x 增大，Z 坐标必须严格单调递增或递减。
    2. 不同前缀间（如 2-x vs 1-y），前缀 2 的整体 Z 坐标范围不能与前缀 1 发生重叠（全大于或全小于）。
    """
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    # 数据结构: { prefix_int: [ (index_int, z_float, filename), ... ] }
    groups = {}
    
    # 1. 扫描文件并提取 Z 轴信息
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.lower().endswith('.dcm'):
                continue
            
            # 正则匹配形如 "1-5.dcm", "2-010.dcm" (允许前后带些其他字符)
            match = re.search(r'^(\d+)-(\d+).*\.dcm$', filename, re.IGNORECASE)
            if not match:
                continue
                
            prefix = int(match.group(1))
            index = int(match.group(2))
            filepath = os.path.join(root, filename)
            
            try:
                # 极速模式：只读 Header 不读像素
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                z_loc = get_z_location(ds)
                
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((index, z_loc, filename))
                
            except Exception as e:
                print(f"⚠️ 无法读取或解析 {filename}: {e}")

    if not groups:
        print("⚠️ 未找到符合 '前缀-序号.dcm' 命名规则的有效文件。")
        return

    violations = []
    group_z_ranges = {}  # 记录每个组的 [min_z, max_z]
    
    print(f"🔍 开始校验空间顺序，共发现 {len(groups)} 个前缀组...")

    # ==========================================
    # 2. 校验同前缀内部的空间单调性 (Intra-prefix)
    # ==========================================
    for prefix, items in groups.items():
        # 按照文件名里的序号从小到大排序
        items.sort(key=lambda x: x[0])
        
        if len(items) == 0:
            continue
        elif len(items) == 1:
            group_z_ranges[prefix] = (items[0][1], items[0][1])
            continue
            
        # 寻找初始的空间扫描方向 (1 代表 Z 轴增大，-1 代表 Z 轴减小)
        expected_dir = 0
        for i in range(1, len(items)):
            diff = items[i][1] - items[i-1][1]
            if abs(diff) > 1e-4:  # 排除浮点数精度极小误差
                expected_dir = 1 if diff > 0 else -1
                break
                
        min_z = max_z = items[0][1]
        
        # 逐层校验是否一直保持这个方向
        for i in range(1, len(items)):
            curr_name, prev_name = items[i][2], items[i-1][2]
            curr_z, prev_z = items[i][1], items[i-1][1]
            
            min_z = min(min_z, curr_z)
            max_z = max(max_z, curr_z)
            
            diff = curr_z - prev_z
            actual_dir = 1 if diff > 1e-4 else (-1 if diff < -1e-4 else 0)
            
            if actual_dir != 0 and actual_dir != expected_dir:
                violations.append(f"[内部冲突] 前缀 {prefix}: {curr_name}(Z={curr_z:.2f}) 与 {prev_name}(Z={prev_z:.2f}) 的空间顺序倒挂！")
                
        # 记录当前前缀组的绝对空间范围
        group_z_ranges[prefix] = (min_z, max_z)

    # ==========================================
    # 3. 校验跨前缀之间的空间界限 (Inter-prefix)
    # ==========================================
    sorted_prefixes = sorted(group_z_ranges.keys())
    
    for i in range(1, len(sorted_prefixes)):
        prev_p = sorted_prefixes[i-1]
        curr_p = sorted_prefixes[i]
        
        prev_min, prev_max = group_z_ranges[prev_p]
        curr_min, curr_max = group_z_ranges[curr_p]
        
        # 校验：两个组的空间区间不能有任何交集
        # 条件 A: current 组完全在 previous 组之上 (curr_min > prev_max)
        # 条件 B: current 组完全在 previous 组之下 (curr_max < prev_min)
        if not (curr_min > prev_max or curr_max < prev_min):
            violations.append(
                f"[跨组冲突] 前缀 {curr_p} 的空间范围 [{curr_min:.2f}, {curr_max:.2f}] "
                f"与前缀 {prev_p} 的空间范围 [{prev_min:.2f}, {prev_max:.2f}] 发生重叠混淆！"
            )

    # ==========================================
    # 4. 打印校验报告
    # ==========================================
    print("\n" + "="*50)
    if not violations:
        print("✅ 空间校验完美通过！")
        print(" -> 同一前缀内的序号与其 Z 轴绝对空间顺序完全一致。")
        print(" -> 不同前缀组（Bed Positions）的空间范围互相独立，没有发生穿插。")
    else:
        print(f"❌ 警告：发现 {len(violations)} 处空间顺序异常：")
        for v in violations[:15]:
            print("  ", v)
        if len(violations) > 15:
            print(f"  ... 还有 {len(violations) - 15} 处错误未显示。")
    print("="*50)

# --- 运行示例 ---
# my_target_folder = "test_dicom_folder"
# verify_spatial_sorting(my_target_folder)


if __name__ == "__main__":
    # my_dcm = "2-011.dcm"
    # my_json = "2-011_head.json"
    # header_info = extract_dcm_header_to_json(my_dcm, my_json)
    my_test_dir = "D:\\D_Work\\Datasets\\NBIA-PSMA-manifest-1772126181965\\PSMA-PET-CT-Lesions\\PSMA_01a52e26ce5b5e26\\10-15-1999-NA-PETCT whole-body PSMA-89953\\102.000000-PET-72841"
    #verify_dcm_chronology(my_test_dir)
    verify_spatial_sorting(my_test_dir)