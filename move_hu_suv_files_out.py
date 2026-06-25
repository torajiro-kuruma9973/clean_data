import json
import shutil
from pathlib import Path


def reorganize_nii(
    root_dir: str,
    mapping_json: str,
    ct_target_dir: str,
    pet_target_dir: str,
):
    """
    遍历 root_dir，找到每个 study 下 CT 文件夹里的 ct.nii.gz 和
    PET 文件夹里的 pet_resampled.nii.gz，按 mapping_json 的规则
    mv（移动）到指定目录并重命名。

    目录结构假设：
        root_dir/
            <projectID>/
                <studyID>/
                    xxxx-CT-xxx/           -> ct.nii.gz
                    xxxx-PET-xxx/          -> pet_resampled.nii.gz
                    xxxx-Segmentation-xxx/ （忽略，即使名字里含 ct）

    mapping_json 形如：
        { "1.nii.gz": "01-01-2003-NA-PETCT whole-body PSMA-65095", ... }
    即 “目标文件名 -> studyID 名”。函数内部会反向查表。

    参数:
        root_dir:       根目录
        mapping_json:   命名映射 json 文件路径
        ct_target_dir:  CT 文件移动到的目标目录（如 ct_hu_clipped_no_normed）
        pet_target_dir: PET 文件移动到的目标目录（如 pet_suv_clipped_no_normed）
    """
    root = Path(root_dir)
    ct_out = Path(ct_target_dir)
    pet_out = Path(pet_target_dir)
    ct_out.mkdir(parents=True, exist_ok=True)
    pet_out.mkdir(parents=True, exist_ok=True)

    # 读取 json，构造反向映射： studyID 名 -> 目标文件名
    with open(mapping_json, "r", encoding="utf-8") as f:
        name_to_study = json.load(f)
    study_to_name = {study: fname for fname, study in name_to_study.items()}

    def find_sub(study_path: Path, keyword: str):
        """在 study 文件夹下找含 keyword 但非 Segmentation 的子文件夹。"""
        hits = [
            d for d in study_path.iterdir()
            if d.is_dir()
            and "segmentation" not in d.name.lower()
            and keyword in d.name.upper()
        ]
        if len(hits) > 1:
            print(f"[WARN] {study_path} 下找到多个 {keyword} 文件夹: "
                  f"{[h.name for h in hits]}，取第一个")
        return hits[0] if hits else None

    def move_one(study_path: Path, keyword: str, src_filename: str,
                 out_dir: Path, target_filename: str):
        sub = find_sub(study_path, keyword)
        if sub is None:
            print(f"[WARN] {study_path} 下未找到 {keyword} 文件夹，跳过")
            return
        src = sub / src_filename
        if not src.is_file():
            print(f"[WARN] 未找到 {src}，跳过")
            return
        dst = out_dir / target_filename
        if dst.exists():
            print(f"[WARN] 目标已存在将被覆盖: {dst}")
        shutil.move(str(src), str(dst))
        print(f"[OK] {src}  ->  {dst}")

    # 遍历 project / study
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        for study in sorted(s for s in project.iterdir() if s.is_dir()):
            study_id = study.name
            target_filename = study_to_name.get(study_id)
            if target_filename is None:
                print(f"[WARN] json 中没有 studyID '{study_id}' 的映射，跳过 "
                      f"(project={project.name})")
                continue

            move_one(study, "CT", "ct.nii.gz", ct_out, target_filename)
            move_one(study, "PET", "pet_resampled.nii.gz", pet_out, target_filename)


if __name__ == "__main__":
    # ---------- sample code ----------
    reorganize_nii(
        root_dir="../PSMA-PET-CT-Lesions",                       # 参数1：根目录
        mapping_json="name_mapping.json",           # 参数2：命名映射 json
        ct_target_dir="../ct_clipped_no_normed",     # CT 目标目录
        pet_target_dir="../suv_clipped_no_normed",  # PET 目标目录
    )