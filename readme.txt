0. 剔除naming json里面没有的study ID（因为没有lesions）：delete_folders_not_shown_in_json.py 
1. 先看看数据整体的suv情况，画出分布图。data_statist_info.py。同理，看看整体CT数据的HU值得分布：CT_HU_stat.py

2. 再看看病灶的suv分布情况, lesion_suv_stat.py 会收集所有病灶点的坐标，raw value和suv值，最后保存到csv文档里面。然后可以根据csv文档画图。注意，由于部分pet图有遮盖，会看到很多病灶点为0.带有raw=suv=0的病灶点的seg应该被修改，原本为前景的值变回背景, 用 revise_seg_fg_val.py。然后再产生一次lesion的统计信息。


3. 把dcm都合成成nii文件（pet nii的存储格式已经从float64压缩为float32,而且pet值都是suvbw，ct值都是hu。方便后续的截断和归一化）：dcm2nii.py。注意，后面用check_ctpet_nii_if_z_mached.py检测ct nii和pet nii的z轴顺序是否匹配的时候，发现有6个studyID的z轴起始点差了太多，需要剔除：
[FAIL] 04-16-2002-NA-PETCT whole-body PSMA-15680: max_z_diff>2mm; CT first/last -1435.000/-616.000, PET first/last -1436.000/-766.000; CT min/max -1435.000/-616.000, PET min/max -1436.000/-766.000; diff min/max 1.000/150.000 mm

[FAIL] 05-31-2002-NA-PETCT whole-body PSMA-85128: min_z_diff>2mm; CT first/last -996.790/-46.790, PET first/last -1021.250/-46.789; CT min/max -996.790/-46.790, PET min/max -1021.250/-46.789; diff min/max 24.460/0.001 mm

[FAIL] 05-22-1998-NA-PETCT whole-body PSMA-69841: min_z_diff>2mm; CT first/last -1036.760/-216.760, PET first/last -1073.500/-216.755; CT min/max -1036.760/-216.760, PET min/max -1073.500/-216.755; diff min/max 36.740/0.005 mm

[FAIL] 02-04-2004-NA-PETCT whole-body PSMA-41850: min_z_diff>2mm; CT first/last -352.500/376.500, PET first/last -487.500/376.500; CT min/max -352.500/376.500, PET min/max -487.500/376.500; diff min/max 135.000/0.000 mm

[FAIL] 08-27-2002-NA-PETCT whole-body PSMA-80741: max_z_diff>2mm; CT first/last -960.000/0.000, PET first/last -961.960/12.501; CT min/max -960.000/0.000, PET min/max -961.960/12.501; diff min/max 1.960/12.501 mm

[FAIL] 01-24-2001-NA-PETCT whole-body PSMA-21855: min_z_diff>2mm; CT first/last -999.425/-184.425, PET first/last -1041.165/-184.420; CT min/max -999.425/-184.425, PET min/max -1041.165/-184.420; diff min/max 41.740/0.005 mm

Checked studies: 537; passed: 531; failed: 6


4. 开始进行pet，seg到ct的重采样
ctpet一体机只能保证两者的配准。即ct空间中某个物理坐标[x,y,z]和pet空间中同样数值的物理坐标[x,y,z]，确实是对应着病人的同一个部位。但因为分辨率等问题，还是需要对pet进行插值以得到和ct相同的shape。seg也是同理。重采样后，seg和pet的shape都和ct一样了。seg不全的z坐标对应的帧会被画成全黑（都是背景）。

5. ct截断以及归一化
窗口值需要观察之前对全局hu和病灶hu的值的统计数据。这个版本暂时放到[-1000, 2000]。ct_clip_norm.py做了截断和归一化。

6. pet截断以及归一化
截断值需要观察之前对全局suv和病灶suv做的统计数据。这里放在50. 用pet_clip_norm.py来完成。

7.对nii数据进行augmentation，只做左右和上下的反转。data_aug.py