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
命令行：python .\ct_clip_norm.py ../PSMA-PET-CT-Lesions --clip-window -1000 2000 --overwrite

6. pet截断以及归一化
截断值需要观察之前对全局suv和病灶suv做的统计数据。这里放在50. 用pet_clip_norm.py来完成。

7.挪动所有nii去final data的对应的子文件夹里。move_rename_nii.py
命令行：python .\pet_clip_norm.py ../PSMA-PET-CT-Lesions --clip-window 0 50 --scale-mode gamma --overwrite

移动硬盘上的数据到这一步为止。本地硬盘接着往下走了。

8.对nii数据进行augmentation，只做左右和上下的反转。data_aug.py 这一步要求7必须完成，因为这一步是对已经归类好的3个最终nii文件夹做处理。

--- 以下都是基于final data数据做的（即数据都重采样，归一化完毕了） ---
1. 新增：考虑掩码数量太少，使得被机器学习到的slice也非常少，决定对seg（这时候已经完成ct重采样了）进行加强。把原本是空的seg slice，找到对应的pet（这时候pet已经重采样，归一化了）上4个最大值的坐标，在seg文件上标2.完成的文件都放在resampled_enhanced_segs，resampled_enhanced_test_segs文件夹里面。
enhanced_seg.py

2. 把2都换成1
seg2to1.py

3. 重新获得加强版掩码idx索引
get_enhanced_foreground_info.py （注意，因为之前的训练数据和test数据分开了，所以得到两个前景点json。需要用merge_train_test_foreground_info.py来合并成新的）
新的前景点json：all_enhanced_foreground_idx_info.json

4. 对数据进行interval划分。8张为一个interval。把划分的信息记录在enhanced_orinal_intv_info.json （也包括test的数据）
gen_intvs_partition_info.py

5. 进行实例划分：把完整的nii文件划分成以interval为单位的nii文件。
gen_nii_partitions.py

6. 乱序缝合, 生成的文件依次放入带有baches关键字的文件夹。这里只包括了training data，因为test data不用乱序。test data的intervals划分就按照all_enhanced_foreground_idx_info.json即可。
stitch_intvs.py






