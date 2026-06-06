0. 剔除naming json里面没有的study ID（因为没有lesions）：delete_folders_not_shown_in_json.py （注意！这个路径下数据集是已经做过剔除的。然后ct和pet目录下包含合成的nii文件，但都是float 64的）

1. 先看看数据整体的suv情况，画出分布图。data_statist_info.py。

2. 再看看病灶的suv分布情况, lesion_suv_stat.py 会收集所有病灶点的坐标，raw value和suv值，最后保存到csv文档里面。然后可以根据csv文档画图。注意，由于部分pet图有遮盖，会看到很多病灶点为0.带有raw=suv=0的病灶点的seg应该被修改，原本为前景的值变回背景, 用 revise_seg_fg_val.py。然后再产生一次lesion的统计信息。


3. （注意，这条可以掠过，因为0）把dcm都合成成nii文件（注意，这时候都是float 64的）：dcm2nii.py