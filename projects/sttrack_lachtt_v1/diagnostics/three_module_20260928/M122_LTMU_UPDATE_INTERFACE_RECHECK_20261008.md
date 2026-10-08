# M122：LTMU 更新接口复核（2026-10-08）

[根README作者信息][authors]列出该CVPR2020工作的作者；[CVF论文页](https://openaccess.thecvf.com/content_CVPR_2020/html/Dai_High-Performance_Long-Term_Tracking_With_Meta-Updater_CVPR_2020_paper.html)仅作出处链接，本次打开403，未核对正文。固定 commit：`121ddd07914a8577e336fed24fa47763b6780fdd`（git ls-remote实测）。确有公开实现；此处核对的 DiMP_LTMU 被[README明确称为简化版][version]，其全图恢复使用 GlobalTrack。没有重试403页面，未做完整论文视觉审核。

- **局部位置**：[track_updater][local]返回 xywh、响应图、分类特征、尺度/采样位置、flag、原始分数，分类器更新留给外层。
- **独立验证**：[MDNet fc6类1][verify]评估候选框；[metricNet初帧锚L2距离][metric]供MU外观输入。它们有独立模型。
- **更新资格**：[MU实码][gate]取最近20帧归一化xyxy、最大响应、外观距离及19×19响应图，二类logit比较后调用update_classifier。[参数][opts]规定20帧；[运行配置][config]从start_frame=200启用，此前按native flag；启用后MU覆盖flag判定。**局部更新发生在MDNet验证之前**（[调用顺序][order]），不可称为验证后安全写入。实际更新涉及[分类记忆与滤波器优化][write]，不是模板替换。
- **重新检测**：[GlobalTrack接口][global]以初框初始化query，再对图像update返回检测结果；[外层][recover]取分数前10、转xywh，[MDNet复核][recover2]后最佳验证分数>0且累计count≥5才重设局部位置/尺寸。此接口不能据论文替写成SiamRPN。

**监督定义**：[DiMP_MU生成器][labelgen]用同帧GT与预测框计算IoU（小尺寸=-1、NaN=0），[记录器][record]写入索引5（第6列）。[采样][sample]只看历史窗口最后一行：>0.5正、==0负，其余末帧不采；[交叉熵][loss]训练二分类。因此是历史窗口末帧，即决策当前帧质量；不是整条视频末帧或写/不写后的未来效用。[训练输入][feed]仅取[零基索引0–3、6、7][opts]，排除帧号4与GT IoU 5；线上[拼接几何、响应、外观距离][gate]，已读MU决策路径没有GT IoU输入。M122部署只接运行可观测历史，不传GT诊断字段。本项目冻结任务边界为无效GT=未知（每pass16426帧）；NaN=0仅描述LTMU原实现，不能照搬为身份负例或物理不存在，此边界与未来效用监督分开。注意简化DiMP_LTMU的[返回IoU固定0][zero]，不能直接用其记录器生成有效正例。

**最小接线提案（未执行）**：本地唯一冻结源SHA256=`0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`；[103–114行][m122]保存selected_feature并记录quality/observation，写模板仍取native间隔与阈值。固定波次闭合后，可保留A+B位置输出和原模板底座，新增verify(image,selected_box,anchor)与allow_write(history,verification)，接到105–109行写入点；跨区域恢复另接propose→verify→重设搜索位置，不能仅靠quality日志实现C。

直接导入受[TF1.15/PyTorch1.4环境][deps]、19×19输入、DiMP分类记忆接口以及[MDNet/metricNet/GlobalTrack/SiamMask权重][weights]约束；未核验权重可得性或编译运行。学习更新资格、独立验证与全图恢复已有先例；如主张未来写入效用，须另建动作监督及公平对照，不能把LTMU同帧IoU标签当作因果证据。未移植、未测涨点、未触碰32/52文件门。

[authors]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/README.md#L35-L44
[version]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/README.md#L6-L15
[local]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/pytracking/tracker/dimp/dimp.py#L156-L218
[verify]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L166-L226
[metric]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L313-L339
[gate]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L373-L415
[opts]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/meta_updater/tcopt.py#L32-L41
[config]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/run_tracker.py#L16-L25
[order]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L468-L475
[write]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/pytracking/tracker/dimp/dimp.py#L594-L626
[global]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Global_Track/Global_tracker/global_track.py#L46-L76
[recover]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L147-L164
[recover2]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L490-L506
[labelgen]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_MU/Dimp.py#L105-L129
[record]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_MU/run_tracker.py#L157-L170
[sample]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/meta_updater/train_lstm.py#L58-L80
[loss]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/meta_updater/train_lstm.py#L153-L178
[zero]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L528-L529
[m122]: https://github.com/666666666666gao/Track/blob/6fbcc764530da9b223cb165b0d0ad1e6018ab410/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py#L103
[deps]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/requirements.txt#L71-L79
[weights]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/README.md#L42-L51

[feed]: https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/meta_updater/train_lstm.py#L231-L240
