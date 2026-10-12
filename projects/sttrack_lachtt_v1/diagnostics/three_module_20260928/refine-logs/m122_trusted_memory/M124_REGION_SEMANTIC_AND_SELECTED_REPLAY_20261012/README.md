# 区域语义研究与最终最佳权重回放准备

仅论文/官方源码研究与诊断源码审查；未执行神经网络回放、未改变运行中训练。

- 当前模型已有 CLIP Surgery 稠密特征与定位负样本监督。首帧有显式 ROI 采样，当前语义分支未直接读取每个候选框的范围。
- RegionCLIP/Alpha-CLIP 提供区域语义编码对照，不保证实例身份、候选几何或长期跟踪，也没有解释本次跳框的唯一原因。完整主源链接见 research-notes/REGION_SEMANTIC_EVIDENCE_PRIMARY_SOURCES_20261012.md。
- replay_m124_selected_best_cdtb_onsets.py 绑定整个 M124 完成后选定的同一 best/final/plan/bank/bundle，在自身历史中只读候选。不能拿旧 CurrentC 的失败状态代替新权重。
- R1 发现 best 元数据被候选索引覆盖、末尾无法写结果；保留原字节后最小修复。R2 源码 PASS/0 阻塞，总体 WARN。语法和 stdlib fixture 通过不等于真实模型运行通过。
- 实际 suite 终态、完整产物链、GPU 释放与运行环境仍需核验。formal_nine_metrics 仅表示指标齐全，不能当达标；若全部九项已达标应按用户要求停止，不追加诊断。
- 回放替代读出不提交状态；GT 在预测/候选封存后读取。候选 oracle、单帧替代与保存轨迹一致性不能宣称完整救回。
- 审查归属 same-family/provisional；actual model/effort UNATTESTED。私有调用 trace 未公开。

当前训练与指标保持 §5.506—§5.508 的记录；没有新增 GPU/NN/训练进度观察。
