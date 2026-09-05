# 原生STTrack的DepthTrack Test完整结果

2026-09-06。同一原生STTrack_Vot22基础权重完成50条序列、76,373帧的单起点OPE；跟踪、正式分析和控制器均exit0，03:40:42 CST核实终态。主跟踪耗时6,680.96秒，实际动态模板更新817次。没有新增训练参数、优化器步骤、语言或候选关联头。

| 指标（%） | 当前原生STTrack | 项目最低目标 | 当前减目标（百分点） | 达标 |
| --- | ---: | ---: | ---: | --- |
| Precision | 62.415336 | 65.200000 | -2.784664 | 否 |
| Recall | 62.682004 | 64.900000 | -2.217996 | 否 |
| F-score | 62.548386 | 65.100000 | -2.551614 | 否 |

以上三个数位于同一个全数据集最大F-score工作点，confidence阈值为0.387226。保持原评测器的bounded VOT矩形重叠、100点全局分数阈值网格、逐序列PR后宏平均，再求F最大值。轨迹与置信度为六位小数输出，初始化confidence=1；不增加报告框缩放或测试时后处理。不是三条曲线各自独立取最大，也不是训练开发mean IoU或VOT ROB。

## 模型与证据

基础网络沿用ViT/BSI、TSG固定query窗口、Mamba、Center头、两个RGB-D模板、factor-4/256搜索及原生50步confidence>0.75动态更新。基础权重SHA256仍为`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`，与M39及本次CDTB/full127参照一致。Python推理/分析入口`tools/run_sttrack_native_ope.py`的SHA256为`11748e8e005e9cf0706a1ff24aa59bf9830dcaf23160db5274fa702720a04ec4`；shell控制器`run_dataset.sh`的SHA256为`9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7`；正式评测源SHA256为`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`。历史启动记录中的`runner_sha256`指shell控制器；Python文件单独绑定在spec.source_sha256中，两者不要混用。当前服务器、发布源码及最初准备提交中的Python文件hash一致，没有因本轮分析修改运行源码。

完整推理先保存50份框及50份confidence文件的hash，封存后评测器才读后续GT。原始GT与图像只存放在私有复核目录；公开[download_binding.json](download_binding.json)记录全部框、分数、GT与首帧图像的校验值。其完整性核验不等于独立重跑跟踪。指标[metrics_depthtrack.json](metrics_depthtrack.json)的SHA256为`bd89f02b8be95699cc845dae1a2473cc553a02771ee94093b84504878fa3892d`，跟踪[receipt](depthtrack_receipt.json)为`003c21968c258a301e80ad862542e917e7dc57d5f2f1b56227973d0c1d5c43b4`。

[per_sequence.csv](per_sequence.csv)和[per_sequence.json](per_sequence.json)导出了全部50条序列在全局选定阈值处的P/R/F、有效GT帧数与选中帧数。它们没有逐序列单独优化阈值；全局F由宏平均P/R计算，不等于逐序列F的均值。导出程序[export_per_sequence.py](../export_per_sequence.py)重新验证下载绑定并复用固定重叠口径，聚合结果与正式报告一致。程序只在封存后分析，不参与跟踪、训练或阈值部署。

独立只读[完整结果复核](EXPERIMENT_AUDIT.md)使用原始预测/GT及VOT区域库，从头计算重叠、100点网格和宏平均PR，没有直接调用项目的evaluate_depthtrack_results。所有指标与正式结果差异为0；100份输出、50份GT和50张图像的hash及行数全部匹配，实际有效GT帧为73,389，50条图像尺寸均为640×360。完整性PASS，目标结果与范围WARN；这是GPT同家族Type-A建议性审计，不是跨家族认可。

另行[逐序列复核](EXPERIMENT_AUDIT_PER_SEQUENCE.md)对全部50行的P/R/F、总帧、有效帧与选中帧重新比较，JSON/独立复算、CSV/独立复算及CSV/JSON均0处不一致；全局阈值处共选中74,052帧。复核者没有访问服务器，远端Python源hash由主执行器通过SSH核验，复核者独立验证的是本地/Git及下载数据；其原文“live/local/Git”中的live措辞已在单独[范围澄清](AUDIT_SCOPE_CLARIFICATION.md)中更正，原审计保留不改。

## 对后续实验的影响

原先“DepthTrack已达标”属于历史SRTrack配置，不能继承给这组STTrack权重。本次只补全原生参照，未重新训练，不能作为用户要求的新模块训练完成证据，也不能用它证明M54已有收益。最终同一模型必须同时弥补当前DepthTrack缺口并满足CDTB和完整VOT目标。

M54保持已冻结的63条DepthTrack Train拟合、20 epochs读取头训练、22条训练开发完整递归及原有晋升门；不根据本次公开Test结果改训练标签、阈值或选择权重。通过Train及低22仍不能自动宣称三数据集达标，必须实际评测同一base、读取头权重和运行策略。CDTB与full127在本节首次终态观察时仍未完成，后续分别登记其完整结果。
