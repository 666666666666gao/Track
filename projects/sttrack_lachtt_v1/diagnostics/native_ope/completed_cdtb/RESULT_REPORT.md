# 原生STTrack的CDTB完整结果

2026-09-06。原生STTrack_Vot22完成CDTB全部80条序列、101,956帧的单起点OPE；跟踪、正式分析和控制器均exit0，04:11:08 CST实际核实终态。跟踪耗时8,415.20秒，动态模板更新1,614次。使用与M39及本次DepthTrack参照相同的基础权重和原生运行策略，没有新增优化器步骤、语言或学习关联头。

| 指标（%） | 当前原生STTrack | 项目最低目标 | 当前减目标（百分点） | 达标 |
| --- | ---: | ---: | ---: | --- |
| Precision | 69.933113 | 72.900000 | -2.966887 | 否 |
| Recall | 68.067744 | 75.600000 | -7.532256 | 否 |
| F-score | 68.987821 | 74.200000 | -5.212179 | 否 |

三项取自同一个全数据集最大F工作点，confidence阈值为0.580139（100点网格的零起点索引89）。保留固定评测器的bounded VOT矩形重叠、无效GT Special(0)、全局分数阈值网格、逐序列PR宏平均后最大F；不是分别最大化P、R、F。框和置信度保留六位小数、初始化confidence=1，无额外报告框尺度调整。

## 固定模型与文件绑定

底座仍为ViT/BSI、TSG固定query窗口、Mamba、Center头；两个RGB-D模板，factor-4/256搜索，每50步confidence严格大于0.75时动态更新。基础权重SHA256为`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`。Python入口`tools/run_sttrack_native_ope.py`为`11748e8e005e9cf0706a1ff24aa59bf9830dcaf23160db5274fa702720a04ec4`，shell控制器`run_dataset.sh`为`9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7`，评测器源为`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`。execution_binding.runner_sha256指shell，Python另在spec绑定。

正式推理先封存全部160份框与confidence文件，随后分析才读取后续GT。[metrics_cdtb.json](metrics_cdtb.json) SHA256为`a00f5db6e853dd9ac59ef2e174ed7a06e991701e61bfd85badf29ae273c1b731`，[cdtb_receipt.json](cdtb_receipt.json)为`6ea74472df6647ba6e1c3ca50b11bce5a03ed903bf883da62585611a40ed43fe`。本机私有复核副本包含160份预测/分数、80份GT和80张实际首图，全部文件hash、序列顺序与帧数已核对；原始GT和图像不进入公开仓库。[下载绑定](download_binding.json)保存完整校验值。

[per_sequence.csv](per_sequence.csv)与[per_sequence.json](per_sequence.json)列出全部80条序列在同一全局工作点的P/R/F及分母，合计91,300个有效GT帧、91,650个选中帧。由固定[封存后导出程序](../export_per_sequence.py)复算，聚合与正式指标一致。没有为每条序列单独选阈值，整体F也不是逐序列F的平均。

## 独立复核

[完整审计原文](EXPERIMENT_AUDIT.md)从私有原始框、分数、GT及首图使用VOT区域库独立计算重叠和100点宏平均PR，没有调用项目评测函数、重跑推理或使用GPU。聚合指标、全部80行JSON、全部80行CSV及CSV/JSON比较均0处不一致；91,300有效GT帧、91,650选中帧和全局工作点一致。全部160份输出、80份GT和80张图像hash匹配，实际图像边界为31条640×360、23条768×432、23条960×540、3条1920×1080。

未发现具体完整性缺陷，完整性PASS，范围与目标结论WARN，evaluation_type为real_gt。复核者没有连接服务器，只独立验证本地/Git和下载数据；远端退出码、进程终态及源码由主执行器SSH核实。这是同GPT家族Type-A建议性审计，不是跨家族认可或新模块性能证明。原文SHA256为`fb4bcaf262c91ba511af727ee3f29f91f80d40df33c171f62f08af3fc6a2344d`，原文与结构化范围分别封存，不改写复核意见。

## 对后续工作的影响

当前原生STTrack在完整DepthTrack Test和CDTB均未达标。历史SRTrack的达标数值不能与这组权重拼接，也不能用低22改善替代三数据集同一模型验收。本次是补全原生参照，不是M54结果。

M54已于04:09:59 CST按既定条件启动DepthTrack Train采集，随后执行固定20 epochs读取头训练和完整22条训练开发递归。训练标签、预算、checkpoint选择和晋升门均不因本次Test/CDTB结果改变。M54通过开发和低22之后，仍需用同一个base＋读取头权重＋运行策略实际验证全部三个数据集。VOT full127原生任务尚无完成态指标；不在此填入推算分数。
