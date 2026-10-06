# 与候选竞争监督有关的代码参照

2026-10-07核对官方仓库：GitHub API确认OpenSpaceAI/UVLTrack默认分支为master，
先前main分支URL实际404，不据此宣称代码不存在。随后读取以下两份官方源码。

UVLTrack的训练actor取GT中心响应作为正项，框外最高的9个响应作为竞争项，
拼成候选集合并用交叉熵监督；它明确训练正项相对困难背景的优势。
[官方actor：contractive_learning / compute_losses](https://github.com/OpenSpaceAI/UVLTrack/blob/master/lib/train/actors/uvltrack.py)

其Head以归一化搜索特征和动态prompt计算对比分数，再结合中心响应选位置，
说明相对竞争确实参与最终框读出；不能仅照搬分数相乘到本项目，也不能把
这项基础操作作为新的贡献。
[官方Head：contractive_learning / convert2bbox](https://github.com/OpenSpaceAI/UVLTrack/blob/master/lib/models/heads/modality_adaptive_box_head.py)

这是相关机制依据，不是M113退化的因果证据，也不证明移植后一定涨点。
项目M47已真实训练过多正例匹配，M78/M89已有竞争与保持目标，必须保留它们
的负结果和边界。M114先读出当前人工文本final的实际分数；只有测到相对
竞争不足等现象后，才确定下一项训练对照，不预先改正在审核的诊断源码。
