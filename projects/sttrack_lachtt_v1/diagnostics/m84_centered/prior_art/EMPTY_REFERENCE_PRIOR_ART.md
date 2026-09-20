# M84 空词参照差分与 CLIP Surgery：机制核查

核查日期：2026-09-20。仅核查公开一手资料及本地 M84 代码/方案；未访问远端、修改实验或启动任务。官方仓库固定到 [d4696d47f49cfe70f49140afe5eb94f94c5f59bc](https://github.com/xmed-lab/CLIP_Surgery/commit/d4696d47f49cfe70f49140afe5eb94f94c5f59bc)。

**结论：空字符串参照相减已有非常直接的先例，不能作为 M84 独创点。** M84 的具体实现与 CLIP Surgery 不等同，但“同一输入下减去无内容文本响应”属于明显相近的中心化思路；实现差异本身尚不能证明研究新颖性。

## 1. 官方单文本分支：确实用了空字符串

作者 Yi Li 在 [2023-04-18 回复](https://github.com/xmed-lab/CLIP_Surgery/issues/2#issuecomment-1512353799)区分了标签集合和单文本情形，并为后者给出空字符串参照。现有 [demo.py 第117–132行](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/demo.py#L117-L132)仍实现该路径：用 `encode_text_with_prompt_ensemble(model, [""], device)` 得到参照，再调用 feature surgery；整个演示处于 `torch.no_grad()`。

注意，参照是把空字符串填入多个提示模板、分别编码归一化、取均值后再次归一化；不是零向量，也不等同于单次编码裸空字符串。[文本编码实现](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/clip/clip.py#L245-L269)

[clip_feature_surgery 第289–290行](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/clip/clip.py#L287-L308)直接计算：

\[
S_q=V\,(T_q-T_\emptyset)^\top
    =VT_q^\top-VT_\emptyset^\top.
\]

这里先减的是文本特征，最终输出每个图像 token 对文本的标量相似度；线性展开后也等于减去空文本相似度图。差分文本不再归一化，因而不宜将结果直接称作两个单位向量的普通余弦相似度。该代码没有把学习出的稠密增量加回跟踪特征。

## 2. 标签集合分支：另一种参照估计

[论文 v2 §3.3，式8–11](https://arxiv.org/html/2304.05653v2#S3.SS3)先构造图文逐通道乘积 `Fm[i,k,c]=V[i,c]T[k,c]`，根据全局图像与各文本的相似度获得权重 `w[k]`，以跨类别加权均值估计冗余项 `R[i,c]=mean_k(w[k]Fm[i,k,c])`，再对 `Fm-R` 沿通道求和。此处参照来自当前标签集合，不能笼统写成全方法都减空字符串。

**论文与固定代码还有一处细节差别：** [代码第300–306行](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/clip/clip.py#L300-L306)先原地 `feats *= w`，然后求均值并相减；实际输出是 `sum_c(w[k]Fm[i,k,c]-R[i,c])`，目标项也带权。论文式11的目标项没有这一步权重。复述应分别标明论文公式和该版本实现。该差别不影响上述空参照分支的直接证据。

CLIP Surgery 同时包含架构修改，但论文明确使用预训练 CLIP 参数、在推理阶段实施，不额外训练；不能把整篇方法缩写成仅一次相减。[论文 §4.1](https://arxiv.org/html/2304.05653v2#S4.SS1) 最新 arXiv 版本为 2024-09-16 v2，题名为 *A Closer Look at the Explainability of Contrastive Language-Image Pre-training*；官方仓库给出 Pattern Recognition 162 (2025), 111409 的期刊信息。[arXiv 元数据](https://arxiv.org/abs/2304.05653v2)；[官方书目信息](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/README.md)

## 3. M84 的具体差别与可声称范围

本地 `centered_semantic_adapter.py` 调用同一个非线性 `SemanticSpatialAdapter` 两次，以零 fused 取得两份增量，计算：

\[
F'=F+D_\theta(x,q)-D_\theta(x,q_\emptyset).
\]

`x` 包含 RGB、深度、初始参考等当前输入；空参照向量广播到相同文本槽位，保留相同 mask。`empty_text` 是固定 buffer，但空分支的适配器运算没有 detach；两分支都对共享参数求导。相减单位是与 STTrack fused 同维度的稠密特征增量，之后仍经过冻结跟踪网络的后续计算。根据 `EXPERIMENT_PLAN.md`，任务监督训练的是适配器，底座和 Center Head 冻结；这与 CLIP Surgery 的推理阶段相似度修正存在实际区别。

相同输入及确定性运算下，`q=q_empty` 时增量按构造抵消；这只是参数化恒等性质。它不能证明非空增量只含语义，也不能排除“所有非空词触发共同适配”。双支梯度只说明训练机制，不能自动称为因果适配或因果效应识别。递归跟踪的总效果还涉及后续状态改变。

建议表述为：**“借鉴空文本参照中心化思路，在冻结 STTrack 上研究共享可训练语义适配器的差分残差参数化。”** 不应声称首次空词消冗余、首次条件/空条件相减，或已提取纯语义因果成分。本轮没有读取 M84 开发结果；按任务交接状态，M84 仍在训练，尚无开发性能结论。既有先例不能证明 M84 无效；实现差异也不能证明其原创或有效。

本地依据：`centered_semantic_adapter.py`、`code/lib/models/sttrack/semantic_spatial_adapter.py`、`EXPERIMENT_PLAN.md`；只读核查。
