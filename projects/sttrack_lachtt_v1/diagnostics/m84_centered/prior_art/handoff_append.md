

M84 等待期间的直接先例核查（2026-09-20）：

独立资料核查并复核 CLIP Surgery 官方代码后，确认空字符串参照相减已有直接先例：固定提交 `d4696d47f49cfe70f49140afe5eb94f94c5f59bc` 的单文本路径计算 `V @ (T(q)-T(empty)).T`。空参照通过提示模板集合编码，并非零向量。M84 的区别是共享可训练非线性适配器的两份稠密增量相减，两支都保留梯度，再加回冻结 STTrack 特征。实现差异本身不证明原创性、纯语义因果解释或性能。

因此当前方法定位为“借鉴空文本参照中心化思路，在冻结 STTrack 上研究共享可训练语义适配器的差分残差参数化”。空词恒等不保证非空增量仅包含词义，也不能恢复过去已改变的递归状态。本次只补引用与机制边界，不修改 M84 冻结训练、评测或晋升规则；尚无 M84 完成态性能结论。

证据：`projects/sttrack_lachtt_v1/diagnostics/m84_centered/prior_art/EMPTY_REFERENCE_PRIOR_ART.md`。官方来源：[单文本实现](https://github.com/xmed-lab/CLIP_Surgery/blob/d4696d47f49cfe70f49140afe5eb94f94c5f59bc/clip/clip.py#L287-L308)、[作者说明](https://github.com/xmed-lab/CLIP_Surgery/issues/2#issuecomment-1512353799)。报告同时区分论文与固定代码的标签集合加权细节，不能将另一条标签集合路径笼统写成减空字符串。
