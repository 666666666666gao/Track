# 区域语义证据：RegionCLIP、Alpha-CLIP 与当前 dense tracker

日期：2026-10-12。性质：论文及官方源码研究；不是完整性门禁、模型结果或新颖性结论。遵循本地 `research/SKILL.md` 与 `shared-references/local-codex-policy.md`，由受委派的 Codex 研究代理完成，未继续委派。

**结论：人工确认类别/属性正确、画面中只有一个该类物体，并不保证当前网络把该语义绑定到正确的空间区域。两项工作提供了“给定区域后判断其语义”的可借鉴证据；它们尚未解释或修复本项目这次 cart 跳变。** 本任务给定的既有保存结果是：首次大幅跳变时目标仍在 crop 内，类别正确、没有第二辆 cart；本次未重新计算这些结果，尚无 dense-candidate NN 回放。因此不能把候选缺失、语义排序、视觉分数、质量或框回归中的某一项写成已测根因。

## 1. 当前代码实际读取了什么

- **并非只用全图 CLS。** `FullDenseTracker.observe` 从上一状态取 256×256 RGB-D 搜索 crop，把前三个 RGB 通道交给冻结 CLIP 的 224 输入预处理，使用 `dense_patch_features` 返回的 256×768 tokens，丢弃同时返回的 CLS。该函数在最后六个 ViT block 读取 value-value attention，产生 CLIP Surgery 风格的 16×16 稠密特征。`native_valid/clip_valid` 来自 crop 的 padding 有效性，表示有效像素覆盖，并不是前景分割 mask。见 [tracker L53–71](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:53)、[encoding L11–35](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_encoding.py:11)。
- **初始 ROI 与当前候选的处理不对称。** 初始框从 RGB/depth/fused/CLIP 四张网格各采 4×4 点，作为 reference；当前分支投影整张 16×16 网格，逐位置与五个文本槽计算 cosine，再经 reference/phrase attention 和 learned head 产生语义残差。当前预测框的宽高、框内区域没有进入 `branch` 的语义编码：当前候选的 box extent 与 initial ROI 的证据机制不同。见 [observe/branch L51–89](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py:51)、[采样 L12–27](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_decoder.py:12)、[4×4 点 L10–13](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/instance_ab_prototype.py:10)。
- **最终分数是联合响应。** 文本贡献为 `Δtext = branch(text) − branch(empty)`；最终 `response = native_response × odds_residual(native_score, Δvisual+Δtext)/native_score × sigmoid(quality)`，随后取最大值。类别文本没有形成“当前框已经确认为该类”的硬约束；geometry 分支不读取当前文本。`step` 提交所选框作为下一状态。见 [decoder L91–123](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py:91)、[step L88–114](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:88)。
- **已有定位负监督。** 配套 `objective` 有中心 focal 的负位置、框回归、以当前框 IoU 为目标的 quality BCE、crop 可观测性 BCE，以及对低 IoU 高响应候选的 preservation 项；`train_m122_full_causal.py` 导入并调用它。这些是跟踪定位监督，不能等同于外部大规模“区域—文本”预训练，也不能将本系统描述成没有负样本。见 [objective L31–66](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m121_dense_target.py:31)、[调用 L11、25–27](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m122_full_causal.py:11)。

## 2. RegionCLIP：先给框，再学习框区域与概念的对应

核对范围是官方 `microsoft/RegionCLIP` 主线的 RN50-C4 路径，固定提交 [`4b8513b56e24827e3d6468e1f2105869f35c2d0b`](https://github.com/microsoft/RegionCLIP/commit/4b8513b56e24827e3d6468e1f2105869f35c2d0b)。

- **输入与空间汇聚。** RGB 图像先得到共享 feature map；区域来自独立 RPN，或评估时给定的 GT boxes。每个框经 ROI pool/Res5，再经 CLIP attention pool 得到区域向量；RN50-C4 示例配置使用 res4、14×14 ROI，attention pool 将区域空间特征的均值 token 与各位置一起做注意力，返回首 token。它接受框位置，不以 alpha mask 作为识别输入；所核对 COCO 配置 `MASK_ON=False`。见 [ROI L112–135、248–266](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/roi_heads/clip_roi_heads.py#L112-L135)、[attention pool L73–107](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/backbone/clip_backbone.py#L73-L107)、[配置 L8–38](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd.yaml#L8-L38)。
- **预训练与背景要分阶段。** 论文用 CLIP teacher 为区域匹配概念伪标签，学习区域—概念对比、概念分布 KL 蒸馏及全图—文本对比；区域对比的负项是其他概念，并不是逐像素前景/背景分割损失。检测迁移阶段另有背景类别、背景损失权重与框回归；不能将迁移训练的背景能力直接算到任意预训练 checkpoint 上。代码以 teacher 最高概念分数筛区域、按相同概念构造正样本矩阵。见 [论文 §3.2–3.3](https://arxiv.org/html/2112.09106v1)、[预训练损失 L558–635](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/meta_arch/clip_rcnn.py#L558-L635)、[伪标签 L637–661](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/meta_arch/clip_rcnn.py#L637-L661)。
- **分数含义。** 检测头对归一化区域向量与文本向量求 cosine/temperature，附加固定零向量背景 logit，再对类别 softmax；可选与 RPN objectness 取几何平均。它是依赖概念集合、温度与 proposal 的检测评分，不是“首帧同一物体”的概率，也不是框 IoU。零样本示例明确关闭 box delta，所以换区域识别器不会自动修复 proposal 几何。见 [背景与分类 L441–464、546–566](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/roi_heads/fast_rcnn.py#L441-L464)、[检测监督 L609–623](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/roi_heads/fast_rcnn.py#L609-L623)、[推理 L703–711、808–811](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/detectron2/modeling/roi_heads/fast_rcnn.py#L703-L711)、[零样本配置 L1–8](https://github.com/microsoft/RegionCLIP/blob/4b8513b56e24827e3d6468e1f2105869f35c2d0b/configs/COCO-InstanceSegmentation/CLIP_fast_rcnn_R_50_C4_ovd_zsinf.yaml#L1-L8)。

## 3. Alpha-CLIP：RGB 保留上下文，alpha 指定本次关注区域

官方作者仓库 `SunzeY/AlphaCLIP` 固定提交 [`ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff`](https://github.com/SunzeY/AlphaCLIP/commit/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff)。

- **输入与汇聚。** `visual(image, alpha)` 接受 RGB 和与其对齐的单通道区域图；ViT 将 RGB patch convolution 与 alpha convolution 相加，再过 Transformer，输出投影后的 CLS。这是 mask 条件化的编码，不是对最终 patch tokens 做 masked mean，也不把背景像素硬删除。alpha 可由矩形框或更精细的 mask 构造；默认 loader 对 RGB 使用 resize+center crop，mask 需匹配同一空间变换。训练 dataloader 先联合 pad RGBA 成方形，再分别 resize，alpha 按 `(alpha−0.5)/0.26` 归一化。见 [模型 L333–375](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/alpha_clip/model.py#L333-L375)、[loader L79–86](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/alpha_clip/alpha_clip.py#L79-L86)、[数据 L27–60、90–130](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/train/dataset/alpha_grit.py#L90-L130)。
- **训练目的。** 论文使用 GRIT box/text 配合 SAM 伪 mask 构造 RGBA—文本对，另研究 ImageNet 来源的区域数据；冻结文本编码器，训练图像编码器。官方训练代码使用归一化区域条件图像向量与文本向量的双向 batch 对比交叉熵；没有独立的二分类背景拒绝头或框回归目标。alpha 的 0/1 表示区域提示，并不等价于一个训练好的“所有非目标均拒绝”输出。原图/全 1 alpha 的混合由 `common_pair` 控制，所附启动脚本设为 0.1。见 [论文 §3、表3](https://arxiv.org/html/2312.03818v2)、[训练 L74–122](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/train/train/train_grit_1m.py#L74-L122)、[启动脚本 L23–32](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/train/train_slurm.sh#L23-L32)。
- **分数与定位来源。** API 输出归一化图文向量的缩放点积 logits；类别 softmax 仅表示给定文本集合内的相对分配，不是目标存在性或身份置信度。官方 REC 示例另外依赖 detector proposals、SAM masks，并组合 full/blur 表征；不能把其 REC 数字归因于单独一个 alpha 编码步骤。模型本身没有生成候选框或搜索目标的接口。见 [logits L481–514](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/alpha_clip/model.py#L481-L514)、[REC 官方说明 L9–49、61](https://github.com/SunzeY/AlphaCLIP/blob/ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff/eval/rec_zs_test/README.md#L9-L49)。

## 4. 对本问题能支持的推论与边界

**有依据的研究方向：** 将“当前候选区域到底对应什么语义”变成显式输入—输出关系：RegionCLIP 用 `image + box → region embedding`，Alpha-CLIP 用 `image + alpha → focused embedding`。对只有一辆车的画面，这种证据仍可服务于“车所在区域 vs 人/背景所在区域”的排序；不要求先出现第二个同类实例。依据是上述结构和预训练目标，是否改善本跟踪器仍需匹配候选的实际测量。

**不能由此承诺：** 两者不提供跨帧实例身份保证；区域错误、候选没有覆盖目标、遮挡或目标已出当前 crop 时，额外语义分数也不会生成缺失的几何信息。它们保留上下文，不能保证完全消除背景相关性。当前系统已包含初始 reference、深度与定位监督，需要评估新增证据的增量，不能用“普通全局 CLIP 总会失败”概括它。只在旧 dense tokens 上增加 ROI 平均，也不等价于 RegionCLIP 的区域预训练；Alpha-CLIP 的作用依赖专门训练的图像编码器，而非仅增加一个 mask 参数。

**尚缺的因果证据：** 必须先知道跳变前后是否存在几何正确的 dense candidate，以及其原始/修正框、视觉响应、语义残差、quality 与最终响应的关系，才能区分候选覆盖不足、框几何错误和候选排序错误。本次只核对源码与论文，没有执行这一 NN 回放。本文不提出测试序列专用规则，不迁移外部阈值，不更改训练或跟踪行为。

## 核对记录

- 官方分支 SHA 由只读 `git ls-remote ... refs/heads/main` 取得，源码通过固定 SHA 的官方 raw URL 读取。论文采用作者 arXiv 全文：RegionCLIP v1、Alpha-CLIP v2；没有把无法读取的 CVF PDF 当成已核对全文。
- 本地阅读时 repo HEAD：`99d1a31e45bb408ca8707299183bfcfefb54a15c`；下列为实际工作文件 SHA256（不以 HEAD 代替文件内容）：

| 文件（均在 `diagnostics/three_module_20260928/`） | SHA256 |
|---|---|
| `dense_target_decoder.py` | `055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2` |
| `full_dense_tracker.py` | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| `dense_region_encoding.py` | `71715a497147eb917e6a93ba6916ebbe01bc4a859cb275401289df2706627f33` |
| `dense_region_decoder.py` | `66b9317529f741aba818aa9e9feba7fcebdb54c285f0bd9184f98f4b1f202e7b` |
| `train_m121_dense_target.py` | `40d00f84c9f2220deb6f557a0e516f4385922c86d276da1b5d3b17f8891502ee` |
| `train_m122_full_causal.py` | `ffa40a08d167d9a7f185cf83ecf0851e9c5ed548c2b1280bb969d6b8c689f891` |

仅新增本笔记；没有下载权重或数据、安装依赖、SSH/GPU/训练进度查询、NN 执行、训练修改、外发消息或 Git commit/push。源码与论文均以释义记录，没有长段摘引。
