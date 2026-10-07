# M121 源码审查（修正后）

时间：2026-10-07T11:01:26.467062+08:00。结论：**PASS，未解决阻断项 0**。这仅为独立上下文、同系列 provisional 源码审查；请求 gpt-6-astra / max，实际后端没有工具证明，**不是 M121 runtime PASS**。

第一轮 `M121_SOURCE_REVIEW_20261007_105320.json/.md` 原文与初始源码哈希完整保留，没有重写失败记录。

- **M121-B1 已修复**：15 个实际架构组均覆盖梯度与相对 initial_state 的参数变化计数、绝对变化和最大变化。sanity 的已有 post-update 输出真实检查 box、score、feature、quality 与同一 selected 索引一致；controller 必须收齐三臂真实 sanity 成功后，才 fresh 初始化完整训练。重复同槽控制不要求每个 q/k 张量都产生梯度。
- **M121-B2 已修复**：新增纯 stdlib analyzer 报告三个训练臂的全部四种读出，保留 native、自身 Empty、M101 历史定位参照和两种同容量控制；覆盖分层、逐序列、0.5 跨界、<=0.1 严重收益损害、双方合格时的细 IoU、固定原生位置精修和静态/精修 full256 上界。controller 在 CPU 报告实际 exit 0 后才写终态；指标失败仍封存，不自动晋升。事件覆盖字段已明确命名为 new_events_with_qualified_candidates / lost_events_with_qualified_candidates，不是候选框数量或物理身份计数。
- **M121-N1 已修复**：观察源窗口上界使用 image_shape.flip(-1)-1，与实际 native sample_target 的 padding/切片一致，没有新增推测性分支。未声称修正翻转了当前某个 Train 标签。

主路径核对结果：四路首帧 4×4 固定区域参照、256 当前位置、五槽人工短语、四次 64 维交叉 attention 的维度和遮罩相容。初始化不按当前候选重定义；当前文字只通过共享 q-minus-empty 支撑支路进入决策，几何/质量/交集观察仅读视觉表示。GT 不传入 forward；无效 GT 排除，部分框相交用交集中心生成支撑、回归仍针对完整 GT。IoU 质量目标 detach；focal + 2 GIoU + 5 crop-normalized L1 + quality BCE + intersection BCE + 可靠九竞争者相对保持，与计划一致。原生 score 的实际 clamp 源码、Hann 来源和零残差初始化检查相容。

固定训练为 seed2027、相同初始化和批次次序、B64、AdamW lr3e-4/weight_decay.01、16 epochs/640 updates、只报告 final。人工标签实际 SHA、152 条 130/22 划分及 704 槽内短语出现次数已复核；M117 完整 152/3502 本地回执通过。主流程只优化 Train fit，development 是复用诊断集；Test/CDTB/VOT 不参与本轮拟合或选择。

实际审查检查：16 个 Python 文件（含私有部署器）的 Python 3.8 AST、2 个嵌入脚本通过；6 份实际 native 来源及 head 共 7 份 SHA 回执一致。15 个模型声明组与 sanity 组完整对应。新 paired/panel 函数复现 M101 对 native 的真实原汇总，并复核 M118 三臂×四条件各 495 条真实保存行的全部分层 fine-IoU/收益损害；按 key 反序配对不改变结果。这些仅检验新报告函数，**不是 M121 新结果**。

部署器只使用已有 known_hosts 与默认拒绝策略，密码从 stdin 读取。原 M120 exit 0、当前两卡空闲、空间和依赖字节验证位于上传启动之前。源码审查没有 SSH、Torch、GPU、模型执行、包安装或实验源码修改。直接与传递依赖的 16 个 D 文件哈希、私有 deployer 哈希、辅助来源及产物哈希分别记录于 JSON，避免将历史来源混入部署列表。

最近检查的本地 M120 状态为 `actual_M120_first_hour_pair_running`（2026-10-07T10:51:54.088866+08:00），仍未完成。后续必须真实通过完整来源、部署资源门槛、三臂 sanity、fresh full 和配对报告；这次 PASS 不证明训练收敛、内容增量、递归跟踪、C 能力或公开九项最终目标。当前整体目标仍未完成。
