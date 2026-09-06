**总体结论：WARN，适合进入一个新的、范围明确的文本信息实验，现有证据不足以宣称当前 STTrack 已获得独立语言收益。** 这是同 GPT 家族的 Type-A 只读建议性审阅。保留原生 STTrack 作为直接对照合理；“文本与 RGB-D 相互引导”仍需新的对照证据。下述历史实验不能归入当前 full127 成绩。

本次已直接读取指定脚本和 JSON，核对两轮共 144 条原始 VLM 回答与结果解析一致、M44 两臂全部拟合/开发事件的汇总计数及完整递归聚合一致，并核对可在指定文件内闭合的源码和结果哈希。没有重新计算原图与 GT 的 IoU；原始图像、训练标签、特征缓存、完整预测和部分被引用的评测依赖不在本次允许读取范围内。

**已经测试过的语言与结果**

| 证据范围 | 实际输入和结果 | 能支持的结论 |
|---|---|---|
| 早期 STTrack response adapter | 冻结 CLIP **整段结构化语言表示**与首帧模板联合预测身份残差；361,474 参数，660 个四帧 predicted-crop 事件。留出评估 mean IoU 为 0.531050→0.532911，语言相对 zero-language 的**累计**贡献为 +0.026568；存在 1 个 catastrophic，恢复覆盖不足。 | 已经训练过语言残差，不能称“首次把文本接入 STTrack”；没有完整部署递归或公开增益。`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:8908,8916-8944` |
| 更深的 STTrack response 微调 | 5,689,155 参数，factor-6/query-reset tentative 分支；30 条审计序列中 27 条产生 291 个事件，每事件实际递归 10 帧。public/tentative mean IoU 为 0.479047/0.563602，但 harm 12.5962%、126 个 catastrophic；语言相对 zero-language **累计贡献 −2.880166**。 | 有真实短窗分支恢复能力，但语言增量为负，且未提交为贯穿整条序列的公开策略。不能把整体分支收益归因于文本。`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:8958-8985` |
| LACHTT 候选选择历史 | 使用候选池化 RGB/Depth/Fused、CLIP 图像及不可变首帧 **768 维短文本锚**；M1/M2 跨序列选择门失败。 | 已测试过全局描述和文本相似度；后续缓存 rescue 证明动作存在价值，未隔离文本贡献。`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:12065-12078,12205-12245,12335-12343` |
| position_text v1/v2 | Qwen2.5-VL-3B 输入 RGB 参考图、当前图和提示词；`stable_identity` 是**生成输出字段**，并非供给 STTrack 的属性标注。每版四条序列、24 窗口×3 臂。v1 坐标/时间角色语义失败；v2 三臂“接受/正确/错误”为 **14/4/4、11/1/6、0/0/0**。 | 是 GT 失败时点抽样的封存窗口建议，tracker commits 和 training steps 均为 0。没有当前 ordinal 定位或递归收益。`projects/sttrack_lachtt_v1/overlay/tools/run_position_text_pilot_v2.py:18-47,60-107,120-168`；`projects/sttrack_lachtt_v1/diagnostics/position_text_v2/result.json:4-91` |
| 当前正式原生 STTrack | full127 EAO/ACC/ROB 为 **77.321654/82.471190/93.669109%**；语言关闭、优化步数 0。M55 也明确关闭语言、候选关联和读取器。 | 强原生对照已经存在；这些数值不能计作文本或 M55 收益。`projects/sttrack_lachtt_v1/diagnostics/native_vot_full127/result.json:15907-15921`；`projects/sttrack_lachtt_v1/diagnostics/m55/EXPERIMENT_PLAN.md:18,80-82` |

v2 的全弃权也不能写成“完全没有定位信息”：16 个未接受的有效原始框中有 3 个 IoU≥0.5、9 个 IoU≤0.1，但 **0 个左右名次**。忽略不确定性字段会同时纳入大量错误；零接受下的零错误不构成安全性收益。19 个窗口使用了动态模板，文件记录其写入时定位均超过 0.5，因此不能把这次失败直接归因于错误模板。证据：`projects/sttrack_lachtt_v1/diagnostics/position_text_v2/abstained_box_diagnosis.json:3-10,254-255`；`projects/sttrack_lachtt_v1/diagnostics/position_text_v2/paired_diagnosis.json:2-122`；`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:16961-16965`。

另需明确：identity-only 的 low22 失败数 200→195、Qwen current-anchor 的 195→202 属于历史 SUTrack；CPSD 属于历史 SRTrack。这些可以说明研究风险，不能替代当前 STTrack 的语言消融。`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:13888-13894,16842,518-536`。

**现成接口提供了什么，尚未提供什么**

- `CandidateSetAssociation.forward(current, previous, references, geometry, scores, previous_choice)` 已接收前后各 10 个候选、两模板，局部特征为每对象 `2×16×768`；先逐格投影，再展平形成对象描述。现接口**没有文本参数、文本编码器或属性 mask**。几何是图像归一化的中心/宽高，不能称为目标相对位移或真实物体排名。`projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_candidate_set.py:15-45`；`projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_candidate_set_observation.py:7-15`。
- 可以复用候选观察、模板引用、候选自身回归和完整状态提交入口。`NONE` 仅映射到**该策略当前状态下**的 candidate0；它不会恢复独立原生轨迹。辅助 affinity 在 runtime 被丢弃；训练报告中的匹配正确数还通过 **GT current_target 选择矩阵行**，不能当作在线身份传播能力。`projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_candidate_set.py:35-58`；`projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_candidate_set.py:51-66`；`projects/sttrack_lachtt_v1/overlay/tools/train_sttrack_m44.py:80-95`。
- M44 缓存由原生预测裁剪产生，所有训练 `previous_choice=0`；bag05 递归已出现 49 次非零前一选择。缓存可用于信息筛查，不能标成新策略自身状态训练。后续 M52 的一次策略状态扩充也未改善递归，因此“补状态就会成功”仍是未证实机制。`projects/sttrack_lachtt_v1/overlay/tools/collect_sttrack_m44.py:44-73`；`projects/sttrack_lachtt_v1/diagnostics/m44/input_state_coverage.json:4-24`；`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:17970-17974`。

**一个有依据的下一实验：稳定属性短语对候选局部 RGB-D 的可学习对齐。**

新假设是：**保留属性短语/词级表示，并让它与候选局部格点交互，可能比一个整句 CLIP 相似度更能利用颜色、纹理、形状和标记等身份线索。** 依据是正确候选存在却选择相邻实例的具体案例；这只是动机，不是成功证据。M42 的局部空间臂与池化臂在 375 个缓存窗口上选择完全相同；M44 显式几何递归反而差于坐标置零对照，M51 相对几何替换也失败，均不能被概括为“加空间信息已证明有效”。`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:17059-17065,17172-17178,17842,17866-17876`；`projects/sttrack_lachtt_v1/diagnostics/m44/recursive_result.json:5-29,639`。

最小实现边界是在现有 `cell` 投影之后、对象特征展平之前增加**一个文本—局部格点对齐块**：属性 token 读取 RGB/Depth 格点，局部视觉证据反馈到属性对当前候选的权重，再进入既有集合上下文和身份输出。稳定属性字符串保持不变；反馈更新的是当前匹配表示。保留原生底座、候选生成、回归、query 和默认模板更新，训练新增块及相同范围的候选头。这里提出的是待验证结构，尚未证明“互相引导”或注意力图等于可靠语义解释。

本轮保持相同的当前几何上下文，可以检验文本在当前候选位置关系下的增量；**不能把这个实验称为已经实现“当前画面同类物体从左到右第几个”的文本能力**。真实 ordinal 能力目前缺少两项输入：

1. 与本次初始化目标及 63/22 序列划分绑定、来源可追溯的稳定属性文本，以及词级编码器/权重/分词和缺失值契约。旧 768 维句向量缓存不能还原词级表示。
2. 当前帧的独立物体实例、同类对象集合及对应关系标注，和部署时从可用图像/预测历史生成该关系的来源。M44 仅有目标框监督，其他身份未标注；93.12% 的“含正确候选”拟合输入含多个正确框，候选编号不能直接变成物体序号。`projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_candidate_set.py:56-65`；`projects/sttrack_lachtt_v1/overlay/tools/collect_sttrack_m44.py:46,62-73`；`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:17458-17474`。

若以后使用 M55 的新底座，还需要在该底座上重新导出特征；现缓存绑定旧模型，而已记录的 TSG 改动触及这些局部特征来源。不能把缓存复用当作新底座输入已验证。`projects/sttrack_lachtt_v1/diagnostics/m44/spec.json:5-19`；`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:18351-18357`。

**归因和因果边界应预先固定**

- 同时保留独立原生 STTrack、相同参数预算/训练预算的无文本模型、正确文本模型。正确文本模型还做冻结后的空文本、同类错属性及打乱文本检查；空文本推理消融与“重新训练的无文本对照”不能混为一项。错/打乱文本须实际改变属性语义，并在相应数据划分内构造。
- 若要分别声称稳定属性和当前关系的贡献，应在**同一个块**中分别去掉这两个输入，并配对相同当前图像视野；额外全景图片、额外 VLM 计算和文字表述不能混合归因。
- 63 条 fit 与 22 条 development 按序列隔离；这 22 条已多次使用，不是全新未见测试。文本来源、负例构造、epoch/seed及小幅增益判据应事前固定，不在 low22 上选择阈值。`projects/sttrack_lachtt_v1/diagnostics/m44/spec.json:28-54`；`projects/sttrack_lachtt_v1/diagnostics/m55/v2/training_spec.json:216,229-244`。
- 当前 GT 生成的目标位置句、GT 选择的上帧候选或最佳模板只能用于训练监督/明确标注的特权容量诊断。部署文本必须只依赖初始化及截至当前已观测信息。相对位置需标明当前帧、视野和播放方向，不能永久沿用首帧排名；水平翻转必须同步变换关系。`projects/sttrack_lachtt_v1/overlay/tools/run_position_text_pilot_v2.py:23-47,74-79`；`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:13846`。
- 动作编号和物理身份标签分开：多个框可以属于同一目标；未知干扰物不能被编造跨帧匹配。候选置换时同步重映射上一选择和动作标签。
- 最终比较必须从相同初始化分别运行全部 22 条完整预测状态递归，报告 pooled/macro IoU、低重叠帧、H10及原成功序列损害。缓存增益不能代替递归增益：M44 geometry 缓存均值略升，完整递归却从 0.652226 降至 0.617098。初始化/无效 GT 的统计口径也应固定。`projects/sttrack_lachtt_v1/diagnostics/m44/training_result.json:15738-15750`；`projects/sttrack_lachtt_v1/diagnostics/m44/recursive_result.json:5-29`；`projects/sttrack_lachtt_v1/diagnostics/m55/EXPERIMENT_PLAN.md:55-66`。

按 experiment-audit 的 A–F 检查：**A GT 来源 WARN**，评估代码读取数据集 GT，但本次缺少原始标签和轨迹，不能完整独立重算；**B 分数归一化 PASS（已读代码范围）**，窗口均值/有效帧均值没有除以模型自身最大输出；**C 结果存在与一致性 PASS（指定产物）**，历史引用仅为文档证据；**D 能力接线 WARN**，辅助匹配已计算但未被在线选择使用；**E 范围 WARN**，小型 GT 时点 pilot、缓存/短窗分支和重复开发集均不能扩写为完整公开语言收益；**F 类型为 real_gt 定位评估，ordinal/count 为未验证的定性输出**。相应计算入口见 `projects/sttrack_lachtt_v1/overlay/tools/analyze_position_text_pilot_v2.py:23-52`、`projects/sttrack_lachtt_v1/overlay/tools/run_sttrack_m44_recursive.py:20-51`。