# M82 实验完整性审阅

- 审阅日期：2026-09-20。
- 审阅者：gpt-6-astra，reasoning effort=max；全新上下文审阅代理 `/root/m82_completed_integrity`。
- `review_independence: same-family`；`acceptance_status: provisional`。
- 范围：本目录提供的冻结源码、计划、回执、结果、四组完整开发轨迹、22 条 GT，以及随后提供的描述性分析和 `handoff_append.md` 草稿。
- 操作：只读已有实验材料；另写本报告及机器可读报告。没有连接服务器、加载训练权重运行网络、启动训练、运行新推理或发布内容。

## Overall verdict: WARN

已直接从四组轨迹和 GT 独立复算全部开发指标，未发现虚假 GT 替换、指标自归一化、轨迹缺帧、数字不一致或冻结源码哈希变化。开发集结果可以作为限定范围的完成态证据归档。

完整的功能保持/语言改进晋升条件未满足：主条件 10/10、Category 相对匹配 M78 的增量条件 4/4，但固定 Category 权重的内容条件只有 4/8。Category-head Empty 在四项指标上均优于原 Category 内容。不能把训练条件间的收益直接归因为推理时类别文字有效，也不能把 `primary_pass=true` 当成完整晋升通过。

另有证据范围限制：本地没有两臂最终权重、逐序列训练日志、抽样训练状态日志、文本 bank 和历史原生/M78 原始轨迹。本次重验了它们在回执中的引用及其与现有材料的一致性，不能声称重新计算了权重哈希或重新审阅了上游文本生成过程。该限制不表示远端文件不存在。

## 独立确定性核验

本代理使用 Node.js 原生文件、哈希和标量算术直接读取所有 88 条轨迹及 22 个 `groundtruth.txt`；没有导入执行者的 Python 统计实现。共 1,243 项现有材料检查通过，数值比较最大绝对差为 `7.275957614183426e-12`，来自浮点求和次序。缺失文件单独记录，没有计入通过项。

- 4 个家族 × 22 条序列，共 132,520 个保存位置；每组 33,130 个位置、28,897 个有效评测位置。每组排除 22 个初始化位置和 4,211 个无效 GT 位置。
- 全部 88 个轨迹哈希与各自 receipt 一致；帧号从 0 连续到末帧，首框与冻结初始化框一致，全部框有限且宽高为正。
- 全部 22 个本地 GT 哈希、长度、首框与 `recursive_spec.json:13-255` 一致。130 个 fit 名称与 22 个 development 名称无交集；fit 帧数合计对应 186,694 次跟踪调用。
- 8 个 `frozen.json:7-15` 指定源码、训练/递归 spec、队列、integration、inventory、原生/M78 reference 均与冻结哈希一致。integration 的 160 项中本地存在 146 项且全部哈希一致；14 个缺失项均在 `code/lib/train/data_specs/`，本次不能宣称整个 160 文件快照完整落盘。
- 10 个本地退出码文件均为 0。训练完成回执中的参数数目、调用数、初始状态标识、结果绑定互相一致；原始训练权重/日志的重验边界见 C。
- 主条件、内容条件、增量条件、44 个 leave-one-sequence-out 数值均重算一致。另逐项比对了 8 行 aggregate CSV、110 行 sequence CSV、26 行 gate CSV，没有不一致。
- 从保存的 `score` 按实际运行规则独立重建更新次数：Category 260、独立 Empty 282、同权重 Empty 270、Swapped 275，逐序列更新帧列表也与描述性文件完全一致。

结果身份：`recursive_result.json` SHA256 为 `823d0560db3acfdd594c53ecfae239ebb59ef77b3b2776bc9037dcdaa3a259ea`。

| 条件 | 有效帧平均 IoU | 序列等权平均 IoU | IoU≤0.1 帧 | H10 段 |
|---|---:|---:|---:|---:|
| 原生 reference | 0.6522262631762438 | 0.6843364163600396 | 7397 | 75 |
| M78 Category reference | 0.7190085849647154 | 0.7015137299609854 | 5223 | 73 |
| M78 Empty reference | 0.6345474449952171 | 0.6514465893569041 | 8125 | 83 |
| M82 Category | 0.7295208956047248 | 0.7447772742984768 | 4952 | 67 |
| M82 独立 Empty | 0.6853177948014881 | 0.6909410033387766 | 6414 | 80 |
| M82 Category-head Empty | 0.7366947943941015 | 0.7516163189672695 | 4694 | 66 |
| M82 Category-head Swapped | 0.7075750516684154 | 0.7354797164541026 | 5744 | 67 |

M82 四行从框与 GT 重算；原生和 M78 行来自哈希绑定的历史 reference，原生逐序列统计的再聚合亦一致。没有把历史 reference 的引用核验冒称为重新运行这些模型。数值位置：`recursive_result.json:6-46,953-977`；历史源：`native_reference.json:5-12`、`M78_reference.json:6-29`。

## A. GT 来源与训练/推理隔离：WARN（几何 GT 和当前调用链 PASS；上游材料不完整）

几何 GT 从数据集文件读取，而非由预测生成。训练入口使用 `training_spec.json:12` 指定的 DepthTrack Train 路径，逐序列核对 GT 哈希并读取四元组（`train_causal.py:98-107`）；开发阶段逐一核对 GT 哈希后读取（`run_recursive.py:147-153`）。本地所有 22 个 GT 与冻结 spec 的哈希一致。例如 `dataset_gt/bag05_indoor/groundtruth.txt:1-4` 为实际 xywh 标注，第 52-54 行为 `nan`，没有用预测补齐无效 GT。`recursive/category/bag05_indoor.json:15-23` 的第 1 个跟踪框也不等于该帧 GT。

训练当前帧先调用 `tracker.step`，随后才把当前 GT 传给 loss（`train_causal.py:124-129`）。`CausalTrainingTracker.step` 没有 GT 参数；搜索裁剪来自上一个预测框（`causal_training.py:21-31`），当前状态和模板更新由 Hann 解码后的学生预测/分数决定（`causal_training.py:51-69`）。GT 只控制已完成预测后的标签、回归位置和损失 mask（`causal_training.py:72-100`），优化后影响未来训练预测属于训练过程本身。

同状态教师实现符合描述：原生正向不传 `semantic_context`，从该次正向的原始搜索特征分别产生教师和学生读出；教师框在学生框提交前解码，教师响应显式 detach（`causal_training.py:28-48`；`code/lib/models/sttrack/sttrack.py:147-169`）。教师使用学生当前的裁剪、模板和 query；它没有独立的原生框/模板历史。有效且中心在裁剪内的标签才计算教师 IoU，且 IoU≥0.5 才增加 KL（`native_preservation.py:23-44`）。无效、出界、教师错误样本无该 KL；不存在推理时 GT 门控。

开发推理只消费首框、初始化文本/实例特征和连续 RGB-D 帧（`run_recursive.py:89-110`）。四组输出和 receipt 均在 GT 评测前验证完整（`run_recursive.py:126-153`）；队列先等待四组完成，再启动分析（`run_pair.sh:27-43`）。这是源码及已保存产物所支持的隔离；没有把 `subsequent_gt_opened=false` 的文字声明当成独立运行时系统调用跟踪。

WARN：文本 bank 仅有冻结路径/哈希（`training_spec.json:2555-2582`）和“初始化自动描述”协议（`training_spec.json:2524,2603`），本地缺 bank、原始 caption 与生成日志，无法在本次确认全部文本确由首帧产生、slot/mask 真正匹配且无后续 GT 信息。训练 fit 的 GT 原文件也未随本包提供。本次没有发现泄漏路径，但上游来源核验不能据此写成全面完成。

## B. 统计与归一化：PASS

开发指标为连续 xywh 交并比，没有使用模型自身最大值、均值或分位数充当评测分母（`recursive_metric.py:20-30`）。初始化和无效 GT 被排除；按帧平均用有效 GT 帧数，宏平均按序列等权（`run_recursive.py:154-160`）。H10 是 IoU≤0.1 连续至少 10 个有效时间位置的低重叠段数，跨无效 GT 时断开（`recursive_metric.py:14-17,22-30`），不是 VOT ROB、EAO 或官方 DepthTrack 测试指标。

`native_preservation.py:6-13` 把正的原始 score 在空间位置上归一化，目的是定义 KL 概率分布，属于训练正则，不是对开发成绩作自归一化。输入由 Center Head sigmoid clamp 保证严格为正（`code/lib/models/layers/head.py:175-179`）。实现按位置求和、按 batch 求均值，没有再除以 256。

该 KL 不约束响应绝对尺度；空间分布相同也可能产生不同更新阈值行为。模板规则使用 Hann 响应绝对最大值（`code/lib/test/tracker/sttrack.py:119-139`），实际配置为两模板、50 帧间隔、严格大于 0.75（`code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml:11-16,61-68`）。更新次数重建方法正确（`summarize_completed.py:48-66`），只能说明按固定规则触发了多少次，不能判定写入质量、模板污染或因果效用。

## C. 产物、数字与状态：WARN

四组轨迹、receipt、结果和新增 CSV 均实际存在并一致。三个使用 Category 权重的家族共享头 SHA `581a044b…f47f859`，独立 Empty 绑定 `9c94d277…de67515`（四个 `*_recursive_receipt.json:2-6`）。每组 22 条/33,130 个位置与 `*_recursive_receipt.json:7-144` 一致。不存在以局部轨迹冒充四组完成的证据。

两条训练回执均为 130 条、186,694 次调用、5,798 次优化、289,154 个可学习参数；Category/Empty 的 KL 启用帧分别为 150,053/149,792，累计 KL 为 3736.785264164384/3691.065195655392（`training/category/result.json:2-35`、`training/empty/result.json:2-35`）。这是回执内容，不能把两个不同训练状态分布上的 KL 次数或均值作为固定输入上的机制对照。

WARN：本地两臂都缺少 `final.pth`、`sequence_log.jsonl`、`sampled_state_trace.jsonl`，虽然对应引用保存在各臂 `result.json:19-22`，且既有标量核验脚本确实包含这些文件检查（`audit_m82_completed_20260909.py:20-44`）。`saved_development_audit.json:10-11` 记录过去检查通过；本次只能验证该回执及其源码/结果哈希，不能重复它的全部训练原始材料检查。本地亦无 native checkpoint、零初始化 checkpoint、文本 bank、原生/M78 原始轨迹或完整 M78 训练配置/源码；匹配 M78 的哈希声明有绑定，全部因果配对条件未在本次逐项重验。

`EXPERIMENT_TRACKER.md:3` 仍是“等待首次优化回执/没有性能结果”的启动时快照，应在完成态记录中明确标记其历史时点或补充最终状态。冻结计划/spec 中的 `frozen_before_training`、历史 `REVIEW_UNAVAILABLE` 应保留原样，新的审阅身份和完成状态另附，不能改写历史。

## D. 评测代码是否真正用于产物：PASS（源码调用和产物重算层面）

M82 实际分析入口是 `run_recursive.py --analyze`（`run_pair.sh:43`；`run_recursive.py:211-221`）。它导入 `recursive_metric.statistics` 并对四组的每条序列调用（`run_recursive.py:119,147-153`），进而使用 `episodes`（`recursive_metric.py:14-30`）。这些函数输出的全部字段均存在于结果，并通过独立全量重算。

`recursive_metric.py:33-87` 保留的是另一套旧入口，它不是 M82 队列入口；不能误称该旧 `main()` 运行过，也不能把其未调用视为当前统计未执行。被复用的 `statistics/episodes` 是实际相关路径。当前训练入口调用 `native_preservation.supervision`（`train_causal.py:56-57,124-129`），再调用 Raw 竞争和基础 GT 监督（`native_preservation.py:25-26`；`window_competition.py:45-56`）。没有将未启用的 support-loss 分支列为 M82 的已执行贡献。

本次没有复现 GPU 网络执行；零残差 101 帧 parity、96 帧/3 次优化 smoke 属于既有回执和检查源码证据（`check_causal.py:75-92,97-145`），不能充当完成后长递归安全证明。

## E. 范围、内容归因与晋升：WARN；完整晋升条件 FAIL

范围固定为一个 seed2027、130 条 fit 和反复使用的 Train development22。没有新独立测试集，没有随机种子方差估计；本审阅遵守只用 seed2027 的任务约束，不要求增加 seed。结果明确限制为开发集（`recursive_result.json:1126-1129`；`training_spec.json:2602,2607-2615`），不能使用“完整三数据集提升”“泛化验证完成”或“保证原生安全”的措辞。

冻结主条件全部通过（`recursive_result.json:940-952`）；Category/M78 的均值、宏均值、低重叠、H10 四项非退化增量全部通过（`recursive_result.json:980-994`）。Category 相对 M78 的按帧 IoU 增量为 1.051231064 个百分点、宏平均 4.326354434 个百分点，低重叠少 271 帧、H10 少 6 段。这支持已绑定参考下的开发集聚合改善报告。

固定头内容条件为 4/8（`recursive_result.json:995-1009`）：

- 原 Category 内容相对同权重 Empty 按帧 IoU 低 0.717389879 个百分点，宏平均低 0.683904467 个百分点，多 258 个低重叠帧、多 1 个 H10。四项内容条件全部失败。
- 原 Category 内容相对 Swapped 按帧高 2.194584394 个百分点、宏平均高 0.929755784 个百分点、低重叠少 792 帧；H10 均为 67。最后一项通过的是冻结的非退化规则，不是“减少 H10”。
- 相对同权重 Empty 的 leave-one-out 只有 1/22 为正；移除 colacan01 后变为 +1.561829479 个百分点。相对 Swapped 的 leave-one-out 有 21/22 为正，但移除 cup13 后变为 -0.126302046 个百分点（`recursive_result.json:1010-1057`）。该分析显示聚合结果对个别序列敏感；没有删序列、重定主指标或构成额外泛化证据。

Category 对独立训练 Empty 的差值同时包含不同训练内容带来的参数与训练状态变化。固定 Category 头的内容干预也通过其自身递归改变后续裁剪/模板历史。两类差值不能直接相减得到“纯词义贡献”；Empty 输入仍通过已训练 adapter，未适配原生不是它的同义词（`run_recursive.py:75-83`；`code/lib/models/sttrack/semantic_spatial_adapter.py:31-51`）。

`training_spec.json:2627` 明确要求主条件、八项内容条件、四项 Category 增量条件共同满足 preservation-improvement claim。当前只满足其中两组，故完整晋升未通过；不得用“与 Swapped 比较通过”替代失败的 Empty 内容条件，不得事后把 Category-head Empty 换成最终主条件。原生/本轮独立 Empty 的零 H10 序列保护通过，也不等同于所有历史方法逐帧无损；`recursive_result.json:1114-1117` 明确保留了先前受保护列表中的 glass03 破坏记录。

## F. 评测类型：PASS，开发跟踪指标为 real_gt

| 对象 | 分类/性质 | 可支持结论 |
|---|---|---|
| M82 四组开发轨迹 vs 数据集 `groundtruth.txt` | `real_gt` | seed2027、Train development22 上的实际连续框指标及限定对照 |
| 原生/M78 reference | 按其保存协议为 `real_gt`；本次仅哈希/聚合引用核验 | 绑定历史参考的数值比较；不冒充重新执行 |
| 同状态原生空间 KL 教师 | 模型生成的训练正则目标；若单独报告一致性，应标作 `synthetic_proxy` | 同状态功能保持约束，不是几何 GT、独立原生轨迹或测试成绩 |
| 自动 caption 和 Swapped/Empty 内容 | 模型生成输入及受控输入替换；不是语义 GT | 输入条件差异，不保证类别描述正确 |
| 构造张量的 KL/梯度/竞争检查 | 合成数值契约检查，属于 `simulation_only` 层面的检查证据 | 实现数学/梯度契约，不支持数据集性能 |

未发现把模型教师输出当成开发 GT 的用法。也没有 human_eval 或 cross-family 接受证据。新的模型审阅仍为 same-family/provisional，不能用本报告改写冻结记录的历史 reviewer 状态。

## 对完成态草稿的审阅

所读草稿 `handoff_append.md` SHA256 为 `8b90a14db528fe5d2bc7b74130b377f398aee60a9fc25ecde2170c391d831584`。其表格、差值、代表性序列数字、4/8 内容判定、Swapped H10 相等、更新次数均与产物一致。建议发布前作两处文字收窄：

1. 第 3 行“功能保持增量成立”改为“聚合指标改善，冻结内容条件未通过”，避免读者把完整 preservation claim 理解为通过。第 23 行应保持“限定开发集、匹配历史参考下的聚合改善”的口径。
2. 第 33 行“分离当前选峰/框变化与已积累的历史状态”改为“测量固定 Category 历史下的当前内容干预”；不能承诺精确分解完整递归差值中的历史贡献。

草稿第 5、35 行涉及本次远端连接、重算最终权重哈希、磁盘/GPU；第 23 行具体称为 M65 Control 的历史归属。它们不由本次本地审阅直接验证，应绑定执行者保留的相应原始证据。本报告没有公开发布任何内容。

## 唯一建议的最小后续诊断

拟议的“在 M82 Category 原始 22 条完整递归状态上进行同状态只读内容干预”适合当前证据，可作为独立后验诊断；不需要等待该诊断才能归档当前完成态结果。

每帧固定原 Category 历史，只用已计算的 adapter 前 RGB/Depth/fused 特征比较原 Category、Empty、Swapped 和未适配 head。反事实仅生成读出，不推进 query、框或模板；唯一推进状态的仍是原 Category 分支。首先按现有冻结输出逐帧核对原分支 bbox/score；若未复现一致，应先解释偏差，不能拿另一条状态历史作当前反事实的基准。保存四种读出的 raw/Hann 选峰索引、框和分数等小型产物，全组封存后才打开 GT。无需保存全量特征：按本次 33,108 个非初始化位置、三组 256×768 float32 张量估算约 72.7 GiB；可逐帧消费后释放。

该诊断能识别“在 Category 已访问状态上，当前文本替换是否直接改变响应/选峰/框”，不能量化全部历史效应，也不能把只读未适配头称为独立原生追踪器。它不训练、不选新 checkpoint、不改参数/阈值、不替换主条件；原内容失败和完整晋升失败结论继续保留。不得将诊断计划写成已获得机制结论。

## Claim impact

- “四组 M82 开发轨迹完成、上述数字可复算”：supported，限定本地已验证产物与 real_gt 开发协议。
- “配对训练完成 130 条/186,694 调用/5,798 优化”：supported by completion receipts；本次未重验缺失的最终权重和训练日志原件。
- “相对原生/匹配 M78 的开发集聚合指标改善”：supported with qualifier；历史参考由哈希绑定，本次未重跑。
- “正确类别词具有稳定正收益/语言贡献已成立”：unsupported；固定头 Empty 对照失败。
- “功能保持全部晋升条件通过、独立原生轨迹安全已保证”：unsupported。
- “官方 DepthTrack Test/CDTB/VOT 同一最终模型全部完成”：unsupported；没有此类结果。
