# M88 completed experiment integrity audit

Date: 2026-09-21. Generated UTC: 2026-09-21T05:56:24.731864+00:00.

**Overall verdict: WARN. Integrity status: WARN. Deterministic evidence checks: PASS. Frozen performance acceptance: FAIL, 12/14.**

审计由 fresh-context Codex reviewer `/root/m88_completed_integrity` 直接读取文件并独立复算；`review_independence=same-family`、`acceptance_status=provisional`。确定性复算可作为已核验的文件事实，不能把同模型家族的语义判断写成跨模型家族接受。本审计没有选择模型、训练、跟踪、SSH 或正式评测。

**模型与范围。** M88 `semantic_spatial_local_reference_v1`，唯一训练的 Category，固定 seed2027，最终完整一轮 checkpoint。适配器文件 SHA256 为 `a6705a512444f0d715e50d2655edb1611dcbb7801be83329ce7bff2d13611c78`。声明的 native base SHA256 为 `cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`；适配器哈希已经核验，推理时实际 base 文件的字节哈希有下述独立警告。每个开发组均为22条反复使用的 DepthTrack Train 开发序列、33130个保存位置，其中33108个非初始跟踪位置、28897个有效计分位置、4211个无效GT位置。三个内容组共享同一适配器文件，各自提交递归状态，不能描述为三个训练模型。依据：`evidence/frozen.json:2-13`、`evidence/run_recursive.py:30-73`、`evidence/recursive_result.json:1050-1052`。

## A. Ground truth provenance — PASS, with provenance limits

计分GT来自 `dataset_root/<sequence>/groundtruth.txt` 的保留副本。全部22份GT的SHA与冻结case吻合，也与更早的native冻结spec吻合；初始框与第0行一致。计分从第1帧开始，排除非有限或宽高非正的GT，未使用预测值合成GT。GT文件路径、载入与比较见 `evidence/run_recursive.py:101-105`，完整scope验证见该文件82–96行；本审计直接读取并核验 `evidence/development_gt/*.txt`。

推理只给 `tracker.initialize` 传t0初始框和冻结文本bank，后续 `tracker.track` 只接收当前图像（`evidence/run_recursive.py:53-64`）。`STTrackSemantic.initialize` 的RoI来自同一t0图像/框（`evidence/code/lib/test/tracker/sttrack_semantic.py:20-30`、`sttrack_initial_instance_observation.py:8-20`）。已读到的实际推理路径没有后续GT输入。基础tracker的调试显示分支存在 `info['gt_bbox']`，但本次参数 `debug=0`，该显示代码不参与预测（`run_recursive.py:39-41`、`code/lib/test/tracker/sttrack.py:141-150`）。

训练会加载fit GT，这是有监督训练；每帧先 `tracker.step(image)` 提交预测状态，再将该帧GT送入损失，且没有后续GT重初始化（`evidence/train_causal.py:97-129`、`evidence/causal_training.py:21-69`）。这里的“GT在预测之后”仅指当前训练帧的前向/状态提交；先前监督更新会影响后续训练预测，不能误读为无监督训练。fit130与development22名称集合不相交。

bank的五个实际文件哈希及其内容结构已直接解析：Category仅保留原自动category slot0，有效属性槽使用CLIP Empty，padding保持零；三种开发条件的mask相同，22个Swapped均只改变slot0的向量。其第一帧caption生成协议、caption来源哈希和原文本被保留，但原始caption调用输入/生成记录、视频帧内容不全在这个包中，因此不能独立重放caption来源，更不能认证这些词是语义真值。依据：`evidence/training_spec.json:2520`、`audit/bank_metadata.json:1723-1726,2015-2018`、`audit/supplemental_reference_checks.json`。

## B. Score normalization — PASS

独立实现以连续xywh框的交集/并集计算IoU；pooled IoU分母为有效GT帧数，macro为22个序列均值的等权平均。低IoU帧定义为IoU≤0.1；H10是原帧索引上持续至少10帧的最大连续低IoU段，遇到无效GT会中断。没有除以预测分数自身最大值/均值的性能归一化，也没有把信心值当IoU。依据：`evidence/recursive_metric.py:14-30`、`evidence/run_recursive.py:106-116`、`audit/reviewer_checks.py:92-139`。

`native_preservation.spatial_kl` 确实把student和teacher响应各自归一化为空间分布；这是明确标记的训练KL项，不是上述性能计分，也不把native预测冒充数据集GT（`evidence/native_preservation.py:6-13,23-44`）。

| Arm | Pooled IoU | Macro IoU | IoU≤0.1帧 | H10段 |
|---|---:|---:|---:|---:|
| M88 Category | 0.714800772043 | 0.707712887178 | 5373 | 65 |
| M88 same-weight Empty | 0.652226263176 | 0.684336416360 | 7397 | 75 |
| M88 same-weight Swapped | 0.710027126405 | 0.719890230022 | 5477 | 65 |
| M84 Category | 0.708521427828 | 0.700454335697 | 5541 | 72 |
| Native extracted reference / M84 Empty | 0.652226263176 | 0.684336416360 | 7397 | 75 |

以上每一个逐序列及汇总值均从保存的框和GT复算，与作者结果一致，浮点比较容差为1e-8；整数和门槛布尔值精确相等。

## C. Result existence, artifact binding and budget — PASS for stored evidence; base binding WARN

原始manifest的367个文件逐项核验SHA及长度，归档368个成员（含manifest）逐字节匹配。161个integrated源码哈希匹配冻结integration；三组M88共66个raw序列文件、M84 Category/Empty共44个raw文件，序列名、帧数、连续索引、初始框和SHA全部核验。另核验native提取包25个manifest条目（22个native序列及3个来源文件），其source spec哈希与原有native结果绑定。输入哈希保存在 `audit/audited_input_hashes.json` 与 `audit/supplemental_reference_checks.json`。

直接以受限pickle/zip存储解析读取final checkpoint，未执行torch或模型代码。检查architecture、seed、训练spec哈希、base声明哈希、完成标志、130条完成序列及5798次更新；23个AdamW参数状态的step均为5798。模型含289154个参数和一个768维Empty buffer，参数有限；24个state张量中22个改变，取消的最终bias及Empty buffer保持不变。按实际state顺序重构初始tensor SHA，匹配M88训练回执及绑定的M84训练回执。

130条逐序列日志独立求和得到186694次track、5798次更新，3917条采样状态的预期索引完整。按32帧窗口计有5896个可能窗口，日志实际少98个更新窗口；源代码只有窗口内存在有效梯度帧时更新，完整fit GT与逐帧state流未导出，故不声称独立证明全部98个窗口的内部标签组成或重放整个训练。训练label计数为inside158073、outside16406、invalid12215。依据：`evidence/training/category/result.json:6-26`、`evidence/train_causal.py:124-180`、`audit/recomputed_training_budget.csv`。

M84父训练spec、integration和inventory均由M88冻结哈希绑定，银行文件、fit顺序、seed、损失、优化预算和更新规则一致；161个源码摘要只在local-reference adapter和architecture-tag loader两项不同。旧M84初始checkpoint字节未收进补充包；已验证的是M88初始字节重构哈希等于绑定的M84初始tensor回执，以及 `evidence/prepare.py:34-39` 的复制/相等检查源码。依据：`audit/M84_training_spec.json`、`audit/M84_training_result.json:15-18`、`audit/supplemental_reference_checks.json`。

**W1 — 推理时实际base字节缺少哈希证明。** 训练在 `evidence/train_causal.py:49` 核验真实base文件，71行取得起始tensor摘要，179–180行断言base参数/buffer未变化及base梯度为空，终态回执记录相等（`training/category/result.json:16-17`）。但没有before/after base tensor副本可供复算。更关键的是，`run_recursive.py:13-27,39-46` 在推理前只传入声明的base哈希；`sttrack_semantic.py:14` 比较的也是声明字符串，真正的base载入在 `code/lib/test/tracker/sttrack.py:24`，没有重新计算文件字节SHA。本地也没有该532MB基础权重。因此，三组推理绑定同一final adapter的结论通过，推理期间实际base字节一致性不能独立认证。未发现base被替换的证据；这是一处实际检查缺口，不是已证实的数据错误。

`EXPERIMENT_PLAN.md:38-42` 仍保留准备期pending，`training_spec.json` 与 `recursive_spec.json` 的status也保留 `prepared_not_frozen`。它们是被外部 `frozen.json` 哈希封存的历史文档，不是当前执行状态。实际完成状态依据终态训练/递归回执及exit文件。不要为消除这些历史字段而改写封存文件。

## D. Dead code and execution path — PASS

实际入口 `run_recursive.py` 第80行导入 `statistics`，第105行调用；该函数调用 `episodes`（`recursive_metric.py:14-30`），输出进入保存结果。`recursive_metric.py:33-87` 的老M42 `main` 在本任务未被调用，不用它的其他gate证明本任务完成。Hann解码、提交状态和模板更新的调用链可追到 `code/lib/test/tracker/sttrack.py:96-139`；semantic context实际传入网络并调用适配器（`code/lib/models/sttrack/sttrack.py:147-154`）。

## E. Scope and interpretation — WARN

这些是反复使用的Train development22结果，固定一个训练seed，无正式DepthTrack Test、CDTB或VOT输出。计划本身已限制这一级别（`EXPERIMENT_PLAN.md:17-36`）；本审计按用户固定seed2027要求核验，不提出多seed或新的机制试验。

**冻结性能条件未通过。** 相对M84的4项通过；相对native的5项通过4项；相对Swapped的4项通过3项；Empty/native逐序列指标一致通过1项。全部14个独立比较见 `audit/recomputed_gates.csv:2-15`。两个失败为：

1. Category macro IoU =0.707712887178，小于Swapped =0.719890230022，差−0.012177342844（约−1.217734个百分点）；见该CSV第11行。
2. native零H10保护失败：`container01_indoor` 和 `mobilephone02_indoor` 在native中均H10=0，在M88 Category中均H10=1；见 `audit/recomputed_per_sequence.csv:11,22` 及 `audit/recomputed_gates.csv:14`。

所有44个content LOO逐项复算。Category−Empty的22个LOO均正，范围0.028126263187至0.072679988635。Category−Swapped有21个为正；删除 `colacan01_indoor` 后差值为−0.012375965642，见 `audit/recomputed_leave_one_out.csv:31`。这不能支持“正确词始终优于错误词”或“可靠实例语义”结论；Swapped是固定内容替换，自动caption和donor词并未被标注为真/假语义。

**同状态与递归轨迹必须分开。** 训练native teacher来自student当前crop/template/query上的同一次base前向，teacher只读，不提交一条独立native轨迹（`evidence/causal_training.py:28-48`、`native_preservation.py:23-38`）。它的局部KL不能证明整条native轨迹受到保护。开发集三内容组各自递归，首次bbox不同也不能定位首次身份错误；严格损害段和模板写入只是不同历史轨迹的事后描述。

## F. Evaluation type — real_gt

主表与下述损害/改善统计均使用保留的数据集框GT。模型合成张量check和96帧丢弃smoke只证明功能/可训练性，本审计没有将其算作性能、语义GT或独立实验重复（`evidence/model_check.json`、`evidence/preflight_result.json`、`EXPERIMENT_PLAN.md:25-30`）。

## Exact Empty reference and retained provenance boundary

M88 Empty与封存M84 Empty的全部33130个bbox和33130个score字段逐值一致，其中22个score为初始化null。进一步直接读取已存在于本机的native提取包，确认33130个bbox和33108个非初始score精确相等；native逐序列指标及五个native gate也重新计算。记录位于 `audit/supplemental_reference_checks.json`。

**W2 — 原始native分片未本地复核。** 已核验的是从两个既有remote分片导出的22条native轨迹、提取脚本、manifest及其source-spec哈希关系（`audit/native_reference/extract_native_reference.py:11-29`、`reference_provenance.json:2-13`）。原始两个完整分片没有收进本机，所以不能声称本审计从原始分片重新提取或重算它们的SHA，也没有重新运行native跟踪或比对所有内部query/template张量。该补充验证加强了逐帧一致性证据，但没有消除W1的实际base推理哈希缺口。

## Strict intervals, scheduled writes and all sequence rows

严格损害：Category IoU≤0.1且参考IoU≥0.5，持续至少10帧；改善交换两者角色。无效GT会中断区间。103个区间全部复算并与原CSV对应。以下单元格为“段数 / 帧数”。

| 参考 | Category损害 | Category改善 |
|---|---:|---:|
| M84_control | 14 / 617 | 19 / 743 |
| category_empty | 19 / 1143 | 30 / 3187 |
| category_swapped | 10 / 654 | 11 / 791 |

1101个模板写入事件按已绑定代码规则重构：`frame>0 && frame%50==0 && score>.75`，模板数为2。正确/严重/中间分别用IoU≥0.5、≤0.1和中间区间分类，无效GT单列。原raw没有保存template write标志或模板像素，因此这里是由源码条件和得分重构的计划写入事件，不能声称直接检查了写入模板内容。依据：`code/lib/test/tracker/sttrack.py:129-139`、`code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml:61-68`，均相对于evidence目录。

| Arm | 正确 | 严重 | 中间 | GT无效 |
|---|---:|---:|---:|---:|
| category | 251 | 12 | 1 | 11 |
| category_empty | 256 | 27 | 0 | 9 |
| category_swapped | 248 | 12 | 0 | 10 |
| M84_control | 243 | 7 | 2 | 12 |

M88相对M84的严重写入由7增至12，虽M84的四个聚合门都通过，这个负面描述仍须保留。它没有被事后加入冻结门槛。

以下保留全部22行，数值是Category减参考的逐序列mean IoU，负值均不删。相对M84为13胜9负，相对Empty为11胜11负，相对Swapped为14胜8负。

| 序列 | Δ vs M84 | Δ vs Empty | Δ vs Swapped |
|---|---:|---:|---:|
| bag05_indoor | +0.069801882 | +0.016258078 | +0.076891638 |
| ball07_indoor | +0.009178142 | +0.416040484 | -0.001069331 |
| ball19_indoor | -0.000521833 | +0.004305770 | +0.000748757 |
| beautifullight01_indoor | -0.259616444 | -0.192075383 | +0.010445464 |
| book06_indoor | +0.126404329 | +0.152576551 | -0.008686734 |
| car01_indoor | -0.052523882 | -0.000299409 | +0.003709112 |
| car02_indoor | -0.000183813 | -0.367296515 | +0.000932605 |
| colacan01_indoor | -0.000379032 | +0.383517536 | +0.164550959 |
| colacan04_indoor | +0.368886883 | +0.504054770 | +0.156095772 |
| container01_indoor | -0.055781784 | -0.056801305 | -0.056992474 |
| cup08_indoor | +0.002462415 | +0.060360146 | -0.000523555 |
| cup10_indoor | +0.000764592 | -0.000296678 | +0.026731506 |
| cup13_indoor | +0.078625829 | +0.034145998 | +0.001772138 |
| egg_indoor | +0.022964094 | -0.001631559 | -0.008183160 |
| flower02_wild | +0.007070510 | -0.020583362 | -0.139605592 |
| flower03_indoor | +0.000969564 | +0.007241967 | +0.002457377 |
| flowerbasket_indoor | -0.003252718 | +0.000546141 | +0.011099608 |
| ghostmask_indoor | -0.193667907 | -0.092734399 | -0.236632096 |
| glass03_indoor | +0.005348850 | +0.132383068 | +0.000573934 |
| mobilephone01_indoor | +0.032845426 | -0.167269107 | +0.000901059 |
| mobilephone02_indoor | -0.001085152 | -0.292074108 | -0.277820623 |
| notebook02_indoor | +0.001378181 | -0.006086327 | +0.004702092 |

## Claim impact and actions

- 支持限定表述：该final adapter完整完成了固定seed2027开发评测，四项M84聚合比较通过，但全部冻结条件仅12/14，故原promotion判据失败。
- 不支持：已满足三大正式benchmark目标、语义真值被验证、实例身份因果得到证明、全序列性能得到保护。
- 保留此模型及所有失败/负面证据，不启动新的机制试验，不因聚合改善改写门槛。
- 后续由主任务选定已完成模型并冻结三数据集同checkpoint评测时，回执应记录实际读取的adapter和base字节SHA、冻结文本输入、实际配置/runner及官方评测版本；在每组运行前后验证同一组文件哈希。此动作针对W1中已发现的缺口，不修改本次封存结果。
- 不要求补跑多seed、训练或native重放；若后续整理证据，可独立追加原始native分片或可信摘要来源，但当前结论保留W2限定。

复算命令（在completion_collection目录；标准Python，无第三方依赖）：

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' .\audit\reviewer_checks.py
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' .\audit\supplemental_reference_checks.py
```

原始sealed evidence未修改。审计源码、细粒度复算CSV、受限读取的bank元数据、补充native来源和JSON结果保存在 `audit/`；完整审阅request/report与metadata trace位于 `.aris/traces/experiment-audit/2026-09-21_m88_completed/`。这份审阅的语义判断保持same-family/provisional；确定性检查PASS和性能条件FAIL是两个独立字段。
