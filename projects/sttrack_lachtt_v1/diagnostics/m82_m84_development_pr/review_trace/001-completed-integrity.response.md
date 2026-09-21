# M82/M84 已封存开发轨迹 PR 补算完整性审阅

日期：2026-09-21。审阅者：gpt-6-astra，reasoning_effort=max，fresh context；原生 Codex 子任务 `/root/m82_m84_pr_audit`。实际调用参数见 `reviewer_invocation.json`。`review_independence: same-family`；`acceptance_status: provisional`。

## Overall Verdict: WARN

**Integrity Status: warn。未发现指标伪造、预测导出改造、分数归一化、来源哈希冲突或漏报某组结果。** 七组指标、完整导出及来源链的本地确定性检查通过；非平凡 VOT 栅格重叠没有在本审阅中由框和 GT 独立重算，不能称为“全链独立验证”。WARN 表示下述证据边界，不表示数值复算失败。可按现有开发补充统计的限定表述报告。

审阅只读原始产物，未运行原评价器、跟踪器、torch、模型、GPU 或 SSH；没有安装依赖，没有修改评价源码、指标、原轨迹、GT、训练或晋升条件。新写文件仅为审阅源码、复算曲线/结果及本报告/trace。

## 确定性结果

独立源码 `reviewer_recompute.py` 仅使用 Python 标准库，没有导入被审源码、NumPy 或 VOT。它从封存 overlap/visible/confidence 数组重新建立全局置信度秩阈值网格，逐序列用 `math.fsum` 计算 P/R，再对 22 条曲线等权平均并取最佳 F。七组共 700 个曲线点保存在 `reviewer_pr_curves.csv`；完整源码哈希、输入哈希、154 份原轨迹绑定及每组误差保存在 `reviewer_recompute.json`。本地执行退出码为 0，报告状态为 `deterministic_checks_pass`。

| 组别 | P（%） | R（%） | F（%） | 最佳 F 阈值 |
|---|---:|---:|---:|---:|
| M82 Category | 75.97636948727633 | 72.33543798265445 | 74.11121281504643 | 0.420002 |
| M82 独立训练 Empty | 70.95897326846293 | 66.61975183633429 | 68.72093321267802 | 0.469584 |
| M82 Category 权重、Empty 输入 | 77.95350625185132 | 72.75940131924924 | 75.26695008448819 | 0.417210 |
| M82 Category 权重、Swapped 输入 | 75.60677448388238 | 71.04108538243857 | 73.25285655714924 | 0.436108 |
| M84 Category | 75.83858398764416 | 66.74002401753538 | 70.99899469641859 | 0.470153 |
| M84 同权重 Empty | 69.65522154923387 | 66.76948699475307 | 68.18183390655601 | 0.471730 |
| M84 同权重 Swapped | 76.74186652433467 | 68.37231364333782 | 72.31573043398373 | 0.436249 |

表中数值取自封存结果；独立复算相对这些报告值的最大差异为 **1.4210854715202004e-14 个百分点**，七组阈值完全一致。每组为 22 序列、33,130 帧；七组合计 154 份轨迹、231,910 个记录位置。每组 28,919 个有效 GT 帧含 22 个初始化，4,211 个无效 GT 帧；排除初始化后的有效数为 28,897，与旧连续 IoU 分母相符，但指标计算口径不同。

345 条 manifest 记录全部通过路径集合、SHA256 和字节数核对；归档含这些 345 个文件以及 manifest 自身，共 **346 个普通文件**，逐成员内容哈希与本地副本一致。manifest 不包含自身哈希是正常的非循环设计；manifest、归档本身的 SHA 单列如下。

## A. Ground Truth Provenance: PASS，附来源限制

GT 从 `/root/autodl-tmp/depthtrack/train/sequences/<sequence>/groundtruth.txt` 加载；`evaluate_sealed_pr.py:68-89` 先核验七组全部已封存预测，再访问数据集 GT。其 22 份 GT SHA 与冻结 spec、M82 完成包 `dataset_gt/<sequence>/groundtruth.txt`、M84 完成包 `dataset_gt/<sequence>.txt` 三方一致，且初始化框与 GT 首行一致。没有从预测生成 GT 的路径。`evidence/spec.json:79-323` 列出全部序列和来源哈希。

分类为 **real_gt**，指使用数据集提供的数值 GT。此检查验证了本地副本与既定来源的绑定，没有向数据集发布方重新取样验证，也没有通过图像判断所有无效 GT 是否代表真实目标不存在。`NARRATIVE_REPORT.md:5` 已正确保留该区别。

`evidence/depthtrack_pr.py:40-60` 把有限且宽高大于 0 的 GT 转为 VOT Rectangle，把其他 GT 转为 Special(0)，随后调用带图像边界的 `calculate_overlaps`。本审阅独立确认了所有 visible 标记与 GT 条件一致、所有初始化 overlap=1、所有无效 GT overlap=0，并检查其余 overlap 有限且在 [0,1]。**这些检查没有独立证明其余 202,279 个有效、非初始化预测位置的 VOT 栅格重叠值正确。**

## B. Score Normalization / Prediction Integrity: PASS

全部 154 份原始轨迹 JSON 的 SHA 均匹配对应原始 receipt；七组 receipt、父 result、training spec、recursive spec、runner 和相关 tracker 源码均与本地原来源的哈希链相符。每份轨迹的 arm、sequence、连续 frame 索引、帧数和初始化框均核对。M82 为两份训练 head、四种开发条件；M84 为一份 head、三种输入条件，不能当成七份独立训练或七个 seed。

`M82/run_recursive.py:95-108` 和 `M84/run_recursive.py:59-70` 直接保存 `target_bbox` 和 `best_score`。两套经 integration SHA 验证的 `code/lib/test/tracker/sttrack.py:118-130,183-186` 从 Hann-window × score-map 的 response 取最大值并返回 best_score，没有用预测统计量作分母。这里的 M82/M84 路径分别指下文来源目录。

`evaluate_sealed_pr.py:99-105` 的变换只有框与原分数六位小数导出，以及声明的初始化 null→1。独立按 Python `%.6f` 格式重建后，**全部 308 份 TXT 内容逐字节相等**，不是仅比较近似均值。最大框取整误差 5.000000555810402e-7，最大分数取整误差小于 5e-7；这只是既定序列化精度，没有额外分数校准或后验框干预。

## C. Result Existence / Source Binding: PASS，附运行证据限制

七组全部结果存在；`evidence/result.json:7` 起的 metrics、`evidence/metrics.csv:2-8` 和 `run.log:1-7` 的对应数值完全相符。`run.exit:1` 为 0。评价器在生成结果前断言每组 22 序列、33,130 帧；源码、spec、arrays、每份输出的哈希均与 result 相符。父 M82/M84 result 都为 complete_recursive_development；原始七组递归退出文件均为 0。M84 recursive_spec 的历史状态字段仍为 prepared_not_frozen，但它的精确字节 SHA 与 frozen.json 和完整回执一致，不能只凭该旧字段误判本次结果尚未完成。

M84 Empty 的指标字典与历史 native/raw 全字段相等；更强的确定性证据是 **22 序列 × bbox/score 的 44 个文件 SHA 全部与历史 native/raw 一致**。历史比较没有使用 constant 或 GT_overlap_oracle 分支；那些分支仅存在于未修改的历史对照文件中。

`launch.json:4-6` 记录 gpu_disabled=true、新调用/优化为 0；`evaluate_sealed_pr.py:57-137` 的代码路径仅加载文件和 CPU 数值评价，没有模型调用，并在 `:64` 设置 OpenCV 线程数为 1。但本审阅没有观察原远端进程环境，launch 不含完整启动命令或全部线程环境变量；不能把“CUDA 屏蔽、CPU 线程统一限制为 1”升级为本审阅独立测得的事实。文件退出码与日志是完成证据，不是全系统调用追踪。

## D. Actual Metric Calls: PASS

`evaluate_sealed_pr.py:62-64` 动态加载固定 SHA 的评价源码；`:107-110` 实际保存 VOT overlap 数组并调用 `evaluate_depthtrack_results`，结果随后进入 JSON、CSV 和日志。核心指标不是未调用的备用函数。`depthtrack_pr.py:157-158` 的 format_metrics 只是未使用的输出格式工具，不是另一条评价支路。

阈值算法（`:25-37`）从每组 33,130 个保存置信度整体降序排序中取 98 个秩点，delta=floor(33130/98)=338，再加 +inf/-inf 两端，共 100 个曲线点。逐帧选择为 confidence≥threshold。序列 P=选中帧 overlap 总和/选中帧数；R=选中帧 overlap 总和/该序列可见 GT 帧数；无选中帧时 P=1、R=0。无效 GT 的零 overlap 仍可能进入 P 分母，而不进入 R 可见性分母。初始化参加统计。

`depthtrack_pr.py:117-142` 先宏平均每个阈值的 P/R，再计算 F=2PR/(P+R)，取首个 argmax；不是先逐序列取最优 F、不是平均各序列最优 F，也不是帧池化 P/R。本独立实现重算了这条完整“数组→阈值→曲线→最佳点”链。

## E. Scope / Claims: WARN，当前限定表述可用

两项既有实验均 seed=2027，开发集是反复使用的 DepthTrack Train 22。此次只补算长期 PR；不是新的训练、独立验证集、DepthTrack Test、CDTB 或 VOT benchmark 结果，也不增加晋升门或模型选择步骤。

`NARRATIVE_REPORT.md:3-19` 的七行表、比较方向和边界与数据一致：M82 Category 的 F 高于独立训练 Empty，但低于同权重 Empty；M84 Category 高于 Empty，但低于 Swapped。这支持描述已保存轨迹在该统计口径下的差异，不建立正确语义的因果贡献。不能从最高某行或最佳 F 阈值反推部署策略、改 0.75 模板阈值、改变旧 IoU/H10 结论或宣称新训练收益。

`NARRATIVE_REPORT.md:23` 对 M88 和 Qwen 的持续运行状态属于执行者的系统状态记录。本审阅确认补算源码没有读取 M88 路径或产生 M88 指标，未独立检查 M88/Qwen 当前运行状态。任何对其持续状态的表述须由执行者自己的即时证据承担。

## F. Evaluation Type

**real_gt / repeated-development / posthoc supplementary reporting**。计算完成，确定性指定检查通过；语义审阅 same-family/provisional，不是跨模型族接受，也不是正式外测资格。

## 未独立复现的环节与具体来源

- VOT 的 `Rectangle`/`calculate_overlaps` 依赖实现及其版本未随包封存。本审阅使用的标准库 Python 环境没有 VOT、NumPy、OpenCV，也没有安装它们。非平凡有界栅格重叠只验证输入和数组约束，没有从 GT 与框独立重算；不得用连续矩形 IoU 替代后宣称等价。
- geometry 的 22 组 640×360 尺寸及首帧图像 SHA 来自原执行记录；本审阅没有打开/解码对应首帧图像。22 条完整远端图像路径和 SHA 逐条列于 JSON 的 unverified_remote_inputs，原 geometry SHA 见下表。这一项同时限制独立验证图像边界。
- M82 Category/Empty 两份 final.pth 在给定本地完成包没有文件字节可读，只核对 receipt 与 training result 的一致 SHA；远端 head SHA 分别为 `581a044bbba8514fb5f26d283a9dd038f46e27f8608c08724228eebfff47f859` 和 `9c94d2776f7898699b6a7af8e107e8c74f41fa7f411f41f20882a231ade67515`。M84 本地 final.pth 可读且重算 SHA=`c63605ebcb1f702de66f96967255e5301bfdca1e3c40450b6d4b8a952791b1e0`，仅 hash 文件、未加载模型。
- 历史原生来源 shard0/shard1 的远端 SHA 仅沿用历史 result 记录，未重新 SSH/读取这些远端 trace；精确路径/值列于 JSON。当前 M84 六位 TXT 与历史 raw 的 44 个出口哈希已独立匹配，但这不替代对历史模型执行过程的复审。
- 原执行环境与远端文件现在是否仍保持相同，未通过此本地审阅重新观察；本报告绑定的是现有封存本地字节。原 result 保留 independent_audit_completed=false，没有被事后改写。

无需为本次开发补充表启动新的训练、安装 VOT 或扩大评价。本次需保留的处理是：随表呈现上述协议和证据边界，运行资源/M88状态注明是执行记录，不写“全链已独立验证”。

## 主要审阅输入 SHA256

| 输入 | SHA256 |
|---|---|
| evaluate_sealed_pr.py | b700d348d84c19d8d3fd04dc1b555d31cb95d53d3999803e011ea0946f8c831c |
| evidence/spec.json | 7d77948aabd40da7b8c65ccb4c49fe1e69200696f625f31b6d3f0512b357e308 |
| evidence/result.json | a871828dc0bc9499469bb287b6dcab61da5ef865c9992143c7ef12b9497afe31 |
| evidence/depthtrack_pr.py 与历史源码 | 05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc |
| evidence/overlap_score_arrays.json | 2429884d2f59b0a7dfcce81aebb56c9170a29837b2e2fb89712dd03e7e311b76 |
| evidence/geometry.json | bcdfa1343b658bcb5883031de4f6c3408f96d918e248bfe888292095e64fcfa2 |
| evidence/manifest.json | 51663db697fc1e90da62635521fb39e0961ac965e13b5f87151a1767dcc30b31 |
| completed_evidence.tar.gz | 3c8d6abfab2d00f99dd909913bbcd678e1c9a06b978bc11128f45886d15e2d6d |
| 历史 native PR result.json | 90fc1533446309921299c61f25002a465774783ea06ebda64012982c6d895f37 |
| M82 原 recursive_result.json | 823d0560db3acfdd594c53ecfae239ebb59ef77b3b2776bc9037dcdaa3a259ea |
| M84 原 recursive_result.json | a9281e0833b8ba1c01220374f3d6a93dbc9d22e6e8648c3495bb009083f2d5f7 |
| NARRATIVE_REPORT.md（本次审阅版本） | c2ad7dd660ebffa838ebd7404d548f30669c17db9ac3fe4f362c02d04cab2d55 |

审阅根目录 R=`D:\Program Files\UserCache\gb\codex\tmp\sttrack_m82_m84_development_pr_20260921`。M82 原来源根为 `D:\Program Files\UserCache\gb\codex\tmp\m82_complete_20260920`；M84 为 `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920\completed`。历史对照位于 `C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\m68\historical_score_reference\result.json`。其余每份输入的完整路径和 SHA，以及复算产物的 SHA，见本报告 JSON 与 reviewer_recompute.json。
