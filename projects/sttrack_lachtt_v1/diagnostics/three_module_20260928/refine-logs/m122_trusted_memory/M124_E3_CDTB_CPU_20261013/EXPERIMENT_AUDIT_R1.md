# M124 epoch-03 CDTB 保存轨迹 CPU 诊断：独立完整性审查 R1

日期：2026-10-13（Asia/Shanghai）。总体 **WARN**；在“本地源码语义、捕获结果一致性、stdlib 算术复核”这个限定范围内，**阻塞数 0，FAIL 数 0**。这不是原始服务器数据独立复评、运行时因果审计或最终全九项验收，也不授权把任务标为完成。

审查路由：`review_independence=same-family`、`acceptance_status=provisional`。skill/policy 要求的请求模型为 `gpt-6-astra`、请求 effort 为 `max`；本审查上下文没有可核验的运行时模型/effort 回执，二者的 actual 均记 **UNATTESTED**。不能由请求值推断真实运行时。原调用 envelope/response 元数据由父调用方保存。

只进行了本地文件读取、源码/AST 检查和 Python stdlib 计算。没有执行 SSH wrapper、网络请求、GPU 查询、NN、torch/numpy/cv2/VOT 模型或包导入，也没有修改待审输入、训练或其他项目。初始完整 prompt 与两次追加路径消息分别保存在指定私有 trace；此报告纳入了封存前收到的 `selected_onsets.json`、native binding 和 wrapper。

## 证据定位约定

以下路径都相对于 `C:/Users/gb/.codex_track_publish_m29_20260902/`。引用中的行号是包含空行的实际物理行号。三个大型原始 JSON 均只有一行，故使用 `:1` 加 JSON 键路径定位；不把展开后的展示行号当作原文件行号。

| 缩写 | 文件 |
|---|---|
| E3 | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_m124_epoch03_cdtb_saved_trajectory.py` |
| E2 | 同目录 `analyze_m124_epoch02_cdtb_saved_trajectory.py` |
| CMP | 同目录 `compare_m124_cdtb_epoch02_epoch03_saved_trajectory.py` |
| METRIC | `projects/sttrack_lachtt_v1/diagnostics/native_ope/source_snapshots/depthtrack_pr.py` |
| RAW3 | `.aris/m122_full_causal_20261007/M124_epoch03_CDTB_CPU_diagnostic_original_stdout_20261013.json` |
| RAW2 | 同目录 `M124_epoch02_CDTB_CPU_diagnostic_original_stdout_20261012.json` |
| NATIVE | 同目录 `M124_epoch03_CDTB_CPU_diagnostic_native_receipts_20261013.json` |
| BIND | 同目录 `M124_epoch03_CDTB_CPU_native_binding_20261013.json` |
| WRAPPER | 同目录 `M124_epoch03_CDTB_CPU_diagnostic_native_20261013.py` |
| OBS | 同目录 `M124_twentythird_hourly_original_observation_stdout_20261013.json` |
| SUMMARY | 同目录 `M124_epoch03_CDTB_saved_comparison_20261013/summary.json` |
| CSV | 同目录 `M124_epoch03_CDTB_saved_comparison_20261013/all80_sequence_diagnostics.csv` |
| ONSETS | 同目录 `M124_epoch03_CDTB_saved_comparison_20261013/selected_onsets.json` |
| TRACKER | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md` |
| MASTER | `docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md` |

全部原始输入及补充源码的绝对路径、SHA-256、字节数、物理行数收录在同名 JSON 报告中。MASTER 的本次声明审查聚焦相关 M124 章节及 §5.506–5.513；不声称验证了整本历史文档的所有实验。

## A–F 检查

| 检查 | 状态 | 证据与结论 |
|---|---|---|
| A. GT 来源 | **WARN** | E3:20、26、48–58；E2 同行；METRIC:40–60、95–113。GT 读取 `dataset_root/sequence/groundtruth.txt`，与 cases 的 `gt_sha256` 比较；预测和分数来自保存 TXT，没有用预测生成 GT。`visible` 的含义是 GT 有限且宽高为正，其余转 `Special(0)`，不能直接称为真实物理缺失。本地未独立读取这些远端 GT/cases/image 原始字节，故来源可由源码和捕获支持，但不能给原数据独立认证 PASS。 |
| B. 分数归一化 | **PASS** | E3:59–61、73–76；METRIC:25–37、117–151。P 是被阈值保留框的 IoU 均值，R 是保留框 IoU 总和除以有效 GT 数，然后按序列等权平均；F 由宏平均 P/R 调和得到。未发现除以模型自身最大/最小输出以抬分。阈值从 confidence 的排序采样并按全数据最佳 F 选取，是此处报告协议，不是异常的分数归一化。 |
| C. 文件、键、状态、数值 | **WARN** | RAW2:1、RAW3:1 的 `status/aggregate/per_sequence/inputs_sha256`；SUMMARY:2–31、516–534；CSV:1–81；OBS:1 的 epoch-02/03 `metrics.json`；NATIVE:2–10；BIND:3–31。全部本地输入存在且 JSON/CSV 可解析；80 行和汇总复算通过，六份历史 metrics 捕获文本与 history 哈希一致。`complete` 是已捕获 OPE/诊断状态；TRACKER:179、MASTER:28470、28474、28480 仍保留全任务未完成。原始 plan/receipt/checkpoint 字节及 native runtime 仍有下述证据边界。 |
| D. 计分调用可达性 | **PASS** | E3:24–25、52、57、70、73–80；CMP:13–79、82–83。诊断实际路径依次加载行、调用 `_vot_overlaps`、分段、聚合、校验正式点并打印。`run_m124_epochwise_suite.py:48–57` 调 `ope_analyze`；`run_m124_epochwise_official.py:45–61、81–84` 实际调用 `evaluate_depthtrack_results`；METRIC:117、139 连接阈值扫描与最佳 F 结果。没有支撑这些指标的不可达计分函数。`format_metrics` 只是格式化工具，不承担本次计分。诊断复算封存阈值点，本次审查没有重新跑完整阈值扫描或原始 overlap。 |
| E. 范围与声明 | **WARN** | E3:77；CMP:67；SUMMARY:534；ONSETS:221；TRACKER:113、159、179；MASTER:28390、28470–28474、28478。实际是同一训练链连续两个 epoch 的两份自身历史 CDTB80 轨迹，各 101956 帧；不是两次独立 seed，也不是同状态、同候选或文本/模块干预。Test50/CDTB80 用于逐轮 checkpoint 选择已披露，不能称未参与选择的最终泛化测试。尚无本次 M124 VOT/全九项联合通过。现有诊断限定语恰当，WARN 是证据适用范围，未发现本次材料把它写成完整验收。 |
| F. 评价类型 | **PASS** | E3:48–61、77；METRIC:49–60；RAW3:1 `evaluation_type`；SUMMARY:529。分类为 **real_gt**：基于数据集 GT 评价已保存预测。crop/持续段是此类真实 GT 上的几何描述，不是 synthetic/self-supervised proxy，也不是新运行的恢复实验。人工核验文字是模型输入来源，不使该指标成为 human_eval。此分类不消除 A/C 的来源证据限制。 |

## 源码、输入一致性及 native 回执

本地 SHA-256 与 RAW2/RAW3 声明完全一致：

| 源码/结果 | SHA-256 |
|---|---|
| E3 | `8f6c08fc5aa4d9f0119774622e3968108c978dd7d7a5f99d7541b71c7796ef6a` |
| E2 | `9ff1df469a6f58b17cba2de34410be0031af552146576addbff8019e2d51f32c` |
| CMP | `d9b0d8037490eb3fe113fb27d36d9ff78ee5ff7d923a08774eeab5acbb95d193` |
| METRIC | `05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc` |
| RAW2 | `aa54318793d5666fa8c5aa7cc831ea6916a0dbed37082a9e2dfdea21b47a90b9` |
| RAW3 | `04fc5990f7fe6955d464a16e5a2243c2823c8e04095c7aa379b10085d35a79f9` |

把 E2 中的 epoch 标签、FINAL、SAVED 替换成 E3 对应元数据后，两份分析源码逐字相同，计分/GT有效性/crop/分段计算没有额外差异。80 个 sequence 的顺序、帧数、有效 GT 数、GT 哈希、首帧图像尺寸均一致；两个 `inputs_sha256` 的共同 162 项为 cases、metric、80 个 GT、80 个首帧图像，全部声明哈希一致。共同 cases 哈希为 `b3170466a090fa1a8949e0297566c9e3ab744e789ca0c7f1bcc71633ccfae887`。两轮均有 **91300 有效 GT / 10656 无效 GT 帧**，GT 有效性恢复事件的帧索引也逐序列相同。这是两份捕获之间的一致性，不是对这些 162 份远端文件重新取哈希。

每轮 325 项输入绑定严格对应 5 个总文件（plan/cases/metrics/receipt/metric）加每序列 bbox/score/GT/首图各 1。全部行的 bbox/score/GT 哈希与各自 map 对应。OBS:1 中 E2/E3 CDTB `metrics.json` 文本以原 UTF-8 编码重算 SHA 分别为 `129a2af0c7f6586c1bee42ae5d27a6e7a9e69f005341bec192d009d8f5d5f07a` 和 `808b1b263c94b58a84fdfe3b30f7ca7e3f3ec9ee0d7a289a51773b81b4cd6452`，与诊断常量、捕获 map 和 history 相符。P/R/F 与诊断聚合的差异仅浮点归约末位，均小于 `1e-10`。E3 final 字段为 `3b9e613b4af8c3a2f009200ae8ecb6facf94b56b2033647ef344f08be07175af`。

**原始字节边界必须保留：**E3:35、38 是 final/bundle 哈希字段之间的比较，诊断没有读取 checkpoint 或 bundle 原文件；325 输入清单也不含 checkpoint/bundle/bank 原始字节。本地审查没有独立读取原 bbox/score/GT/首图/cases/plan/prediction receipt。不得把“远端 CPU 输出捕获 + 本地哈希/算术一致”改写成“审查者已经独立复核这些原始服务器字节”。METRIC 的本地快照已校验，实际安装的 `vot.region.calculate_overlaps` 依赖版本/字节未独立获取；本报告不另外认证官方工具的完整安装与运行环境。

NATIVE:2–10 记录 session **44398**、invocation chunk `6fec66`、authentication chunk `f45084`、terminal chunk **185b26 / exit 0**。原 terminal 返回只有控制台复位字符，没有诊断正文，也没有 `session_id` 字段。封存前追加的 BIND:3–31 提供完整本地 wrapper 调用、join request 的 session 44398，以及 wrapper/stdout/source 哈希；这些值与原 NATIVE 逐项相同。WRAPPER 哈希为 `275a94a22203ba417b750a2f961e5e7605f5d12d8856b96ac557ca73552470f0`。

对 WRAPPER:3 只做 AST/base64/zlib 解码，没有执行它：内嵌分析源码逐字等于 E3；远端 launcher 校验写入前后 SHA 后 `runpy.run_path(..., run_name='__main__')`。WRAPPER:4–6 以 `xb` 捕获 stdout 并传播子进程 returncode；命令设置 `CUDA_VISIBLE_DEVICES=''` 和单线程 CPU 环境。**调用者记录的命令—session—stdout—源码绑定一致性 PASS；宿主真实运行时认证仍 UNATTESTED。**BIND:26、31 明确保留 terminal 无 session_id 的事实，不能补造终态字段。新增绑定改善的是调用者留存链，不是新得到的原始预测/GT/checkpoint 证据。

RAW3 的 CPU 零 NN/GPU/训练进度查询字段与受审分析源码相容，但属于源码与捕获层面的核对；OBS:1 本身另有 `GPU_queries=1`、`NN_progress_queries=1`，不应把 CPU 诊断的零计数推广成整个历史观察过程均为零。

## 独立算术结果

使用单独的 stdlib verifier 从 RAW2/RAW3 原行构造结果，没有执行 E2/E3/CMP。**CSV 80×28=2240 个单元逐值完全相同；SUMMARY 的两组各 8 条排序记录共 448 个单元完全相同。**所有 summary 计算字段、输入哈希、计数、序列映射都通过。证据：CMP:27–67；CSV:1–81；SUMMARY:5–31、32–515、516–528。

| 项目（% 或 pp） | Epoch 2 | Epoch 3 | E3−E2 |
|---|---:|---:|---:|
| 正式 P | 74.403461552958 | 71.856294048470 | −2.547167504488 |
| 正式 R | 68.729441474690 | 72.256096876543 | +3.526655401853 |
| 正式 F | 71.453987841364 | 72.055640887358 | +0.601653045993 |
| 全部保存框 R | 70.214997710010 | 73.089458480645 | +2.874460770635 |
| 报告分数筛选 R 损失 | 1.485556235320 | 0.833361604102 | −0.652194631219 |
| 数据集最佳 F 的报告阈值 | 0.318970 | 0.013903 | 各轮单独选择 |

`ΔR = ΔR_all − Δloss = 2.8744607706345846 − (−0.6521946312188049) = 3.5266554018533895 pp`。

全部保存框 R 仍比目标 75.6% 低 **2.5105415193553426 pp**。这是固定预测框、固定后续状态下，对仅改变“报告时分数过滤”的 R 上界。它没有生成替代框、选择 GT-best 候选或得到可部署 P/F；如果改变分数同时改变搜索/模板/后续状态，则超出这个上界适用条件。不能据该算术把“定位轨迹整体累计重叠改善”进一步指定为 A、B、C、文字语义或几何分支的独立原因。

宏平均贡献按每条 `ΔR_all/80` 计算，**51 条提高、29 条下降、0 条相等**。最小绝对变化约 0.00324517 pp，本次符号计数并非仅由 `1e-14` 归约误差造成；仍无显著性检验或跨 seed 稳定性证据。个别行筛选损失为约 `−1e-14` 是两种求和的浮点末位，本次按 `1e-10` 复核容差处理，不能解释为真实负损失。

最大 8 条损害的宏贡献合计 **−1.6410116600332052 pp**，其余 72 条 **+4.51547243066779 pp**；最大 8 条收益合计 **+3.94407159582604 pp**，其余 72 条 **−1.0696108251914556 pp**。两种分组都回到净 +2.8744607706345846 pp，没有把大收益例子当作全部序列。代表行：`humans_corridor_occ_1` CSV:47，`jug` CSV:53，`trashcans_room_occ_1_A` CSV:73，`person_outside` CSV:57，`boxes_humans_room_occ_1` CSV:36。

## 持续低 IoU、crop 和追加案例

定义由 E3:62–71 给出：**有效 GT 且 IoU≤0.1**，无效 GT 会断段；保留长度至少 10 的连续段；索引为 **0-based、[start,end)**。分段起点只是满足这一定义的开始，不能自动视为第一次因果分叉。依据 RAW2/RAW3:1 `/per_sequence` 逐段核对了范围、顺序、长度、均值范围、中心计数及事件索引，并独立相加；没有从未提供的原始轨迹重建每个 mask。

| 统计 | Epoch 2 | Epoch 3 |
|---|---:|---:|
| 严重低 IoU 帧 | 9795 | 7674 |
| 其中 GT 中心在重建 crop 外 | 7897 | 5705 |
| 其中中心在内 | 1898 | 1969 |
| 至少 10 帧持续段 | 97 | 80 |
| 持续段总帧数 | 9486 | 7280 |
| 持续段起点中心在内 / 在外 | 35 / 62 | 43 / 37 |
| 起点在内且段内后来在外 | 17 | 19 |
| 有效 GT 恢复事件 / 中心在外 | 159 / 84 | 159 / 57 |

SOURCE 支持 factor 4 方形裁剪：`full_dense_tracker.py:24–30、53–54` 与 `lib/train/data/processing_utils.py:25–39` 的 ceil/sqrt/round 几何一致。`trusted_template_tracker.py:30–45` 用自身上一状态并更新后续状态；`run_m124_epochwise_official.py:33–35` 保存为 **6 位小数**。因此以保存框重建 crop 是有源码依据的几何描述，**不是原未舍入状态/特征/query/template 的逐位回放**，更不是 observation BCE、目标整框可见或候选可用性的等价标签。这里只读追踪本地实现，未认证远端实际导入依赖的字节。

ONSETS:2–222 的 4 个 panel、两 epoch 共 **56 个顶层字段（含完整嵌套段列表）**与原行完全相同；输入 SHA 也一致。

- 推车：ONSETS:5–97，CSV:39。两轮首个长段都从 395 开始，E2 `[395,546)` 151 帧，E3 `[395,415)` 20 帧，之后仍有 `[468,546)` 78 帧；两次 E3 起点中心都在内，且之后出现外部帧。后面三个段 `[556,593)`、`[607,624)`、`[685,999)` 仍存在。全框 R 只从 32.671561 升至 33.373571，不能称已修复，也不能移植旧 CurrentC 的 175→176 跳框作为 E3 实测首次原因。
- 背包：ONSETS:100–127，CSV:35。E2 有 137 个严重低 IoU 帧，E3 为 0，全框 R 60.201192→72.047342。这支持该阈值定义下严重错误消失，不支持“所有帧完全正确”。
- jug：ONSETS:130–173，CSV:53。548→12 个严重低 IoU 帧，E3 没有 ≥10 帧段；不能把“没有长段”改成“没有低 IoU 帧”。
- robot：ONSETS:176–211，CSV:59。1084→0 个严重低 IoU 帧，全框 R 21.407647→74.387899；仍是自身历史轨迹比较，没有同状态候选/文字/模块干预证据。

## 声明影响及后续证据要求

本报告支持“已捕获的 E3 CDTB80 全保存框 R 比 E2 增加 2.874461 pp，正式 R 增加还包含报告筛选损失减少，持续低 IoU 总量下降且仍有局部损害”。不支持部署 oracle、真实物理缺失、唯一神经根因、模块/文字独立贡献、全部错误修复、独立未选择测试集或同一最终权重的完整九项通过。

OBS:1 六份指标文本及 `epoch_history.json` 的 F 均值/选模算术一致：E3 两 F 均值 68.38346083616585%，仍低于 E1 68.5415903449273%；E3 相对 E2 是一项 F 降、一项升，best 仍 E1，dual_decline=false/streak=0。这些是捕获状态，不能把本文 E3 CPU 诊断提升为最终选中权重的验收。

限定范围内不要求修改诊断或比较实现。报告新结果时继续保留上述限制；若后续要升级为原始数据独立复评，需要实际获取并校验原预测/分数/GT/cases/plan/receipt、完整计分依赖与原始权重产物链；若要讨论模型机制，需要另行取得同状态受控或候选回放证据。本次未执行这些工作，也没有将其算作已通过。

## 审计留痕

私有目录：`.aris/traces/experiment-audit/2026-10-13_m124_epoch03_cdtb_cpu/`。

- `reviewer_prompt.txt` 为初始完整 prompt；追加路径另存 `002-additional-path.request.txt`、`003-native-binding.request.txt`，未改变初始 prompt。
- `verify_saved_capture_stdlib.py` / `stdlib_verification.json`：实际本地执行退出 0，tool chunk `84973c`，复核 CSV/聚合/段/追加案例。
- `verify_native_binding_stdlib.py` / `native_binding_verification.json`：实际本地执行退出 0，tool chunk `9e16af`，只解码和比较 wrapper 源码，没有执行它。`decoded_remote_launcher.txt` 为解码后只读源材料。
- `E:/python.exe` 的实际版本输出为 Python 3.12.6。最初默认 `python` 指向失效的 `E:/Scripts/python.exe`，以 `No pyvenv.cfg file` 退出，未读写实验输入；随后使用已存在的解释器，没有安装依赖或改动实验。
- 所有被纳入证据表的输入在封存前再次核对 SHA；没有覆盖或修改待审文件。父调用方应保存本报告原文及实际可获得的调用元数据，缺失字段保持未知。

**最终审查结论：WARN，限定范围阻塞 0；same-family / provisional；actual model/effort UNATTESTED。最终验收未执行。**
