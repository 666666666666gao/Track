# Experiment Audit — R3 actual terminal raw results

日期：2026-10-10。审查者：fresh native Codex agent `/root/m122_r3_actual_terminal_result_integrity_20261010`。请求 `gpt-6-astra` / `max`；实际模型、后端、effort 均 **UNATTESTED**；`same-family`、`provisional`，不宣称跨族接受。

**总体 WARN；blocking_count = 0（仅针对归档已核验终态结果）。`R3_terminal_raw_audit_complete = true`。科学推进门失败，不能据此启动 R4、新 Full152 或正式评价。**

A = `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007`；P = `C:\Users\gb\.codex_track_publish_m29_20260902`。本报告中的行号按原 UTF-8 文件计数。原始请求在 `ORIGINAL_REVIEW_REQUEST.txt`，本报告全文同时保存在 trace response。

## 独立结论与实际范围

原始257文件（221,114,638字节）审前/审后全部匹配原清单；ZIP67成员逐项匹配，66个导出数据文件加导出清单自身。另读取9个原有R2原始补充文件（123,492,666字节），与旧审计的原输入hash绑定并审后未变。已有审查只用作来源/历史范围信息，未以其PASS替代本次复算。

独立检查219,802条完整教师前缀的时序与写入规则；三组各32序列、49,366个非初始化帧均逐帧检查，其中46,088帧有有效GT、3,278帧为未知GT。有效GT仅由数据标注有限值与正宽高判定，初始化帧不计入；未知GT不记零准确率、不计低IoU段。独立标量IoU与H10扫描、全部逐序列/汇总数值及写入计数均一致，浮点比较容差绝对1e-12/相对1e-13；实际最大逐序列IoU_sum误差 4.547473508864641e-13、mean_IoU误差 2.220446049250313e-16，整数/区间不一致0。

| 轨迹 | macro IoU | frame IoU | IoU≤0.1帧 | H10段 | qualified / accepted | 未知GT写入 | 低IoU写入 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.764334383531 | 0.740338754077 | 6955 | 63 | 277 / 277 | 13 | 27 |
| current | 0.774631107471 | 0.755051096280 | 6205 | 60 | 265 / 255 | 8 | 7 |
| future | 0.762334417857 | 0.746099497196 | 6702 | 62 | 269 / 139 | 9 | 8 |

`native`是固定P1第三pass A+B decoder（final SHA `3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811`）+固定STTrack/CLIP/人审文本、采用50帧且同选中位置原生响应>0.75的写入规则，**不能标成原版STTrack基准成绩**。来源 `P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\run_template_write_pilot.py:22`、`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\full_dense_tracker.py:92`；实际两臂input字典和该P1/hash一致。

Future−native macro = -0.001999965674，Future−current = -0.012296689613；两个预定的“严格优于”都失败；H10 62≤63通过。Current−native = 0.010296723939（约+1.0297个百分点），32条中3条改善、0条下降、29条完全相同；Future相对native为14改善/8下降/10相同，相对Current为13/9/10。该结果如实支持“本次未来监督增量未建立”，并非造假或缺失结果。预定规则 `P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_PLAN_20261008_235856.md:64`；原结果失败标志 `A\R3_terminal_actual\comparison\result.json:2217`。

写入分母逐臂取各自实际历史：native277/277=100%，Current255/265=96.2264%，Future139/269=51.6729%；对应原规则不具备资格的帧49,089/49,101/49,097，qualified占全部非初始化帧约0.5611%/0.5368%/0.5449%。qualified中的GT有效数264/253/257，未知数13/12/12。不能将277统一充当其他臂分母，也不能用模型输出最大值归一。资格事件(n,frame)交集为native/current264、native/future234、current/future233；同帧键相交不保证改变历史后的状态相同。

## A–F 清单

### A_gt_provenance — PASS

All152 supplied groundtruth files match the frozen training spec; all32 development trajectories use the dataset annotations, excluding initialization and invalid GT. All1297 current/future teacher labels are independently recalculated from dataset GT and exported W/K boxes. Preflight equality uses model outputs as a disclosed proxy, not accuracy.

证据：`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:140`；`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:29`；`A\R3_terminal_actual\code\train_template_write_C.py:68`；`A\R3_terminal_actual\code\train_template_write_C.py:55`；`A\R3_terminal_actual\code\run_template_write_C_preflight.py:186`。

### B_score_normalization — PASS

IoU uses geometric intersection/union; frame mean divides by valid dataset-GT frame count, macro by 32 sequences, action rates by explicit own-history qualified-event counts. H10 counts contiguous valid-GT IoU<=0.1 runs of at least10 frames; unknown GT breaks runs. No own-output max/min normalization. Feature L2/motion transforms are inputs, not performance denominators.

证据：`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:153`；`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:46`；`A\R3_terminal_actual\code\template_write_features.py:38`。

### C_existence_numeric_completion — WARN

257 original input hashes, 67 ZIP members, all6 stages, aggregate/supervisor exits and metrics agree. A separate 9-file raw R2 supplement is hash-bound to the sealed prior audit and independently read. Historical tracker table/master still predate terminal results; update the current status after archiving this review. Remote model/dependency and initializer tensor limitations remain explicit.

证据：`A\R3_terminal_actual\code\m122_R3_process_supervisor_20261010.py:20`；`A\R3_second_hour_observer_original_stdout_20261010.txt:2`；`A\R3_second_hour_observer_actual_native_receipts_20261010.json:2`；`A\R3_terminal_actual\fit_current\result.json:11`；`A\R3_terminal_runtime_integrity_audit_20261010\FIT_PREFLIGHT_RECOMPUTED.json:26`；`A\R3_terminal_runtime_integrity_audit_20261010\BINDINGS_CHECKED.json:227`；`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_TRACKER.md:10`；`P\docs\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:27105`。

### D_actual_calls_dead_code — PASS

Main controller calls preflight, two fits, two complete development workers and comparison; native launch receipts/log outputs match these commands. Overlaps is used by summarize, summarize is called for all3x32, fixed_panel is called for both arms. Source dependencies contain older pilot/candidate-coverage entry points that are not invoked as R3 performance metrics; their definitions are not claimed as executed R3 results.

证据：`A\R3_terminal_actual\code\m122_R3_C_controller_20261010.py:127`；`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:149`；`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:159`；`A\R3_terminal_actual\code\evaluate_template_write_C.py:54`。

### E_scope — WARN

Single seed2027, one paired fit, one DepthTrack Train-derived C optimization holdout of32 sequences. A+B saw all152. Of120 C-fit partition sequences,113 supply993 eligible events;29 of32 development sequences supply264 common fixed-teacher events. Own-history performance covers all32. No formal Test50/CDTB80/VOT127 metrics or whole-model unseen validation are certified.

证据：`A\R2_terminal_actual\inputs\C_split.json:11`；`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_PLAN_20261008_235856.md:26`；`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_PLAN_20261008_235856.md:64`；`A\R3_terminal_actual\code\train_template_write_C.py:164`。

### F_evaluation_type — PASS

- own_history_native_current_future: real_gt (custom Train/C development tracking metrics)

- fixed_teacher_current_MSE: real_gt-derived current-IoU surrogate regression

- fixed_teacher_future_MSE_and_sign_panels: real_gt-derived short-horizon action-utility surrogate, only at original teacher states

- constant_action_preflight: synthetic_proxy (reference predictions/query/features from the same fixed tracker)

- text_bank_and_state_hash_checks: self_supervised_proxy / deterministic identity checks, not semantic contribution

- formal_three_dataset_performance: not evaluated by this R3 audit

证据：`A\R3_terminal_actual\code\analyze_template_write_C_dev.py:140`；`A\R3_terminal_actual\code\train_template_write_C.py:164`；`A\R3_terminal_actual\code\run_template_write_C_preflight.py:186`；`A\R2_terminal_actual\inputs\C_split.json:11`。

## 共同拟合、固定教师与自身历史

两臂使用相同519维输入与70721参数结构。1297原教师事件的当前IoU及W/K最多32帧未来有效GT均值差独立复算，78,594个有效未来W/K IoU数值核对通过。共同合格1257事件，993 fit来自113/120拟合分组序列，264 fixed-teacher development来自29/32开发分组序列；排除当前GT未知35、未来GT不可用5。fit正/负/零494/423/76，dev113/126/25，未删除负/零样本。每个C输入当前128/过去128/初始256/质量3的实际数值和hash匹配，motion4独立重算最大误差2.9802322387695312e-08；人审bank及已导出初始文本/框对应，不重新声称执行过人工核验。

共同dataset SHA `5894dd48e03ca466cddd56311e8efbf31a9c4d979897b35d588ae10bc3ce5dc9`；seed2027排列的order SHA `f6407db612286d2ce5568b7ac0b03c5929f034a9f9dbc9c3af19a3d509e99b3f` 均独立重建一致。两臂各固定10 epoch、batch32、AdamW lr3e-5/weight_decay0.01、320步；10行原epoch日志、终态结果和launch/exit相符。训练代码只将fit组交给优化器，固定final，无best选取；固定教师开发损失仅观测。证据 `A\R3_terminal_actual\code\train_template_write_C.py:123`、`A\R3_terminal_actual\code\train_template_write_C.py:140`、`A\R3_terminal_actual\code\train_template_write_C.py:164`。

**初始化未独立重建通过：** 两臂真实回执共同摘要 `d83b325dfdc6163a20bb7cc5bcb710f05e3b36816d82f954f7de772b254d65c8`；本地Windows Python3.7.6 / Torch1.13.1+cpu seed2027构造得到 `94c2419332b132d5b8ce8fa1cf06626494d67f91b127f65e312f966aa65e6815`。原始初始权重没有导出，差异原因未确定；本报告不把“平台差异”写成已证实原因。相同初始化有固定源码和两臂相同运行摘要支持，不能升级成独立初始张量完全相等证书。两次严格重建检查的exit1保留；后续退出0表示其他已列项目核验完成，并没有将这项改成PASS。证据 `A\R3_terminal_actual\fit_current\result.json:11`、`A\R3_terminal_runtime_integrity_audit_20261010\FIT_PREFLIGHT_RECOMPUTED.json:26`。

两个实际final.pt CPU加载均6个float32 state项、70721参数；文件SHA和state digest与开发前后摘要一致。独立CPU线性代数计算264个固定教师预测/臂，共528个，最大GPU记录差Current2.384185791015625e-7、Future7.450580596923828e-8（检查上限1e-6），阈值动作不一致0。Current原权重SHA `60d0a200941801bf4097c25311200885c9c9bf6263dc503a29ce91b2e5b12259`；Future `9020c67d50bcddfa48ab004d39cd093424ce0c1a203f6e0165b818d43f15ac48`。

| 固定教师dev264 | 保存预测独立MSE | 正收益接受 | 负收益接受 | 零收益接受 | 全部接受 |
|---|---:|---:|---:|---:|---:|
| Current | 0.031545851130 | 113/113 | 126/126 | 7/25 | 246/264 |
| Future | 0.000538952640 | 60/113 | 70/126 | 9/25 | 139/264 |

补充确定性诊断：Future同一264标签的恒零预测MSE 0.000159181205，低于该模型0.000538952640；它只帮助解释当前固定教师拟合，不能视为部署基线或用来事后改变既定门。Future保留正收益60/113、误接纳负收益70/126，不能仅因写入更少就称“更安全”。上述标签属于原教师状态，**不能转贴到改变策略后的自身历史**。own-history Future接受139恰好与fixed-teacher的139同数，二者分母269与264、状态集和含义均不同。

动作代码先提交同一选中候选box/query/feature，再按原资格与Current>0.5 / Future>0决定是否替换模板，没有相乘定位分数、改本帧候选排序或GT控制输入。所有自身历史scalar记录检查previous_bbox链、frame顺序、selected范围、资格、C_prediction空值约束和严格阈值；源码同候选断言位置见 `A\R3_terminal_actual\code\trusted_template_tracker.py:44`、`A\R3_terminal_actual\code\template_write_C.py:24`。所有学习臂的完整candidate/query/feature张量未导出，因此这里不宣称逐帧神经输出重建。

## 终态、预检和证据层次

实际六阶段preflight/fit_current/fit_future/development_current/development_future/comparison全exit0；controller.exit、supervised.exit与监督器terminal也为0。控制器结果结束于2026-10-10T12:35:27.392699Z，监督器12:35:27.410128Z；原第二小时observer在13:19:36.400301Z记录进程已无cmdline。原生observer session65866 terminal fcd80e exit0、原timer18966 terminal be3592 exit0与原stdout/同句柄join一致。此处核验提供的原始回执，不重新查询远端或声称独立访问历史工具服务。215项cross-file/source/launch/log/hash检查通过。

preflight仅cube04_indoor的256帧：original与constant1两条保存历史共512步，加frame250的constant0拒绝分支1步，合计513 calls。256对共有原record字段、query与selected_feature张量完全相等，原始记录还与教师prefix逐帧一致；唯一拒绝事件的本帧box/query/feature保留、C输入相同并与教师519维输入一致。它是接口一致性synthetic_proxy，不能证明训练C的准确率。保存的原始preflight文件没有全部before-template快照及所有模型out张量，因此“旧模板保留/全部out一致”的完整证明仍包含源码实际断言层，不伪称全部本地逐张量重算。

冻结STTrack/CLIP/A+B前后摘要各阶段相符；小C权重后验state digest与before/after直接对应。完整视觉网络/图像/GT线上获取没有重跑；90项gate中的24项本地字节匹配，66项仅由原运行gate断言见证，逐项范围在 `GATE_BINDING_SCOPE.json`。这些明确限制不妨碍已提供预测的真实GT数值复算，但禁止扩大为全部远端模型依赖或完整神经执行再现认证。

## 非阻断限定与后续动作

- **N1 Initial tensors are not independently reconstructed**：Both remote arm receipts record d83b325dfdc6163a20bb7cc5bcb710f05e3b36816d82f954f7de772b254d65c8. Fresh local Windows Python3.7.6/Torch1.13.1+cpu seed2027 construction yields94c2419332b132d5b8ce8fa1cf06626494d67f91b127f65e312f966aa65e6815. The cause is unestablished; no original initial tensor file was exported. This mismatch is not reclassified as an exact PASS. Common initialization is supported by the matched actual digests and source seed placement, but independent initial-tensor equality/reconstruction is not certified. It does not change directly recomputed terminal predictions or the failed Future advancement gate. 证据 `A\R3_terminal_actual\fit_current\result.json:11`；`A\R3_terminal_runtime_integrity_audit_20261010\FIT_PREFLIGHT_RECOMPUTED.json:26`。

- **N2 Full runtime tensor and remote dependency scope**：Only the small final C states, human text bank and saved preflight histories are loaded locally. Full visual models/images and original R2 PT cache files are not reconstructed. 24/90 gate entries match supplied local bytes;66 are runtime gate-assertion witnesses. Learned own-history logs omit519-D features and full candidate/query tensors; all scalar state chains/actions are checked, while same-candidate/query semantics and freezing beyond the saved preflight arrays rely on pinned source and actual recorded assertions/digests. Preflight has only cube04 frames1..256 and one eligible event at250; prior template/full-output tensors are not all exported. 证据 `A\R3_terminal_actual\code\trusted_template_tracker.py:44`；`A\R3_terminal_actual\code\evaluate_template_write_C.py:68`；`A\R3_terminal_actual\code\run_template_write_C_preflight.py:186`；`A\R3_terminal_runtime_integrity_audit_20261010\BINDINGS_CHECKED.json:227`。

- **N3 Engineering development scope and proxy distinction**：No multi-seed or formal-three-dataset robustness claim. Teacher future targets are actual GT-based W/K utility labels, but apply only to original states; they cannot label changed C-own-history states. Current gains and Future failure are descriptive within C-dev32. No learned identity, language semantics, depth reliability or crop-out recovery claim follows. 证据 `A\R2_terminal_actual\inputs\C_split.json:11`；`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_PLAN_20261008_235856.md:64`；`A\R3_terminal_actual\code\train_template_write_C.py:164`。

- **N4 Tracker/current narrative lag**：The bound tracker R3 row still says SOURCE_PASS and latest appendix/master5.453 says development running. These are stale intermediate statements, not evidence that terminal files are missing. Update the current tracker after sealing this audit; preserve old review inputs and historic status records. 证据 `P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_TRACKER.md:10`；`P\docs\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md:27105`。

可以归档本次两臂权重、完整raw预测、精确失败条件和逐序列损益，更新当前Tracker/main handoff为实际终态。可以据此形成待审核的Current质量C后续方案或检查Future标签/预测/状态分布；现有冻结计划的R4路径以Future B2通过为前提，该条件失败。此审查不自动将Current晋升为替代分支、不授权新Full152/新正式NN运行，也不声称三数据集九项、语言贡献或整体目标达到。`P\projects\sttrack_lachtt_v1\diagnostics\three_module_20260928\refine-logs\m122_trusted_memory\EXPERIMENT_PLAN_20261008_235856.md:68`。

## 实际工具退出与失败保留

最终确定性指标检查 `metrics_attempt2` exit0；权重/教师/预检限定检查 `fit_preflight_attempt3_scoped` exit0；215项接线核验 `bindings_attempt2` exit0；审前/后输入核验与最终evidence检查 exit0。所有这些实际stdout/stderr、命令、时刻、退出码和输出hash均以同名receipt保存。没有SSH/SCP/GPU查询、训练、完整视觉网络前向、依赖安装、新代理或输入改写；仅CPU小C线性计算。三次fit/preflight核验尝试各有一次seed初始化构造，共三次；仅最后一次继续到两批共528个固定教师CPU预测。

原六次失败均保留：metrics_attempt1的审查脚本语法错误（exit1，修正独立脚本后复算成功）；bindings_attempt1使用当前Python3.7不支持的ast参数（exit1，改为本机语法解析，不虚报Python3.8目标编译）；fit_preflight_attempt1与attempt2_diagnostic为上述初始化摘要不匹配（均exit1，未强行证明一致）；report_generation首次引用源码路径少了一层父目录、第二次字符串格式化括号遗漏（均exit1，修正报告生成器）。这些不是原实验阶段失败；原实验各阶段exit0与科学Future门失败分别记录。失败源文件与完整stderr位置见 `EVIDENCE_FINAL_CHECK.json`；历史失败未删除或覆盖。

## 全部32条逐序列结果

以下为GT有效非初始化帧均值；H10列依次native/current/future。完整96行IoU_sum/有效帧/写入/区间见 `PER_SEQUENCE_RECOMPUTED.csv`，每帧IoU、valid/write/qualified数组见 `METRICS_ALL_FRAME_VALUES.npz`（每个数组4×N，顺序为IoU、valid、write、qualified）。

| sequence | native IoU | current IoU | future IoU | H10 N/C/F |
|---|---:|---:|---:|---:|
| skateboard02_indoor | 0.904139342 | 0.904139342 | 0.903962425 | 0/0/0 |
| bottle02_indoor | 0.651703243 | 0.651703243 | 0.497243719 | 3/3/4 |
| bottle01_indoor | 0.809338788 | 0.809338788 | 0.816500216 | 3/3/2 |
| mushroom02_wild | 0.914317542 | 0.914317542 | 0.913214693 | 0/0/0 |
| skateboard01_indoor | 0.344860057 | 0.344860057 | 0.434337994 | 5/5/4 |
| book01_indoor | 0.850585993 | 0.850585993 | 0.851520084 | 0/0/0 |
| glass04_indoor | 0.804507449 | 0.804507449 | 0.810824413 | 1/1/1 |
| parkingsign_wild | 0.893158963 | 0.893158963 | 0.938801994 | 1/1/0 |
| bottle06_indoor | 0.861003928 | 0.861003928 | 0.856533391 | 0/0/0 |
| toy08_indoor | 0.846106398 | 0.846106398 | 0.847988432 | 1/1/1 |
| cube06_indoor | 0.770794713 | 0.770794713 | 0.771803181 | 3/3/3 |
| hat03_indoor | 0.883891337 | 0.883891337 | 0.880924023 | 0/0/0 |
| glass05_indoor | 0.838996138 | 0.838996138 | 0.843122963 | 1/1/1 |
| hat02_indoor_320 | 0.837703301 | 0.837703301 | 0.548759204 | 0/0/3 |
| pigeon05_wild | 0.025251288 | 0.025251288 | 0.025251288 | 2/2/2 |
| duck02_wild | 0.924877744 | 0.924877744 | 0.925058576 | 0/0/0 |
| glass02_indoor | 0.902480776 | 0.902480776 | 0.897996058 | 0/0/0 |
| cup09_indoor | 0.905417827 | 0.905417827 | 0.905417827 | 0/0/0 |
| container02_indoor | 0.943540647 | 0.943540647 | 0.943986275 | 0/0/0 |
| flower01_indoor | 0.420118328 | 0.516087525 | 0.516087525 | 7/6/6 |
| cup03_indoor | 0.560792959 | 0.778713809 | 0.789052431 | 5/4/3 |
| cat03_indoor | 0.908067641 | 0.908067641 | 0.908067641 | 0/0/0 |
| ball13_indoor | 0.399628222 | 0.415233342 | 0.399628222 | 18/17/18 |
| pigeon06_wild | 0.802827727 | 0.802827727 | 0.802827727 | 0/0/0 |
| paintbottle_indoor | 0.841623440 | 0.841623440 | 0.841623440 | 0/0/0 |
| human03_wild | 0.731080222 | 0.731080222 | 0.761114413 | 3/3/3 |
| human06_indoor | 0.891243127 | 0.891243127 | 0.765582948 | 0/0/1 |
| pine01_indoor | 0.790990067 | 0.790990067 | 0.790990067 | 1/1/1 |
| book06_indoor | 0.800135766 | 0.800135766 | 0.806962901 | 3/3/3 |
| cup13_indoor | 0.695815196 | 0.695815196 | 0.695815196 | 4/4/4 |
| flower03_indoor | 0.850748472 | 0.850748472 | 0.850748472 | 0/0/0 |
| notebook02_indoor | 0.852953632 | 0.852953632 | 0.852953632 | 2/2/2 |

## 签发与文件

`EXPERIMENT_AUDIT.json`为机器可读限定结论；`AUDIT_SEAL.json`绑定本报告、JSON、原清单、审前/审后核验、确定性代码/结果及完整trace。`R3_terminal_raw_audit_complete`只表示上述原始材料审查已完成；不覆盖列出的未重建张量，也不是推进门PASS或部署许可。
