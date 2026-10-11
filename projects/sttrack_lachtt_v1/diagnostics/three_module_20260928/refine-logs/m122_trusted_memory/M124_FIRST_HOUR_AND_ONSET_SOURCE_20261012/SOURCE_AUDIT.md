# CurrentC onset replay — SOURCE_ONLY integrity audit

**source_verdict: SOURCE_ONLY_PASS；blocking_count: 0。总体 WARN：尚未执行，作用域受限。**

审查日期：2026-10-12（北京时间）；生成时间：2026-10-11T17:17:53.466888+00:00。
请求 reviewer：`gpt-6-astra` / `max`；实际 backend/model/effort：`UNATTESTED`。
独立上下文直接读取原始材料；`review_independence: same-family`，`acceptance_status: provisional`。未把 executor 的声明或历史审查结论当作本次结论。

15 个清单输入全部存在、与清单 SHA256 相符；另审阅 18 个必要源码/绑定依赖。33 个文件及 INPUT_MANIFEST 审查前后字节未变。
本次 0 NN、0 GPU 查询、0 SSH、0 模型加载、0 优化器步、0 训练进度查询、0 输入源码修改、0 部署。

**这不是模型运行 PASS，也不放行执行。必须先有实际 M124 主完整 suite 终止和 GPU 释放证据；当前两者均未查询、未核验。**

## A–F 核查

### A_gt_provenance: PASS

- 正式 CurrentC 类复用 base plan、bundle、训练结果、final、native checkpoint、CLIP 与 bank/binding 哈希检查；加载 A_B 与原 base final 逐张量相等，C 使用 current arm。此处检查的是源码约束，未加载实际权重或 bank 张量。
- 每条序列只在 frame=0 使用 case.init_bbox，并按 RGB SHA + 原始 double xywh 绑定文字。官方 binding 的 1895 个 key 已独立重算一致，三个选中 CDTB 条目均为第 1 帧、human_confirmed。
- 后续 groundtruth.txt 在 replayed_predictions.json 写入及 SHA256 生成后才被哈希和解析；GT 只进入重叠/诊断标注，未进入 tracker step、Empty 分支、状态提交或模板决策。
- 人工文字合法初始化绑定不等同于盲首帧协议：已有资料明确使用多帧/视频辅助人工审阅，仅编码类别与稳定属性；不能把文字或 phrase cosine 当作当前实例身份真值。

源码证据：
- [replay_currentC_cdtb_onsets.py:39](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:39)；相关行 39, 41, 43, 53, 58, 63, 64, 66, 114, 117, 124, 125, 126, 169。
- [m122_current_C_official_runtime.py:9](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m122_current_C_official_runtime.py:9)；相关行 9, 11, 18, 20, 23, 30, 47, 48, 49, 50, 56, 58, 59, 61, 63, 68, 71。
- [m122_official_runtime.py:9](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m122_official_runtime.py:9)；相关行 9, 11, 12, 18, 19, 20。
- [bind_m122_official_initializations.py:13](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/bind_m122_official_initializations.py:13)；相关行 13, 33, 35, 37, 50, 53, 54。
- [encode_m122_official_human_text.py:14](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/encode_m122_official_human_text.py:14)；相关行 14, 18, 19, 24, 25, 27, 29, 33。
- [prepare_m122_external_human_labels.py:25](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/prepare_m122_external_human_labels.py:25)；相关行 25, 26, 31, 34, 45, 46。

### B_score_normalization: PASS

- 候选 IoU 由锁定的 VOT bounded rectangle overlap 与数据集 GT 计算，不以预测自身 max/min/mean 作指标分母。native/full/visual/quality/semantic 原始候选数值均被保存。
- response / quality 是移除同一候选乘法质量因子的局部读数，原 full response 同时保留；它不是性能百分比归一化，也不被提交为跟踪动作。浮点舍入仍存在。
- 锁定指标源码的 PR 阈值取自 confidence 只用于选择集合；precision 对选中重叠取均值，recall 分母为有效 GT 数，F 使用通常调和式。本诊断不调用全套 PR/F，也不产生新官方指标。
- native 候选从 data.boxes.float() 保存，因此仅为 float32 的观察表示；源码结果显式否认 exact float64 native output。不能扩大成完整 native tracker baseline。

源码证据：
- [replay_currentC_cdtb_onsets.py:83](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:83)；相关行 83, 85, 86, 91, 96, 98, 99, 100, 120, 136, 148, 149, 160, 162, 163, 184。
- [dense_target_decoder.py:109](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py:109)；相关行 109, 111, 114, 115。
- [full_dense_tracker.py:42](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:42)；相关行 42, 46, 49, 51, 63, 68。
- [CurrentC_actual_locked_CDTB_metric_source_20261012.py:40](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_actual_locked_CDTB_metric_source_20261012.py:40)；相关行 40, 47, 49, 55, 58, 117, 124, 128, 129, 136, 139。

### C_result_existence: WARN

- 15 项清单文件实际存在且初始/最终哈希均匹配；onset_rows.json 是已有保存轨迹的 CPU 诊断，不能用作新的神经回放证据。
- 新计划明确 prepared_not_deployed_not_executed / actual_NN_calls=0 / optimizer_steps=0；审查目录所在 .aris 子树未发现 replayed_predictions.json。该本地文件盘点不证明远端进程状态，本次没有进行远端查询。
- 代码只有完成前缀、冻结状态检查、GT 后标注与最终 prediction SHA 校验后才写 complete_readonly... 的 result.json。尚未执行这些路径，不能把待执行 assert 或返回字典中的 True 字段当作实测通过。
- M124 publication closeout 仅能说明已有发布收据与当时状态字段，不是主完整 suite 终止或 GPU 释放证明；部署/执行准入保持未满足、未核验。

源码证据：
- [CurrentC_readonly_onset_replay_PLAN_20261012.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_readonly_onset_replay_PLAN_20261012.json:2)；相关行 2, 15, 16, 17, 22。
- [onset_rows.json:2](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_failure_onset_images_20261011/onset_rows.json:2)；相关行 2, 973, 974, 975, 976。
- [replay_currentC_cdtb_onsets.py:110](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:110)；相关行 110, 111, 114, 117, 169, 170, 175, 187。
- [M124_actual_launch_publication_final_closeout_20261012.json:18](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/M124_actual_launch_publication_final_closeout_20261012.json:18)；相关行 18, 19, 28, 33, 35, 36。

### D_dead_code_and_real_interfaces: PASS

- main 由 __main__ 入口调用；actor.trusted.step 与正式 actor.track->trusted.track->step 使用同一动作路径。step 返回 out,data,record，诊断没有误解返回协议。
- PreprocessorMM.process 返回一个 torch Tensor；原生初始化 z_dict=[template]*num_template，模板检查直接 clone/equal 张量，未错误使用 .tensors 字段。原生 network.forward 返回 out_dict 列表，observe 正确取 outputs[0] 与 candidate_features。
- 模板写入只在正式 step 的 native 50 帧/.75 条件与 current C prediction>.5 后发生，append(template)/pop(1) 保持首模板；Empty 分支仅调用 decoder(alternate)，没有 native forward、tracker step、query/template 写入。
- 预测封存后，_load_rows 与 _vot_overlaps 调用结果确实写入 diagnostics/result；全量 evaluate_depthtrack_results 和 format_metrics 不属于新诊断的执行路径。
- train_template_write_C 仅提供 state_digest；其训练 main 受 __main__ 保护，不因导入执行。analyze_train_states 与历史 DenseRegionDecoder/InstanceCandidatePrototype 的指标/模型类并非本诊断路径，不可据其存在宣称执行。

源码证据：
- [replay_currentC_cdtb_onsets.py:66](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:66)；相关行 66, 75, 79, 106, 107, 108, 109, 126, 136, 148, 156, 168, 187, 192, 193。
- [trusted_template_tracker.py:27](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/trusted_template_tracker.py:27)；相关行 27, 32, 35, 44, 45, 46, 48, 55, 56, 58, 59, 61, 62, 63, 66, 67, 68, 71, 72, 73。
- [template_write_C.py:21](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_C.py:21)；相关行 21, 24。
- [full_dense_tracker.py:56](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:56)；相关行 56, 59, 73。
- [train_template_write_C.py:18](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_template_write_C.py:18)；相关行 18, 31, 102, 186, 187。
- [data_utils.py:23](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009/lib/test/tracker/data_utils.py:23)；相关行 23, 25, 26, 27。
- [039_sttrack.py:31](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R0_completed_audit_sources/039_sttrack.py:31)；相关行 31, 77, 79, 84, 85。
- [038_sttrack.py:59](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/R0_completed_audit_sources/038_sttrack.py:59)；相关行 59, 65, 147, 149, 150, 161, 162。

### E_scope: WARN

- 计划恰为 cart 0..197、trashcans A 0..492、backpack 0..216，三条自身历史前缀共 908 帧（含 3 次初始化）、905 次跟踪步；另有 30 次 Empty decoder 调用，30x256 候选。905 不是所有底层神经模块调用总数。当前实际新回放为 0。
- 固定事后选出的 30 个失效附近样本，不是全 CDTB80/完整轨迹恢复实验，不是独立测试集泛化证明；25 帧 GT 有效，backpack 的 184/192/193/194/195 无有效 GT，不可用这些帧的零重叠或 oracle 排名判定候选失败。
- 每条序列从首帧连续走自身 history，不读取保存框重置，不回放单帧 GT 状态。保存框/分数只在封存后比较。5.01e-7 容差对应已有 6 位小数保存边界。
- 若前缀框或分数与保存轨迹不同，只能描述当前回放状态。即使 bbox 在容差内一致，也没有保存原始 query/template 全状态作比对，不能据此证明原始故障内部状态精确复现。
- Empty、native score、without-quality 及几何替代只是同一回放状态的候选读数，不提交反事实动作，因此不能宣称因果完整轨迹改进或恢复。不得用 CDTB 调阈值/训练，亦无新结构设计。
- 进入执行前，仍必须实际取得 M124 主完整 suite 终止与 GPU 释放证据，使用指定官方 plan/cc71... final 与合法 bank 的锁定链；本审查没有准入操作，不能作为 GPU 或模型 runtime PASS。

源码证据：
- [replay_currentC_cdtb_onsets.py:21](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:21)；相关行 21, 22, 23, 24, 58, 63, 66, 70, 71, 127, 131, 132, 133, 138, 139, 140, 141, 171, 172, 179, 180, 181, 182, 183, 184, 185。
- [CurrentC_readonly_onset_replay_PLAN_20261012.json:3](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_readonly_onset_replay_PLAN_20261012.json:3)；相关行 3, 8, 9, 10, 11, 12, 13, 16, 21, 22, 23, 24。
- [run_m122_current_C_official.py:33](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_m122_current_C_official.py:33)；相关行 33, 34, 35。
- [onset_rows.json:670](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_CDTB_failure_onset_images_20261011/onset_rows.json:670)；相关行 670, 671, 696, 697, 722, 723, 748, 749, 774, 775。

### F_evaluation_type: PASS

- 拟执行的是数据集 GT 事后标注的三前缀、同状态候选诊断；旧 prediction 只作轨迹一致性参考，不是 GT。
- 不是 synthetic_proxy、自监督代理、simulation_only 或 human_eval 性能测量；人工标签是模型输入协议。评审中纯数据 fixtures 仅验证元数据/控制流，不是模型性能实验。

源码证据：
- [replay_currentC_cdtb_onsets.py:124](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:124)；相关行 124, 126, 127, 136, 137, 147, 148, 149, 185。
- [CurrentC_actual_locked_CDTB_metric_source_20261012.py:40](C:/Users/gb/.codex_track_publish_m29_20260902/.aris/m122_full_causal_20261007/CurrentC_actual_locked_CDTB_metric_source_20261012.py:40)；相关行 40, 49, 50, 51, 54, 55, 58。

## 完整模型与状态路径

`@torch.no_grad()` 覆盖新 main。native/STTrack 与 CLIP 使用 eval 并关闭参数梯度；A_B/C 载入正式 final 后 eval，C 参数关闭梯度。源码在前缀前后比较 native、CLIP、A_B、C 的 state_dict digest，并重新调用 checked_plan。正式跟踪状态正常沿 CurrentC 自身历史变化；所谓“替代状态不变”仅指同帧额外替代计算没有再次提交状态。

证据：[replay:32](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:32)、[replay:47](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:47)、[replay:110](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/replay_currentC_cdtb_onsets.py:110)、[runtime:60](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m122_current_C_official_runtime.py:60)、[FullDenseTracker:31](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py:31)、[state_digest:31](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_template_write_C.py:31)。这些是待执行的正确约束，不能记为实测通过。

## 已完成的确定性检查

- 27 份 Python 文件 AST 解析通过，未 import 项目模块、torch、cv2 或 VOT。
- 9 组源码/纯数据检查通过：清单哈希、AST、固定范围、未执行状态字段/准入声明、30 行 crop 元数据、1895 个初始化 key、7 项原生源码归档映射、封存顺序/调用点、6 位小数容差和 float32 表示边界。
- crop fixture 重算的是已保存框的几何元数据；不是新 GT/IoU 重算，更不是神经运行。人工 key 重算使用已记录 RGB SHA 与合法 bbox，未重读远端 RGB bytes 或加载 bank。
- 新 replayed_predictions.json 的本地盘点为空；未检查远端进程或文件，不能将本地缺失提升成全局未运行的证明。

## 结论与后续约束

没有发现需修改此只读诊断源码的阻塞项。可保留为通过本轮 SOURCE_ONLY 审查的待执行材料；原输入不改。
未来部署/执行仍需主 M124 完整 suite 实际终止、GPU 释放，以及计划指定的 CurrentC 官方 plan、`cc71e0b4f42c2e489325627d0a16dcdfc4fa6bcb941908c30bc20c8c3c5828c0` final、合法 bank 的真实哈希检查。此次没有加载或证明远端权重/plan/bank 的当前字节。
回放后必须保留真实差异与失败，不能为了匹配旧框强制 state、重试挑结果，不能把 30 个候选状态的容量称为完整恢复。即使框通过保存舍入比较，也不能宣称原始内部状态精确复现。

## 审查产物

- `SOURCE_AUDIT.json`：机器可读结论与逐项文件/行号。
- `INPUT_SHA256_BEFORE.json` / `INPUT_SHA256_AFTER.json`、`DEPENDENCY_SHA256_BEFORE.json` / `DEPENDENCY_SHA256_AFTER.json`：前后字节校验。
- `_private_trace/`：完整审查请求/报告响应、元数据、逐行源材料快照、独立检查脚本与确定性证据；私有、不发布。
- `AUDIT_SEAL.json` 与 `AUDIT_SEAL.sha256`：输入及私有审查产物封存。

操作记录：系统默认 `python` 最初因 `No pyvenv.cfg file` 退出，未执行审查脚本；随后使用已安装的独立 Python 3.12.12 执行纯 stdlib 检查。无下载或环境安装。
