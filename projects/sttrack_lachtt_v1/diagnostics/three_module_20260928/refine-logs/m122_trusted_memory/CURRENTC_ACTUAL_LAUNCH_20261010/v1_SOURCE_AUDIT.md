# Current-quality C Full152 源码与部署完整性审查

**source_verdict = FAIL；blocking_count = 2；CurrentC_source_gate_ready = false。** 本轮只审所列版本的源码能否进入实际启动验证，未部署、未训练、未计算本分支的正式指标，也未放行原 Future R4。

审查者为 fresh native Codex agent `/root/m122_current_c_full152_source_integrity_20261010`；请求 gpt-6-astra / max，实际 backend/model/effort 均 **UNATTESTED**。审查是 **same-family / provisional**，确定性文件、算术和非神经进程检查按各自范围成立。

文内 trainer 指 train_m122_current_C_full152.py，controller 指 run_m122_current_C_full_suite.py，launcher 指 launch_CurrentC_full152_remote_20261010.py，runner 指 run_m122_current_C_vot_shards.py，collector 指 collect_m122_current_C_results.py。

路径简写：A = `C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007`；D = 同仓库 `projects\sttrack_lachtt_v1\diagnostics\three_module_20260928`；本目录 = A 下 `CurrentC_full152_source_audit_20261010`。行号按原 UTF-8 文件计数。

## 必须修复的两项

### B1：部署准备必然在 split 绑定计数处失败

- 位置：`A/prepare_CurrentC_full152_deployment_20261010.py:46`、`:53–55`；原 `A/R3_paired_actual_deployment_20261010/gate.json:11`、`:372–379`、`:397–399`、`:462–464`。
- 新代码令 split = old['split']，即 `/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/TRAIN_SEQUENCE_SPLIT_20261008_235856.json`。
- 但同一个旧 gate 的 bound_files 中，已绑定的 split 是 `/root/autodl-tmp/m122_full_train_template_teacher_20261010/C_split.json`，SHA256 `ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6`；不存在前一个精确路径。
- 因而过滤只选出 spec、teacher_audit、base_final 三项，随后 `assert len(gate['input_sha256']) == 4` 必失败。未生成 gate/ZIP，也谈不上实际启动。
- 确定性复现：`review_trace/commands/017_deploy_contract_expected_failure/`，真实 exit **1**，完整 stdout 列出四个 wanted、三个 selected 和唯一 missing，stderr 保留原 AssertionError。
- **最小修复**：让新 gate 的 split 使用旧 bound_files 中已有的确切 `teacher_root/C_split.json` 路径与原 SHA；不改 split 内容、不删除计数断言、不加入路径 fallback。

### B2：部署包布局与训练结束时的源码路径不一致

- 位置：`A/prepare_CurrentC_full152_deployment_20261010.py:13–23`、`:49–52`、`:61–66`；`D/train_m122_current_C_full152.py:84–85`。
- ZIP 仅把九个 NEW 写入新 `code/`。SHARED 只在 source_sha256 中指向旧 BASE，未复制进新 code。
- 新 trainer 在 400 步、final 保存及张量往返检查之后，使用自身 `Path(__file__).with_name('train_template_write_C.py')` 与 `with_name('template_write_C.py')` 读取新 code 目录下的两个 sibling 文件。九成员部署包中两个文件均不存在。
- 即使 BASE 导入碰巧可用，代码仍会在记录 result.json 之前出现 FileNotFoundError，控制器停止，后续三数据集不会运行。不能把留在目录中的 final.pt 解释为完整训练成功。
- 另一个已证据化的路径限制：旧 gate 只在 `/root/autodl-tmp/m122_R3_C_paired_20261010/code/` 见证 train_template_write_C.py、template_write_C.py、trusted_template_tracker.py；并未见证新代码假定的 BASE 路径。这里没有查询服务器，故不声称 BASE 文件实际不存在。
- 确定性复现：`review_trace/commands/029_packaged_sibling_expected_failure/`，只按实际 NEW 清单复制源码到独立 fixture，真实 exit **1**；同一 sibling hash 读取产生 FileNotFoundError。没有调用 trainer 或构造模型。
- **最小修复**：将所需 C 共享模块的已审字节放入新 code/，使实际导入位置、source_sha256 键和 trainer 的 sibling 路径一致；至少解决上述两份直接读取及 C 依赖的实际部署位置。无需变更模型、目标、学习率、阈值或旧 P1 源码，也无需兼容分支。

## A–F 结论

| 检查 | 状态 | 证据与限定 |
|---|---|---|
| A Ground truth provenance | PASS，源码范围 | trainer:36–40 只取原教师 current_iou；共享 common_samples:40–88 绑定 R2 教师 result/shards/PT SHA，保留真实 GT 合格规则。正式 OPE 在完整预测 receipt 后才打开 GT（run_m122_current_C_official:44–58）；VOT failure helper 对数据集 groundtruth 计算 overlap。原教师神经执行真实性依已封存 R2/R3 见证，本轮未重建视觉模型。 |
| B Score normalization | PASS | C 的 MSE 分母是样本数；OPE 锁定 05879f… metric；VOT 固定官方 JSON fraction×100。模型特征 L2、motion 变换与 response 的 odds residual 是推理内部操作，不充当成绩归一化。OPE 用预测置信度形成 PR 阈值、选择协议 F 点，不是除以自身输出最大值，也不是选 checkpoint。 |
| C Results and completion | PASS，源码范围 | 新计划:3 明确无训练/部署/正式结果；R3 原 comparison:2217 为 false，R3 audit:323–330 保留科学门失败。本审查未将历史 Current dev 增益转成新分支指标。新控制器的 complete 字段仅位于十阶段成功之后；B1/B2 使当前源码无法走到完整终态。 |
| D Calls and lifecycle | FAIL | 各阶段调用及参数相接，非神经 fixture 覆盖十个调用和失败留证；但真实包准备被 B1 阻断，绕过后仍被 B2 阻断。不能以函数存在、AST 通过或替身成功声称正式阶段执行。 |
| E Scope | WARN | 单 seed2027；新的 all152 是教师覆盖范围，1,257 合格事件来自142序列。原32已纳入 C 优化，A+B 早已见过全部152。三正式数据集仍是待运行计划；源审查不证明泛化、语言贡献、身份恢复或整体目标。 |
| F Evaluation type | PASS | 新 C 拟合 = real_gt-derived current-IoU surrogate regression；正式 OPE/VOT = intended real_gt；人审 bank = 已有人工初始化监督，不是本轮 human_eval；本轮 hash/AST/替身进程检查 = deterministic / synthetic_proxy，不是模型准确率。 |

## 训练与部署策略的逐项核对

1. **独立的新 Current 计划，原 Future 门不变。** 新计划:5–10、:20–33、:48–58 将 Current 作为新的质量控制分支，并说明不能执行、替换或回写原 Future R4。R3 实际审查 WARN/0 仅表示限定的原始终态可归档，不能单独晋升。新准备器:31–39 同时要求本次 R3 actual audit 完成、0 阻断以及本次 SOURCE_AUDIT 的 PASS/ready；本轮 FAIL 会被拒绝。没有把 Future failed 科学条件改写为 true。

2. **训练集合与固定配方正确。** trainer:36–67 拼接 fit+development，按 sequence_index/frame 排序，固定 1,257 / 原1,297；目标为 current IoU，没有把 C 预测或 Future delta 作为当前标签。共同合格规则仍要求当前 GT 有效且有有效未来帧，故保留35当前未知与5仅未来缺失排除。补充原 common_dataset_manifest 实际全表核对得到 993+264、142事件贡献序列、规范化 digest `5894dd48e03ca466cddd56311e8efbf31a9c4d979897b35d588ae10bc3ce5dc9`。十轮×ceil(1257/32)=400，seed2027、AdamW 3e-5/.01、batch32、epoch10 final；未加载公开 Test/CDTB/VOT 优化数据，也没有 best checkpoint 分支。新代码不重新采集 C 自身历史 teacher。B2 是结果记录路径问题，不是训练配方问题。

3. **一份 composite 与接口。** trainer:68–74 将旧 P1 state 放入 A_B、C state 放入 C，并逐张量比对保存/重载；m122_current_C_official_runtime:56–63 再对旧 P1 校验 key 集合及全部张量并 strict load。519→128→32→1 的参数数按结构独立算术为70,721。TrustedTemplateTracker:31–54 保留同一 observe、decoder、bbox、query、selected feature、selected score；:55–67 只在原每50帧且同候选原生响应>.75合格后追加 C>.5 写入门，初始模板不弹出。它可能影响后续自身历史，不声称与旧 P1 全轨迹相同，也不是新的未来价值贡献。

4. **文本和初始化。** m122_current_C_official_runtime:47–53 检查 bank 的1895唯一键、1895×5×768 tokens、5槽 mask、非空类别、encoder/binding SHA；:70–72 按 RGB SHA+精确 double xywh 得到初始化 key。独立遍历 official_initialization_binding 的全部1895行，重建全部 key，确认50/80/1765、VOT127序列、OPE帧数76373/101956，全部记录 human_confirmed、非空类别及1–5个短语槽。未重新做人审，历史 multiframe aids 和 conflicting/status 字段不被重新解释为本轮人工复核。

5. **启动 hash 根与路径。** 新准备器消费 R3 审计和本次 source review，校验 reviewed_source_files，再将审计 SHA 置于 gate input_sha256。launcher:40–43 与 controller:37–40 在执行前检查新/共享源码及输入。旧 checked_plan:7–21 仍负责原 P1 schema、训练结果、原final/native/CLIP/bank/binding；新 checked_plan:9–36 接受旧 plan+旧 bundle 返回，额外校验新 composite schema/training/final，再注入 base_bundle，三者字段契约一致。但 B1 的 split 路径与 B2 的部署布局必须先修复。未执行远端路径存在性验证；旧 gate 是历史见证，不能当成本次实际服务器 hash 读回。

## 三数据集闭环

- `prepare_m122_current_C_evaluation.py:22–46` 读取同一个旧 P1 bundle，为三数据集写同一新 composite bundle，保留相同 bank/binding 以及 OPE cases/dataset_root/metric source。补充读取已归档 hour_1021 P1 bundle 与 OPE plans/cases，验证 bundle SHA、cases SHA、50/76,373 和80/101,956，与当前新字段契约相容。
- `run_m122_current_C_official.py:12–41` 创建全新 predictions 目录、逐帧调用 tracker、写出 bbox/confidence 和回读精度检查，完整预测封存后才在 analyze:44–60 校验/读取真实 groundtruth。没有导入旧预测轨迹。native/CLIP 摘要与 C state digest 前后检查存在；A_B 的冻结依据是载入逐张量相等、eval、Trusted 的 no_grad 和无优化路径，OPE frozen_digest 本身只包含 native/CLIP，不虚称它还独立计算了 A_B 的前后 digest。
- `prepare_m122_current_C_evaluation.py:48–87` 只复制旧 shard sequences/config/metadata，不复制 results；wrapper 将新 code、BASE 放入 sys.path，TraX module/paths/env_PYTHONPATH 均指向新 VOT 目录，神经进程使用既有 sttrack。原桥接器 SHA230acf… 的 rectangle/rgbd/两个路径/report 接口与新调用一致。
- 旧 P1 shard manifest 与锁定 full127 原 manifest 的 source、127序列、四组 expected_trajectories 完全相同；独立检查1765唯一轨迹、总锚点和4 shard分组。新准备器依据旧 execution.source_sha256 验证所复制 metadata，并逐一重新绑定 source anchor.value；runner:25、:46 在前后检查封存输入。
- `run_m122_current_C_vot_shards.py:30–45` 两波、每波GPU0/1，既有 mplt 调用官方 evaluate，child.wait 阻塞等待；新 tracker results 目录须不存在，exit0与每 shard 预计锚点完成后才进入下一波。merge:47–61 按原序列+末尾anchor命名复制并逐文件 SHA 比对，要求1765/5295，无旧轨迹复用。
- controller:66–72 用既有 mplt 调用 official analysis，再 seal，再 collect。analysis 文件名与新 model name 严格相接。已归档 toolkit0.7.1 官方 JSON 的 results.baseline.results 槽位与当前 EAO[0][0][0]、ACC/ROB[2][0][0/1] 解析一致；此为布局核对，未把旧 P1 数字作为新 Current 结果。当前 wrapper 对 JSON 的身份元数据没有新增独立断言，身份依赖该新目录中明确 tracker 的官方调用；完成后的实际审计仍须检查 toolkit/tracker/127序列/protocol。
- failure helper SHA e96a… 与 common SHA cbbe… 均匹配已归档源；collect_confirmed_failure_outcomes 使用官方 workspace、find_anchors、各方向真实 dataset GT 和有效轨迹长度，得到1765 outcomes/127 per-sequence，返回值与新调用相接。H10 不冒充 VOT ROB。
- collector:15–43 消费两份 OPE receipt/metrics 与 VOT merge/analysis/result hashes，再比较一行九项：DepthTrack≥65.2/64.9/65.1、CDTB≥72.9/75.6/74.2、VOT>77.9/82.1/93.7。独立 AST 提取比较表达式验证恰等边界：前六 true、后三 false；VOT高于边界后九项 true。仅为规则测试，没有伪造正式成绩。写出的 independent_completed_audit=false 保留终态独立审查需求。

## 调度、失败记录与实际测试

- launcher:22 在 nvidia-smi:32 之前拒绝早于 **2026-10-10T14:19:36.400301Z** 的准入。源码中只有该一次 GPU snapshot，随后一次 disk_usage；控制器和 VOT runner 均无 NN 进度轮询、自动重试或环境安装路径。首次启动失败后的再次手动调用不在本源码自动行为之内，本轮未尝试准入。
- run_stage:21–29 在子进程返回后先保存真实 .exit/.receipt 与原 stdout/stderr 合流日志，再对非零退出断言；supervisor:20–27 保存真实控制器返回码，返回同值；launcher:47–49 将监督器和控制器输出接入 controller.log。并发 OPE 退出后收齐现有 futures，只有两者成功才继续。
- **实际 CPU 测试**：原25份 Python 全部 ast.parse + compile 成功；未 import torch/clip/numpy、未构造模型。E:\python.exe 为已存在 Python3.12.6，本轮没有安装环境；不将本地编译等同服务器 ABI/import/GPU 验证。
- 生命周期 fixture 实际启动的都是本地短小打印/退出子进程：stage exit7 的 stdout/stderr/.exit/.receipt 保存；原 supervisor 对替身 controller exit7 的真实返回/terminal/log保存；原 VOT runner 对替身子进程 [7,0] 均 wait 并保留日志/exit，第二波不启动；成功路径用 stage 替身覆盖十个控制器调用与五个 mplt metric 阶段。证据 `review_trace/LIFECYCLE_FIXTURE_RESULT.json` 与 `commands/028_lifecycle_fixtures/`。这些 fixture 的 synthetic status 文件位于专用 review_trace/fixtures，绝非正式模型结果。
- B1/B2 的真实本地失败已保留；另保留最初 PATH 的 E:\Scripts\python.exe 失效（No pyvenv.cfg，三次并发只读命令均exit1），随后直接使用已存在 E:\python.exe。没有删除原失败、降低断言或改动原实验源码。

## 原审计限定如何保留

本轮对 R3 原 seal 中120个输出文件逐一检查字节/哈希，全部相符；同时验证 R3审计→comparison/fit_current、fit_current→R2审计的 SHA 链。重算的是原 comparison 中全部96条逐序列汇总的加总与均值，不是重新跑神经网络或逐帧 GT IoU：Current macro 0.774631107471、native 0.764334383531、Future 0.762334417857；Current 3改善/29相同/0下降；Future两个宏均值条件均 false、H10条件 true。

R3 共同初始摘要与本地重建不一致、未导出初始张量的问题仍未解决；原 full-runtime、完整视觉/候选/query/feature张量与远端依赖限制全部保留。R2原 PT 的历史CPU见证不被写成本轮全量重新读取。Current 在原32上的开发结果没有“未见验证”含义，本轮更不能证明未来监督增量、三数据集目标或完整研究目标已达成。

## 封存、阅读范围与后续最小动作

36个清单输入全部逐字节读取、UTF-8解码并复制原文至 `review_trace/originals/`；25个Python源码和新计划逐行检查，大型JSON全量解析/遍历，重复1895初始化行逐项做确定性键/类别/槽位核验。原文阅读账本为 ALL_ORIGINALS_READ.json；额外18个历史静态文件单独审前封存并保留补充核对范围。完整原始请求、子命令argv/cwd/时刻、未截断stdout/stderr和真实退出码保存在 review_trace/commands；宿主工具回执另存 TOOL_RECEIPTS.json。启动阶段的宿主显示截断按原回执保存，不把显示截断内容虚称为完整原stdout；正式审查命令均由记录器捕获完整流。

`SOURCE_AUDIT.json` 给出全部26份要求覆盖的源码/计划路径与SHA及18份补充文件绑定、2个阻断、A–F、范围和测试结果；36份原输入与18份补充输入审前/后哈希一致；审后输入 seal 与 AUDIT_SEAL.json 绑定原文、报告、JSON、完整最终答复和工具证据。trace 内含本机启动上下文，按 review-tracing policy 保持私有，不自动提交或发布。

下一轮只需修复 B1/B2 的路径/打包契约，保留本轮原文、失败和审查，使用新的输入清单再审。源审查通过也只可进入实际启动验证；正式结果仍须完成运行、封存和新的实际结果审计。

