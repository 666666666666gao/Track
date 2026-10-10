# Current-quality C Full152 v2：B1/B2 最小修复复核

**source_verdict = PASS；blocking_count = 0；CurrentC_source_gate_ready = true。** 这只表示本清单源码可以进入实际启动验证；不表示已经部署、训练、运行三数据集或取得正式指标，不表示原 Future 科学门通过或 Future R4 已执行。

本轮是同一原生 Codex 审查员的 v2 延续复核，不伪称第二个独立 fresh reviewer。请求模型 gpt-6-astra / max；实际 backend、model、effort 均 UNATTESTED，same-family / provisional。按 experiment-audit 与 local-codex-policy 保留 v1 的全部研究范围限定。

## 差异与历史封存

v2 清单有40项：v1 原36路径，加 v1 的报告、JSON、seal 与失败版准备器归档。实际读取/哈希比较确认原36项中35项未变；只有 `prepare_CurrentC_full152_deployment_20261010.py` 从4650字节、SHA256 `5dad0d8207533866deeb44af8adaf6b610ed805f37e1228098ba87c9e4ec7103` 改为4647字节、`a8fe5e5eefe0bcec37e98a07073fa12eedd1f3937a6fa8ed6829a8abf3983013`。

原准备器的 `prepare_CurrentC_full152_deployment_20261010_v1_source_failed.py` 与 v1 字节完全一致。v1 AUDIT_SEAL 的257个封存输出全部再次核对通过，seal 自身 SHA256 仍为 `1034f9eac141979216b361a5e5e9b2fb4f0bce93cf8841e351fbf218e4644fdf`。v1 的 FAIL/2、原始失败、原文快照、报告及 seal 均保留；本轮仅关闭 v2 中修复后的阻断，不回写 v1。

九个新模块、14个共享模块、launcher、新计划、R3 actual audit 和其余原输入均未修改。因此复用 v1 已完成的训练、指标、GT、生命周期与范围审查，没有重复运行神经或正式评测。证据：`review_trace/V1_V2_DIFF.json`、`DEPLOYMENT_DIFF.patch`。

## 两项阻断的闭合

| 历史问题 | v2 修复与实际证据 | 结论 |
|---|---|---|
| B1：split 路径未命中 bound_files，原四项筛选只得到三项 | 准备器第47行使用 old['teacher_root'] + '/C_split.json'；第53–55行按 gate['split'] 筛选并保留恰4项断言。CPU 执行原 builder AST，得到 split、spec、teacher_audit、base_final 四个精确旧路径/原SHA；随后加入当前两份审计，共6个输入绑定。 | CLOSED in v2 |
| B2：九源码包缺少 trainer 用 with_name 读取的两个 sibling | 第24行 PACKED=NEW+SHARED；第50行 source_sha256 与第62–63行 ZIP 均使用同一23项 PACKED，全部目标统一为新 code/。隔离实际 ZIP 包含23源码与3份元数据，逐文件比对清单、源文件、解包文件、gate SHA一致；trainer:84–85 的两个同目录读取均成功。 | CLOSED in v2 |

四个原绑定输入保持以下原哈希，未修改数据：

- C_split.json：`ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6`
- training_spec.json：`3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425`
- 旧 P1 final.pt：`3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811`
- R2_TERMINAL_AUDIT.json：`e082d4a5d92579fb39abe4b8b4db300bac4127ccb9e90b227d98287fe7197cc2`

这里验证的是旧 gate 的静态精确映射，不是本轮远端原文件的存在性或字节读回。该限制保留到实际启动的哈希检查。

## 导入位置、源码门与隔离 fixture

准备器76行全文和完整 diff 已逐行复核。第33行确实消费本 v2 SOURCE_AUDIT.json；第35–38行仍要求本次 R3 actual terminal audit 完成/0阻断及本次 source PASS/ready，第39–40行校验 reviewed_source_files。没有放宽原科学门或任何失败断言。

launcher:45、controller:15 均按新 code → BASE → repository 设置 PYTHONPATH，且 cwd 是新 code；TraX wrapper（prepare_m122_current_C_evaluation.py:47、55–57）也将新 code 置于 BASE 之前。23个本地模块全部通过实际 PathFinder 解析到解包后的新 code，C 的 train_template_write_C、template_write_C、template_write_features、trusted_template_tracker 及递归本地依赖均在其中；两个 sibling 的 SHA 分别仍为 `7089754f…`、`4bf01cd8…`。源码表不再把这14份共享模块绑定到未被证明存在的 BASE 位置。

其中10个仅在模块级使用标准库的模块还进行了真实 import 并核对 __file__ 为新 code；神经模块只解析路径、AST/compile，不执行模块导入，不构造模型。既有 FullDenseTracker 在构造时将 repository 加到路径首位供 lib.* 导入；当前 runtime 在构造前已导入这些本地 C/decoder 模块。外部 STTrack/CLIP/lib 和原 P1 bundle 所引用的旧共享文件仍需按 v1 契约在实际启动时验证；本轮没有虚称23份包涵盖所有第三方依赖或旧52份来源。

CPU 证据 `review_trace/SOURCE_CONTRACT_FIXTURE.json`：

- 负向：在隔离 HERE 下用**真实 v1 FAIL 报告**调用原 v2 main，第37行实际 AssertionError，exit1，未生成 gate/ZIP。完整 stdout/stderr 保留于 commands/005_actual_FAIL_gate_rejection。
- 正向：仅执行原准备器第41–72行未改写的 builder AST；HERE 重定向到 review_trace/fixtures，ROOT 明确为 /SYNTHETIC_CPU_FIXTURE_NOT_DEPLOYABLE/…，review 使用 ready=false、NOT_AN_AUTHORIZATION 的 fixture 标记。没有模拟 PASS 报告，也没有以 fixture 通过授权 NN。完整原 main 的正向准入未在本轮执行。
- 原 AST 实际写出并解开 ZIP，验证成员数26、代码数23、全量 SHA、一致的 source_sha256 映射、恰4个原输入、2份审计绑定、C递归依赖、同目录 sibling 及新 code 解析优先级；23个解包源码编译通过。该通过属于 deterministic / synthetic_proxy。
- 首次 packaging fixture 在审查脚本自己的 launcher 文本断言处 exit1，原因是期望字符串省略逗号后的空格；原产品源码正确。失败脚本/包/stdout/stderr未删。另存修正后的 fixture，仅修正字面空格及独立输出目录，commands/009_corrected_source_package_fixture 实际 exit0。没有改生产源码，没有重启任何实验。

正式部署路径未写入；没有 SSH/SCP、GPU查询、NN导入/前向、训练、官方指标执行或环境安装。

## A–F 与研究限定继承

| 检查 | v2 状态与范围 |
|---|---|
| A GT 来源 | 继承 v1 PASS（源码范围）：既有教师真实GT派生的 current_iou；OPE完整预测封存后再读真实GT，VOT由锁定官方工具与failure helper读取数据GT。 |
| B 指标归一化 | 继承 v1 PASS：没有模型输出归一化冒充成绩；特征归一化与推理变换不被当成正式指标。 |
| C 完成声明 | PASS（源码范围）：新 Current Full152 仍未执行；原 Future R3 科学门失败和未执行的 R4 保留，没有新正式数值。 |
| D 调用与生命周期 | PASS（源码准入）：B1/B2闭合；其余十阶段调用、blocking wait、exit/stdout保留由未变源码与 v1 非NN检查支持。没有宣称正式十阶段已被执行。 |
| E 范围 | WARN 保留：单seed；all152为teacher覆盖，1257合格事件来自142序列，原32已进入C训练；不证明泛化、语言贡献或整体目标。 |
| F 监督/代理 | PASS：Current训练为GT派生current-IoU surrogate，拟运行OPE/VOT为real_gt；bank是既有人审初始化监督；本轮为静态/CPU synthetic_proxy。 |

固定方案不变：993+264=1257、35+5排除、seed2027/10epoch/B32/lr3e-5/wd.01/400更新/固定final；不使用Test/CDTB/VOT优化或选checkpoint，不重采集C自身历史标签。A_B逐张量保持旧P1；C为519输入/70,721参数，原native每50帧且>.75资格后再加C>.5；同一bbox/score/query/feature接口及初始身份、人审类别+最多4属性保持。

三数据集计划仍使用同一 composite 和人审bank，保留50/76373、80/101956、127/1765/5295，重新生成轨迹，既有sttrack用于NN、mplt用于指标。九项目标的六个≥与三个严格>规则不变。Current质量头不因此成为新的未来价值贡献。

v1 关于 R3初始摘要不一致且未导出初始张量、历史神经/教师运行见证、远端存在性、官方JSON身份、OPE冻结摘要范围、人审记录来源等全部限制继续生效。R3 WARN/0、v2 source PASS/0都不能当作研究性能晋升证明。

## 封存

40项v2输入逐字节复制为本目录的原文快照；审前、审后及签发前核验均须一致。SOURCE_AUDIT.json列出26份当前源码/计划的路径和SHA、两项历史阻断的关闭证据与继承范围。完整请求、实际命令、未截断stdout/stderr、退出码、失败fixture、最终答复与报告一起封存；完整报告读回后签发。所有trace留在本机，不自动提交或发布。

下一步可以用本次真实报告构造正式源包，再执行实际启动准入验证。准入时间门、一次资源检查、输入/源码hash验证、失败原证据留存、正式终态后的独立结果审查仍须遵守；本审查不执行这些后续工作。

