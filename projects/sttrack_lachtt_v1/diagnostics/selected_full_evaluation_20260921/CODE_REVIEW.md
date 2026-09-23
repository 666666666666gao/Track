# 2026-09-24 全量评测恢复代码审查

**结论：PASS。没有发现必须修改后才能恢复队列的代码缺陷。** 父代理按此结论启动恢复后，一次健康检查确认迁移GPU比对 `migration_preflight.exit=0`、两模型均复现已保存Train前缀，M67 CDTB已开始。失败时阻断正式推理的首stage逻辑正确。

审查者：fresh Codex agent，gpt-6-astra / max；`review_independence: same-family`，`acceptance_status: provisional`。这是实现与续跑审查，不是六项最终结果的完成审计。未启动远端任务，未编辑冻结源码。

审查范围：以 `remote_snapshot_20260924` 中从当前服务器取得的实际文件为依据，审查 `prepare_full.py`、`run_full.sh`、`analyze_full.py`、五个接口、VOT分片运行器、文本生成器及协议、selection/bundle、M67 DepthTrack完成记录、两模型实际 direct/OPE/TraX预检；另外审查父目录新增 `resume_20260924.sh`、`migration_preflight_20260924.py`、`launch_resume_20260924.py`，以及补充取得的官方初始化导出、冻结VOT manifest和失败统计函数。

## 阻塞项

无。

## 已核实的正确行为

- **剩余队列准确且保留已完成结果。** `resume_20260924.sh:14-29` 先运行两个final的迁移GPU比对，然后M67 CDTB→M67 VOT→M82 DepthTrack Test→M82 CDTB→M82 VOT。既有M67 DepthTrack不再运行。所有stage检查退出码，性能低于目标不改变队列。无新训练或seed试验。
- **中断CDTB不会被混入完成结果。** `launch_resume_20260924.py:35-60` 核验已完成DepthTrack的50组预测/置信度文件，要求CDTB无终结receipt/退出记录、下游正式目录尚不存在，并在重跑前把中断的predictions及log迁入独立存档。它解决了 `interface/run_semantic_ope.py:45` 必须创建新输出目录的实际约束。新CDTB重新执行完整80序列，保留原plan。
- **迁移检查实际运行两模型。** `migration_preflight_20260924.py:10-40` 为M67/M82分别创建独立进程，载入冻结final/base/source，使用原Train序列16帧，将15次预测与已保存直接调用逐项比较。它不打开后续GT，不把预检输出当正式成绩。`resume_20260924.sh:14` 的失败处理有效。
- **模型和接口绑定正确。** `prepare_full.py:45-78`、`interface/semantic_runtime.py:10-57` 使用M67 Control final与M82 Category final，分别为历史seed2026/2027，均要求完整130序列训练状态。未以外部成绩重新选择checkpoint。独立复算了快照12项冻结运行/生成文件与两bundle各5项接口hash，22项均匹配；bundle与selection匹配。
- **OPE没有把其他模型输出当GT。** `interface/run_semantic_ope.py:36-72` 仅以初始化框、图像及初始化文本跟踪；`75-102` 在全部预测封存并验证后才读取数据集GT，调用冻结 `depthtrack_pr.py`、resolution100计分。6位小数输出、初始化置信度1和原始 `best_score` 均保留。NATIVE引用只提供冻结输入列表、GT hash和计分源码，不提供候选模型预测。
- **M67 DepthTrack完成记录内部一致。** 50序列/76,373帧；cases→plan→receipt→metrics、bundle和receipt hash均一致。记录P/R/F为63.66596112335755 / 62.114945894106974 / 62.88089065734882%。M67 CDTB已绑定80序列/101,956帧。此审查没有本地重算完整OPE分数；远端预测逐文件验证由已审launcher执行。
- **实际TraX证据有效。** 对每模型保存的两条Train序列各101次报告，重新计算float32存储→4位小数文本→float32传输，共404次框全部匹配；score最大误差M67为2.9635620069079494e-8、M82为2.9799652079276484e-8。两模型小数框probe均正常退出。`check_entries.py:63-88` 对真实OPE文件和直接结果逐项比对。父代理已报告迁移机Python3.8.20 / vot-toolkit0.7.1 / vot-trax4.0.2的查询记录位于远端 `migration_versions_20260924.json`；本审查未冒称旧TraX预检是在新主机重跑。
- **VOT完整覆盖与协议一致。** 实际冻结manifest的4片为441/441/441/442个anchor，共1765唯一轨迹、127序列、867前向/898后向，逐项方向与长度合计1,327,004位置。caption_inputs的1765项恰为anchors的id/image/bbox投影；manifest、exporter和3个已下载导出文件hash匹配。`export_initializations.py:48-86` 从官方MultiStartExperiment的实际初始化取得矩形，保留正反向索引；生成器 `initialization_captions.py:59` 使用完整TraX wire转换。没有手工GT字符串解析或后续帧文本更新。
- **VOT分片不是复用native结果。** `prepare_full.py:164-203` 只复制冻结sequence/config，创建新的模型专属tracker/run/result目录，把4个worker映射到两GPU。运行器 `139-189` 仅合并新tracker的1765组轨迹/置信度/时间文件，并拒绝重复、缺项或已有master。`analyze_full.py:14-38` 要求5295结果文件、调用实际VOT分析并对全部1765轨迹和127序列统计失败。导入的low22模块只调用 `collect_confirmed_failure_outcomes`；其历史promotion gate不在调用链中。
- **统一BF16处理有实证且保持因果输入。** 已保存同一输入数值诊断显示float16从首步151936个logit全NaN，BF16同一步全有限。实际生成器 `147-172` 统一BF16且每步要求有限logit，无数值替换或Empty失败兜底。仅用初始化图像与实际初始化框。类别保留及有效属性slot替换Empty由 `prepare_full.py:91-97` 完成，两模型使用相同外部观察。历史训练文本精度不同已经在计划中披露。

## 非阻塞交付事项

1. **最终对比CSV尚无写出逻辑。** `analyze_full.py:42-57` 仅输出 `all_results.json`，未实现计划要求的最终comparison CSV。六项完成后，从已核验JSON独立导出即可，不需要改冻结评测源码，也不阻塞续跑。
2. **每小时轮询覆盖值已准确记录。** 最终 `resume_20260924.sh:19,27` 均使用 `--poll-seconds 3600`。原冻结 `prepare_full.py:203` 仍把 `execution.json.poll_seconds` 写为240；最终launcher在恢复记录中明确保存 `actual_vot_poll_seconds=3600` 和 `original_execution_metadata_poll_seconds=240`。此元数据差异已说明，不需要修改冻结评测源码。

## 最终恢复文件与启动时序

最终文件复核与上述审查一致，唯一追加变化为每小时轮询参数及其记录：

- `resume_20260924.sh`: `725ca63b444d6e43d25a4f64a9d72d421044fd3905868ba8675d34e752cfec99`
- `launch_resume_20260924.py`: `97371f0562004ddf20e575c954d5a6fc3b76107411e22646539a2bac74c7dcc9`
- `migration_preflight_20260924.py`: `cfba51648aaeacdd0ce9908b08f7bb5bc0c65d6f2e7678871a17209a5b621905`

已向父代理明确报告无阻塞、VOT实物验证通过后，父代理报告恢复控制器于UTC16:54:34启动，PID2437，首先执行迁移GPU witness。随后一次健康检查确认 `migration_preflight.exit=0`，日志为 `Both migrated final models reproduce saved Train prefixes.`，PID2437存活，GPU1使用2444MiB/62%，M67 CDTB已开始。本审查记录的是父代理提供的启动与健康状态，未独立发起SSH或启动任务。后续按用户要求每小时检查。

## 验证与边界

16个主要Python文件通过AST解析；源码/接口、两个OPE输入绑定、DepthTrack完成链、两模型404次TraX输出及完整VOT anchor清单均完成上述独立检查。新增launcher将进一步在运行机核查所有final/base/模型源码、三份完成BF16文本bank、已完成DepthTrack预测、两GPU空闲、无原tracker进程以及剩余磁盘空间。

本审查没有下载完整数据集/权重/预测以重新推理，也没有把尚未完成的五项评测宣称完成。后续仍需要五项正式完成以及最终指标证据审计；这些工作由现有恢复队列和结果收集阶段完成。用户追加的“两个模型全量完成后，任一同模型全达标即停止；否则再分析并继续新思路”应在六项完成后判定，不改变本次冻结评测。
