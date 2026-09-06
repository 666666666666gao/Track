**结论：WARN。未发现原生递归、GT 使用顺序、指标门或新队列退出顺序的具体 FAIL；防混组绑定还缺两处现成证据的核验。** 建议在递归启动前补齐这两项直接一致性检查。无需修改训练源码、训练进程或研究方案。

本结论对应本次读取的递归 runner SHA `02a532765ec89324…`。未执行 SSH、GPU 或训练，未见 M55 两组完成态权重，也未判断训练性能。

1. **WARN — 结果记录的组别已检查，但最终 checkpoint 自身的组别与轮次元数据未检查。**

   入口检查 `result['variant']`、训练 spec、3,840 步及最后第 15 轮，并验证 `result['weight_path']` 所指文件的 SHA；随后把这个路径交给 `STTrack`。然而原生 tracker 只加载 checkpoint 的 `['net']`，没有核对二进制自己保存的 `variant`、`spec_sha256`、`epochs`、`optimizer_steps`。

   这些字段已经由训练器写入最终 `model_final.pth`，可以直接使用。当前代码能拒绝直接交换两份带错误 variant 的结果 JSON，但不能独立识别“结果记录声明 clone，所指二进制实际是 control”这种指向错误；两组网络结构兼容，`strict=True` 不能替代组别核验。**目前没有证据表明已发生混组。**

   最小补齐：限定为该组的 `training/<arm>/model_final.pth`，加载时核对内部 `variant == arm`、训练 spec SHA、第 15 轮和 3,840 步。无需增加兼容或 fallback。

   证据：[run_recursive.py:33–42、68–71](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/run_recursive.py:33)、[v2/train.py:280–283](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/v2/train.py:280)、[已绑定原生 tracker:24](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py:24)。

2. **WARN — 初始状态只做组间相等比较，尚未连回训练启动绑定。**

   入口核对实际 `batches.jsonl`、`steps.jsonl` 的 SHA，比较两组批次 SHA 和数据流 SHA，并比较两份 `initial_state_sha256.json` 的文件 SHA。这些配对检查有用，但它没有使用完成结果中的 `execution_binding_sha256`。

   训练启动记录已经包含 `mode`、`variant`、训练 spec、trainer SHA、模型源码 SHA 和 `initial_state_file_sha256`。当前没有核验这条链，因此“两个初始状态文件相等”尚未独立证明它们就是各自最终训练记录所绑定的那两份启动状态。

   最小补齐：验证 `execution_binding.json` 的实际 SHA 等于完成结果所记摘要，再将其中的组别、spec、trainer／模型源码和初始状态文件摘要与已有冻结字段对应起来。这是利用已有证据闭合绑定，不需要新增训练记录格式。

   证据：[run_recursive.py:43–51](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/run_recursive.py:43)、[v2/train.py:182–191、291–293](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/v2/train.py:182)。

3. **PASS — 固定训练预算、代码来源和原生单路径配置对应。**

   已独立核验递归 runner、准备器、输入清单、v2 训练 spec、preparation、trainer、两组模型源码、配置及原生 tracker／指标源码的 SHA，均匹配各自冻结记录。训练 spec 确为 `b53991ca…`；控制／克隆两组分别指向各自冻结代码目录，递归在独立子进程中导入对应目录。

   训练器只在完整训练结束后保存最终模型；入口要求两组训练及训练控制器均正常退出，并检查 15 轮、3,840 步、30,720 clips、15,360 microbatches、122,880 search frames。该预算与冻结 spec 一致，没有按开发结果选择轮次的路径。

   运行直接调用原生 `STTrack`：模板数 2、更新间隔 50、置信度严格大于 0.75，模板／搜索尺寸和比例保持原配置；绑定 YAML 的 query 设置为 `TRACK_QUERY=1`、`TRACK_QUERY_OLD=4`、`FIX_QUERY_WINDOW=true`。每条序列初始化都会重置模板、query、框和帧号，没有把上一序列状态传入下一序列。

   证据：[recursive_spec.json:4–9](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/recursive_spec.json:4)、[run_recursive.py:33–71](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/run_recursive.py:33)、[v2/training_spec.json:165–174、216–219](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/v2/training_spec.json:165)、[绑定 YAML:33–37、61–68](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/code/control/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml:33)、[原生 tracker:73–89、128–138](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py:73)。

4. **PASS — 仅首帧框进入推理，所有轨迹核验后才解析后续 GT；指标与原生对照同口径。**

   实际输入清单为 **22 条、33,130 帧／组**，序列名称恰好匹配训练 spec 的 development 集合，与 fit 无交集。推理只把首帧框传给 `initialize`，后续逐帧只读取 RGB-D 图像。

   分析先核对独立原生轨迹，再检查两组退出码、receipt、权重绑定、全部序列、预测文件 SHA、连续帧号及初始框；完成这些检查后，第 151 行才解析 GT 数值。准备阶段对 GT 做哈希绑定没有向 tracker 提供后续 GT 数值。

   复用的统计函数使用连续矩形 IoU，排除初始化与无效 GT；无效 GT 会中断低重叠连续段，H10 是连续至少 10 帧、IoU ≤ 0.1 的段数。晋升门使用有效帧合并均值：clone 对 native 至少 `+0.005`，对 control 至少 `+0.002`，低重叠帧及 H10 不超过两者，并保护原生零 H10 序列。没有将宏平均或别的指标代入主门。

   证据：[run_recursive.py:76–102、115–170](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/run_recursive.py:76)、[prepare_recursive.py:26–38](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/recursive_preparation/prepare_recursive.py:26)、[统计函数:14–30](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/tools/analyze_sttrack_m42_recursive.py:14)、[v2/training_spec.json:229–242](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/v2/training_spec.json:229)。

5. **PASS — 新队列没有提前成功标记，训练和递归退出文件名称一致。**

   训练 wrapper 实际写出的 `train_<arm>.exit` 和 `train_<arm>_controller.exit` 与 runner 要求一致。新队列等待对应 M56 外层控制器结束，再检查显存低于 500 MiB；它记录 M56 退出码但不要求 M56 成功，符合此次明确的依赖关系。

   每组推理返回后先记录实际退出码，失败即退出；GPU0 等待 clone 结果并检查状态，完成分析且分析成功后才写自身队列成功标记。外层 shell 的 EXIT trap 记录真实退出状态；clone 队列若在产生结果前终止，GPU0 有对应终止检查。未发现上一轮 M56 那种先写成功再分析的路径，也没有自动启动公开评测。

   **排程时间是就绪检查的起点，不是训练完成证据。** runner 仍会要求 M55 两组完成态全部成立；本次没有声称 13:00 一定满足这些条件。

   证据：[v2/run_training.sh:10–14](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m55/v2/run_training.sh:10)、[queue_sttrack_m55_recursive.py:23–70](C:/Users/gb/.codex_remote_staging/queue_sttrack_m55_recursive.py:23)、[run_sttrack_m55_recursive_queue.sh:7–9](C:/Users/gb/.codex_remote_staging/run_sttrack_m55_recursive_queue.sh:7)。

本轮两项 WARN 都集中在读取完成态材料时的绑定。若补齐入口断言，应同步更新递归 spec 中的 runner SHA 和发布绑定；M55 已冻结训练 spec、源码及正在运行的进程可以保持原状。最终权重和实际 GPU 轨迹仍需在完成后另行验收。