**本轮源码接线审阅：PASS，另有 1 项队列状态 WARN；未发现具体 FAIL。** 上轮的 checkpoint 与 runtime 绑定问题已补上，静态反事实、序列文本索引、递归 GT 使用顺序和晋升门实现均与当前计划一致。

本次未执行 SSH、训练或 GPU 路径，也未读取开发数值结果。CPU 正式训练于 10:09 启动是任务提供的状态；下述结论不代表三组训练已完成或递归已运行成功。

1. **WARN — `queue_gpu0.exit=0` 写得过早，会把仍在等待或随后失败的协调队列标成成功。**

   第 49 行在 GPU0 完成 attributes、empty 后立即写入成功标记，但随后还要等待 pooled、检查其退出码并运行分析。若 pooled 返回非零，第 53 行会失败；若分析返回非零，第 58 行会退出失败，但此前的 `queue_gpu0.exit` 仍为 `0`。即使最终成功，该文件也会在整个队列进程结束前出现。

   最小修正是让 GPU0 的该成功标记在协调分析成功后写入；当前版本应以 `recursive_analysis.exit` 和对应完整结果为最终分析状态，不能用 `queue_gpu0.exit=0` 宣布整条流程完成。这是状态报告问题，没有发现它会改变模型预测或指标。

   证据：[queue_sttrack_m56_recursive.py:49–58](C:/Users/gb/.codex_remote_staging/queue_sttrack_m56_recursive.py:49)。

2. **PASS — 递归入口正确核对正式权重，并按序列名接入训练时的文本表示。**

   `bound()` 检查三组完成记录、20 轮／960 步、结果和 checkpoint 哈希；加载具体组别时进一步核验 checkpoint 的训练 spec、variant、base、text bank 绑定。构造 tracker 后再次确认组别，并用 `bank['sequences'].index(case['sequence'])` 提取该序列的原始短语与 mask，再由已审 runtime 调用同一个 `condition_text`。

   因而源码上不存在把 pooled 权重配上 attributes 语义、按遍历位置取错文本，或把空串控制替换成全遮罩输入的问题。运行返回字段 `target_bbox`、`best_score`、`association_candidate`、`association_none` 也与继承的 tracker 实际返回一致。

   证据：[run_sttrack_m56_recursive.py:19–28、46–68](C:/Users/gb/.codex_remote_staging/run_sttrack_m56_recursive.py:19)、[sttrack_attribute_candidate_set.py:32–43](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_attribute_candidate_set.py:32)、[sttrack_candidate_set.py:58](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_candidate_set.py:58)。

3. **PASS — 静态反事实使用正确的固定权重，配对范围和预定分层没有接错。**

   三组主比较各加载对应最终 checkpoint；`attributes_empty`、`attributes_shuffled`、`attributes_conflict` 均复用 attributes 权重。空串干预调用 empty 条件化；打乱／冲突干预保留 attributes 条件化，替换为封存供体的文本行与 mask。冲突无供体的序列被排除，真实文本结果按事件键取相同覆盖范围后再比较。

   类别保持、短语数保持以及两者同时保持的分层直接读取训练前冻结的控制字段；没有使用开发 IoU 选择供体或筛选分层。静态脚本在读取开发标签之前，先检查最终训练完成记录和权重／结果哈希。

   证据：[analyze_sttrack_m56_static.py:19–34、71–106、121–139](C:/Users/gb/.codex_remote_staging/analyze_sttrack_m56_static.py:19)、[bind_sttrack_m56.py:69–77](C:/Users/gb/.codex_remote_staging/bind_sttrack_m56.py:69)。

   这里封存的是**三组最终权重**，静态脚本随后先解析标签、再计算固定模型预测；标签仅用于事后 IoU，没有进入模型输入或控制映射。这与完整递归要求“先封存轨迹再解析后续 GT”的更严格顺序有明确区别。

4. **PASS — 完整递归的后续 GT 数值解析位于全部预测封存检查之后。**

   `run()` 的逐帧路径只读取 RGB-D 图像、初始框和冻结文本，不解析后续 GT。每条预测写出后记录文件 SHA，整组完成后再写 receipt。

   `analyze()` 先核对三组退出码、receipt、权重绑定、22 条序列集合、逐序列文件 SHA、完整帧号和初始框；上述检查全部结束后，第 137 行才用 `np.loadtxt` 解析后续 GT。准备阶段对 GT 做文件哈希绑定，没有把 GT 数值传给 tracker。

   原生对照复用父实验已绑定的独立原生轨迹；它不是本次新跑的第四条 GPU 任务，也没有把当前 attributes 轨迹充作 native。

   证据：[run_sttrack_m56_recursive.py:59–90、102–140](C:/Users/gb/.codex_remote_staging/run_sttrack_m56_recursive.py:59)、[prepare_sttrack_m56_recursive.py:18–36](C:/Users/gb/.codex_remote_staging/prepare_sttrack_m56_recursive.py:18)。

5. **PASS — 比较范围、指标定义和晋升门准确接入。**

   准备入口固定 22 条 development、每组 33,130 帧，并检查 RGB 与 depth 文件名完整连续；分析要求每组 receipt 覆盖全部序列，并逐帧检查轨迹完整性。复用的 `statistics()` 排除初始化与无效 GT，无效 GT 会中断连续低 IoU 段；H10 是连续至少 10 帧、IoU ≤ 0.1 的段数。

   晋升比较使用有效帧合并均值，要求 attributes 相对 native 至少 `+0.002`、相对 empty 和 pooled 各至少 `+0.001`；低 IoU 帧数及 H10 对三组均不得增加，原生零 H10 序列不得新增 H10。这里没有把百分点误写成比例，也没有把 sequence macro mean 用作主门。静态结果明确不做晋升决定，队列亦不据静态结果选择 GPU 组别。

   证据：[prepare_sttrack_m56_recursive.py:18–24](C:/Users/gb/.codex_remote_staging/prepare_sttrack_m56_recursive.py:18)、[run_sttrack_m56_recursive.py:118–165](C:/Users/gb/.codex_remote_staging/run_sttrack_m56_recursive.py:118)、[analyze_sttrack_m42_recursive.py:14–30](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/tools/analyze_sttrack_m42_recursive.py:14)、[analyze_sttrack_m56_static.py:140–145](C:/Users/gb/.codex_remote_staging/analyze_sttrack_m56_static.py:140)。

6. **PASS — 正常队列路径的训练退出标记、设备映射和跨 GPU 衔接一致。**

   CPU controller 实际写出的文件就是 `train.exit`，与队列检查一致。GPU0 顺序执行 attributes、empty，GPU1 执行 pooled；`CUDA_VISIBLE_DEVICES` 将各子进程绑定到指定物理 GPU。每个 runner 正常返回时，无论成功与否都会写该组退出码；因此 pooled runner 返回失败会让 GPU0 明确失败，不会在这个正常返回路径上无限等待。空闲等待和组间等待使用 240 秒轮询。

   未发现需要增加 fallback 的当前证据。GPU 是否按预期释放、环境是否能完成完整前向仍属于尚未执行的运行验证，不应由本次源码 PASS 代替。

   证据：[run_sttrack_m56_cpu.sh:6–14](C:/Users/gb/.codex_remote_staging/run_sttrack_m56_cpu.sh:6)、[queue_sttrack_m56_recursive.py:19、30–58](C:/Users/gb/.codex_remote_staging/queue_sttrack_m56_recursive.py:19)。

本次审阅的本地源码 SHA 前缀为：静态分析 `d373c417c1c1`、递归 runner `150e8eeb5675`、递归准备 `54adabf95611`、队列 `5be7327b4f96`。给定的训练 spec `cafc4471…` 和递归 spec `bdc367d0…` 是任务提供的冻结标识；本次没有取得这两份实际 JSON 的本地副本，未独立核验其文件 SHA。当前结论是对上述入口源码及其检查逻辑的审阅，不是已完成权重文件和 GPU 输出的验收。