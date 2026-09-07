# M73：按用户要求取消多seed，仅保留seed2027

2026-09-07，用户明确要求“不做多seed实验”。本目录记录最终结果产生前的范围修订，替代父目录及completion_tools/candidate_evaluation/full_evaluation中的双seed调度条件。

- seed2027 Category/Empty两组原训练进程继续运行，网络、输入、预算、11项开发要求不变。
- seed2028完整训练与开发评测取消；此前准备和短预检记录仍保留。
- 唯一候选仍是原先指定的seed2027 Category final；不再作多seed稳定性或重复验证声明。
- 固定权重内容对照、真实OPE/TraX保真、低22及同一模型完整三数据集的条件保持。
- 没有新增正式指标，两个Qwen继续保留。

`plan.json`是范围修订；`patch_receipt.json`列出运行文件的新旧哈希；`running_verification.json`记录真实进程核验。`controller__*`是当前独立接管队列的记录。旧双seed总控和旧screen管理进程暂停，现有seed2027配对继续；原配对结束后才清理旧管理进程。旧completion的非零终止回执属于用户取消调度，不是训练失败。

源码中的PID用于接管本次已经运行的进程，不是新服务器的通用启动命令。不要重新运行父目录的历史`run_m73.sh`，它仍保留原双seed计划以保存历史证据。新环境复现应只执行原seed2027配对训练，并按本目录的单seed审计条件接续。

本次CPU检查核验了实际源码/数据契约及文本库内容不变，并实际执行缺少完成态审计时的入口拒绝检查。这不是最终权重、GPU接口保真或评测完成的证明。旧准备产物保留原哈希，当前修订以本目录为准。

修订中的候选shell路径随后补齐；`entry_link_correction.json`及`patch_receipt.json`记录当前依赖。其预期拒绝信息写入binding.log，已按真实重定向日志核验，未发生模型评测。详见统一交接文档§5.137。
