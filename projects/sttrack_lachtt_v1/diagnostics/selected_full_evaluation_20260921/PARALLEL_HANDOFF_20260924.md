# 双GPU续跑交接（2026-09-24）

用户在M67 CDTB运行期间要求尽量使用两张GPU。原控制器PID2437于UTC约17:22被SIGSTOP，仅停止负责切换阶段的shell；其正在运行的M67 CDTB子进程PID2647继续在GPU1执行。保存的M67 DepthTrack结果未动。

M67 VOT在独立进程PID3660中于2026-09-23T17:23:40.542462+00:00启动，先由原冻结`prepare_full.py bind_vot`完成绑定，再按原127序列、1765anchor、四个TraX分片在GPU0/1推理。VOT实际runner轮询3600秒，原`execution.json`的240秒为此前计划元数据。启动检查：M67/vot_bind.exit=0，runner PID3663存活；GPU0 4883MiB/97%，GPU1 7322MiB/97%，M67 CDTB子进程仍存活。`parallel_vot_launch.json`和`parallel_m67_vot_20260924.sh`保留启动绑定，VOT完成态看`parallel_m67_vot.exit`、`M67/vot_metrics.exit`及`M67/vot/result.json`，不能凭PID文件推断成功。

旧`resume_20260924.sh`不能恢复执行，否则会尝试重复绑定M67 VOT。待M67 CDTB子进程终止为zombie且完整receipt出现，运行`finish_m67_cdtb_20260924.py`；它先确认80序列/101956帧封存，再终止已暂停的旧shell，调用原冻结OPE分析器逐项验证预测和GT，写`M67/cdtb_metrics.exit`、`metrics.json`与回执。它明确记录CDTB子进程真实exit code未直接观察；验证依据是完整receipt及冻结分析器通过。不在子进程仍运行时调用。

M67 CDTB与VOT均完整分析通过后，运行`parallel_m82_20260924.sh`。脚本先验证两个M67 exit/result，再顺序绑定M82 DepthTrack与CDTB，分别在GPU0/GPU1同时执行OPE，等待两者均完成，计算两组指标，随后使用双GPU完成M82 VOT。最后收集六项正式指标并导出comparison CSV。后验逐序列CSV只在六项完成后另行运行，不改变正式指标。

用户要求每小时检查一次长任务；本次额外即时检查是用户改变GPU调度所需的启动与安全确认。新并行代码不修改模型、权重、文字输入、输出协议或指标实现。旧脚本及日志保留以便追溯。M67先完成三数据集正式结果，再执行M82；两模型的预测不可拼成一个成绩。