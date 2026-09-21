M88 完成态收集代码审查

**PASS；blocking_issues = 0；nonblocking_issues = 0。** 本次为 experiment-bridge 限定范围的新上下文审查，review_model = gpt-6-astra，reasoning_effort = max，review_independence = same-family，acceptance_status = provisional。结论仅覆盖完成态收集实现及其与现有运行记录的衔接。

审阅文件为 COLLECTION_PLAN.md、collect_completed.py、run_collection.sh、launch_collection.py。上下文读取了原 launch.json、frozen.json、prepare.py、run_m88.sh、run_recursive.py、准备快照中的 training_spec/recursive_spec、初始 adapter、原审查回执及前检结果；仅对照了 M87 既有 collector 的继承部分，没有重新展开模型或旧实验全面审计。审查者未 SSH、部署、启动训练、运行推理，也未修改原实验源码或权重。

1. **等待原控制器正确。** launch_collection.py 要求原 launch.json 的 controller_pid 为 48174，且当前 ps 参数包含原 run_m88.sh。run_collection.sh 只以 CPU 环境启动收集器；collect_completed.py:135-142 对仍活动的同一控制器每 240 秒记录一次状态，消失或僵尸后才进入收集。没有 kill、重新启动训练或恢复队列逻辑。原始本地 launch.json 对应 controller_pid 48174、training_pid 48175；本次审查不把该历史回执当作远端当前进程观察。

2. **完成条件与队列输出一致。** collect_completed.py:21-38 要求 controller、training_category、recursive_analysis、category_recursive、category_empty_recursive、category_swapped_recursive 六个退出码均为 0；同时要求完整训练状态、130 序列、186694 次 track、5798 次更新及 final.pth 与训练/分析回执一致。原 run_m88.sh 正好产生这六个退出文件。失败会停止，不会用 latest 或中间状态制造完成结果。

3. **固定 M88 与三组封存条件一致。** collector 固定 frozen SHA 为 473c0e5e83d5a093cfb5dd541ca5de467fe58355c20e5f917ad4391f3b545cbc；本地实际 frozen 文件和原 launch 回执均匹配。准备快照中的 training_spec SHA 为 0d82b358e88525b17474e50215afd84f39898f40c7ff1b0df13b44bc92da4428，recursive_spec SHA 为 06c6920905baa4613c1c1c4a846a1838346d19b9c7a7a55fe2e065806ff4262f，均与 frozen 一致。case 数为 22、总帧数 33130。collect_completed.py:68-87 先核对三个内容条件的回执、共同最终 checkpoint、spec、序列顺序、逐文件 SHA 和完整帧号，再读取 22 个 development groundtruth.txt。原 run_recursive.py 的分析阶段同样在三组封存后使用真实 dataset GT。

4. **14 门保持原定义。** 固定方案和 run_recursive.py 对应 4 个 M84 增量条件、5 个 native 条件、4 个 Swapped 条件、1 个 Empty/native 指标一致性条件。collector 仅携带既有分析和通过项数，没有改阈值或重新定义门；无论 all_gates_pass 为真或假都收集。Empty 与 M84 Empty 的逐帧 bbox/score 精确比较单独保存，frozen_M88_gate_changed 为 false，不影响原 14 门。正式评测和独立完成态审计标志仍为 false。

5. **证据归档覆盖本次必要文件。** 最终小型 adapter、初始 adapter、原始 sequence_log.jsonl 与 sampled_state_trace.jsonl、66 个预测、训练/递归日志及回执、实际 integration 源码、五份冻结 bank、方案/spec/准备/冻结/启动/审查记录、GT、原始分析，以及本收集器的源码、方案、审查和 launch 记录均纳入清单。sequence_log 的 130 条实际内容、最终参数有限性、指标、LOO、strict 损害/改善属于计划明确保留的后续完整审查，本次没有声称已读到尚未产生的最终文件。

6. **初始化和两次前检路径匹配已有准备方式。** prepare.py:33-49 确实写入 native_parity/category_zero.pth；本地该文件 SHA 为 d36353b6fcea8634a94d0c0051cb2d0c2f19061c9cd2e6cffb35e253e44b5334，与 training_spec 匹配。首次前检取 preflight_attempt1 下的 result/log/exit/launch 与两份历史 spec，第二次取根目录对应文件和最终 spec；RUNTIME.md 与 queue_line_endings.json 保留 CRLF 导致冻结失败及 LF 修正后重复前检的说明。最终 preflight_result 绑定当前 spec，两次短训练仍为丢弃的准备成本。主执行代理另行报告已在远端核验这批易缺静态路径存在；本审查未复做远端检查，也未把该报告标注为审查者实测。

7. **M84 双参照保留。** M84 Category 和 Empty 的回执分别绑定已冻结 parent_result 的 receipts 条目，22 个原始轨迹逐个核验并归档。M84 Category 用于结构参照；M84 Empty 用于本次完整 bbox/score 比较。关于 M84 Empty 已与独立 native 对齐的历史结论明确沿用已有审计；collector 不执行新 native 推理、不把模型输出当作性能 GT。

8. **执行边界正确。** 收集代码只用 Python 标准库读取、比较和打包，启动环境显式 CUDA_VISIBLE_DEVICES 为空；没有 GPU、caption、训练、在线文本更新、消息发送或正式三数据集评测入口。manifest 和 archive SHA 使用既有机制。run_collection.sh 实际字节未含 CR，未重现原队列的 CRLF 缺陷。本次仅做静态审阅和本地既有 SHA 绑定核验，未执行收集器或启动器，未构造模拟结果。

审查中发现并已由执行代理修复 1 个实际归档遗漏：原清单没有 prepare.py、preflight_m88.py、freeze_m88.py、launch_m88.py，而归档内准备/前检/启动/原审查回执引用它们。最终 collect_completed.py:46 已把四个现有脚本加入 root_names，build_collection.py 也同步该条目。审查者已复核修后源；没有新增 hash 框架、兼容层或防御分支。当前无遗留阻断项。

此 PASS 允许主线程继续既定收集部署准备。它不表示原训练已完成、归档已生成、性能已通过 14 门或完成态指标已独立复算。最终运行成功仍需实际 collector.exit、collection_result、下载归档与后续独立核验。
