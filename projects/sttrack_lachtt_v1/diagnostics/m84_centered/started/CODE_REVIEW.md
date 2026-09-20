# M84 中心化语义残差实现审阅

审阅日期：2026-09-20。结论：**PASS（当前实现与已完成前检）；0 个部署阻断问题，0 项必须修改的实现补丁。** 研究有效性仍为 **WARN / 未测**，本报告不表示 M84 性能成功。

这是新上下文的同系列 Codex 代码审阅，结论为 same-family / provisional，不是外部独立研究审稿。审阅仅在本地读取 M84、M82 文件并执行语法与门槛逻辑检查；没有启动远端任务、修改实验源码或读取凭据。执行者后来新增的 `freeze_m84.py` / `launch_m84.py` 不在本次代码审阅范围内。

审阅根目录为 `D:\Program Files\UserCache\gb\codex\tmp\sttrack_m84_centered_20260920`，父实现为 `D:\Program Files\UserCache\gb\codex\tmp\m82_complete_20260920`。下文未注明父目录的文件均相对 M84 根目录。

可以在完成既定封存后进入计划限定的单个 seed2027 Category 正式训练。必须继续使用固定 final、三个完整内容递归和全部 14 项事先门槛，不得将本报告或 smoke 的可学习性写成开发集收益。

## 实现与协议核对

| 核对项 | 结论与代码位置 |
|---|---|
| 共享空词差分 | **PASS。** `centered_semantic_adapter.py:12–20` 对同一个父 adapter 连续调用两次，分别传入原文本和固定 `empty_text`；两次都传零 fused，先算 `delta - empty_delta`，最后加回真实 fused。父 `code/lib/models/sttrack/semantic_spatial_adapter.py:25–52` 只在第 51 行使用 fused，因此该取 delta 方法与当前父实现严格对应。没有在 head 输出上相减。 |
| 两支梯度与空词恒等 | **PASS。** 两次父 forward 都在正常 autograd 内，没有 detach 空词分支；只有编码本身作为不训练 buffer 保存（`centered_semantic_adapter.py:7–10,15–20`）。父 adapter 无 dropout、无 BatchNorm 状态写入。768 维末层 bias 在差分中抵消；计划第 9 行已披露 289154 个存储参数与最多 288386 个有效维度的区别。 |
| 原始零初始化、单训练臂 | **PASS。** `prepare_m84.py:48–54` 从 M82 `native_parity/category_zero.pth` 取初始状态，只增加空词 buffer 和新架构标识，检查末层原为零、原张量保持不变。`train_causal.py:35,67,86–91` 只接受 Category，从该零状态开始，保存 centered 架构。`run_m84.sh:7` 只有一个训练调用，不存在新增 Empty 训练。 |
| M82 mask-control 的复用前提 | **PASS（依据准备脚本和已下载 receipt）。** `prepare_m84.py:37–47` 对 fit Category、dev Category、dev Swapped 的每条类别槽检查 mask 为真且实际向量与 empty 不相等，并核对相同空词向量。`preparation_receipt.json:3–27` 分别记录 130/22/22 条全部非空，最小向量最大绝对差约 1.5971/2.5740/2.1020。训练初始化只传入这些固定 bank，没有文本 dropout（`train_causal.py:73–74,108–110`；`code/lib/test/tracker/sttrack_semantic.py:20–30`）。因此“仅 Empty 归零、非空不变”控制在这些既定非空训练/推理输入上与 M82 相同。实际完整 bank 与初始权重位于远端，本审阅没有在本地独立逐向量重算；没有将 receipt 说成审阅者重新运行所得。 |
| 冻结 base / head，正确保留当前帧训练梯度 | **PASS。** `causal_training.py:15–18` 冻结整个模型，只打开 adapter 的参数梯度并保持 eval。第 28–31 行在 no_grad 下算 native 特征，第 43–46 行重新通过 adapter 和冻结但可对输入求导的 head；base 不接受优化更新。正式运行结束检查 base 参数及 buffer 状态与梯度（`train_causal.py:179–182`）。 |
| 递归状态与 GT 边界 | **PASS。** `causal_training.py:21–69` 的 `step` 没有 GT 参数；当前 native 教师框只读，Category 的 Hann 解码预测提交 bbox，query 来自当前 native backbone，模板按预测框及原始 50 帧 / 0.75 规则更新。`train_causal.py:125–129` 在状态提交后才把当前 GT 传给损失。没有后续 GT 重置、跨时间反传或由 GT 选取公开预测。首帧 ROI 提取只读且不提交辅助 query（`code/lib/test/tracker/sttrack_initial_instance_observation.py:8–20`）。 |
| 损失、顺序与优化预算 | **PASS。** `causal_training.py`、`window_competition.py`、`native_preservation.py`、`support_loss.py` 与本地 M82 对应文件逐字节相同；raw focal、2 GIoU、5 L1、原始 raw 竞争和 weight-1 同状态 native KL 保留。`native_preservation.py:9–13,23–44` 教师 detach，GT 有效且中心在 crop 内、native IoU≥0.5 才启用空间 KL。`train_causal.py:139–163` 每 32 个观察帧结算，并按该窗口有效监督帧数平均。130 条顺序及所有 bank 的 spec 与 M82 完全相同，fit/dev 不相交。总跟踪转移 186694；5896 是按原始帧窗口计算的上界，5798 是预期实际优化数，不能互换。正式末尾第 177–178 行分别断言转移和实际步数；该 5798 也与 M82 完成 receipt 一致。 |
| 保存和加载新权重 | **PASS。** final 包含 centered 架构、adapter 完整 state_dict（包括 `empty_text`）、base 标识、训练 spec、优化器及完成状态（`train_causal.py:86–91,183–196`）。加载器按 centered 架构创建对象并 `strict=True` 加载（`code/lib/test/tracker/sttrack_semantic.py:12–18`）。`run_recursive.py:22–27,42–50` 要求训练成功、130 条 / 186694 次 / 5798 步完成，只加载已绑定的固定 final，并检查开发 bank 空词等于保存的空词 buffer。没有从中间 loss 选权重。 |
| 三个完整内容递归先于 GT | **PASS。** `run_m84.sh:17–25` 先完成 Category、同权重 Empty，再完成同权重 Swapped，最后分析。`run_recursive.py:53–74` 的推理循环仅使用既定首框、RGB-D 帧和固定文本，存完整逐帧框/分数；第 81–96 行先核对全部三组的退出码、完整 receipt、预测顺序与有限合法框，再在第 102–105 行读取开发 GT。每组 22 条、33130 个含首帧的预测；指标排除首帧及无效 GT，合计 28897 个有效评测帧。 |
| 14 项前瞻门槛 | **PASS。** `run_recursive.py:113–123,132–137` 与计划第 19–22 行逐项对应：4 项 M82 增量、5 项 native（含原生零 H10 序列保护）、4 项 Swapped 内容、1 项逐序列 Empty/native 指标恒等。均值严格提高，宏均值允许相等，低 IoU 帧与 H10 不增。Empty 对每条序列分别检查有效帧数 / 低 IoU 帧 / H10，且 IoU sum 绝对差≤1e-8；没有把 Empty/native 再记作独立训练收益。 |
| 结果报告范围 | **PASS。** 第 100–112 行保留每组和每条序列的真实 GT 指标；第 118–129 行保留零 H10 损害名单和两个内容 LOO；第 135–136 行明确是反复使用的 Train development22、一个训练模型的三个内容递归、自动 caption 未经语义真值校验，且不自动启动公开评测。负结果不会触发切换主模型或更改门槛。 |

## 实际核查证据

- 本地验证了 integration 中全部 **161 个文件**的现有摘要。与 M82 原 manifest 相比 **159 项保持相同**，只修改 `lib/test/tracker/sttrack_semantic.py`，新增 `lib/models/sttrack/centered_semantic_adapter.py`。其中本地父快照可直接比对的 145 个未改文件逐字节相同；父本地快照缺少的 14 个 data-spec 文本仍与父 manifest 值相同，未将 manifest 比较描述成直接文件比较。
- `train_causal.py` 相对 M82 的差异仅为 Category 参数选择、新根目录、新架构名、说明文本及最终 5798 步断言。底座、head、Hann、首帧 ROI、crop、query、模板更新实现及配置未改变。
- 11 个被审阅 Python 文件通过本地 `ast.parse`；`run_m84.sh` 通过 `bash -n`，退出 0。
- 直接抽取当前 `run_recursive.py:113–123` 的实际门槛语句，用合成逐序列/聚合字典执行 **22 项检查，全部通过**。覆盖严格均值、其他指标的相等边界、各组单项拒绝、总 H10 不增但原生零 H10 序列受损、Empty 的每个整数字段不一致、IoU sum 容差上下两侧。这是门槛程序检查，不是实验成绩。
- 核对计划、训练 / 递归 spec、父结果、bank 配置与现有代码绑定均一致。审阅时本地正式 `training/` 与 `frozen.json` 均不存在，属于准备状态。

执行者已完成的 CUDA 前检由 `preflight.exit=0`、`preflight_result.json` 和匹配的源码 / spec 标识支持；本审阅检查了这些文件，没有独立重复 CUDA 运行。该 receipt 记录：

- 零参数 Category 与独立 native 的 **101 帧** bbox、score、query、template 精确一致；非零 M82 参数配 Empty 再次 **101 帧**精确一致。两次都实际在第 100 帧出现模板写入，覆盖了更新后的状态核对。
- 96 个实际训练转移，其中 **88 个有效监督帧、3 次优化**。第一步只有末层 weight 获得任务梯度；后续 rgb/depth/text 权重均有非零梯度，末层 bias 梯度为零。非零梯度证据与参数变化共同支持当前参数化可学习，不把 AdamW 衰减单独算作任务学习。
- base 参数和 buffer 状态为 `c07022117c6efa8399b6df57e275ec81c6f3efce44eca5e4bc0282b52dd0dae0`，与 M82 已完成训练的初始 base 状态一致；smoke 后保持不变，Empty 特征再次精确等于 fused。smoke 权重没有保存，正式优化步数为 0。
- 该 receipt 的 `training_spec_sha256` 为 `0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5`，与本次审阅的训练 spec 一致；`preflight_m84.py` 摘要也一致。

## 非阻断 WARN 与结论边界

**WARN：尚无 M84 性能结果。** 101 帧旧 M82 权重接口读出、合成 CPU 代数检查和 96 帧 smoke 均不能证明新中心化训练会更好。原方法审阅建议的完整固定权重开发集读出没有在本轮追加；`EXPERIMENT_PLAN.md:26` 明确记录了这项取舍。本代码审阅确认实现与已定计划一致，不将该取舍写成研究审阅者已经批准效果。一个 seed、反复使用的开发22、可能错误的自动类别及增加的 adapter 算量都限制未来结论范围；计划已披露这些限制。

**WARN：现有 latest 是恢复用快照，但训练入口未实现直接续跑。** `train_causal.py:86–91,175` 保存优化器、已完成序列数和计数；但参数解析只有 `--arm`，第 67 行始终装载零初始化，第 76/97 行拒绝复用既有输出目录/日志。因此进程中断后不能把原命令直接重跑描述成从 latest 续训。它是沿用 M82 的现有行为，不阻断这次完整新运行；若确实中断，应保留快照与日志，再做限定的恢复处理，并验证已完成序列和累计步数，不覆盖旧输出或误报为独立完整训练。当前无需为尚未发生的中断新增恢复框架。

**无需更改当前实验实现。** 以本报告审阅的文件及已通过前检为准，单臂训练、固定 final、三组完整递归和 14 项门槛均可按计划执行。最终研究判定必须等真实完整结果，再如实保留全部通过与未通过项。

