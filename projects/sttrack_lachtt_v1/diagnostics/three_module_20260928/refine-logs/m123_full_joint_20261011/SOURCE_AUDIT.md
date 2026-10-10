# M123 full Train152 A+B+C source audit

**结论：SOURCE_ONLY_PASS，确定源码阻断点 0。** `source_gate_ready=true` 仅表示当前只读源码审计未发现必须先修复的实现问题，可据此封存正式部署 gate。它不是训练成功、双 GPU 实测、实际参数更新或正式指标的证明。该新路线尚未启动。

审计使用已安装 `experiment-audit` 及 `local-codex-policy.md`。请求路由为 `gpt-6-astra`、`max`；当前工具未提供可验证的实际后端身份，记录为 **unattested**。本审计为新上下文、same-family、provisional；不声称跨模型家族验收。

直接逐行阅读计划 `source_sha256` 中全部 26 个文件，以及 `prepare_M123_joint_plan_20261010.py`。26/26 文件当前 SHA-256 均与计划一致。对具体损失、原生设备行为、数据读取和官方度量依赖补充检查，未导入项目模块、构造模型、连接 SSH、查询 GPU、运行训练或评估。未修改实现。

文中未另标前缀的 `.py:行号` 位于 `C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928`。`R0/` 指 `.aris/m122_full_causal_20261007/R0_completed_audit_sources`；`STATE/` 和 `BACKBONE/` 分别指同目录保留的 `runtime_state_dependency_sources_20261009` 与 `runtime_backbone_dependency_sources_20261009`。

## 已核对的源码行为

| 检查 | 源码结论 | 证据与边界 |
|---|---|---|
| 全量 Train152 与完整预算 | PASS | `train_m123_ABC_joint_full152.py:34`、`:80`、`:94`、`:158`：按整个固定 `sequence_order` 运行 3 遍，无序列筛选、development 路由或缩短帧数参数。每遍 219,802 次 track，共 659,406 次、456 个完整序列运行。 |
| 当前训练无缓存特征拟合 | PASS | `train_m123_ABC_joint_full152.py:83`、`:85`、`:90`、`:95` 每次读取真实 RGB-D，初始化后沿自身状态生成下一次观测。旧 `train_m121_dense_target.main`、`train_template_write_C.main/common_samples` 未被新训练调用；导入其中损失、常量或摘要函数不会启动旧训练。 |
| GT 因果边界 | PASS | `JointFullABCTracker.step(image)` 不接收 GT。新训练 `:95` 完成定位、query、feature 与模板写入之后，`:99–117` 才用当前真实 GT 构造损失。合法首帧盒用于初始化；之后不以 GT 重置轨迹。`train_m122_full_causal.py:14–22` 固定 GT 文件哈希，并保留已有 toy07 的 1,367 个有效 RGB-D 帧截断。 |
| 双 GPU 分配 | PASS（源码） | 新训练 `:55–62` 在 GPU0 构造 decoder/writer，在 GPU1 构造冻结 native/CLIP。`m123_joint_full_tracker.py:23`、`:39`、`:78` 分别包住初始化、观测和模板预处理；`:25–28`、`:41–45` 将初始和当前观察输入搬至学习设备。没有在 GPU1 上执行 A+B/C 损失。 |
| 原生当前设备语义 | PASS（源码） | `R0/039_sttrack.py:29–36`、`STATE/lib/test/tracker/data_utils.py:18–26` 的 `.cuda()`，`R0/038_sttrack.py:71`、`:82` 的新 query，以及 `BACKBONE/lib/models/sttrack/utils.py:20` 的 mask 都依赖当前 CUDA 设备。上述 GPU1 上下文覆盖这些实际调用。后端 CUDA 扩展二进制未验证，不能推出双卡已运行成功。 |
| A+B 与 C 的梯度路径 | PASS（源码） | A+B 前向位于 `m123_joint_full_tracker.py:46`，不在观测 `no_grad` 区域内；当前盒残差经过 `.double()` 与 clip 仍连接 A+B。`template_write_features.py:28–40` 返回 detach 特征，但 `m123_joint_full_tracker.py:70` 的 writer 前向在该装饰器返回之后执行，保持 C 参数梯度。新训练 `:104` 和 `:113` 分别反传独立图，C 不需第二次反传 A+B 图。 |
| 冻结与状态一致性 | PASS（源码） | `full_dense_tracker.py:31–33` 冻结并 eval STTrack/CLIP，`:55–73` 无梯度观测。`m123_joint_full_tracker.py:55–57` 将提交盒、query、selected feature detach；C 决策先于 template crop。初始参考是每序列开始的 detach 快照；中途 A+B 更新不会重写已发生的历史状态。 |
| 损失、累积与更新 | PASS（源码） | A+B 使用既有 `precision_objective(...,1)`；真实 IoU 与目标相交监督来自 `train_m121_dense_target.py:19–66`、`m121_dense_inputs.py:97–108`。C 目标为当前选中盒的真实 GT IoU，非未来效用。新训练 `:122–140` 按各自有效监督帧数归一化、分别裁剪，然后同一 AdamW step；无 C 样本时其 grad 保持 None。末尾不足 32 帧的窗口也处理。 |
| 两模块均可学习 | PASS（源码） | `m123_joint_full_tracker.py:18–19` 断言所有学习参数 requires-grad；新训练 `:69–70` 同时将所有 decoder/writer 参数交给优化器。A+B 的 support、geometry、quality、observation 损失覆盖其相应模块；C 接收独立 MSE。静态层维度算术核对 A+B=380,167、C=70,721、总计 450,888。实际非零梯度和实际更新尚未观察；最终参数变化检查本身也不能证明有用学习。 |
| 固定最终模型与快照 | PASS（源码） | 新训练 `:155–157` 保存序列边界模型及优化器快照；`:158–172` 完整预算、冻结状态、分组变化检查之后只保存一份 `final.pt`，再 CPU reload 比较两组 state_dict。`latest.pt` 是进度快照；没有声称实现任意中断恢复。最终评估 `m123_joint_official_runtime.py:60–64` 严格装载相同 A_B/C 两组。 |
| 同一 final 的正式评估 | PASS（源码） | `prepare_m123_joint_evaluation.py:24–47` 继承同一 P1 数据/人审文本/协议，统一新 bundle 与 final SHA；`m123_joint_official_runtime.py:9–38` 重新核验。OPE 跟踪只用首帧初始化，全部输出封存后才在 `run_m123_joint_official.py:54–58` 打开随后 GT。VOT 通过合法 TraX 初始化，复用 127 序列/1,765 anchors，分片合并要求 5,295 份结果文件。 |

## 完整预算与已执行的静态检查

本地保留 M82 spec 的 SHA 为 `3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425`，与新训练硬编码 pin 一致。独立读取全部记录得：152 个唯一序列、每遍 219,954 张图像、219,802 次跟踪调用；三遍 659,406 次调用。逐序列计算的梯度累积窗口上限是 **20,829**；有效 GT 全缺失的窗口不会更新，故不能把该上限写成已完成的 optimizer steps。第一跟踪帧无 previous feature，C 可预测帧数上限为 **658,950**，不等于已获得的 C 监督数。

本地已执行 `review_trace/static_checks.py`，只用 Python 标准库读取文件、SHA-256、`ast.parse`、维度算术及 JSON 元数据核验。26 个直接文件与计划生成器均解析成功，原保留 52 项评估来源哈希逐项匹配。保留的正式输入元数据是 Test50/76,373 帧、CDTB80/101,956 帧、VOT127/1,765 anchors；Train152 与 Test50 的序列名不相交。**这些是源码/元数据检查，不是真实模型或 GPU 测试。**

最初执行默认 `python --version` 返回 `No pyvenv.cfg file`，没有执行语法解析；随后明确使用已存在的 uv CPython 3.11.14 完成标准库检查，没有安装依赖。该原始失败也保留在 trace 中。

## 继承依赖与部署封存范围

26 项直接映射不是整个 import 依赖集合。`train_m121_dense_target.py` 的实际损失调用 `m121_dense_inputs.observation_targets`；另有 7 个本地模块在顶层导入链中被载入，但其旧训练/缓存拟合入口没有被调用。新 source gate 应绑定**实际部署解析路径**的这些已有文件及原生 repository 依赖。不能仅把 Windows 的 26 路径映射原样当作远端完整封存。此处是部署封存工作，并非已发现的必然运行故障；父执行器已明确将补入继承 gate pins。

已直接核对的重要原始路径与 SHA-256 如下。完整本地路径、保留来源、读取范围、SHA 和逐字节快照见 `SOURCE_AUDIT.json` 及 `review_trace/reviewed_source_manifest.json`。

| 实际原始依赖路径 | SHA-256 |
|---|---|
| `/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/m121_dense_inputs.py` | `89651f43d40f1b13e16ab586d6c0aaa19e3edfcf91f59d3286f74cf093ef2d85` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/test/tracker/sttrack.py` | `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/models/sttrack/sttrack.py` | `d62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/test/tracker/data_utils.py` | `873337d0419ecfdd08b64123cdc22a4936ccdd6af98a092bc2106c32cf401fe5` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/train/data/processing_utils.py` | `7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/train/dataset/depth_utils.py` | `f97336af94a3ec196ad239603896fc0a2387dd1d61b48a4fbe2e055d9466e92e` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/models/layers/head.py` | `62cb2f363ea8bce491c9e6fc528076dd1e14c7d371ad69d6efd0b4062d063326` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/models/sttrack/vit_care.py` | `05096a6cec0448e49f82264ad34647e267a6a50a6d96d3ee07ba10d61ce6ef0e` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/models/layers/mamba.py` | `a11e49551cea9c7a188b9b7783b541b315a1f684b86e1de364bb3ad6c04f00be` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/lib/models/sttrack/utils.py` | `557ce8e3d352e3ec69b2d21fcb0f8c18703b3152eff6f3f2e1a7470585e634ea` |
| `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml` | `b6bda3238c9dd001aab62d87234d9ccecf1ae1cee3bd7bea5f21d6368ff4b344` |

新训练 `:40–47` 硬编码约束 spec、人审训练标签、训练文本 bank、STTrack checkpoint、CLIP 权重、warm final 和 warm result；`load_truth` 逐序列验证 GT 哈希。已保留 warm result 的当前本地字节与 `b9cbaf752c839afbf5854138c36c1e5cd35f58d024c45a694e56b3164e74435c` 一致，其声明的 final SHA 为 `cc71e0b4f42c2e489325627d0a16dcdfc4fa6bcb941908c30bc20c8c3c5828c0`。本次没有重新读取远端权重文件，故不独立重证该二进制存在或内容。

该 warm result 明确是过去固定 teacher 特征的 C 拟合，只有 142 个序列产生 eligible events。新计划将其仅作为 warm initialization；新代码重新遍历完整 152 个序列，既不将历史 400 个 C updates 抵扣三遍预算，也不把该旧产物当作新完整联合训练。这一界限必须保留。

## experiment-audit A–F

| 项 | 结论 | 说明 |
|---|---|---|
| A. GT provenance | PASS（源码） | 新损失来源是真实训练集 GT；外部 GT 只在输出封存后的分析器中使用。人审文本延续已声明的 multiframe-aids 协议，不能声称纯首帧自动文本。 |
| B. Score normalization | PASS（源码） | `R0/000_depthtrack_pr.py:117–142` 以置信度选阈值、计算真实 overlap 的标准 PR/F；没有除以模型自身最大输出来伪造分数。F 的最佳阈值是评价协议，不是 checkpoint selection。VOT 只将固定官方结果乘 100。 |
| C. Result existence | UNASSESSED（新路线未运行） | 没有新 M123 final、训练回执或九指标可验。本审计不产生这些结果。 |
| D. Dead code | PASS（目标调用链） | 新控制器明确依次调用完整训练、prepare、两套 OPE track/analyze、VOT track/analyze/seal、collect。旧 cached-fit 主函数属于未调用历史工具，不是新训练的执行路径。 |
| E. Scope | PASS（计划与源码一致） | 唯一新训练范围是完整 Train152、固定 seed2027、三遍；无新 pilot/split。结果有效性与性能范围尚无新实测证据。 |
| F. Evaluation type | real_gt（设计） | 官方数据集真实 GT，附带人审初始化文本协议；尚无已完成的新评价记录。 |

## 限制及审计边界

- 没有实际 forward/backward、双 GPU 内存/设备测试、训练、评估、冷加载推理或正式指标重算；不能将 AST 成功升级为运行成功。
- 不证明所有参数在实际数据上有非零梯度、C 在真实轨迹中有效写入、训练有限数值稳定或指标提升。源码里的最终断言尚未执行。
- 原生代码、正式输入元数据和 warm result 来自已保留的本地材料；未查询当前远端状态。数据帧字节和预训练权重没有在此重新逐一哈希。
- `mamba.py` 的实际 Python 调用路径已查；已安装的 selective-scan/CUDA/CLIP/PyTorch 二进制与版本没有独立验证。
- 本次检查确认模型权重最终序列化的源码闭环；不声称支持从任意中断位置恢复 optimizer、RNG 或 actor 状态。
- 26 个直接文件均完整阅读。补充大型原生库只检查具体调用/设备路径，其余只保留原文件 SHA 与明确读取范围，未冒充全库语义审计。
- 完整部署 gate 的依赖并集、远端实际路径/哈希及原任务资源准入仍需执行器正常封存；本报告不创建 gate、不修改原计划、不触发启动，也不要求增加新 pilot 或开发集。

## 阻断项

`blocks=[]`。本次没有找到可由当前源码和保留输入直接证明的实现阻断点。唯一接受的结论是**当前源码可继续正式部署封存，实际完整训练及同 final 三数据集验收仍未发生**。
