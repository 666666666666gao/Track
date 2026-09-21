# M88 代码审查

结论：**PASS，blocking_issues = 0；same-family / provisional**。当前实现可以进入方案规定的真实 GPU 预检；预检通过后继续冻结、完整训练和三组完整递推。本审查不代表 GPU 预检通过、性能达标或跨模型家族认可。

审查模型：`gpt-6-astra`；reasoning effort：`max`。审查从独立上下文读取本地源码及准备证据。未修改实验源码，未执行远程命令，未启动 GPU、训练或推理。已读取 experiment-bridge 及 shared local-codex-policy。

## 阻断问题

无未解决的阻断问题。准备阶段的 bank 整数 metadata 遍历错误已改为枚举五个实际 bank；CPU 置换与 Empty 梯度的数值说明已在冻结前写明，并通过 `numerical_clarification.json` 重新绑定当前方案与两份 spec。此次修改没有增加模型分支、改变损失或放宽性能门槛。

## 核对结果

1. **局部检索与 centered 语义正确。** `semantic_spatial_adapter.py:24-28` 的张量形状为 query `B,K,H`、search `B,P,H`、reference `B,J,H`；两项 logits 广播相加得到 `B,P,K,J`，沿 J softmax，读取结果为 `B,P,K,H`，对应方案的 `(q_k+r_p)·z_j/sqrt(H)`。`forward` 用局部 bound 同时生成 evidence 与 context，modality 仍只由初始化静态 bound 决定。屏蔽槽在 null softmax 前被置为负无穷，各槽的检索/归一化没有跨槽耦合，屏蔽文本不会污染有效 context。模块定义和参数顺序与 M84 相同，没有新增可训练参数。

2. **保留共享双分支与准确的数值边界。** 继承的 `centered_semantic_adapter.py:12-20` 先计算 `delta-empty_delta` 再加 fused，两支均保留梯度。CPU 记录显示 Empty 输出精确恒等；参数梯度的最大浮点余量是 `2.384185791015625e-7`，原 M84 同张量/同权重对照也为同一数值。因此可以声明数学抵消及本次数值近零，不能声明全部参数梯度逐位为零。末层 bias 梯度仍实测为 0。原 M84 数值对照使用 sum loss，因两支上游导数分别为正负 1，其加法括号差异不改变这次梯度对照的含义。

3. **M84 是实际结构对照。** 对比完整训练源码，M88 与 M84 只有实验根路径和 architecture tag 两处差异。准备 spec 的 banks、fit 顺序、seed 2027、AdamW 学习率/weight decay `1e-4`、clip `1.0`、32 帧窗口、preservation 权重及总预算保持相同。130 个 fit 序列与 22 个 development 序列交集为 0；目标为 186694 次 tracking 和 5798 次 optimizer step。准备脚本从校验过的 M84 zero checkpoint 载入，仅更改架构 metadata，并在写回后检查模型张量相等；没有载入 M87 的新文本 bank，也没有继承训练 optimizer 状态。初始化文件在准备快照中的实际 SHA 与当前 spec 一致。

4. **模型接入绑定正确。** 准备快照的 161 个 integration 源文件均通过本地 SHA 核对；相对 M84 仅 `lib/models/sttrack/semantic_spatial_adapter.py` 与 `lib/test/tracker/sttrack_semantic.py` 改变。已确认前者与本次根目录审阅文件一致，后者要求 `semantic_spatial_local_reference_v1`。训练最终 checkpoint、评估入口和 loader 使用同一新 tag。继承的 causal、support、competition、preservation 与 metric 文件都与已存 M84/M87 支持证据一致，当前 spec 也绑定了实际文件。

5. **因果状态与 GT 顺序正确。** `CausalTrainingTracker.step(image)` 没有 GT 参数；crop 来自上一步自身预测，native 只在同一 crop/query/template 上作为 detached loss teacher，公开 bbox 与模板先按适配后预测提交，`train_causal.py:125-129` 随后才把当前 GT 传入损失。base 网络保持 eval/frozen，历史 query、crop 和 template 不携带跨帧梯度。初始化 RoI 仅由第 0 帧及其允许的 init box 提取。`run_recursive.run` 不打开后续 GT；`analyze` 先核验三组完整轨迹、退出状态与 receipt，再加载 development GT。指标直接对照数据集 GT；native/M84 只作为已完成结果比较项，不替代监督真值。

6. **14 个门禁与方案一致。** `run_recursive.py:114-137` 分别给 M84、native、Swapped 计算 pooled IoU 严格提高、macro IoU 不降低、low-IoU 帧数不增加、H10 不增加；native 再增加零 H10 序列保护，Empty 再增加逐序列 native metric parity，合计 `4+5+4+1=14`，推广要求全部成立。门禁引用的 parent 路径是已绑定的 M84，未误用 M82/M87。metric 排除初始化与无效 GT，H10 是至少 10 个连续有效低 IoU 帧；无效帧会打断连续区间。LOO、所有逐序列原始指标均保留，能够支持完成后的描述性分析。

7. **预检恢复与失败停止顺序正确。** 真实预检先比较 zero Category 与非零权重 Empty 各 101 帧，再恢复初始 adapter 张量、重新初始化 tracker、创建全新 AdamW 做 96 帧/3 step smoke。smoke 参数不落盘；正式训练在独立进程重新读取初始 checkpoint 并创建新 optimizer。`freeze_m88.py` 要求本审查无阻断、源码一致、preflight 退出码 0、成功 receipt、3 个丢弃 step 及 0 个正式 step。队列在训练或任一递推失败时写非零状态并停止后续分析；只有三组递推完成才执行 GT 分析。该控制流可以支持用户要求的完整实验，不需要增加额外框架。

## 已审阅的 CPU 证据

`prepared/model_check.json` 为执行代理生成并由本审查读取核对的记录，不是审查代理独立重跑：

| 检查 | 记录 |
|---|---:|
| 参数数量 | 289154 |
| 显式循环与向量化最大误差 | 7.152557373046875e-7 |
| search 置换最大误差 | 2.384185791015625e-7 |
| Empty 最大参数梯度 | 2.384185791015625e-7 |
| 原 M84 Empty 最大参数梯度 | 2.384185791015625e-7 |
| RGB / Depth / text 投影梯度 | 均有限且非零 |
| 末层 bias 梯度 | 0 |
| tracker calls / optimizer steps | 0 / 0 |

batch、屏蔽槽、reference 置换、search 置换、当前 search 改变 bound 与精确 Empty 输出检查均已在 CPU 脚本中执行至 PASS。当前两份 spec 与数值说明的 SHA 关系、本地方案/训练/队列/预检文件及 integration/初始化绑定均已由本审查重新核验。

## 后续实验义务与结论边界

没有待修补的源码问题。真实 GPU 预检仍待实际执行，必须通过才能冻结和启动完整训练。完成后按方案补充逐帧 Empty/native 或已保存 M84Empty 的 bbox/score 相等检查、strict damage/improvement、LOO 与全部负向行；这些属于结果审计，不是新增推广门，也不要求塞入推理入口。保存三组全部轨迹后再做这些分析即可。

本次是一个固定 seed 的重复 development 实验。CPU/GPU 正确性检查不能证明语义贡献、tracking 提升或三个官方数据集达标。初次 `preparation_receipt.json` 记录的是数值说明前的 spec；当前版本应以 `numerical_clarification.json`、本审查所列当前 spec 及后续 frozen receipt 为准，保留原回执而不改写历史。

## 冻结前队列换行修正复核（2026-09-21）

本节更新当前状态；上文首次审查按当时证据保留。**复核结论仍为 PASS，0 个阻断问题，same-family / provisional。**

执行代理报告首次 GPU 预检退出码为 0、耗时约 32 秒，但随后冻结阶段的 `bash -n` 因队列 CRLF 换行失败，完整训练未启动。首次预检已保存至 `preflight_attempt1`。本次只复核这一实际失败对应的队列文本修正和 spec 绑定，不扩大到模型、损失或实验方法。

本地已确定性验证：

- 当前 `run_m88.sh` 不含 CR；将每个 LF 还原为 CRLF 后，文件 SHA 精确恢复为上一版收据中的 `c94d778f4a51ca094940f3eb44d4e0ef7167ae17766e5ef15c032ac13a8c5f33`。因此改动只有换行，没有任何 Bash 命令或控制流变化。
- 其余九个根目录审阅文件 SHA 全部与首次收据相同。
- 当前 training spec 只更新 `run_queue_sha256`；当前 recursive spec 只更新队列 SHA 与其引用的 training spec SHA。把这些字段还原后，两份文件的 SHA 都精确恢复为首次审阅值。方案、模型、bank、损失、优化器、数据和性能门禁均未改变。
- `prepared/queue_line_endings.json` 的三个当前 SHA 与实际本地文件一致。执行端的 `bash -n` 通过由该修正回执记录；本次复核未执行远程命令或重复 GPU 检查。

当前绑定：

| 文件 | SHA256 |
|---|---|
| run_m88.sh | 2c62c710fe009f72219443e58958428b710a2625652fadc5bc3c117711dd402f |
| training_spec.json | 0d82b358e88525b17474e50215afd84f39898f40c7ff1b0df13b44bc92da4428 |
| recursive_spec.json | 06c6920905baa4613c1c1c4a846a1838346d19b9c7a7a55fe2e065806ff4262f |

`numerical_clarification.json` 继续保留数值说明历史；换行修正后的当前 spec 以 `queue_line_endings.json` 和更新后的本审查收据为准。旧报告及旧收据已经按首次审查 UTC 时间另存，未覆盖其内容。

下一步是在当前最终 spec 绑定下重复原定短 GPU 预检，成功后继续冻结和完整训练。旧的成功预检不冒充新绑定的成功结果，不需要向冻结脚本加入兼容分支。
