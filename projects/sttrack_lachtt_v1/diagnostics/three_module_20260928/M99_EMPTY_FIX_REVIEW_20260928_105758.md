# M99 Empty 修复预部署源码复核

- verdict: PASS
- review_independence: same-family
- acceptance_status: provisional
- reviewer_model: gpt-6-astra
- reasoning_effort: max
- context: fresh agent / fork_turns none；实际 spawn 配置由主代理确认
- reviewed_at: 2026-09-28 10:57:58 +08:00
- scope: 本地源码与已保存实际证据复核；未使用 SSH/GPU，未改生产代码，未读取私有人工 mapping。

本次单行修复与已观测的 Empty 布局差异对应；更新后的 `check_empty_replay.py` 忠实保留原生产评分断言和首轮优化路径。没有尚未解决的 BLOCKING 或需要新增代码的 NON-BLOCKING 问题。此 PASS 是源码复核结论；修正生产源码的真实 red/green 回归、新 GPU sanity 以及最终训练尚未由本复核执行或认证。

## 原始证据与修复对应关系

读取了 `M99_AB_VISUAL_CONTROL_PLAN.md`、`M99_EMPTY_FIX_PREPARATION.md`、生产模型/producer/回归/诊断脚本，以及 `m99_failed_initial/` 和 `m99_empty_layout/replay.json` 的相关原始回执。

1. `m99_failed_initial/final_train.log` 明确在 `train_ab_visual_control.py:193` 的 `torch.equal(selection_logits, visual_selection_logits)` 失败，`final_train.exit` 为 1。归档回执记载失败训练没有 checkpoint；本地归档亦没有最终权重。现有两步 GPU sanity 曾通过，不足以覆盖此次第四批失败。
2. `replay.json` 的原布局分支在零起始 batch 3、已有 3 次 optimizer update 后复现同一评分相等失败；输入 stride 为 `(0, 0, 1)`。失败时 semantic 最大差为 `7.152557373046875e-7`，phrase 最大差为 `1.7881393432617188e-7`，score 最大差为 `4.76837158203125e-7`。四次 forward 均有限，前三批 loss、梯度和更新后参数均有限；失败批在原断言处停止，未声称其尚未执行的 loss/backward/update 已经通过。
3. 同一 seed、模型初始化、AdamW 和 fit 顺序下，仅 caller 的 `text.contiguous()` 变体完成 40 次更新；40 批评分均精确相等，semantic、phrase、score 的全程最大差均为 0，所有记录的 forward/loss/梯度/参数检查均有限。
4. 当前 `instance_ab_prototype.py:51` 相对失败源码的唯一变化为 `self.text(text)` → `self.text(text.contiguous())`。`train_ab_visual_control.py` 与失败归档逐字节相同。原 full 输入是 producer 的 expanded view；null 输入在 `instance_ab_prototype.py:84-85` 经相同 Empty 值和全 active mask 的乘法物化。共享 `phrase_branch` 在两条分支投影前规范布局，与实际 caller 变体所隔离的差异一致，保持值、形状和梯度关系，不新增参数、硬 Empty 绕过、fallback 或近似容差。

这些证据支持将故障归因到两条数值相同输入的布局相关计算差异；未追踪具体 CUDA kernel，因此不声称已定位底层 kernel 缺陷。生产修复的实际 GPU 效果仍由拟执行的 regression 验证。

## 回归是否忠实

当前 `check_empty_replay.py` 直接复用生产 `load_inputs`、`batch`、`freeze_unused` 和 `check_frozen`，读取同一 M90/M98/M95 缓存和 Empty bank 的 `empty`，只优化 `fit`。输入加载函数没有随机采样；在构建同一模型前重置 Python/NumPy/Torch/CUDA seed2027，保持同一初始化及首轮独立 Generator(seed2027) 的排列。模型保持默认 train 状态，没有改变训练路径的 eval 切换。

它保留生产最初 64 项的 native 零残差检查、原始 expanded stride `(0, 0, 1)`、相同参数冻结、batch64、AdamW `3e-4` 默认配置和两个等权 IoU BCE。每批首先检查原评分逐值相等断言，再执行相同 loss/backward/update，并检查有限 loss、梯度和更新后参数。semantic/phrase 最大绝对差累积到完成 40 批后要求都为 0；这避免较早的微小语义差异在原评分症状之前终止 red 回放。最终还检查被冻结参数没有变化。

`2544 = 39×64 + 48`，因此首轮确为 40 次更新。回归没有调用 development/public evaluation，也没有 checkpoint 保存。它会通过共享 loader 加载 development 数据，但模型输入和优化只取 fit；这不等于发展集评估。原源码 red 必须来自保存的失败源码运行，修正源码 green 必须完整完成此脚本；当前源码复核没有把 caller 诊断的 green 冒充生产修复 green。

## 训练、标签与输出边界

- producer 未变：former-fit130/2544 优化，former-development22/495 仅在最终训练后诊断；seed2027，batch64，AdamW3e-4，12 epochs × 40 updates = 480 updates。没有 best/development checkpoint 搜索，最终仅保存 `final.pt`。
- `batch` 排除 `iou`，非 tensor 的事件 key/strata 不进入 forward；GT 只在 loader 中计算现有候选 IoU，并作为 visual selection/quality BCE 目标及最终指标。`geometry_features` 来自 native 候选分数、cell、box、prior 等已有视觉状态。没有把 IoU 改称身份或语义真值。
- 文本输入始终是同一五槽 Empty 和全 active mask；没有使用类别、属性、人工短语、私有 mapping 或序列依赖文本 mask。语义/observation 参数冻结规则保留。
- 源码没有 tracker state 写入、模板更新、C/recovery 控制或公开数据集评测。Empty 精确相消只证明此对照的数值/函数一致性，不能证明非空语义有效或正式追踪性能。
- 既定后续顺序合理：原源码 regression red → 修正源码 regression green → 现有 GPU sanity → 独立 `corrected_run` 下重新初始化并完整训练 12 epochs/480 updates。保留失败输出与日志，不从失败任务恢复、不覆盖它；该新输出路径属于待执行命令的约束，本次没有伪称已验证尚未执行的 launch。

## 已完成的本地核验

使用现有可用 Python 3.13 对四份源码执行内存 `compile()`，全部通过，未导入 Torch、未运行模型、未产生 bytecode。核对原型 diff 仅一行且 producer 与失败归档字节相同。复核 replay 已有 source/prototype 摘要分别对应当前诊断脚本与失败原型；逐项检查原布局 4 个 batch 记录及 contiguous 40 个记录，字段和上述计数一致。检查回归 JSON 写入后缀为真实换行、可被 JSON parser 接受。没有新增摘要体系、flags、兼容层或额外运行门控。
