**验收结论：PASS。未发现完成态数字、文件绑定或公开说明中的具体 FAIL；没有新增需要阻止既定递归评测的问题。** “尚无独立静态属性语言优势”准确反映了当前结果。原有证据边界继续成立，完整 GPU 递归仍未验收。

1. **PASS — 私有包、正式权重和公开投影的真实字节对应。**

   我独立计算了归档 SHA，确为：

   `f5706dc87cb93388f91882a8e4f730c263363f39b092bd963c9572c5e1281c60`

   归档 manifest 中 **36 个文件的字节数和 SHA 全部匹配**；公开 `cpu_completed/`、`source/` 中 **34 个对应文件与私有包逐字节一致**。三组正式权重分别匹配各自训练记录、总训练记录及静态分析引用的 checkpoint SHA：

   | 权重 | 实际 SHA 前缀 | 文件大小 |
   | --- | --- | ---: |
   | attributes | `c68818cc78aacb05` | 1,958,431 B |
   | pooled | `ce38d9ab614e7c41` | 1,958,183 B |
   | empty | `64c8eb2eb72d4a3d` | 1,958,121 B |

   本次也直接核验了训练 spec `cafc4471…`、递归 spec `bdc367d0…` 和 text bank `93cd7285…` 的真实文件 SHA，补齐了上次仅有冻结标识、未取得实际 JSON 的证据缺口。模型、训练器、静态分析器、绑定器，以及本地包内 7 项冻结输入均与 spec 对应。

   来源：[cpu_bundle_manifest.json:71–145](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/cpu_bundle_manifest.json:71)、[training_result.json:3–16](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/training_result.json:3)、[spec.json:29–48](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/spec.json:29)、[static_result.json:3–6](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:3)。

2. **PASS — 实际训练完成态支持三组各 20 轮／960 步及配对公平性。**

   我逐项比较了 `train.log` 的 **60 条 epoch 记录**与三份训练结果：各组 epoch 连续为 1–20，累计步数每轮增加 48，最终为 960；全部记录的 loss、identity loss、matching loss 和耗时字段一致，损失数值有限。`train.exit`、`static.exit`、`cpu_controller.exit` 均为 `0`。

   三组完成记录中的 **60 项初始状态摘要完全一致**，采样顺序 SHA 和事件键顺序 SHA 完全一致，均为 484,387 参数、63 条 fit、1,511 个事件。实际 `fit_labels.json` 也恰有 1,511 个键、63 条序列，全部属于 fit／fold 2、3、4；与静态开发事件键没有交集。

   这里“初始化一致”是对完成记录中逐张量摘要、源码绑定和执行日志的验收，未重新构造初始化张量。

   来源：[train.log:1](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/train.log:1)、[attributes_result.json:2–74](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/training/attributes_result.json:2)、[pooled_result.json:2–74](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/training/pooled_result.json:2)、[empty_result.json:2–74](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/training/empty_result.json:2)。

3. **PASS — 六种静态模式、119 份逐序列统计及反事实配对均重算一致。**

   五种完整覆盖模式各有 **590 个唯一事件、22 条 development 序列，其中有效 GT 495 个**；冲突模式有 **225 个事件、9 条序列，其中有效 GT 191 个**。事件键、序列、默认 IoU、oracle IoU 和有效 GT 标记在各模式之间一致；相同事件选择同一候选时，记录的候选 IoU 也一致。

   我独立重算了六种模式的聚合、119 份逐序列统计、三种反事实配对和全部预定分层。整数统计全部一致；双精度行聚合与原 float32 汇总的最大均值差为 **`4.98×10⁻⁸`**，没有影响显示值或结论。

   | 同预算模型 | 报告平均 IoU | IoU ≥ 0.5 | 严重原错→正确 | 原正确→严重错 |
   | --- | ---: | ---: | ---: | ---: |
   | attributes | 0.450517714 | 275 | 7 | 3 |
   | pooled | 0.451725274 | 277 | 7 | 0 |
   | empty | 0.450666726 | 276 | 7 | 0 |

   原生候选 0 为 **0.440273792、正确 268 个**。attributes 比 pooled 低约 **0.001207560**，比 empty 低约 **0.000149012**；仅凭相对候选 0 的提高，不能得出独立属性语言优势。

   来源：[static_result.json:5914–5925](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:5914)、[static_result.json:12120–12131](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:12120)、[static_result.json:18326–18337](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:18326)。

4. **PASS — README 没有隐藏主比较伤害，局部反事实结果也没有被用于晋升。**

   attributes 的三次严重破坏可以准确定位为：

   - `colacan04_indoor@264`：默认 IoU **0.707831 → 0**。
   - `egg_indoor@46`：默认 IoU **0.698304 → 0**。
   - `egg_indoor@3547`：默认 IoU **0.851642 → 0**。

   README 表格明确报告 **3 次破坏，对照均为 0**，并保留完整事件结果，没有只报告救回数量。固定 attributes 权重的空串为 **0.449481308**、16 次选框变化；整套文本错配为 **0.450699627**、7 次变化，错配均值没有下降。9 条冲突子集真实文本 **0.472446471**、冲突文本 **0.468270272**；6 条同短语数子集为 144 个事件，结果与 README 一致。

   “局部正信号存在，但不足以抵消主比较未优于空文本／池化、完整错配不下降的限制”是与数据相称的表述，不需要改写成正向语言收益。

   来源：[static_result.json:2433](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:2433)、[static_result.json:3543](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:3543)、[static_result.json:3893](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:3893)、[static_result.json:33428–33640](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/static_result.json:33428)、[README.md:14–24](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m56/README.md:14)。

5. **PASS — 发布和递归状态边界准确，上轮队列标记 WARN 已关闭。**

   README 明确区分 590 个快照与 33,130 帧完整递归，明确说明未晋升、未获得 VOT 收益；公共核验脚本也明确只核验已发布事件行与完成记录。公开目录没有 `.pt`、`.pth` 或原始图像文件。

   当前包没有递归结果；启动收据为 `queued_not_tracking_yet`。队列修订后的成功标记位于协调分析成功之后，修订源码 SHA `2163be44…` 与首次队列启动收据及两份 queue plan 一致，所以上轮“提前成功”问题已在首次排队前修正。

   来源：[README.md:3、28–36](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m56/README.md:28)、[recursive_queue_launch.json:2–20](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/recursive_queue_launch.json:2)、[queue_recursive.py:49–61](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/queue_recursive.py:49)。

**WARN（既有证据边界，非新增失败）**：仍须保留七条历史跨帧／裁定文本来源、单槽与多槽的有效注意力容量差异、重复使用 development，以及原生轨迹缓存／`previous_choice=0` 的限制。当前 README、计划和训练记录已经保留这些限制，不能用本次 CPU PASS 消除它们。

本轮没有安装依赖、反序列化权重、重新运行模型或重算原始候选与 GT 的重叠；二进制验收限于真实字节 SHA，数值验收是对实际完成日志、结果和事件行的独立重算。完整 GPU 递归仍需以它自己的预测、退出码和分析收据验收。