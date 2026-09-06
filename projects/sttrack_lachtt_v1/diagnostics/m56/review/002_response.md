**结论：WARN。未发现应阻止已定义的三组 fit 训练启动的具体 FAIL。** 训练与 runtime 的组别转换、fit-only 数值监督、M45 标签规则、相同初始化与优化预算均可定位为 PASS。尚不能把当前检查视为完整递归运行通过；以下问题主要约束递归接线和最终归因。

本次只读审阅使用 `experiment-audit`，未执行 SSH、GPU、训练或文件修改。

1. **WARN — checkpoint 记录了绑定信息，但 runtime 尚未核验。**

   训练 checkpoint 保存了 `variant`、`m56_spec_sha256`、`base_checkpoint_sha256` 和 `text_bank_sha256`；runtime 实际只读取 `variant`、加载 `model`，文本则直接取自调用方的 `info`。因此，当前包装器本身不能证明递归调用使用了对应 STTrack 底座、文本库和序列映射。现有材料中**未发现实际错配**，但这项完整性工作应由完整递归 runner 落实。

   最小处理是在递归入口绑定 checkpoint、底座和本次文本来源，并核对序列到文本的映射；当前 development 可核对既有 bank SHA。以后换数据集时，应绑定同一编码器与文本处理规则及该次评测 manifest，不能要求新数据集等于训练 bank 的整体 SHA。现有检查没有覆盖正式 checkpoint 保存后重载到 runtime 的完整链路。

   证据：[train_sttrack_m56.py:139–141](C:/Users/gb/.codex_remote_staging/train_sttrack_m56.py:139)、[sttrack_attribute_candidate_set.py:32–43](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_attribute_candidate_set.py:32)、[check_sttrack_m56.py:93–97](C:/Users/gb/.codex_remote_staging/check_sttrack_m56.py:93)。

2. **WARN — 三组参数量相同成立，但 pooled/empty 的第二次注意力具有确定的单槽退化，归因不能超出这个设计。**

   `pooled` 和 `empty` 都只保留一个有效文本槽。第二次 `guide_visual` 以视觉 cell 为 query、文本为 key/value；只有一个有效 key 时，softmax 恒为 1，同一对象内所有视觉 cell 得到相同的广播残差，query/key 分支不能通过注意力权重学习不同 cell 的文本选择。`attributes` 使用多个有效槽，具有这一选择能力。

   因而：
   
   - `attributes > pooled` 可以支持“保留独立短语并做多槽交互优于短语均值”的结论。
   - `attributes > empty` 同时包含类别与属性文本的贡献。
   - 单靠这三组，不能把增益进一步全部归为“属性语义起效”，或称为“有效空间注意力能力完全匹配”的对照。

   这是当前架构的实际性质，**不要求增加训练组或阻止训练**；保留计划中的固定权重文本干预，并准确限定结论即可。

   证据：[lachtt_attribute_candidate_set.py:9–21、38–46](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_attribute_candidate_set.py:9)。

3. **WARN — 现有“打乱属性”实际替换整套类别＋属性文本；冲突子集还存在槽数变化。**

   依据当前 manifest，按最新绑定脚本的确定性规则独立重算：
   
   | Development 对照 | 覆盖 | 同时改变类别 | 改变有效槽数 |
   | --- | ---: | ---: | ---: |
   | 划分内打乱 | 22 条 | 15 条 | 10 条 |
   | 同类别颜色词冲突 | 9 条 | 0 条 | 3 条 |

   三条冲突槽数变化为 `cup13_indoor`、`mobilephone01_indoor`、`mobilephone02_indoor`。因此打乱结果应称为“整套文本错配干预”；同类别冲突结果应明确报告 9 条覆盖及其中 6 条槽数一致的结果，避免把槽数变化导致的响应全部解释为颜色语义。这里可以利用既定逐序列输出完成报告，无需新增训练。

   最新计划已经正确限定为“标注颜色词冲突”，没有声称逐帧独立证真；这一修订有效。

   证据：[bind_sttrack_m56.py:43–68](C:/Users/gb/.codex_remote_staging/bind_sttrack_m56.py:43)、[text_manifest.json](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/text_manifest.json)、[EXPERIMENT_PLAN.md:40–42](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m56/EXPERIMENT_PLAN.md:40)。上表是对 manifest 的只读重算，不是已运行的跟踪结果。

4. **PASS — fit-only 数值训练实现成立，未发现 development 数值标签参与优化。**

   准备脚本显式解析旧混合标签后按 fit 名单物化；训练只读取 `fit_labels.json`，要求标签序列集合恰为 63 条 fit，逐缓存核对序列、fold、父 spec 和 SHA，并要求事件键与标签键完全相等且共 1,511 条。虽然冻结文本 bank 包含 85 条，进入模型的文本行索引只由 fit 视觉事件建立。

   这支持“训练脚本仅使用 fit 数值监督”，不支持把准备阶段描述成“从未解析过 development 标签”。当前收据已如实区分两者。

   证据：[prepare_sttrack_m56.py:68–75](C:/Users/gb/.codex_remote_staging/prepare_sttrack_m56.py:68)、[train_sttrack_m56.py:53–87、120–124](C:/Users/gb/.codex_remote_staging/train_sttrack_m56.py:53)、[text_preparation.json:51–54](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/text_preparation.json:51)。

5. **PASS — 三组初始化、事件顺序、预算和标签规则匹配。**

   三组各自重设 seed 2026、重新实例化同一网络和 AdamW、使用重新设种子的独立采样生成器；训练后比较初始权重摘要、事件顺序摘要、事件键摘要、参数量和步数。固定 20 轮、每轮 48 batch，共 960 步，仅保存最终轮。默认候选 IoU ≥ 0.5 时优先保留默认标签，current/previous 均沿用 M45 规则；匹配损失直接复用原函数。未发现 M55 权重载入或按 development 选择 checkpoint 的路径。

   证据：[train_sttrack_m56.py:91–109、116–157](C:/Users/gb/.codex_remote_staging/train_sttrack_m56.py:91)、[train_sttrack_m45.py:65–81](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/tools/train_sttrack_m45.py:65)。

6. **PASS — 训练和 runtime 的文本组别语义一致；实际工程收据与当前文件对应。**

   两端调用同一个 `condition_text`：attributes 保留有效短语，pooled 对相同短语取均值，empty 使用有效 CLIP 空串；runtime 从 checkpoint 取得组别，没有把 pooled/empty checkpoint 当 attributes 执行的源码错误。

   本次独立计算的准备脚本、检查脚本、模型源码和 manifest SHA，均与实际收据一致。收据记录的 448,739 父参数、35,648 新增参数、484,387 总参数、零残差精确保真、padding 精确保真和置换误差 `3.3378601e-6`，因此可归属到当前版本。

   **检查证明的范围仍需保持准确：**它用 attributes 做了三步优化，另外两组只检查有限前向；参数变化记录结合 AdamW weight decay，不能独立证明每个注意力子分支都有非零数据梯度。它足以支持已报告的有限反向与参数变化工程检查，不是三组正式训练或性能证据。

   证据：[train_sttrack_m56.py:112](C:/Users/gb/.codex_remote_staging/train_sttrack_m56.py:112)、[sttrack_attribute_candidate_set.py:33–43](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_attribute_candidate_set.py:33)、[contract.json:3–36](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/contract.json:3)、[check_sttrack_m56.py:67–97](C:/Users/gb/.codex_remote_staging/check_sttrack_m56.py:67)。

7. **PASS，附既有证据边界 — 历史标注来源及原生状态分布均已披露。**

   七条历史跨帧／裁定复核文本确实保留在 manifest，其中 development 是 `bag05_indoor` 和 `egg_indoor`。因此未来结论应属于“使用这些既有审阅文本的协议”，不能升级成严格只凭首帧自动生成文本的证明。cat04 裁剪差异已明确记录，未看到修改 GT 的代码。`previous_choice` 全 0 在训练中有显式断言，计划也已声明没有关联头改选后的状态补齐；不存在把本轮说成已解决该分布问题的当前文档错误。

   证据：[text_manifest.json:28–43](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/text_manifest.json:28)、[text_manifest.json:1072–1087](C:/Users/gb/.codex_remote_staging/m56_language_evidence_20260906/text_manifest.json:1072)、[prepare_sttrack_m56.py:58–66](C:/Users/gb/.codex_remote_staging/prepare_sttrack_m56.py:58)、[train_sttrack_m56.py:74](C:/Users/gb/.codex_remote_staging/train_sttrack_m56.py:74)、[EXPERIMENT_PLAN.md:50、60](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/m56/EXPERIMENT_PLAN.md:50)。

当前可支持**按冻结设计执行三组 fit 训练**。完整递归启动前需要落实第 1 项绑定；第 2、3 项应进入结果解释和对照覆盖报告。本次没有复跑真实缓存检查，未读取本地不存在的完整 bank／fit 标签二进制；相关实际数值由已匹配源码 SHA 的收据支持。