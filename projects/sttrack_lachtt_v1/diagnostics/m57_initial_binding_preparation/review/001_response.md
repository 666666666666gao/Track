**方案审阅结论：可以进入 M57 通用模块和 CPU contract 准备，未发现必须否定该 2×2 设计的 FAIL。** t0 提取可复用现有网络而不提交 query 状态；需要明确两处接线位置，并保留多槽对照的解释边界。当前没有 M57 新实现或真实 t0 提取结果可供验收。

1. **PASS — 当前 `initial` 的确在 t1 冻结；同图额外 forward 可以把来源限制到 t0。**

   原生 `initialize()` 只生成模板张量，随后将 `track_query_before=None`、`frame_id=0`。候选 tracker 到第一次正常 `track()` 才执行联合 template/search forward，然后调用 `before_decision()` 冻结 `initial`。因此，旧参照准确说是**首次正常跟踪时、受 t1 搜索图影响的模板段 ROI**。

   新方案可在初始化完成后，用同一 t0 图像和合法初始化框裁出搜索输入，直接调用：

   `network.forward(template=..., search=[...], track_query_before=None, ..., return_candidate_features=True)`

   网络的 query 在该次调用中创建并通过返回值输出；真正将它提交到 tracker 的代码位于 `track()`。直接 forward 后丢弃返回 query，不调用 `track()`、不执行预测框和模板更新，即可避免这些 tracker 状态变化。应沿用 `eval()` 与 `torch.no_grad()`。

   **该参照仍是“t0 同图搜索条件化的模板 ROI”**，不能描述成未受搜索上下文影响的纯模板特征。

   证据：[原生初始化:73–89](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py:73)、[候选 track:25–34](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_candidate_set.py:25)、[网络 forward:70–76、147–159](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/sttrack.py:70)。

2. **最小接线要求 — 在新 bank 创建之后，只赋值 `initial`，不要借用 `before_decision()` 完成预热。**

   M57 的初始化 hook 应放在 `STTrackCandidateSet.initialize()` 完整返回之后，因为其中会创建新的 reference bank。随后复用：

   `template_roi(features, 0, init_bbox).detach().clone()`

   并只赋给 `reference_bank.initial`。现有 `template_roi()` 在当前 128 尺寸、模板因子 2 的接口下，正好返回 `2×16×768`。

   **不能为了取 initial 而调用 `before_decision(t0_features, dynamic_tensor)`**：它同时设置 `dynamic` 和 `encoded_dynamic`，会令动态参照也从 t0 开始缓存，直到模板对象变化。只预置 `initial`，其余字段保持构造后的状态，才能只实现本次预定修改。

   `before_decision()` 的返回顺序是 `[initial, dynamic, previous]`；候选头使用 `refs[:2]`，再拼接于 10 当前、10 上一候选之后。因此所有四组统一预置后，**输入对象 index 20 对应 t0 参照**成立。

   证据：[bank 与 ROI:31–34、51–66](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_local_spatial_observation.py:31)、[bank 创建与输入截取:19–22、39–40](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_candidate_set.py:19)。

3. **最小交互要求 — 来源开关只替换第一次注意力的视觉 K/V。**

   按当前模块结构，清楚的 2×2 接线是：

   - `read_visual` 的 query：短语／空内容表示，加共享的 slot embeddings。
   - `read_visual` 的 K/V：各对象自身 cells，或**交互前**的 `cells[:, 20]`，按对象广播。
   - `guide_visual` 的 query：始终保留各对象自身 cells。
   - `guide_visual` 的 K/V：上一步得到的条件化文本槽。

   这样来源因子只改变“文字从哪里读取视觉信息”。若把第二次注意力的 query 也替换成 t0，就同时改变了候选局部视觉的读取路径；若从 `self.context()` 之后的 index 20 取源，该表示已经混入当前候选信息，不能再称为固定 t0 视觉源。

   此外，四组都保留 index 20 和后续对象间注意力，所以这个对照检验的是**文本条件化的视觉来源**，不是“有无初始参照”。要解释 t0 与文本内容的交互，应同时看两个来源下真实文本相对空内容的增量，不能只比较两组真实文本。

   证据：[现有两次注意力:38–46](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_attribute_candidate_set.py:38)、[对象拼接及后续 context:57–68](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/lachtt_attribute_candidate_set.py:57)。

4. **WARN — 多槽方案能消除已知的单槽退化，但 slot 初始化必须打破槽间对称。**

   四组使用相同有效 mask、相同数量的 slot embeddings，并让空内容覆盖所有有效槽，是合理的匹配方式。**这些可学习槽必须彼此具有不同的初值，同时四组共享同一套初值。** 如果所有槽从全零或同一个向量开始，空内容下各槽的输入、注意力输出和梯度仍相同，槽间对称可能一直保持；“有多个参数槽”本身不能证明已获得独立视觉查询。

   另有两点解释边界：

   - 空内容组保留由原描述决定的有效槽数，应称为“去除文本内容、保留槽结构”的对照。
   - 加入与槽位置绑定的 embeddings 后，不能继续要求“仅置换短语而不调整槽对应关系，输出仍不变”；短语到槽的对应关系应固定并在四组中一致。

   这不要求增加训练组，也不要求将所有 padding 槽改成有效槽。

5. **WARN — 当前纯 CPU contract 不能验收原生 t0 提取本身。**

   原生 tracker 构造时使用 `.cuda()`，网络 forward 内创建 query 也硬编码到 CUDA。因此，现阶段 CPU contract 可以验证新交互模块的形状、四组参数／初始化、mask 和槽行为；不能据此宣布未改动的真实网络已完成 t0 提取及状态保真验证。

   后续正式接线还必须让缓存与 runtime 使用同一 t0 提取方式。现有 M56 缓存中的 initial 是 t1 来源，不能仅改名为 t0，或只在 runtime 替换而训练仍使用旧参照。

   证据：[原生构造:24–30](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py:24)、[forward 中 CUDA query:70–82](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/models/sttrack/sttrack.py:70)、[当前 initial 冻结:59–64](C:/Users/gb/.codex_track_publish_m29_20260902/projects/sttrack_lachtt_v1/overlay/lib/test/tracker/sttrack_local_spatial_observation.py:59)。

最小实现应放在 **M57 新文件中的初始化 hook 和交互类**，复用现有 ROI 函数；不要修改 M56 已绑定的 `condition_text`、交互模块或 reference bank 默认行为。这样可保持当前 M56 源码／spec／队列完整，同时准备这项明确限定的初始化参照与文本内容交互实验。