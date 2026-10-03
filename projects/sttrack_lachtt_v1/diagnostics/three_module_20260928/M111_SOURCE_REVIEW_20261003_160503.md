# M111 源码审查

结论：**PASS，限源码与本地标签绑定检查**。首次检查发现一个实际 FP16/FP32 阻断，主执行代理已修复，复读最终源码后没有未解决的阻断。可进入计划中的 GPU sanity；本报告不代表 GPU sanity、最终训练或效果已经通过。

时间：2026-10-03T16:05:03.282194+08:00。审查模型 `gpt-6-astra`，`reasoning_effort=max`；独立新上下文，`review_independence=same-family`，`acceptance_status=provisional`。没有模型回退，没有远程命令，审查代理没有修改实验源码。

1. **已修复的阻断 R1。** 初始化 RoI 在 `collect_initial_feature_origins.py:66` 存为 FP16，初始 context 在 `collect_train_contexts.py:103-104` 也存为 FP16；载入及 `initial_panel` 没有转换。初版 `category_inputs` 仅执行 `.to(device)`，会把 FP16 输入交给 FP32 Linear。当前 `train_m111_explicit_category_evidence.py:42-43` 已加入与旧 batch 完全相同的转换：bool 掩码保留，其余转 float32。这是已有缓存的实际类型要求，未增加 fallback。本次只验证源码修复，没有伪称复现 GPU 异常。

2. **原始短语绑定和拆分通过。** 私有 CSV、weak_labels 和 manifest 的 117 条记录逐项一致：原始 `generated_category.strip()`、状态、clear 可观察性、sequence、split、类别索引均匹配；四项 eligibility hash 与实际文件一致。fit 为 80/17/1，development 为 15/3/1，98/19 序列互不重叠；另外 35 条被排除。117 条 split 全部匹配保存的 M98 缓存回执，原 130/22 序列集合也无交集。与 M90 preparation hash 相同的训练标签副本独立重计为 2544/495 个有效 IoU 状态。没有把旧候选标签继承给纠正后的描述。

3. **初始化输入与梯度边界通过。** `INITIAL_FIELDS` 仅有初始 RoI、context、两个 mask 和深度有效性；同一初始化观测在新类别任务中作为单个 current candidate，其他事件 RoI、候选 GT、后续帧、审核说明不进入该前向。收藏代码把初始 context/masks 绑定到同一个 frame0 crop，ring 顺序与原型一致。原始类别只在 slot0，其他四槽 mask=False。证据张量为 `[B,1,5,3]`，`[:,0,0]` 给 CE `[B,3]`，目标为 long 类别索引；CE 前无 detach。绝对 evidence logits 接受监督，而非 Empty 差值。类别梯度进入 text、phrase slots、phrase_read 与 evidence MLP；text_read 的训练路径仍来自定位损失。

4. **对照目标、冻结 B 与 Empty 约束通过。** weight0 直接返回 `localization + preservation`，未把零权重 CE 分支接入 backward；weight1 仅增加一项未加权 CE。原型不存在运行均值状态或非零 dropout，独立 CPU generator 不改变定位采样。两臂种子、初值、采样、预算和 AdamW 参数相同；旧 M110 可训练前缀原样复用。其他参数、所有 buffers、全部 3039 状态的 Empty score/quality 都有精确相等断言。`instance_ab_prototype.py` 唯一 diff 是返回已有 evidence 张量，没有新参数或候选计算改动。上述是源码结论，未声称历史 M110 checkpoint 数值重现。

5. **sanity、最终 checkpoint 和结果边界通过。** 两臂先各跑独立 3 步 sanity，检查有限梯度与冻结/Empty；sanity 不保存 checkpoint，不读出开发类别准确率。driver 等待两者成功后才启动全新 12 epoch/480 update finals，不继承 sanity 权重，不按 dev 选 epoch。final.pt 会反序列化并逐张量比较；这证明序列化一致，不等同于新模型再次前向验证。定位评价仍使用真实标注框生成的 IoU；类别读出明确是模型弱标签一致性。代码计算 Empty/generic/reviewed 三条件、own-Empty 配对变化、rescues/harms 和旧四项容量检查，没有自动 promotion、递归训练或公共 benchmark 调用。

6. **聚合诊断与私有信息边界通过。** 独立重算 fit/development 原始短语种类为 59/14，混标签短语组 6/1、样本 21/3，多数类基线 80/98 和 15/19。新增 readout 的多数类与混标签子集正确数只用于 no_grad 评估。M111 manifest/weak-label 文件被 Git ignore；训练结果未输出原始类别字符串或逐例类别标签。报告仅包含聚合计数和 hash；本 driver 不发布类别 bank。最终发布仍应沿用计划，仅携带允许的源码、计数和聚合结果。

限制：类别来源是获授权的模型初审，不是人工确认或数据集类别真值；clear 过滤不消除多帧审核偏差。fit/development 各只有一个 uncertain 样本，类别分布明显不均衡，混标签子集只有 21/3 条；不能据此宣称通用视觉证据利用、当前候选身份或可见性判断。原始类别与纠正后的候选描述也不同；是否转移到候选选择，必须看固定 final 相对自身 Empty 的实际结果，不能用类别准确率代替。

验证范围：实际完成五个 Python 文件的 Python 3.8 语法 AST 检查、私有 JSON/CSV 绑定与计数检查、保存的缓存回执拆分检查及 git diff/ignore 检查。默认 python 曾报 `No pyvenv.cfg file`；随后使用已存在、明确定位的 Python 3.11 完成语法/JSON检查，该解释器无 Torch。本地原先假定的 inference_inputs.json 不存在；没有把缺失输入称为已验证，拆分证据来自以上明确列出的 M98 回执和同 hash 标签副本。未执行 GPU、Torch forward/backward、编码、训练或实际特征载入；真实运行和资源容量仍由后续 GPU sanity gate 验证。

最终源码 hash 和逐项机器可读结论见同名 JSON。M110 既有结果仍为 reviewed/Empty 同是 272/495，mean IoU 0.5304029679763121 对 0.5307076979870673；本审查没有产生新的 M111 指标或正式三数据集成绩。
