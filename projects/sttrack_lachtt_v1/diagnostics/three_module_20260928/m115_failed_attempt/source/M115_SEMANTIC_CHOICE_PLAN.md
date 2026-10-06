# M115：人工初始化文本的直接候选竞争读出

所有后续训练/正式推理使用用户确认的初始化类别和稳定属性，序列名前缀仅作
核查线索，不覆盖confirmed_category或不确定项。Test/CDTB/VOT人审仅用于
各协议初始化输入，不进入优化、开发阈值或事件采样。人工曾参考多帧，是额外
离线输入监督，不能称首帧自动生成。generic/Empty仅为明确标记的机制对照。

M113实际272/495等于自身Empty；M114实际人工开发增量能量92.18%为共同平移，
223个好/严重偏离框状态中204个好框优势缩小。它支持检验独立选择读出，
不构成BCE损失唯一根因或全体语言无效的证明。现有10候选开发上界305，
190状态没有合格框；不能据此推断全256候选或crop外容量。

新增semantic_choice_prototype.py，不修改已封存InstanceCandidatePrototype。
严格继承M101视觉父模型，旧semantic/phrase读出置零并冻结；视觉特征、
视觉选择及质量冻结。训练父模型91,395文本交互参数，加8,512独立评分头，
合计99,907。评分头读取64维视觉、64维语义增量、3维短语增量，131→64→1，
最后无bias且零初始化。完整语义与同视觉零语义的评分相减，然后在候选维
去均值；加回父模型选择logits。Empty结构恒等，零更新所有条件等于Parent。
所选框、分数、质量、特征全部按同一index提交。本项不提交跟踪历史；新的
选择logits不当作模板更新概率，旧0.75阈值不机械沿用。

三臂同一seed2027、初值、152人审bank、130fit/22development划分、12轮480
AdamW更新、batch64/lr3e-4、固定final。先GPU0 human_bce / GPU1 human_choice
分别3更新sanity通过，再各480；随后GPU0 generic_choice的3更新sanity再480。
不跑填空任务，不重新编码或调用Qwen。预计总2–4分钟，根据M113实测；
启动后240秒观察一次，不频繁轮询，不自动重试。

| 臂 | 文本 | 定位目标 | 保持目标 |
|---|---|---|---|
| human_bce | 人工类别/稳定属性 | 每候选真实GT IoU的BCE | 可靠Parent好框对严重偏离框的原间隔 |
| human_choice | 同一人工文本 | IoU≥0.5候选按IoU加权归一化的softmax CE | 相同 |
| generic_choice | 有效槽均object | 相同相对CE | 相同 |

相对目标没有≥0.5候选时不伪造正例；IoU只监督定位，不作为不同物理实例
或属性冲突标签。保持项在冻结Parent选中框GT IoU≥0.5时，保护该框相对
IoU≤0.1候选的分数差；教师detach，GT仅Train优化/开发计分。本项从M113的
native候选0保持改为实际Parent可靠关系，明确披露，因此M113→M115不是
单因素因果消融；M115两人工臂才是匹配定位目标对照。

sanity/完整检查实际非零有限文本/评分梯度、视觉参数/buffer不变、全部3039
Empty分数/质量逐值相同、非空质量不变、所有选择字段一致、final序列化。
优化期间保持eval模式、梯度仍启用：冻结Parent的候选自注意力因此与原
推理采用相同路径。已记录环境torch1.13.1，官方实现依train/eval选择不同
attention路径；不通过放宽逐值检查回避这一差异。全部attention dropout=0，
没有BatchNorm，本项eval不是no_grad，不阻断文本/评分参数更新。
保存各臂fit及开发三内容（Empty/object/人工）GT逐条记录与子组、相对自身
Empty救回/损害、健康/transition保护；人工choice应改善自身Empty、优于
同预算generic且不增加健康损害，才能成为后续递归候选。未过不自动晋升。
这些不是正式P/R/F或EAO/ACC/ROB；全量九项仍须同一final实际评价。

这是最小新增推理评分职责，不宣称首次空参照、去均值、softmax/多正例。
M47跨帧partial matching多正例负结果保留，UVLTrack等候选竞争为明确近邻。
本项若有效，后续仍须自身状态递归、B几何/候选容量、C记忆/新观察各自
证明，不能仅一个小头宣称三个创新点。部署前fresh Astra/max源审查。
