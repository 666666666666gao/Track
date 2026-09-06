# M57 初始化实例绑定：接线准备，尚未选定底座或训练正式权重

本准备承接M56四个事实：独立属性版未优于空文本/池化对照；当前每个候选用自身视觉调制文本；NativeReferenceBank.initial在首次regular track才提取；M55两种底座仍在配对验证。只准备权重无关的模块与CPU工程contract，正式特征采集、训练和评测等待底座决定。

## 最小结构

在原生initialize创建模板和reference bank之后，使用同一t0 RGB-D图像及合法初始框构造搜索crop，调用raw network.forward，传入track_query_before=None。仅提取template index0的2×16×768 RoI并设为bank.initial；丢弃这次输出query。不能调用before_decision来初始化整个bank，否则会额外冻结dynamic/encoded_dynamic。

这一路仍是初始化帧的joint template/search表示，不称为隔绝背景或搜索交互的固有模板。TSG语义随最终底座，不能在未经过GPU验证时声称其token严格对应可解释部件。

四个候选头均使用相同t0初始RoI和默认动态模板参考，当前/上一候选数量仍为10+10。交互前的object20固定为t0参照；后续候选不能通过交互结果回写它。结构维持先文本读视觉、再对象局部格点读条件化文本。

| 预备对照 | read_visual的视觉K/V来源 | 文本输入 | 其余配置 |
| --- | --- | --- | --- |
| candidate/real | 各对象自己的局部格点 | 真实短语 | 相同槽数、mask、参数和初始化 |
| candidate/visual_queries | 各对象自己的局部格点 | 空文本embedding+可学习槽查询 | 同上 |
| initial/real | 固定t0参照object20 | 真实短语 | 同上 |
| initial/visual_queries | 固定t0参照object20 | 空文本embedding+可学习槽查询 | 同上 |

每组都含5个不同随机初始化、训练可学习的32维slot embeddings，四组共享同一初始化。视觉对照仍保留原始mask和槽数，不把5槽折叠为1槽；它保留文本的结构元信息，因此只能称为无词义内容的视觉查询对照。加入槽embedding后不再假定短语顺序完全无关，类别首槽与属性顺序来自固定manifest。参数为484547：原候选头448739、交互35648、共享槽查询160。

此2×2用于分别检验条件化来源与词义内容。四组使用相同t0参照可以排除“只有实验组额外获得初始化视觉”的解释；正式训练需同事件、初始化、顺序和预算。不能仅用initial/real超过原生就声称文本有效。当前不实现持续ID、模态属性可靠性、在线Qwen或身份/视角记忆，也不声称三个主创新完成。

## 当前工程contract

仅复用已绑定的chair01_indoor拟合缓存和M56 fit标签，22条真实事件循环组成32样本检查批。缓存内旧reference是t1来源，只检验张量接线及反向传播，不能充当新t0特征。每组3个临时优化步后丢弃全部模型，不保存checkpoint，不读取development标签。

核对四组初始张量/参数一致、零输出残差时与父头完全一致、padding不改变输出，以及可学习槽和有效查询的初始值确实不同。通过实际forward的guide_visual hook观察条件化短语：逐一对22个对象施加特征维度非均匀扰动；candidate组只能改变被扰动对象的条件化短语，initial组仅在object20改变时影响全部对象，其他对象改变时参照应保持。逐组记录optimizer执行前各交互子模块及两次注意力Q/K/V投影各自的数据梯度，避免把AdamW权重衰减或仅V分片的梯度当作查询学习证据。

原生STTrack构造及TSG前向含硬编码CUDA。CPU contract不执行真实t0提取，不能证明额外前向对bbox、模板、query、frame_id、网络状态与随机状态无影响。底座确定后，必须用真实初始化帧完成此GPU接线验证，并核对初始参照在后续默认模板写入后仍保持，才能采集正式特征或启动训练。

## 后续决定

先接收M55/M56已冻结完整递归结果，再冻结底座身份、t0提取协议及正式2×2训练方案。若采用不同底座，重新采集全部候选与参照特征；旧缓存只保留历史诊断用途。新增参数须在DepthTrack Train训练，再按完整开发递归、冻结低22及同一最终模型三数据集顺序验证。此准备没有新的跟踪指标、候选收益或晋升结论。
