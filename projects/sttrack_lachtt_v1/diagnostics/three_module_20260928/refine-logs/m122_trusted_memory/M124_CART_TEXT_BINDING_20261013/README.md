# 推车案例：人工文本与保存评测输入的绑定核查

本次只读CPU检查真实exit0（native d2a795），未查询服务器、GPU或训练进度，未执行神经网络。

cart_room_occ_1的人工确认输入为 `cart`、`blue tubular rails`、`dark flat platform`、`dark wheels`。本地人工CSV、外部标签清单和官方初始化绑定的短语完全一致；绑定第87行（零基）、合法首帧框及999帧序列信息一致。由保存的首帧图像摘要和精确bbox重新计算的初始化key与人工review_id一致。

保存的CDTB80评测plan、初始化绑定文件和编码回执的SHA一致；编码器源SHA一致。已固定的CurrentC runtime源码要求继承父模型的同一bank/binding，并按RGB摘要与精确初始框选择行。`exact material`是不确定属性，未加入短语。CSV的status=`conflicting`描述旧提案冲突；human_confirmed=true且确认类别非空，不能解释成尚未审核或缺类别。

这是现有本地文字与绑定元数据检查，没有新读取远端bank张量或首帧RGB原始字节，没有重放候选；不构成文字实际增量、独立完整运行审计或首次跳框根因证明。首个跳框仍需区分候选不存在、评分未选中与几何精修。原CurrentC的跳框不可直接套给正在训练的M124权重。准确类别继续保留，候选NN诊断仍待整套M124训练、选定best完整VOT终态并释放GPU。

原03:16:11.250160小时检查节点与native68855/functions359等待不变。运行中44份锁定源码、loss、学习率、搜索、模板和checkpoint选择未修改。Goal ACTIVE_UNMET。
