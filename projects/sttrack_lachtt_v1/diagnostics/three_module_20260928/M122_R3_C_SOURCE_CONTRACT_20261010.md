# R3 C源码准备：同输入当前质量/未来写入收益

此文件是源码接口与待执行条件，不是运行回执。R2原双GPU教师仍由唯一控制器运行；不提前读取它的NN进度，不增加GPU任务。R3必须等完整R2回执、原始标签/状态结果审核、源码审核和实际接线检查后执行。

- 固定原计划：P1 final、人审bank、seed2027、C fit120/dev32、10epoch、AdamW 3e-5/0.01、batch32、固定final。
- C网络为519→128 ReLU→32 ReLU→1线性，70721参数；两臂结构、CPU初始化、数据公共资格与CPU排列生成器一致。current标签为当前框真实GT IoU，future标签为W/K真实未来32帧有效GT平均差；负/零事件不删除。本帧或未来GT未知不强赋零。
- `train_template_write_C.py`在构造模型前要求完整R2、fresh结果审核完成/0阻断/`R2_terminal_raw_audit_complete=true`，且实际aggregate和两个replay结果SHA分别等于审核的`audited_teacher_result_sha256`与`audited_teacher_shards_sha256[0/1]`。这些是后续实际审核必须提供的明确字段，目前没有对应完成态结果，不得伪造。
- 训练仅从真实事件PT取已绑定519维float32；不载入图像/模板到GPU，不构造或更新A+B。保存fit/dev公共数据清单与真实排除数、输入/顺序/初始化摘要、10轮MSE、final及重载回执。开发teacher-MSE不用于选best或优化。
- `TrustedTemplateTracker`保留原step的RGB-D输入、float64框映射、候选/分数/特征/query一致提交；新增C决策位于模板裁剪前。每次初始化重建固定首帧参照，本帧C输入只用当前/立即上一帧特征、首帧ROI及预测质量/运动；当前及未来GT不进入控制。C只在原50/.75同位置资格上预测，current>.5、future>0，拒绝时保持原模板；不修改当帧输出框、分数、查询或搜索因子。源码原FullDenseTracker及正在运行的R2三源不修改。
- 完整C-dev32预测器不打开groundtruth.txt；预测完成后独立CPU分析加载真实GT。指标只覆盖非初始化且GT有效帧；H10为有效GT且IoU≤.1连续至少10帧，未知GT断开；两臂和已保存原生前缀采用同口径，不与历史dev22或正式P/R/F混用。
- 开发分析器同样要求实际R2终态审核完成、0阻断及`R2_terminal_raw_audit_complete=true`；aggregate SHA须匹配`audited_teacher_result_sha256`，两个原生完整前缀须分别匹配该审核的`audited_teacher_prefixes_sha256[0/1]`，两臂训练回执中的teacher结果及审核SHA须与当前实际输入一致。后续终态审核必须从真实原前缀产生这些摘要；当前没有对应终态结果，不得伪造或用启动回执替代。
- 固定教师开发事件的有益/有害写入保留只在同一原教师状态上统计。不可将原教师事件的未来标签转移到C改变后的历史。完整C历史另报实际资格/写入总数、未知GT写入、低IoU写入与逐序列轨迹；其动作因果收益需新的W/K回放。R3的macro IoU/H10计算门仍须结果审核与覆盖检查，不能仅凭一项布尔推进。
- 推进条件保留：future macro IoU严格胜原规则与current，H10相对原规则不增加，报告全部写入覆盖与逐序列正负项；之后才允许R4自身历史/Full152 C。不是当前正式目标已满足或C有效的证据。

新源码只有准备状态；未构造新C、未前向、未训练、未部署。MLP参数量为源结构算术，不能当作参数更新/梯度/效果实测。下一个R3部署包必须绑定实际R2已审结果、C数据/初始化/预算和所有新旧执行来源；不接入在线MLLM、多视图库、跨区域搜索或数据集专属阈值。
