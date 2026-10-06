# M113：人工核验初始化文本的固定状态双卡训练

2026-10-06用户确认已完成全部人审，并明确今后统一采用人工核验文本。
当前文件为Train152、Test50、CDTB80、VOT1765，共2047条human_confirmed=true。
只将Train152用于新增训练输入；其余用于后续统一评价的初始化输入核查。
序列名前缀只作核查线索，实际输入使用人工CSV的confirmed_category，
不把文件名前缀自动替代人工结论，不再运行Qwen生成类别。

继承M101视觉父模型、M90/M95/M98真实状态缓存、M110的语义参数冻结边界
及定位BCE+可靠原生候选保持目标。初始类别放slot0，人工stable_attributes按
原管道符切分，依原顺序最多取4条；其余保存为omitted_confirmed_attributes。
uncertain_attributes、note和模型后帧备注不编码。所有初始化输入均为已确认
Train文本。原130fit/22development划分保持，只从既有manifest读取划分，
不采用其自动描述或候选弱标签。人工审阅曾参考Train后帧，属于额外离线
监督来源；不宣称这些类别来源于自动首帧生成，不用于零监督比较。

M112旧准备使用202项模型当前候选短语标签，与此次初始化人审不是同一
标签任务。M112保留未执行，不将其human_confirmed改成true，不启动旧计划。
M113不训练当前候选support/conflict/unknown CE，也不将初始化人工确认
当作当前候选物理身份标签。

GPU0训练human_text，GPU1训练generic（object，各有效槽及mask一致）。两臂
均从同一M101父模型开始，语义读出共同清零，视觉参数及全部buffer固定；
仅更新同一95,683参数，seed2027、AdamW3e-4、batch64、12轮/480次更新，
固定最终checkpoint，不扫描seed或选择最佳开发checkpoint。先编码同一
冻结CLIP ViT-L-14文本bank，再执行两卡各3次更新sanity；两项实际exit0、
梯度有限且非零、冻结/Empty逐值保持之后才启动两卡完整训练。

报告2544fit/495development固定状态的GT定位结果；每个final分别测Empty、
generic、human_text，记录每条选择及all/healthy/transition三组相对自身
Empty的IoU差；rescue计≤0.1到≥0.5，harm计≥0.5到≤0.1，并报告两条件
各自IoU≥0.5命中数。核对全部3039状态的Empty分数与质量、冻结参数
和buffer均等于父模型，并验证保存/重载。此处development是固定训练内
划分与人工初始化协议下的开发证据，不是三个外部基准的正式成绩。

该对照首先回答人工语义输入是否具有实质增量，仍须后续完整递归、可信
状态推进及同一final的DepthTrack/CDTB/VOT验证才能达成原九项指标目标。
没有递归结果时不宣称三数据集达标，不把本轮固定状态成绩代填P/R/F或
EAO/ACC/ROB。服务器清理只退役已确认不用的权重，保留CLIP、视觉底座、
M101/M111及缓存/最终参照。预计编码、sanity及双卡训练合计3–6分钟，
依据既有M108/M110量级估计，尚非M113实测；180–300秒检查一次。
