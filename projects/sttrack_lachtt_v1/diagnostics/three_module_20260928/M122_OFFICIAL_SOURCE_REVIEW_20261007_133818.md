# M122 官方入口 SOURCE 审查

时间：2026-10-07T13:38:18.784277+08:00

**PASS；blocking 0，non-blocking 0。** 本次只放行当前 canonical labels 的一次 CPU 初始化绑定；不代表绑定已执行，不放行 GPU 编码或完整正式 suite，也不证明 M122 成绩。

审查身份：fresh delegated reviewer `/root/m122_official_source_review`，same-family / provisional。按安装策略请求 gpt-6-astra / max；实际后台模型与 effort 无独立 attestation，记为 UNATTESTED。

本次只读取本地源码、AST、归档元数据和人审文本；Torch、模型、神经执行、优化器、SSH、网络、GPU 查询、训练进程操作均为 0。没有修改实现代码，只保存本审查及 MANIFEST 条目。

- **confirmed_external_text_provenance — PASS_SOURCE_AND_STATIC_METADATA**：逐条核对当前 CSV、网站人审字段和 canonical labels：Test50、CDTB80、VOT1765 共1895条全部一致，第一槽为确认类别，之后最多四个确认稳定属性；未从 sequence 前缀、模型建议、uncertain/context/后帧备注生成短语。VOT88/backpack、1006/cup 与新 CSV b83f0b42…一致，外部类别全非空。Train152 CSV另行只复核确认/非空计数，四数据集合计2047。
  依据：`prepare_m122_external_human_labels.py:15`、`prepare_m122_external_human_labels.py:29`、`encode_m122_official_human_text.py:17`。

- **canonical_site_serialization_provenance — PASS_STATIC_METADATA**：原 labels 所记 VOT site SHA 34b8edae…精确对应当前网站 JSON 的 CRLF 字节版本；当前 LF SHA 056ed502…，不是人审内容变更。执行者已保留旧 labels 并刷新 canonical labels 的该来源摘要；本审查独立确认全部1895行数据对象完全相同，当前三份CSV/网站SHA均与 canonical labels 相等。当前 labels SHA 6a1d128d58f47a8b5ecb41153614c595bf2442a0808fc533c024b45e9c52520f。
  依据：`.aris/m122_full_causal_20261007/EXTERNAL_PROVENANCE_REFRESH.json`。

- **legal_initialization_key_and_membership — PASS_SOURCE_AND_STATIC_METADATA**：binder 锁定原 OPE inputs 和 VOT initialization plan 的实际 SHA。OPE 使用原始 xywh 和第1帧 JPEG；CDTB再匹配人审 review_id。VOT 使用已冻结 wire xywh，不重做坐标修正，键为 RGB SHA 原始32字节加 >4d 后的SHA。以归档元数据独立重算1765个VOT键全部一致，序列/帧身份和四shard anchor集合一致（867 forward、898 backward、127序列）。OPE50/76373及80/101956成员/预算一致。CDTB与VOT的实际 review_id 交集为0，不报告虚构的跨数据集冲突。真实RGB字节、CDTB实际key和1895全局唯一性由待运行的CPU binder断言验证，本审查未打开原图片。
  依据：`bind_m122_official_initializations.py:13`、`bind_m122_official_initializations.py:23`、`bind_m122_official_initializations.py:28`、`bind_m122_official_initializations.py:39`、`m122_official_runtime.py:39`。

- **CPU_binding_transport_boundary — PASS_SOURCE**：binder仅使用stdlib，打开标签/冻结初始化元数据以及合法初始化RGB，未打开后续GT或depth、未import Torch/CLIP。私有transport先要求本审查PASS且source/自身SHA完全一致，采用已知host key、stdin凭据和逐参数shlex.quote，只上传该CPU binder并逐字节核对远端源码/当前labels，执行一次绑定后回收私有产物与聚合回执；无NN、GPU/训练进度查询、后台job、自动重试或训练进程操作。此transport未由审查者运行。
  依据：`bind_m122_official_initializations.py:2`、`bind_m122_official_initializations.py:32`、`bind_external_inputs.py:8`、`bind_external_inputs.py:15`、`bind_external_inputs.py:19`。

- **frozen_text_encoding_and_bank_identity — PASS_SOURCE**：与Train采用同一指定ViT-L-14权重b8cca3fd…、float32/eval/no_grad、batch32、tokenize(truncate=False)；直接编码已绑定phrases，不翻译、不补词。现有1895行产生822种唯一短语（含空串）；编码器检查finite768维并生成(1895,5,768)和bool(1895,5)，类别mask全true。运行时核对bank格式、人审标记、1895唯一keys、tensor形状/finite以及binding和encoder权重SHA；plan还核对bank文件本身SHA。实际tokenization/编码与bank产物尚未执行，本结论不是bank运行验收。
  依据：`encode_m122_official_human_text.py:13`、`encode_m122_official_human_text.py:19`、`encode_m122_official_human_text.py:23`、`m122_official_runtime.py:29`、`prepare_m113_human_text.py:19`。

- **final_training_identity_gate — PASS_SOURCE**：checked_plan要求bundle SHA/schema、所列source字节及final/training_result/native_checkpoint/clip_weight SHA一致；training_result必须complete_M122_full152_training、659406 track_calls、456 sequence_runs、3 epochs、seed2027、冻结前后相等且final roundtrip相等，precision_weight与final SHA对应bundle。659406正确，不是656406。decoder用strict=True加载该final。最终bundle的实际来源完备性及两个臂各自跨三数据集同final仍需后续builder/scheduler审查，当前无bundle/final完成证明。
  依据：`m122_official_runtime.py:7`、`m122_official_runtime.py:15`、`m122_official_runtime.py:36`、`train_m122_full_causal.py:122`、`train_m122_full_causal.py:136`。

- **one_score_geometry_and_template_runtime — PASS_SOURCE**：OPE与TraX均调用OfficialDenseTracker→同一FullDenseTracker/DenseTargetDecoder。选中位置的最终double像素框、response分数、quality和feature使用同一selected_index；提交后的state/query推动下一crop。模板使用该同一位置未适配native Hann response，沿用50帧/.75阈值和128/2.0 crop，保留永久首模板；没有用新support×quality分数套旧模板阈值或另一个候选峰值。旧runtime/native相关字节均与已有M122审查一致。
  依据：`m122_official_runtime.py:24`、`run_m122_official.py:17`、`run_m122_official.py:63`、`full_dense_tracker.py:88`、`full_dense_tracker.py:104`、`dense_target_decoder.py:114`。

- **sequence_reset_and_GT_boundary — PASS_SOURCE**：每个合法初始化重置native template/state/frame_id/query及初始参照、固定短语；track路径no_grad，只输入当前RGB-D，不接收GT，不在线修改文字/优化器。OPE inference循环仅读取RGB/depth；GT仅在单独ope_analyze且完整prediction receipt存在、各结果SHA复核后打开。TraX只接收协议合法初始化框及后续图像，后续submit同runtime输出。人审使用多帧/视频辅助已在计划中如实披露；不能据此声称纯首帧自动caption协议。
  依据：`full_dense_tracker.py:75`、`full_dense_tracker.py:117`、`run_m122_official.py:21`、`run_m122_official.py:48`、`run_m122_official.py:67`。

- **OPE_sealing_and_locked_official_metric — PASS_SOURCE**：OPE预测验证完整帧数、框/分数finite和正尺寸，以六小数写框及_all_scores并复读检查舍入；全部序列结束后检查冻结digest并重新检查plan/bundle，再写完整SHA回执。分析先复核plan、final、bank、case顺序/帧数、结果SHA和GT SHA，随后按锁定metric源码05879f2e…调用100阈值宏平均P/R/F。已直接读取并校验该本地metric原始源码，调用签名和返回sequences/frames匹配；未运行metric、未重算任何M122结果。
  依据：`run_m122_official.py:28`、`run_m122_official.py:36`、`run_m122_official.py:43`、`run_m122_official.py:53`、`native_ope/source_snapshots/depthtrack_pr.py:63`。

- **TraX_bridge_and_remaining_suite_scope — PASS_SOURCE_WITH_EXPLICIT_PENDING_INTEGRATION**：bridge通过bundle['vot_bridge_path']显式spec_from_file_location加载，已核对canonical full152_paired_20260925/evaluation/interface/m39_vot_bridge.py SHA230acf10…；未错误地假定D目录中存在同名模块。wrapper接收RGB-D初始化wire rectangle并逐帧report bbox/confidence。当前entry本身不是VOT suite controller；四shard冻结划分、环境/timeout、完整5295结果文件合并、官方EAO/ACC/ROB分析和同final验收尚待准备/审查，因此不放行完整正式评测或宣称其已ready。
  依据：`run_m122_official.py:62`、`full152_paired_20260925/evaluation/interface/m39_vot_bridge.py:24`、`full152_paired_20260925/evaluation/interface/m39_vot_bridge.py:74`、`M122_OFFICIAL_EVALUATION_PLAN.md:13`。

- **source_import_closure_and_archived_native — PASS_SOURCE**：独立AST展开10个官方入口本地Python传递依赖；其top-level只定义类/函数与安全常量，不导入旧trainer或读取历史缓存。另绑定26个D源码/计划文件（24个Python3.8 AST通过），其中先前训练审查19个source/plan摘要均未变化；对于额外历史训练链仅复用原审查并验证字节，未冒称重新审查所有历史算法。private transport、2个外部Python源码、6个归档native Python也通过3.8 AST；7份native/YAML原字节与已有审查一致。Torch/CLIP/OpenCV/TraX/VOT包内部与实际远端解析环境未由此次执行验证。
  依据：`local_import_graph`、`source_sha256`、`native_source_sha256`、`external_source_sha256`。

- **claim_boundary — PASS_SOURCE**：计划明确当前仅准备输入/编码/统一runtime/OPE+TraX入口；未宣称已生成M122正式预测、metrics、full complete、三模块贡献或联合达标。完整目标仍需每个固定final分别完成Test50/CDTB80/VOT127-1765全部九项。此次SOURCE PASS只放行当前canonical labels的一次CPU绑定准备步骤，不提前执行encoder或占用当前训练卡。
  依据：`M122_OFFICIAL_EVALUATION_PLAN.md:3`、`M122_OFFICIAL_EVALUATION_PLAN.md:7`、`M122_OFFICIAL_EVALUATION_PLAN.md:13`。

后续仍需完成：

- 只检查本地源码、AST、原始归档元数据和当前人审CSV/网站文本；没有SSH、网络、原图读取、Torch导入、模型执行或GPU/训练状态查询。
- 真正1895个初始化RGB字节及全部key唯一性仍须CPU binder执行成功；其中1765个VOT key本次只从已保存image_sha256+wire bbox独立重算，不能称本次重新hash原图。
- 文本编码未运行，没有已生成/验收的官方bank；实际tokenization、模型加载、形状/finite检查和绑定SHA需由编码产物证实。
- 最终training_result/final/bundle及两个臂的OPE case计划尚未验收。后续bundle必须包含本次所审runtime闭包、native配置与canonical bridge/metric，不能把任意source map通过当作完整部署验证。
- VOT完整四shard调度、协议/超时配置、失败确认、5295文件合并及官方分析尚待实现/审查；单TraX入口PASS不代表完整suite ready。
- 当前远端路径、包版本/导入解析、TraX握手和运行时数值行为未在本次运行验证。无需为此加入无证据的fallback/兼容层。
- 已有M122训练链19个源码/计划仅确认与先前PASS审查字节相同并读取本次相关final/runtime代码；本次没有重新执行训练、重放历史sanity或查询正在运行的训练。
- 该审查具有新上下文、同模型家族的独立审查身份，结论为provisional；实际后台模型/effort没有独立证明，不能声称跨家族接受。

已保存字节身份：当前 labels `6a1d128d58f47a8b5ecb41153614c595bf2442a0808fc533c024b45e9c52520f`；私有 transport `8eb76d4705f766f9de2eceea989be458fc192624e1f8f2b183fc178991626b88`。完整26个D源码/计划、7个native源、canonical bridge/metric及静态证据摘要见同名JSON。

| 当前入口 | SHA256 |
|---|---|
| bind_m122_official_initializations.py | a617e1161b252125de5a0237248e6eacafc3f7ac19798602c58c571f485271a1 |
| encode_m122_official_human_text.py | 6f50da069c5a4a7e902a05dc1d3227032bb67a9952f9687ca4a95279fab3348f |
| m122_official_runtime.py | a1fb1e04cf2f5394a96ed1fb1da6142f56f6dfa6477ec047e34c5b9d0b5b3e7b |
| run_m122_official.py | d0eeacba480a19ccb717b36b7cf0d7d3fe3cb5395e7bf19d19b37868dd3354f9 |
| M122_OFFICIAL_EVALUATION_PLAN.md | 6515dc143fccbcff6b0466f941fbfc4d0ef293000552f27dfb248c508a858ba3 |

验证数量：10 个官方本地导入闭包模块；24 个 D Python、1 个私有 transport、2 个外部 Python、6 个归档 native Python 的 Python 3.8 AST 通过；旧19个训练source/plan与7个native字节均未变化。六份原始官方元数据及收取回执逐项SHA/长度一致。外部1895行、四数据集2047个确认非空类别、1765个VOT元数据key和anchor身份已独立复核；真实RGB原字节未由本次审查读取。
