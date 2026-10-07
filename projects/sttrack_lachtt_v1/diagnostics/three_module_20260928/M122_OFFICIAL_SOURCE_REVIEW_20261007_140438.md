# M122 官方绑定 dataset 命名空间 SOURCE 复审

时间：2026-10-07T14:04:38.524672+08:00

**PASS；blocking 0，non-blocking 0。** 本次只放行当前固定源码和 canonical labels 的一次修正后 CPU 初始化绑定。GPU 文本编码、final/bundle builder、OPE 正式运行与完整 VOT suite 仍未放行；本报告不是绑定完成回执或 M122 成绩。

审查身份：fresh delegated reviewer `/root/m122_binding_namespace_review`，same-family / provisional。安装策略请求 gpt-6-astra / max，实际 backend/model/effort 独立 attestation 不可用，记为 UNATTESTED。

本次没有 SSH、网络、Torch/CLIP 导入、模型加载/forward、优化器、GPU或训练进度查询；没有改动实现。只写本审查的时间戳/固定版本和 MANIFEST 条目。

原 13:38:18 PASS 只允许首次 CPU 绑定，并将全局键唯一性交给该真实步骤验证。随后实际诊断得到 1895 行 / 1859 原始观测键，原断言失败；旧报告和修复前源码保留。本次已按新源码重新审查，不能沿用旧 PASS。

本地独立复算：Test50 原始首帧 JPEG 与50份原始receipt一致；VOT1765由保存的RGB SHA与wire框重算；CDTB80使用canonical review_id。结果为36组Test↔VOT碰撞、1895个dataset唯一键。36组完整短语都不同，其中6组类别不同。原始观测键、框与各自人工文字必须保留。

- **observed_collision_and_local_reproduction — PASS_LOCAL_STDLIB**：独立读取50个Test原始首帧JPEG，逐一匹配原始收取receipt的SHA/大小/序列/框；以1765个VOT归档image_sha256+wire bbox重算键，CDTB80使用canonical review_id元数据。复得1895行、1859个raw observation keys、36组碰撞、1895个dataset keys。36组均恰为Test与VOT各1条；Test重新计算的RGB SHA与VOT保存SHA相等，>4d框字节相等。逐组核对私有诊断和公共摘要。该复算不冒称重新读取VOT/CDTB服务器原图。
  依据：`initialization_key_collision.json`、`M122_OFFICIAL_BINDING_COLLISION.json`、`bind_m122_official_initializations.py:49`。

- **minimal_dataset_namespace_repair — PASS_SOURCE**：与before_dataset_key_namespace及原133818审查逐字diff：binder仅在2处将bank key改为dataset+":"+原键并保存observation_key；原RGB SHA算法、>4d框、raw/wire坐标、review_id和整个人审row原样保留。runtime增加dataset合法值检查/保存，并用相同前缀查找。未增加alias、跨数据集替换、fallback、重新标注或GT输入。旧23个其余source/plan、全部19个训练source/plan字节相同。
  依据：`bind_m122_official_initializations.py:35`、`bind_m122_official_initializations.py:37`、`bind_m122_official_initializations.py:47`、`m122_official_runtime.py:35`、`m122_official_runtime.py:41`。

- **collision_label_and_runtime_index_preservation — PASS_SOURCE_AND_LOCAL_METADATA_LOOKUP**：36组碰撞的完整phrases全部不同，其中6组类别不同；不能合并或以一条替代另一条。encoder仍按binding rows顺序写tokens/mask/keys/datasets，1895个新key唯一。抽取当前源码的initialization_key和initialize两个纯函数，在已保存Test RGB摘要/真实框和记录sink上执行72次碰撞lookup，全部命中各自dataset行及原phrases；没有加载runtime类、Torch或模型。该检查是命名空间元数据验证，不是神经推理验收。
  依据：`encode_m122_official_human_text.py:23`、`encode_m122_official_human_text.py:27`、`m122_official_runtime.py:36`、`m122_official_runtime.py:40`。

- **confirmed_external_text_provenance — PASS_SOURCE_AND_STATIC_METADATA**：重新逐行核对1895个canonical labels与三份最终CSV/网站人审字段；只编码confirmed_category及最多前四stable_attributes，顺序/omitted属性/人审标记/序列帧身份均一致。当前labels SHA6a1d128d…与旧审查字节相同；Train152另复核确认非空，总2047。共822种短语含空串。VOT88=backpack、1006=cup的用户更正来源仍是b83f0b42…CSV；不使用模型建议、名称前缀补词、uncertain/context或后帧备注。
  依据：`prepare_m122_external_human_labels.py:15`、`prepare_m122_external_human_labels.py:29`、`.aris/m122_full_causal_20261007/external_human_labels.json`。

- **legal_initialization_and_protocol_membership — PASS_SOURCE_AND_STATIC_METADATA**：OPE输入SHA61541e35…及VOT plan SHA b4c23250…仍锁定。Test50/76373与CDTB80/101956成员/预算正确；OPE首帧raw xywh保持原值，CDTB仍要求原键等review_id。VOT仍验证实际RGB SHA、wire框键、序列/帧身份；1765个anchor与冻结四shard集合完全一致，127序列、867 forward/898 backward。dataset命名空间仅管理各自文字输入身份，不改变协议框或跟踪算法。
  依据：`bind_m122_official_initializations.py:23`、`bind_m122_official_initializations.py:28`、`bind_m122_official_initializations.py:39`。

- **CPU_binding_transport_boundary — PASS_SOURCE**：private transport与旧版本完全相同，执行前要求当前review PASS/无blocking、全26个D source摘要和自身摘要相同；采用已有known_hosts/default RejectPolicy、stdin密码、固定43811和逐参数shlex.quote。只上传并复核CPU binder/当前labels、一次前台运行及回收binding/聚合receipt。binder stdlib-only且仅打开合法初始化RGB及静态元数据，无后续GT/depth、NN/GPU/训练进度查询、后台job、自动重试或训练操作。本审查不执行transport；当前修正仍需真实一次CPU回执。
  依据：`.aris/m122_full_causal_20261007/bind_external_inputs.py:8`、`.aris/m122_full_causal_20261007/bind_external_inputs.py:15`、`bind_m122_official_initializations.py:2`。

- **frozen_text_encoding — PASS_SOURCE_GPU_NOT_RELEASED**：encoder字节未改：同Train指定ViT-L-14 b8cca3fd…、float32/eval/no_grad、batch32、tokenize(truncate=False)，直接编码已绑定phrases，输出1895×5×768及bool mask。新key随同序列写入bank；runtime核对格式、人审标记、1895唯一key、形状/finite、binding及encoder SHA，checked_plan另验bank文件SHA。实际tokenization/GPU编码/bank仍未运行，不在本次放行范围。
  依据：`encode_m122_official_human_text.py:13`、`encode_m122_official_human_text.py:19`、`prepare_m113_human_text.py:19`、`m122_official_runtime.py:29`。

- **final_and_training_identity — PASS_SOURCE_FINAL_BUILDER_NOT_RELEASED**：checked_plan继续核对bundle/source/final/training_result/native/CLIP/bank/binding摘要和固定schema；训练完成条件仍为complete_M122_full152_training、659406调用、456序列、3pass、seed2027、冻结exact及final roundtrip exact，precision_weight/final与bundle相等；strict=True加载同final。未更改当前训练源码/预算/选择规则。实际final builder、完整source/native/bridge/metric清单和两臂各自三数据集同final尚待实现及后续审查。
  依据：`m122_official_runtime.py:7`、`m122_official_runtime.py:15`、`m122_official_runtime.py:37`、`train_m122_full_causal.py:122`。

- **same_selected_state_history_and_GT_boundary — PASS_SOURCE**：重新读取统一runtime和本地10模块闭包。OPE/TraX均使用OfficialDenseTracker→FullDenseTracker/DenseTargetDecoder；同selected_index提交double像素框/score/quality/feature，之后crop/query/history继续自身状态。每个合法初始化重置状态、永久首模板与固定短语；同位置native Hann response触发原50帧/.75模板规则，未把新支撑×质量分数代入旧门槛。track no_grad，仅当前RGB-D，不传GT、文字在线更新或优化器。人审多帧辅助协议在计划中披露。
  依据：`full_dense_tracker.py:75`、`full_dense_tracker.py:88`、`full_dense_tracker.py:104`、`dense_target_decoder.py:114`、`run_m122_official.py:21`。

- **OPE_sealing_and_official_metric — PASS_SOURCE_FULL_EVALUATION_NOT_RELEASED**：OPE完整帧数、finite/正尺寸、六小数写回及冻结digest检查不变；所有序列预测封存并有完整SHA receipt后，ope_analyze才复核结果/GT SHA并读取正式GT。直接读取锁定05879f2e…metric，调用100阈值宏平均P/R/F签名与返回帧数匹配。namespace不改变输出/置信度/阈值/评价集合。未执行metric或产生正式结果。
  依据：`run_m122_official.py:28`、`run_m122_official.py:36`、`run_m122_official.py:43`、`run_m122_official.py:53`、`native_ope/source_snapshots/depthtrack_pr.py:63`。

- **TraX_and_suite_boundary — PASS_SOURCE_VOT_SUITE_NOT_RELEASED**：直接读取并校验canonical bridge 230acf10…；runtime由bundle明确路径加载RGB-D TraX bridge，使用合法wire rectangle和同runtime后续bbox/confidence。plan.dataset必须由未来VOT计划固定为vot。当前仅入口，不含完整四shard调度/环境及timeout/失败确认/5295文件合并/官方EAO-ACC-ROB分析；全部仍未放行。
  依据：`run_m122_official.py:62`、`full152_paired_20260925/evaluation/interface/m39_vot_bridge.py:24`、`M122_OFFICIAL_EVALUATION_PLAN.md`。

- **source_closure_and_claim_scope — PASS_SOURCE_ONLY**：独立AST重建10个D模块闭包，24个D Python、1私有transport、1碰撞诊断器、2外部Python、6归档native Python通过Python3.8 AST；7 native/YAML摘要与旧审查相同。旧训练链其余算法按原审查复用并核对字节，不冒称重新运行或全链语义重审。当前仅放行一次corrected CPU binding；原PASS历史保留，本报告不证明GPU bank、full训练完成、正式九项、三模块贡献或联合达标。
  依据：`source_sha256`、`local_import_graph`、`native_source_sha256`、`external_source_sha256`。

后续边界：

- 仅源码、本地stdlib元数据/原始首帧SHA审查。SSH、网络、Torch/CLIP导入、模型加载/forward、优化器、GPU或训练进度查询均为0。未执行真实CPU binder或任何remote transport。
- 本次真正重新hash的初始化原图为本地Test50首帧；VOT1765仅独立重算已保存RGB SHA与wire bbox，CDTB80使用canonical review_id。当前服务器全部1895原图及修正binder完整回执仍须一次CPU执行确认。
- 72次lookup为从当前源码抽取的纯初始化键查找，使用已保存摘要和记录sink；不是Torch bank加载、真实tracker初始化或神经数值验证。
- GPU文本编码和产物未验收；未证明实际tokenization、远端依赖/路径解析或TraX握手。
- 完整final/bundle builder、精确source/native/bridge/metric清单及两臂三数据集同final固定计划尚待后续实现和审查。
- VOT完整四shard调度、协议/timeout、失败确认、1765 anchor/5295文件合并与官方EAO/ACC/ROB分析仍未放行。OPE完整正式运行同样未放行。
- 该审查为fresh context、same-family、provisional。策略请求gpt-6-astra/max，实际backend/model/effort独立attestation不可用；不能声称跨家族接受。

当前发布门禁字节：

| 文件 | SHA256 |
|---|---|
| bind_m122_official_initializations.py | 4023d263cde4e10436f2f2dc78247d2b8875df5316352935ea85b2a1ad71f69a |
| m122_official_runtime.py | 25aa67000cf581c454b160c524c8dcf8f55ec30d227b387990cc03ed18be3157 |
| encode_m122_official_human_text.py | 6f50da069c5a4a7e902a05dc1d3227032bb67a9952f9687ca4a95279fab3348f |
| run_m122_official.py | d0eeacba480a19ccb717b36b7cf0d7d3fe3cb5395e7bf19d19b37868dd3354f9 |
| M122_OFFICIAL_EVALUATION_PLAN.md | 3f5b417f863b1d9609e139ca3336127d4429d7868572b3d2d9954572be973676 |
| private bind_external_inputs.py | 8eb76d4705f766f9de2eceea989be458fc192624e1f8f2b183fc178991626b88 |
| canonical external_human_labels.json | 6a1d128d58f47a8b5ecb41153614c595bf2442a0808fc533c024b45e9c52520f |

完整26个D源码/计划、7份native/YAML、canonical bridge/metric、历史失败证据、50个Test首帧与其receipt摘要见同名JSON。24个D Python、private transport/诊断器、2个外部Python、6个归档native Python的Python3.8 AST通过。10模块闭包相同，19个训练源码/计划字节未变。
