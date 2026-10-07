# M122 完整评测源码审查

**PASS — 0 个未解决阻塞。** 一处实际OPE阶段顺序问题已由主代理作最小修复并重读；当前只放行摘要所列字节的一次被动排队部署。

审查时间：2026-10-07T14:17:59.532618+08:00。Fresh context、same-family、provisional；策略请求gpt-6-astra/max，实际后端/model/effort独立证明不可用。

本审查没有SSH、网络、Torch/CLIP导入、模型加载或forward、优化器、GPU查询、安装、实验启动，也没有编辑实现。仅标准库读取源码、AST、归档数据及回执；报告和私有检查证据是本代理唯一写入。

- **original_training_release_and_two_fixed_finals — PASS_SOURCE**：原24539控制器先真实写driver.exit=0及complete pair；builder要求原四次sanity/train子任务exit0，两份实际training result与pair逐对象相同，每臂3pass/659406 calls/456 sequences/seed2027、冻结参数及buffers、final回读、无后帧GT重初始化、无外部优化/测试best；strict load同一final、380167参数。两臂共享warm及initial state，final各自固定并跨三个数据集复用。 位置：`run_m122_evaluation_suite.py:8`, `prepare_m122_evaluation_suite.py:31`, `prepare_m122_evaluation_suite.py:62`, `run_m122_full_causal.py:47`。

- **confirmed_1895_binding_namespace — PASS_LOCAL_METADATA_AND_SOURCE**：已存在修正后CPU binding，实际文件SHA3bf7aa31…；本审查独立重算所有1895个保存RGB摘要+>4d框键，并逐字段比较canonical labels。50/80/1765、1859 raw keys、36 Test/VOT碰撞、1895 dataset keys；36组短语全不同，其中6组类别不同，dataset前缀保留各自记录，没有跨数据集替代、翻译、补词或改框。此处不冒称重新读取远端1895原图。 位置：`bind_m122_official_initializations.py:28`, `m122_official_runtime.py:40`, `official_initialization_binding.json`。

- **same_frozen_text_encoding — PASS_SOURCE_WITH_EXISTING_CPU_RECEIPT**：encoder沿Train冻结CLIP SHA b8cca3fd…，float32/eval/no_grad、batch32、五槽及前四稳定属性，truncate=False；bank rows/keys/datasets/binding SHA、1895x5x768、bool mask、finite均受检查。读取主代理已完成CPU tokenizer回执：822 unique、最长47/77、无超限；本审查没有执行CLIP或编码，GPU bank仍待未来生成。 位置：`encode_m122_official_human_text.py:13`, `prepare_m122_evaluation_suite.py:36`, `prepare_m113_human_text.py:19`, `M122_EXTERNAL_TOKEN_LENGTH_CHECK.json`。

- **uniform_own_history_runtime — PASS_SOURCE**：OPE/TraX均走OfficialDenseTracker/FullDenseTracker/DenseTargetDecoder，固定首帧视觉参照与同bank文字，当前位置和query/template只沿自身历史；同selected cell提交bbox/response/quality/feature。模板仍按同位置native Hann response执行50/.75，未用适配score套原阈值。track是no_grad；无外部优化、在线文字、后帧GT或数据集专属gate。重读运行10模块闭包，旧训练19文件字节未改。 位置：`m122_official_runtime.py:24`, `full_dense_tracker.py:75`, `full_dense_tracker.py:88`, `dense_target_decoder.py:91`, `run_m122_official.py:12`, `run_m122_official.py:62`。

- **all_OPE_predictions_before_metric_GT — PASS_SOURCE_AFTER_MINIMAL_FIX**：发现并反馈原dataset外层循环会先计算Test指标再预测CDTB；主代理仅交换两层循环，当前为Test track→CDTB track→Test analyze→CDTB analyze，每phase两臂同时运行且全部exit0才继续。因此两臂全部260条预测先封存，再打开OPE GT并使用锁定05879f2e…源码、100阈值、原macro P/R/F。各臂50/76373及80/101956、六小数回读、结果SHA、GT SHA和冻结digest/0 optimizer均闭合。 位置：`run_m122_evaluation_suite.py:44`, `run_m122_official.py:21`, `run_m122_official.py:36`, `run_m122_official.py:42`, `depthtrack_pr.py:63`。

- **VOT_frozen_partition_and_metadata — PASS_SOURCE_AND_LOCAL_METADATA**：FROZEN原manifest SHA8e76256f…、VOT metadata SHA1633e2f…、每个原master anchor SHA均核对。四分片441/441/441/442，成员恰为1765合法初始化、127序列、867 forward/898 backward。原creator把GT/tags/图像作指向原数据的symlink，仅anchor生成过滤覆盖；新copytree保留symlink、逐文件核对6938份metadata，四分片GT/tag一致，完整master重新指回原数据并保留原anchor。没有旧预测复用。 位置：`prepare_m122_evaluation_suite.py:46`, `prepare_m122_evaluation_suite.py:95`, `prepare_m122_evaluation_suite.py:104`, `official_analysis_sources/metadata_sha256.json`, `create_vot_failure_family_shards.py:184`。

- **two_GPU_VOT_waves_and_exact_merge — PASS_SOURCE**：每臂按[0,1]、[2,3]两个wave，GPU[0,1,0,1]，两个进程且每300秒检查；前wave exit0及各anchor数量完整才运行后wave。原600s TraX timeout/restart=false，输出目录全新。1765个expected names对应唯一5295个.bin/confidence/time文件，合并禁止同名覆盖并逐文件SHA相等；完整结果回执后才官方analysis。 位置：`run_m122_vot_shards.py:22`, `run_m122_vot_shards.py:30`, `run_m122_vot_shards.py:50`。

- **official_VOT_analysis_and_postseal_failures — PASS_SOURCE**：原mplt执行vot analysis，命名与master/analysis路径吻合；[0][0][0] EAO、[2][0][0] ACC、[2][0][1] ROB与锁定common parse_analysis及M67/M82已完成入口相同。failure e96a…与dependency cbbe…字节及绝对路径被纳入gate，正式分析后再调用原collect_confirmed_failure_outcomes(expected_anchors=1765)，每条实际轨迹长度须等于原forward/backward proxy，返回1765 outcomes/127 sequences；失败数未冒充EAO/ACC/ROB。 位置：`run_m122_evaluation_suite.py:50`, `analyze_m122_vot.py:12`, `analyze_m122_vot.py:22`, `finalize_vot_full127.py:373`, `finalize_vot_transaction_low22.py:193`, `full152_paired_20260925/evaluation/analyze_full.py:16`。

- **mplt_import_boundary_and_Python38 — PASS_LOCAL_STDLIB_SOURCE**：29个D Python、6个归档native Python、4个外部Python、private deployer及1段嵌入remote脚本全部Python3.8 AST通过。analyzer顶层实际导入闭包仅analyze/bind/prepare；torch与DenseTargetDecoder在prepare函数内。本地实际以非main方式导入analyzer成功，无Torch/CLIP/NumPy/OpenCV/VOT导入；failure/common顶层为stdlib。真实Linux mplt环境与VOT依赖仍待实际运行证明。 位置：`analyze_m122_vot.py:1`, `prepare_m122_evaluation_suite.py:27`, `local_import_graph`, `verification`。

- **six_results_nine_metrics_per_final — PASS_SOURCE**：每个model逐数据集核对final/bundle/plan/receipt/bank与原结果文件SHA；六份完整结果才写all_results和两行metrics.csv。每行九项门槛独立判定：DepthTrack>=65.2/64.9/65.1、CDTB>=72.9/75.6/74.2、VOT严格>77.9/82.1/93.7，同一final九项全过才joint_pass；没有两final拼分或测试best选择。independent_completed_audit仍明确false。 位置：`collect_m122_official_results.py:10`, `collect_m122_official_results.py:28`, `collect_m122_official_results.py:39`。

- **queued_transport_no_running_source_overwrite — PASS_SOURCE**：部署器当前review PASS/无blocking及32 D source、自身SHA、actual binding一致才可运行；只上传新增11个评测源/计划，原训练19链仅读/验，不写。源门禁包括19个绝对外部路径及当前binding，queue启动先校验源再被动sleep3600，此前不读取训练进度/GPU。之后每小时只看driver.exit并kill(pid,0)，原失败不会重启；只有实际complete之后才查两卡空闲、编码及评测。没有安装、优化器或训练重启。 位置：`deploy_complete_evaluation_queue.py:8`, `deploy_complete_evaluation_queue.py:19`, `deploy_complete_evaluation_queue.py:31`, `run_m122_evaluation_suite.py:8`。

**摘要范围**：旧26个D源码/计划加新增5个Python和完整计划，共32；29个D Python通过3.8 AST。19个绝对外部路径包含native7、6份原始正式输入元数据、OPE metric、TraX bridge、失败统计及依赖、VOT metadata和config。部署器自身摘要单列。完整摘要、10模块运行闭包和3模块分析器顶层闭包见同名JSON。

**实际输入边界**：已存在CPU binding `3bf7aa31a3f819ec746b37dc221e55c531a1235726cda3efedc4990c607cb26e`；本地逐字段和摘要核对通过，822短语最长47/77 token的实际CPU回执亦匹配。GPU bank尚未生成，两个训练final、正式bundle和六组正式结果均未由本审查核实完成。

**结论限制**：
- fresh delegated context、same-family、provisional；策略请求gpt-6-astra/max，实际后端/model/effort无独立attestation，不能称跨家族接受。
- 本审查仅本地源码、stdlib AST/import、已归档元数据和回执检查：SSH、网络、Torch/CLIP导入、模型加载/forward、优化器、GPU查询及实验启动均为0。
- CPU binding和tokenization已存在实际回执且本地摘要/字段核对通过；本审查没有重新读取远端1895图像，也没有自行运行真实CPU binder/tokenizer。
- 源码PASS仅放行这些字节的一次被动排队部署及既定运行门禁；不证明原Full152训练已结束、GPU编码成功、bank/bundle/final存在或GPU初始化/TraX握手/完整正式指标通过。
- Python3.8 AST与Windows stdlib导入不等于Linux sttrack/mplt包和GPU的实际兼容性；真实依赖、显存、速度、数值和环境仍由未来原始运行回执验收。
- 直接D运行闭包、指定native7及原bridge/metric/failure源码已查；未声称重审全部第三方Torch/CLIP/OpenCV/VOT或native backbone的内部实现。
- 没有源码层三模块有效性、C模块贡献、九项达标或SOTA成功结论；正式结果仍需实际产物及独立完成审查。

私有标准库证据：`.aris/m122_full_causal_20261007/complete_evaluation_reviewer_source_checks.json`，SHA `079bb5cd02241b40481363d4a854d0b01e60ea60fba0a1d8bb4afe9e5c6caa3d`。
