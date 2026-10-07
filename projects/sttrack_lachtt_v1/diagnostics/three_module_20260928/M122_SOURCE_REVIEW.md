# M122 source review

**PASS - source-only; same-family / provisional.**

当前审查范围内未发现需要修复后才能运行短 sanity 的源码缺陷。该结论仅放行现有真实 sanity 门禁；不证明 M122 运行成功、完整训练或九项正式结果。

Blocking: none. Non-blocking source defects: none.

Requested by installed policy: gpt-6-astra / max. Actual backend/model/effort: independently UNATTESTED.

Reviewed at: 2026-10-07T12:29:52.108949+08:00

18 local Python modules in the direct/transitive import closure, the M122 plan, private deployer, and 7 archived native source/config files. Python 3.8 AST: 18 + 1 + 2 embedded blocks passed. No SSH, Torch import, model execution, GPU query, installation, experiment launch, or implementation edit.

## Findings

- **native_pixel_mapping - PASS_SOURCE**: 所有 256 个 cell 调用原生 cal_bbox；*256/resize 保留在原生 dtype，随后转 double，按原生 Python map_box_back 的中心/half_side 顺序计算。clip 对应 margin=10。不会把现有 M89 浮点分叉风险用缓存 float32 替换掉。运行逐值等价由零输出 64 帧 sanity 验证，源码检查不代替它。
  Source: full_dense_tracker.py:42, dense_target_decoder.py:9, native sttrack.py:120, native sttrack.py:191, native head.py:142, native box_ops.py:97.

- **submitted_box_and_loss - PASS_SOURCE**: raw_boxes-data.boxes 的 learned delta 加到 double native pixel base 后统一 clip；out.boxes 与 selected_box 被替换为该提交框，precision_objective 将这组提交框转 float 用于原 objective 和几何保护损失。box、response、quality、feature 均用同 selected_index；状态提交后 detach。
  Source: full_dense_tracker.py:88, dense_target_decoder.py:104, train_m122_full_causal.py:25, train_m121_dense_target.py:39.

- **own_history_and_GT_boundary - PASS_SOURCE**: 序列首帧 legal first_box 初始化；以后 action 使用自有 state 取 crop、上次 native query、当前动态模板，不接收 current GT。actor.step 在 truth[frame] 有效性判断及 loss 之前；无效 GT 跳 loss 但仍 action、推进 query/state/template 和帧计数。没有后续 GT 重置或跨离散 crop 的时间反传。
  Source: full_dense_tracker.py:53, full_dense_tracker.py:75, full_dense_tracker.py:101, train_m122_full_causal.py:81.

- **same_cell_native_template_rule - PASS_SOURCE**: 写模板使用最终 selected cell 的未适配 native Hann response，沿用 interval50/threshold0.75 和 128/2.0 模板 crop；不把 support×quality 新 score 套旧阈值，也不读取另一候选峰值。仅保持接口基线，未声称 C 贡献。
  Source: full_dense_tracker.py:104, native sttrack.py:128, deep_rgbd_256_lachtt_v1.yaml:67.

- **real_zero64_then_warm64_then_fresh_train - PASS_SOURCE**: sanity 先用 zero decoder 在首序列真实推进 1..64（经过 frame50），对每帧 native bbox/score 逐值检查；随后恢复同一 warm model 做真实 64 帧、2 次更新。两臂均 exit0、完成状态和参数组检查后，driver 才启动两个新的 train 进程，从 warm final 重新加载；不沿用 sanity 权重。
  Source: train_m122_full_causal.py:62, train_m122_full_causal.py:132, run_m122_full_causal.py:27, run_m122_full_causal.py:47.

- **freeze_core_clip_parameters_buffers - PASS_SOURCE**: native 与 CLIP .eval().requires_grad_(False)，特征生成处 no_grad；decoder.train() 不切换它们。前后 state_dict digest 覆盖参数及 buffer，另检查 frozen model 无梯度和 decoder buffer 逐值相同。运行检查尚未执行。
  Source: full_dense_tracker.py:14, full_dense_tracker.py:31, full_dense_tracker.py:55, train_m122_full_causal.py:59, train_m122_full_causal.py:117.

- **accumulation_and_actual_update_gate - PASS_SOURCE**: 每 32 实际 action 或序列末块更新；每有效帧 loss/32，更新前乘 32/有效梯度帧数，得到有效帧均值。全无有效帧块不 step；无效帧仍占时序位置。clip_grad_norm1.0、AdamW3e-5/.01，与计划一致；sanity 检查 2 updates、15 组非零末次梯度、15 组实际参数值变化。
  Source: train_m122_full_causal.py:86, train_m122_full_causal.py:99, train_m122_full_causal.py:120.

- **precision_loss_semantics - PASS_SOURCE**: 几何保护以 native IoU≥.5 的全部位置为资格，优化 relu(native.detach()-refined)，允许提升且不因学生已降到.5以下而关闭；权重0/1之外两臂算法与预算相同。继承 focal +2 GIoU +5 normalized L1 +quality BCE +intersection BCE +response preservation；GT作为监督而非模型输入。
  Source: train_m122_full_causal.py:25, train_m121_dense_target.py:39.

- **human_text_and_matched_budget - PASS_SOURCE**: 两臂同 seed2027、同 human-confirmed labels/bank、同 M121 human final 指定 SHA、同 native/CLIP 权重与同 spec 路径。完整训练固定 3 次 Train152 顺序；源码断言每臂659406 track calls、456 sequence runs。两臂 initial_state、bank、warm_final相同门禁存在；无多 seed、测试集优化或 best checkpoint 选择。原22开发序列进入优化后的数据只称训练内诊断。
  Source: train_m122_full_causal.py:41, run_m122_full_causal.py:8, M122_FULL_CAUSAL_PLAN.md.

- **import_closure - PASS_SOURCE**: 通过 AST 展开本地直接/传递 import 闭包。M121→M113→M110→M107、visual trainer 和 collector 等历史模块只在 import 时定义函数/类与常量；旧 main 均受 __main__ 保护，不会触发旧缓存数据加载、训练、评估或额外 NN。M122 实际调用 objective/box_overlap/observation_targets、dense reader 和原生 runtime。
  Source: 18 D Python files listed in source_sha256.

- **deployment_and_sanity_gate - PASS_SOURCE**: 部署入口先核对本审查与已存在源码绑定、原生7文件；读取 M121 原始完成状态及全部6子任务exit0，确认输出不存在、两卡idle和剩余磁盘，再传输并一次启动控制器。仅 CUDA_VISIBLE_DEVICES0/1；没有自动重试、安装、下载、功率/温度查询或失败后无条件放行。此审查没有执行部署。
  Source: deploy_m122.py:11, deploy_m122.py:36, deploy_m122.py:60, run_m122_full_causal.py:31.

- **claim_and_evaluation_boundary - PASS_SOURCE**: 源文件明确 no_public_evaluation_yet。M122只是完整训练候选；两组final后续仍须各自完成同权重 Test50/CDTB80/VOT127-1765 九项。当前源码不包括正式评估完成证明，也不证明三模块有效、C贡献或联合目标达成。
  Source: M122_FULL_CAUSAL_PLAN.md, train_m122_full_causal.py:131, run_m122_full_causal.py:48.

## Limits and remaining runtime evidence

- 本审查只使用本地源码、AST 和归档原生源字节；未独立连接远端核实运行环境、当前进程或模型后端。
- 逐帧 bbox/score 相等、frame50 模板行为、两次更新、冻结 digest、显存/吞吐与完整三pass需由真实 sanity 和训练报告证实。未执行模型重放。
- 已核对提供的原生 tracker/model/head/crop/clip/depth/YAML 源；未声称复审全部第三方 Torch/CLIP/OpenCV 或原生 backbone/Mamba 包内部。
- training_spec 与 inference_inputs 的实际数据内容未由此审查读取；源码预算、索引和门禁已核对。启动前/运行期断言及原始 artifacts 仍是实际输入证据。
- M121 的历史数字在此仅作为计划中已披露的先前结果引用；未在本次重算、也未称为 M122 成果。
- latest.pt 保存进度和优化器状态；当前入口按协议 fresh 启动且没有恢复执行分支，不应把保存 checkpoint 说成已验证自动恢复。

## Current source binding

The JSON source_sha256 binds the 18 actually read D Python modules plus M122_FULL_CAUSAL_PLAN.md; the private deployment entry is separately bound by private_transport_sha256. The append-only MANIFEST index is not executable source.

| Source | SHA256 |
|---|---|
| M122_FULL_CAUSAL_PLAN.md | 8f7a300dc6e37cd3ee3f9989593daaa516b1e7a1e8ef7255a125df67cdb82943 |
| analyze_train_states.py | 2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def |
| collect_ab_interface_panel.py | 361559ef53623dc53665534243a0b4cfb078821dc852e20b34a75d14dca45086 |
| collect_train_states.py | 3961c894e42411bb0cbb2eb245e0162b8765a742b08d5cd72ee9d8dfaa6618f3 |
| dense_region_decoder.py | 66b9317529f741aba818aa9e9feba7fcebdb54c285f0bd9184f98f4b1f202e7b |
| dense_region_encoding.py | 71715a497147eb917e6a93ba6916ebbe01bc4a859cb275401289df2706627f33 |
| dense_target_decoder.py | 055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2 |
| full_dense_tracker.py | 0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778 |
| instance_ab_prototype.py | 4485a7ca29a90fa13cc868c32e4e4083c351cfba7b318d341c0165c32a886e4f |
| m121_dense_inputs.py | 89651f43d40f1b13e16ab586d6c0aaa19e3edfcf91f59d3286f74cf093ef2d85 |
| region_patch_evidence.py | 85b7f245dc79dd5ed8c04b075854cbd8b0863bf5428a01c49b4e4951f44af7e1 |
| run_m122_full_causal.py | 6f13ab6905166faf7cd1c80f0e2a260f42d048d218fc2436c4d2dd2c4161c9da |
| train_ab_visual_control.py | b1f5c1a458c380eae481b5f4ebccdc619524b8eb8a275ceb5d25fc7a24f49323 |
| train_fixed_visual_selector.py | a8c644a8ca2201c6c36a67c7146725a08ee8a9624fd9ef111da9ca596a930784 |
| train_m107_weak_semantics.py | a9a1f17c2b15caed6b00dfa7d3fa73abfe1f7492f73ee05bc50652e1644bccf4 |
| train_m110_no_weak_rank.py | ad140851b189f363cd16a0fc5610a88720993bb89c59bfe50bc0632149ed4ef9 |
| train_m113_human_initialization.py | aa07a6959b7c83aae5c4ecd4374ea4f6a54691d58b97168e26bcfb31daab15e3 |
| train_m121_dense_target.py | 40d00f84c9f2220deb6f557a0e516f4385922c86d276da1b5d3b17f8891502ee |
| train_m122_full_causal.py | ffa40a08d167d9a7f185cf83ecf0851e9c5ed548c2b1280bb969d6b8c689f891 |
| private deploy_m122.py | df5f7244b5cd06b7c1bbf5df9de4da9d588d65da6ec56daf511e2143ba94c5be |

No changes were requested or made to implementation. A source PASS is not a measured M122 result, final checkpoint acceptance, module contribution claim, or completion of the shared nine-metric goal.
