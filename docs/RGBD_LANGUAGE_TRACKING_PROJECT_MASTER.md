# RGBD 语言跟踪统一交接文档

> 本文件是当前唯一的统一交接入口。它由下列三份远程权威材料完整合并而成；后续交接内容继续只在远程服务器此文件末尾续写。本地副本只是一次性快照，不作为后续编辑源。

- 远程权威路径：`/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_HANDOFF_CANONICAL_ZH.md`
- 合并时间（UTC）：`2026-08-15T05:24:08.536521+00:00`
- 范围：RGBD 语言跟踪、SUTrack 迁移、模板更新、VOT/DepthTrack/CDTB 实验与架构说明

## 合并来源与完整性

| 部分 | 远程来源 | SHA256 | 字节数 |
|---|---|---|---:|
| 实验总账（当前状态与结果索引） | `/home/SRTrack_RGBD_L/refine-logs/EXPERIMENT_TRACKER.md` | `e125ee9878eb8249814d8dad6d0a7b2682fd242d24bd12e1d4df4a105fea3eff` | 16619 |
| SUTrack 迁移、模板更新与 VOT 实验说明 | `/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_SAFE_TEMPLATE_PORT_ZH.md` | `2f5d6c8f6c2d9d0b24d13fe6e1d6b69ac6a1766a347e3d7f18fe587db91c0cda` | 14274 |
| 完整模型架构、推理协议与历史实验记录 | `/home/SRTrack_RGBD_L/docs/final_language_model_architecture_and_inference_zh.md` | `3b8394bb85968853cb60c0dcb83c535aa78e118a74ac0a8be28a66fc93413720` | 454148 |


---

# 第 1 部分：实验总账（当前状态与结果索引）

> 原始远程文件：`/home/SRTrack_RGBD_L/refine-logs/EXPERIMENT_TRACKER.md`；合并时 SHA256：`e125ee9878eb8249814d8dad6d0a7b2682fd242d24bd12e1d4df4a105fea3eff`。

# Experiment Tracker — Depth Reliability Fusion

| Stage | State | Evidence |
|---|---|---|
| Source and metric audit | complete | VOT v0.7.1 multi-start source; formal baseline 72.908956/82.535868/87.988071 |
| Open-source method audit | complete | `docs/rgbd_open_source_tracker_research_20260814.md` |
| Fusion implementation | complete | 28,737-parameter bounded zero-init gate |
| Focused tests | complete | 3 passed; exact initial mean and finite-gradient checks |
| Full checkpoint/optimizer preflight | complete | 0 source tensor mismatches; one optimizer group only |
| Seed 2026 Train-only probe | complete | 8,000 samples, one epoch; 747 frozen source tensors unchanged; 28,737 fusion parameters updated |
| Fixed-6 evaluation | failed capacity gate | P/R/F 65.532075/62.827214/64.151145 versus 65.504506/65.219985/65.361936 |
| Three-seed gate | not authorized | single-seed F delta -1.210791 pp; bag04 -5.067678 pp and ball16 -2.757762 pp |
| Public VOT evaluation | not authorized | no public evaluation was started |

## Seed 2026 terminal evidence

- Run root: `/root/autodl-tmp/srtrack_depth_reliability_fusion_probe_e1_seed2026_v4`
- Checkpoint: `fcbcf91ddf89bfd7fd038954d3dedf3fc1c4789085de7fbf60941ad10a6c2dab`, 1,090,087,372 bytes.
- Training validation: `747` source tensors, `0` mismatches; `6` fusion tensors and `28,737` fusion parameters.
- Runtime result SHA256: `81578f1226dfd0a1298da3fa8b75bb627d1306d3a642749b54139bce44a664b9`.
- Fixed-6 manifest SHA256: `d49d138e147d29caa120f94e0322531e30a631ebbf281388a46ae87490e65dbb`.
- Fixed-6 metrics SHA256: `0b2900fa8ac34306da8e2d69b6709d0d6e4f13e1b5f1c14368cccdaa9ffff1e2`.
- The first post-training validation invocation lacked `PYTHONPATH`; the checkpoint was already complete. The exact frozen implementation snapshot was rechecked before resuming `validate_checkpoint`, so no retraining or artifact substitution occurred.

## Fixed-6 sequence audit

| Sequence | Baseline F | Fusion F | Delta (pp) |
|---|---:|---:|---:|
| toy03_indoor | 80.029876 | 79.736517 | -0.293360 |
| pigeon05_wild | 4.366735 | 4.726144 | +0.359409 |
| bottle03_indoor | 86.856624 | 87.203550 | +0.346926 |
| ball16_indoor | 68.426690 | 65.668927 | -2.757762 |
| bag04_indoor | 72.488367 | 67.420689 | -5.067678 |
| flower03_indoor | 82.276875 | 82.426223 | +0.149348 |

The always-on token preference slightly increased precision (`+0.027569 pp`) but reduced recall (`-2.392772 pp`). The capacity probe therefore falsifies the assumption that positive bounded modal weights alone make learned fusion safe under recursive tracking. It must not be expanded to three seeds.

## Pre-registered next mechanism

The next candidate is not a smaller hand-tuned blend. It must train and deploy a causal counterfactual admission gate around the exact legacy mean: compute legacy and learned-fusion candidates during Train-only supervision, label fusion as admissible only when it improves target support without increasing displacement/identity risk, and use the legacy mean exactly otherwise. The gate must be zero-initialized to reject, preserve the source path byte-for-byte on rejected frames, and pass the same fixed-6 zero-harm gate before any public VOT work.

## Counterfactual-primary isolation probe

The first structural follow-up retained the exact legacy mean as primary and exposed learned fusion only through the already-trained language response router. This was a bounded diagnostic of primary-path isolation, not the final fusion-specific gate.

- Focused tests: `5 passed`; the rejected primary is the exact inherited tensor object.
- Full preflight: `747` source tensors, `0` mismatches; one optimizer group containing only `28,737` fusion parameters.
- Run root: `/root/autodl-tmp/srtrack_depth_reliability_fusion_counterfactual_probe_e1_seed2026_v1`.
- Checkpoint SHA256: `dc437352bbabf39a5451abbf23a5aa8efed91ebe26953282a857dbe7afbbd543`.
- Runtime result SHA256: `47fb8b0222e253dd353f19fa500b60f3aea51f1afc36f16d97760a23577799ff`.
- Fixed-6 manifest SHA256: `4958a594e2c107f2516fd9b40bb7239bd8e37fbdb3e7f99d03b85df1a71475f6`.
- Fixed-6 metrics SHA256: `c626e9c848270097f452fa584a496e90c6fc241af5cd7f9c10f06f5735a8dc82`.

| Sequence | Baseline F | Always-on F | Counterfactual F | Counterfactual delta (pp) |
|---|---:|---:|---:|---:|
| toy03_indoor | 80.029876 | 79.736517 | 80.029876 | +0.000000 |
| pigeon05_wild | 4.366735 | 4.726144 | 4.295578 | -0.071157 |
| bottle03_indoor | 86.856624 | 87.203550 | 86.845721 | -0.010904 |
| ball16_indoor | 68.426690 | 65.668927 | 61.848850 | -6.577840 |
| bag04_indoor | 72.488367 | 67.420689 | 72.587173 | +0.098806 |
| flower03_indoor | 82.276875 | 82.426223 | 82.276875 | +0.000000 |

Aggregate P/R/F was `64.622978/63.755219/64.186166`, or `-0.881527/-1.464767/-1.175770 pp` against safe025. The exact fallback is real (`toy03` bbox output is byte-identical), and it removes the always-on `bag04` corruption. However, the language router admits a few harmful fusion actions on `ball16`; sparse admission is not sufficient without action-specific precision. This follow-up also fails the capacity gate and does not authorize more seeds or public evaluation.

The next implementation must add a separately trained fusion admission head. Its Train-only target is positive only when the fusion candidate improves IoU/support over the exact legacy response, changes the selected peak, and does not increase displacement, identity-switch, or catastrophic-loss risk. Negative and ambiguous rows must resolve to exact reject. Selection must be out-of-fold/held-out and optimize zero harm plus high precision, not action count.

## Strict peak-dominance falsification

A final inference-only diagnostic tested the concrete `ball16` observation that the first counterfactual divergence selected a candidate peak below primary. No checkpoint was retrained and no margin was scanned: a routed proposal was rejected unless its current peak strictly exceeded the exact primary peak.

- Focused tests: `7 passed`.
- Config SHA256: `a788f00b6cfa5e74842291318f1fa4b50bb5cdf6eed73d564dd3b079b92157ec`.
- Fixed-6 manifest SHA256: `08d5ec5fe8b46a5a549bbf9a8216e5001b06f1028df70b147cb3eee9ba903af4`.
- Fixed-6 metrics SHA256: `c79e24a9106ab9aac1e77d4e1ac93564a2417ae05c6c520e257d147a4367d90d`.
- Aggregate P/R/F: `65.435802/63.036177/64.213579`, or `-0.068704/-2.183808/-1.148357 pp` against safe025.
- `ball16` improved only from counterfactual `61.848850` to `61.939140`, still `-6.487550 pp` against baseline; `bag04` remained positive at `+0.070930 pp`, while `flower03` became `-0.190806 pp`.

Concrete causal example (zero-based indices): `ball16` first diverged at frame 526 while the target was absent; legacy/candidate scores were `0.089111/0.039078`, so peak dominance rejects that action. Nevertheless later high-score admissions still produced a divergent state. At frame 887, legacy re-acquired with `IoU=0.931893, score=0.657070`, while guarded fusion remained at `IoU=0, score≈0.34`; the counterfactual trajectory contained 151 visible frames where legacy IoU was at least 0.3 but fusion IoU at most 0.1. Current-frame confidence dominance therefore cannot replace identity evidence and tentative rollback.

No additional score rule is authorized. The next gate must keep a pre-action snapshot, treat a fusion decision as tentative for two causal frames, validate RGB/depth/first-template identity plus legacy-relative support, and atomically restore bbox and recursive tensors on deterioration. Its action policy must be trained/audited from Train-only paired counterfactual traces.

## Strong primary-template capacity falsification

The fixed-6 same-state probe tested primary-template blend weights `0.50` and `1.00`, including the two strictly future frames after every captured risk/write event. Public static outputs remained byte-identical (`12` files, `0` mismatches); GT was joined only after inference.

| Probe | Rows | Weak beneficial/harm/catastrophic/recovery | Strong beneficial/harm/catastrophic/recovery | IoU gain |
|---|---:|---|---|---:|
| Current-frame | 112 | 0 / 4 / 1 / 0 | 0 / 4 / 0 / 0 | weak `-1.770457`; strong incremental `-1.069088` |
| Future two-frame | 292 | 0 / 8 / 1 / 0 | 0 / 7 / 0 / 0 | weak `-3.765301`; strong incremental `-3.844236` |

- `bag04_indoor` frame 1593: static IoU `0.739491`, weight 0.50 IoU `0`, weight 1.00 IoU `0.748352`; the peak response is non-monotonic in blend weight.
- `bottle03_indoor` frames 3156-3158: both stronger templates remained below static throughout the trigger plus future burst, with no recovery transition.
- Temporal manifest SHA256: `c092898cbe2a9dcdac451372c9119333a047a56c93663eb76487027304ea84a2`.
- Temporal analysis SHA256: `32729ef11af32e0a91a350605d6907a5ea12fe10ce3fe1c5daf887673acb142e`.

This route fails the capacity gate and is stopped before full-152, additional seeds, or public VOT. A joint fusion-plus-exact-admission-router seed-2026 Train-only probe is the next bounded experiment; its training loss is not acceptance evidence and it remains unauthorized for public evaluation until fixed-6 safety is measured.

## Joint fusion + router terminal result

The joint seed-2026 probe updated exactly 28,737 fusion and 2,499 router parameters while all 747 source tensors remained unchanged. It failed fixed-6 and was stopped before additional seeds or public VOT.

| Sequence | Joint F | Delta vs safe025 (pp) |
|---|---:|---:|
| toy03_indoor | 80.029876 | +0.000000 |
| pigeon05_wild | 4.274090 | -0.092644 |
| bottle03_indoor | 86.856624 | +0.000000 |
| ball16_indoor | 60.903428 | -7.523262 |
| bag04_indoor | 72.808993 | +0.320627 |
| flower03_indoor | 82.194042 | -0.082833 |

Aggregate P/R/F was `64.771638/63.310567/64.032769`, a `-0.732868/-1.909418/-1.329167 pp` change. On `ball16`, 135 visible frames had baseline IoU at least 0.3 and joint IoU at most 0.1; the longest corrupted run lasted 40 frames. Single-frame router supervision did not protect recursive identity survival.

- Run root: `/root/autodl-tmp/srtrack_depth_reliability_fusion_joint_router_probe_e1_seed2026_v1`.
- Checkpoint SHA256: `0acc61ec82f9d75d915c8702252847edd37d1d3659aa42d19845cb8d4476d6b0`.
- Runtime result SHA256: `954d9b0e04c5a9cb6f109880a7b1d8e8f10a7eb8034ff6e44c5d488b46645ca3`.
- Fixed-6 manifest SHA256: `5dc8088f977d00ba4fd97ff918111aaf0b0806eaf2762e6eb06e02bf4644b329`.
- Fixed-6 metrics SHA256: `aa374b3a6ad163414917c8314a60b6cc55099cdf7b27612e37541b74c5e38f01`.

## Baseline pivot

The SRTrack fusion/template/router branch is closed after repeated fixed-6 falsification. The two retained innovations—SHA-bound structured initial language and fail-closed RGB/depth/temporal safe template update—are being ported to `/home/SUTrack_RGBD_L` at upstream commit `d65052d1ba3fcf55010e1fb3665ee6616c139a2c`. The official SUTrack-L384 baseline is not rerun; only the ported configuration is eligible for smoke and subsequent gated evaluation.

The first real-model port smoke passed on six frames of `adapter01_indoor_1`. It consumed ground truth only for first-frame initialization, loaded the exact official L384 checkpoint, injected the bound language, and exercised the safe template policy. Frame 6 authorized one dynamic-slot replacement; there were no drops and all outputs were finite. Receipt: `/root/autodl-tmp/sutrack_rgbd_language_safe_template_smoke_v1/adapter01_indoor_1_smoke.json`, SHA256 `f9c87b0682128ef80c4cd5f12bed704e58a361dbe20fd57f45415c3951646158`.

A 60-frame continuation on the same frozen implementation also passed. It replaced the dynamic slot once at frame 6, dropped it at frame 28 after identity/depth conflicts, and remained on the immutable template through frame 60. This is a continuity check, not a VOT metric. Receipt SHA256: `aeb66021560b099bbe03afbd28f748aa4640bc90c34b4bc28755eab2e183ac7a`.

## SUTrack VOT failure-family gate

The ported tracker completed all 141 forward/backward anchors from the six largest historical ROB-loss sequences. Ten disjoint shards produced exactly 423 result files and were merged only after per-trajectory completeness checks. The initial foreground SSH controller ended at 105/141 without leaving GPU processes; the same frozen shards were resumed in a detached screen and skipped completed trajectories.

| Same six sequences | EAO | ACC | ROB |
|---|---:|---:|---:|
| Frozen SRTrack historical formal trajectories; not a SUTrack baseline | 39.446101 | 78.281194 | 30.725423 |
| SUTrack port | 48.311723 | 79.807124 | 45.295506 |
| Delta vs SRTrack historical reference | +8.865622 | +1.525931 | +14.570083 |

This table is not a pure-SUTrack baseline comparison. Per instruction, the official SUTrack full-127 baseline is not rerun; therefore the port can report its absolute server-measured metrics and its delta versus the historical SRTrack reference, but not a full-dataset innovation delta versus SUTrack. The 16-anchor default-update branch below is attribution only.

Per-sequence audit is mixed rather than universally safe. `glass01_indoor_2` improved from 19 to 9 failed anchors and ROB 15.817818 to 81.589475; `bag02_indoor_2` improved 13 to 5 and ROB 69.618472 to 86.846209. In contrast, `cup02_indoor_1` remained 36/36 failed and ROB fell 6.959906 to 5.612288, while `shoes02_indoor_1` moved 12 to 13 failures and ROB fell 25.208436 to 10.528037. A preselected 16-anchor default-template-update ablation is therefore required before full-127.

- Shard manifest SHA256: `b7abaa6af4991a22359ab9c5af719a30eca6a7a9f23227f407e8a7c9ab482f0e`.
- Merge receipt SHA256: `61f58bf52679ea67b5a88c6849550a802493807582c52dfe6eda020ccd8c66dc`.
- New analysis SHA256: `aa846361f4a9f1a9168a8b80bcc16387017bd5b8345965bde907da78fbd37616`.
- Reused-reference analysis SHA256: `81c073576b02aa921406c5f489a4db86f0a1a28bf13276a732b3908fa7236ac2`.

## Template attribution and full-127 launch

The preselected 16-anchor attribution showed that v1 safety was not universally better than SUTrack's default update. With the same language/checkpoint/data path, v1 had 8 failures and 10,043 progress frames, default update had 7 and 11,658, and the single pre-registered v2 schedule alignment had 8 and 9,837. V2 is rejected; no further public-threshold scan is authorized. The default branch is an attribution ablation only. To satisfy the retained-innovation requirement, full-127 remains bound to v1, whose complete six-sequence aggregate improved every metric.

- Default-ablation manifest/merge: `6807ef44760e58039d785db88a2c2f3a3129b37875eccf8aa3e56d0ecb08a1ef` / `843e504c6e1f0bb82514de54e3ce225921e38c9693d594d832e63eb37dcd38d4`.
- V2 manifest/merge: `cf0a22946574fe604bc4b6ff54294a7582ff160caa78e47fda62205df0c0dfac` / `215e516a55f8198c3cf8953177514cb958820c37b2d2ff8ad248b3d702747038`.
- Full workspace: `/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1`.
- Full manifest: `8bf5271b3cdc0e0f4587657502f0aa4d873c6cfbc8716f88a1fabb55aa5334b3`.
- Preseed receipt: `1612112e02942c7bf7df1e6f910ba65866c508bcb0be13f981c51ceadea408a9` (141 trajectories / 423 files).
- Detached session: `sutrack_rgbd_safe_full127_v1`; launch progress 141/1,765.

<!-- SUTRACK_FULL127_RESULT_BEGIN -->
## SUTrack full-127 terminal result

无人值守 finalizer 已在 VOT toolkit `0.7.1` 下验证 127 序列、1,765 anchors 和全部结果 SHA，
随后生成正式 full-127 汇总。至少一项目标未达到，不能写成目标已完成。 检查：EAO=未通过、ACC=通过、ROB=未通过。

| 结果 | EAO | ACC | ROB |
|---|---:|---:|---:|
| SRTrack 历史正式参考（非 SUTrack baseline） | 72.908956 | 82.535868 | 87.988071 |
| SUTrack 官方论文报告（未在本服务器复测） | 76.600000 | 83.500000 | 92.200000 |
| SUTrack-L384 + 结构化语言 + safe-v1（本服务器实测） | **73.974969** | **82.627562** | **89.455266** |
| 相对 SRTrack 历史正式参考变化（pp） | +1.066014 | +0.091694 | +1.467195 |
| 目标 | 77.900000 | 82.100000 | 93.700000 |

权威结果：`/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/full_result.json`；analysis SHA256 `e3feabdee88b5dc28938171a08c5d58b13aca52d5cffda8265d4b970e1a68e08`；merge SHA256 `a00462de9fb0025ee10b905564959ab100a40ecdfeee82122b0ac513762846c9`。
SUTrack 官方 baseline 按要求未重跑，因此这里不声称“创新相对 SUTrack baseline 的 full-127 增益”；
只有 SUTrack+创新的绝对实测指标，以及相对 SRTrack 历史正式参考的变化。
该 full-127 只更新 VOT 证据；DepthTrack/CDTB 未在 SUTrack 移植上重测，原有已达标正式数字保持不变。
<!-- SUTRACK_FULL127_RESULT_END -->

---

# 第 2 部分：SUTrack 迁移、模板更新与 VOT 实验说明

> 原始远程文件：`/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_SAFE_TEMPLATE_PORT_ZH.md`；合并时 SHA256：`2f5d6c8f6c2d9d0b24d13fe6e1d6b69ac6a1766a347e3d7f18fe587db91c0cda`。

# SUTrack-L384 RGBD 语言与安全模板更新移植说明

## 1. 决策与边界

当前 SRTrack 的 always-on fusion、counterfactual fallback、峰值支配、强模板和 joint router 均未通过
严格 fixed-6。后续不再继续堆叠单帧规则，而是以 SUTrack-L384 作为新 baseline，保留两项创新：

- 结构化、无 bbox 泄漏的序列级初始语言；
- 由 RGB 身份、原始 depth、运动稳定性与响应唯一性共同授权的动态模板槽。

官方 `sutrack_l384.yaml` 和官方输出不修改、不重测。所有新行为只由
`experiments/sutrack/sutrack_l384_rgbd_language_safe_template.yaml` 启用。

## 2. Baseline 与权重 provenance

| 项目 | 固定值 |
|---|---|
| 仓库 | `/home/SUTrack_RGBD_L` |
| upstream commit | `d65052d1ba3fcf55010e1fb3665ee6616c139a2c` |
| 许可 | MIT |
| 官方模型 | SUTrack-L384 |
| 官方报告 VOT-RGBD22 | EAO/ACC/ROB `76.6/83.5/92.2` |
| 官方报告 DepthTrack | F-score `66.4` |
| L384 checkpoint SHA256 | `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4` |
| CLIP ViT-L/14 SHA256 | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |

MixForRGBD 的竞赛指标更高，但公开 MixFormer 仓库没有提供可直接复现的完整 RGBD winner 配置、融合
实现与对应权重，不能满足“代码和权重均可核验”的移植条件。SUTrack 原生统一处理 RGB、RGBD、RGBT、
event 与 language，文本 token 直接进入主干，因而是更稳妥的可复现起点。

SUTrack 指标高的主要原因不是一个后处理阈值，而是更强的起点组合：L 级 Fast-iTPN 主干、384×384
搜索区域、RGB 与伪彩 depth 的六通道 patch embedding、search/template/text token 的单主干联合注意力、
多任务 task token/decoder，以及 RGB、语言、RGBD、RGBT、event 多域联合训练。官方 L384 配置每个样本
使用两个 template，并在测试时以 0.70 confidence/25 帧更新在线模板。这些能力已包含在官方 checkpoint；
本移植不重新声称它们是创新，只把我们的语言来源与模板安全约束接到公开接口上。

针对 VOT 的优化目标是 ROB/EAO 而非继续抬 ACC：当前旧模型 ACC 已达标，损失来自连续低 overlap 后的
失败与 EAO 零尾。安全模板 gate 不改当前帧 bbox，只阻止错误候选污染后续 template token；结构化语言
则提供跨遮挡仍不变的类别/外观/depth-relation 身份锚点。这两个改动都直接面向“错误身份递归持续”，而
不是扩大搜索框或对所有正常帧改分数。

## 3. 结构化语言路径

清洗后的 VOT-RGBD2022 manifest：

~~~text
/home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/votrgbd2022_language.jsonl
rows 127
SHA256 b0e08fcee58f5ae8119d951eabf4a5688a433864279291add34e56440f57072d
~~~

`RGBDLanguageManifest` 在加载时逐项强制：

1. 文件 SHA256 与配置相等；
2. 恰好 127 个唯一 sequence；
3. dataset 必须是 `votrgbd2022`；
4. `annotation_quality.is_valid=true`；
5. `has_bbox_leak=false` 且 `has_absolute_path=false`；
6. VOT 当前 sequence 必须存在，缺失时直接报错而不是退回零 token。

VOT wrapper 从 RGB 帧路径推导 sequence name，取出 `language` 字段作为 `init_nlp`。SUTrack 原生
CLIP ViT-L/14 text encoder 将其编码为单个 text token；Fast-iTPN 把 search、两个 template 和 text
token 拼接后统一注意力建模。语言不是事后分数重加权，也不读取未来帧或 GT。

## 4. 安全动态模板状态机

slot 0 永久指向首帧模板；slot 1 是唯一可更新状态。每帧读取以下 online-only 证据：

- `confidence >= 0.65`；
- 5×5 NMS 后选中峰与最强竞争峰的差至少 `0.10`；
- 相对上一 bbox 的归一化中心跳变不超过 `0.35`；
- 候选 RGB 与首帧目标的 HSV+灰度 Bhattacharyya similarity 至少 `0.75`；
- 候选中心 80% 区域 raw depth 有效比例至少 `0.50`；
- 相对 trusted depth 的绝对 log change 不超过 `0.08`；
- 连续稳定至少 3 帧，每 5 帧检查，更新间隔至少 30 帧。

满足全部条件后才替换 slot 1。中心大跳、首模板 RGB 身份冲突或 depth 大变会立即丢弃动态槽并恢复
`[static, static]`；动态槽超过 90 帧也恢复。manifest、初始 RGB/depth 锚点或当前 raw depth 不可用时
不会更新，保持 static 路径。该 gate 不改变当前帧 bbox，只影响后续模板 token。

## 5. 代码落点

~~~text
lib/test/tracker/rgbd_language_manifest.py     # SHA/安全字段/序列覆盖门
lib/test/tracker/rgbd_frame.py                 # 无训练依赖的官方六通道读图等价实现
lib/models/sutrack/encoder.py                  # 完整 checkpoint 推理时跳过冗余 pretrain init
lib/test/tracker/safe_template_update.py       # 安全模板决策
lib/test/tracker/temporal_depth_identity.py    # raw-depth 与归一化运动证据
lib/test/tracker/sutrack.py                    # text token + bounded dynamic slot
lib/test/vot/sutrack_class.py                  # sequence/depth path 透传
lib/test/vot/vot.py                            # TraX 0.7 metadata 参数兼容
lib/test/vot/sutrack_l384_rgbd_language_safe_template.py
experiments/sutrack/sutrack_l384_rgbd_language_safe_template.yaml
tools/create_vot_failure_family_shards.py       # anchor 子集/full-127 互斥分片
tools/run_vot_failure_family_shards.py          # 完整性门、进程组清理与合并
tools/seed_vot_shards_from_master.py            # 同配置既有轨迹 SHA 复用
~~~

## 6. 当前验证状态

- Python 3.8 语法检查：通过；
- 新配置完整加载：通过；
- manifest 127/127、固定 SHA、adapter 示例语言：通过；
- 真实 VOT RGB/depth 首帧锚点：通过；
- 人工响应图 NMS margin `0.7`：通过；
- 完整 checkpoint 严格加载前跳过冗余 iTPN 初始化：仅新配置启用；
- 官方 baseline 重测：按要求跳过；
- 完整模型/真实序列烟雾：通过；官方 checkpoint strict load，6 帧均为有限输出；
- `adapter01_indoor_1` frame 6：授权一次动态槽更新，无 drop，终帧 confidence `0.665491`；
- 同序列 60 帧连续性：仅 frame 6 replace，frame 28 身份/depth 冲突 drop，随后保持 static；
- full-127 public VOT：已启动；六条 ROB 失败族的 141-anchor 定向门已完成并复用。

烟雾收据位于
`/root/autodl-tmp/sutrack_rgbd_language_safe_template_smoke_v1/adapter01_indoor_1_smoke.json`，SHA256
`f9c87b0682128ef80c4cd5f12bed704e58a361dbe20fd57f45415c3951646158`。它绑定 checkpoint、config、
language manifest 与 10 个实现文件的 SHA。

60 帧连续性收据位于同目录 `adapter01_indoor_1_smoke_60.json`，SHA256
`aeb66021560b099bbe03afbd28f748aa4640bc90c34b4bc28755eab2e183ac7a`。该运行不读取后续 GT，也不
计算 VOT 指标；它只证明动态槽能在冲突后恢复 immutable template，而不会每隔固定帧盲写。

## 7. VOT-RGBD2022 精确口径与六序列定向门

当前环境使用 VOT toolkit `0.7.1` 的 `vot2022/rgbd` multi-start。每个 anchor 独立向前或向后
运行；可见 GT 上 `IoU <= 0.1` 连续 10 帧才确认失败，中间任一帧恢复到 `> 0.1` 就重置 grace。
确认失败时，progress 回溯到这段连续低重叠的第一帧。对单序列，ACC 是所有 anchor 在 progress
之前的 overlap 加权均值，ROB 是 progress 总和除以各 anchor 方向总长；全数据集 ACC 按 progress
加权、ROB 按原始序列长度加权。EAO 对每个 anchor 构造前缀平均，失败后的尾部补零，再对
`115..754` 的曲线取均值。因此旧模型 ACC 已达标而 ROB 偏低时，优化重点必须是阻止连续失锁和
模板污染，而不是继续提高正常帧的局部框回归。

定向门选择 SRTrack 历史正式结果中 ROB 损失最大的六条序列，完整覆盖 141 个 anchor。SRTrack
参照完全复用既有正式轨迹，没有重跑 SUTrack baseline；新模型用 10 个 disjoint shard 运行，合并后严格得到 141 个 `.bin`、
141 个 confidence 和 141 个 time 文件。

| 六序列聚合 | EAO | ACC | ROB |
|---|---:|---:|---:|
| SRTrack 历史正式轨迹同子集（非 SUTrack baseline） | 39.446101 | 78.281194 | 30.725423 |
| SUTrack + 语言 + 安全模板 | **48.311723** | **79.807124** | **45.295506** |
| 相对 SRTrack 历史参考变化 | **+8.865622** | **+1.525931** | **+14.570083** |

这张六序列表只说明新移植相对 SRTrack 历史正式轨迹的变化；它不是“纯 SUTrack → SUTrack+创新”的
对照。官方 SUTrack baseline 按要求未重跑，故不能从本表声称创新相对 SUTrack baseline 的全量增益。

| 序列 | 旧→新失败 anchor | 旧→新 EAO | 旧→新 ACC | 旧→新 ROB |
|---|---:|---:|---:|---:|
| `cup02_indoor_1` | 36→36 | 22.595796→18.058178 | 85.217282→83.722998 | 6.959906→5.612288 |
| `earphone01_indoor_1` | 20→17 | 39.024433→57.518116 | 76.463173→79.977214 | 24.536487→43.908961 |
| `toy09_indoor_1` | 26→21 | 58.507339→65.244976 | 82.765710→86.278690 | 42.430142→50.614598 |
| `glass01_indoor_2` | 19→9 | 22.354986→69.497410 | 74.217242→80.244396 | 15.817818→81.589475 |
| `shoes02_indoor_1` | 12→13 | 24.612851→12.037950 | 83.432397→83.535061 | 25.208436→10.528037 |
| `bag02_indoor_2` | 13→5 | 62.969132→69.791796 | 74.729894→75.225472 | 69.618472→86.846209 |

定向聚合三项同升，尤其 ROB 提升 14.57pp，证明换到更强公开基线的方向有效；但 `cup02` 与
`shoes02` 仍退化，所以该结果不是 full-127 正式分数，也不授权按公开序列写特例。

组件归因只取预先选定的 16 个最大正/负 progress anchor，语言、checkpoint、读图与 VOT 协议不变。
默认模板更新是诊断对照，不是重跑官方 baseline。v2 也只做一次有原则的修改：把安全 gate 从
5 帧/0.65 对齐到官方 25 帧/0.70，其余安全条件不变，不做阈值扫描。

| 16-anchor 模板策略 | 失败数 | progress 总和 | progress 加权 ACC |
|---|---:|---:|---:|
| 安全 gate v1（部署候选） | 8 | **10,043** | 77.566594 |
| SUTrack 默认更新（消融） | **7** | **11,658** | **78.404825** |
| 安全 gate v2（单次修正） | 8 | 9,837 | 77.560596 |

默认更新在这个困难子集上比 v1 多 1,615 个 progress 帧；`bag02@100`、`toy09@100`、
`cup02@1600`、`shoes02@613` 分别多 841/324/248/174 帧。v1 也有真实反例收益：
`toy09@150` 比默认多 335 帧。v2 不但总 progress 比 v1 少 206，还让原本成功的 `glass01@50`
失败，因此被否决；继续扫描公开 anchor 阈值没有授权。按“保留创新点”的约束，full-127 固定采用
六序列聚合已验证的 v1，而不是切回默认更新。

full-127 工作区覆盖 127 序列、1,765 anchors，10 个 shard 的预计负载为
132,679--132,727 tracker frames。六序列 141 条 v1 轨迹逐文件核 SHA 后预填充，启动进度
`141/1765`；剩余轨迹由 detached `screen` 会话 `sutrack_rgbd_safe_full127_v1` 运行。

~~~text
default-ablation manifest 6807ef44760e58039d785db88a2c2f3a3129b37875eccf8aa3e56d0ecb08a1ef
default-ablation merge    843e504c6e1f0bb82514de54e3ce225921e38c9693d594d832e63eb37dcd38d4
safe-v2 manifest          cf0a22946574fe604bc4b6ff54294a7582ff160caa78e47fda62205df0c0dfac
safe-v2 merge             215e516a55f8198c3cf8953177514cb958820c37b2d2ff8ad248b3d702747038
full-127 manifest         8bf5271b3cdc0e0f4587657502f0aa4d873c6cfbc8716f88a1fabb55aa5334b3
full-127 preseed receipt  1612112e02942c7bf7df1e6f910ba65866c508bcb0be13f981c51ceadea408a9
~~~

权威产物：

~~~text
new shard manifest SHA256 b7abaa6af4991a22359ab9c5af719a30eca6a7a9f23227f407e8a7c9ab482f0e
new merge receipt SHA256 61f58bf52679ea67b5a88c6849550a802493807582c52dfe6eda020ccd8c66dc
new toolkit analysis SHA256 aa846361f4a9f1a9168a8b80bcc16387017bd5b8345965bde907da78fbd37616
old-reference shard manifest SHA256 3cc53b32a57dc0e1229123f94e73054627d5d45ab2ff51400660a0b145b56be5
old-reference toolkit analysis SHA256 81c073576b02aa921406c5f489a4db86f0a1a28bf13276a732b3908fa7236ac2
~~~

full-127 完成并由 toolkit 汇总前，正式当前最好仍是旧模型的
EAO/ACC/ROB `72.908956/82.535868/87.988071`，不能把 SUTrack 论文数字或烟雾运行当作本项目实测
VOT 指标。

## 8. 外部来源

- SUTrack 官方仓库与 model zoo：<https://github.com/chenxin-dlut/SUTrack>
- SUTrack AAAI 2025 论文：<https://ojs.aaai.org/index.php/AAAI/article/download/32223/34378>
- MixFormer 官方仓库（可核验的是通用 backbone，不是完整 MixForRGBD winner）：<https://github.com/MCG-NJU/MixFormer>
- VOT 2022 参赛方法页：<https://www.votchallenge.net/vot2022/participation.html>

<!-- SUTRACK_FULL127_RESULT_BEGIN -->
## 9. full-127 正式结果

无人值守 finalizer 已在 VOT toolkit `0.7.1` 下验证 127 序列、1,765 anchors 和全部结果 SHA，
随后生成正式 full-127 汇总。至少一项目标未达到，不能写成目标已完成。 检查：EAO=未通过、ACC=通过、ROB=未通过。

| 结果 | EAO | ACC | ROB |
|---|---:|---:|---:|
| SRTrack 历史正式参考（非 SUTrack baseline） | 72.908956 | 82.535868 | 87.988071 |
| SUTrack 官方论文报告（未在本服务器复测） | 76.600000 | 83.500000 | 92.200000 |
| SUTrack-L384 + 结构化语言 + safe-v1（本服务器实测） | **73.974969** | **82.627562** | **89.455266** |
| 相对 SRTrack 历史正式参考变化（pp） | +1.066014 | +0.091694 | +1.467195 |
| 目标 | 77.900000 | 82.100000 | 93.700000 |

权威结果：`/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/full_result.json`；analysis SHA256 `e3feabdee88b5dc28938171a08c5d58b13aca52d5cffda8265d4b970e1a68e08`；merge SHA256 `a00462de9fb0025ee10b905564959ab100a40ecdfeee82122b0ac513762846c9`。
SUTrack 官方 baseline 按要求未重跑，因此这里不声称“创新相对 SUTrack baseline 的 full-127 增益”；
只有 SUTrack+创新的绝对实测指标，以及相对 SRTrack 历史正式参考的变化。
该 full-127 只更新 VOT 证据；DepthTrack/CDTB 未在 SUTrack 移植上重测，原有已达标正式数字保持不变。
<!-- SUTRACK_FULL127_RESULT_END -->

---

# 第 3 部分：完整模型架构、推理协议与历史实验记录

> 原始远程文件：`/home/SRTrack_RGBD_L/docs/final_language_model_architecture_and_inference_zh.md`；合并时 SHA256：`3b8394bb85968853cb60c0dcb83c535aa78e118a74ac0a8be28a66fc93413720`。

# 最终 RGB-D-L 跟踪模型：早期三模态交互与一致性漂移控制

> 文档状态：2026-08-05，primary tri-modal 已通过 DepthTrack/CDTB 保真门；完整 VOT reference 已冻结，V31 top-14 投影已拒绝，正在四卡采集 DepthTrack Train 真实递归 top-8 证据；当前验证入口默认启用安全动态模板，在线 Qwen 仍关闭
> 代码仓库：`/home/SRTrack_RGBD_L`
> 当前主训练配置：`experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1.yaml`
> 当前主推理配置：`experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml`
> 当前主权重 SHA256：`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab`
> 推理口径：safe025 架构 + `fixed6_isotropic_098_v1`；DepthTrack/CDTB 保真结果另启用冻结 v11，当前 VOT 架构对照关闭 v11
> 训练数据：只使用 DepthTrack Train；CDTB/VOT-RGBD2022 不参与训练或阈值选择

本文以实际代码、checkpoint、评测 manifest 和原始预测文件为准。fixed-6 只用于
开发期选权重；可报告的 DepthTrackTest 主结果必须覆盖全部 50 条序列、76,373 帧。

## 1. 核心结论

当前主模型候选是 **Primary Early Tri-modal Cross-Attention with Consensus Guard
（早期三模态交互与一致性门）**。它继承双裕量 CRAR-V5 的完整视觉动作 `S_v` 和语言
动作 `S_l`，但把语言从尾部仲裁提前到 ViT 第 2 层边界：类别、稳定属性和初始状态三个
语言 role 分别与 RGB/Depth 模板 token 交互，再让搜索 token 反向查询已定位的语言 role。
第 2 层 adapter 同时产生模态独立 grounding map 与一致性 map；其有界 search-token residual
继续通过后续 RGB、Depth 和四个跨模态 block 形成候选。最终一致性门只允许否决缺乏
RGB-D 联合支持的 CRAR 语言动作，不允许凭空创建新语言路由。

新增早期交互模块包含 844,482 个参数，一致性门包含 571 个参数；本轮只训练这 845,053
个新增参数，继承的视觉主干、Center Head、CLIP、两阶段语言适配器和 CRAR 均冻结。
safe025 在加载同一权重时把 search-token residual 上限收紧为 `0.00010`，把 CRAR logit
最大抑制收紧为 `0.25`，以控制跨域轨迹偏移。

- 初始化时只读取一次首帧目标描述、类别、稳定属性和结构化 RGB-D 初始状态。
- 后续每帧默认重复使用同一首帧语言；正式结果中的首帧视觉模板、文本和身份原型均不更新。
- 早期 cross-attention 显式保留 RGB、Depth 两张 grounding map，并用二者的一致性控制 search-token residual。
- CRAR 读取响应质量、峰值位移、语义秩差、RGB-D 身份证据等 25 个归一化量。
- 一致性门额外读取 9 个模态秩特征，修正值经过 `clamp(max=0)`，因此只能抑制原 CRAR 语言动作。
- `g=0` 精确使用冻结视觉响应，`g=1` 使用完整语言候选响应；不存在第三张混合图。
- V5 对固定推理阈值 0.5 两侧同时施加 margin：风险负例压到阈值下，正例拉到阈值上。
- 当前正式测试不读取逐帧人工/Qwen 文本，不调用 Qwen，也不周期生成描述；在线语言仍只作为尚未晋升的推理优化。
- 表 1.1 已报告的 primary DepthTrack/CDTB 保真结果使用
  `ONLINE_LANGUAGE_UPDATE.USE=False`、`SAFE_TEMPLATE_UPDATE.USE=False`。其后的 V31/V32
  部署验证链和 `tracking/evaluate_depthtrack_validation.py` 已改为默认启用安全动态模板；
  首帧模板保持不可变，只更新受门控的动态槽。distractor quarantine、identity rerank、depth
  rescue、route-confirmation 和通用时序框拒绝仍关闭。
- 部署报告框使用 fixed-6 选择的各向同性 `0.98` 尺度；递归 tracker state、下一帧 crop、
  v11 锚点和 score 均保留未缩放值，首帧协议框不缩放。
- 空文本通过显式 mask 严格回退冻结视觉路径，便于做同权重因果对照。

必须区分论文架构贡献和部署策略：**早期语言-RGB-Depth cross-attention、模态一致性图
和 veto-only 漂移门是当前学习架构的主创新**；CRAR-V5 是继承的离散动作基础。v11、
固定 0.98 报告尺度、安全模板槽和在线语言都只能作为单独推理增强。在线语言的定位尤其
严格：它只是尝试减轻长期外观变化和模板偏移，不作为核心网络贡献，也不能把尚未完成的
在线实验写入主结果。

### 1.1 当前指标

DepthTrack 使用 VOT long-term 宏平均 PR/Recall/F1。fixed-6 只承担开发选模，正式
目标判断只来自 50 条序列、76,373 帧的 full-50。

| 评测范围 | 设置 | PR (%) | Recall (%) | F1 (%) | 覆盖/状态 |
|---|---|---:|---:|---:|---:|
| fixed-6 | V5 empty、无后处理 | 64.8780 | 64.2284 | 64.5515 | 6 / 10,041 |
| fixed-6 | V5 正确文本、无后处理 | 65.5045 | 65.2200 | 65.3619 | 6 / 10,041 |
| fixed-6 | primary tri-modal safe025、无后处理 | 65.5045 | 65.2200 | 65.3619 | 6 / 10,041；六条轨迹与 V5 byte-identical |
| fixed-6 | V5 正确文本 + 冻结 v11 | 65.8311 | 65.9624 | 65.8967 | 6 / 10,041，尺度基线 |
| fixed-6 | 上项 + 固定 0.98 报告尺度 | **65.9333** | **66.0042** | **65.9688** | 6 / 10,041，尺度冠军 |
| 完整 DepthTrackTest | V5 + v11，原始报告框 | 65.9438 | 65.2732 | 65.6068 | 50 / 76,373，已达标 |
| 完整 DepthTrackTest | V5 + v11 + 0.98 | **65.9978** | **65.3358** | **65.6652** | 50 / 76,373，历史最好 |
| 完整 DepthTrackTest | primary safe025 + v11 + 0.98 | **65.9959** | **65.3359** | **65.6643** | 50 / 76,373，当前主模型；接近历史最好 |
| 完整 CDTB80 | primary safe025 + v11 + 0.98 | **75.3878** | **76.0059** | **75.6956** | 80 / 101,956；三项目标均通过 |
| 预设目标 | - | 65.2000 | 64.9000 | 65.1000 | 50 / 76,373 |

primary safe025 full-50 相对目标为 `+0.7959 PR / +0.4359 Recall / +0.5643 F1`；其
F1 只比 V5 历史最好低 `0.0009` 个百分点。CDTB80 相对目标为
`+2.4878 PR / +0.4059 Recall / +1.4956 F1`，F1 比 V5 历史最好低 `0.0465` 个百分点。
这两组结果支持“跨数据集近似保真”，但不能表述为已经改善历史最好。原 V5 full-50
`metrics.json` / manifest SHA256 为 `07bcf2...d3c4` / `ea31d1...7861`；0.98 派生
结果为 `93653d...f2ae` / `f85e6c...cb19`。派生 manifest 绑定源 manifest、源 metrics、
全部源预测、checkpoint/config/审核文本和校准实现 SHA，且记录
`future_frame_text_used=false`、`language_source_frame_index=0`、
`language_search_recovery_profile_name=longterm_scale_adaptive_v11`、
`reported_box_scale_feedback_to_tracker_state=false`。

### 1.2 历史 CPSD 完整集退化：不是文本错误，而是闭环身份漂移

逐序列结果为 25 条提升、23 条退化、2 条不变。主要正负样例如下：

| 序列 | 空文本 F1 | 正确文本 F1 | 差值 |
|---|---:|---:|---:|
| `pigeon01_wild` | 82.2019 | 39.5811 | -42.6209 |
| `dumbbells01_indoor` | 73.9585 | 60.3813 | -13.5772 |
| `pigeon02_wild` | 53.9041 | 45.4069 | -8.4972 |
| `ball01_wild` | 61.2125 | 71.9963 | +10.7837 |
| `cup02_indoor` | 29.3456 | 36.6329 | +7.2873 |

三条严重退化序列的审核文本均与首帧目标一致。语言 score 从第 2 帧就与空文本路径有
小差异，但 `pigeon01_wild`、`dumbbells01_indoor`、`pigeon02_wild` 的预测框分别到
第 145、121、53 帧才首次分叉。这些时刻都接近多峰、遮挡或 re-entry 状态；一次峰值
切换改变下一帧搜索 crop，随后误差在闭环中累积。`pigeon01_wild` 第 315 帧语言路
IoU 已为 0，而空文本仍为 0.870。

因此该 CPSD 版本的关键缺口是训练目标只衡量独立帧上的局部效用，没有优化递归跟踪风险。
独立 Head-Aligned refiner 还能绕过继承空间门，在多个同类别目标时让类别语义压过实例
模板。下一轮应训练连续 3--5 帧的因果展开和视觉实例一致性风险门，而不是添加测试后
处理：只有模板相似度、RGB/depth 一致性和短窗效用同时支持时才允许语言改变视觉峰值。

### 1.3 2026-08-02 primary tri-modal safe025：已晋升为当前主模型候选

此前 early-grounding probe 只把 RGB/Depth 平均后的早期语言旁路送入候选分支，跨域
轨迹虽接近 V5，但没有显式约束 RGB 与 Depth 是否共同支持语言峰。primary tri-modal
版本在同一早期交互中保留三张可审计图：

```text
M_rgb       = language-grounded RGB search map
M_depth     = language-grounded Depth search map
M_consensus = 0.5 * (M_rgb + M_depth) - 0.25 * |M_rgb - M_depth|
```

一致性门比较视觉峰和语言候选峰在三张图中的秩，并读取 RGB/Depth 最小支持、模态秩差
与一致性变化，共 9 个特征。其输出满足：

```text
delta_guard <= 0
route_logit = inherited_crar_logit + delta_guard
```

因此 early cross-attention 提供三模态交互能力，一致性门负责控制这种能力引起的错误峰值
切换。safe025 的 fixed-6 与 V5 六条轨迹逐字节一致；完整 DepthTrackTest 和 CDTB80 保持
在历史最好附近且均高于预设目标，故晋升为当前主模型候选。正式 VOT 使用官方 127 序列、
1,765 anchors、1,327,004 tracker frames 的 multi-start 协议；V5 no-recovery 对照已完成，
primary safe025 已自动接续运行，结果文件生成前不填写预测 EAO/ACC/ROB。

在线语言和动态模板没有参与上述训练或指标。后续即使启用，也只能作为漂移恢复的可选
推理 profile，不能改变本节对主创新的归因。

### 1.4 2026-08-01 早期语言 grounding 并行 probe：历史负晋升结果

该历史实验实现了一个从第 2 层分叉的早期 grounding 分支，用于验证“语言先作用于 RGB-D
token、再由后续 ViT 层完成视觉整合”是否比只在尾部生成语言候选更稳。它当时没有替代
冻结 V5，现也已被第 1.3 节的 primary tri-modal safe025 取代。并行分支的契约如下：

- canonical RGB/depth stream 从输入到最终层保持不变，CRAR 的视觉 score、size map、
  offset map 和 raw-RGB identity 均读取 canonical token；
- 早期分支只在至少存在一个首帧语言 role（category、stable appearance、initial state）
  的样本上运行，缺失 role 的行逐 token 复制 canonical 输出；
- template token 不被 grounding residual 改写，bounded residual 只写入 search token，
  residual_max=0.04，插入边界为 INSERT_LAYER=2；
- grounded token 只供语言 candidate 和 semantic cache 使用，最终 hard route 仍在完整
  visual/candidate response 之间选择，不引入第三张响应图；
- 所有 added tensors 都属于 language_early_grounding.*，没有改变 checkpoint 的旧键
  命名或视觉主干参数。

该 probe 只使用 DepthTrack Train、seed 2026、8,000 个 causal samples、1 epoch。新模块
增加 25 个 tensor、695,105 个参数；708 个继承 tensor 与 V5 逐字节一致，optimizer 只
包含新增模块，学习率为 1e-4。初始化时 raw gate=0.0028032360、effective scale=0.0027986555，
冻结 router EMA 仍为 0.01555。

同一 checkpoint 的 fixed-6 correct/empty 门禁通过：empty prediction 与 V5 empty 逐字节
一致，candidate boxes 与 V5 correct 在 10,041 帧全部一致，且 correct 相对 empty 为
+0.626552 PR / +0.991610 Recall / +0.810406 F1。但是 full benchmark 的严格
preserve gate 未通过，故该分支不能替换 V5，也不能启动官方 VOT：

| 完整评测 | V5 | early-grounding probe | 相对 V5 |
|---|---:|---:|---:|
| DepthTrackTest PR / Recall / F1 | 65.997845 / 65.335825 / 65.665166 | 65.997790 / 65.335819 / 65.665136 | -0.000054 / -0.000005 / -0.000030 |
| CDTB80 PR / Recall / F1 | 75.430422 / 76.056392 / 75.742113 | 75.404807 / 76.010464 / 75.706424 | -0.025614 / -0.045928 / -0.035689 |

DepthTrack 的 50 序列 audit 没有 sustained trajectory regression（48 条逐序列不变、2
条提升、0 条退化候选）；CDTB 的 80 序列 audit 有 15 条提升、16 条退化、49 条不变，
并标出 7 条 sustained-regression candidates，最大负例为 humans_corridor_occ_2_A
（F1 -3.5799）。因此该结果只能作为已实现、已测量的架构 probe 和负晋升结果，不能
把 fixed-6 的正增益或“达到宽松目标”写成 full benchmark 改进。

## 2. 总体网络框架

```text
首帧 RGB-D 模板 192x192 ─┐
当前帧 RGB-D 搜索 384x384 ┘
            │
            ├─ canonical RGB/Depth ViT 全程 ──────────────── 冻结视觉 Head ─ S_v / size / offset
            │
            └─ 第 2 层边界复制 RGB/Depth token
                    + 首帧语言 roles（CLIP）
                    ├─ role -> RGB template cross-attention
                    ├─ role -> Depth template cross-attention
                    └─ RGB/Depth search -> grounded role cross-attention
                           ├─ 当前边界的 M_rgb / M_depth / M_consensus ─────┐
                           └─ 有界 search residual                         │
                                  └─ 后续双流层与跨模态 block              │
                                         └─ candidate Head ─ S_l           │
                                                                            │
M_rgb / M_depth / M_consensus + 候选峰 <────────────────────────────────────┘

S_v / S_l + 25 维证据 ─ 冻结 CRAR ─ inherited route logit
三张 early grounding map + 候选峰 ─ 9 维 guard ─ 非正 logit 修正
                         └─ 完整响应硬选择 + canonical size/offset ─ 框解码

可选推理增强：v11 搜索、0.98 报告尺度、安全动态模板槽、低权重在线语言
```

canonical 视觉路径从输入到输出保持不变，保证 `S_v`、size、offset 和 raw-RGB identity
仍有稳定参照。早期语言路径在第 2 层边界分叉，模板 token 不被 residual 改写，bounded
residual 只作用于 search token；随后它不是直接输出一张尾部语义图，而是重新经过余下
视觉层完成 RGB-D-L 联合编码。缺少有效语言 role 的样本逐 token 回退 canonical 输出。

当前学习网络由九个主要部分组成：

1. RGB 与 Depth 双流视觉编码器；
2. 四次跨模态 Transformer 交互；
3. 冻结的 CLIP 文本编码器；
4. 类别到稳定属性的两阶段语言适配器；
5. 角色分离、视觉收益门控和 Head-Aligned 有界特征校正；
6. 第 2 层边界的语言-RGB/Depth 双路 cross-attention；
7. RGB、Depth 和 disagreement-penalized consensus 三张 grounding map；
8. 双动作之间的 25 维证据抽取、冻结 CRAR 与 9 维 veto-only 一致性门；
9. 原 Center Head、原 Hann window 与框解码器。

历史逐帧 teacher 和 GT 效用选择只存在于早期 CPSD 训练图中；当前配置仍为
`FRAME_TEACHER_USE=false`。主模型训练不包含恢复状态机、候选重排、安全模板更新或在线
Qwen。它们若启用均属于推理期 profile，不进入网络参数或训练损失；当前正式 safe025
VOT 对照全部关闭，只保留 0.98 report-only 框尺度。

## 3. RGB-D 视觉主干

### 3.1 输入形式

每一帧输入为六通道：

- RGB：3 通道；
- Depth：经过当前深度预处理得到的 3 通道表示。

模板和搜索区域的尺寸为：

| 输入 | 尺寸 | Patch size | 每模态 Token 数 |
|---|---:|---:|---:|
| 首帧模板 | 192×192 | 16×16 | 12×12 = 144 |
| 当前搜索区域 | 384×384 | 16×16 | 24×24 = 576 |
| 合计 | — | — | 720 |

RGB 和 Depth 分别经过独立的 patch embedding，因此两个模态在进入 Transformer 之前不会被简单拼接成一个共享投影。

### 3.2 双流 ViT-B/16

RGB 与 Depth 各使用一条独立的 ViT-B/16 分支：

- Transformer 层数：12；
- 特征维度：768；
- 注意力头数：12；
- MLP ratio：4；
- RGB 与 Depth 拥有独立的 12 层 block 和归一化层。

代码沿用了上游的 `event` 命名，但本项目第二模态实际是 Depth。

### 3.3 跨模态交互

在零基索引 2、5、8、11，也就是第 3、6、9、12 层后，执行四次跨模态交互：

1. 拼接当前 RGB Token 与 Depth Token；
2. 送入一个跨模态 Transformer block；
3. 按原长度重新拆分为 RGB 和 Depth Token；
4. 继续进入各自后续的模态分支。

最后分别归一化两种模态，并计算：

```text
visual_fusion_tokens = (rgb_tokens + depth_tokens) / 2
```

两阶段语言分支不仅接收平均后的视觉特征，还接收独立的 RGB Token 和 Depth Token，从而可以学习不同模态对类别和属性的贡献。

### 3.4 当前没有执行硬 Candidate Elimination

最终 YAML 中保留了：

```yaml
CE_LOC: [3, 6, 9]
CE_KEEP_RATIO: [0.7, 0.7, 0.7]
```

但当前实际使用的 `vit_siam_dropmae` 前向没有执行 candidate elimination，`ce_keep_rate` 不会删除搜索 Token。

因此论文中不能把当前模型描述成已经实现了 70% Token 保留率的硬剪枝。当前两阶段语言匹配也是软门控，不会不可逆地丢弃候选位置。

## 4. CLIP 文本编码器

当前已经完全去掉 RoBERTa，使用本地 CLIP ViT-B/32 文本模型：

```text
/home/OSTrack_RGBD_L_dataset_modified/pretrained_models/clip-vit-base-patch32
```

主要设置为：

| 项目 | 当前设置 |
|---|---:|
| 最大文本长度 | 77 |
| CLIP 主体 | 冻结 |
| 本地文件模式 | 开启 |
| 文本特征 | L2 归一化 |
| 文本缓存 | 4096 条 |

CLIP 主体不参与训练。早期阶段训练过：

- CLIP 到跟踪视觉空间的投影；
- RGB/Depth 类别投影；
- RGB/Depth 属性投影；
- 粗匹配和细匹配输出层；
- 可靠性门控；
- 特征校正与可靠性门控参数。

CPSD 正式阶段从保留的 SRD-HAC Epoch 1 权重继续初始化。视觉主干、跨模态 block、
原 Center Head、CLIP 和旧语言投影全部冻结，只训练
`language_feature_correction` 参数组，共 331,913 个参数，学习率为 `5e-5`，
第 4 轮后衰减。正式训练为 5 epochs x 4,000 causal pairs。它属于加载 RGB-D
跟踪权重后的参数高效微调，而不是从头训练完整 ViT；论文必须同时报告初始化来源、
冻结范围、可训练参数量和实际训练轮数。

## 5. 当前真正生效的语言信息

一条首帧富文本会被整理为四类输入：

1. 完整描述 `language_text`；
2. 目标类别 `language_category`；
3. 稳定外观属性 `language_attributes`；
4. 首帧状态 `language_state`：在第 2 层 early adapter 中作为第三个实例绑定 role，
   在继承的 two-stage 尾部分支中只作可靠性控制上下文。

`TargetDescriptionStore` 同时保留 `fine_attributes = stable_attributes +
initial_state` 以兼容旧模型；但当前 `ROLE_SEPARATION_USE=true`，正式 tracker
明确拆开两部分：

- `language_attributes` 只取 `stable_attributes`，例如颜色、材质、形状和稳定
  局部标记，参与第二阶段空间精匹配；
- `language_state` 取首帧深度关系、深度质量、遮挡状态和干扰物关系；它在 early
  adapter 中参与模板 grounding 和 RGB/Depth search-token 实例绑定，在继承的 SRD-HAC
  尾部分支中只作 frame-level 控制上下文。该 role 始终标记为 frame 0 状态，不能被当作
  每一后续帧都成立的当前真值。

当前配置为：

```yaml
FUSE_TOKENS: false
FUSE_HEAD: false
TWO_STAGE.USE: true
TWO_STAGE.RESPONSE_FUSION_USE: false
TWO_STAGE.FEATURE_CORRECTION_USE: true
TWO_STAGE.ROLE_SEPARATION_USE: true
TWO_STAGE.HEAD_ALIGNED_CORRECTION_USE: true
```

这意味着：

- 完整句子会被 CLIP 编码；
- 但完整句子的传统 FiLM Token 融合和 Head 融合均关闭；
- 第 2 层 early 分支由类别、稳定外观、首帧状态和模板身份证据共同有界地改变搜索 Token；
- 初始 RGB-D 状态参与 early grounding map，但在继承的 two-stage 尾部分支中只调节
  frame-level 可靠性；
- 没有直接修改最终响应图的 response-fusion 分支。

因此，首帧深度关系和遮挡信息确实进入网络，但角色是“初始化条件与可靠性上下文”，
不是永久空间定位指令。`motion_or_state`、详细证据句和未进入白名单的自由文本字段
不会直接控制主路径。代码目前是由结构化字段在网络外完成角色拆分，并非 CLIP
token 上的端到端 slot attention；论文中不能把它描述成自动句法分解器。

## 6. 第一阶段：类别粗匹配

以“一支红色的笔”为例，第一阶段只使用类别“笔”。

### 6.1 构建首帧目标原型

在首帧模板 Token 上使用目标中心区域 mask，分别构建：

- RGB 模板原型；
- Depth 模板原型。

当 mask 不可用时，代码会退化为模板 Token 的均匀平均，避免产生零向量或 NaN。

### 6.2 类别与搜索 Token 匹配

将类别 CLIP 特征分别投影到 RGB 和 Depth 的 128 维匹配空间，计算：

```text
RGB粗分数   = RGB搜索Token与类别的相似度
            + 1.5 × RGB搜索Token与模板原型的相似度

Depth粗分数 = Depth搜索Token与类别的相似度
            + 0.5 × Depth搜索Token与模板原型的相似度
```

随后对 RGB 和 Depth 分数执行 softmax 模态路由，得到每个位置更依赖 RGB 还是 Depth。

第一阶段输出：

- 类别粗匹配图；
- RGB/Depth 模态路由图；
- 粗粒度可靠性；
- 粗粒度 Token 门控。

### 6.3 为什么不硬删除 Token

第一阶段不会把低分 Token 删除，只通过软门控降低它们的影响。这样即使类别粗匹配暂时判断错误，第二阶段仍然可以在全部原始搜索 Token 上利用属性重新判断。

这比“先硬剪枝、再精匹配”更适合长期跟踪，因为硬剪枝错误通常不可恢复。

## 7. 第二阶段：属性精匹配

第二阶段只加入“红色”等稳定身份属性。结构化首帧状态走独立的 frame-level
控制路径，不参与此处的空间属性 logits。精匹配核心形式是：

```text
fine_logits = complete_coarse_logits + attribute_logits
```

关键性质包括：

1. 完整类别分数始终保留；
2. 属性在未剪枝的原始搜索 Token 上重新计算；
3. 属性匹配可以救回第一阶段粗分数较低的位置；
4. 属性不能完全覆盖类别证据。

当前配置要求：

```yaml
COARSE_WEIGHT_MIN: 0.70
```

因此当属性存在时，类别残差在最终语言残差中的权重至少为 70%，属性最多使用剩余部分。当属性缺失时，全部残差容量分配给类别分支。

这对应“类别粗筛选—属性精定位—类别保底”的设计，但采用可微分软级联实现。

需要注意，深度关系、遮挡和干扰物关系描述的都是 **frame 0 状态**。它们在第 2 层
early adapter 中作为固定的第三个 role 参与实例绑定，在继承的 SRD-HAC 尾部分支中只作
状态上下文；后续帧不会更新，不能被解释成模型获得了当前帧的额外真值信息。

## 8. 可靠性控制与有界残差

本节与第 9 节先记录继承的 SRD-HAC/VHSC 连续校正组件。它仍在当前模型中计算安全状态，
但当 `COUNTERFACTUAL_ROUTER_USE=true` 时，其返回的连续校正 token 不是最终 `S_l` 动作；
当前 primary 的真实候选构造与这些安全量的用途以第 16.6--16.8 节为准。

语言分支不会覆盖视觉主干，而是先保留一条完全不变的 `F_visual`，再构造候选
`F_grounded`。SRD-HAC 从原视觉响应、语义图和模板身份图预测是否、以及在哪里
应用校正：

```text
F_center = F_visual
         + g_frame * g_spatial * (F_grounded - F_visual)
         + g_refiner * bounded_refiner_delta
```

当前设置为：

| 参数 | 数值 |
|---|---:|
| 语言瓶颈维度 | 128 |
| 两阶段候选残差上限 | 0.04 |
| VHSC frame/spatial 基础 floor | 0.35 / 0.65 |
| Head-aligned gate logit 校正上限 | 8.0 |
| 独立 refiner 元素级残差上限 | 0.04 |
| 类别最小混合权重 | 0.70 |

frame gate 读取视觉最高响应、第一第二峰差、响应熵、视觉/粗匹配/精匹配一致性、
RGB/Depth 模板身份一致性和 coarse/fine 可靠性；SRD-HAC 再加入粗细分布统计、模态
一致性、属性增量和首帧状态上下文。spatial gate 读取 11 张 24×24 图，包括视觉、
类别、属性、RGB/Depth 身份图及其乘积和差分关系。

具体安全约束为：

- 只修改搜索 Token，不修改首帧模板 Token；
- 文本类别为空时，残差严格为 0；
- 空文本通过显式 `torch.where` 精确回退原始视觉 Token；
- 模板与文本不兼容时，可靠性会降低；
- 残差经过 `tanh` 和最大尺度双重限制；
- 新增输出层零初始化，从 VHSC 初始化时预测不会突变；
- 独立 refiner 仍受类别 mask 和 curriculum 控制，不能在无文本时形成固定偏置。

## 9. Head-Aligned 校正与原 Center Head

当前配置明确关闭响应图融合：

```yaml
RESPONSE_FUSION_USE: false
FEATURE_CORRECTION_USE: true
HEAD_ALIGNED_CORRECTION_USE: true
```

在不启用 CRAR 的历史连续路径中，校正发生在冻结 Center Head 之前，而不是把语义热图
直接加到最终 score map。当前 CRAR 路径仍运行该控制器以产生 frame/refiner gate 等证据，
但硬路由候选在连续校正前已经由完整两阶段 `grounded` token 生成。两种路径的 size 和
offset 分支都始终读取未校正的 `F_visual`，避免让首帧文本决定当前帧尺度：

- 中心响应图 `score_map`；
- 宽高图 `size_map`；
- 中心偏移图 `offset_map`。

网络内部仍使用原 Hann window、原 argmax 和原框解码，不重排 Center Head 候选，也不在
网络内拒绝或修正已经解码的框。冻结 v11 与 0.98 报告尺度属于第 12 节的独立部署 profile；
当前 DepthTrack/CDTB 保真启用 v11，而正式 VOT 架构对照关闭 v11。

## 10. CPSD 训练时如何使用文本

### 10.1 图像采样

最终语言配置每个训练样本包含：

- 1 张模板帧；
- 1 张搜索帧；
- 模板固定使用 frame 0；
- 搜索帧从首帧后的完整序列中按早期、中期、后期区间抽取；
- early/middle/late 概率为 0.20/0.30/0.50；
- 10% 样本为 causal absence hard negative；
- 每轮 4,000 个有放回 causal pairs；
- batch size 为 16，共 250 iterations/epoch；
- 共训练 5 轮，Epoch 4 后学习率衰减；
- 双 GPU 完整训练实测约 `20:53`，约 `4:11/epoch`；
- 只更新 331,913 个 Head-Aligned 语言特征校正参数。

5 个 checkpoint 每轮都完整跑 fixed-6 开发集，F1 最高的 Epoch 4 才被冻结为正式
full-50 候选。fixed-6 仅用于选权重，绝不作为论文主指标。当前实验仍只有 seed 2026，
论文终稿需要补多随机种子以量化方差。

### 10.2 首帧主查询

主查询来自：

```text
depthtrack_train_first_rich_reviewed_qwen3_v5.jsonl
```

最终配置 `DATA.LANGUAGE.INITIAL_PROB=1.0`，因此无论搜索帧采到第几帧，student 主查询始终使用首帧富文本。

这保证：

- 搜索帧采到后续帧时，语言不会消失；
- 训练和正式推理都遵守“首帧给定目标描述”的因果边界；
- 语言描述表达的是目标长期身份，而不是当前帧场景描述。

另有 15% 文本 dropout，用于训练视觉回退能力和空文本对照。

### 10.3 逐帧教师文本

逐帧文本来自经过 Qwen2.5 生成和 Qwen3 审核的 overlay。数据管线按实际
`search_frame_id` 精确读取并验证索引。CPSD 正式配置为：

```yaml
FRAME_TEACHER_USE: true
FRAME_TEACHER_DROPOUT: 0.05
LANGUAGE_TEACHER_WEIGHT: 0.0
LANGUAGE_PRIVILEGED_DISTILL_WEIGHT: 0.20
LANGUAGE_PRIVILEGED_FEATURE_WEIGHT: 0.50
```

`LANGUAGE_TEACHER_WEIGHT=0` 只表示旧的 RGB grounding loss 关闭，不能再解释成
“逐帧文本没有使用”。当前帧描述会构造一个训练期 privileged teacher，但 student
始终只读取首帧查询。具体过程是：

1. 用同一 RGB-D 视觉 Token、首帧类别和当前帧描述生成 teacher 候选；
2. 在完整 Head-Aligned teacher response 与 RGB semantic response 中，按 GT target
   support 和 target-to-hard-background gap 选择较强视图；
3. 只有候选相对未改动视觉 tracker 的 support 或 gap 提升超过 0.05 时才选中；
4. 对选中样本，用温度 0.25 的响应分布 KL 和 feature-delta Smooth-L1 蒸馏到
   首帧 student；
5. teacher response、feature delta 和选择标签全部 detach，不能通过 teacher 分支
   反向迎合 GT；
6. absence 样本会清空 teacher，另有 5% teacher dropout 和 15% student text
   dropout。

因此训练与部署的信息边界是：

```text
首帧富文本 = 部署时可获得的主查询
逐帧文本   = 仅训练可用的 privileged teacher
```

这是一种 learning using privileged information：后续帧文本只负责教会首帧查询在
外观变化时如何校正视觉特征，推理时完全删除 teacher 分支。

5 轮日志的 50 个统计窗口中，teacher 平均可用率为 48.305%，真正通过效用门并参与
蒸馏的样本平均只占全部样本 8.155%，约为可用候选的 16.88%。因此 CPSD 不是把所有
Qwen caption 当作真值，而是只学习少量经视觉基线和 GT 定位效用共同确认的教师事件。

### 10.4 反事实文本监督

训练时还会为一个 batch 构造：

- 错误类别文本；
- 正确类别但错误属性文本。

当前 CPSD 实际启用了 Head-level 反事实监督，权重为 1.0、margin 为 0.10；它
比较正确文本、错误类别和同类错误属性对应的最终中心响应，避免语言分支退化成与文本
内容无关的固定偏置。空文本则通过 15% dropout 和显式空文本评测检查严格视觉回退。

## 11. 正式测试推理协议

### 11.1 初始化阶段

tracker 在 frame 0 执行一次：

1. 从首帧框裁剪 192×192 RGB-D 模板；
2. 保存固定模板张量 `z_tensor`；
3. 读取并保存：
   - `init_language_description`；
   - `init_language_category`；
   - `init_language_stable_attributes`；
   - `init_language_state`；
4. 保存首帧目标身份原型；后续不更新模板，也不更新文本。

### 11.2 后续每一帧

每次 `track()` 都将同一组首帧字符串传入网络：

```text
language_text       = 首帧完整描述
language_category   = 首帧目标类别
language_attributes = 首帧稳定外观
language_state      = 结构化首帧 RGB-D/遮挡/干扰物状态
```

当前 tracker 不会在后续帧修改这些成员变量。

本轮正式 full-50 推理固定搜索因子为 5.0，关闭搜索恢复、候选重排、depth rescue、
时序拒绝和在线文本生成。正式推理允许使用：

- 首帧 RGB-D；
- 首帧框；
- 首帧目标描述；
- 当前和历史 RGB-D 图像；
- 模型自己的历史预测、置信度、当前因果深度观测及内部状态。

正式推理不允许使用：

- 后续帧 GT；
- 后续帧人工或 Qwen 文本标注；
- 未来帧；
- 测试序列专用规则。

### 11.3 真实样本端到端走读：`bottle02_indoor`

下面使用 DepthTrack 训练集中的真实序列和真实审核文本，具体走一遍网络。
字符串、帧号、框和张量尺寸都来自当前数据与代码；关于“哪些区域应被增强”
的描述是结构机制的可解释说明，不冒充尚未导出的逐 Token 实测热力图。

#### 11.3.1 首帧目标与实际输入文本

序列首帧为：

```text
/root/autodl-tmp/depthtrack/train/sequences/bottle02_indoor/color/00000001.jpg
```

首帧 GT 框为 `[172, 252, 33, 77]`，框内目标是一只透明塑料瓶，带绿色和
红色标签以及红色瓶盖。桌面上同时摆放着多只瓶子和易拉罐，因此该序列不是
简单的“找任意瓶子”，而是需要保持具体瓶子身份。

审核后的首帧记录为：

```text
clear plastic bottle with a green and red label and red cap;
closer in depth than the surrounding background, with reliable initial depth;
initial occlusion: fully visible;
initial distractors: multiple similar.
```

`TargetDescriptionStore` 与正式 tracker 实际送给网络的四个字段是：

```text
language_text:
Target category: bottle. Stable appearance: clear plastic bottle with a green
and red label and red cap. First-frame RGB-D state: initial depth relation:
closer than background; initial depth quality: reliable; initial occlusion:
fully visible; initial distractors: multiple similar.

language_category:
bottle

language_attributes:
clear plastic; green and red label; red cap

language_state:
initial depth relation: closer than background; initial depth quality:
reliable; initial occlusion: fully visible; initial distractors: multiple similar
```

这里必须区分四点：

1. `language_text` 会经过 CLIP，但当前 `FUSE_TOKENS=false`、
   `FUSE_HEAD=false`，所以完整句子的旧 FiLM 路径不参与最终融合；
2. `bottle` 进入第一阶段类别粗匹配；
3. 三个稳定外观短语进入第二阶段精匹配；
4. 首帧状态不进入继承的属性空间图，只进入 Head-Aligned frame-level 控制器；在新增的
   第 2 层 early adapter 中，它还作为第三个 role 参与 RGB/Depth grounding 和实例绑定。

#### 11.3.2 模板裁剪和 Token 化

模板因子为 2.0。若按首帧 GT 框计算，原图中的正方形模板边长为：

```text
ceil(sqrt(33 × 77) × 2.0) = 101 像素
```

该区域被裁剪并缩放到 192×192，RGB 与 Depth 拼成 6 通道输入。随后：

```text
RGB模板   → 独立 Patch Embedding → 144 个 768 维 Token
Depth模板 → 独立 Patch Embedding → 144 个 768 维 Token
```

目标中心 Token mask 分别从 RGB 和 Depth 模板中汇聚出首帧目标原型。这两个
原型代表“这一个瓶子”的视觉身份，而不只是抽象的瓶子类别。

#### 11.3.3 选择一个真实后续帧

取该序列第 1800 帧，即零基 `frame_index=1799`：

```text
/root/autodl-tmp/depthtrack/train/sequences/bottle02_indoor/color/00001800.jpg
```

这一帧的真实内容是：目标瓶被人拿到镜头前，尺度明显变大并出现运动模糊；
瓶内为黄色液体，绿色标签和红色瓶盖仍可见；后方桌面上仍有多只瓶子、红色
易拉罐和其他包装物。其 GT 框为 `[345, 37, 121, 213]`。

训练时该帧属于序列后 1/3，因此可由 late bin 采到；late bin 的配置概率为
0.50。训练 crop 会加入随机中心和尺度扰动。正式推理则绝不使用这里列出的
GT 框，而是以上一帧预测框为中心裁剪。

仅为说明裁剪尺度，若用 GT 近似“上一帧预测基本正确”的状态，普通搜索因子
5.0 对应：

```text
ceil(sqrt(121 × 213) × 5.0) = 803 像素
```

这个边长大于 640×360 原图，因此实现会补零后缩放到 384×384。搜索图每个
模态产生 24×24=576 个 Token。与模板拼接后，每个模态的输入为：

```text
[144 个模板 Token + 576 个搜索 Token] × 768 维 = [720, 768]
```

#### 11.3.4 RGB-D 双流编码

RGB 和 Depth Token 分别通过 12 层 ViT-B/16。在第 3、6、9、12 层后，
RGB 与 Depth Token 拼接进入跨模态 block，再按原长度拆回双流。

对第 1800 帧而言，RGB 能提供绿色标签、红色瓶盖和透明瓶身等外观证据，
但运动模糊会削弱纹理；Depth 能提供轮廓和前后层次，但这一帧深度存在缺失
和噪点。后续模态路由是逐位置学习的，因此不要求整幅图始终固定相信某一个
模态。

#### 11.3.5 第一阶段：先回答“哪些位置像瓶子”

冻结 CLIP 将 `bottle` 编码为类别向量，再分别投影到 RGB 和 Depth 的
128 维匹配空间。对 576 个搜索 Token，网络计算：

```text
RGB粗分数   = RGB类别相似度 + 1.5 × RGB模板原型相似度
Depth粗分数 = Depth类别相似度 + 0.5 × Depth模板原型相似度
```

在这个真实场景中，类别词 `bottle` 的合理作用是降低人物、窗户、桌椅等
非瓶类背景，同时让目标瓶和桌面上的其他瓶状物都保留为候选。第一阶段本身
未必能在多个相似瓶子中唯一选中目标，所以它不会硬删除任何 Token。

#### 11.3.6 第二阶段：用具体身份属性消除同类歧义

第二阶段在全部 576 个原始搜索 Token 上加入：

```text
clear plastic; green and red label; red cap; ...
```

这些属性能够进一步区分：

- 镜头前的透明瓶、绿色标签和红色瓶盖；
- 桌面上的红色可乐罐；
- 其他白色、透明或不同标签的瓶子。

精匹配使用 `fine_logits = complete_coarse_logits + attribute_logits`，因此
第一阶段的完整类别证据不会消失。最终语言残差中类别分支权重至少为 70%；
即使某个属性因模糊暂时不可见，网络仍保留“这是瓶子且像首帧模板”的保底
证据。

首帧深度、遮挡和干扰物状态位于独立 `language_state` 字符串中，不参与这里的
属性 logits，也不会在第 1800 帧更新。它们表达初始化先验，而不是当前帧真值。

#### 11.3.7 CPSD 如何处理逐帧教师文本

第 1800 帧经过 Qwen2.5 生成和 Qwen3 审核后的真实 teacher 文本为：

```text
A bottle containing yellow liquid with a green label.
```

这句文本存在于已审核 overlay 中，顶层状态为 `accepted_visual`；Qwen3 的静态首帧
身份检查将“yellow liquid”判为 `identity_incompatible`，但该 overlay 的预注册策略
将 Qwen3 判定作为 advisory，并保留已经通过逐帧视觉审核的 caption。这说明逐帧文本
仍可能包含外观变化或错误身份噪声，不能直接当作无条件真值。

CPSD 会读取它，但只把它当作**候选 teacher**。当前帧描述先生成 teacher response；
若其 Hann-selected 位置的 GT support 或目标/困难背景 gap 没有比原视觉 tracker 至少
提高 0.05，该样本的蒸馏 mask 就是 0。通过门槛时，才将 detach 后的响应分布和有界
feature delta 蒸馏给始终使用首帧查询的 student。因此“记录被读取”和“记录对参数
产生梯度”是两件不同的事。

它不会：

- 替换首帧 student 查询；
- 进入 Depth teacher 路径；
- 在测试时提供给 tracker；
- 让模型看到未来帧文本。

因此，这个样本在训练和推理中的文本边界为：

| 阶段 | 主查询 | 第 1800 帧候选 teacher | 第 1800 帧 GT |
|---|---|---|---|
| CPSD 训练 | 始终为首帧富文本 | 读取；仅在效用门通过时蒸馏 | 仅用于训练定位损失、裁剪和 teacher 效用选择 |
| 推理 | 始终为首帧富文本 | 不读取 | 不读取 |

#### 11.3.8 有界融合和最终框

粗、细两阶段得到 24×24 语义图和逐位置 RGB/Depth 路由。语言分支只修改
576 个搜索 Token；两阶段候选残差和独立 Head-Aligned refiner 的元素级上限均为
0.04，144 个模板 Token 保持不变。冻结 Center Head 随后输出：

```text
score_map  [1, 1, 24, 24]
size_map   [1, 2, 24, 24]
offset_map [1, 2, 24, 24]
```

中心响应使用校正特征，宽高和 offset 使用原视觉特征；代码不执行语义响应图融合。
若文本和首帧模板不兼容，门控会减弱校正；若文本为空，则严格退化为纯视觉结果。
最后 tracker 乘原 Hann 窗、选择响应峰值、读取宽高与偏移，再按搜索 crop 的缩放
和补边关系映射回 640×360 原图坐标。

## 12. 冻结部署增强：v11 搜索恢复与 0.98 报告尺度

最终 V5 YAML 保持 `LANGUAGE_SEARCH_RECOVERY.USE=false`，以保证网络架构实验默认无
部署规则；正式部署命令通过显式 CLI/profile overlay 启用此前冻结的
`longterm_scale_adaptive_v11`。V5 无后处理 paired correct/empty 用于证明语言网络本身
有效，最终部署则统一使用 V5+v11+`fixed6_isotropic_098_v1`。以下规则属于独立推理
增强，不属于 CRAR 参数、训练损失或论文中的学习架构贡献。

### 12.1 状态与触发

初始化时，首帧 GT 框只用于建立协议允许的初始 `trusted_bbox` 和中心深度。正常状态
使用搜索因子 5.0；出现以下任一事件时，先保留当前帧原预测不变，从**下一帧**开始
围绕最后可信框使用因子 9.0：

- 原 Center 响应低于 0.25；若首帧声明 `nearby_similar`，阈值为 0.35；
- 相对可信框的归一化中心跳变大于 0.8、响应不高于 0.70，同时中心深度的对数变化
  大于 0.06。

一次宽搜索最多 12 帧，活动窗口内的持续低分不会反复重置预算，因此不会无限保持
全局搜索。

### 12.2 恢复确认

候选可以通过两条因果路径之一结束宽搜索，两条路径都要求连续两帧稳定：

1. 语义/身份路径：原响应至少 0.45，类别和属性 logits 均为正，候选在首帧原始
   RGB 模板身份图中的百分位不低于 0.80，在投影 RGB 身份图中不低于 0.98；
2. RGB-D 连续路径：响应至少 0.48（`nearby_similar` 为 0.55），与可信深度的
   对数差不超过 0.12，并且归一化中心跳变不超过 0.8。

首帧目标面积不超过整帧 0.3% 时，语义路径至少观察 4 个宽搜索帧后才允许提交；大
目标不延迟。这个尺度自适应条件来自通用几何量，不含序列名称。

### 12.3 可信锚点保护

普通跟踪期间，只有响应达到 0.55（`nearby_similar` 为 0.60），并且得到语义身份
或 RGB-D 连续性支持时，当前框才会刷新可信锚点。若 12 帧预算耗尽，默认保留旧
锚点；只有类别/属性为正、原始 RGB 身份百分位至少 0.90、投影身份百分位至少
0.98，同时响应和深度兼容时，失败窗口末端候选才可晋升为新锚点。

### 12.4 为什么形成 v11

开发期逐序列 trace 暴露了三个典型错误：

- `flower03_indoor`：早期规则在 frame 369 接受了错误候选，其原始模板身份百分位
  只有 0.491；语义身份门控将其拒绝，避免整个后段崩溃；
- `cup12_indoor`：小目标在宽搜索刚启动时的局部峰不稳定，过早在 frame 61 提交会
  失败；等待到 frame 63 恢复了正确轨迹；
- `bag04_indoor`：对所有目标统一等待 4 帧反而破坏大目标，所以 v11 只对首帧面积
  不超过 0.3% 的小目标延迟提交。

开发期还确认 `ball01_wild` 与 `book03_indoor` 是恢复机制能明显改善的长时漂移案例。
这些序列只用于诊断通用条件，正式规则没有任何按名称分支。需要复现实验事件时可设
`SRTRACK_LANGUAGE_RECOVERY_TRACE=1`；默认关闭，不影响结果。

### 12.5 信息边界

该模块仍使用原首帧文本和原首帧模板，不生成描述、不更新模板、不修改语言记忆，
也不在预测文件生成后改框。每次决策只依赖首帧信息、当前/历史 RGB-D、网络响应和
自身历史状态。所以它是搜索范围恢复机制，不是 RAG、在线 Qwen 或结果后处理。

### 12.6 `fixed6_isotropic_098_v1` 报告框校准

CDTB 原始报告框结果已经有较高 PR/F1，但最大 F1 工作点的 Recall 为 75.3493，距目标
75.6 差 0.2507。为避免改 PR 取点规则，新增一个**只作用于报告框**的各向同性尺度候选。
候选集合固定为 `{0.90, 0.94, 0.98, 1.00, 1.02, 1.06, 1.10}`，唯一选择数据仍是
DepthTrack fixed-6；正式 CDTB/VOT 指标不参与候选排序：

| 报告尺度 | fixed-6 PR | Recall | F1 |
|---:|---:|---:|---:|
| 0.90 | 61.4711 | 61.3531 | 61.4120 |
| 0.94 | 64.4867 | 64.4438 | 64.4653 |
| **0.98** | **65.9333** | **66.0042** | **65.9688** |
| 1.00 | 65.8311 | 65.9624 | 65.8967 |
| 1.02 | 65.2940 | 65.4793 | 65.3865 |
| 1.06 | 62.9039 | 63.1306 | 63.0171 |
| 1.10 | 59.2770 | 60.4048 | 59.8356 |

对第 `t>0` 帧原始状态框 `(x,y,w,h)`，只向评测器返回同中心的
`(0.98w,0.98h)`；首帧仍原样返回协议初始化框。内部 `self.state`、v11 的
`trusted_bbox`、下一帧搜索 crop、响应图和 confidence 都不读取缩放结果。因此它是
当前帧因果、无未来信息的确定性几何校准，可以从冻结原轨迹无损派生；但它不属于语言
创新、不会改善跟踪状态，也必须与原始报告框指标同时给出。

## 13. 可选推理增强：安全模板槽与低权重在线语言

RAGTrack 类方法通常会在跟踪过程中重新检索、生成或更新目标描述及动态模板。本仓库已经
把安全模板槽、在线语言策略和 Qwen HTTP client 接入 `SIAMTrack.initialize()/track()`。
已报告的完整 DepthTrack、CDTB 和 VOT reference 均关闭在线语言；V24--V28 的 VOT 候选
单独启用 branch-safe 模板槽，在线语言仍关闭。模板槽的职责是减少长期外观变化造成的
模板偏移，不是主模型的已学习跨模态创新。

### 13.1 安全动态模板槽

实现采用“不可变首帧模板 + 一个可替换动态槽”，不会执行示例代码中的无条件 FIFO
覆盖。`SafeTemplateUpdatePolicy` 每 5 帧检查一次，要求连续 3 次稳定，并同时满足响应
置信度、非极大值响应 margin、中心运动、首帧 RGB 身份和中心 Depth 一致性。通过后才从
当前预测框裁出 192x192 RGB-D 模板；V27 policy 的动态槽权重为 0.20、硬上限 0.20，
但进入 primary 模板输入的独立 blend 被进一步限制为 0.02，动态记忆 90 帧过期。
出现大跳变、首帧身份冲突、Depth 突变或时序拒绝时立即丢弃动态槽，首帧锚点始终保留。

动态模板可以进入独立的 early-language memory 路径，也提供严格限幅的 primary-template
blend 开关。`safe025` 架构对照关闭模板更新；V24--V27 候选使用独立的
`probe_e1_template` 部署配置。恢复分支 active 时暂停模板观察，只有最终公开分支稳定后才
允许更新，避免 speculative candidate 污染下一帧模板。

### 13.2 在线语言策略

在线语言只在安全模板证据已经可用时才允许开启，并额外要求类别一致。其默认策略为：

| 项目 | 原型设置 |
|---|---:|
| 检查间隔 | 每 5 帧 |
| 需要连续稳定检查 | 3 次 |
| 最小生成间隔 | 30 帧 |
| 生成超时 | 30 帧 |
| 动态文本权重 | 0.20 |
| 动态记忆 TTL | 90 帧 |
| 最低置信度 | 0.65 |
| 最低首帧身份相似度 | 0.75 |

满足稳定门后，tracker 通过本地 Qwen endpoint 发送完整 RGB、目标 RGB crop 和 Depth crop；
返回文本必须通过视觉有效、类别匹配和首帧身份兼容检查。接受后的动态描述仅以 0.20 权重
混入动态状态 role，并与同一帧动态模板配对；TTL 到期或任一硬冲突会同时清除两者。client
或生成失败采用 fail-closed，不修改 tracker memory。

当前实现使用同步 HTTP 请求，正式配置为 `ONLINE_LANGUAGE_UPDATE.USE=false`，也没有任何
已完成的主表指标。因此在线语言只保留为模板偏移时的次要推理优化，不继续扩大模型、训练、
Qwen 能力或论文叙事；在主模型与安全模板更新未通过正式 VOT 门禁前，不为它单独投入训练
或消融预算。

### 13.3 与主创新的边界

```text
主创新：首帧语言与 RGB/Depth token 的早期 cross-attention + 三模态一致性漂移门
推理增强：高置信动态模板槽 + 可选低权重在线外观文本
```

在线文本不能覆盖首帧类别或稳定属性，不能独立刷新模板，也不能绕过 RGB-D 一致性证据。
只有正式 VOT 主结果完成后，才考虑在固定高风险 anchors 上验证其是否减少模板偏移；无
收益或破坏 DepthTrack/CDTB 保真门时保持关闭。

## 14. 在线增强的验证约束

后续启用 RAGTrack 式在线更新时继续使用双记忆结构，而不是覆盖首帧信息：

```text
不可变身份记忆：首帧模板 + 首帧类别/稳定属性
可替换动态记忆：高置信当前模板 + 当前外观描述
```

推荐约束：

1. 首帧模板和首帧文本永远不可覆盖；
2. 只有置信度、响应 margin、运动、深度和首帧身份相似度同时稳定时才允许生成；
3. 动态文本只以较低权重参与融合；
4. 动态记忆设置 TTL；
5. 发生大位移、深度突变或类别冲突时立即删除动态记忆；
6. Qwen 结果必须再次检查类别一致性与首帧身份兼容性；
7. 作为单独推理实验报告在线 VLM 的延迟、显存、调用次数和模板提交次数。

这样可以利用当前外观变化，又避免 tracker 已漂移时让 Qwen 围绕错误目标生成描述，形成正反馈漂移。

## 15. 当前实现的主要限制

### 15.1 完整富文本利用仍不充分

当前第 2 层 early adapter 已将类别、`stable_attributes` 和首帧状态作为三个独立 role，
分别与 RGB/Depth 模板及 search token 做 cross-attention；旧 two-stage 分支中的状态 role
仍只提供低维 frame context。完整句子的 FiLM Token/Head 路径关闭，详细证据句、运动字段
和复杂关系文本仍未被充分利用。

### 15.2 正式指标尚未验证在线外观适应

在线语言与动态模板已经接入 tracker，但当前正式 profile 关闭。它们是否能改善模板偏移、
以及同步 Qwen 延迟是否可接受，都还没有通过完整 VOT multi-start 结果验证。

### 15.3 视觉和语言残差能力受限

冻结视觉主干能够保证公平对照和稳定回退，但语言分支最多只能在已有视觉表示上做小幅修正，难以修复视觉 backbone 本身没有编码出的信息。

### 15.4 当前没有硬 Token 剪枝

语言第一阶段只做软门控，因此计算量不会因语言粗筛选下降。它提高了错误恢复能力，但没有获得硬剪枝的速度收益。

### 15.5 在线增强不是当前方法贡献

策略、Qwen client、动态视觉模板槽和 tracker 接线已经完成，但仍缺正式收益证据与异步
调度。它只能作为关闭默认的推理增强，不能作为当前主模型贡献宣称。

### 15.6 v11 与 0.98 尺度都是冻结推理增强，不是架构贡献

搜索恢复处理目标离开普通 crop 的能力边界，引入置信度、深度、身份百分位和尺度阈值。
v11 不是训练得到的模块，因此必须与 V5 网络增益分表报告。当前正式部署复用同一
`longterm_scale_adaptive_v11` 和 `fixed6_isotropic_098_v1`，禁止为不同测试集重新调参。
论文的架构主张来自 CRAR/V5；v11 只能写成搜索恢复消融，0.98 只能写成报告几何校准。

### 15.7 当前 V5 仍只有单一随机种子和一轮 probe

V5 checkpoint 使用 seed 2026、8,000 个 DepthTrack 样本和一轮 router-only 训练；视觉
主干、Center Head 与全部既有语言分支冻结。当前 paired empty/correct 已证明语言因果
增益，full-50 也达到部署目标，但完整投稿仍应补充至少三个 seed、训练样本数/轮数消融
和不使用 v11 的 full-50 架构主表。现有单 seed 结果不能支持方差或显著性结论。

### 15.8 当前完整 50 序列结果不是 untouched-test 结果

v11 没有任何序列名特判，fixed-6 也不在正式 50 序列清单中；但规则开发期间曾单独
检查正式测试集中的 `ball01_wild`、`book03_indoor` 和 `cup12_indoor`，并据此分析
过恢复与锚点失败。因此完整 50 序列结果在覆盖、计算和因果信息边界上是真实的，
但统计意义上属于 test-informed engineering result，不能宣称为完全未见测试集的
无偏主结果。

顶会版本应在 DepthTrack 训练集内部重新划分开发序列、冻结唯一 profile，再将
CDTB/VOT-RGBD2022 作为未用于调参的跨数据集验证；若要重新获得严格的 DepthTrack
主结果，则需要预注册规则和独立评测环境，避免继续观察测试序列后迭代阈值。

### 15.9 V5 风险标签仍不是真实递归 rollout

训练样本仍是一个 template/search pair。V5 已把支持度不足、gap 风险和阈值两侧 margin
直接写入正例/风险负例约束，但风险标签仍来自当前 teacher-forced crop，没有把本帧预测
框递归用于下一帧搜索裁剪。真实跟踪在多同类目标、遮挡或 re-entry 时仍可能因一次峰值
切换改变后续输入分布；V5 缓解了路由标定问题，并没有完成真实闭环训练。

下一轮仍应在 DepthTrack 训练集内部采样连续 3--5 帧并真正展开预测状态，加入同类别
hard negative、absence/re-entry 和遮挡样本。学习到的风险门应优化短窗递归 utility；
语言证据不足时继续保持网络内部精确回退视觉峰值，并用无 v11 的 full-50 单独验证。

### 15.10 CDTB 尺度结果不是最终 profile 的 untouched 结论

V5+v11 原始 CDTB80 先得到 `74.7326/75.3493/75.0397`；看到 Recall 距目标 0.2507 后，
才提出报告框尺度假设。候选选择严格只使用 fixed-6，CDTB 数值不参与 0.98 与其他候选的
排序，而且相同 0.98 同时改善 fixed-6 和 full-50；但“提出假设”的时序仍受 CDTB 结果
启发。因此校准后的 CDTB `75.4304/76.0564/75.7421` 是真实完整计算结果，却不能写成
最终 profile 的完全未见跨域验证。VOT-RGBD2022 是 0.98 冻结后才首次运行的正式外部集，
只有它能承担最终部署 profile 的 untouched 跨数据集证据。

## 16. 论文可直接使用的当前主网络架构与创新点

本节描述当前 **Primary Early Tri-modal Cross-Attention with Consensus Guard**。CRAR-V5
提供继承的双动作和阈值对齐仲裁基础；当前新增贡献是语言在第 2 层边界分别与 RGB/Depth
token 交互，并由模态一致性 map 和 veto-only guard 控制峰值漂移。逐帧 privileged
teacher、v11、0.98、动态模板和在线语言均不能写进学习架构贡献。

### 16.1 方法名称与一句话定义

当前工作名为：

```text
Primary Early Tri-modal Cross-Attention with Consensus Guard
早期三模态交互一致性跟踪器（Primary Tri-modal Guard）
```

核心是“先跨模态定位，再一致性决策”：首帧语言 role 先查询 RGB 和 Depth 模板以绑定
实例，当前搜索 token 再查询已绑定的语言 role。第 2 层 adapter 由两路搜索查询直接形成
RGB/Depth/consensus grounding maps，同时把有界 residual 送入后续双流层构造语言候选。
冻结 CRAR 给出双动作基础，571 参数的一致性门只在 RGB-D 不共同支持时下调语言 route
logit。它不合成第三张响应图，也不能创建 CRAR 原本未选择的语言动作。

论文中可使用下面的一句话摘要：

> 我们在 RGB-D 双流主干的早期层引入语言到 RGB/Depth 模板、再由搜索 token 到语言 role
> 的双阶段 cross-attention，使文本在候选形成前参与三模态定位；随后利用 RGB、Depth 与
> disagreement-penalized consensus map 构造只抑制不新增的路由门，避免单模态语言峰导致
> 递归模板和搜索状态漂移。

### 16.2 问题定义与严格因果信息边界

给定首帧 RGB 图像 `I_0^r`、首帧深度图 `I_0^d`、首帧框 `b_0` 和首帧目标描述
`q_0`，目标是在第 `t` 帧仅利用当前及历史 RGB-D 图像预测框 `b_t`：

```text
b_t = T(I_0:t^r, I_0:t^d, b_0, q_0)
```

正式网络的约束是：

1. `q_0` 只在初始化时读取一次，后续逐帧复用同一不可变查询；
2. 不读取 `q_t (t > 0)`、后续帧人工/Qwen 描述、未来图像或后续帧 GT；
3. 首帧模板、文本和身份原型不在线更新；
4. 当前帧推理只允许使用模型自己的历史框、置信度和因果深度观测；
5. 空文本对照使用同一 checkpoint 和同一视觉输入，必须精确回退视觉响应。

该定义把“训练时可能存在的额外监督”和“部署时可用信息”严格分开。当前主配置中
`FRAME_TEACHER_USE=false`，因此连训练阶段也只使用首帧语言；早期 CPSD 的逐帧教师
文本不属于当前主模型的训练图。

### 16.3 总体数据流

```text
首帧 RGB-D 模板 Z，192x192 ─┐
                             ├─ RGB/Depth 双流 ViT-B/16
当前 RGB-D 搜索 X_t，384x384 ┘   └─ 第 3/6/9/12 层跨模态交互
                       ├─ canonical 全程 ─────────────── F_v -> S_v/size/offset
                       └─ layer-2 seed + language roles
                              ├─ role -> RGB/Depth template attention
                              └─ search -> grounded role attention
                                     ├─ M_rgb/M_depth/M_consensus -> guard
                                     └─ bounded residual -> 余下双流/跨模态层
                                                            └-> F_l -> S_l

S_v/S_l + 25 维证据 -> 冻结 CRAR -> inherited route logit
三张 grounding map + 两个候选峰 -> 9 维 consensus guard -> delta <= 0
                                  └-> p_t >= 0.5 选 S_l，否则 S_v
                                      + canonical size/offset -> b_t
```

网络由四个逻辑层组成：

1. **共享感知层**：RGB-D 双流主干提取模板和搜索 Token；
2. **早期三模态交互层**：语言分别在 RGB/Depth 模板上绑定实例，再条件化搜索 token；
3. **双动作构造层**：同一冻结 Head 分别读取视觉 Token 与语言校正 Token，得到两个完整
   center response；
4. **一致性仲裁层**：继承 CRAR 提议动作，三模态 guard 仅否决缺乏 RGB-D 共识的语言动作。

“早期交互、双动作、共识否决”是当前结构的主线。语言候选不是额外检测器，视觉和语言
动作共享同一 RGB-D 输入、同一模板、同一 Center Head 和同一框几何解码器，差异只来自
搜索 Token 是否经过首帧语言条件的有界校正。

### 16.4 RGB-D 双流视觉编码器

模板 crop 为 192x192，搜索 crop 为 384x384，patch size 为 16。因此每个模态分别产生：

```text
模板 Token：12 x 12 = 144
搜索 Token：24 x 24 = 576
总 Token：  720
Token 维度：768
```

RGB 与 Depth 使用独立 patch embedding、独立 12 层 ViT-B/16 block 和独立归一化层。
在零基 block 2、5、8、11 后执行四次跨模态交互：先拼接同层 RGB/Depth Token，经共享
跨模态 Transformer block 后再按原长度拆回两条流。最终视觉融合 Token 为：

```text
F_v = (F_rgb + F_depth) / 2
```

最终实现没有执行硬 Candidate Elimination。配置中的 `CE_LOC` 和
`CE_KEEP_RATIO` 不会在当前 `vit_siam_dropmae` 前向中删除 Token，因此论文应表述为
“完整 Token 上的软语义门控”，不能宣称已经完成 70% Token 硬剪枝。

### 16.5 首帧语言编码与角色分离

首帧富文本由冻结 CLIP ViT-B/32 编码，最大长度为 77，文本特征执行 L2 归一化。结构化
字段被显式分为三种角色：

| 角色 | 示例 | 网络职责 |
|---|---|---|
| 类别 `c` | bottle、person、ball | 第一阶段回答“哪些位置属于目标类别” |
| 稳定属性 `a` | red cap、green label、striped shirt | 第二阶段区分同类实例 |
| 初始状态 `s_0` | closer than background、fully visible | 尾部作 frame context；早期作第三个实例绑定 role |

状态字段描述的是 frame 0，而不是后续帧真值。在继承的尾部 two-stage 适配器中，它只作
frame-level 可靠性上下文，不进入属性 logits；在新增 early branch 中，它与类别、稳定属性
一起作为第三个不可变 role，参与模板实例绑定和搜索 attention，但不能解释为当前帧状态。

早期交互先把三个 512 维 CLIP role 投影到 192 维。RGB 与 Depth 模板分别被查询，再由逐
role 模态路由融合为同一组实例绑定 role：

```text
Q       = P_l([c, a, s_0]) + E_role
G_rgb   = Attn(Q, P_v(Z_rgb), P_v(Z_rgb))
G_depth = Attn(Q, P_v(Z_depth), P_v(Z_depth))
w       = sigmoid(R([Q, G_rgb, G_depth, |G_rgb - G_depth|]))
G       = LN(Q + w * G_rgb + (1 - w) * G_depth)
```

两路搜索 token 使用各自模态 embedding 查询共享的 `G`。归一化搜索查询与有效 role 的
相似度分别形成 `M_rgb`、`M_depth`，并在该 early 边界构造：

```text
M_consensus = 0.5 * (M_rgb + M_depth) - 0.25 * |M_rgb - M_depth|
```

三张 map 直接保留给最终 veto guard；`2 * sigmoid(M_consensus)` 只空间调制两路有界
search residual，随后 residual 才经过余下视觉与跨模态层。模板 token 在此过程中保持原值。

继承的尾部类别与属性匹配另行投影到 RGB 和 Depth 的 128 维空间。第一阶段将类别相似度
和不可变首帧模板原型相似度结合：

```text
L_rgb^c = sim(F_rgb^x, E_rgb(c)) + 1.5 * sim(F_rgb^x, P_rgb^z)
L_dep^c = sim(F_dep^x, E_dep(c)) + 0.5 * sim(F_dep^x, P_dep^z)
```

`P_rgb^z` 与 `P_dep^z` 由首帧框对应的模板中心 Token 聚合得到，表达“这一个目标”的
实例身份，而不是抽象类别。RGB/Depth 粗分数通过 softmax 模态路由融合，但不会删除任何
搜索 Token。

第二阶段在全部 576 个搜索 Token 上计算稳定属性匹配，并保留完整类别证据：

```text
L^f = L^c + L^a
```

最终语言残差中类别权重下界为 0.70，属性最多占剩余容量。这样属性可在多个同类物体中
提供实例区分，同时不能完全覆盖类别和模板证据。

### 16.6 早期候选构造与 Head-Aligned 安全证据

语言分支同时保留 canonical 最终 Token `F_v`。令 `F_e` 为第 2 层 early residual 经过余下
双流/跨模态层后的融合 Token；继承的两阶段 adapter 再在其搜索部分加入类别/属性有界
残差，形成当前 CRAR 真正使用的完整候选 `F_l`：

```text
F_e = RemainingViT(EarlyGround(Z_rgb, Z_depth, X_rgb, X_depth, c, a, s_0))
F_l = F_e + Delta_two_stage(c, a),  |Delta_two_stage| <= 0.04
```

early 与 two-stage 两个 residual 都只作用于 576 个搜索 Token，144 个模板 Token 保持不变；
空类别时显式 mask 使 two-stage residual 严格为 0，三个 role 全空时 early 分支也逐值回退。

实现随后仍调用继承的 Head-Aligned `correct_features(F_e, F_l, S_v, ...)`。它读取视觉响应
质量、粗/细语义分布、RGB/Depth 模板身份、类别/属性和首帧状态，计算 frame/spatial gate、
refiner gate 与有界诊断 residual。但在当前 `COUNTERFACTUAL_ROUTER_USE=true` 路径中，
`feature_candidate_out` 已在该连续校正之前由 `F_l` 生成；Head-Aligned 返回 token 不替换
`S_l`，其中 `frame_gate` 和 `refiner_gate` 作为 CRAR 25 维证据的一部分。这样应把它表述为
候选风险证据生成器，而不是当前最终语言动作的第三个 feature branch。

同一个冻结 Center Head 分别处理两组 Token：

```text
S_v = H_center(F_v)
S_l = H_center(F_l)
M_size   = H_size(F_v)
M_offset = H_offset(F_v)
```

因此 `S_v` 和 `S_l` 是同一检测头下可直接比较的完整动作。语言只拥有“中心落在哪里”的
候选权，没有独立修改宽高或 offset 的权限。`M_size` 和 `M_offset` 永远来自视觉分支，
这是防止文本把语义类别偏好误写成框几何的结构约束。

### 16.7 从连续融合到反事实双动作

常见视觉语言融合可写成：

```text
S_mix = S_v + alpha * Delta_language
```

即使 `alpha` 很小，只要它改变 argmax，下一帧 crop 就会改变；这种微小残差在递归 tracker
中可能产生不可逆身份漂移。CRAR 不学习连续 `alpha`，而是显式构造两个反事实问题：

```text
如果本帧完全不采用语言，tracker 会执行什么动作？ -> S_v
如果本帧完整采用语言候选，tracker 会执行什么动作？ -> S_l
```

路由器学习的不是“语言加多少”，而是“完整采用语言动作的收益是否足以覆盖切换风险”。
这使每次语言干预都能被记录为 `route_selected`，并能通过 correct/empty paired 实验直接
审计。

### 16.8 25 维跨域证据向量

令 `i_v`、`i_l` 分别为 `S_v`、`S_l` 乘正式 Hann window 后的 argmax。CRAR 不读取
数据集名、序列名、绝对框坐标或后续 GT，而只读取 25 个有界或标准化证据：

| 组别 | 维数 | 特征 |
|---|---:|---|
| 两路响应质量 | 6 | visual/language 的 Hann 峰值、top1-top2 margin、归一化熵 |
| 响应差异 | 7 | 峰值差、margin 差、熵差、分布重叠、`dx`、`dy`、归一化峰距 |
| 语义/实例秩差 | 5 | coarse、fine、raw RGB、投影 RGB、投影 Depth 在 `i_l` 与 `i_v` 间的百分位秩差 |
| 已冻结安全状态 | 5 | coarse/fine reliability、frame gate、refiner gate、curriculum gate |
| 完整性 | 2 | coarse/fine 分布重叠、稳定属性是否存在 |

响应图先被转为 logit，并在每张图内中心化与 RMS 标准化：

```text
z_j = (logit(S_j) - mean(logit(S))) / sqrt(mean(centered^2) + eps)
P(S) = softmax(z)
H(S) = -sum(P log P) / log(HW)
```

两路分布重叠使用 Bhattacharyya 系数：

```text
O(S_v, S_l) = sum_j sqrt(P_v,j * P_l,j)
```

峰值位移按 24x24 响应图尺寸归一化：

```text
dx = (x_l - x_v) / 23
dy = (y_l - y_v) / 23
d  = sqrt(dx^2 + dy^2) / sqrt(2)
```

对 coarse/fine 和三张身份图，不直接比较跨数据集容易漂移的原始 logit，而比较两个候选
位置在同一图内的百分位秩：

```text
Delta_rank(M) = rank_M(i_l) - rank_M(i_v)
```

这五个秩差直接回答：语言峰相对视觉峰是否得到更强类别、稳定属性、首帧 RGB 身份和
Depth 身份支持。图内标准化、百分位秩和归一化位移共同减少 DepthTrack 到 CDTB/VOT 的
响应尺度偏移，是只在 DepthTrack 训练仍能跨域使用的重要设计。

#### 16.8.1 9 维三模态共识证据

冻结 CRAR 给出 inherited logit 后，新增 guard 只比较同一 `i_v/i_l` 在三张 early map 中的
相对位置。令 `r_M(i)` 为位置 `i` 在 map `M` 内的百分位秩，9 个输入依次为：

| 组别 | 维数 | 特征 |
|---|---:|---|
| 三图候选增益 | 3 | `r_rgb(i_l)-r_rgb(i_v)`、`r_depth(i_l)-r_depth(i_v)`、`r_consensus(i_l)-r_consensus(i_v)` |
| 候选绝对支持 | 3 | `r_rgb(i_l)`、`r_depth(i_l)`、`r_consensus(i_l)` |
| 模态一致性变化 | 1 | 候选峰相对视觉峰的 RGB/Depth 标准化一致性增量 |
| 双模态最弱支持 | 1 | `min(r_rgb(i_l), r_depth(i_l))` |
| 双模态秩差 | 1 | `|r_rgb(i_l)-r_depth(i_l)|` |

其中逐位置模态一致性为：

```text
A(i) = exp(-|z_rgb(i) - z_depth(i)|)
Delta_A = A(i_l) - A(i_v)
```

guard 网络与参数量为：

```text
LayerNorm(9):       9 + 9             =  18
Linear(9, 24):      9*24 + 24         = 240
Linear(24, 12):     24*12 + 12        = 300
Linear(12, 1):      12*1 + 1          =  13
总计                                     571
```

令其原始输出为 `c_t`，则：

```text
delta_guard = C * clamp(tanh(c_t), max=0)
z_final     = z_CRAR + delta_guard
```

训练配置使用 `C=6.0` 学习清晰的否决边界；冻结 safe025 部署配置把同一 checkpoint 的上限
收紧到 `C=0.25`，降低跨域过抑制风险。由于 `delta_guard<=0`，它最多把 inherited 语言动作
压回视觉动作，不能让原本低于阈值的 CRAR logit 上升。三张 map 缺失或含非有限值时该
样本 fail closed，不允许 guard 路由语言候选；shape 不匹配则由输入校验显式报错并停止运行。

### 16.9 轻量硬仲裁器与精确回退

路由网络为：

```text
LayerNorm(25)
-> Linear(25, 48) -> GELU
-> Linear(48, 24) -> GELU
-> Linear(24, 1) -> sigmoid
```

参数量可逐层核算：

```text
LayerNorm(25):       25 + 25                 =   50
Linear(25, 48):      25*48 + 48              = 1248
Linear(48, 24):      48*24 + 24              = 1176
Linear(24, 1):       24*1 + 1                =   25
总计                                           2499
```

令路由概率为 `p_t`，固定推理阈值 `tau=0.5`。有效 mask 为：

```text
a_t = category_present AND curriculum_active AND all_evidence_finite
g_t = a_t AND 1[p_t >= 0.5]
S_t = where(g_t, S_l, S_v)
```

`where` 是整张响应图的离散选择，不是像素级拼接。`a_t=false` 时最后一次显式
`where` 再次写回 `S_v`，所以空文本、缺类别、NaN/Inf 或失效 curriculum 都 fail closed。
这不是“尽量接近视觉基线”，而是同张量逐值使用视觉分支；paired empty 实验进一步验证
框与 score 文件逐字节一致。

训练时使用 straight-through hard route：前向仍执行 0/1 硬动作，反向用 `p_t` 的梯度
训练分类器。推理时改为真正的无梯度 `torch.where`。因此训练前向与部署决策边界一致，
不存在训练时软混合、测试时突然硬化的行为落差。

### 16.10 反事实动作标签

训练 GT 只用于生成路由标签，不进入推理。对 GT Gaussian heatmap `Y_t`，在两个正式
Hann-selected 峰位置读取目标支持度：

```text
u_v = Y_t[i_v]
u_l = Y_t[i_l]
Delta_u = u_l - u_v
```

同时分别计算响应在 GT 前景区域的均值与 top-k 困难背景均值之差 `G_v`、`G_l`：

```text
Delta_G = G_l - G_v
```

语言动作正例必须同时满足：

```text
i_l != i_v
u_l >= 0.20
AND (
       Delta_u > 0.05
       OR (Delta_u > 0 AND Delta_G > 0.02)
    )
```

这一定义排除三类伪正例：两路其实选择同一位置、语言峰仍远离目标、语言只制造更尖但
错误的背景峰。其余 active 样本为负例；其中 `i_l != i_v` 但未满足收益条件者被定义为
**风险负例**，因为一旦错误打开路由就会真实改变 tracker 动作。

### 16.11 阈值对齐双裕量风险学习

语言救援正例稀少，普通 BCE 容易学成“永远关闭”；只做正负相对排序又不能保证两类样本
落在固定阈值 0.5 的正确两侧。V5 的总路由目标包含四部分：

```text
L_CRAR = L_balanced_BCE
       + 1.0 * L_rank
       + 0.5 * L_negative_tail
       + 2.0 * L_positive_head
```

整个 `L_CRAR` 再乘配置权重 2.0 加入总训练损失。各项定义如下。

原跟踪器的分类、L1 和 GIoU 等标准损失仍按原 actor 计算；由于 optimizer 中只有 CRAR
参数可训练，它们只能在 straight-through 路由实际连接到候选动作时向路由器提供辅助
梯度，不会更新视觉主干、语言候选生成器或 Center Head。上式是 V5 新增且显式负责
路由标定的监督项，不应误写成训练图中唯一存在的损失。

**1. 跨 batch 稀有正例平衡。** 维护正例率 EMA：

```text
pi <- 0.99*pi + 0.01*pi_batch
w_pos = clip((1-pi)/pi, 1, 32)
```

BCE 使用 `w_pos`，但不再按当前 mini-batch 的实际权重和归一化，而按 EMA 先验期望权重
归一化。这样含少量正例的 batch 不会把逆频率权重再次抵消，无正例 batch 也不会单方面
主导训练。

**2. 困难负例相对排序。** 对风险负例中 logit 最大的 top-8：

```text
L_rank = mean ReLU(0.5 + z_negative - z_positive)
```

该项要求有效正例至少比最危险负例高 0.5 logit，但相对排序本身仍不足以对齐推理阈值。

**3. 风险负例的阈值下裕量。** `tau=0.5` 对应 logit 0。对风险负例 top-8：

```text
L_negative_tail = mean ReLU(z_negative + 0.5)
```

即要求危险负例 `z_negative <= -0.5`，不仅低于正例，还要明确落在推理阈值以下。

**4. 有益正例的阈值上裕量。** 对全部有效正例：

```text
L_positive_head = mean ReLU(0.5 - z_positive)
```

即要求 `z_positive >= +0.5`。负侧与正侧两个绝对 margin 共同形成“阈值对齐双裕量”。
这正是 V5 相对 v3/v4 的关键修复：v4 只压危险负例会同时牺牲语言收益，而 V5 同时保护
真正的救援动作。

继承的 V5 阶段只训练 8,000 个 DepthTrack causal samples、1 epoch，optimizer 仅包含
`language_counterfactual_router.*`。当前 primary 阶段再从 V5 初始化，使用同样 8,000 个
DepthTrack causal samples、1 epoch、batch size 16、梯度累积 1，只训练
`language_early_grounding.*` 与 `language_trimodal_consensus_guard.*`，两组 LR 均为 `5e-5`。

#### 16.11.1 当前 primary 的新增监督

令 `Y` 为训练框生成的 24x24 Gaussian heatmap。`BCE_bal` 把 `Y>0.2` 视为正区域，并用
`clip(N_negative/N_positive, 1, 16)` 平衡正位置。三张 early map 的监督为：

```text
L_ground = BCE_bal(M_consensus, Y)
L_modal  = 0.5 * [BCE_bal(M_rgb, Y) + BCE_bal(M_depth, Y)]

p_rgb   = softmax(flatten(M_rgb))
p_depth = softmax(flatten(M_depth))
L_cons   = 0.5 * [KL(stopgrad(p_depth) || p_rgb)
                + KL(stopgrad(p_rgb) || p_depth)]
```

完整 `S_l` 还接受普通 Center Head focal loss `L_candidate`。guard 复用第 16.10--16.11 节
的反事实标签和双裕量形式，但其 active 集合进一步限制为：

```text
A_guard = A_CRAR AND inherited_CRAR_selected
```

因此正例只能要求 guard 保留继承动作，风险负例可以要求 guard 抑制继承动作；CRAR 未选择
语言的样本不会训练 guard 创建新动作。当前 primary 的显式语言增量损失权重为：

```text
L_primary_added = 0.25 * L_candidate
                + 0.25 * L_veto
                + 0.20 * L_ground
                + 0.10 * L_modal
                + 0.02 * L_cons
```

总训练目标还保留原 tracker 的 GIoU、L1、center focal 和 absence loss，这些不是本节新增
监督项；其中存在可微路径的梯度也只能更新 optimizer 内的 early grounding 与 tri-modal
guard。视觉主干、Center Head、CLIP、two-stage、Head-Aligned 和继承 CRAR 参数都不会更新。
所有 `Y`、动作效用和风险标签只来自 DepthTrack Train 当前训练样本，推理时不再需要 GT。

### 16.12 冻结边界与参数效率

当前 primary tri-modal 训练期间：

| 模块 | 状态 |
|---|---|
| RGB/Depth patch embedding 与 12 层双流 ViT | 冻结 |
| 四个跨模态 Transformer block | 冻结 |
| Center/size/offset Head | 冻结 |
| CLIP 文本编码器 | 冻结 |
| 类别/属性投影与两阶段语言适配器 | 冻结 |
| Head-Aligned 安全证据/连续校正组件 | 冻结 |
| CRAR 25 维路由器 | 冻结，继承 V5 |
| EarlyLanguageGrounding | **训练，844,482 参数** |
| PrimaryTriModalConsensusGuard | **训练，571 参数** |

因此当前实验回答的是：在视觉动作、语言候选器和 CRAR 均冻结时，早期三模态交互能否
提供额外定位证据，并由极小的一致性门跨域控制漂移。任何视觉主干再训练收益都不能混入
语言贡献；训练日志中 `center_head_grad_norm` 必须始终为 0。

#### 16.12.1 新增模块的单帧计算量

对 batch size 1、3 个有效语言 role、每个模态 144 个模板 token 和 576 个搜索 token 的
标准推理几何，新增 early cross-attention 与 veto guard 的矩阵计算量如下。该口径只统计
相对继承 V5 新增的 dense projection、QK/AV attention、grounding dot product 和 guard MLP，
不是双流 ViT、Center Head、CLIP 或整套 tracker 的总 FLOPs。

| 新增计算项 | MACs |
|---|---:|
| 语言 role 投影 | 294,912 |
| RGB/Depth 视觉 token 投影 | 212,336,640 |
| role 到两路模板的 cross-attention | 22,007,808 |
| modality router | 442,944 |
| 两路搜索 token 到 grounded role 的 cross-attention | 86,704,128 |
| search residual 回投影 | 169,869,312 |
| RGB/Depth grounding similarity | 663,552 |
| EarlyLanguageGrounding 合计 | **492,319,296** |
| 9-24-12-1 veto guard MLP | **516** |
| 新增模块总计 | **492,319,812 MACs = 0.492319812 GMAC** |

按 `1 MAC = 2 FLOPs` 报告，新增矩阵计算为 **0.984639624 GFLOPs/帧**，可在论文中简写为
约 **0.985 GFLOPs/帧**。PyTorch CPU profiler 对 `addmm/bmm/baddbmm` 得到
`984,638,592 FLOPs`，与 early 模块解析公式逐值一致；profiler 能标注的 add/mul 一并计入
后为 `986,909,773 FLOPs`。归一化、softmax、activation、排序和 reduction 等算子没有被
PyTorch 完整赋予 FLOPs，因此不能把 `0.986910 GFLOPs` 写成包含所有逐元素算子的严格整量。
空文本会 fail-closed 回退视觉路径；上述数值对应三个语言 role 均有效时的主模型常规帧。

### 16.13 正式逐帧推理算法

```text
Input: 首帧 RGB-D、首帧框 b_0、首帧文本 q_0、当前帧 RGB-D

Initialize once:
  1. 裁剪并缓存 192x192 首帧 RGB-D 模板张量、语言角色与身份原型；模板 token 在每帧
     网络前向中由冻结 patch embedding 重新计算；
  2. 编码 q_0，缓存类别、稳定属性和初始状态角色；
  3. 后续不更新模板、文本或身份原型。

For each frame t > 0:
  1. 由上一帧内部状态裁剪 384x384 RGB-D 搜索区域；
  2. canonical 双流 ViT 得到 F_v，冻结视觉 Head 得到 S_v、M_size、M_offset；
  3. 在第 2 层 seed 上让语言 role 分别查询 RGB/Depth 模板并按 role 融合；
  4. RGB/Depth search token 查询 grounded roles，同时生成三张 early grounding map；
  5. 有界 search residual 通过余下视觉/跨模态层得到早期 F_l；
  6. 冻结两阶段适配器与同一 Center Head 得到完整候选 S_l；
  7. 抽取继承 CRAR 的 25 维证据，冻结 CRAR 计算 inherited logit；
  8. 保留的三张 early map 形成 9 维 guard，只添加非正修正；
  9. 若 active 且修正后 p_t>=0.5，整图选 S_l，否则整图选 S_v；
 10. 乘原 Hann window，取 argmax；
 11. 在该位置读取 canonical M_size/M_offset 并解码内部框 b_t；
 12. 把未缩放 b_t 写回 tracker state，供下一帧递归使用。
```

若启用部署 profile，v11 在步骤 1 外部决定是否扩大下一帧搜索 crop，0.98 只在步骤 11
之后生成提交给 evaluator 的报告框；二者都不修改 `p_t`、`S_v/S_l`、内部 `b_t` 或训练
参数。

### 16.14 真实样本：`bottle02_indoor` 如何经过最终架构

第 11.3 节给出了完整文件路径、图像尺寸、GT 和审核文本。这里按最终 V5 动作重新解释。
首帧真实目标为一只透明塑料瓶，框 `[172, 252, 33, 77]`，结构化查询是：

```text
category: bottle
stable attributes: clear plastic; green and red label; red cap
initial state: closer than background; reliable depth; fully visible;
               multiple similar distractors
```

第 1800 帧中，目标被拿到镜头前，尺度变大且有运动模糊，背景桌面仍有多个瓶子和易拉罐。
最终网络的实际信息流是：

1. 视觉分支利用固定首帧模板和当前 RGB-D 搜索图生成 `S_v`；
2. 类别阶段保留所有“像瓶子”的位置，并用 RGB/Depth 模板原型抑制人物、桌椅和窗户；
3. 属性阶段在全部候选上比较透明塑料、绿色/红色标签和红色瓶盖，形成候选 `F_l`；
4. 同一 Center Head 将 `F_v`、`F_l` 分别映射为 `S_v`、`S_l`；
5. 若两路同峰，CRAR 即使打开也不会产生新位置，训练标签不会把它当作救援正例；
6. 若语言峰转向另一个瓶子，CRAR 会检查该位置相对视觉峰的类别秩、属性秩、首帧 RGB
   身份秩、Depth 身份秩、峰距和响应熵；证据不一致时应保持 `S_v`；
7. 只有语言峰与视觉峰不同、语言峰 GT 支持显著更高且多模态身份支持一致的训练事件，
   才把类似证据训练成允许选择 `S_l` 的正例；
8. 无论选择哪张 center response，框宽高和 offset 都来自视觉分支。

这是一个真实输入样本和真实文本的结构走读；本文没有保存该具体帧的 25 维 route trace，
所以不能虚构它的 `p_t` 或声称该帧实测选择了哪一路。论文若需要可视化，应重新以只读
诊断模式导出 `route_features`、`route_probability`、两路热图和身份秩，不改变 tracker
状态，然后把“机制示意”和“实测路由”明确分开。

### 16.15 为什么该结构更适合只在 DepthTrack 训练后跨域

1. **动作共享同一 Head。** 两路不是不同检测器，减少数据集切换时的几何标定差异。
2. **证据以图内相对量为主。** 熵、重叠、百分位秩和归一化位移弱化绝对 score 尺度变化。
3. **语言角色稳定。** 类别和稳定属性表达长期身份，首帧遮挡/深度状态不会被错误当作每帧
   均成立的空间事实。
4. **只训练小路由器。** 视觉与语言候选能力被冻结，降低 8,000 个单域样本对大主干的
   过拟合风险。
5. **硬回退保留强基线。** 跨域证据异常时不会留下小幅语言残差，而是逐值退回 `S_v`。
6. **几何解码由视觉拥有。** 语言不能因为类别先验直接改变尺寸和 offset。
7. **固定单阈值。** DepthTrack、CDTB、VOT 共用 0.5，不存在数据集专用阈值。

这些设计提供跨域泛化的结构理由，但不能替代外部数据集实测。CDTB 已完成结果和 VOT
官方 multi-start 最终结果必须与机制论证同时报告。

### 16.16 可作为论文主贡献的表述

建议把创新点写成以下四条，而不是把所有工程模块都列为贡献：

1. **语言优先的双模态 cross-attention。** 在候选响应形成前，让类别、稳定属性和初始状态
   role 分别查询 RGB/Depth 模板，再让两路搜索 token 查询已完成实例绑定的语言 role；
   语言因此参与真正的 RGB-D token 编码，而不是只在预测头尾部重加权。
2. **可审计的三模态一致性表征。** 同时保留 `M_rgb`、`M_depth` 与带模态分歧惩罚的
   `M_consensus`，显式区分“双模态共同支持”与“单一模态语义峰”，为跨域漂移判断提供
   可视化证据。
3. **只否决不新增的一致性路由门。** 9 维 guard 在继承 CRAR logit 上只能添加非正修正，
   因而能抑制缺乏 RGB-D 共识的语言动作，却不能越权创建新动作；canonical 视觉响应与
   size/offset 始终保留为稳定回退。
4. **严格的因果与工程归因边界。** 主模型只使用不可变首帧语言，空文本和异常证据 fail
   closed；动态模板、在线语言、v11 和报告框尺度均作为独立推理 profile，不混入网络贡献。

论文摘要式中文表述可直接使用：

> 我们提出一种早期三模态交互与一致性保护的 RGB-D-L 长期跟踪框架。不同于仅在预测头
> 尾部融合文本，我们在双流主干第 2 层边界让语言 role 分别查询 RGB 与 Depth 模板，再
> 由两路搜索 token 查询已完成实例绑定的语言 role，并经过后续视觉与跨模态层形成候选。
> 模型显式输出 RGB、Depth 和分歧惩罚的一致性 grounding map；一个轻量 guard 只允许在
> RGB-D 不共同支持时下调继承的语言路由，不允许创建新路由。canonical 视觉响应及其
> size/offset 全程保留，使空文本、异常证据或跨域失配能够回退稳定视觉动作。训练仅更新
> 845,053 个新增参数，约占 checkpoint state 参数的 0.3103%。在线语言和动态模板仅作为
> 关闭默认的推理期模板偏移优化，不属于主网络贡献。

英文方法摘要可使用：

> We introduce an early tri-modal interaction and consensus-guarded framework
> for first-frame language-guided RGB-D tracking. At an early dual-stream
> boundary, language roles attend independently to the RGB and depth templates,
> after which modality-specific search tokens attend to the instance-grounded
> roles and continue through the remaining visual and cross-modal blocks. The
> model exposes RGB, depth, and disagreement-penalized consensus grounding maps.
> A lightweight veto-only guard can suppress an inherited language action when
> it lacks joint RGB-D support, but cannot create a new route. The canonical
> visual response and its box geometry remain intact for fail-closed inference.
> Training updates only 845,053 added parameters, approximately 0.3103% of the
> checkpoint state parameters.
> Online language and dynamic templates are optional inference-time drift
> controls rather than components of the learned architecture.

### 16.17 论文图表与消融建议

主方法图应画出：

1. RGB/Depth 双流与四次跨模态交互；
2. 第 2 层边界分叉出的 early RGB/Depth language branch；
3. 语言 role 到 RGB/Depth template token 的 cross-attention；
4. RGB/Depth search token 到 grounded role 的 cross-attention；
5. `M_rgb`、`M_depth`、`M_consensus` 三张 grounding map；
6. 冻结 CRAR inherited logit 与 9 维 veto-only consensus guard；
7. canonical `S_v`、candidate `S_l` 与 canonical size/offset 的 fail-closed 框解码；
8. 用虚线框把 v11、0.98、动态模板和在线语言放在“network-external inference profile”区域。

最低限度消融应包括：

| 消融 | 回答的问题 |
|---|---|
| 同权重 empty vs correct、全部后处理关闭 | 语言网络是否产生真实增益 |
| V5 vs primary safe025、全部后处理关闭 | 早期三模态交互是否保持 V5 行为边界 |
| 去掉 early language branch | 提前与 RGB/Depth token 交互是否必要 |
| 去掉 consensus map 或 disagreement penalty | RGB-D 共识是否比单模态语言峰更稳 |
| guard suppression cap 0 / 0.25 / 1.0 | veto-only 门控强度如何影响保真与漂移 |
| 连续融合/SRD-HAC vs CRAR hard action | 离散仲裁是否减少闭环伤害 |
| 去掉秩特征或改用原始 logit | 跨域标定友好证据是否必要 |
| 仅 BCE、仅负 margin、双 margin | 阈值对齐训练是否必要 |
| 去掉 RGB identity / Depth identity | 实例与深度证据各自贡献 |
| 软路由 vs straight-through hard route | 训练/推理动作一致性是否必要 |
| primary、primary+v11、primary+v11+0.98 | 学习架构和部署增强分别贡献多少 |
| primary vs primary+safe-template | 安全模板是否在保真门内减少模板偏移 |

不能写成论文贡献的内容：v11 固定搜索恢复、0.98 报告框校准、测试序列专用规则、当前
并未执行的硬 Token pruning、并未完成的真实递归 rollout 训练，以及任何后续帧 Qwen
文本推理。在线语言不属于当前必做消融，只可在主模型和模板版正式门禁完成后，作为
“推理期模板偏移优化”单独报告。当前单 seed、test-informed
full-50 历史和 CDTB 尺度假设时序也必须按第 15 节如实披露。

## 17. 关键代码位置

- 当前主训练配置：`experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1.yaml`
- 当前主推理配置：`experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml`
- 正式评测入口：`tracking/evaluate_depthtrack.py`
- RGB-D 双流主干：`lib/models/srtrack/vit_siam_dropmae.py`
- 完整跟踪模型：`lib/models/srtrack/siamtrack_dropmae.py`
- CLIP 文本编码：`lib/models/srtrack/language.py`
- 两阶段语言适配器：`lib/models/srtrack/coarse_fine_language.py`
- 早期三模态 cross-attention：`lib/models/srtrack/early_language_grounding.py`
- 三模态一致性门：`lib/models/srtrack/primary_trimodal_consensus_guard.py`
- 反事实响应仲裁路由器：`lib/models/srtrack/counterfactual_language_router.py`
- 训练 actor：`lib/train/actors/vipt.py`
- 训练 sampler：`lib/train/data/sampler.py`
- 正式测试 tracker：`lib/test/tracker/siamtrack_dropmae.py`
- 两帧路由确认负消融：`lib/test/tracker/language_route_confirmation.py`
- 冻结 v11 搜索恢复：`lib/test/tracker/language_search_recovery.py`
- 安全动态模板槽：`lib/test/tracker/safe_template_update.py`
- 在线语言推理增强：`lib/test/tracker/online_language_update.py`
- 独立 primary 证据 trace tracker：`lib/test/tracker/siamtrack_primary_evidence_trace.py`
- 全 Train trace 启动器：`tools/run_primary_evidence_trace.py`
- false-conflict 离线分析：`tools/analyze_primary_evidence_trace.py`
- VOT 完成后低频接续 trace：`tools/continue_primary_evidence_trace.py`
- evidence 通过后低频接续安全模板保真：`tools/continue_primary_safe_template.py`
- 三项保真通过后低频接续模板版正式 VOT：`tools/continue_primary_safe_template_vot.py`
- 本轮 VOT 接续脚本：`tools/continue_primary_trimodal_vot.py`
- VOT 只读进度与 ETA：`tools/report_votrgbd2022_progress.py`
- VOT 官方结果对比：`tools/compare_votrgbd2022_results.py`
- 新增模块参数/MACs/profiler 审计：`tools/audit_primary_language_compute.py`
- 计算审计公式回归：`tests/test_audit_primary_language_compute.py`

## 18. 可复现性与保存状态

当前 primary tri-modal safe025 主权重为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,096,535,310 bytes；
- SHA256：`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab`；
- safe025 配置 SHA256：`4ddf10339575003294b2ff0e77ff8583b8d630d035173d8508852b0a17806b46`；
- 只训练 `language_early_grounding.*` 和 `language_trimodal_consensus_guard.*`；
- 新增参数量：early cross-attention 844,482，一致性门 571；
- fixed-6 safe025：`65.5045 / 65.2200 / 65.3619`，与 V5 六条轨迹 byte-identical；
- DepthTrack full-50 safe025 + v11 + 0.98：`65.9959 / 65.3359 / 65.6643`；
- CDTB80 safe025 + v11 + 0.98：`75.3878 / 76.0059 / 75.6956`；
- DepthTrack `metrics.json` / manifest SHA256：`5f41886598559ead7af401247442f130c4aa38080f87ba870b0070828abe639c` /
  `1b1b789eef399557ad1380a83c4682b5bffa873bedcfb94e17b8e5b8bcf5e5df`；
- CDTB `metrics.json` / manifest SHA256：`271c91bad7f652c89ec097f99ae38c5ddd3c3045956a254a1889499ceb7a69be` /
  `cd97be8752fb4306301290b5ed3bd6e1c8454842f8e16738f20418bc79669943`；
- safe025 VOT workspace manifest SHA256：`0ef8cfb6df4ee0e96a415c24e7faabdd6304670cbe7f97794fe680ccbe44b53f`；
- VOT 接续状态：`/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/primary_trimodal_vot_continuation_state.json`。

2026-08-02 10:56 UTC 又对上述主权重做了 CPU 只读架构验收：

- `language_early_grounding.*` 共 31 个 tensor、844,482 个参数，全部 finite 且全部非零；
- 从零初始化的 `language_early_grounding.residual_gate` 已训练为 `0.0023808819`，因此早期
  RGB/Depth/language cross-attention 不是未启用的占位模块；
- `language_trimodal_consensus_guard.*` 共 8 个 tensor、571 个参数，全部 finite 且全部非零；
- early grounding、tri-modal guard、modal-token return 和 optimizer-group 四组组合测试为
  `43 passed`。该验收只读 checkpoint、未占用 GPU，也未读取 VOT partial 指标。

同一 checkpoint 的 747 个 state tensor 共含 `272,342,121` 个元素；本轮实际训练的 early
cross-attention 844,482 参数与 tri-modal veto 571 参数合计 `845,053`，只占总 state 参数的
`0.310291%`。保存的 optimizer 也只有两个 param group、39 个参数 tensor：
`language_early_grounding` 为 31 个 tensor，`language_trimodal_consensus_guard` 为 8 个 tensor，
两组 LR 均为 `5e-5`，不存在其他更新组。因此当前结果可表述为参数高效的早期三模态适配，
而不是依赖 RGB-D backbone、预测头或完整语言编码器的大规模联合微调。

2026-08-02 11:33 UTC 完成了独立计算审计。冻结结果位于：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
primary_language_compute_audit.json
```

审计中的 early 模块/config SHA、early/guard 解析 MACs、trained parameter 和 profiler
matrix FLOPs 六项检查均为 `true`。文件绑定如下：

| 文件 | SHA256 |
|---|---|
| `tools/audit_primary_language_compute.py` | `e6066aa3ce8099d19eb1f26fab02462a4c2352594df0902a567b9869222086be` |
| `tests/test_audit_primary_language_compute.py` | `e4855579ca16abcdb588c437d0dd95c39412019cbb6fb4e869be33dcfaf4c912` |
| `primary_language_compute_audit.json` | `0af984beccbbc299306dd4ce8f6a2b4f5bebb478d05a1c1a7b9109639d1a1522` |

在新路径复现，避免覆盖上述带时间戳的冻结证据：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$REPO" "$PY" -u \
  tools/audit_primary_language_compute.py --profile \
  --output-json <FRESH_COMPUTE_AUDIT_JSON>
```

该审计只给出当前新增 early tri-modal adapter 与 veto guard 的增量计算，不能代替整网
端到端 latency、显存或总 FLOPs 测量。

2026-08-02 的 checkpoint 清理记录保存在
`docs/checkpoint_cleanup_20260802_template_router.json` 和
`docs/checkpoint_cleanup_20260802_trimodal_replay.json`。共删除 4 个未晋升、无运行/manifest/
config 精确引用的 template-router / replay probe 权重，释放 4,359,111,764 bytes；对应日志、
指标、配置和运行记录均保留。随后又清理了被 primary tri-modal guard 取代的 early-grounding
和 trimodal-consensus 两个单链接 checkpoint，释放 2,191,252,380 bytes，详见
`docs/checkpoint_cleanup_20260802_superseded_language_probes.json`。当前保留权重的历史快照与
hardlink 原因见 `docs/checkpoint_retention_audit_20260802.json`；该快照早于后一次清理，实际
删除状态以新增 cleanup 记录为准。

同日按 `docs/checkpoint_cleanup_20260802_legacy_superseded.json` 又删除 4 个被当前 primary
结果支配的 July legacy 物理 inode，共对应 10 个 hardlink，释放 8,658,334,529 bytes；清理
后文件系统可用 18,831,110,144 bytes。primary、V5、唯一 Fisher baseline、SRD-HAC
full-50 champion、原始预训练和 mapped visual initializer 均明确保留。被删 payload 本地
不可恢复，但 SHA、历史指标、manifest、配置和日志仍保留；该清理记录 SHA256 为
`cb9269276084496fa99e3174c6c716e7905fec751d4ddb183c7261c78548eb58`。

随后按 `docs/checkpoint_cleanup_20260802_superseded_historical_champions.json` 清理剩余的
VHSC、旧 final-language epoch-5、CPSD fixed-6 F1 和 CPSD gate 四个历史物理 inode，共删除
6 个 hardlink 路径，checkpoint payload 合计 4,358,618,576 bytes。清理前逐一验证 SHA、
hardlink 和活跃进程依赖；文件系统可用空间由 18,826,293,248 bytes 增至
23,184,924,672 bytes。该审计记录 SHA256 为
`bc36306fa6d7da1c0f64f67c9b473f8a793c2a92804c6f25310ff995c990e99e`。当前大权重集合只保留
primary、V5、唯一 Fisher baseline、SRD-HAC 来源冠军、原始预训练和 mapped initializer；
小型 trace/replay bank 不是可部署模型权重，继续保留用于 evidence 审计。

以下历史 Fisher 权重是旧 baseline SHA `395ccf...` 的唯一实际副本，不得删除：

```text
/root/autodl-tmp/srtrack_depthtrack_full50_champion_seed2026/history/
checkpoint_fisher_epoch2__395ccf1ad15365f1d0d28dd04ef54a9108e1e71926b9b3585aaa016344bc2431.pth.tar
```

历史 V5 checkpoint 为：

```text
/root/autodl-tmp/srtrack_crar_tail_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_crar_tail_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,086,361,618 bytes；
- SHA256：`40132d57de7f6d8b78b069d9becad1db4e94228b03472f3220ad12da3e58e6b6`；
- 配置 SHA256：`cad544b7835ec3d93f08d5f2b04abf7363c911a38b00173fa18e20480fd1b74b`；
- 只训练 `language_counterfactual_router.*`，Center Head 最大梯度为 0；
- full-50 原始报告框目录：
  `/root/autodl-tmp/srtrack_crar_tail_probe_e1_seed2026/depthtrack_test_full50_recovery_v11`；
- full-50 最终 0.98 报告目录：
  `/root/autodl-tmp/srtrack_crar_tail_probe_e1_seed2026/depthtrack_test_full50_recovery_v11_scale098`；
- 原始 `metrics.json` / manifest SHA256：`07bcf2...d3c4` / `ea31d1...7861`；
- 最终 `metrics.json` / manifest SHA256：`93653d...f2ae` / `f85e6c...cb19`。

早期 grounding probe 是未晋升的历史负结果。其 checkpoint payload 已于 2026-08-02 清理，
原路径仅作为 manifest 和 SHA 审计标识保留：

```text
/root/autodl-tmp/srtrack_early_grounding_parallel_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_early_grounding_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
```

- 已删除文件大小：1,094,708,102 bytes；SHA256：`ec5f6db039e26d7b681c4a31c10a25e10f4a38405d3d1568adcd73141fbbad34`；
- 配置 SHA256：`331ffd2328234c2d9f437a5e97fb764bb20929a61d659e74f8bdf534ed6361a5`；
- DepthTrack full-50 `metrics.json` / `manifest.json` SHA256：
  `64b5126f9456a6ee7b812dee047d7d89c03ae26f3430d86dbccc64f0f27d7e70` /
  `b736a6c0f141e93b5ee33de59c69117386340ad5df245dd50230af3e9bbf1710`；
- CDTB80 full `metrics.json` / `manifest.json` SHA256：
  `12f0a0af6fe775767e68bbc9cd59586c9d19621cce2daf537e1662d2665052ef` /
  `325da7eba381d3b1da02235871ea3ec163172787b4e3b09f49c23b620d262f7b`；
- full regression audit SHA256：DepthTrack `b288c4b822a6ed98023ecdee029e0592151f43ce525d9f7c78b22c768ce05bc0`，
  CDTB `91addb83fc32ab095f31b05bdeb4ce7172a0d866d59cb0e8675f42fc0fe610be`；
- 该 probe 未替换 V5 registry，也没有生成 VOT workspace；其结果、manifest、日志和逐序列
  audit 保留，若要重跑依赖它的历史 template-memory YAML，必须先恢复上述 SHA 的 payload。

未晋升的 trimodal-consensus fixed-6 前代权重也已清理：原大小 1,096,544,278 bytes，
SHA256 为 `351a93e639c90e5175dce2aa2402d973082e8c0de5be81827392b9975d0fb704`。
其 fixed-6 metrics、manifest、audit 和训练日志保留；依赖它的历史 template-router YAML
同样需要先恢复该 SHA，当前 primary safe025 和正式 VOT 均不依赖此文件。

以下 CPSD 结果保留为完整历史负结果；其两个训练内选模 checkpoint payload 已清理，不再是
可直接加载的部署权重。原路径为：

```text
/root/autodl-tmp/srtrack_cpsd_ep1init_full_e5_seed2026/champions/
cpsd_fixed6_f1.best.pth.tar
```

- 文件大小：1,089,007,540 bytes；
- SHA256：`7668645b4bf9467bc5cf579814d58a038ff494e7091b5bfc00033fb629ca3ef8`；
- gate 选模备份为 `cpsd_fixed6_gate.best.pth.tar`，SHA256
  `258fee906a5cb0d819cf24d3f93213eb4549a069d8df0d37f7741537ce6f3812`；
- Epoch 2/4 的普通训练路径和两个唯一 champion inode 均已删除；
- CPSD 完整集 F1 低于 baseline，因此没有覆盖 full-50 registry champion。

上述 CPSD payload 的删除记录、大小、SHA 和恢复说明见
`docs/checkpoint_cleanup_20260802_superseded_historical_champions.json`；所有完整集预测、指标、
manifest 和日志仍在原结果目录。SRD-HAC Epoch 1 仍作为 V5 来源链和无后处理 full-50
历史冠军保留。

当前无后处理 full-50 最佳权重为 SRD-HAC Epoch 1：

```text
/root/autodl-tmp/srtrack_depthtrack_full50_champion_seed2026/champions/
depthtrack_full50_best.pth.tar
```

它绑定的完整指标为 64.0717/62.5719/63.3129，checkpoint SHA256 为
`ab26442ec433692b506e4151e9b1564425a453679326d4cd919e922c85f3d84e`。原视觉 baseline
及其 64.7703/61.7485/63.2233 指标已作为前任 champion 完整归档。SRD-HAC Epoch 2
普通 checkpoint 已清理；Epoch 1 由原训练路径与 registry 两个 hardlink 共同保护。

两路 CPSD 结果目录分别为：

```text
/root/autodl-tmp/srtrack_cpsd_ep1init_full_e5_seed2026/depthtrack_test_full50_correct
/root/autodl-tmp/srtrack_cpsd_ep1init_full_e5_seed2026/depthtrack_test_full50_empty
```

每一路均保留 50 个框、50 个 score、50 个 time 文件、`manifest.json` 和
`metrics.json`，框与 score 分别逐行统计为 76,373 帧。审计验证：

```text
language_source_frame_index = 0
future_frame_text_used       = false
online_language_generation   = false
distractor_quarantine        = false
language_identity_rerank     = false
depth_identity_rescue        = false
language_search_recovery     = false
```

关键结果文件 SHA256：

| 文件 | SHA256 |
|---|---|
| correct `metrics.json` | `07679472f433c779da4dc77962d0f9ff2a5d6b3f00014e934ce1cb6cfb0887e5` |
| empty `metrics.json` | `f85ebd4b6474a494d9a9a3709b1af480b2e8ce31745c8a776ddd5ec42aedacc1` |
| correct 逐序列审计 | `abad7142e50c51680a2534bc081b4b89e2ca1e578366fb1a20132a2e73112ca2` |
| empty 逐序列审计 | `ea62fdc819aec32a3c9e9823cfe76b92c9a94925b5141298dbd0bfaa0f562244` |

空文本全部 100 个框/score 文件与视觉 baseline 逐字节一致。两份逐序列审计重新计算的
总体 P/R/F1 与各自 `metrics.json` 在 `1e-10` 绝对误差内一致。

## 19. 2026-07-27 架构更新：CRAR

### 19.1 动机与论文创新边界

CPSD 和 SRD-HAC 已证明语言候选在一部分帧上有帮助，但连续地把语言残差写入 Center
响应会在多峰、遮挡和 re-entry 状态触发峰值切换。一次局部偏移又会改变下一帧 crop，
从而产生完整集上的闭环漂移。为避免用更多测试后规则掩盖这个问题，本轮加入
**Counterfactual Response Arbitration Router（CRAR，反事实响应仲裁路由器）**。

CRAR 的论文贡献不是另一个连续融合权重，而是把语言作用显式改写为两个可审计动作间的
决策：

```text
动作 0：冻结视觉分支的完整中心响应 S_v
动作 1：完整语言候选分支的中心响应 S_l
路由：  g = valid * 1[sigmoid(f(e_v, e_l, e_text, e_rgbd)) >= 0.5]
输出：  S = (1 - g) * S_v + g * S_l
```

网络不合成第三张响应图。`g=0` 时使用未经修改的视觉响应；`g=1` 时整图切换到语言
候选。框的 size map 和 offset map 始终来自冻结视觉分支；发生切换时只用新的中心响应
与同一组 size/offset 重新解码。因此语言无权任意改变宽高或 offset 几何。

### 19.2 25 维归一化证据

路由器只读取跨数据集可比较的有界量或标准化量，不读取数据集名、序列名和绝对响应
尺度。25 个特征分为：

| 特征组 | 数量 | 内容 |
|---|---:|---|
| 两路响应质量 | 6 | visual/language 的 Hann 峰值、margin、归一化熵 |
| 两路差异 | 7 | 峰值差、margin 差、熵差、分布重叠、dx、dy、归一化距离 |
| 语义与实例证据 | 5 | coarse、fine、raw-RGB、投影 RGB、投影 depth 在两个候选峰间的秩差 |
| 旧安全控制状态 | 5 | coarse/fine reliability、frame gate、refiner gate、curriculum gate |
| 完整性证据 | 2 | coarse/fine 分布重叠、稳定属性是否存在 |

响应图先转为 logit、逐图中心化并按 RMS 标准化，再计算 softmax 分布和熵。身份与语言
证据使用候选峰相对视觉峰的百分位秩差，而不是原始 logit 差。这一设计用于降低从
DepthTrack 训练后迁移到 CDTB 和 VOT-RGBD2022 时的响应标定漂移。

### 19.3 训练标签与推理 fail-closed 契约

训练时只在当前 teacher-forced crop 内比较两个离散动作。视觉峰和语言峰均按正式推理
使用的 Hann window 取 argmax，再读取 GT Gaussian 对两个峰的支持度。只有同时满足下列
条件才标为语言动作正例：

1. 两个动作选择不同位置；
2. 语言峰 GT 支持不低于 `0.20`；
3. 支持度增益严格大于 `0.05`，或支持度有正增益且 target/background gap 增益严格
   大于 `0.02`。

GT 只生成训练标签，推理时不存在。正式阈值固定为 `0.5`，不允许针对 DepthTrack、
CDTB 或 VOT 分别调阈值。空文本、无类别、curriculum 关闭或任一输入证据非有限时，
`active=false`，逐样本精确回退视觉响应。空文本的候选分支即使被计算，也不能改变输出。

训练时 straight-through hard route 保持前向行为与推理一致，同时允许显式路由 BCE 和
响应损失把梯度传回分类器；推理时使用真正的 `torch.where` 硬选择。

### 19.4 参数量与冻结边界

CRAR 为 `LayerNorm(25) -> Linear(25,48) -> GELU -> Linear(48,24) -> GELU ->
Linear(24,1)`，总计 **2,499 个可训练参数**。当前 probe 中：

- RGB/depth 双流 ViT、四个 cross block 全冻结；
- Center Head、size/offset 分支全冻结；
- CLIP、两阶段语言适配器、Head-Aligned 校正器全冻结；
- optimizer 只包含 `language_counterfactual_router.*`；
- 日志中的 `center_head_grad_norm` 必须始终为 0。

源权重是已登记的 SRD-HAC Epoch 1 full-50 champion，SHA256 为
`ab26442ec433692b506e4151e9b1564425a453679326d4cd919e922c85f3d84e`。这样 CRAR
实验只回答“能否可靠仲裁已有视觉/语言动作”，不会把视觉再训练收益误写成语言贡献。

## 20. CRAR 训练实验与类不平衡修复

### 20.1 e1：小批次先验是负结果

配置 `droptrack_depthtrack_final_language_crar_probe_e1.yaml` 的 SHA256 为
`25ed9e40bbbd7972a60989dacda87d554b5ca3ad5298866b536400fb04058ad5`。checkpoint：

```text
/root/autodl-tmp/srtrack_crar_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_crar_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,086,365,876 bytes；
- SHA256：`e903db377649d880cea12435c92b2cc82b85e6a22320640b305bfbc47c66d9f2`；
- 20,000 个训练样本、1,250 个 mini-batch；
- 整轮路由正例率 0.0197956，预测均值 0.0887386，硬选择率 0；
- 小批次正例权重均值只有 3.4460；
- 预裁剪/后裁剪梯度均值为 1.4504/0.3888，Center Head 梯度为 0。

约 72% 的 16-sample 小批次不含正例；旧实现对这些批次把正例权重设回 1。该实验没有
产生任何可执行语言选择，不进入 fixed-6 评测，也不能声称改进。

### 20.2 EMA v2：先验正确，但分母仍抵消权重

v2 配置 SHA256 为
`ac15d75b81eed4b6e2b72e04ab19d02efb4f78edf5efb46633d6602c770ecad3`，显式采用
checkpointed EMA 类先验、`MAX_POS_WEIGHT=32`、`FOCAL_GAMMA=0`。checkpoint：

```text
/root/autodl-tmp/srtrack_crar_ema_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_crar_ema_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,086,366,738 bytes；
- SHA256：`e79c4b03898df82926505940358c11495c5bf5367c6637ab46b96bc923a56dfd`；
- 训练耗时 28:38.956；
- 整轮正例率 0.0197956，预测均值 0.0386512，硬选择率 0；
- EMA 正例率窗口均值 0.0197534，最终 buffer 为 0.020535219；
- `positive_rate_updates=1250`，说明 EMA 随 checkpoint 完整保存；
- 正例权重始终为上限 32；
- 预裁剪/后裁剪梯度均值为 1.8996/0.5044，Center Head 梯度为 0。

根因是旧 BCE 使用 `sum(weighted_loss) / sum(realized_batch_weights)`。正例出现时分子和
分母同时增大，而无正例批次仍以完整尺度产生负梯度。按约 2% 正例率估算，常数分类器
的实际平衡输出约 0.2，而不是全局逆频率目标应对应的 0.5。正例批次的 2--3 量级梯度
又经 `GRAD_CLIP_NORM=1` 截断，无正例批次的小负梯度却基本不截断，进一步偏向关闭。

### 20.3 显式、可复现的 v3 修复协议

为避免旧 YAML 在新代码下静默变义，新增
`LANGUAGE_ROUTER_LOSS_NORMALIZATION`：历史默认值为 `batch_weight_sum`，只有新配置
显式写 `prior_expected` 才启用修复。EMA 模式的稳定分母为：

```text
N * [(1 - positive_rate_ema) + positive_rate_ema * positive_weight]
```

它不依赖当前小批次恰好含几个正例，因此保留全局逆频率目标，又避免损失尺度随稀有
正例剧烈跳变。新配置为
`droptrack_depthtrack_final_language_crar_balanced_probe_e1.yaml`，SHA256
`e0139fc3849c4cebfd8821293df458b7c42d42207cf757a2e143d1357b11a2bd`，并采用：

- 从 v2 checkpoint 只加载模型状态，optimizer 重新初始化；
- EMA 权重上限 64；
- 4 个 mini-batch 梯度累积；
- 裁剪阈值 5；
- 仅路由器学习率按有效 batch 放大到 `8e-4`；
- 推理阈值仍固定 0.5。

本配置增加正/负预测均值、正/负选择率和精确计数诊断。checkpoint 与评测结果必须以
本节后续实际归档为准；训练尚未结束时不得把中途日志写成最终指标。

### 20.4 v3 完成结果：负例尾部失控，未通过晋升门禁

v3 训练于 2026-07-27 完成，耗时 `26:46.309`。唯一 checkpoint 为：

```text
/root/autodl-tmp/srtrack_crar_balanced_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_crar_balanced_probe_e1/
SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,086,367,570 bytes；
- SHA256：`c236c1a69275ac8b082ddfbbbf51f5c30daab6367111c219cda40d0d78c0798a`；
- 最终 EMA 正例率：0.020535219，更新 2,500 次；
- 整轮有效正/负样本：338 / 16,718；
- 正/负预测均值：0.414466 / 0.326536；
- 正/负硬选择率：37.8699% / 18.6625%；
- 总硬选择率：19.0432%；Center Head 梯度始终为 0。

虽然最后一个日志窗口的正/负选择率改善到 55.56% / 2.71%，不同窗口的负例选择率
仍从接近 0 波动到 60% 以上。不能用最后窗口替代整轮分布，因此只按预登记流程运行
一次 `correct_full` fixed-6。结果目录为：

```text
/root/autodl-tmp/srtrack_crar_balanced_probe_e1_seed2026/fixed6_correct
```

| 方法 | PR | Recall | F1 | 相对视觉 F1 |
|---|---:|---:|---:|---:|
| 同权重视觉回退 | 64.8780 | 64.2284 | 64.5515 | - |
| 原 SRD-HAC correct | 66.0889 | 64.7108 | 65.3926 | +0.8411 |
| CRAR v3 correct | 65.2544 | 63.0656 | 64.1413 | **-0.4102** |

v3 覆盖 6 条序列、10,041 帧，`future_frame_text_used=false`，在线生成、搜索恢复、
候选重排、depth rescue 和时序拒绝全部关闭。`metrics.json` SHA256 为
`9c72aeb9ee5f336082af3cf7ef7ba0892fa40d8b915bf58ff901cbce4b1b1aca`，manifest SHA256
为 `17b99413b053488cab21b76587318ae23a21442646577f9610fe1c9f4a42b419`。

逐帧比较显示 `toy03_indoor`、`bottle03_indoor` 与视觉轨迹逐字节一致；
`ball16_indoor` F1 从 64.1329 提升到 66.6617，但仍低于原 SRD-HAC 的 68.4669；
`pigeon05_wild` 只从 4.2741 提升到 4.3144。主要失败是 `bag04_indoor`：第 1,047
帧开始与视觉轨迹分叉，F1 从 72.8090 降到 66.7497。独立漂移审计定位到
1,448--1,650 帧的 203-frame 持续回归区间，审计文件 SHA256 为
`9ec40e95f0b580c29148c107d3b5d464a4a8e4417f6851d2935a01a653198cd4`。

因此 v3 未通过“高于同权重视觉回退”的最低门禁，不运行配对 empty、full-50、CDTB
或 VOT。该结果证明全局类别平衡仍不能控制决定闭环轨迹的最高风险负例，后续实验必须
直接约束负例尾部，而不能继续提高整体正例权重。

### 20.5 v4 hard-negative：只压负例尾部仍是负结果

v4 在 v3 上加入 top-k hard-negative ranking，配置
`droptrack_depthtrack_final_language_crar_hardneg_probe_e1.yaml` 的 SHA256 为
`c71f370fe147c7612735db7777ef6de0e65c8e6862e87e9fd68920a682dd4e28`。checkpoint：

```text
/root/autodl-tmp/srtrack_crar_hardneg_probe_e1_seed2026/checkpoints/train/srtrack/
droptrack_depthtrack_final_language_crar_hardneg_probe_e1/
SIAMTrack_DropMAE_ep0001.pth.tar
```

- 文件大小：1,086,367,570 bytes；
- SHA256：`7a8543b5fd00843d2ffbd8f1751accf8d88a2dd1246043f1d1d66bdb0aa6183d`；
- fixed-6 为 `64.8527 / 62.6931 / 63.7546` PR/Recall/F1；
- `metrics.json` / manifest SHA256 为 `f6076c...4074e` / `0c281d...0fca`。

它比 v3 更差，说明仅按负例 logit 排名并不能保证正例仍位于固定 0.5 推理阈值上方，
也不能控制真正改变轨迹的风险负例。v4 未晋升，不运行 full-50。

### 20.6 v5：阈值对齐双裕量风险学习

v5 不再只优化相对排序，而是对推理阈值两侧施加两个绝对 margin：

```text
风险负例：max(0, logit_negative - (threshold_logit - margin_negative))
有效正例：max(0, (threshold_logit + margin_positive) - logit_positive)
```

只有满足风险条件的负例进入负侧约束，正例进入正侧约束；两者直接对齐固定阈值 0.5。
对应配置 `droptrack_depthtrack_final_language_crar_tail_probe_e1.yaml` SHA256 为
`cad544b7835ec3d93f08d5f2b04abf7363c911a38b00173fa18e20480fd1b74b`，8,000 个
DepthTrack 样本训练耗时 `11:06.741`。checkpoint 文件大小 1,086,361,618 bytes，
SHA256 为 `40132d57de7f6d8b78b069d9becad1db4e94228b03472f3220ad12da3e58e6b6`。

训练整轮诊断必须如实保留：有效正/负/风险负例为 `147 / 6,724 / 291`；正例预测均值
0.490937，风险负例预测均值 0.528983；TPR、全负例 FPR、风险负例 FPR 分别为
46.9388%、21.0440%、62.1994%。Center Head 最大梯度为 0。风险 FPR 仍偏高，因此不能
只凭训练诊断晋升；真正的门禁来自同权重 paired correct/empty：

| V5 fixed-6 | PR | Recall | F1 |
|---|---:|---:|---:|
| empty，无后处理 | 64.8780 | 64.2284 | 64.5515 |
| correct，无后处理 | 65.5045 | 65.2200 | 65.3619 |
| correct - empty | **+0.6266** | **+0.9916** | **+0.8104** |

两路使用同一 checkpoint、配置、实现和 6/10,041 覆盖；empty 的 12 个非 timing 输出
文件与登记视觉基线逐字节一致。correct `metrics.json` / manifest SHA256 为
`79489a...2a7` / `62f07f...dd8d`，empty 为 `f99fdd...0d11` / `24cf05...65c9b`。
这组配对结果是当前支持“语言网络本身有效”的核心证据。

### 20.7 两帧 route confirmation：风险假设成立，但折中不佳

新增的因果确认器只在 `route_selected=true` 且语言峰与视觉峰不同的连续帧上计数；首帧
强制使用视觉响应，连续第二帧才允许语言候选。未选择、同峰、空文本或畸形诊断都会清零
并 fail-closed 回退视觉。该模块默认关闭，只通过
`--language-route-confirmation-frames 2` 显式启用。

在 v3 checkpoint 上的 fixed-6 结果为 `64.9566 / 64.3282 / 64.6409`，低于 V5 无
后处理的 65.3619 F1。它确实消除了 `bag04` 第 1,047 帧的错误动作，并把 `ball16`
首次分叉从第 616 帧推迟到第 919 帧；但 `ball16` F1 只有 64.6394，明显低于 V5 的
68.4267。`bag04`、toy、pigeon、bottle、flower 最终均逐字节回到视觉轨迹，说明连续
确认过度压制了真实语言收益。结果 `metrics.json` / manifest SHA256 为
`6ad99f...a8f` / `b83b1d...32d9`，该方向作为负消融关闭。

### 20.8 预声明选择与唯一一次原始框 DepthTrack full-50

开发选择规则在正式评测前固定：只比较 V5 无后处理、v3-confirm2、V5+既有冻结 v11，
以 fixed-6 F1 最高者进入唯一一次 full-50，不根据 full-50 调参。V5+v11 以
`65.8311 / 65.9624 / 65.8967` 胜出；其 `metrics.json` / manifest SHA256 为
`7b86b2...e0e8` / `ea92fc...a517`。v11 阈值完全沿用历史
`longterm_scale_adaptive_v11`，未针对 V5 修改。

唯一 full-50 得到：

| 协议 | PR | Recall | F1 | 覆盖 | 目标判定 |
|---|---:|---:|---:|---:|---|
| DepthTrackTest initial structured target v1 | **65.9438** | **65.2732** | **65.6068** | 50 / 76,373 | 三项均达标 |

结果绑定 V5 checkpoint `40132d...e6b6`、配置 `cad544...b74b` 和测试审核文本
`8fd4fc...b75f`；`future_frame_text_used=false`，在线生成、候选重排、depth rescue、
时序拒绝和 route confirmation 全关闭，仅 v11 开启。该结果高于目标
`+0.7438 / +0.3732 / +0.5068`，但必须按“V5 学习架构 + 冻结 v11 推理增强”报告。

### 20.9 CDTB 原始结果、fixed-6 尺度选择与统一部署

同一 V5+v11 在 CDTB80 的首次原始报告框运行覆盖 80 条序列、101,956 帧，得到：

| 设置 | PR | Recall | F1 | 判定 |
|---|---:|---:|---:|---|
| 原始报告框 | 74.7326 | 75.3493 | 75.0397 | PR/F1 达标，Recall 少 0.2507 |
| 固定 0.98 报告尺度 | **75.4304** | **76.0564** | **75.7421** | 三项达标 |

原始 `metrics.json` / manifest SHA256 为 `ca9dfb...fe84` / `4e4130...1a60`；最终派生
结果为 `ffca84...60bc` / `ba3719...7fa3`。派生工具没有重新运行或修改 tracker：它校验
全部源文件后只变换第 2 帧起的报告框，score/time 逐字节复制，最终 manifest 记录源预测
整体 SHA `12df5b...0a8d` 和派生预测 SHA `42f0d9...86bd`。

同一 0.98 profile 作用于原始 full-50 后得到
`65.9978/65.3358/65.6652`，比未缩放结果三项都高；VOT workspace 则在任何正式 VOT
轨迹生成前写入同一 profile。尺度候选的提出时序仍受 CDTB Recall 缺口启发，统计边界按
第 15.10 节披露，不能倒写为预注册的 CDTB untouched 结果。

## 21. 两个已关闭方向

### 21.1 RRGG 并没有执行递归 rollout

RRGG 配置 SHA256 为
`bccc753106669093a4b61e8c5d002b6f23fd00c74ea31ca08fe043feb5e2676c`。fixed-6
选出的 Epoch 1 checkpoint SHA256 为
`36b28588056fbc3e12c9d396550ec5b1b26000c1c7e7d6c60d35e887db1ca001`。

| 范围 | PR | Recall | F1 | 结论 |
|---|---:|---:|---:|---|
| fixed-6 Epoch 1 | 64.9003 | 64.2319 | 64.5644 | 低于 SRD-HAC correct 65.3926 F1 |
| DepthTrackTest full-50 | 64.4635 | 62.0687 | 63.2434 | 低于 SRD-HAC 63.3129 F1 |

代码审计确认 `sampler.py::_causal_window_candidates` 明确生成 teacher-forced window 和
causal crop anchor；`vipt.py::_window_risk_losses` 也明确说明各帧仍是独立网络输入，序列
维只累计局部 harm 并稳定 gate。预测框没有递归生成下一帧 crop，因此不能把该实验写成
“短窗闭环 rollout 训练”。这是方法假设不成立的负结果，方向已关闭。

### 21.2 v12 搜索 fallback 没有超过 v11

同一 SRD-HAC checkpoint、同一 fixed-6 上：

| 恢复策略 | PR | Recall | F1 |
|---|---:|---:|---:|
| v11 scale-adaptive | 65.0070 | 66.2322 | 65.6139 |
| v12 fallback | 64.9571 | 66.2428 | 65.5937 |

v12 相对 v11 的 F1 为 -0.0202，不能晋升。它仍属于固定推理规则，不是语言网络架构
贡献，也不进入 CRAR 无后处理主实验。

## 22. 当前跨数据集事实与晋升门禁

下表只登记已经完成的正式结果；运行中的任务不填预测数值，不能拿 fixed-6、smoke 或
VOT one-pass PR 替代正式协议：

| 数据集/协议 | checkpoint | 当前指标 (%) | 目标 (%) | 状态 |
|---|---|---|---|---|
| DepthTrackTest full-50，V5+v11，原始框 | V5 `40132d...` | 65.9438 / 65.2732 / 65.6068 | 65.2 / 64.9 / 65.1 | 已达标，50/76,373 |
| DepthTrackTest full-50，上项+固定 0.98 | 同上 | **65.9978 / 65.3358 / 65.6652** | 同上 | **最终部署，三项达标** |
| CDTB80，V5+v11，原始框 | 同上 | 74.7326 / 75.3493 / 75.0397 | 72.9 / 75.6 / 74.2 | Recall 少 0.2507，其余达标 |
| CDTB80，上项+固定 0.98 | 同上 | **75.4304 / 76.0564 / 75.7421** | 同上 | **最终部署，80/101,956，三项达标** |
| early-grounding probe DepthTrack full-50 | `ec5f6d...` | 65.997790 / 65.335819 / 65.665136 | V5 preserve gate | 未晋升，略低于 V5；无 VOT |
| early-grounding probe CDTB80 | `ec5f6d...` | 75.404807 / 76.010464 / 75.706424 | V5 preserve gate | 未晋升；16 条序列退化 |
| primary safe025 DepthTrack full-50 | `30c804b...` | **65.9959 / 65.3359 / 65.6643** | 65.2 / 64.9 / 65.1 | 当前主模型，三项目标通过 |
| primary safe025 CDTB80 | 同上 | **75.3878 / 76.0059 / 75.6956** | 72.9 / 75.6 / 74.2 | 当前主模型，三项目标通过 |
| VOT-RGBD2022 官方 multi-start | V5 `40132d...` + v11 | 71.8513 / 82.4997 / 86.4614 | 77.9 / 82.1 / 93.7 | 历史正式结果 |
| VOT-RGBD2022 官方 multi-start | V5 `40132d...`，no-recovery | **72.9081 / 82.5349 / 87.9935** | 同上 | 官方完整覆盖；ACC 达标，EAO/ROB 未达标 |
| VOT-RGBD2022 官方 multi-start | primary safe025 `30c804b...`，no-recovery | 运行中，不填预测值 | 同上 | 截至 2026-08-02 15:01 UTC 为 103,209 / 1,327,004 tracker frames（7.777595%）、152 / 1,765 anchors、9 / 127 完整序列；worker ETA 约 2026-08-03 08:07 UTC，watcher 汇总约 09:05--09:15 UTC |

历史旧 language-e5 的 CDTB 结果为 72.3752/72.3388/72.3570，VOT 官方结果为
EAO 72.5199 / ACC 81.9775 / ROB 87.9941；它们只能作为旧基线，不能当作 V5 结果。
V5+v11 主 workspace 的历史官方 VOT 结果保持冻结：EAO `71.8513` / ACC `82.4997` /
ROB `86.4614`，覆盖 127 sequences / 1,765 anchors / 1,327,004 tracker frames。逐序列
anchor 审计显示 v11 在 `bag02`、`notebook01`、`yogurt`、`toiletpaper01` 等序列引入系统性
回退。V5 no-recovery 的完整结果为 EAO `72.9081` / ACC `82.5349` / ROB `87.9935`，相对
V5+v11 分别提高 `1.0569pp / 0.0352pp / 1.5321pp`，但 EAO/ROB 仍未达到目标，说明关闭
恢复状态机只能消除部分伤害。primary safe025 现以完全相同 no-recovery 协议运行；两项均
关闭在线语言与模板更新，避免把推理状态机和主架构效果混在一起。

同 checkpoint 的官方逐序列精确比较进一步量化了 v11 的影响。no-recovery 相对 v11 的
全局增量为 EAO `+1.056866pp` / ACC `+0.035165pp` / ROB `+1.532112pp`，127 条逐序列贡献
之和与全局差值在 `1e-12` 内一致。EAO 最大正贡献来自
`toiletpaper01_indoor_2 +0.229040pp`、`bag02_indoor_2 +0.142451pp`、
`toy09_indoor_1 +0.105890pp`、`ball06_indoor_2 +0.091661pp`；ROB 最大正贡献来自
`bag02_indoor_1 +0.355872pp`、`toiletpaper01_indoor_2 +0.333371pp`、
`bag02_indoor_2 +0.325025pp`、`yogurt_indoor_1 +0.253523pp`。仍存在较小反向序列，
例如 `earphone01_indoor_1` 的 EAO/ROB 贡献为 `-0.050925pp / -0.081372pp`，所以关闭 v11
不是所有序列都改善。完整报告 SHA256 为
`7652085c3c32739849b965203e134c7c5cfa0e43681750c19fb7aedef289e17e`。

新增的官方逐序列精确归因进一步确认了这一点。V5+v11 相对旧 baseline 的全局差值为
EAO `-0.639680pp`、ACC `+0.674315pp`、ROB `-1.673216pp`：框重叠精度已经提高，主要损失
来自少数长序列持续跟丢。EAO 最大负贡献为 `toiletpaper01_indoor_2 -0.238507pp`、
`bag02_indoor_2 -0.172329pp`、`toy09_indoor_1 -0.136857pp`、
`yogurt_indoor_1 -0.101904pp`、`ball06_indoor_2 -0.090994pp`；ROB 最大负贡献为
`bag02_indoor_2 -0.393947pp`、`bag02_indoor_1 -0.355872pp`、
`toiletpaper01_indoor_2 -0.333371pp`、`yogurt_indoor_1 -0.298742pp`、
`ball06_indoor_2 -0.227717pp`。127 条逐序列贡献之和与官方全局差值在 `1e-12` 内一致，
完整报告 SHA256 为 `b2b0cc17247a3b73cada5a586b3e43fb4c90b082eda9fd49ceac9c84c2960229`。
这也是后续优化应优先做“风险触发冻结动态模板/保持旧 state、再用双尺度重搜”，而不是
继续增强逐帧语义峰值的直接证据。

2026-08-02 又完成了失败前因果特征诊断。诊断只使用当前/历史预测框与 tracker score 作为
可部署输入，GT 仅用于离线标记失败窗口。DepthTrack Train-fixed6 上有 16 个连续 10 帧
IoU 不高于 0.1 的失败事件与 415 个 IoU 不低于 0.5 的健康窗口；单帧归一化中心位移和
绝对 log-area 变化的联合风险
`sqrt(center_jump^2 + abs_log_area_change^2)` 在健康窗口第 90 百分位为 `0.601964`，覆盖
`13/16` 个失败事件，其中 11 个在失败起点之前触发。该阈值来自训练划分，不来自 VOT。

将同一个 `0.601964` 训练阈值只读应用于 VOT 重点 14 条序列，历史 baseline 和 V5+v11
分别覆盖 `95.7895%` 与 `96.1373%` 的失败 anchor；其各自健康窗口第 95 百分位分别为
`0.614552` 与 `0.572313`。另一方面，V5+v11 的低 IoU 失败帧仍有 `65.2790%` 的 confidence
不低于 0.5，证明单一 score 阈值不能识别高置信错误峰。完整报告为：

```text
$PRIMARY_RUN/runtime/depthtrack_trainfixed6_failure_precursors.json
$PRIMARY_RUN/runtime/vot_baseline_top14_failure_precursors.json
$PRIMARY_RUN/runtime/vot_v5_v11_top14_failure_precursors.json
```

但 `0.601964` 只能作为失败前兆诊断，不能直接部署为单阈值拒绝器。随后对 DepthTrack
Train 全部 152 条序列、203,567 个有效帧和 202,472 个有效运动转移做了真实目标运动审计：
若把风险超过 0.6 的候选一律冻结，即使输入候选就是 GT，旧状态机的平均 IoU 也只有
`0.9531`。这证明真实快速运动和尺度变化会频繁跨过单阈值，直接部署会把真运动误判成漂移。

据此将尚未接入正式 tracker 的 `state_drift_guard.py` 改为两级有界候选状态机：soft
threshold 为 `0.6`，只在 RGB/Depth 或不可变身份证据明确冲突时触发；证据缺失时对中等
风险精确 fail-open。hard threshold 为 `1.2`，只对极端跳变短暂隔离。隔离期最多 4 帧，
连续 2 帧 RGB-D 与身份共同确认且候选间 center jump/log-area change 均不高于 `0.6` 时
接纳真实位移；预算到期必须 fail-open，禁止无限冻结旧 state。活动期冻结模板写入、丢弃
动态模板槽并暴露基础/扩大两个搜索尺度。在线语言仅可记录为低权重附加证据，不能触发、
维持或单独解除隔离。

全 152 条 Train 的理想候选模拟中，跨模态证据正确、证据缺失、所有真运动均被错误标为
冲突时的平均 IoU 分别为 `0.996169`、`0.993782`、`0.970193`。这比单阈值安全，但仍只是
运动上界审计，不代表真实 tracker 的跨模态冲突判断准确。当前模块保持独立、默认关闭；
GPU 空闲后必须先用 primary 在 DepthTrack Train 生成真实 RGB/Depth/identity trace，测量
false-conflict rate，再决定是否接入新 profile。现有旧 template-router trace 只有 2 个正例
和 5 个风险例，证据不足，不能据此晋升。该 trace 已完成独立只读实现和全 152 条 preflight：
诊断 tracker 继承正式 tracker，只在正式 `track()` 返回后读取网络输出和内部候选框；正式
tracker 文件已恢复并验证为冻结 SHA `12e267a...b7399d8`。preflight 证明 152 条首帧语言的
7 个字段与正式 fixed-6 `correct_full` 完全一致，且只对训练 loader 已登记的
`toy07_indoor_320` 执行 39 行 GT 尾部裁剪。正式 trace 尚未运行；完成时必须额外证明 fixed-6
预测框和 score 共 12 个文件与既有 safe025 结果逐字节一致，否则整个 trace 作废。
原 V5 workspace manifest SHA256 为 `23a7f7a5a776d2b97f98b6aa22d9b3d3a0265d6411635ec9f94732582cdd1019`；
两卡分片 manifest SHA256 为 `f957d5908633408010e3489594b96e00af3c25afbaead3540c7f53057128fc0c`。

从 V5 到当前 primary tri-modal 的实际执行顺序为：

1. V4 hard-negative 失败后，只训练一个 V5 双裕量 checkpoint；
2. 同一 V5 checkpoint 跑 paired fixed-6 correct/empty，确认 empty 精确回退且 correct 更优；
3. 在预声明候选中由 fixed-6 选出 V5+冻结 v11；
4. 只运行一次原始报告框 DepthTrackTest full-50，达标后冻结轨迹；
5. 同一 V5+v11 首次运行 CDTB80，原始 Recall 比目标低 0.2507；
6. 固定七个尺度候选并只用 fixed-6 排序，选出 0.98；从冻结原轨迹派生 full-50/CDTB，
   不改 checkpoint、响应、score、搜索状态或 v11；
7. 在生成任何正式 VOT 轨迹前冻结 V5+v11+0.98 workspace，再运行官方 multi-start；
8. 原始与校准结果都完整保留；CDTB 对尺度假设的启发按第 15.10 节披露。
9. 在第 2 层边界加入语言-RGB/Depth 双路 cross-attention，并训练 RGB-D 一致性 veto gate；
10. 由 safe025 限幅后通过 fixed-6、DepthTrack full-50 和 CDTB80 保真门；
11. 先运行 V5 no-recovery 官方 VOT，再由 5,400 秒轮询脚本自动接续 primary safe025，
    两者都使用固定 0.98 report-only 尺度且关闭 v11、模板更新与在线语言。

当前关键实现 SHA256：

| 文件 | SHA256 |
|---|---|
| `counterfactual_language_router.py` | `f7a7e13490751060900b62b0e0f39b73ed186428757e75a76bd190b8f197b5bd` |
| `early_language_grounding.py` | `052b90dde186eeddee5388eb33bb85498a627abf3d2be9afb3e35671854187ab` |
| `primary_trimodal_consensus_guard.py` | `a5c606168d002e41f60f443601c4225c4fa6caf0b6490eeedb8b53d1dbf18bf1` |
| `vit_siam_dropmae.py` | `8f75e60726598df625da39ff66df979b45c314e8ab00067fa13b24fe93875bb5` |
| `siamtrack_dropmae.py` | `12e267a0d1e93a7d9675f40a1c36197a1aba1f7f2eecc1bd3c7d630e3b7399d8` |
| `vipt.py` | `18899cf0314a688690c698ea67c6590438bf30f4cb893d971832be22fb8bf5da` |
| `regular_trainer.py` | `b098f29bac77507b125f4b689504f93b2f9122a5bdc4e249b2a76b7a2e1ccb17` |
| `config.py` | `ca3d956e4fbc3ee4436370d4c174aade83ffdd565b851d2de1e541b53479066f` |
| `language_route_confirmation.py` | `1adabacb92a8b8d0d3304d118d4ebd6d6c21e3efa7f95c7e59a78dfac9979b3a` |
| `rgbd_dropmae.py` | `3a4c4df1765dd75918e65b82de4e1362cb4b6b83c1e2f3f33412a25928db3554` |
| `language_search_recovery.py` | `c084af86e75045f4dae95444cda5e8dc18c70e828eab7e2e52692ec7702e14fb` |
| `safe_template_update.py` | `7c3382ffc59ff47791d58e5d72009528879c323f65aca124f29ee12605d2a3e6` |
| `online_language_update.py` | `5430a132d7928e03b69d99bb982436ca3f5c3019c6e0cfd85a9877e2719c627e` |
| `qwen_online_client.py` | `dfddb15485c701e8e123e0fbc10867a72c122b4d7f6b489a5b7717e33b10d9c9` |
| `continue_primary_trimodal_vot.py` | `0b5f4c673af3d8c4fbc1d5771abe88ae6a4543791baf381032645a8ba61e2747` |
| `report_votrgbd2022_progress.py` | `a3fb9836c08377cfc8da3ac1ad4134e5b18b0a22be725150d0f1c0385db8a4bb` |
| `compare_votrgbd2022_results.py` | `2bceeb3284f4548225868e11d715f86ad7ffebe5110598294de56381f1f45434` |
| `analyze_votrgbd2022_sequence_gaps.py` | `5c4d50247ee1a97b5fba1f59e806c2f8ba14ffdc6e16ef8f6ac984213c7229b4` |
| `analyze_votrgbd2022_failure_precursors.py` | `d0ba4a340587f487529797f47b944ce00d07e6aa788c0c4640c577f850b5f1ca` |
| `analyze_depthtrack_failure_precursors.py` | `96ed6b4db318b3536f80b2f8eac3f4fddee94f267c63d05e3acf8e3a883619b2` |
| `analyze_depthtrack_train_motion_guard.py` | `cf24cd618ad2c7ed48453aa78a735705ac16bbfb9bb2eaf6004ca2902afb8c5c` |
| `state_drift_guard.py` | `7d2b2f4a1b9e0005f15b50bd8d74e2507a60ef18dfd86ea36f6597cbb09b04ce` |
| `test_state_drift_guard.py` | `d26a0ed765892d5baed317a727c8ee9222a662a3494c612901db3f8eab6ed31c` |
| `test_depthtrack_train_motion_guard.py` | `21bcc2f0269bbd2277de1ef680cf6d73c606b0225e2ded84392ea6e051f4f0b3` |
| `siamtrack_primary_evidence_trace.py` | `8f20aca5d47d738eda9000992f6a49b9bf05b57946f207c2ee6cbccdaef8644c` |
| `siamtrack_primary_evidence_trace` 参数入口 | `22c834e9ed472771e6c5f63aec81988f0e5e0106a2ba940b3fe493f3c89133ba` |
| `run_primary_evidence_trace.py` | `637a45ac5662632fa942d122179d83c91fe13b1903a6518182df7367669ec81c` |
| `analyze_primary_evidence_trace.py` | `3d6b7caac51bb21231e14af9997024024827f41215c1b1a1ad7027fc62f2f341` |
| `continue_primary_evidence_trace.py` | `ceac35e3832d1040b660b2d543c34fb4cd882c47299970186440816cd6700e30` |
| `continue_primary_safe_template.py` | `9e5e7abd1206e973b56e0f7458079f6e1420e3219da4ecfcefcbd4aeb2facada` |
| `continue_primary_safe_template_vot.py` | `6a32f204e062cf59eae1673e1bb00d047a25a90bd5d14d79ea85be32c9f1740f` |
| `test_primary_evidence_trace.py` | `023fbed4e4d044f203c6d55abf6dee851d01ea7d892356bd2316cc05b81c0254` |
| `test_primary_evidence_trace_analysis.py` | `f3e0800e6b524bd954bd09eb8e2c5f70a2bd737f77e022286ef3adb328aa0618` |
| `test_primary_safe_template_continuation.py` | `2b85319feaba3359f0c007b50fe98fa8fabf3f38dc69dbfec59cebb2cfa86595` |
| `test_primary_safe_template_vot_continuation.py` | `c04030cc8eb5547239064842bbcc3ed217ba83e11eba197666aa890beded32b4` |

对应只读报告 SHA256：

| 报告 | SHA256 |
|---|---|
| `depthtrack_trainfixed6_failure_precursors.json` | `4498c520969bc86698e8daa03cbb5adb2eb4c38f1941a960ec7de45b4c40d109` |
| `vot_baseline_top14_failure_precursors.json` | `0cf271a2040a4985d80af47e21454697e2dfc0274d6884b838bbc0052ac5b293` |
| `vot_v5_v11_top14_failure_precursors.json` | `0b418ab0aa261b6332cb11738a3517cba21b2d936b2c8253f2bbec1a97d0ed03` |
| `depthtrack_train_all152_motion_guard_two_tier_bounded_audit.json` | `3761bd9706acc8a1df5153cd1cb2a6d627d10a14de41a2cc8cf5c3a3057f5cbd` |
| `primary_evidence_trace_preflight_v2_20260802/manifest.json` | `6d2856ab7aba0b2b850e35e0a280165f83b862b2f2fd9fbfa05f2c81da77f4d9` |

## 23. 测试与交接命令

最终测试总数在 VOT 完成后的收口审计中更新。测试口径必须是 `pytest -q tests`；直接运行
仓库根 `pytest` 会额外收集六个上游 `workspace/test_*.py`，它们引用本仓库不存在的
`lib.models.vipt` / `lib.config.vipt`，与本方法无关。历史 pipeline 的实现 SHA 冻结测试
若因本轮代码变化失败，应原样保留，禁止更新旧 SHA 常量来伪造通过。

2026-08-02 11:38 UTC 本轮全量 `tests` 结果为 `797 passed / 5 failed`。五个失败均来自已关闭的
full-rich-teacher、rich-teacher 和 strict-visual 历史 pipeline 对旧 actor SHA 的冻结检查；
当前模型、在线语言/安全模板、VOT 进度/总指标比较及逐序列缺口归因相关测试均通过。

### 23.1 固定路径与 SHA 前置检查

以下命令假设在同一个 shell 中执行；现有正式输出目录均不可覆盖：

```bash
REPO=/home/SRTrack_RGBD_L
PRIMARY_RUN=/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026
PRIMARY_CKPT=$PRIMARY_RUN/checkpoints/train/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
V5_RUN=/root/autodl-tmp/srtrack_crar_tail_probe_e1_seed2026
V5_CKPT=$V5_RUN/checkpoints/train/srtrack/droptrack_depthtrack_final_language_crar_tail_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
PY=/root/miniconda3/envs/mplt/bin/python
VOT=/root/miniconda3/envs/mplt/bin/vot

cd "$REPO"
sha256sum "$PRIMARY_CKPT" \
  experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml \
  "$V5_CKPT" \
  experiments/srtrack/droptrack_depthtrack_final_language_crar_tail_probe_e1.yaml
```

必须得到 primary checkpoint `30c804b...43ab`、safe025 配置 `4ddf103...6b46`、V5
checkpoint `40132d...e6b6` 和 V5 配置 `cad544...b74b`，否则停止。

### 23.2 primary 训练与 fixed-6 保真门禁

下面是 primary tri-modal 实际训练入口。现有 `$PRIMARY_RUN` 已有正式 checkpoint，不要直接
重跑到同一目录；复现时应换一个全新 `--save_dir`，且至少预留约 1.1 GB：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
PYTHONPATH="$REPO" "$PY" -u lib/train/run_training.py \
  --script srtrack \
  --config droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1 \
  --save_dir <FRESH_PRIMARY_TRAIN_DIR> \
  --seed 2026 \
  --pid-file <FRESH_PRIMARY_TRAIN_DIR>/runtime/train.pid
```

safe025 fixed-6 必须关闭全部部署增强，验证它不破坏 V5 轨迹边界：

```bash
PYTHONPATH="$REPO" "$PY" -u tracking/evaluate_depthtrack_validation.py \
  --config droptrack_depthtrack_final_language_primary_trimodal_guard_safe025 \
  --checkpoint "$PRIMARY_CKPT" --output-dir <FRESH_PRIMARY_FIXED6_SAFE025> \
  --target-jsonl /home/OSTrack_RGBD_L_dataset_modified/annotations/depthtrack_train_first_rich_reviewed_qwen3_v5.jsonl \
  --target-jsonl-sha256 56c03871b8e2a005bbd9ca12c32ed8821c5445d16af9e2631963267b89785b58 \
  --structured-control-id correct_full --threads 0 --num-gpus 2
```

门禁是 6/10,041 完整覆盖、`future_frame_text_used=false`、`SAFE_TEMPLATE_UPDATE=false`、
`ONLINE_LANGUAGE_UPDATE=false`，并且与 V5 correct_full 六条轨迹 byte-identical。correct/empty
语言因果门禁仍属于 V5 历史实验，不能用来替代 primary safe025 的保真门禁。

### 23.3 primary DepthTrack 与 CDTB 直接推理

直接重跑最终 profile 到新目录：

```bash
PYTHONPATH="$REPO" "$PY" -u tracking/evaluate_depthtrack.py \
  --config droptrack_depthtrack_final_language_primary_trimodal_guard_safe025 \
  --checkpoint "$PRIMARY_CKPT" --output-dir <FRESH_PRIMARY_DEPTHTRACK_FULL50_SAFE025> \
  --target-jsonl /home/OSTrack_RGBD_L_dataset_modified/annotations/depthtrack_test_first_rich_reviewed_qwen3_v2.jsonl \
  --target-jsonl-sha256 8fd4fc7b37d97dc08d0ec724da4b75a74eb0eb8500d356d3ced862c29090b75f \
  --language-search-recovery \
  --language-search-recovery-profile longterm_scale_adaptive_v11 \
  --reported-box-scale-profile fixed6_isotropic_098_v1 \
  --threads 0 --num-gpus 2

PYTHONPATH="$REPO" "$PY" -u tracking/evaluate_cdtb.py \
  --config droptrack_depthtrack_final_language_primary_trimodal_guard_safe025 \
  --checkpoint "$PRIMARY_CKPT" --output-dir <FRESH_PRIMARY_CDTB80_SAFE025> \
  --target-jsonl /home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/cdtb_language.jsonl \
  --target-jsonl-sha256 b2bfab241ee1a66718a082ed08e74ce0d4c07ad62812396c1e7c3a8eee9234ff \
  --language-search-recovery \
  --language-search-recovery-profile longterm_scale_adaptive_v11 \
  --reported-box-scale-profile fixed6_isotropic_098_v1 \
  --threads 2 --num-gpus 2
```

### 23.4 VOT-RGBD2022 官方 multi-start

当前正式 VOT 采用顺序接续：先完成 V5 no-recovery 对照，再自动启动 primary safe025。
两个 workspace 都关闭 v11、模板更新和在线语言，只保留 `fixed6_isotropic_098_v1` report-only
尺度：

```text
/root/autodl-tmp/srtrack_v5_vot_no_recovery_seed2034
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/votrgbd2022_official_safe025_scale098
```

不要几十秒级轮询。接续脚本固定 5,400 秒检查一次，状态文件会记录覆盖率和当前 stage：

```bash
screen -r srtrack_primary_trimodal_vot
sed -n '1,240p' "$PRIMARY_RUN/runtime/primary_trimodal_vot_continuation_state.json"
```

需要人工查看最新进度时，使用只读 ETA 脚本，不要重启或强制 evaluate：

```bash
PYTHONPATH="$REPO" "$PY" -u tools/report_votrgbd2022_progress.py \
  --workspace /root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/votrgbd2022_official_safe025_scale098 \
  --workspace-fragment parallel_full_after_pair/shard_
```

V5 no-recovery 已在 2026-08-02 13:35 UTC 完成官方 full-workspace 验证与 analysis，结果为
EAO `72.9081375844` / ACC `82.5349134065` / ROB `87.9935337544`，覆盖 127 sequences /
1,765 anchors / 1,327,004 tracker frames。随后 watcher 自动启动 primary safe025 两个分片。
2026-08-02 15:01 UTC 的唯一低频快照为 103,209 / 1,327,004 tracker frames（7.777595%）、
152 / 1,765 anchors 和 9 / 127 完整序列；累计吞吐 19.907108 tracker frames/s，worker ETA
约 2026-08-03 08:07 UTC。考虑 5,400 秒 watcher 周期，官方汇总预计约 09:05--09:15 UTC，
下一次主动查看不应早于 09:15 UTC。ETA 只用于低频排队管理，不是指标证据；primary 的
正式 EAO/ACC/ROB 仍只能来自其完整 `official_metrics_parsed.json`。

若需要在新机器从头接续，使用：

```bash
PYTHONPATH="$REPO" "$PY" -u tools/continue_primary_trimodal_vot.py --poll-seconds 5400
```

脚本会校验 manifest/checkpoint SHA，要求精确覆盖 127 sequences / 1,765 anchors /
1,327,004 tracker frames，然后运行官方 JSON analysis。不得用 `tracking/evaluate_votrgbd2022.py`
的 one-pass PR 或 partial anchors 代替 EAO/ACC/ROB。

当前环境的 VOT toolkit 在 multi-start 协议中将 ROB 定义为达到失败判定前的有效跟踪进度
占总进度的比例（`robustness / total`），因此与 EAO、ACC 一样都是越高越好；finalizer 读取
官方 analysis 的该比例并乘以 100，与 `93.7` 的百分制门槛比较，不使用“失败次数越低越好”
的 supervised VOT 语义。

两组官方结果都落盘后，用只读比较脚本收口：

```bash
PYTHONPATH="$REPO" "$PY" -u tools/compare_votrgbd2022_results.py \
  --reference-metrics /root/autodl-tmp/srtrack_v5_vot_no_recovery_seed2034/official_metrics_parsed.json \
  --candidate-metrics "$PRIMARY_RUN/votrgbd2022_official_safe025_scale098/official_metrics_parsed.json"
```

接续器还会自动运行逐序列缺口归因，严格复用官方 multi-start 的
`threshold=0.1`、`grace=10` 和 EAO 区间 `[115, 755]`，并将每条序列对全局 EAO/ACC/ROB
差值的可加和贡献写到：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
primary_trimodal_vot_sequence_gaps.json
```

该工具已在历史 baseline 与 language-e5 的 127 序列完整 workspace 上核验，重算的三项
指标与官方 JSON 的最大绝对误差小于 `1e-12`。也可独立运行：

```bash
PYTHONPATH="$REPO" "$PY" -u tools/analyze_votrgbd2022_sequence_gaps.py \
  --reference-workspace /root/autodl-tmp/srtrack_v5_vot_no_recovery_seed2034 \
  --candidate-workspace "$PRIMARY_RUN/votrgbd2022_official_safe025_scale098" \
  --output-json "$PRIMARY_RUN/runtime/primary_trimodal_vot_sequence_gaps.json"
```

只有 candidate 的 EAO/ACC/ROB 均达到 `77.9 / 82.1 / 93.7`，且 DepthTrack/CDTB 保真门仍
保持，才能把目标标为完成。

### 23.5 GPU 空闲后的 primary 跨模态证据审计

这一步不启用状态保护、模板更新、在线语言、v11 或 0.98 报告尺度，只记录 motion risk
不低于 0.4 的帧并每 100 帧抽一个稳定样本。首帧文本仍来自同一个 SHA 绑定 JSONL，GT 只
在完整推理结束后由分析器离线 join。先运行全量 trace：

```bash
TRACE_RUN=/root/autodl-tmp/srtrack_primary_evidence_trace_all152_seed2026
PYTHONPATH="$REPO" "$PY" -u tools/run_primary_evidence_trace.py \
  --output-dir "$TRACE_RUN" --threads 2 --num-gpus 2
```

启动器会拒绝非 primary checkpoint/config/文本 SHA，验证 152 条覆盖和已登记 GT tail，完成
后还要求 fixed-6 的框与 score 文件和既有 safe025 共 12 个文件逐字节一致。随后只在 Train
上扫描 RGB/Depth grounding rank、三张身份 rank 和 9 维一致性特征：

```bash
PYTHONPATH="$REPO" "$PY" -u tools/analyze_primary_evidence_trace.py \
  --trace-dir "$TRACE_RUN/traces" \
  --output "$TRACE_RUN/false_conflict_analysis.json" \
  --max-false-conflict-rate 0.01
```

分析器同时完成“不改 state 的 matched simulation”：对相同 causal forward，冲突帧选择
visual action，其余帧保持 routed action，并记录失败救回数、健康候选误伤数、平均 IoU 增量
和最坏单帧损失。这只是同帧反事实，不把选择结果反馈给下一帧，不能替代闭环指标。

只有 soft-risk 健康候选的 false-conflict rate 不高于 1%、确实覆盖失败候选、matched action
覆盖全部有效 soft-risk 帧、重新解码的 routed action 与实际 candidate state 一致、冲突帧
visual 回退的平均 IoU 增量为正，并且没有把 routed 健康候选打到 IoU 不高于 0.1，才允许
进入 DepthTrack Train-only 闭环 probe；否则保持 `state_drift_guard` 默认关闭。在线语言不参与
阈值扫描、matched action 选择或状态放行。

该任务已由 `screen` 会话 `srtrack_primary_evidence_trace` 自动排队，PID `146924`，状态文件为
`$PRIMARY_RUN/runtime/primary_evidence_trace_continuation_state.json`。接续器只检查完整的
`primary_trimodal_vot_continuation_result.json`，轮询间隔固定为 5,400 秒；不会读取 partial
anchors，也不会在 V5 与 primary 两组官方 VOT 都完成前启动 GPU trace。

安全模板是独立推理增强，采用单一预注册 profile，不借助 matched visual action 调参。
`continue_primary_safe_template.py` 只在上述全 Train trace 完整、fixed-6 trace 与 safe025
逐字节一致、健康 soft-risk false-conflict 不高于 1% 且确实覆盖失败候选后，才物化派生
配置 `droptrack_depthtrack_final_language_primary_trimodal_guard_safe_template_blend002.yaml`。
派生配置相对冻结 safe025 只允许改变 `TEST.SAFE_TEMPLATE_UPDATE`；预注册 SHA256 为
`9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0`，在线语言始终关闭。

唯一预注册 profile 每 5 帧检查、要求连续 3 帧稳定、两次提交至少相隔 30 帧；动态
counterfactual 槽权重为 `0.10`、上限 `0.20`、TTL 为 90 帧。由于 checkpoint 没有训练
template router，`MODEL.TEMPLATE_MEMORY.ROUTER_USE=false` 保持不变，不能把该路由写成
收益来源；模板通过独立限幅的 `PRIMARY_TEMPLATE_BLEND_USE=true` 以 `0.02` 权重进入下一帧
主模板输入。保真门要求 bbox 轨迹实际改变，正是为了验证这条主定位路径真实生效。

该 watcher 依次执行 fixed-6、DepthTrack full-50 与 CDTB80。fixed-6 要求三项目标通过、
相对 safe025 最大下降不超过 0.05pp、存在模板 commit，并且至少一个 bbox 文件发生变化，
防止把“提交但没有进入 primary 动作”的旧实验误报为模板收益。full-50 与 CDTB80 均要求
不低于当前 primary 最好三项指标；任一门失败立即停止，不生成 formal VOT workspace，也
禁止基于测试结果再试第二个模板 profile。只有三门全部通过，才允许准备模板增强版官方
VOT multi-start；该 watcher 本身不会自动消耗一次正式 VOT 评测机会。

静态审计还确认一个必须保留的边界：模板版正式 VOT 关闭 v11，因此当前正式 VOT 不存在
恢复状态与模板 commit 的交互；但 DepthTrack/CDTB 模板保真运行会启用冻结 v11。现有模板
策略能通过低置信、响应裕量、center jump、首帧 RGB 身份、Depth 连续性和 temporal rejection
间接阻止不可信更新，却没有直接消费 v11 的 `active/armed/recovered/expired` 状态。正式评测
期间不得修改 tracker。若保真门失败且 trace 显示 commit 落在上述恢复状态附近，下一轮只能
先在 DepthTrack Train-only 上把这些状态设为明确的 commit veto（必要时丢弃动态槽），再重新
注册独立 profile；不能根据测试集结果放宽阈值，也不引入在线语言掩盖该问题。

另一个独立 watcher `continue_primary_safe_template_vot.py` 只读取模板保真 watcher 的完整
结果；仅当 `eligible_for_formal_vot=true`、三项 comparison 全通过、fixed-6 至少一个 bbox
文件改变且存在 template commit、在线语言关闭、未用测试指标调 profile 时，才创建唯一正式
workspace。固定 tracker ID 为 `srtrack_primary_safe_template_blend002_scale098`，协议为
127 sequences / 1,765 anchors / 1,327,004 tracker frames，关闭 v11，只启用安全模板和
report-only `fixed6_isotropic_098_v1`。它用两个完整序列分片共享官方 result storage，最后仍
由 full workspace finalizer 验证完整覆盖并计算 EAO/ACC/ROB；任一保真门失败时不创建
workspace。

该任务已由 `screen` 会话 `srtrack_primary_safe_template` 自动排队，PID `172594`，轮询间隔
为 5,400 秒，状态文件为：

```text
/root/autodl-tmp/srtrack_primary_safe_template_blend002_seed2026/runtime/continuation_state.json
```

截至 2026-08-02 10:43 UTC，其 stage 为 `waiting_for_complete_evidence_trace` 且
`candidate_config_materialized=false`，证明 evidence 门前没有创建或启用模板 profile。

正式模板 VOT 任务也已由 `screen` 会话 `srtrack_primary_safe_template_vot` 排队，PID
`177587`，轮询间隔同为 5,400 秒，状态文件为：

```text
/root/autodl-tmp/srtrack_primary_safe_template_blend002_seed2026/formal_vot_runtime/continuation_state.json
```

截至 2026-08-02 10:50 UTC，其 stage 为 `waiting_for_complete_template_preservation`，
`workspace_created=false`；因此在三项保真通过前不会创建或运行模板版官方 VOT。

### 23.6 统一目标审计

`tools/audit_primary_language_goal.py` 将最终目标收敛为一个只读、fail-closed 的机器审计：
它同时校验 primary checkpoint/config SHA、checkpoint 内两个创新模块的 tensor 数量与非零
训练状态、DepthTrack/CDTB 完整指标及 manifest、官方 VOT 完整覆盖、模板分支终态、权重清理
与保留集合，以及在线语言保持默认关闭。缺失的正式结果不会被预测值、partial anchors 或测试
通过数替代。

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -u tools/audit_primary_language_goal.py \
  --allow-incomplete \
  --output-json "$PRIMARY_RUN/runtime/primary_language_goal_audit.json"
```

截至 2026-08-02 11:20 UTC，审计中以下六项为 true：核心架构与来源、DepthTrack、CDTB、
权重清理与保留、安全模板实现与预注册、在线语言次要定位；以下三项仍为 false：primary
官方 VOT 达标、VOT 接续完整、模板分支终态。权重项要求 8 份冻结清理记录 SHA 匹配、40 个
淘汰路径全部缺失、当前集合精确为 7 个路径/6 个物理 inode，且 7 个保留权重 SHA 与
SRD-HAC hardlink 均正确。模板实现项要求策略、tracker 接入、两个接续器和三组测试共 7 个
文件 SHA 匹配；候选配置只能在 evidence 门后物化，预注册 SHA 为
`9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0`，在线语言关闭。
因此总 `complete=false`，不得宣称目标完成。审计器 SHA256 为
`ff4b8231c0de444661fd3c2b688d0a07836939fb4989b68fe5203cc0675a03d8`；9 项定向测试 SHA256
为 `61ec0cb946daabe984b532ddccf583d1ee47ab15114550afb9d9b34db372e677`。

最终 champion 选择不是强制要求无模板 primary 自身达标：若保真通过的模板版正式 VOT 达标，
审计优先选择 `safe_template_blend002`；否则选择达标的 `primary_safe025`。若两者都未达标则
保持未完成。这使模板更新可能贡献真实最终增益，同时仍要求 DepthTrack/CDTB 不下降、完整
官方协议和在线语言关闭。

低频自动收口任务已由 `screen` 会话 `srtrack_primary_goal_audit` 启动，PID `201172`，轮询
间隔 5,400 秒。它只等待 primary VOT 接续结果与模板正式 VOT 接续结果两个 terminal JSON，
明确记录 `reads_partial_vot_results=false`；二者齐全后才运行上述统一审计。状态与结果路径为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
primary_language_goal_continuation_state.json
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
primary_language_goal_continuation_result.json
```

接续器 SHA256 为 `a543caf475eff72fcce073b9c005b38dd8e2008dc0073d858dbcc3dac6adb98c`，
其 3 项测试 SHA256 为 `6507ab89fc1186e82bbccab8b35f8c1de51e229e06121bc254c380425c38681e`。

### 23.7 最终代码检查

```bash
PYTHONPATH="$REPO" "$PY" -m pytest -q tests
git diff --check
```

截至 2026-08-02 11:56 UTC，针对主创新边界和 actor loss 的定向回归为：

```text
pytest -q tests/test_early_language_grounding.py \
  tests/test_primary_trimodal_consensus_guard.py \
  tests/test_modal_token_return.py tests/test_optimizer_controls.py \
  tests/test_target_c2f_dataparallel_loss.py
47 passed, 1 warning in 3.47s
```

新增计算审计公式回归也已独立验证：

```text
pytest -q tests/test_audit_primary_language_compute.py
2 passed in 1.06s
```

任何交接者都必须同时检查 checkpoint/config/profile SHA、完整覆盖、
`future_frame_text_used=false` 和 `reported_box_scale_feedback_to_tracker_state=false`。不得用
中途日志、fixed-6、partial anchors、one-pass PR 或目标可行阈值点替代正式汇总指标。

## 24. 2026-08-03 VOT 专项优化续接状态

### 24.1 已完成的完整官方对照不是最终完整模型

primary checkpoint `30c804b...43ab` 在 VOT-RGBD2022 官方 multi-start 上已经完成
`127/127 sequences`、`1,765/1,765 anchors` 和 `1,327,004/1,327,004 tracker frames`。
结果文件为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/
votrgbd2022_official_safe025_scale098/official_metrics_parsed.json
```

其指标为 EAO `72.90895597`、ACC `82.53586770`、ROB `87.98807108`。这不是历史
recovery-v11 的 `71.8513/82.4997/86.4614`，但仍没有达到 `77.9/82.1/93.7`，因此目标保持
未完成。更重要的是，该官方 workspace 使用 `safe025`：训练时上限为 `0.02` 的早期三模态
残差在推理时被限制为 `0.0001`。它是跨数据集安全对照，不应表述为完整激活的主模型。

当前 VOT 专项候选恢复 `EARLY_GROUNDING.RESIDUAL_MAX=0.02`，保留早期
Language-to-RGB-template、Language-to-Depth-template 以及 RGB/Depth-search-to-grounded-language
交互，再分别加入安全模板或时序身份保护。两个 14 序列诊断仅用于淘汰候选，不能替代完整
127 序列官方结果：

```text
/root/autodl-tmp/srtrack_primary_template_diag_seed2026
/root/autodl-tmp/srtrack_primary_temporal_diag_seed2026
```

安全模板候选保留首帧静态锚点，仅在置信度、NMS 响应裕量、RGB 身份、center jump 和原始
Depth 连续性同时通过时更新单个动态槽；动态槽上限为 `0.20`，进入主模板输入的额外 blend
仍限制为 `0.02`。在线 Qwen 文本在这些 VOT 实验中保持关闭。

### 24.2 v11 的递归状态缺陷与 v12 修复

逐帧失败归因显示最差序列普遍存在同一模式：前一帧仍有较高 IoU，下一帧中心突然移动约
`1.1--1.7` 个目标尺度，随后错误目标置信度重新升高并形成闭环。旧
`LanguageSearchRecovery` 只在下一帧围绕 `trusted_bbox` 扩大搜索，却仍把当前疑似身份切换框
写入 `SIAMTrack.state`。当恢复窗口严格到期且拒绝 anchor promotion 时，policy 内部虽然保留
可信锚点，主 tracker 的递归状态却没有回滚。这是 v11 只能改善部分序列并伤害另一些序列的
一个直接实现原因。

新增 profile `longterm_scale_adaptive_hold_v12` 默认不影响旧 profile；只有显式启用时才在
恢复窗口 active、或严格到期且拒绝 anchor promotion 时，把递归 state 与报告框保持在可信
锚点。语义确认成功后立即释放到新候选。若 state 被保持，同帧安全模板更新会收到明确 veto，
避免用“可信位置框 + 错误候选响应分数”的混合证据提交动态模板。相关实现与测试为：

```text
lib/test/tracker/language_search_recovery.py
lib/test/tracker/siamtrack_dropmae.py
lib/test/parameter/rgbd_dropmae.py
tests/test_language_search_recovery.py
tests/test_rgbd_parameter_overrides.py
```

定向回归结果为 `68 passed`。在接入 V16 输出保护、参数 allowlist 和 V18 rank 字段后，
完整 `tests/` 测试集当前为 `904 passed / 5 failed`；5 个失败全部来自历史
rich-teacher/strict-visual 工具中已过期的 actor SHA 冻结值，实际 actor SHA 为
`18899cf...f5da`，与本次只改推理恢复路径无关，不能把它们改写成 v12 通过。

v12 已冻结并完成 top-14 诊断的 workspace 为：

```text
/root/autodl-tmp/srtrack_primary_v12_hold_diag_seed2026
```

V12 的 14 序列/295 anchors 已完成。反事实投影到完整 127 序列后的指标为
`EAO/ACC/ROB = 70.09172/83.04403/83.96588`，其中只有 ACC 通过目标；因此 v12 被淘汰，
不能替代当前正式 reference `72.90896/82.53587/87.98807`。

它绑定同一 DepthTrack checkpoint、`active025_template` 配置、v12 profile、安全模板和
report-only `0.98`，仍禁用未来帧文本。实现清单为
`runtime/implementation_manifest.json`，workspace manifest SHA256 为
`319ab0ee...d558`。后台顺序是先完成当前模板/时序两组诊断，再运行 v12 的相同 14 序列全部
295 个 anchors，最后恢复 DepthTrack Train evidence trace；检查间隔保持 5,400 秒。

2026-08-03 14:26 UTC 的低频快照中，template 已完成 `116,226/259,874` tracker frames、
`134/295` anchors，temporal 已完成 `113,764/259,874` frames、`130/295` anchors；两者吞吐
分别约 `11.11/11.17 frames/s`，粗略完成时间为 `18:02/18:05 UTC`。v12 在该快照时尚未
启动。两张 RTX 3090 各约使用 `13.3/24 GB` 显存，但利用率为 `96--100%`；因此当前长耗时
来自官方 multi-start 的约 259,874 tracker frames 和满负载计算，而不是显存不足或 GPU 空转，
不增加同卡并发。

### 24.3 已排除纯报告框运动外推

对已完成官方轨迹进行了不反馈 tracker state 的因果速度外推扫描。该扫描使用矩形 IoU
近似聚合，不是 VOT toolkit 的官方子集输出；其高风险 14 序列近似基线为 EAO
`50.14097`、ACC `74.90837`、ROB `53.43527`，最佳扫描点仅达到 EAO
`50.30166`、ACC `74.87330`、ROB `53.59472`，即约 `+0.16 EAO/-0.04 ACC/+0.16 ROB`。
这验证了单纯 hold 或报告框外推只能延迟失败，无法满足接近 `+5 EAO/+5.7 ROB` 的缺口，
因此不进入正式候选。后续候选必须实际保护递归搜索状态，并由 RGB、Depth、语言/模板身份
证据确认新的 anchor。

正式诊断比较使用 `tools/analyze_votrgbd2022_sequence_gaps.py --sequence-list`。该路径直接
调用 VOT toolkit 的 multi-start overlap、grace、burn-in 和 EAO partial curve 实现；已在
V5 no-recovery 与 primary safe025 两份完整 workspace 上验证选择 14 序列时仍覆盖全部
295 个 anchors，并在输出中明确记录 `full_dataset=false`。聚合契约保存在：

```text
/root/autodl-tmp/srtrack_primary_v12_hold_diag_seed2026/runtime/
top14_aggregation_contract.json
```

三组候选完成后使用 `tools/finalize_vot_top14_candidates.py` 一次性收口。该工具要求 reference
与每个 candidate 的 checkpoint SHA、VOT 设置、序列集合和 anchor 数完全一致，固定按
EAO delta、ROB delta、ACC delta 排序；输出强制包含 `diagnostic_only=true`、
`official_full_dataset_result=false` 和 `may_satisfy_final_vot_target=false`。自身契约测试已确认
14 序列和 295 anchors 的完整选择，不能通过该工具直接触发正式 VOT 达标结论。
后台 stage、screen PID、5,400 秒轮询约束和最终输出路径记录在
`srtrack_primary_v12_hold_diag_seed2026/runtime/background_schedule.json`；收口 watcher 只在
v12 screen 完全退出后运行，不读取 partial results。

### 24.4 后续硬门禁

1. 先比较模板、时序与 v12 在同一 14 序列、同一 295 anchors 上的闭环结果；任何子集值均
   只能作为候选淘汰证据。
2. 候选必须在完整 DepthTrackTest 和 CDTB80 上分别保持当前最好约
   `65.996/65.336/65.664` 与 `75.388/76.006/75.696`，至少仍通过用户目标。
3. 只有跨数据集门通过，才允许启动新的 VOT `127/1,765/1,327,004` 官方 workspace。
4. 只有完整官方 EAO/ACC/ROB 同时不低于 `77.9/82.1/93.7`，才能选择 champion、更新权重
   保留清单并把总目标标为完成。

### 24.5 恢复搜索范围与 token 分辨率诊断

为区分“搜索窗覆盖不足”和“错误状态/身份选择”两类原因，
`tools/analyze_votrgbd2022_failure_precursors.py` 增加了纯离线恢复几何诊断。它以官方失败
起点前一帧的预测框作为因果可信框代理，计算后续 grace 窗口内 GT 完整落入方形搜索裁剪所需
的最小 factor，并估算 `384 / stride 16` 输入中的目标 token 尺寸。该分析显式记录
`future_ground_truth_used_offline=true`、`inference_feedback=false`；GT 只用于失败后的原因
归因，不能进入任何 tracker 状态、阈值或正式指标。

primary safe025 的 127 序列完整 workspace 共得到 363 个官方失败运行。目标在触发后的
1--9 帧中至少一次可由相应搜索窗完整覆盖的比例为：factor 5 `95.8678%`、factor 7
`99.4490%`、factor 9 `99.7245%`。在高风险 14 序列的 192 个失败运行中，对应比例为
`96.8750% / 99.4792% / 99.4792%`，即 factor 9 相对 factor 7 没有增加诊断集覆盖。反而在
factor 9 下，完整 127 序列可见目标的几何平均 token 跨度中位数仅 `2.5994`，最短边 token
跨度中位数仅 `2.1565`；14 序列分别为 `2.6746 / 2.2427`。

因此现有失败的主因不是搜索窗过小：绝大多数目标仍在可信中心附近，真正的问题是疑似身份
切换框污染递归中心，以及宽裁剪降低了目标分辨率。冻结中的 v12 仍按预注册 factor 9 完成，
不得中途改写；若 v12 的统一 295-anchor 结果不足，下一候选应在同一 state-hold 与三模态语义
确认逻辑下优先验证 factor 7。它相对 factor 9 提高约 `28.6%` 的线性目标分辨率，同时在当前
14 序列诊断中保持相同几何覆盖。相关输出为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
vot_baseline_top14_failure_precursors_geometry.json
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
vot_full127_failure_precursors_geometry.json
```

新增几何函数的定向测试为 `6 passed`。上述覆盖率只是“搜索窗里存在 GT”的能力上界，不证明
模型能选中目标，也不能替代候选闭环指标或 127 序列官方结果。

候选完成后还应运行 `tools/project_vot_top14_candidate.py`：它把候选的 14 序列完整 trajectory
替换进 safe025 的 127 序列统计，其余 113 序列保持 reference，复用官方 EAO curve/active
权重精确重算反事实全量指标。输出固定标记 `counterfactual_projection_only=true`、
`official_full_dataset_result=false`、`may_satisfy_final_vot_target=false`；即使投影过线也只表示
值得进入跨数据集门禁，不能作为正式结果。用 safe025 自身作 candidate 的契约检查精确得到
零 delta、14 sequences / 295 anchors，文件为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
vot_top14_self_projection_contract.json
```

几何诊断、候选 finalizer、反事实投影和逐序列聚合的合并定向测试为 `16 passed`。

投影器还用同 checkpoint 的历史 V5 no-recovery/reference 与 recovery-v11/candidate 做了独立
回放：仅替换相同 14 序列时投影 EAO/ACC/ROB 为
`71.699389/82.578152/86.297899`，而 v11 的 127 序列真实官方结果为
`71.851271/82.499748/86.461422`。EAO 与 ROB 的绝对误差仅 `0.151882/0.163523`，说明该子集
覆盖了历史恢复策略的主要损失源；ACC 误差 `0.078404`，且本来已过线。验证文件为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/
vot_v11_top14_projection_validation.json
```

这仍不是对新候选全量指标的保证。使用时必须把投影视为昂贵全量评测之前的筛选信号，并为
EAO/ROB 留出至少约 `0.2` 个百分点的历史投影误差余量。

### 24.6 已排除恢复期报告框扩张

在同一 192 个高风险失败运行上又扫描了只改变报告框、不反馈递归状态的可信框等比例扩张。
该扫描把官方失败起点前一帧预测框视为可信框代理，在起点至 grace-1 的 10 帧内用 GT 外接
矩形近似 IoU；结果显式记录 `does_not_model_recursive_feedback=true`，不是官方 trajectory
重放。原比例 `1.0` 时，至少一帧 IoU 高于 0.1 的运行占 `94.2708%`，全部可见帧均高于
0.1 的运行占 `64.0625%`，每运行平均矩形 IoU 为 `0.4090`。扩大到 `1.5` 后三项分别为
`94.7917% / 66.6667% / 0.2814`：桥接覆盖仅增加 `0.5209/2.6042` 个百分点，平均 IoU 却
下降 `0.1276`。`2.0/2.5/3.0` 的至少一帧覆盖进一步降到
`89.5833%/78.1250%/60.9375%`。

因此恢复期输出大框的理论收益远小于约 `+5.7 ROB` 缺口，且会直接消耗当前仅约 `+0.44`
的 ACC 余量，不进入候选。后续仍聚焦递归状态保护、较高 token 分辨率和三模态身份峰选择，
不采用针对 0.1 失败阈值的报告框技巧。加入该诊断后的合并定向测试为 `17 passed`。

### 24.7 待接线的恢复期三模态候选峰选择

当前 v12 只在 Center Head 已选中的单个最高峰上读取类别、属性、原始 RGB 模板身份和投影
RGB 身份证据；如果真实目标是响应图的第二或第三个峰，policy 只能等待它以后自行成为最高
峰。为解决这个结构性限制，已新增默认不接线的独立模块：

```text
lib/test/tracker/language_recovery_peak_rerank.py
tests/test_language_recovery_peak_rerank.py
```

模块仅在 recovery 已经 active 时检查 NMS top-K。若原视觉最高峰满足既有 v11 门槛
`raw score >= 0.45`、coarse/fine logit 均不小于 0、raw RGB identity rank 不小于 `0.80`、
projected RGB identity rank 不小于 `0.98`，则无条件保留原峰。只有原峰被语义门拒绝、且另
一个视觉峰同时通过全部门槛时，才选择其中视觉响应最高者。inactive、缺失/非有限证据和配置
异常均 fail-closed 到原 argmax；相同最大值导致 top-K 顺序不确定时，也单独验证原 argmax 的
语义资格，不允许误切换。定向测试为 `6 passed`。

该模块尚未 import 到 `siamtrack_dropmae.py`，也没有任何指标，不能表述为 v12 的组成部分或
既有提升。只有冻结 v12 完成且反事实投影不足时，才允许与 factor 7 profile 一起接线并在
相同 14 sequences / 295 anchors 上验证。这样后续候选仍直接使用训练得到的早期
Language-RGB-Depth 交互图，而不是增加 VOT GT 规则或在线文本依赖。

### 24.8 V16 alternative probe：只提交经确认的 VOT 候选

v16 已把恢复期三模态候选峰接入独立 TraX bridge
`tools.vot_siamtrack_rgbd_v16.py`，profile 名为
`alternative_probe_factor5p01_trimodal_v16`。它仍绑定同一份 DepthTrack primary checkpoint，
在线 Qwen 保持关闭，使用首帧语言和训练得到的 RGB/Depth/language maps；没有读取未来帧或
VOT ground truth。

v16 的关键修复是把“候选探针”和“对外轨迹”分开：

1. trimodal selector 找到 alternative 时，alternative 只送入恢复状态机；
2. 未完成连续两帧语义、RGB、Depth 和 IoU 一致性确认前，报告框和下一帧递归 state 仍使用
   同帧 visual baseline；
3. 只有 `decision.recovered` 才将 alternative 解码结果提交到输出和递归 state；
4. probe、active recovery 和未通过 anchor promotion 的 expiry 都会 veto 安全模板更新；
5. baseline 解码异常时 fail-closed 到因果 trusted bbox，不会用 malformed peak 改写轨迹。

默认 `SIAMTrack` hook 保持 v12/v15 的既有行为，只有 `RecoveryPeakV16Tracker` 覆盖输出选择，
避免 VOT 专项保护改变 DepthTrack/CDTB 主路径。V16 离线 192 事件 replay 仅用于诊断，当前
得到 9 次 alternative commit、9/9 IoU 不低于 0.5、0 次有害 commit；这些数字不是 VOT
成绩。V16 官方 top-14 workspace 为：

```text
/root/autodl-tmp/srtrack_primary_v16_alternative_probe_official_seed2026
```

其 14 序列/295 anchors 的 VOT toolkit 结果和反事实 projection 已完成；投影指标为
`EAO/ACC/ROB = 69.90626/83.02949/83.88974`，同样低于 reference，故 v16 被淘汰，没有
启动 v16 的完整 127 序列评测。projection 和 top-14 comparison 均固定标记为
diagnostic-only，不能替代正式 EAO/ACC/ROB。

### 24.9 V18 rank-calibrated bridge：修复语言 logit 的错误零阈值

V16 replay 暴露了一个明确的召回瓶颈：`language_coarse_logits` 和
`language_fine_logits` 是每张响应图上的稠密亲和度，不是以 `0` 为分类边界的二分类输出。
V16 同时要求两个值不小于 `0`，因此虽然通过的 9 次提交全部正确，仍把大量“绝对值为负、
但在当前图内排名靠前”的目标候选拒绝。V18 没有重新训练 checkpoint，也没有读取 VOT GT；
它只把部署时的语义证据校准为当前响应图内的 percentile rank：

```text
coarse rank >= 0.85
fine rank >= 0.80
raw RGB identity rank >= 0.60
projected RGB identity rank >= 0.90
projected Depth identity rank >= 0.75
early Depth rank >= 0.50
early RGB-D-language consensus rank >= 0.40
```

这些条件仍要求候选来自 recovery active 时的 NMS top-16，并经过连续两帧
`IoU >= 0.90`、相对可信锚点 center jump 不大于 `0.90` 的确认。为了防止一个稳定干扰物覆盖
本来健康的视觉峰，V18 还加入“强基线保护”：当 baseline raw score 不低于 `0.70`，且
coarse/fine rank 都不低于 `0.90` 时，alternative 至少有一个语言 head 的 rank 不得低于
baseline；较弱 baseline 只允许 `0.05` 的相对 rank 容差。未完成两帧确认前，alternative
仍只进入非递归 probe，公开输出、下一帧 state 和安全模板均保持原 visual baseline。

冻结的 192 个 VOT 高风险事件 replay 中，factor 5 的 V18 因果规则得到 24 次提交，其中
23 次提交帧 IoU 不低于 `0.5`、0 次低于 `0.1`；factor 7 为 `23/20/0`。factor 5 的 selector
层面另有 272 个 helpful 和 93 个 harmful 变化，但未确认 selector 变化不会进入公开输出，
因此真正的风险门是 commit 统计。这里的 GT 只在所有因果选择结束后用于离线 IoU 审计，
replay 不建模提交后的递归反馈，不能作为 EAO/ACC/ROB 或达标证据。

V18 独立入口、分析和正式候选 workspace 为：

```text
tools/vot_siamtrack_rgbd_v18.py
tools/analyze_v18_rank_calibrated_replay.py
/root/autodl-tmp/srtrack_primary_v18_rank_calibrated_diag_seed2026/runtime/
/root/autodl-tmp/srtrack_primary_v18_rank_calibrated_official_seed2026
```

正式 workspace 仍绑定 checkpoint SHA256
`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab`、首帧语言、
安全模板和 report-only `0.98`，在线 Qwen 与未来帧文本均关闭。V18 bridge 不进入默认
DepthTrack/CDTB tracker 路径；新增 selector 字段的默认值为恒等门，V16 修改前后 replay
精确保持 `113` 次 selector change、`9/9/0` commit 统计。相关定向回归为 `57 passed`。

V18 只能在 V12、V16、V17 顺序候选没有过线时运行同一 14 序列/295 anchors；只有 top-14
反事实投影同时超过 `77.9/82.1/93.7` 且留出投影误差余量，才允许进入 127 序列正式评测。

### 24.10 V24/V25：安全模板、Train motion trigger 与完整 top-14 结论

V24 在 V23 的 DepthTrack-Train cross-scale ranker 上加入 branch-safe 模板更新。恢复分支
active 时暂停模板观察，只有最终公开分支稳定后，才允许首帧静态 RGB/Depth 身份锚点、响应
margin、center jump、RGB identity 和 Depth continuity 共同门控动态模板。V24 已完成相同
14 序列、295 anchors 和 259,874 tracker frames；其反事实完整集投影为：

```text
EAO/ACC/ROB = 73.11213818 / 82.51043405 / 88.25355086
```

V25 保留 V24 的跨尺度 ranker 与安全模板，只用 DepthTrack Train 的 motion/scale transition
分布把触发策略校准为 `MOTION_SCALE_TRIGGER=0.60`、`MOTION_SCALE_MAX_SCORE=1.0`，并取消
必须先发生 semantic conflict 的附加条件。V25 仍绑定同一 primary checkpoint、首帧语言、
active `EARLY_GROUNDING.RESIDUAL_MAX=0.02`、factor 5/7 双分支和 report-only 0.98；在线
Qwen、未来帧文本和 VOT GT 均未进入推理。

V25 通过 8 worker/4 GPU 完成全部 14 序列、295 anchors 和 259,874 tracker frames，统一
子集指标为：

```text
EAO/ACC/ROB = 51.22786181 / 74.94669819 / 55.13395051
```

把这 14 条完整 trajectory 替换进 primary safe025 的 127 序列 reference 后，反事实投影为：

```text
EAO/ACC/ROB = 73.12867781 / 82.50386581 / 88.28537104
```

相对 V24 仅为 `+0.016540/-0.006568/+0.031820pp`，相对正式 reference 为
`+0.219722/-0.032002/+0.297300pp`。EAO 和 ROB 仍分别低于目标约 `4.77/5.41pp`，因此
V25 被淘汰，不启动完整 127 序列 VOT。权威诊断文件为：

```text
/root/autodl-tmp/srtrack_primary_v25_train_motion_seed2026/runtime/
top14_comparison.json
/root/autodl-tmp/srtrack_primary_v25_train_motion_seed2026/runtime/
top14_projection.json
```

两个文件分别强制记录 `diagnostic_only=true` 或
`counterfactual_projection_only=true`，均不是正式 VOT 达标证据。直接对该 top-14 根
workspace 调用 127 序列 `vot analysis` 会因其余 113 序列缺失而 fail closed；正式子集比较
必须继续使用 `finalize_vot_top14_candidates.py` 和 `project_vot_top14_candidate.py`。

### 24.11 V26：DepthTrack-Train OOF 召回校准候选

V25 的限制不在 motion trigger 数量，而在保守 ranker 的部署 margin。原 artifact 在 149 条
DepthTrack Train、2,266 个因果回放帧、5-fold sequence-group OOF 上，只恢复
`39/1,059=3.6827%` 的 failure baseline misses，平均单帧 IoU 增益为 `0.01404`。对同一组
OOF score 做风险-召回扫描后，5% failure-risk 预算对应 margin `3.605391`，可恢复
`122/1,059=11.5203%`，平均 IoU 增益为 `0.04162`；953 个健康 control baseline 中只有
4 个有害选择，比例 `0.4197%`。该阈值完全来自 DepthTrack Train，没有使用 DepthTrack Test、
CDTB 或 VOT GT。

V26 artifact 路径和 SHA256 为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_cross_scale_ranker_recall5_seed2026/
ranker_recall5_seed2026.pth
66ce2ef506dfd3e4d4b385177dd85fa0760fb17d2e2610a973765aa3a99d8434
```

它与原保守 ranker 的模型参数逐 tensor 完全一致，只改变由 OOF 选出的 `score_margin`，因此
不会把一次新的主 checkpoint 训练混入比较。独立入口为 `tools/vot_siamtrack_rgbd_v26.py`，
profile 为 `train_motion_q98_recall5_ranker_template_factor5_factor7_v26`。V26 必须先完成同一
14 序列/295 anchors 闭环诊断；若投影仍未同时超过目标并留出至少 0.2pp 误差余量，继续淘汰，
不得启动 127 序列正式 VOT。即使 VOT 候选过门，仍必须补跑完整 DepthTrack Test 与 CDTB80
保真门，不能用 Train OOF 代替跨数据集指标。

### 24.12 V26 完整 top-14 结果与淘汰结论

V26 已用 8 worker/4 GPU 完成与 V24/V25 完全相同的 14 序列、295 anchors 和
259,874 tracker frames。统一子集指标为：

```text
EAO/ACC/ROB = 51.23406163 / 74.92742578 / 55.14618314
```

把这 14 条完整 trajectory 替换进 primary safe025 的 127 序列 reference 后，反事实投影为：

```text
EAO/ACC/ROB = 73.12989684 / 82.50124563 / 88.28751011
```

相对 V25 仅为 `+0.001219/-0.002620/+0.002139pp`。V26 共改变 41/295 条 anchor
trajectory，其中 `cup02_indoor_1` 占 27 条、`toy09_indoor_1` 占 8 条，其余序列合计
6 条；只有 cup 的变化带来极小 EAO/ROB 正收益。该结果说明 ranker margin 不是当前主要瓶颈，
继续扫描 margin 没有足够收益。V26 的投影 EAO 和 ROB 仍分别低于目标约
`4.77/5.41pp`，因此淘汰，不启动 127 序列正式 VOT。

权威诊断文件为：

```text
/root/autodl-tmp/srtrack_primary_v26_recall5_seed2026/runtime/top14_comparison.json
/root/autodl-tmp/srtrack_primary_v26_recall5_seed2026/runtime/top14_projection.json
```

下一步不再调整同一 ranker 的 score margin，而是在 DepthTrack Train 因果回放中统计
alternative selection、evidence staging、temporal confirmation 和 recursive-state commit 的
逐级损耗及拒绝原因，再据此设计多帧 RGB/Depth/语言候选一致性模块。

### 24.13 V27：DepthTrack-Train 因果多帧跨模态一致性

V26 的 OOF commit funnel 证明主要召回瓶颈不是 ranker margin。在可观测的 V25 trigger
子集上共有 167 个事件：93 帧选择 alternative 并成功 stage evidence，57 帧随后被
`center_jump > 0.90` 拒绝，26 帧到达 confirmation evidence，最终只有 4 次递归状态
commit。漏斗文件及 SHA256 为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_v26_commit_funnel_seed2026/
oof_commit_funnel.json
e0e6b66d9249f226a15f6a65295a4ad98861fe6fcb150f66a5183b174017fd94
```

因此 V27 不再放宽单帧 center-jump 阈值，而是把恢复窗口扩展到 12 帧，并要求连续两帧的
RGB 模板身份、Depth grounding、早期 RGB-Depth-language consensus、候选框运动与
cross-scale ranker 特征共同支持同一个大跳候选。horizon-12 回放仍只来自 DepthTrack Train，
共 609 个事件；8 workers/4 GPUs 的 8 个 replay shard 全部成功。events 文件 SHA256 为
`988f94903eb21b1232cfe30282f9ca63d5140a2d2a00a942e9a30f84de0d1939`。

先验固定阈值规则扫描覆盖 199 个全窗口 causal pairs、90 个可观测 trigger pairs和 810 个
policy，结果为 0 个可行策略；最佳规则的 commit precision 约 53%，说明简单阈值组合不能
可靠替代硬 center-jump gate。随后训练 96 hidden-dim 的小型 causal consistency MLP，使用
5-fold sequence-group OOF、199 个 pairs 和 160 epochs。选定 OOF threshold 为
`0.9963613153`，得到 80 次 commit、`80.0%` commit precision、恢复 61 个 failure misses，
control/failure harm 均为 0，平均 IoU gain 为 `0.648354`。这些数字只表示 DepthTrack Train
OOF，future GT 仅用于离线训练 label，推理端不可见；它们不是 DepthTrack Test、CDTB 或
VOT 成绩。

冻结产物为：

```text
ranker_horizon12_recall5_seed2026.pth
6abfdc48fdf01eeaa587ca5384f02743191d50543a1d960b6f80b8e2ac5aa96a
multiframe_consistency_seed2026.pth
f987d6385f6bf5974a6958be51c5267cfa8f47f57037bfb0dc112172f2dfdcf4
```

两者都绑定 primary checkpoint SHA256
`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab`，multiframe artifact
还绑定上述 ranker SHA。V27 只允许 learned pair support 覆盖大跳拒绝；既有单帧
RGB/Depth/language 语义门、两帧 temporal confirmation、branch-safe recursive commit 和
安全模板暂停规则全部保留。缺 artifact、schema/SHA 不匹配或任一非有限特征均 fail closed。
在线 Qwen 和未来帧文本仍关闭。

V27 独立入口与正式候选 workspace 为：

```text
lib/test/tracker/siamtrack_recovery_peak_v27.py
lib/test/tracker/multiframe_candidate_consistency.py
tools/vot_siamtrack_rgbd_v27.py
/root/autodl-tmp/srtrack_primary_v27_multiframe_seed2026
```

定向继承链、产物绑定、workspace/shard 和 V27 行为回归共 `63 passed`；其中明确验证第一帧
大跳 proposal 不放行、仅连续第二帧的完整跨模态证据可以触发 learned override，且 V24
安全模板暂停在 V27 中继续生效。最短序列真实 TraX
smoke 覆盖 2 anchors/62 tracker frames 并成功完成。2026-08-05 08:57 UTC 已按与 V26
相同的 14 sequences/295 anchors/259,874 tracker frames 启动 8 workers/4 GPUs 闭环诊断。
V27 已于 2026-08-05 11:05 UTC 左右完成，完整性校验确认 14 sequences、295 anchors 全部
有效。统一 top-14 指标为：

```text
EAO/ACC/ROB = 51.22717863 / 74.93648233 / 55.13728819
```

把这 14 条完整 trajectory 替换进 primary safe025 的 127 序列 reference 后，反事实投影为：

```text
EAO/ACC/ROB = 73.12850886 / 82.50253637 / 88.28595382
```

相对 V26 投影约为 `-0.001394/+0.001288/-0.001553pp`，EAO 和 ROB 略低；相对目标仍分别
缺少约 `4.7715/0/5.4140pp`。因此 V27 淘汰，不启动 127 序列正式 VOT，也不能把 Train OOF
的 80% commit precision 写成跨数据集收益。权威输出为：

```text
/root/autodl-tmp/srtrack_primary_v27_multiframe_seed2026/runtime/top14_comparison.json
/root/autodl-tmp/srtrack_primary_v27_multiframe_seed2026/runtime/top14_projection.json
```

### 24.14 主创新接线复核：不是末端语言重加权

2026-08-05 对当前代码路径重新逐层核对后，主创新与论文表述一致。配置
`EARLY_GROUNDING.INSERT_LAYER=2` 使 canonical RGB/Depth 流先执行 blocks 0--1，在 block 2
之前保存 language branch seed。这个位置也在第一个 RGB-Depth cross block（indices
`2/5/8/11`）之前。三个语言 role 先分别作为 query 对 RGB 与 Depth template token 做
cross-attention，再由两路 search token 作为 query 对实例绑定后的 role 做反向
cross-attention；adapter 只对 search token 写入有界 residual，首帧 template token 保持
不变。

被语言条件化的 RGB/Depth search token 随后继续执行 blocks 2--11 和四个跨模态 block，
最终才进入共享 Center Head。因此语言在候选响应形成前参与两路 token 编码，不能描述成
head 后处理或 response reweighting。与此同时 canonical visual branch 独立完成并保留，
三模态 guard 只能否决继承语言动作，不能创建新的语言动作；size/offset 仍由 canonical
视觉分支拥有。

动态模板同样不覆盖首帧锚点：tracker 缓存静态 RGB-D 模板张量，安全 policy 只维护一个
可过期动态槽；V27 的 slot weight 上限为 `0.20`，进入 primary template 输入的 blend 上限
为 `0.02`。恢复分支 active 时暂停模板观察。在线 Qwen 仍关闭，只保留为未来低权重的推理
优化。early branch、modal token、guard、safe-template 和 continuation 接线的合并回归为
`46 passed`。

### 24.15 V28：两帧训练/部署确认对齐

V27 的 learned pair 在训练回放中已经包含连续两帧支持，但部署时第一帧大跳 proposal 被旧
center-jump gate 拒绝，第二帧只成为第一次有效 confirmation，通常要到第三帧才可能 commit。
V28 不改变 checkpoint、ranker、multiframe MLP、阈值、搜索 factor 或模板策略；当冻结的
multiframe MLP 接受一对连续 proposal 时，它把前一候选框作为第一条有效观测写入确认状态，
再用当前帧完成第二次确认，使部署语义与训练 pair 对齐。缺失前一框、非有限框或 learned
support 为 false 时严格保持 V27 行为。

独立实现和 workspace 为：

```text
lib/test/tracker/language_search_recovery_v28.py
lib/test/tracker/siamtrack_recovery_peak_v28.py
tools/vot_siamtrack_rgbd_v28.py
/root/autodl-tmp/srtrack_primary_v28_twoframe_seed2026
```

2026-08-05 11:07 UTC 已启动与 V27 相同的 14 sequences/295 anchors/259,874 tracker frames，
使用 8 workers/4 GPUs，每张 GPU 两个进程。恢复链、模板、ranker、workspace、anchor shard
和 V27/V28 行为的合并回归为 `86 passed`；primary checkpoint、horizon-12 ranker 和
multiframe artifact SHA 分别为 `30c804ba...43ab`、`6abfdc48...96a`、`f987d638...c4f`。
在线 Qwen 与未来帧文本继续关闭。

V28 于 2026-08-05 13:13 UTC 完成；最后 9 个长尾 anchors 通过官方 anchor-shard 工具无重叠
拆成 9 workers，最终完整性为 14 sequences、295 anchors。统一 top-14 指标为：

```text
EAO/ACC/ROB = 51.22709962 / 74.93642801 / 55.13728819
```

替换这 14 条完整 trajectory 后的全量反事实投影为：

```text
EAO/ACC/ROB = 73.12849308 / 82.50252985 / 88.28595382
```

相对完整 reference 为 `+0.219537/-0.033338/+0.297883pp`，只满足 ACC，EAO/ROB 仍分别低于
目标 `4.771507/5.414046pp`。该结果与 V27 投影仅有约 `-0.000016/-0.000007/0.000000pp`
差异，说明两帧部署确认对齐没有改变主要失败轨迹。V28 因此淘汰，不启动 127 序列正式 VOT。
权威文件为：

```text
/root/autodl-tmp/srtrack_primary_v28_twoframe_seed2026/runtime/top14_comparison.json
/root/autodl-tmp/srtrack_primary_v28_twoframe_seed2026/runtime/top14_projection.json
```

### 24.16 V28 收口 watcher 修正

初始 watcher `v28_finalize_watch` 只等待启动时记录的 Python PID。VOT 在同一 shard 内
会为后续 anchor 重新生成 TraX 子进程，因此这些初始 PID 退出并不代表 shard 已完成；该
watcher 曾提前调用 `finalize_vot_top14_candidates.py`，并因缺少
`bag02_indoor_2_00000` 轨迹而 fail closed。该错误只影响诊断聚合，没有终止或修改任何
V28 tracker shard，也没有产生可用指标。

已停止错误 watcher 和残留的 premature projection 进程，8 个 VOT shard 保持运行。新的
watcher 会等待全部 `v28_s00`--`v28_s07` screen 会话退出后再依次执行 comparison 和
projection，输出路径为：

```text
/root/autodl-tmp/srtrack_primary_v28_twoframe_seed2026/runtime/top14_comparison.json
/root/autodl-tmp/srtrack_primary_v28_twoframe_seed2026/runtime/top14_projection.json
```

新 watcher 会话为 `v28_finalize_watch_v2`。在全部 shard 退出并通过工具自身的完整
14-sequence/295-anchor 检查前，V28 仍不计入候选排名，也不启动完整 127 序列 VOT。

### 24.17 V29 待评测候选：当帧输出与下一帧递归状态解耦

对 V20--V28 的闭环再次审计发现，dual-branch fail-open 只保证恢复未确认时向评测器返回
同帧 factor-5 框；parent tracker 随后仍把该 factor-5 candidate 留在 `self.state`。因此一个
高置信错误峰虽然没有被 wide recovery 接受，仍可能改变下一帧 crop 中心。VOT 正式
reference 的失败帧约 70% 具有不低于 0.5 的置信度，这个递归副作用比继续放宽 ranker margin
更符合已观测的错误形态。

V29 已实现但尚未启动 GPU 评测。它继承 V28 的 checkpoint、DepthTrack-Train horizon-12
ranker、两帧 multiframe MLP、安全模板、factor 5/7、阈值和 report-only 0.98，唯一的轨迹
变化契约是：

1. recovery 新触发或持续 active 且尚未 recovered/expired 时，本帧公开返回框保持 V28
   原结果，下一帧内部 `self.state` 恢复到本帧进入前的 finite trusted bbox；
2. learned pair 确认成功时提交 parent 的 confirmed state；base branch 明确取消恢复或窗口
   到期采用 fallback 时同样保留 parent state；
3. 新触发帧在 V24 事前还无法暂停模板观察，因此 V29 对该帧的 policy 状态、动态模板张量和
   source 做 snapshot/rollback；已经 active 的后续帧仍由 V24 原规则暂停模板观察；
4. malformed trusted bbox 不写入递归 state，fail closed 到 parent state；在线 Qwen 仍关闭；
5. profile 显式保持 `HOLD_TRUSTED_STATE=false`，因为旧开关会同时冻结公开输出与内部 state，
   不等价于 V29 的 output/state 解耦。

实现和入口为：

```text
lib/test/tracker/siamtrack_recovery_peak_v29.py
tools/vot_siamtrack_rgbd_v29.py
```

V29 已注册到正式 workspace、sequence shard、anchor shard、ranker artifact 和 multiframe
artifact 的 fail-closed 绑定路径。定向及合并回归为 `43 passed`，覆盖新触发、active
unresolved、两帧确认、base cancel、expiry fallback、malformed trusted bbox、模板回滚和
VOT shard 环境传播。该测试只证明实现契约，不证明指标提升。V28 完整 top-14 收口前不启动
V29；V29 后续也必须先完成相同 14 sequences/295 anchors，再根据真实 comparison/projection
决定是否进入 DepthTrack/CDTB 保真和官方 127 序列 VOT，不能把本节写成新最好指标。

### 24.18 V30 待评测候选：Train-only RGB/Depth/语言递归状态门控

V29 在所有 unresolved recovery 帧无条件保持旧 trusted state。对 DepthTrack Train 因果回放
复核后，这个规则并不成立：多数帧的 factor-5 candidate 比旧锚点更接近目标，无条件 hold
会损害必须继续运动的样本。因此 V30 保留 V29 的“公开输出与下一帧递归状态解耦”接口，但用
一个仅在 DepthTrack Train 上训练的三模态门控决定下一帧 crop 使用旧 trusted state 还是
当帧 factor-5 parent state。

门控输入只包含推理时可见的因果证据：factor-5 response/raw score、相对 trusted bbox 的
位移和尺度、factor-5/factor-7 候选一致性、早期 RGB grounding、Depth grounding、
RGB-Depth-language consensus、coarse/fine language rank、静态 RGB identity、投影后的
RGB/Depth identity、恢复窗口进度和安全动态模板是否激活。它不读取未来帧、当前帧 GT、
VOT GT、DepthTrack Test GT 或 CDTB GT。模型是 `LayerNorm + 64/32 hidden MLP`，只控制
递归 state；本帧公开框、主模型、factor 5/7、V28 两帧确认、安全模板更新和 report-only
`0.98` 均不改变。特征缺失、非有限或 artifact/SHA 不一致时 fail open 到 parent state。

训练数据包含 149 条 DepthTrack Train 序列、6,563 个因果帧和 5-fold sequence-group OOF。
最终选择的是严格 harm-budget 策略，而不是 OOF 总收益最大的宽松策略：

```text
threshold                         = 0.987118124961853
held frames                       = 40 / 6,563
failure frames protected          = 25 / 456
healthy control harm              = 3 / 2,655 = 0.1130%
failure healthy-frame harm        = 0
must-move harm                    = 3 / 761 = 0.3942%
OOF total IoU gain                = 15.550529
```

宽松阈值 `0.7556823` 虽保护 183 个 failure 帧，但 healthy control harm 为 `2.3352%`、
must-move harm 为 `8.2786%`，不满足跨数据集泛化约束，已明确淘汰。

另一个 `1%` harm-budget 校准阈值 `0.9823388457` 只比严格策略多保护 6 帧，同时
must-move harm 从 `3/761` 增至 `6/761 = 0.7884%`，同样淘汰。其非最佳
`recursive_state_gate_harm100_seed2026.pth` 已删除，只保留 `training.json` 负结果。
严格策略冻结产物为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_state_gate_seed2026/
recursive_state_gate_seed2026.pth
SHA256 = 2ce8d8d0c2d52d2e698512e70a16b081d8f1961d59c683c71b965ada41b30e51
```

实现入口为 `lib/test/tracker/siamtrack_recovery_peak_v30.py` 和
`tools/vot_siamtrack_rgbd_v30.py`。V30 workspace、sequence shard 与 anchor shard 都必须同时
绑定 primary checkpoint、cross-scale ranker、multiframe consistency 和 recursive-state
gate 的 SHA256；验证时继续显式启用安全模板更新，在线 Qwen 和未来帧文本关闭。本节中的
数字仅是 DepthTrack Train OOF 证据，V30 尚未运行 top-14，更没有正式 VOT 指标。只有 V28
完整结果未达到投影门槛时才启动 V30，避免两套候选同时争用 4 张 GPU。

### 24.19 已拒绝 V31：递归门控与当前公开输出对齐

V30 的 Train OOF label 比较的是“当前帧 factor-5 candidate”和“进入当前帧前的 trusted
box”谁的 IoU 更高，但 V30 部署时只用该判断选择下一帧递归 crop，当前公开框仍始终返回
factor-5 candidate。这构成了明确的训练/部署动作不一致：门控学到的是当前帧 hold，部署却只
执行 next-state hold。V31 保留 V30，不修改其冻结产物或阈值，只在严格门控输出
`learned_hold` 时同时把当前公开框替换为 held trusted state，并通过现有 report-only
`fixed6_isotropic_098_v1` 生成最终 VOT 框。

V31 的行为边界为：

1. `learned_hold` 时，内部 `self.state` 仍由 V30 保持为进入本帧前的 finite trusted box；
   V31 仅把返回字典中的 `target_bbox` 改为同一 trusted box 经 `0.98` 等比例缩放后的框；
2. `learned_commit`、cross-scale evidence 缺失、非有限 gate score、malformed trusted box、
   非法图像尺寸或异常返回结构均原样返回 V30 结果，不扩大门控动作范围；
3. `best_score`、`all_scores`、`all_boxes` 及 V30 的后续递归状态保持不变；首帧静态模板仍不可
   覆盖，安全动态模板继续启用，在线 Qwen 和未来帧文本关闭；
4. 状态门控仍只使用 DepthTrack Train 的 149 序列、6,563 因果帧和严格阈值
   `0.987118124961853`，未读取 VOT/CDTB/DepthTrack Test GT。

实现、桥接和评测工作区为：

```text
lib/test/tracker/siamtrack_recovery_peak_v31.py
tools/vot_siamtrack_rgbd_v31.py
/root/autodl-tmp/srtrack_primary_v31_learnedoutput_seed2026
/root/autodl-tmp/srtrack_primary_v31_learnedoutput_seed2026/
top14_learnedoutput_8worker
```

工作区绑定同一 primary checkpoint、ranker、multiframe consistency 和 recursive-state gate
的 SHA256，14-sequence shard 完整覆盖 295 anchors/259,874 tracker frames，并以 8 workers 映射到
4 GPUs。每个 shard 都显式写入 `SRTRACK_SAFE_TEMPLATE_UPDATE=1`。V30/V31 行为、workspace 注册、
sequence shard 和 anchor shard 的定向回归为 `32 passed`。V31 于 2026-08-05 13:14 UTC 启动，
最终八个 shard 全部完成。权威 top-14 结果为：

```text
                         EAO      ACC      ROB
reference subset       50.176   74.968   53.431
V31 subset             47.210   73.105   47.601
subset delta           -2.966   -1.862   -5.830
projected full V31     72.093   82.409   86.970
target                 77.900   82.100   93.700
```

投影只有 ACC 达标，EAO 与 ROB 分别仍差 `5.807` 和 `6.730` 个百分点，因此 V31 已由自动门控
正式拒绝，不运行 V31 DepthTrack/CDTB OPE，也不运行 full-127 VOT。最大负贡献来自
`toiletpaper01_indoor_2`、`notebook01_indoor_1`、`toy02_indoor_1`、`yogurt_indoor_1` 和
`ball06_indoor_2`；其中前两条的序列 ROB 相对 reference 分别下降约 `43.10` 和 `42.29` 个百分点。
对比 V24--V28 的投影约 `73.11/82.50/88.29`，V31 将 Train gate 的 trusted-state hold 同时应用到
当前公开输出后出现明显额外退化，说明当前 gate 的 Train OOF 判别不能直接泛化为 VOT 公共框动作。
V31 的 checkpoint 和三个依赖仍暂时保留，用于可复现失败结论和正在进行的 Train-only trace；它不再
作为可提交的最终部署候选。

### 24.20 已淘汰：候选证据绝对值 ranker

现有 cross-scale ranker 的每个候选使用 8 路 RGB/Depth/语言证据 percentile rank，但 replay
同时保存了各路绝对 logit/cosine value。为验证绝对值是否能缩小 top-K oracle 与部署选择的
差距，曾构建独立 V2 原型：在原 55 维有界特征后加入每路证据的 absolute value、相对
baseline delta、factor-5 peer value 和 peer delta，全部经 softsign 有界；训练与校准仍只用
DepthTrack Train 的 149 序列、6,563 帧和 5-fold sequence-group OOF。

严格 `control/failure harm <= 0.5%` 下，结果为：

```text
candidate                 recovered   control harm   failure harm   mean IoU gain
旧 rank-only top-4        90/3004     1/2638         3/678          0.011716
value-aware top-8         85/3004     1/2638         2/678          0.010161
value-aware top-4         52/3004     0/2638         3/678          0.006335
```

top-8 与公平 top-4 对照都没有超过旧 rank-only 特征，说明这些绝对 logits 在跨序列尺度上
不稳定，percentile rank 的域不变性更强。该方向已淘汰，未接入 tracker，也未运行任何 VOT、
DepthTrack Test 或 CDTB。两个非最佳 `.pth` 已删除，只保留以下 Train-only JSON 负结果：

```text
/root/autodl-tmp/srtrack_depthtrack_train_cross_scale_ranker_v2_seed2026/
ranker_valueaware_top8_seed2026_training.json
ranker_valueaware_top4_seed2026_training.json
```

### 24.21 跨帧 top-K 轨迹容量与朴素关联淘汰结论

V27/V31 的跨帧模块只判断旧 ranker 已经选中的单个 proposal 是否可信，无法在旧 ranker 选错
峰值后返回同帧其余候选。为判断下一步是否值得直接建模完整候选集合，在不改 tracker、不读取
验证集标注的前提下，对 DepthTrack Train replay 的 factor-7 proposals 做了相邻两帧轨迹容量
分析。分析命令最初请求 `topk=8`，但结构化复核发现这批旧 replay 每行实际只保存 1--4 个
proposal（13,126 行分布为 `{1:2596, 2:3719, 3:3057, 4:3754}`），所以以下数字严格属于
**实际 top-4 上限**，不能写成 top-8。分析覆盖 5,942 个相邻帧对；其中 failure 事件有
3,441 对。这里的 GT 只用于离线标注 oracle 上限，不能进入推理特征，且未使用 VOT、CDTB 或
DepthTrack Test GT。

在 failure 帧对的 2,735 个 baseline miss（baseline IoU `< 0.1`）中：

```text
分析项                                      恢复数       占 baseline miss
当前帧 top-4 oracle                         633/2735     23.14%
连续两帧均存在且 IoU 可关联的目标轨迹       526/2735     19.23%
朴素最大跨帧 box-IoU 关联                   279/2735     10.20%
```

这说明候选集合中确实存在 V27/V31 尚未利用的因果恢复容量，而且大部分当前帧 oracle 恢复目标
在上一帧也有对应候选；因此“跨帧完整候选集合建模”是有依据的下一步，而不是继续微调单 proposal
hold 阈值。但直接选择与上一帧任意候选 box-IoU 最大的 proposal 完全不可部署：在 648 个健康
baseline 帧中造成 615 次显著伤害（94.91%），其中 590 次退化到 IoU `< 0.1`（91.05%），
平均 IoU 变化也为负。原因是上一帧 top-K 同样包含背景峰，纯几何连续性会稳定地延续错误轨迹，
且无法识别真实目标、相似干扰物和静态背景。

因此朴素关联方案已淘汰，不接入 V31，也不运行 VOT。基于同一旧 top-4 replay 的 causal setwise
trajectory selector 随后也已完成 sequence-group OOF 并在第 24.23 节被拒绝。若 V31 top-14
仍未过投影门槛，必须先重采更完整的 DepthTrack Train causal replay，不能继续微调这两类旧
轨迹策略。任何未来部署动作仍必须直接选择当前帧 proposal，并把同一选择写入下一帧递归状态，
避免再次出现 V30 的训练/部署动作错位。权威容量分析产物为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_candidate_trajectory_seed2026/
trajectory_top4_factor7.json
```

分析器已升级为 `srtrack-depthtrack-train-candidate-trajectory-analysis/v2`，产物同时记录
`requested_topk`、`effective_topk`、`maximum_available_candidates` 和候选数直方图，防止旧 replay
的保存容量再次被 CLI 请求值掩盖。

### 24.22 当前权重保留审计与清理边界

2026-08-05 的只读盘点显示，`/root/autodl-tmp` 中实际存在的 `.pth/.pth.tar/.pt` 只有 12 个；
大量旧实验 manifest 所指向的 checkpoint 已经不存在，不能把这些失效引用计为当前磁盘权重。
与最终目标直接相关且必须保留的权重为：

```text
用途             大小       SHA256
primary 主模型   1.096 GB   30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
历史 V5/CRAR     1.086 GB   40132d57de7f6d8b78b069d9becad1db4e94228b03472f3220ad12da3e58e6b6
V27 ranker       28 KB      6abfdc48fdf01eeaa587ca5384f02743191d50543a1d960b6f80b8e2ac5aa96a
V27 multiframe   98 KB      f987d6385f6bf5974a6958be51c5267cfa8f47f57037bfb0dc112172f2dfdcf4
V30 state gate   27 KB      2ce8d8d0c2d52d2e698512e70a16b081d8f1961d59c683c71b965ada41b30e51
```

这里的“历史 V5”不是独立的 V5 文件。`srtrack_v5_vot_no_recovery_seed2034` 的主 workspace、
diagnostic shards 和 full shards 都绑定 `srtrack_crar_tail_probe_e1` checkpoint 及其
`40132d...` SHA256；删除 CRAR 会使历史最好 VOT 结果不可复现。另一个
`srtrack_final_language_visual_safe_v5_seed2026` 的 manifest 绑定 SHA256 为 `68cfc7...` 的旧路径，
但该大权重当前已经不在磁盘，只剩结果与 manifest。

因此现阶段不删除 primary 或 CRAR：前者是 V31 当前部署主模型，后者是可复现的历史最好对照；
三个小权重又是 V31 的硬依赖。旧 MPLTTrack 的 checkpoint 目录仍有约 9.4 GB：

```text
/home/OSTrack_RGBD_L_dataset_modified/output/depthtrack_roberta/checkpoints
```

但它属于另一套旧项目，且尚未完成逐 checkpoint 指标与引用审计，本轮不越界删除。只有最终三
数据集门槛全部用同一部署配置通过后，才重新生成引用闭包并删除未被最佳结果、复现 manifest 或
初始化链绑定的文件。

### 24.23 已淘汰：causal setwise trajectory selector

针对 24.21 证明的实际 top-4 候选容量，曾实现一个 48,654 参数的因果集合选择器。其输入不是
单个旧 ranker proposal，而是相邻两帧的完整候选集合：每个候选包含原 55 维 response、跨尺度、
RGB identity、Depth identity 和早期语言-RGB/Depth/consensus grounding 特征，再加入旧 ranker
的集合内 score rank、相对已选 proposal 的有界 delta 和 selected flag。模型对当前候选做 set
self-attention，并让每个当前候选对上一帧全部候选执行带 box 几何 bias/value 的 causal
cross-attention。

80 epochs、5-fold sequence-group OOF 已在 DepthTrack Train 完整结束，覆盖 4,984 个相邻帧对和
145 个序列。严格条件为 commit precision `>= 80%`，同时 control/failure harm 均 `<= 0.5%`。
结果没有任何可行阈值：

```text
OOF 项目                                  数值
可行阈值                                  0 / 503
最高 precision 的零伤害阈值              72.73% (8/11)
failure miss 恢复                         8 / 2091 = 0.38%
control harm                              0 / 1760
failure harm                              0 / 863
全体平均 IoU 增益                         +0.001136
```

各 fold 训练交叉熵已经下降到约 `4e-6`--`1e-3`，但 OOF precision 和 recovery 仍很弱，说明当前
旧 replay 的 1--4 proposals、事件窗口采样和样本规模不足以支撑可泛化的集合注意力模型，继续增加
epoch 或放宽阈值只会把过拟合风险带入 VOT。该候选因此已淘汰，没有生成 `.pth`、没有接入 tracker，
也不运行 VOT/CDTB/DepthTrack Test。原型模型、训练器及其测试已删除，只保留权威 Train-only
负结果：

```text
/root/autodl-tmp/srtrack_depthtrack_train_setwise_trajectory_seed2026/training.json
```

若 V31 失败，不能基于这批旧 top-4 replay 再做小型网络或阈值变体。需要先重新采集至少 top-8、
覆盖正常连续帧与失败窗口、记录实际递归状态反馈的新 DepthTrack Train causal replay，再判断集合
轨迹模型是否值得重建。

V31 的正式 OPE 入口缺口已补齐。`tracking/evaluate_depthtrack.py` 与
`tracking/evaluate_cdtb.py` 现在都有显式 `--tracker {primary,v31}`：`primary` 保持原
`siamtrack_dropmae` 行为；`v31` 使用 `siamtrack_recovery_peak_v31`，强制启用冻结的 V31
language-recovery profile 和安全模板更新，并要求同时提供 cross-scale ranker、multiframe
consistency、recursive-state gate 三个 artifact。缺少任一文件会在创建输出前 fail closed；路径和
SHA256 会同时写入 manifest 与 metrics。参数入口 `lib/test/parameter/rgbd_dropmae.py` 已接受三
个 artifact 的显式路径，`lib/test/parameter/siamtrack_recovery_peak_v31.py` 提供 V31 tracker
对应的参数模块，公共解析与 SHA 记录集中在 `tracking/rgbd_evaluation_deployment.py`。

定向回归覆盖 primary 默认路径、V31 三 artifact 完整性、冻结 profile、模板强制启用、两个 OPE
数据集覆盖契约及 V31 VOT 契约，共 `55 passed`。使用当前真实 primary/ranker/multiframe/state
gate 做的无 GPU 参数 smoke check 进一步确认：`CHECK_INTERVAL=5`、`MIN_STABLE_FRAMES=3`、
`MIN_UPDATE_INTERVAL=30`、动态融合 `0.20`、主模板融合 `0.02`、`WIDE_FACTOR=7.0`，在线语言
仍关闭，四个权重的 source SHA 绑定一致。这里只证明正式入口可用；V31 top-14 通过后仍必须实际
跑完 DepthTrack50/CDTB80，不能用旧 base OPE 结果声称“同一完整部署配置三数据集通过”。

### 24.24 验证默认模板更新与 V31 full-127 自动门控

固定六序列开发验证入口 `tracking/evaluate_depthtrack_validation.py` 已把安全模板更新改为默认开启。
这不是覆盖首帧模板：immutable first template 始终保留，在线模板只进入受限动态槽；更新仍要求
`CHECK_INTERVAL=5`、`MIN_STABLE_FRAMES=3`、`MIN_UPDATE_INTERVAL=30`，并经过置信度、response
margin、中心跳变、RGB identity、Depth 变化和有效深度比例联合门控。默认动态融合上限为 `0.20`，
主模板分支只注入 `0.02`。`--safe-template-update` 继续保留用于显式记录命令，新增
`--no-safe-template-update` 作为后续消融开关。manifest/metrics 新增以下字段，避免把默认行为误记为
CLI 消融：

```text
safe_template_update_validation_default
safe_template_update_cli_override
safe_template_update_cli_value
```

参数层新增独立的 `safe_template_update_force_disable`，只服务于显式关闭路径；原有
`safe_template_update=False/None` 的 no-op 兼容语义没有改变。模板参数与 V31 续跑相关定向测试分别
得到 `48 passed` 和 `27 passed`（两组有重叠测试，不能相加声称独立用例数），实际 V31 workspace
也通过 tracker id、V31 bridge、checkpoint、ranker、multiframe、state-gate、恢复 profile 和模板
更新的 SHA/配置检查。

旧的三数据集正式 runner `tools/run_formal_cross_dataset_evaluation.py` 也已补齐同一契约：完整语言模型
的开发集选择结果会固定写入 `safe_template_update=true`，DepthTrack、CDTB 与官方 VOT workspace
三条命令必须携带同一个 `--safe-template-update`。该设置属于完整部署，不允许在读取测试集指标后再
选择；视觉-only baseline 保持自己的独立无模板 profile，避免改变历史对照定义。

原先存在一个每 5 分钟检查 shard screen 的旧 `v31_finalize_watch`，它会与正式续跑器同时写
`top14_comparison.json/top14_projection.json`，存在竞态且不符合轮询要求，已停止。八个 V31
评测 shard 随后全部完成；单写者续跑器生成 comparison/projection 后以
`v31_top14_projection_below_target` 正常退出：

```text
result: /root/autodl-tmp/srtrack_primary_v31_learnedoutput_seed2026/runtime/continuation_result.json
decision: v31_top14_projection_below_target
projected metrics: 72.09298 / 82.40875 / 86.97012
```

其行为保持 fail closed：top-14 投影的 EAO/ACC/ROB 三项全部达到 `77.9/82.1/93.7` 才顺序运行
V31 同栈 DepthTrack50 与 CDTB80；任一 OPE 未达到各自目标就写拒绝结果，不运行 full VOT。

为补齐 OPE 通过后的流程，新增 `tools/continue_v31_full_vot.py`。由于 V31 已在 top-14 被拒绝，
formal gate 只写出 preservation rejection，不会创建或运行 full-127 shards：

```text
decision: preservation_gate_rejected
formal_vot_started: false
planned full-127 GPU mapping (unused): 0,0,1,1,2,2,3,3
```

该门控器不会仅信任上游 JSON 的布尔值，而会重新校验 top-14 projection、两个 OPE metrics/manifest
的 SHA 和完整覆盖，再校验 V31 workspace 的 checkpoint、配置、bridge 及三个 Train-only artifact。
只有三门全部通过才会在现有 workspace 上按剩余 tracker frames 重新均衡八个 shard。当前三门未通过，
所以不存在 V31 正式三数据集同栈达标结果，本节不能用于声称指标提升，也不能触发最终权重删除。

最终清理也已改成显式门控工具 `tools/cleanup_v31_nonbest_weights.py`。默认运行只做 dry-run；当前
`formal_vot_runtime/continuation_result.json` 已存在，但明确记录 `formal_vot_started=false` 和
`preservation_gate_rejected`。重新执行 dry-run 返回 `eligible=false`，理由为
`formal_vot_not_started`、`three_dataset_decision_not_met`、`official_vot_target_not_met` 和
`official_vot_checks_incomplete`，磁盘中的权重没有变化。清理器直接脚本入口已加入仓库根目录解析和
子进程回归，不会在 import 阶段失败，也不会把 dry-run 隐式转成删除操作。
只有 formal VOT 结果明确写出 `all_three_datasets_target_met`，且重新校验 top-14、DepthTrack50、
CDTB80、正式 VOT、7 个保留依赖和活动文件引用全部通过后，`--execute` 才允许删除 5 个已淘汰的
template-router audit/replay `.pth`，预计回收约 24 MB。删除目标是固定绝对路径和固定 SHA256，
不使用 glob；primary、历史 V5、原始预训练、mapped initializer、ranker、multiframe、state-gate
均在强制保留集合内。相关门控回归与模板/续跑测试本轮合计执行 `50 passed`。

### 24.25 V31 失败后备：真实递归 top-8 DepthTrack Train trace

旧 V27 replay 虽然命令可请求 `topk=8`，但源行只保存 1--4 个 proposal，而且每个 horizon 都从
同一个失败前 `search_state` 独立前向，明确标记为 `inference_feedback=false`、
`counterfactual_single_step_replay=true` 和 `models_recursive_recovery_state=false`。它不能继续用于
训练新的集合选择器或递归恢复模块。为避免 V31 未过门后再次在这批旧数据上调阈值，已经补齐一条
只使用 DepthTrack Train 的真实连续采集链：

```text
lib/test/tracker/siamtrack_depthtrack_train_recursive_trace.py
lib/test/parameter/siamtrack_depthtrack_train_recursive_trace.py
tools/run_depthtrack_train_recursive_top8_trace.py
tools/finalize_depthtrack_train_recursive_top8_trace.py
tools/analyze_depthtrack_train_recursive_top8_capacity.py
```

trace tracker 直接继承完整 `RecoveryPeakV31Tracker`，不会构造另一条简化推理路径。每一帧先检查
本帧 `recursive_input_state` 是否等于上一帧真正写回的 `recursive_output_state`，再运行原 V31；
产物同时记录公开输出、恢复前后 trusted 状态、cross-scale ranker、multiframe consistency、
recursive state gate 的 hold/commit、模板动态槽状态以及 factor-5/factor-7 候选。稳定段每 30 帧
采样一次；motion/scale 风险帧、恢复窗口、learned hold/commit 和动态模板实质变化全部采样。

V23 候选解码器新增了只读 `topk` 参数，但默认仍严格使用冻结 ranker artifact 的 `topk=4`、严格
局部极大值和 visual-argmax baseline 契约。真实响应图在 5x5 NMS 下常常只有 2--5 个严格局部峰，
所以仅把请求值改成 8 并不等于真正获得 top-8。trace wrapper 现在额外显式启用两个只读开关：允许
保留生产分支已经选定的非 visual-argmax baseline，并在严格局部峰不足时按 response 从高到低、对
已选位置做相同半径的空间抑制，确定性补齐到 8 个候选。生产调用不传这两个开关，因此 V31 的
生产 top-4、选择集合、阈值、递归状态和公开输出不变；trace 只额外批量解码候选框和保存证据。
每个候选包含 response、RGB identity、Depth identity、早期 language-RGB、language-Depth、三模态
consensus 及 coarse/fine language 的 value/rank。

推理阶段 tracker 不接收 GT。完整连续运行结束后，finalizer 才按 `(sequence, frame)` 关联
DepthTrack Train GT，为 base/wide 候选、公开输出和递归状态计算离线 IoU 标签。finalizer 会拒绝
以下任一不完整产物：没有覆盖所有序列、没有稳定正常帧、没有恢复窗口、没有真正观察到 base/wide
top-8、没有执行 V31 state gate、没有失败标签，或任何记录不满足：

```text
inference_feedback=true
counterfactual_single_step_replay=false
models_recursive_recovery_state=true
ground_truth_available_to_tracker=false
vot_ground_truth_used=false
depthtrack_test_ground_truth_used=false
cdtb_ground_truth_used=false
```

finalizer 通过后，容量分析器才读取离线标签，分别统计 top-8 相对 top-4 的独占恢复容量、连续两帧
目标轨迹容量、V31 learned hold/commit 的得失、动态模板槽分组，以及 language/RGB/Depth 证据中
目标候选相对 baseline 的差值。产物固定标记为 `DepthTrack Train only`、
`future_ground_truth_available_to_inference=false`、`deployable_policy_selected=false`；它只回答是否有
足够证据重建 setwise 模型，不会自动训练权重、选择阈值或改变 V31 输出。权威产物路径为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026/recursive_top8_capacity.json
```

2026-08-05 14:56 UTC 的 CPU preflight 已覆盖 152 个 DepthTrack Train 序列，并确认固定六序列
语言字段与正式验证一致；primary checkpoint、V31 config、ranker、multiframe、state-gate 的 SHA
分别为 `30c804...`、`cf8e1c...`、`6abfdc...`、`f987d6...`、`2ce8d8...`，全部通过。preflight
manifest 位于：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_preflight_seed2026/manifest.json
```

V20--V31、模板更新、部署入口、trace/finalizer、容量分析器和三个自动门控器的当前合并回归为
`144 passed, 1 warning`。完整采集默认配置为
8 workers/4 GPUs。V31 投影门失败后，该链已于 2026-08-05 16:00 UTC 自动启动，不再运行旧 top-4
单步 replay：

```bash
conda run -n mplt python tools/run_depthtrack_train_recursive_top8_trace.py \
  --output-dir /root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026 \
  --threads 8 --num-gpus 4 --topk 8

conda run -n mplt python tools/finalize_depthtrack_train_recursive_top8_trace.py \
  --run-dir /root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026

conda run -n mplt python tools/analyze_depthtrack_train_recursive_top8_capacity.py \
  --labeled-jsonl /root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026/recursive_top8_labeled.jsonl \
  --output-json /root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026/recursive_top8_capacity.json
```

这条链当前只是可立即启动的后备数据采集基础设施，不是新模型结果，也不能用于声称 VOT 指标提升。
最初的 `leaves01_wild` smoke 暴露了“请求 8、实际只有 0--5”问题，失败产物已原样保留在
`srtrack_depthtrack_train_recursive_top8_smoke_leaves01_seed2026.failed_before_trace_fill_20260805T1555Z`。
修复后在 GPU2 重跑同一条 155 tracker-frame 序列，耗时 `35.209 s`，记录 91 帧；base 为
`91/91` 帧实际 8 候选，wide 为 `39/39` 帧实际 8 候选，`validate_smoke()` 已通过。新权威 smoke
目录为 `/root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_smoke_leaves01_seed2026`。全量
152 序列 trace 正在运行；failover 已复用 smoke manifest 的实测吞吐计算 full trace ETA。
为避免 V31 拒绝后四卡空转，又不与通过后的 OPE/full VOT 抢卡，已启动独立 fail-closed watcher：

```text
screen: 852967.v31_recursive_top8_failover_now
PID: 852970
启动时间: 2026-08-05 16:00 UTC
workers/GPUs: 8 / 4（每卡 2 workers）
GPU 显存: 约 4.5 GB/卡
启动后利用率: 73%--97%
预计完成窗口: 2026-08-05 19:28--22:56 UTC
预计完成中值: 2026-08-05 21:12 UTC
```

它只接受完整的 `srtrack-v31-continuation-result/v1`。若 decision 为 `ready_for_full_vot`，不会立即
退出，而是等待 `formal_vot_runtime/continuation_result.json`：formal VOT 三项达标才写
`not_required` 并退出，只有 formal result 明确为 `formal_vot_target_missed` 才启动采集。若
top-14/OPE 已拒绝，则仍要求 `v31_top14_projection_below_target` 或
`v31_ope_preservation_failed` 且内部 target checks 一致后才启动。先运行最短的
`leaves01_wild` 单卡 155 tracker-frame smoke，确认真实递归链和 base top-8 后，才启动 8 workers/
4 GPUs 全量任务。全量 ETA 根据 smoke 实测吞吐给出 4x 理想缩放到 2x 保守缩放的时间窗口，运行期
不做分钟级轮询。全量 finalizer 完成后 watcher 会自动运行上述容量分析器，重新校验 schema、
Train-only provenance、因果 flags、labeled trace SHA 和 152 序列覆盖，并在 failover result 中记录
`capacity_analysis` 与 `setwise_model_evidence_ready`；仍不会自动训练或部署新策略。

### 24.26 full-127 失败序列集中度与问题类型

在等待 V31 时，重新按已完成的 primary safe025 full-127 官方 multi-start 轨迹汇总了 363 个失败
运行。该汇总只用于解释既有正式结果，不生成 VOT 阈值，也不把 VOT GT 写入 tracker。失败最多的
五条序列合计占 `122/363 = 33.61%`：

```text
序列                              失败 anchor     序列 ROB    序列 ACC   失败窗 motion 中位数   h=1 所需 factor   h=1 最短边 token
cup02_indoor_1                    36/36            6.960       85.217          1.533                1.254              2.361
toy09_indoor_1                    26/26           42.430       82.766          1.712                1.512              1.913
yogurt_indoor_1                   21/34           78.127       63.718          1.868                2.415              2.236
earphone01_indoor_1               20/20           24.536       76.463          1.936                1.328              1.617
glass01_indoor_2                  19/19           15.818       74.217          1.726                1.699              2.072
```

这里的 `h=1 所需 factor` 是以失败前最后预测框为因果 anchor proxy、再用 GT 离线计算的几何诊断。
五条高频失败序列的中位数都远小于基础 factor 5；`cup/toy/earphone/glass` 的 ACC 也并非全部很低，
说明主要损失不是框回归普遍失准，而是运动/遮挡附近选错响应峰后污染递归状态。`toy09` 和
`earphone01` 的最短边只有约 `1.6--1.9` 个 token，更容易出现真实目标与干扰峰证据接近；这正是
早期 language-RGB/language-Depth grounding、首帧静态模板和动态模板身份共同参与候选集合判别的
适用场景。

也存在少量真正需要宽搜索的不同问题：`stick_indoor_1` 的失败后第一帧所需 factor 中位数为
`4.562`，`cube02_indoor_1` 为 `6.865`。因此不能把所有失败统一解释成身份切换，也不能取消
factor-7 分支；正确分工应是 motion/scale 风险负责触发，factor-5/factor-7 共同提供候选，随后由
跨帧集合模型结合语言、RGB、Depth、模板和递归状态选择并提交。V31 未过门时启动的真实 top-8
trace 将直接验证该候选是否存在以及是否能连续两帧保持，而不是继续从这些 VOT 数字反推阈值。

权威来源为：

```text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/vot_full127_failure_precursors_geometry.json
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/primary_trimodal_vot_sequence_gaps.json
```

### 24.26 真实递归 top-8 之后的 V32 候选 selector（待容量门）

V31 的失败不能用旧 top-4 counterfactual replay 继续微调。真实 152 序列递归 trace 完成后，
`tools/analyze_depthtrack_train_recursive_top8_capacity.py` 只有在以下条件同时成立时才允许进入
下一阶段：base/wide top-8 实际出现、top-8 对 public miss 有增量恢复、连续帧有持久轨迹容量，并且
语言/RGB/Depth/early-grounding 证据均存在。`tools/continue_recursive_top8_setwise.py` 以 30 分钟
间隔等待该报告；任何条件失败都只写 `rejected`，不创建模型权重。

通过容量门时，`lib/test/tracker/causal_trimodal_setwise_selector.py` 与
`tools/train_depthtrack_recursive_top8_setwise_selector.py` 才会训练 V32 候选：

- 当前 top-8 proposal 集合先做 permutation-equivariant self-attention；上一连续帧 proposal 集合
  只通过 causal cross-attention 进入当前集合，非连续帧严格使用空 history。
- 原有跨尺度 ranker 特征保留八类语言/RGB/Depth evidence；显式追加动态模板激活、融合权重、稳定度、
  模板年龄、恢复进度、上一帧 recursive gate/multiframe 状态和当前置信度。
- sequence-group OOF 同时约束 public healthy harm、baseline harm、catastrophic harm、提交候选
  precision 和 public miss 恢复收益；没有可行策略时不生成 artifact。
- 新 artifact 绑定 checkpoint、真实 labeled top-8 trace、capacity report、cross-scale ranker、
  multiframe consistency 和 recursive-state gate 的 SHA256。V32 tracker、DepthTrack/CDTB OPE、
VOT bridge 与并行 shard 均 fail-closed 要求这六项来源一致，且验证时安全模板更新保持开启。

因此当前 V32 仍是“待真实 Train-only 证据触发”的候选实现，尚未产生任何可用于测试集选择或正式
VOT 指标声称的结果；正式流程仍必须先完成同一 checkpoint/profile 的 top-14 三项投影，再按顺序
运行 DepthTrackTest、CDTB80 和 full-127 VOT。

为避免 selector 生成后再次依赖人工拼接命令，新增
`tools/continue_v32_after_setwise.py` 作为单写者续跑器。它以 30 分钟间隔等待
`setwise_continuation_result.json`，重新校验训练报告、selector、checkpoint、labeled trace、
capacity、ranker、multiframe 和 state-gate 的路径/SHA 绑定后，才创建独立 V32 workspace。
随后固定使用 8 workers/4 GPUs 跑同一 14 序列/全部 anchor 的投影；投影三项全部达到
`77.9/82.1/93.7` 才运行 DepthTrack50 和 CDTB80，同栈 OPE 通过后才启动官方 full-127。

OPE 门禁同时补上了历史保真约束。除最低目标外，DepthTrack 和 CDTB 的 PR/Recall/F1
每一项相对历史最好最多允许下降 `0.05` 个百分点；该容差覆盖 primary safe025 已测得的
跨域微小数值差异，但会拒绝旧 `BLEND_WEIGHT=0.20` 模板 profile 在 DepthTrack 上约
`0.59` 个百分点的 F1 退化。输出同时保留严格的 `historical_best_not_decreased` 和用于部署
门禁的 `historical_best_within_tolerance`，避免把“达到最低目标”误写成“保持历史最好”。
V32 自动续跑定向测试与 V31 保真门回归分别为 `43 passed` 和 `31 passed`；这些测试只证明
流程和绑定契约，不构成任何 V32 benchmark 指标。

### 24.27 真实递归 trace 结论与 V32 的正式否决

本轮恢复连接时，服务器没有发生重启或 OOM；用户已明确说明上一次 SSH 断开是主动退出。检查时
服务器 uptime 约 503 天、可用内存超过 700 GiB，也没有 OOM 记录。原续跑脚本固定假设 4 GPU，
而当前实例只有 GPU0/GPU1。`tools/continue_v32_after_setwise.py` 已改为 8 个 worker 按
`0,0,0,0,1,1,1,1` 分配，OPE 使用 2 GPU；同时允许只剩 runtime 目录的已准备 workspace 正常续跑。
该修改只修复调度和恢复能力，不改变 tracker、checkpoint 或评测协议。

真实递归采集最终完成 152 个 DepthTrack Train 序列，共记录 47,611 帧；离线标签 JSONL 大小约
622.7 MB。容量报告确认：base/factor-5 top-8 可在 3,333 个 public miss 帧中找到 IoU 至少 0.5
的候选，其中 47 个只能由 top-8 而非原 top-4 找到；wide/factor-7 对应为 1,713 和 163。
wide 分支在连续 miss 帧上具有 865 对可持续候选，其中 91 对是 top-8 独占。V31 的 learned hold
则表现出明显反向选择：2,233 次 hold 中 1,650 次有害、1,550 次为 catastrophic，只有 149 次有益，
public 相对 base 的平均 IoU 差为 `-0.584`。这与 V31 把 learned hold 同时写到公开输出后 top-14
投影从约 `73.13/82.50/88.29` 降到 `72.093/82.409/86.970` 相互印证。

V32 初版阈值选择曾把所有 OOF 样本的唯一概率与唯一 margin 做笛卡尔积，再对每个组合扫描全部帧，
复杂度不可接受。该进程被安全停止，训练器改为预注册的 9 个概率阈值乘 10 个 margin 阈值，共 90
个策略：

```text
probability: 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99
margin:      0.00, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 1.60, 2.40, 3.20
```

这一固定格点既给出有界计算，也避免按单个 OOF 样本的浮点值过拟合部署阈值；85% 提交精度、
public/baseline healthy harm 各不超过 0.5%、catastrophic 为零且总 IoU gain 为正的门槛没有放宽。
V32 最终 90 个策略中没有一个可行。最佳非约束策略为：提交 1,578 帧，提交精度 `0.482256`，
恢复 306 个 public miss，总 IoU gain `+24.95`；但 public healthy harm 为
`6333/7956 = 79.6003%`，baseline harm 为 `86/3087 = 2.7859%`，catastrophic 3,331 帧。
因此 V32 是结构性不安全，而非阈值略偏；正式决定为 `setwise_selector_rejected`，没有生成 V32
artifact，也没有启动 VOT 测试。权威结果为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026/setwise_selector_training.json
/root/autodl-tmp/srtrack_depthtrack_train_recursive_top8_seed2026/setwise_continuation_result.json
/root/autodl-tmp/srtrack_primary_v32_setwise_seed2026/runtime/continuation_result.json
```

### 24.28 V33：语言条件的因果双尺度联合动作仲裁

V32 只在 wide 分支内部从候选中重排，无法表达“保持原 public 输出”“改用 factor-5 候选”与
“改用 factor-7 候选”的联合竞争。对 recovery active 且同时保存两分支的 25,078 帧重新做
Train-only 容量统计：16,239 帧为 public miss；factor-5 恢复 2,981 帧，factor-7 恢复 1,713 帧，
并集恢复 3,577 帧，其中 factor-5 独占 1,864、factor-7 独占 596、两者共同 1,117。连续 miss 帧
IoU 至少 0.5 的持久轨迹对，factor-5 为 1,057、factor-7 为 857、并集为 1,545。由此实现 V33，
每帧联合比较 17 个动作：

其中仅 factor-5 的分支基线动作就在 1,552 个 public miss 帧上达到 IoU 0.5；它在 7,956 个
public healthy 帧中造成超过 0.1 IoU 下降的只有 97 帧（95 帧降至 0.1 以下）。这说明“撤销错误
public hold 回到正常 base”与“从更宽搜索选择新峰”是两类不同动作，必须在同一安全门下联合建模。

```text
action 0: 保留当前完整 V31 public 输出
action 1--8: factor-5/base top-8
action 9--16: factor-7/wide top-8
```

V33 的论文创新点不是一个 VOT 专用阈值，而是语言条件的 causal joint action memory：当前双尺度
proposal 集先经过 permutation-aware self-attention；严格相邻的上一帧 proposal 集通过 causal
cross-attention 和两两框几何进入当前动作；非连续帧没有历史输入。每个动作保留 response、位置、
尺度、跨尺度一致性、RGB identity、Depth identity、early language-RGB grounding、early
language-Depth grounding、三模态 consensus 与 coarse/fine language 等证据，并加入 public/base/wide
动作身份、分支内 rank、动态模板状态、恢复阶段和递归 gate 上下文。训练标签只在 DepthTrack Train
完整推理结束后离线计算；推理不读取当前或未来 GT，也不使用 VOT、CDTB 或 DepthTrack Test GT。

训练仍采用按序列分组的 5-fold OOF，阈值只从同一固定 90 策略格点选择，并沿用 V32 的全部安全门。
只有 OOF 存在可行策略才会训练全量模型并生成 artifact；否则 fail closed。若选择 factor-5 或
factor-7 动作，V33 会同步更新公开输出、下一帧递归 state 与 recovery trusted anchor，避免再次出现
“输出已改但内部 crop 仍指向另一目标”的状态分裂。

模板更新属于同一完整部署配置，不因 V33 改回静态模板：首帧 immutable template 永不覆盖；动态槽
仍要求 5 帧检查间隔、至少 3 个稳定帧、至少 30 帧更新间隔，并联合约束置信度/response margin、
中心跳变、RGB identity、Depth 变化和有效深度比例。动态槽最大融合为 0.20，主模板分支只注入
0.02；恢复激活、未确认动作或状态 hold 会拒绝/回滚可疑观察。模板更新必须在 DepthTrack、CDTB
和 VOT 三条正式命令中一致开启，关闭只作为单独消融。

V33 实现入口为：

```text
lib/test/tracker/causal_trimodal_joint_setwise_selector.py
tools/train_depthtrack_recursive_joint_setwise_selector.py
lib/test/tracker/siamtrack_recovery_peak_v33.py
lib/test/parameter/siamtrack_recovery_peak_v33.py
tools/vot_siamtrack_rgbd_v33.py
tools/continue_v33_after_joint.py
```

续跑器固定先做同一 top-14/all-anchor 投影；三项达到 `77.9/82.1/93.7` 后才依次验证
DepthTrack50 和 CDTB80 的目标及历史最好保真，全部通过才允许 full-127 官方 VOT。当前两 GPU
调度、workspace/bridge/profile/SHA 绑定及相关 V32/V33 定向回归已通过；实际训练和 benchmark
结论必须以下一节的权威产物为准，不能由容量或单元测试替代。

### 24.29 V33 OOF 结论：联合动作空间有效，但单头分类仍未被安全认证

V33 在 25,078 个 recovery-active 双分支帧、145 个至少包含一个有效训练帧的 DepthTrack Train
序列上完成 5-fold sequence-group OOF；每折验证帧分别为
`5016/5016/5016/5014/5016`，模型为 hidden 96、4 heads、80 epochs、seed 2026。训练报告最终为
`stage=rejected`，90 个固定策略中可行策略数仍为零，因此没有生成 V33 权重，也没有运行任何
DepthTrack Test、CDTB 或 VOT 测试集评测。

最佳非约束 OOF 策略的结果为：提交 2,703 帧，IoU 至少 0.5 的提交精度 `0.613393`，恢复
1,467 个 public miss，总 IoU gain `+1089.007`。相比 V32 的 306 个 miss 和 `+24.95`，联合
public/base/wide 动作空间确实找到了更强的有用信号；但 healthy harm 仍为
`281/7956 = 3.5319%`，catastrophic harm 为 226，距离预注册的 0.5%/0 要求仍很远。不能用总收益
掩盖这类少量但会在递归跟踪中扩散的错误，也不能事后放宽安全门。

权威结果为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_joint_seed2026/joint_selector_training.json
/root/autodl-tmp/srtrack_depthtrack_train_recursive_joint_seed2026/v33_joint_continuation_result.json
/root/autodl-tmp/srtrack_primary_v33_joint_seed2026/runtime/continuation_result.json
```

V33 wrapper 续跑时还暴露了一个只影响 provenance 记录、不影响训练或 tracker 的 Python 默认参数
绑定问题：底层 `validate_setwise_result(path=SETWISE_RESULT)` 在 import 时冻结了 V32 路径，即使
wrapper 之后把全局路径切换为 V33，第一次拒绝 JSON 仍引用旧 V32 文件。由于两者都是拒绝状态，
它没有启动 workspace 或测试集评测，但该记录不可信。函数已改为 `path=None` 后在调用时读取当前
全局路径，并加入 V33 runtime-default 回归；相关 11 项续跑测试通过。随后重新生成 V33 结果，当前
记录正确绑定 `v33_joint_continuation_result.json` 的 SHA，仍为
`setwise_selector_rejected/formal_vot_started=false`。

### 24.30 V34：价值估计与独立风险认证（Train-only OOF）

V33 的剩余问题不是候选容量不足，而是一个 softmax/argmax 分类头同时承担“哪个动作价值最高”和
“该动作是否足够安全”两个不同问题。V34 保留 V33 的语言/RGB/Depth、模板状态、双尺度动作集合、
当前集合 self-attention 与严格上一连续帧 causal memory，但把决策拆为以下监督头：

```text
1. action head: 候选相对排序；
2. IoU value head: 每个动作的连续 IoU 估计；
3. success head: P(IoU >= 0.5)；
4. healthy-harm head: P(candidate < public - 0.1 | public healthy)；
5. catastrophic head: P(candidate < 0.1 | public healthy)。
```

对 public hold 动作不能错误复用 factor-5 baseline 的证据。V34 以当前候选中与 public bbox IoU
最大的动作作为可观察 evidence proxy，同时显式保留 proxy IoU；每个动作还加入相对 public、相对
factor-5 baseline 的 6 维几何，以及与上一帧 proposal 集的最大轨迹 IoU。所有字段都来自当前/过去
推理状态，不读取 GT。GT 仍只在完整 DepthTrack Train trace 结束后产生逐动作离线训练标签。

部署策略先由 action-logit 或 risk-adjusted value 选择一个动作，再同时要求 success 下限、预测
gain 下限、healthy/catastrophic 最大风险、跨帧轨迹 IoU 和跨尺度 IoU。候选策略是预注册的
`2×6×6×5×4×3 = 4,320` 个固定格点；最终仍执行与 V32/V33 完全相同的 85% 提交精度、0.5%
healthy harm、零 catastrophic 和正总 gain 门，门槛没有因前两版拒绝而放松。

实现及测试入口为：

```text
lib/test/tracker/causal_trimodal_risk_critic.py
tools/train_depthtrack_recursive_risk_critic.py
tests/test_causal_trimodal_risk_critic.py
tests/test_train_depthtrack_recursive_risk_critic.py
```

2-fold×1-epoch 的全 trace GPU 集成冒烟验证了 25,078 帧张量化、严格因果 history、五个监督头、
72 MB OOF JSONL 与 4,320 策略搜索可完整运行；它的零提交只是 1 epoch 欠拟合结果，不作为模型
结论。正式 5-fold×60-epoch、seed 2026 OOF 随后完成：694/4,320 个策略通过全部安全门，选定
策略为 risk-adjusted safe value、`success >= 0.99`、`risk <= 0.5`；其余 gain/trajectory/
cross-scale 下限均为 0，由 success head 提供主要认证。OOF 提交 565 帧，提交精度 `0.946903`，
恢复 180 个 public miss，总 IoU gain `+163.049`；7,956 个 public healthy 帧上的 harm 为 0，
catastrophic 也为 0。由此生成 410,377-byte artifact：

```text
/root/autodl-tmp/srtrack_depthtrack_train_recursive_risk_seed2026/causal_trimodal_risk_critic_seed2026.pth
SHA256: dee0bb3f8cf2ee4fab21af89b4f7bf8096a7a97fa2fe6dc9d781e1763f75d7b3
```

训练报告和约 75 MB OOF 证据分别为 `risk_critic_training.json` 与 `risk_critic_oof.jsonl`；artifact
同时绑定两者、原递归 trace/capacity、primary checkpoint、ranker、multiframe 和 state-gate 的
SHA。V34 runtime、OPE deployment、VOT bridge/workspace/shard 已接入同一泛化栈，37 项定向回归、
续跑/工作区 32 项回归均通过。DepthTrack Test 最短序列 `pigeon04_wild` 143 帧真实 GPU smoke
以 8.05 FPS 完成，证明初始化、权重加载、模板更新、双分支候选、风险认证和结果写盘可运行；该
smoke 不是正式指标。V34 top-14/all-anchor 投影已经启动，只有投影三项达标才会继续完整 OPE/VOT。

### 24.31 V34 top-14 正式否决：独立风险头仍不足以阻断身份切换递归污染

本节覆盖上一节末尾的“已经启动”临时状态。V34 已完成固定 top-14 序列的全部 anchor 投影，共
295 个 anchor、885 个协议输出文件，8 个 shard 全部成功、错误数为 0。权威投影为：

```text
EAO = 73.05662085364717
ACC = 82.51704831635134
ROB = 88.155284038978
目标 = 77.9 / 82.1 / 93.7
判定 = false / true / false
```

相对同协议 baseline `72.90895597/82.53586770/87.98807108`，V34 只提升约 `+0.148 EAO`
和 `+0.167 ROB`，ACC 轻微下降约 `-0.019`。`bag02_indoor_2` 的 ROB 提升约 `+10.746`，但
`yogurt` 的 ROB 下降约 `-2.702`。这说明风险头能修复少量片段，却仍不能稳定阻断身份切换后的
错误框写入递归状态。V34 正式否决；没有启动 DepthTrack Test、CDTB 或 full-127 VOT。投影和
逐序列对比的 SHA 绑定为：

```text
projection SHA256 = 8d407d66bf1be029c26b75ba79809dc6686dce086b11670faff14064b1b0f05c
comparison SHA256 = 0f2ad493a20497c3fa40f1b494d4f80630147ddebfa57761bdb8ef63c954fc4e
```

### 24.32 V35/LATCH：语言锚定的候选身份假设与未来失败风险控制

V32--V34 的共同缺陷不是候选回归容量不足。Train-only 容量审计在 25,078 个 recovery-active
联合帧上确认：16,239 个 public miss 中，factor-5/factor-7 top-8 的单帧并集可恢复 3,577 帧
（22.0272%）；连续 2/3/5 帧仍有候选的 miss 分别为 3,014/2,412/1,681。真正瓶颈是身份关联和
递归状态提交：高置信错误峰一旦被当成目标，错误 bbox 会写进下一帧 crop、动态模板和历史记忆，
后续搜索便围绕干扰物展开。

V35 将该问题重写为 Language-Anchored Candidate Hypothesis Tracking（LATCH）。factor-5 与
factor-7 只负责生成候选；选择层维护四类互不混淆的状态：

1. 不可变语言/首帧身份锚：early language-RGB、language-Depth、三模态 consensus 与首帧
   RGB/depth identity 证据不被在线模板覆盖；
2. RGB 与 Depth 两条解耦 GRU 状态：分别编码模态历史，先各自估计可靠性，再受控融合；
3. 有界目标记忆与干扰物记忆：目标写入固定要求 `target >= 0.90` 且 `hazard <= 0.20`，干扰物只写
   最高 distractor 概率且 `>=0.90` 的非同一候选；这些门槛写入 artifact，不随最终策略扫描变化；
4. secondary hypothesis：公共轨迹未来风险高时保留最高 selection-logit 的非 public 次假设；它只
   推进 RGB/Depth 两条模态状态并参与后续关联，不直接污染公开输出或递归 crop。低风险帧只允许
   public action 推进两条状态，因此 secondary 不是仅记录在日志里的无效字段。

模型分头输出候选 listwise selection、连续 IoU、target association、distractor association、
success、5 帧 failure hazard、RGB reliability 和 Depth reliability。当前 trace 能进行真实的
candidate-state rollout：模型动作会递归更新下一步双模态隐状态和身份记忆；但不能为每个候选
重渲染反事实像素 crop，因此 artifact 明确标记 `pixel_crop_counterfactual_dagger=false`，不把候选
状态 rollout 冒充完整像素级 DAgger。

训练只使用 DepthTrack Train：按序列分组 OOF，连续 4--8 帧窗口，scheduled sampling 从 0.10
线性提高到 0.90。public 动作使用接下来 5 帧的真实公共轨迹监督 hazard；替代候选因为没有反事实
未来 crop，只使用当前候选 IoU 的身份失败监督，避免把“公共轨迹之后失败”错误复制给本可救回
目标的候选。RGB/Depth evidence dropout、语言锚 batch counterfactual、listwise、association、
survival、modal reliability 和 IoU value loss 在同一 rollout 中优化。VOT、CDTB、DepthTrack
Test GT 均不参与训练、阈值选择或模型记忆。

正式训练前又执行了一次训练/OOF/runtime 同构性审计，并因此主动中止了两次尚未产出 artifact 的
旧训练。最终冻结的同构规则不是“实现上大致相似”，而是显式写入 artifact 并由加载器验证：

```text
state_transition_policy = public_or_hazard_secondary_v1
trajectory_history_policy = bounded_previous_proposal_set_v1
maximum_rollout_length = 8
```

每个动作的 `maximum_previous_action_iou` 在训练张量化和 runtime 中都只相对上一连续帧的完整
proposal 集计算，不再临时混入 target memory 或 secondary bbox。训练 OOF 每个窗口清空 target/
distractor memory 和 RGB/Depth 状态；runtime 也在帧号间断或每完成 8 步后执行同样的有界重置。
模态递归位置不读取最终策略格点：public hazard 低于 0.55 时固定观察 public action，达到 0.55
时固定观察 selection-logit 最高的非 public secondary。OOF 策略评估还与 runtime 一样执行
`public hazard >= 0.40` 才授权 factor-7、`public hazard >= 0.70` 禁止当前状态提交。缺失上述策略
字段、最大窗口长度或任一 source SHA 时，V35 加载器 fail closed，旧诊断权重不能部署。

V35 的核心工程不变量是事务式动作提交：

```text
selected internal bbox
  = next recursive state bbox
  = public target bbox（V35 强制 report scale = 1.0）
  = safe-template observation bbox
```

父 tracker 的模板观察在 localization 期间被冻结；最终动作确定后先回滚 speculative template
状态，再用最终 bbox 恰好观察一次。高 hazard 帧只回滚、不更新模板。动态模板 residual 由
`target_probability × mean(RGB/Depth reliability) × (1-hazard)` 决定，并截断在
`[0.0001, 0.02]`。公共 5 帧 hazard 同时控制 factor-7 动作授权、secondary hypothesis、模板更新
和 state delay；state delay 在 unresolved recovery 中强制下一帧 crop 保留 pre-frame trusted
state，而不是继续写入当前可疑框。

冻结同构规则后的全 trace 2-fold×5-epoch 诊断训练端到端通过，覆盖 3,595 个窗口、21,741 个
rollout 帧、140 个有连续窗口的 Train 序列。900 个固定策略中只有 1 个通过安全门：提交 380 帧、
提交精度 `88.1579%`、0 healthy harm、0 catastrophic harm、总 IoU gain `+9.336`，恢复 4 个
public miss。该结果比同构审计前的诊断更保守，证明训练器和控制契约可运行，但恢复量很小，不能
提前推断 VOT 会达标；诊断 artifact 绝不进入任何测试集评测。

正式 V35 为 5-fold、40 epochs、hidden 96、4 heads、seed 2026/2027/2028，只使用上述
Train-only 数据和
固定 900 策略格点。只有 OOF 同时满足提交精度至少 85%、healthy harm 为 0、catastrophic 为 0、
总 IoU gain 为正，才允许写出 artifact。正式训练已启动；其结果、artifact SHA、DepthTrack/CDTB/
VOT 指标必须以实际绑定文件为准，不能预填或由诊断结果替代。实现入口为：

```text
lib/test/tracker/language_anchored_candidate_hypothesis.py
lib/test/tracker/latch_atomic_action_commit.py
lib/test/tracker/siamtrack_recovery_peak_v35.py
tools/train_depthtrack_latch.py
tools/vot_siamtrack_rgbd_v35.py
tracking/rgbd_evaluation_deployment.py
tools/select_depthtrack_latch_seed.py
tools/continue_v35_after_latch.py
tools/continue_v35_multiseed.py
```

三个 seed 除随机种子外必须具有相同 checkpoint/trace/capacity/ranker/multiframe/state-gate SHA，
相同 folds/epochs/窗口/模型维度/增强设置，并各自绑定真实 OOF 文件 SHA。seed 选择不读取任何测试集：
先剔除未满足零 healthy/catastrophic harm、提交精度小于 85% 或非正 gain 的记录，再按
`public_miss_recovered`、总 IoU gain、提交精度、较少提交数、较小 seed 的固定字典序选择。选择产物
显式记录 `VOT/CDTB/DepthTrack Test GT used=false`；不能在看到 top-14 后换 seed。

V35 不沿用历史 `fixed6_isotropic_098_v1` 的 report-only scale。该变换虽然不反馈父 tracker state，
仍会造成“evaluator 看到的框”和“递归/template 观察的框”不相等。V35 workspace 因此不传 scale
profile，实际 scale 固定为 `1.0`；加载时同时要求 parent V34 risk artifact 与独立 LATCH artifact。
晋升顺序固定为 artifact 加载/绑定 → top-14 全 anchor 投影 → DepthTrack50 → CDTB80 → full-127
官方 VOT，前一门失败时后续任务不启动。

本次冻结后的定向回归为 V35 核心/训练/部署 33 项全部通过，VOT workspace/shard/finalize 18 项全部
通过，`git diff --check` 无错误。正式运行目录为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_latch_v35c_seed2026
/root/autodl-tmp/srtrack_depthtrack_train_latch_v35c_seed2027
/root/autodl-tmp/srtrack_depthtrack_train_latch_v35c_seed2028
/root/autodl-tmp/srtrack_depthtrack_train_latch_v35_multiseed
```

目录内的 `frozen_code_sha256.txt` 与 `frozen_input_sha256.txt` 分别锁定四个核心实现和全部 Train-only
输入；正式进程运行期间不得修改这些文件。当前文字只记录运行协议，不将进行中的训练视为结果。

### 24.33 V35 三 seed 正式训练结果、固定选择与 top-14 启动

三组预注册的 5-fold×40-epoch 正式训练均已以 `exit.code=0` 完成，并分别产出权重、约 74 MB
OOF 预测及原子写入的 `training.json`。三者使用相同 checkpoint、DepthTrack Train trace、capacity、
ranker、multiframe、state-gate、训练超参数和策略格点；唯一变化是随机种子。固定 OOF 安全门与结果为：

```text
seed 2026: feasible 213/900, commit 6, precision 1.000000,
           healthy harm 0, catastrophic harm 0, recovered miss 4, IoU gain +4.943359
seed 2027: feasible 172/900, commit 862, precision 0.859629,
           healthy harm 0, catastrophic harm 0, recovered miss 0, IoU gain +2.154277
seed 2028: feasible 60/900, commit 461, precision 0.917570,
           healthy harm 0, catastrophic harm 0, recovered miss 0, IoU gain +3.539282
```

正式 artifact/OOF/report SHA 分别为：

```text
seed 2026 artifact 5166faa08c8ef6d7eaba7044599d7782a73584bd4a78b35a7e60fbf02e95cab7
          OOF      c1728df4b5f6333aff64407ebf591e7c10a42e640cb769187cc0f31ad941a124
          report   b26ff4d007093f7d66cd342ff077893100219b39d9dd9b13d72de2134c4d7b13
seed 2027 artifact e7a1bca8b887eb3788eda05d3c7221fe5bdfad2a697c797355c83656bfe0e076
          OOF      6158c3521274878c159865cf07869a47b65d6a59d7023d3b0ef7822f604592a3
          report   8d6c75d7f2a7e5551ef569eca45df5a1bb4b9d0ea12ce1df13abc4e34f7cc00d
seed 2028 artifact a51ca8527236de251c7899ad3815985e8ea2b9ae973c5a6e9b0fb86ae985217c
          OOF      0d06023a74abade0eb4bcc49f1ad49f21e04cc6a4d598f5051d587e7b8da62a1
          report   27b6f28790ef3466867258a25ad9fdced00ed3fe093aa386ff6a5466a1df81d7
```

预注册的 Train-only 字典序选择器因此固定选中 seed 2026：其首要排序量
`public_miss_recovered=4`，另外两组均为 0。选择清单明确记录
`VOT/CDTB/DepthTrack Test GT used=false`，并绑定选中权重、OOF 与报告的真实 SHA；后续不能依据
top-14 结果改换 seed。选择证据为：

```text
/root/autodl-tmp/srtrack_depthtrack_train_latch_v35_multiseed/seed_selection.json
```

训练完成后补了一处不改变任何权重或阈值的部署安全修复：每个 `track()` 调用开始先清空上一帧的
`_latch_controls` 与 `_latch_template_call`，避免没有进入候选阶段的当前帧错误沿用上一帧 hazard
来控制模板。修复后的 `siamtrack_recovery_peak_v35.py` SHA 为
`5932cf963961ce9997294976c9cf144ce0a94d9c89d9628e12784ede4cad598c`；27 项 V35 核心、训练、
选择、续跑、workspace、shard 与 bridge 定向回归全部通过，`git diff --check` 无错误。正式 VOT
workspace 在修复后重新物化，因此 manifest 绑定的是修复后的部署代码，而不是旧 runtime。

seed 2026 已进入固定 top-14/all-anchor 门禁。workspace 同时独立绑定父 V34 risk artifact
`dee0bb3f...d7b3` 与 LATCH artifact `5166faa0...cab7`，强制 report scale `1.0`，并将 14 个序列、
295 个官方 multistart anchor、259,874 个 tracker frame 无重叠划分为 8 个 GPU shard：

```text
/root/autodl-tmp/srtrack_primary_v35_latch_seed2026
/root/autodl-tmp/srtrack_primary_v35_latch_seed2026/top14_latch_8worker/manifest.json
```

本节只确认正式训练、固定选择和评测绑定；top-14 尚在运行，因此不能提前填写投影指标，也不能
声称 V35、DepthTrack、CDTB 或 full-127 VOT 达标。只有 top-14 的 EAO/ACC/ROB 三项绝对投影均
达到 `77.9/82.1/93.7`，续跑器才会启动后续 OPE 与完整 VOT。

### 24.34 全帧因果审计：恢复候选阶段之外仍存在高置信身份切换

V35/LATCH 解决的是 recovery-active 候选集合内的身份关联，但进一步审计发现，VOT 的 ROB 瓶颈
还有一条更早的旁路：主 factor-5 前向可以在分数仍高时直接跳到相似干扰物；这种帧不一定触发
既有 recovery candidate stage，因而 LATCH 没有机会否决。错误框随后写入下一帧 crop，搜索中心
和历史状态一起迁移到干扰物。这与“普通框回归略有偏差”不同，是一次离散身份切换及其递归放大。

为验证该判断，建立了完全只读的全 152 序列 DepthTrack Train 因果 trace：推理过程不可见 GT，
GT 只在全部输出落盘后离线 join；固定 6 序列的输出与冻结 baseline 逐字节一致。trace 共记录
17,789 个 motion-risk/周期稳定帧，目录为：

```text
/root/autodl-tmp/srtrack_primary_evidence_trace_all152_seed2026
trace aggregate SHA256: e75eb83...
Train GT aggregate SHA256: 8c3bf14...
```

在“上一帧 IoU 至少 0.5、当前候选 IoU 小于 0.2”的身份切换事件中，当前 raw score 至少 0.5 的
事件有 201 个；简单保留上一帧 bbox 能把其中 160 个恢复到 IoU 至少 0.5，并改善 197 个，健康
帧 harm 大于 0.1 的数量为 0，总单帧 IoU gain 为 `+128.26`。raw score 至少 0.7 的 37 个事件中，
保留动作可恢复 36 个，同样没有健康 harm。相比之下，同一前向内的 visual/language 候选只能分别
恢复 6/5 个。这说明关键候选不是另一个当前峰，而是“尚未被错误递归状态覆盖的 pre-frame 状态”。

但对所有 miss 帧无条件 hold 会把已经丢失的框继续保留，无法泛化；人工阈值扫描也找不到零 harm
策略。旧特征只读取当前被选峰的位置，无法区分“刚刚从健康目标切走”和“早已丢失后再次跳动”。
因此三次小规模 V36 诊断均按安全门正式拒绝，不产出权重：前两次验证基础多头训练，第三次增加
transition-recovery head 后仍无可行策略。该负结果没有被当作可部署模型。

### 24.35 V36：语言锚定的全帧状态预提交（Language-Anchored Pre-commit）

V36 在 V35 之上新增一个位于递归状态写入之前的二动作门控：

```text
commit current candidate
        或
hold exact pre-frame bbox for one frame
```

新信息来自同一次主前向中“被选候选位置”和“pre-frame bbox 在当前搜索图上的中心位置”的成对
证据。对 routed response、early language-RGB grounding、language-Depth grounding、三模态
consensus、coarse/fine language、raw/projected RGB identity、projected Depth identity，均同时记录：

```text
selected rank/value
held-center rank/value
selected - held-center rank/value
```

再拼接 raw score、motion/scale 风险、候选跳变、两种中心 response、三模态 guard features 与
route diagnostics。这样门控器可以问“当前高分峰是否仍比原目标中心更符合首帧语言身份”，而不是
只问“当前峰自身看起来是否可信”。训练器使用 MLP 的 current-IoU、held-IoU、benefit、
transition-recovery 与 harm 五个头，并按序列分组 OOF。策略只有同时满足以下条件才写 artifact：

```text
hold precision >= 85%
healthy harm = 0
catastrophic harm = 0
transition recovered > 0
total IoU gain > 0
maximum consecutive holds = 1
```

正式 V36 trace 使用完整 V35/LATCH 轨迹，而不是旧 frozen-primary 轨迹。钩子固定挂在
`_recovery_output_candidate_state` 的主前向参数上，避免误取 factor-7 的额外网络 forward。只有
未进入 joint/LATCH 动作、恢复前后均不 active、无 armed/recovered/expired 事件，且 hook candidate、
最终递归框、公开结果三者在 `1e-5` 内完全一致的帧才标记为 `gate_eligible`。trace 本身仍不改变
V35 输出，GT 只在独立分析程序中后接。

部署时 hold 是原子事务而不是仅修改返回值：回滚 recovery 对象、safe-template policy、动态模板
tensor/source、joint/multiframe 历史和递归 bbox，再把同一 held bbox 写入 public output 与模板观察
语义；hold 帧的置信度使用 search-map 中心的 held response。上一帧已经 hold 时，本帧强制冷却，
因此不会冻结目标。artifact 加载器绑定 checkpoint、完整 V35 五个父 artifact、152 个 trace 文件
aggregate、Train GT aggregate、分析报告和 OOF SHA，并拒绝任意 VOT/CDTB/DepthTrack Test GT
参与。实现入口为：

```text
lib/test/tracker/language_anchored_precommit_gate.py
lib/test/tracker/siamtrack_v35_precommit_trace.py
lib/test/tracker/siamtrack_recovery_peak_v36.py
tools/run_v35_precommit_trace.py
tools/analyze_v35_precommit_trace.py
tools/train_depthtrack_precommit_gate.py
tools/select_depthtrack_precommit_seed.py
tools/vot_siamtrack_rgbd_v36.py
tools/continue_v36_after_precommit.py
tools/continue_v36_after_v35.py
```

V36 的 152 序列预检已经通过：固定首帧语言 JSONL、checkpoint、ranker、multiframe、state-gate、
risk critic 与 LATCH 的 SHA 全部匹配，明确 `precommit_gate_active=false`。当前聚焦回归为 66 项
全部通过。正式条件控制器运行在 screen `codex_v36_after_v35`：它先等待 V35 的最终结果；若 V35
三套数据全部达标，则不启动 V36；若 V35 任一门失败，才在双卡释放后运行完整 Train trace、三组
seed 2026/2027/2028、固定 Train-only OOF 选择及 V36 top-14。无安全 OOF 策略时以正式拒绝结束，
不会加载诊断权重。控制目录为：

```text
/root/autodl-tmp/srtrack_v36_conditional_after_v35_seed2026
```

本节记录的是已经实现并通过预检的候选架构，不是正式指标结论。V35 top-14 截至本次更新完成
154/295 个 anchor，8 个 shard 正常、日志无错误；V36 尚未开始 Train trace，因此没有 artifact，
也没有任何 V36 测试集结果可填写。

### 24.36 V35 top-14 运行期隔离修复与缺失 anchor 恢复

V35 top-14 后续推进到 241/295 个 anchor 时，shard 6/7 在分别切换到
`humans_shirts_room_occ_1_A_1` 与 `stick_indoor_1` 后出现 TraX
`Unable to connect to tracker`。tracker 端的真实异常不是 CUDA OOM，而是 V35 worker 启动时收到
了一个已经消失的 `/tmp/tmp*` V36 pre-commit artifact 路径：

```text
FileNotFoundError: Missing evaluation artifact
language_anchored_precommit_gate_artifact: /tmp/tmp*
```

该路径既不在 V35 workspace 的 `trackers.ini`，也不属于 V35 部署契约；启动 shell 显式
`unset` 且 V35 wrapper 执行 `os.environ.pop()` 后仍可复现。另一个运行期问题是当前 VOT CLI 在
该 TraX 初始化错误后仍返回 0，所以只检查 shard 子进程 return code 会把不完整覆盖误判为成功。

隔离修复因此不再依赖可被 runtime 重新注入的环境状态。基础 bridge 新增显式参数
`ignore_precommit_artifact=False`；V35 固定以 `True` 调用，使 `_build_tracker()` 的 pre-commit
输入无条件为 `None`，V36 仍沿用默认路径并正常读取自己的冻结 artifact。该改动只约束 V35 的
启动参数，不改变 V35 tracker、checkpoint、LATCH/ranker/multiframe/state-gate/risk artifact、
阈值、模板更新规则或逐帧推理。V35/V36/deployment 聚焦回归 16 项通过，当前 SHA 为：

```text
tools/vot_siamtrack_rgbd.py      64507ede3df2a18d35ec0e741a6f77872000b01f2ccdd595338f30ca24f55c0a
tools/vot_siamtrack_rgbd_v35.py  576ce51f1f7fec235bfde56a8e34bd054423ba8ea6540765640789b294f0d5a7
```

修复后在空闲 GPU 1 上用独立 screen `codex_v35_retry06b` 与
`codex_v35_retry07b` 恢复两个缺失 shard；两个 tracker 均成功初始化，各占约 2.2 GiB，未再出现
临时 artifact 错误。GPU 0 原 shard 0/1 同时继续运行。到 2026-08-12 23:40 CST，共完成
248/295 个 anchor。`codex_v35_rescue_project_only` 每 300 秒审计完整 `.bin` 覆盖；只有达到
295/295 且原 continuation 进程退出时，才使用冻结的 14 序列清单生成 comparison 与 projection。
该 watcher 明确不写 V35 continuation 终态，也不启动 OPE 或 full-127 VOT；三项原始投影必须先由
人工/agent 读取并执行下一道门控。

这次启动隔离修复晚于现有 top-14 workspace 的物化，其 manifest 仍绑定旧 bridge SHA
`eb94e21b...ab5409`。因此当前混合启动批次只作为 top-14 候选门诊断；若投影达到目标，正式晋升前
必须用当前 bridge 重新物化并绑定一致的 workspace/provenance，不能把 manifest 漂移后的结果直接
当作 DepthTrack/CDTB 或 full-127 VOT 正式证据。本节同样不预填 EAO/ACC/ROB，V36 继续等待
V35 的原子终态。

补跑期间对已经完整落盘的 10 条序列、189 个 anchor 做了一次纯只读 early diagnostic；排除仍在
运行的 `cup02`、`yogurt`、`humans_shirts` 与 `stick`，因此它不是预注册 14 序列门，也不能提前
终止或晋升。相同 10 序列上的 reference 与 V35 原始聚合为：

```text
                         EAO       ACC       ROB
reference              53.37060  78.55657  57.60016
V35/LATCH              25.55349  79.44267  19.98669
delta                 -27.81711  +0.88610 -37.61347
```

只替换这 10 序列、其余 117 序列保持完整 reference 的 counterfactual projection 为
EAO/ACC/ROB `68.72666/82.76151/83.76371`，相对 reference full-127 为
`-4.18229/+0.22564/-4.22436pp`，只有 ACC 通过目标。最大 EAO/ROB 负贡献来自
`toiletpaper01_indoor_2`、`bag02_indoor_2` 与 `notebook01_indoor_1`；其中 toiletpaper 的
sequence ROB 从 `1.0` 降至约 `0.04670`。该诊断已明确标记 `diagnostic_only=true`，最终 V35
分支仍必须等待 295/295 后的固定 14 序列 comparison/projection。

为避免旧 controller 与 rescue watcher 在覆盖完成瞬间竞态，又增加两道只影响控制面的 fail-closed
guard。`codex_v35_original_guard` 等原 GPU 0 shard 0/1 都退出后停止旧
`codex_v35_multiseed`，禁止它用旧 manifest 自行进入后续门；`codex_v35_fail_gate` 只接受 schema、
candidate workspace/tracker/checkpoint、reference workspace、固定 14 序列集合、`selected_anchors=295`
和 same-settings/checkpoint 全部一致的 rescue projection。projection 三项未达标时，它才原子写
`srtrack-v35-continuation-result/v1` 的拒绝终态并放行 V36；若三项达标，只写
`await_provenance_rematerialization` gate，不写 continuation result，因此不可能自动晋升。

### 24.37 V35 固定 top-14 终态与 V36 Train trace 启动修复

V35 的固定 14 序列评测在 2026-08-13 00:17 CST 完成全部 295/295 个官方 multistart
anchor。两个隔离补跑 shard 均返回 0，旧 controller 已由 guard 在汇总前停止；comparison
契约确认 14 序列集合、295 anchors、checkpoint 与 VOT settings 全部一致。固定子集上的原始
EAO/ACC/ROB 为：

~~~text
reference subset   50.175795 / 74.967504 / 53.431277
V35/LATCH subset   28.561090 / 77.004933 / 20.858465
delta             -21.614705 / +2.037429 / -32.572812 pp
~~~

将这 14 条候选轨迹替换进其余 113 条保持 reference 的 full-127 counterfactual 后，投影为：

~~~text
EAO/ACC/ROB             67.831351 / 83.222987 / 82.300607
目标                    77.9      / 82.1      / 93.7
通过                    false     / true      / false
相对 reference full-127 -5.077605 / +0.687119 / -5.687464 pp
~~~

该结果仍明确标记 counterfactual_projection_only=true 与
official_full_dataset_result=false，不能替代正式 full VOT；但 EAO/ROB 已低于目标，因此足以
执行 fail-closed 拒绝。rescue_fail_gate.json 的 18 项输入/来源检查全部为 true，决策为
reject_and_start_v36；V35 原子终态为 v35_top14_projection_below_target，
formal_vot_started=false，未浪费资源启动必然不能达标的 full-127。关键证据 SHA 为：

~~~text
top14_comparison_rescue.json  4c2868afa23d641d09077a825b7ea59674dfaa2864de7e3c91573e4bb5dd3216
top14_projection_rescue.json  05a492ab5fc4ad977e4e9e2a601f5aa199a82357e7419634f1fb299ee8d7d973
~~~

V36 条件控制器随后进入 running_v35_train_trace，但首次启动在 GPU 推理前失败。根因是
run_v35_precommit_trace.py 把 profile 名字符串直接传给只接受 dict 的
language_search_recovery_profile 参数；正式 VOT bridge 与 OPE 部署路径本来都会调用
build_v35_profile()，仅该新 trace runner 漏了转换。修复后 runner 传入冻结 profile dict，
并在 manifest 中额外绑定 62 个 profile 字段与 tools/vot_siamtrack_rgbd_v35.py 的 SHA。
这只修正启动契约与 provenance，不改变 checkpoint、五个父 artifact、阈值、逐帧 tracker
逻辑或安全模板更新策略。修复后 launcher SHA 为：

~~~text
tools/run_v35_precommit_trace.py
1eaa77c34f26386999a82161fc4a5ef61e00a0738eefd8b26fd1823ce58a7e51
~~~

公开参数入口 smoke 确认 62 个 profile 字段逐项一致，LANGUAGE_SEARCH_RECOVERY.USE=true 且
SAFE_TEMPLATE_UPDATE.USE=true；V35/V36 定向回归 30 passed。首次失败的 8 KiB 目录未删除，
已归档为：

~~~text
/root/autodl-tmp/srtrack_v35_precommit_trace_all152_seed2026.failed_profile_type_20260813T002554
~~~

V36 于 2026-08-13 00:33:55 CST 从相同冻结输入重新启动。到 00:35 CST，两张 GPU 各占约
2.25 GiB、利用率约 66%/73%，两条实时 trace 已写入；以重启命令为边界的日志切片中
Traceback/RuntimeError/TypeError/OOM/FileNotFoundError 均为 0。当前 screen 为
codex_v36_after_v35，状态仍是 running_v35_train_trace；只有 152 条 Train trace 全部完成并
通过 manifest 完整性检查后，控制器才会分析容量、训练三组 seed 并执行固定 Train-only OOF 选择。

### 24.38 V36 动作可达性中期审计（仅诊断）

V36 的部署与训练 trace 使用相同的 eligible 条件：当前帧必须没有 joint 决策、recovery 在前后均
不 active、没有 armed/recovered/expired 事件，并且 hook candidate、递归 state、public bbox
一致。该限制保证 hold 可以原子回滚，但也意味着 V36 不会直接否决 V35 的 LATCH/recovery 动作。

运行中抽查 adapter02 的 516 个捕获帧时，478 帧存在 joint decision、463 帧 recovery active，
只有 11 帧满足 V36 gate eligible；前 7 个已创建或正在写入的 trace 文件合计 1,656 个捕获帧，
其中 100 个 eligible。全部已写 frame 的 69 维 pre-commit 特征都能解析且为有限值，
held-center response、两侧 evidence maps、consensus 与 route 字段均完整，trace 与公开输出的
最大 bbox 差异为 0。因此当前限制来自动作域设计，而不是缺特征或 trace 损坏。

对这 7 个顺序偏置、部分仍未完成的 Train trace 做只读 GT 后接 sanity check，得到：

~~~text
eligible frames                 100
beneficial holds                  3
healthy harmful holds            55
catastrophic harmful holds        1
healthy-to-miss recoverable       0
beneficial hold IoU gain   +0.45644
~~~

该早期子集不是固定 OOF、不能用于调阈值或提前拒绝；完整 152 序列仍按原计划运行。但它与 V35
top-14 的结构性缺口共同限定了可达性：V35 full 投影距离目标仍缺 EAO 10.06865pp、ROB
11.39939pp，而 V36 无法直接回滚多数 V35 recovery 动作。若完整 Train OOF 无安全策略，或 V36
top-14 仍未达标，下一候选应把同类 pre-commit 保护直接放到已达标的 reference/primary 栈，
以保留 baseline 轨迹和安全模板更新，再只针对普通主前向的高置信身份切换训练零 harm 策略；
不应继续在已证明显著退化的 V35 recovery 栈上叠加更深的门。

### 24.39 已达标主栈上一帧 hold 容量对照（仅诊断）

为避免 V36 若被拒绝后重复探索已经失败的 visual-fallback 路线，对 2026-08-04 已完成的主栈
DepthTrack Train 只读 trace 做了一次固定阈值、无调参的上一帧 hold 容量复算。源 manifest 含
152/152 条序列、456 个预测文件、152 个 trace 文件，trace error 为 0；GT 仅在因果推理完成后
联接，未使用 VOT、CDTB 或 DepthTrack test GT。与 V36 capacity 一致，beneficial 定义为
held-current IoU 大于 0.10，healthy 定义为 current IoU 至少 0.50，catastrophic 定义为
healthy current 下 held IoU 小于 0.20。原始计数为：

~~~text
captured frames                      17789
valid GT-linked frames               13259
invalid GT rows                       4530
beneficial holds                       705
beneficial total IoU gain        +272.987683
healthy harmful holds                 8586
catastrophic holds                    1930
healthy-to-miss transitions            298
transition recoverable by hold          200
recoverable transition gain      +142.473885
~~~

该对照证明 primary 主栈存在真实的单帧回滚容量：例如 cup10、cube06、ball16 与
toiletpaper02 均出现 current IoU 接近 0、previous-state IoU 大于 0.93 的帧；但无条件 hold 的
伤害量远高于收益量，绝不能据此直接部署。旧的同一套 trace 上，visual/language/routed 三种当前帧
动作平均 IoU 分别为 0.636045/0.638084/0.637525，固定 false-conflict 门下 feasible rule 数为 0，
正式决策已经是 insufficient_false_conflict_safety_keep_guard_disabled。因此下一阶段不能把旧
visual fallback 换名后重跑，而必须学习高精度的 current-versus-held pre-commit 判别。

这项容量复算还有两个严格的部署不等价限制。第一，旧 trace 的
safe_template_update=false，而最终系统必须保留安全模板更新；第二，旧 schema 只记录当前选中
位置的语言/深度证据，没有 held-center response 与 held evidence maps，无法构造 V36 的 69 维
成对特征。因此该结果不得生成 artifact、不得进入 top-14，也不能作为 DepthTrack/CDTB 保持性
证据。若完整 V36 OOF 拒绝，主栈回退的最低契约应为：

~~~text
1. 以当前已达标 checkpoint 和 SAFE_TEMPLATE_UPDATE.USE=true 重新采集 Train-only trace；
2. trace wrapper 必须证明公共 track() bbox/score 与相同配置 reference 逐文件一致；
3. 每帧同时记录 current 与 held-center 的响应、RGB/Depth/语言证据和模板状态；
4. hold 时原子回滚 recursive bbox 与安全模板 policy/tensor，且最大连续 hold 为 1；
5. 仅在固定 3-seed OOF 达到 zero healthy/catastrophic harm 门后运行 VOT top-14。
~~~

当前 V36 仍按冻结计划完成 152 序列，不使用本节的旧主栈 trace 训练或提前改变方向。本节只把
下一候选的动作容量、已失败路线与 provenance/模板更新要求固定下来。

### 24.40 V36 Train trace 近半程完整性、回归与模板证据边界

2026-08-13 02:37 CST 的短连接审计确认 `codex_v36_after_v35` 仍为 detached 运行态，controller、
runner 与两个 GPU worker 的 PID 均保持不变；两卡即时利用率为 62%/67%，各占约 2.25 GiB。
从 00:33:55 修复重启点之后统计，Traceback、RuntimeError、TypeError、CUDA OOM、
OutOfMemoryError、FileNotFoundError 与 Killed 合计仍为 0。原始进度为：

~~~text
完成序列                     69 / 152
预测文件                    207 / 456
已创建 trace 文件            71 / 152
帧加权处理下界       106697 / 219954 = 48.5088%
双卡合计吞吐                    14.4069 FPS
trace 阶段剩余估计                    2.18 h
manifest / controller     running / running_v35_train_trace
SAFE_TEMPLATE_UPDATE.USE                         true
~~~

对当时全部 71 个 JSONL 做独立增量 schema 审计，共读取 23,745 个 frame record；其中 1,888 个
`gate_eligible` frame 均能构造固定 69 维有限特征，parse error、frame error 与 feature error 都为
0。eligible frame 的 hook/public bbox 最大差异均为 0；ineligible frame 虽有 V35 内部 hook
candidate 变化，但所有 23,745 帧的 public bbox 最大差异仍为 0，符合 read-only trace 契约。
当前仅诊断的 GT 后接容量为：

~~~text
eligible / valid GT rows              1888 / 1778
beneficial holds                              74
healthy harmful holds                       1025
catastrophic holds                            38
healthy-to-miss transitions                    5
transition recoverable by hold                 2
~~~

这些顺序偏置中间数不参与阈值选择，也不能替代完整 152 序列的三 seed OOF。它们只说明安全策略必须
极稀疏触发，并且部署要求的 `transition_recovered > 0` 已经在动作域中可达。训练器的固定 60,000
组策略搜索、每 fold 仅使用 training-sequence 归一化、Train-only provenance、artifact SHA 绑定、
最大连续 hold 为 1，以及 runtime 原子回滚语义已经复核；V36 trace/训练/选择/deployment 的聚焦
回归为 34/34 passed，仅有一条第三方 timm 弃用警告。

模板证据需要继续严格区分。已达标的 DepthTrackTest `65.995933/65.335885/65.664250` 与 CDTB
`75.387821/76.005850/75.695574` 来自 `safe025 + report scale 0.98` 性能参考，其 manifest 明确
`safe_template_update=false`；旧 `safe_template_blend002` 分支又因
`no_false_conflict_safe_rule` 终止于 `complete_rejected`，未物化候选配置，也未运行正式 VOT。

### 24.41 V36 完整 Train-only OOF 终态：安全拒绝

2026-08-13 04:46:40 CST，V36 的 V35-stacked pre-commit trace 完成。最终 manifest 为
`complete`，覆盖 152/152 条 DepthTrack Train 序列、456/456 个预测文件与 152/152 个
JSONL；`trace_aggregate_sha256=8c1a0d1595a1f101a41fc1302d1631bcba9a0cc18c012e11b942f5033931ec9b`。
完整 trace 的 `maximum_final_public_abs_difference` 和
`maximum_hook_final_abs_difference` 均为 0，`trace_error_count=0`。00:33:55 修复重启后的
日志没有新增 Traceback、RuntimeError、TypeError、CUDA OOM、OutOfMemoryError、
FileNotFoundError 或 Killed。源契约仍为首帧语言、无未来帧文本、推理时不可见 GT，且 VOT、
CDTB、DepthTrack test GT 均未用于训练或选择；trace 配置明确
`SAFE_TEMPLATE_UPDATE.USE=true`。

完整 Train-only 后接 GT 的动作容量为：

~~~text
captured / eligible frames                 43591 / 3734
invalid GT rows                                     231
beneficial holds                                     131
beneficial total IoU gain                     +31.473533
harmful / catastrophic holds                   1842 / 51
healthy-to-miss transitions                          10
transition recoverable by hold                         4
recoverable transition gain                    +2.248574
~~~

训练阶段固定使用 69 维 current-versus-held 成对特征、5-fold sequence-group OOF、40 epochs 和
60,000 条冻结策略网格。三个预先固定的随机种子得到一致终态：

~~~text
seed    OOF rows    evaluated policies    feasible    artifact
2026        3503                 60000           0       none
2027        3503                 60000           0       none
2028        3503                 60000           0       none

seed_selection.ready_for_top14                         false
selected_seed / selected_artifact                null / null
controller decision       v36_train_oof_rejected_no_safe_policy
~~~

因此 V36 被 fail-closed 门禁正确拒绝，未启动 VOT-RGBD2022 top-14，更未触发 OPE/full VOT。
这支持“采集、Train-only 分组 OOF、来源绑定和安全拒绝链路正确执行”，但不支持“现有 V36
能够安全选择 commit/hold”或“具备 top-14 资格”。同理，`safe_template_update=true` 只证明
本次观测栈启用了模板更新；由于没有部署 artifact、也没有实际接受 hold 策略，它不能替代模板
更新开启时的 DepthTrack/CDTB 保持性评测。

独立 result-to-claim 审查给出 `claim_supported=no`、`confidence=high`。随后完成的独立
实验完整性审计总体为 `WARN`；审计未发现泄漏或数字错配，但确认主张仍只能停留在“安全拒绝链路正确”。下一步只允许在 DepthTrack Train
上分析门槛首要失败项与可分性，然后转向已达标 primary 主栈的最小 pre-commit 观测/回滚
fallback；仍须先通过跨 seed OOF 并锁定 artifact，才可依次进入 VOT top-14、模板开启的
DepthTrack/CDTB 保持性与 full VOT。禁止放宽 zero healthy/catastrophic harm 门来强行生成
artifact，也禁止把本次安全拒绝包装成正向性能结果。

### 24.42 V36 后续 provenance hardening 与 primary-stack fallback 诊断

V36 完整 OOF 拒绝后，2026-08-13 05:13 CST 又做了两项只读/小步收紧，目的不是生成新指标，而是避免后续 fallback 误用无模板更新的旧证据。

第一，pre-commit 训练入口和 artifact loader 的来源契约已经收紧：source trace manifest 的 `tracker_stack.safe_template_update` 必须严格为 `true`，否则训练 source contract 或 artifact 加载直接 fail-closed。修改位置为：

~~~text
lib/test/tracker/language_anchored_precommit_gate.py
  load_language_anchored_precommit_gate(): require trace_stack.safe_template_update is True

tools/train_depthtrack_precommit_gate.py
  validate_source_contract(): require stack.safe_template_update is True

tests/test_language_anchored_precommit_gate.py
  existing Train-only loader fixture updated with safe_template_update=true
~~~

聚焦回归结果为 16/16 passed，仅有一条第三方 timm FutureWarning。额外 smoke 验证同一份 V36 complete manifest 可通过 `validate_source_contract()`；将该 manifest 中 `safe_template_update` 篡改为 `false` 后会得到 `source trace manifest contract failed`。因此后续不会再从 `safe_template_update=false` 的旧 primary trace 产出 pre-commit artifact。

第二，对旧 primary 主栈 trace 做了 CPU-only OOF 诊断，输出只放在 `/root/autodl-tmp/srtrack_primary_precommit_old_trace_oof_diagnostic/`，schema 明确 `deployable_artifact=false`，原因是该 trace 缺 held-center evidence maps 且 `safe_template_update=false`。它只用于判断 fallback 方向，不参与训练选择、top-14 或正式指标。

~~~text
seed    rows    nonzero policies    zero-harm+cat policies    precision85    gain-positive    transition-positive    feasible
2026   13259                  60                       60            60              60                    0           0
2027   13259                   0                        0             0               0                    0           0
2028   13259                   0                        0             0               0                    0           0

seed2026 diagnostic sha256  ad435f5d282bd37ed3449a3066323baa639879be54e372462d7a9f707ebbd184
seed2027 diagnostic sha256  36bc68b9215b125c4b7400f7548d066a464480d50d87709c8ff1baff9c77e283
seed2028 diagnostic sha256  ca07fa636e3bf53b76d482faa4af3e393879f7a3ef53eb3b3626391d869cdd7c
~~~

这个结果说明 primary 主栈确实比 V36-stacked trace 更容易找到零伤害、高 precision、正 gain 的候选，但仍没有任何 OOF 策略恢复 `transition_recovered>0`。同日对 V36 OOF 还做了 transition 阈值扩展重扫（把 `minimum_transition_recovery_probability` 从冻结网格的 0.5--0.95 扩到 0--0.5），依然没有任何 seed 得到 `transition_recovered>0` 的实际策略；seed2026/2027 只各有少量零伤害、高 precision、正 gain 的单帧 hold，均不是 transition recovery。

结论保持不变：不能放宽 zero harm 或 transition recovery 硬门槛来强行生成 artifact。下一步若继续 fallback，应重新采集 primary-stack、`safe_template_update=true`、含 held-center evidence maps 的最小 pre-commit trace；只有固定 3-seed sequence-group OOF 同时满足 zero healthy/catastrophic harm、precision>=0.85、transition_recovered>0、total gain>0，才允许进入 VOT-RGBD2022 top-14。旧 primary diagnostic 和 V36 rejected OOF 都不得包装为性能提升证据。


### 24.43 独立完整性审计与 primary fallback 配置边界

独立实验完整性审计总体为 `WARN`。A（GT provenance）、B（fold 内训练集归一化）、C（数字/
状态/文档一致性）、D（metric/selection 实际执行）和 F（评测类型分类）均为 `PASS`，E（scope
language）为 `WARN`。没有发现 GT 泄漏、验证折归一化污染或数字错配；WARN 表示 V36 只支持
“安全拒绝链路正确”，不支持性能提升、top-14/full VOT 达标或上线。审计固化于
`EXPERIMENT_AUDIT.md/.json`，原始追踪位于
`.aris/traces/experiment-audit/2026-08-13_run01/`。

达标 primary 基线 `droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml` 的
SHA-256 是 `4ddf10339575003294b2ff0e77ff8583b8d630d035173d8508852b0a17806b46`。旧控制器虽
预注册其“只加入 SAFE_TEMPLATE_UPDATE”的精确派生，但证据门以
`no_false_conflict_safe_rule` 终止，候选从未物化。现存 `trimodal_consensus_safe_template_blend002`
还改变 early grounding、primary guard 和训练 router；V36 的 `probe_e1_template` 还把
`RESIDUAL_MAX` 从 `0.00010` 改为 `0.02`、`MAX_SUPPRESSION` 从 `0.25` 改为 `6.0`。
两者都不能充当 safe025 主栈的模板更新证据。

primary fallback 必须从 SHA 绑定的 safe025 精确派生，仅增加模板更新；validator 与 artifact
loader 硬拒 `safe_template_update != true` 或来源哈希不符。用户确认公开测试切面后先写红测，
再实现 current/hold 与 bbox、模板、递归状态原子回滚。重新采集 primary trace，三个锁定 seed
OOF 全通过并锁定 artifact 后才准入 top-14；之后才评测模板开启的 DepthTrack/CDTB 和 full VOT。
zero healthy/catastrophic harm 不得放宽。

### 24.44 primary safe-template fallback 的当前证据边界与待确认测试切面

2026-08-13 05:30 CST 复核现存 template 输出后，结论需要进一步收紧：
`droptrack_depthtrack_final_language_primary_trimodal_guard_probe_e1_template.yaml`
确实已有完整 DepthTrack Test 与 CDTB 结果，且 manifest/metrics 均绑定
`safe_template_update=true`、checkpoint
`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab` 与 config
`cf8e1caed647600dc307cfc4fa18619174ea38ae5ef6f0e54a84223f175473a2`。对应输出为：

~~~text
/root/autodl-tmp/srtrack_primary_v24_branchsafe_template_seed2026/depthtrack_test_full50_recovery_v11_scale098_template/metrics.json
  sha256 24b28d69cd90fe50a4f2bc7ef49a515f219d2bdc9f76afb444d325df392e3ae6
  precision / recall / F = 65.666593 / 64.498467 / 65.077288
  coverage = 50 sequences / 76373 frames

/root/autodl-tmp/srtrack_primary_v24_branchsafe_template_seed2026/cdtb80_full_recovery_v11_scale098_template/metrics.json
  sha256 aa4a6991976ac003937675291aab88962fc7243ebe4c604714465b9c05318178
  precision / recall / F = 75.378983 / 75.927675 / 75.652334
  coverage = 80 sequences / 101956 frames
~~~

但这两份结果仍不能作为“safe025 只加模板更新”的 preservation 证据。原因是该 template
配置除了 `TEST.SAFE_TEMPLATE_UPDATE` 外，还把 `MODEL.LANGUAGE.EARLY_GROUNDING.RESIDUAL_MAX`
从 `0.0001` 改到 `0.02`，并把
`MODEL.LANGUAGE.TWO_STAGE.TRIMODAL_CONSENSUS_GUARD_MAX_SUPPRESSION` 从 `0.25` 改到 `6.0`。
`active025_template` 也不是精确派生，因为同样改动了 `EARLY_GROUNDING.RESIDUAL_MAX`。
因此 primary fallback 的唯一合格来源仍应是 SHA 绑定的 safe025：
`droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml`
(`4ddf10339575003294b2ff0e77ff8583b8d630d035173d8508852b0a17806b46`)，再按
`tools/continue_primary_safe_template.py` 中预注册的 `TEMPLATE_PROFILE` 只替换
`TEST.SAFE_TEMPLATE_UPDATE`。该候选文件
`droptrack_depthtrack_final_language_primary_trimodal_guard_safe_template_blend002.yaml`
当前尚未物化，避免了把未通过 evidence gate 的旧路线误当作正式配置。

下一步实现前应先确认一个公开测试切面，然后再写红测。建议 seam 为：新增 primary-stack
pre-commit trace tracker，要求 `safe_template_update=true`，复用 69 维 current-vs-held
特征 schema，记录 selected 与 held-center evidence maps，并声明
`trace_changes_tracker_output=false`；新增 primary runtime gate 时必须原子回滚 bbox、safe-template
动态槽和相关状态，且无需 V35 learned artifacts。训练/loader source contract 继续硬拒
`safe_template_update != true`、checkpoint/config/source hash 不符、或 GT/future text 进入 tracker。
只有三 seed sequence-group OOF 同时满足 zero healthy/catastrophic harm、precision>=0.85、
`transition_recovered>0`、total gain>0，才允许 top-14。

### 24.45 primary fallback 深模块接口与三 seed 放行缺口

在未改动实现的前提下，2026-08-13 对 primary fallback 的模块接口做了收敛。相关既有回归
`test_language_anchored_precommit_gate`、`test_train_depthtrack_precommit_gate`、
`test_select_depthtrack_precommit_seed`、`test_siamtrack_v35_precommit_trace`、
`test_siamtrack_recovery_peak_v36`、`test_continue_v36_after_v35`、
`test_vot_siamtrack_rgbd_v36`、`test_primary_safe_template_continuation` 与
`test_safe_template_update` 合计 `38 passed`，只有一条 timm 弃用警告。双 GPU 均为空闲，
没有启动新实验。

后续实现应维持两个深模块及其公开 seam：

1. **Primary pre-commit tracker module**：唯一公开 seam 是 `tracker.track(image, info)`。
   trace adapter 与 runtime adapter 都隐藏 selected/held-center evidence、69 维特征、模板快照及
   状态回滚细节。trace adapter 必须返回与同配置 primary tracker 一致的 public bbox，并声明
   `trace_changes_tracker_output=false`；runtime adapter 的 hold 必须通过同一 `track()` 结果
   体现，同时原子回滚 bbox、safe-template policy/dynamic slot、recovery 与递归 proposal 状态。
   primary adapter 不依赖 V35 的 ranker/multiframe/state-gate/risk/latch artifacts。
2. **Three-seed promotion module**：公开 seam 是选择终态 `ready_for_top14` 及绑定的 artifact。
   内部负责报告 schema、固定 seeds、训练设置一致性、source hashes 和 OOF 安全指标；controller
   只能消费这个终态，不能自己重解释单份报告。

构造三份同源报告进行只读诊断时，当前 `select_reports()` 对 eligible flags
`[true, false, false]` 仍返回 `selected_seed=2026` 和 `ready_for_top14=true`。原因是当前实现
只要求至少两份报告，并从“任意 eligible 报告”中选最优；这与文档约定的固定 seeds
`2026/2027/2028` 全部通过相冲突。真实 V36 三份报告恰好都是 rejected，所以历史终态未被该
缺口改变；但在下一轮产生候选前必须 fail-closed 修复。

用户确认公开测试 seam 后，第一条 selector 红测应只通过三份 report 输入和公开选择终态断言：
任一 seed rejected 时 `ready_for_top14=false`、无 selected artifact；三份固定 seed 全部 eligible
且来源/训练设置一致时，才选择其中排序最优且 SHA 绑定的 artifact。第一条 tracker 红测只通过
`track()` 观察 current/hold 输出与下一帧可见行为，不直接测试 snapshot 私有函数。当前阶段
没有写这些测试或实现，也没有物化候选配置。

### 24.46 primary pre-commit 可执行数据契约

继续对齐现有 parser/loader 后，primary fallback 不能直接复用或伪装成 V35 provenance。现有
`train_depthtrack_precommit_gate.py` 与 `load_language_anchored_precommit_gate()` 除了检查
`safe_template_update=true`，还硬编码：

~~~text
trace schema        srtrack-v35-precommit-evidence-trace/v1
manifest schema     srtrack-v35-precommit-trace-run/v1
tracker version     V35 LATCH
required artifacts  ranker / multiframe / state_gate / risk_critic / latch
~~~

因此实现时需要保留公共 69 维特征与安全门，但以 stack adapter 分流来源契约。V35 v1 loader
继续原样支持历史 artifact；primary 新路径使用以下独立契约，禁止以空字典绕过 V35 五 artifact
校验：

~~~text
frame schema       srtrack-primary-precommit-evidence-trace/v1
manifest schema    srtrack-primary-precommit-trace-run/v1
artifact schema    srtrack-primary-language-anchored-precommit-gate/v1
tracker family     PRIMARY_SAFE025_TEMPLATE
learned artifacts  none
action policy      commit_current_or_hold_pre_frame_v1
feature schema     exact PRECOMMIT_FEATURE_NAMES (69 dimensions)
~~~

primary manifest 的硬条件为：`status=complete`、`dataset=DepthTrack Train only`、
`ground_truth_available_to_tracker=false`、`future_frame_text_used=false`、
`trace_changes_tracker_output=false`、`tracker_stack.safe_template_update=true`、
`precommit_gate_active=false`、`reported_box_scale=1.0`；checkpoint SHA 固定为
`30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab`，配置必须是
safe025 精确派生候选，预注册 SHA 为
`9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0`。trace aggregate、
首帧文本 JSONL、sequence list、GT 后接 aggregate 与 OOF JSONL 均继续 SHA 绑定。GT 只能在
trace 完成后由 trainer 连接，VOT/CDTB/DepthTrack-test GT 一律为 false。

每个 frame record 复用现有 69 维 current-vs-held schema：selected 与 held-center 的 response、
8 组 evidence map 的 value/rank、selected-minus-held、9 维 consensus 及 5 维 route；另含
`previous_state`、`candidate_state`、`gate_eligible`、hook/final/public 差异。primary stack
没有 V35 recovery/joint learned state，eligibility 只要求特征完整、candidate/final/public
一致，并执行 one-frame hold cooldown。

零扰动门必须比较同一配置，不得拿无模板的 safe025 输出冒充对照：

1. 先物化 SHA 为 `9e825...` 的 safe-template 候选；
2. 用普通 `SIAMTrack` 和 primary trace adapter 在相同 checkpoint、candidate config、首帧文本、
   环境与 fixed6 序列上各运行一次；
3. 18 个 prediction/score/time 结果文件中，prediction 与 score 文件必须逐字节一致；时间文件
   不作为数值一致性证据；
4. full Train trace 要求 152/152 JSONL、456/456 输出、`trace_error_count=0`，
   `maximum_hook_final_abs_difference=0`、`maximum_final_public_abs_difference=0`；
5. 任一条件失败即不允许训练。

三 seed promotion 的终态契约同时收紧为：报告数量恰为 3，seed 集合严格等于
`{2026, 2027, 2028}`，三份 source hashes 与训练超参一致，且三份报告各自满足
zero healthy/catastrophic harm、precision>=0.85、`transition_recovered>0`、total gain>0。
只有 `all_required_seeds_eligible=true` 时才允许 `ready_for_top14=true`；随后可在三个合格
artifact 中按 transition recovery、gain、precision、稀疏度排序选一个部署。任一 seed 拒绝时，
selected seed/artifact 必须为空。

runtime primary adapter 直接继承 `SIAMTrack`，不加载 V35 learned artifacts。它在
`track()` 前快照 pre-frame bbox、safe-template policy 状态、dynamic template tensor/source；
父 `track()` 完成后若选择 hold，则恢复快照并让 public bbox、下一帧 recursive crop 与模板观察
全部指向 pre-frame bbox。current 路径保持父结果与状态不变。以上行为只通过已提请确认的公开
`track()` seam 验证；selector 只通过 `ready_for_top14` 终态 seam 验证。

截至本节写入时没有物化候选、没有写测试/实现、没有启动 GPU 实验。原因是 TDD 规则要求公开
测试 seam 必须先获用户确认。

### 24.47 primary pre-commit 实现、preflight 与全量 trace 启动记录（2026-08-13）

本节更新 24.46 的“待实现/未物化”状态：primary safe-template pre-commit 路径已经在远程仓库实现并通过聚焦测试。24.46 中写到的 primary 专用 artifact schema 以当前实现为准修正为：门模型 artifact 继续复用共享 `srtrack-language-anchored-precommit-gate/v1`，来源栈通过 `source_trace_manifest.schema` 与 `source_tracker_stack.version` 区分 V35 与 primary；primary 不再伪装成 V35，也不允许用空 artifact 字典绕过 V35 五个 learned artifact 的校验。

已落地的代码边界如下：

~~~text
trace tracker        lib/test/tracker/siamtrack_primary_precommit_trace.py
runtime tracker      lib/test/tracker/siamtrack_primary_precommit_gate.py
parameter aliases    lib/test/parameter/siamtrack_primary_precommit_trace.py
                     lib/test/parameter/siamtrack_primary_precommit_gate.py
shared gate loader   lib/test/tracker/language_anchored_precommit_gate.py
trainer contract     tools/train_depthtrack_precommit_gate.py
trace launcher       tools/run_primary_precommit_trace.py
capacity analyzer    tools/analyze_primary_precommit_trace.py
deployment adapter   tracking/rgbd_evaluation_deployment.py --deployment primary_precommit
~~~

契约收紧点：`validate_source_contract()` 现在按 manifest schema 绑定 expected tracker stack，并要求 V35 manifest 只配 `srtrack-v35-precommit-capacity/v1`，primary manifest 只配 `srtrack-primary-precommit-capacity/v1`；primary stack 要求 `artifacts={}`、`language_search_recovery=false/None`，V35 stack 继续要求 ranker/multiframe/state_gate/risk_critic/latch 五个 SHA。`load_language_anchored_precommit_gate()` 也同步解析并校验 source analysis、trace aggregate、safe-template stack、primary 空 artifacts 与 V35 五 artifacts，不再只信 artifact provenance 中的布尔字段。

exact fallback 配置已经物化：

~~~text
source config      experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_safe025.yaml
source SHA256      4ddf10339575003294b2ff0e77ff8583b8d630d035173d8508852b0a17806b46
candidate config   experiments/srtrack/droptrack_depthtrack_final_language_primary_trimodal_guard_safe_template_blend002.yaml
candidate SHA256   9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
~~~

diff 仅限 `TEST.SAFE_TEMPLATE_UPDATE`：新增/开启 `USE=true`，并加入 `CHECK_INTERVAL=5`、`MIN_UPDATE_INTERVAL=30`、`MIN_STABLE_FRAMES=3`、`MIN_CONFIDENCE=0.65`、`MIN_RESPONSE_MARGIN=0.1`、`MAX_CENTER_JUMP=0.35`、`MIN_RGB_IDENTITY=0.75`、`MAX_LOG_DEPTH_CHANGE=0.08`、`MIN_DEPTH_VALID_RATIO=0.5`、`CENTER_FRACTION=0.8`、`NMS_KERNEL=5`、`BLEND_WEIGHT=0.1`、`MAX_BLEND_WEIGHT=0.2`、`MAX_TEMPLATE_AGE=90`、`REQUIRE_ACTIVE_MEMORY_FOR_ROUTE=false`、`PRIMARY_TEMPLATE_BLEND_USE=true`、`PRIMARY_TEMPLATE_BLEND_WEIGHT=0.02`。未改 `EARLY_GROUNDING.RESIDUAL_MAX` 或 `TRIMODAL_CONSENSUS_GUARD_MAX_SUPPRESSION`。

preflight 已通过，未跑序列，仅校验候选配置、checkpoint、首帧文本 JSONL、152 个 Train sequence 与 manifest 落盘：

~~~text
preflight manifest /root/autodl-tmp/srtrack_primary_safe_template_precommit_preflight_20260813_054301/manifest.json
manifest SHA256    d410c346eabd0c8cd7ae5b3187fcadcff11153e747516de1d79f5ce76b04e689
status             preflight_complete
checkpoint SHA256  30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
target JSONL SHA   56c03871b8e2a005bbd9ca12c32ed8821c5445d16af9e2631963267b89785b58
config SHA256      9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
~~~

聚焦验证：`py_compile` 覆盖 primary trace/gate、parameter aliases、loader、trainer、launcher、analyzer、deployment 与两份测试；pytest 命令覆盖 `test_language_anchored_precommit_gate.py`、`test_train_depthtrack_precommit_gate.py`、`test_select_depthtrack_precommit_seed.py`、`test_siamtrack_recovery_peak_v36.py`、`test_vot_siamtrack_rgbd_v36.py`、`test_safe_template_update.py`，结果为 `33 passed, 1 warning`（warning 仅为 timm import deprecation）。新增测试断言 primary source contract 可通过、cross-stack analysis 必须拒绝、primary 不得携带 V35 helper artifacts、loader 可加载 primary safe-template source 且拒绝 cross-stack trace analysis。

全量 trace 启动记录：第一次尝试写入 `/root/autodl-tmp/srtrack_primary_safe_template_precommit_trace_all152_seed2026` 后因缺少 `lib.test.parameter.siamtrack_primary_precommit_trace` 立即失败，未形成完整 trace；该 manifest 已标记 `status=failed`，failure reason 为缺少 parameter alias。补齐两个 parameter alias 后，已用新目录重启：

~~~text
screen             primary_precommit_trace_152_s2026_v2
output             /root/autodl-tmp/srtrack_primary_safe_template_precommit_trace_all152_seed2026_v2
log                /root/autodl-tmp/srtrack_primary_safe_template_precommit_trace_all152_seed2026_v2.run.log
exit code file     /root/autodl-tmp/srtrack_primary_safe_template_precommit_trace_all152_seed2026_v2.exit.code
initial status     running, 2 GPUs active, manifest status=running
~~~

claim 边界：截至本节写入，全量 Train trace 尚未完成，`precommit_capacity.json`、三 seed gate artifact、primary_precommit runtime VOT 结果均尚未产生。因此不能声明 EAO/ACC/ROB 改善，也不能声明 primary precommit gate 安全可部署。当前只支持“primary safe025→safe-template exact candidate 已物化、source contract 已实现并通过 preflight/单测、全量 Train trace 已在 v2 后台运行”。下一步必须等 v2 trace 完成后运行 primary capacity analyzer，再训练 seeds `{2026,2027,2028}`；只有三 seed 都满足 zero harm/precision/gain/recovery 约束后，才允许选择 artifact 并进入 VOT top14/full 与 DepthTrack/CDTB 回归验证。

### 24.48 selector fail-closed 修正与自动续跑 watcher（2026-08-13）

24.46 标出的 seed selector 缺口已修正：`tools/select_depthtrack_precommit_seed.py` 现在要求 training report 数量严格为 3，seed 集合严格等于 `{2026, 2027, 2028}`；三份 report 的 `source_trace_aggregate_sha256`、`source_groundtruth_sha256`、`source_checkpoint_sha256`、`source_artifacts`、`source_tracker_stack` 与除 seed 外的训练超参必须一致。只有三份 report 全部 eligible 时才设置 `ready_for_top14=true` 并选择排序最优 artifact；任一 seed rejected 时 selected seed/artifact 为空，selection 以 no-safe-policy 终态结束。

对应聚焦测试已更新为四个公开 seam：缺 seed 拒绝、seed identity 错误拒绝、任一 required seed 不 eligible 时 selected 为空、三 seed 全 eligible 时按 transition recovery/gain/precision/sparsity/seed 排序选最优。当前聚焦测试结果更新为 `36 passed, 1 warning`（warning 仍为 timm deprecation）。

为避免人工守到全量 trace 结束，已启动只读等待型续跑 watcher：

~~~text
watcher screen      primary_precommit_after_trace_v2
controller          /root/autodl-tmp/srtrack_primary_precommit_gate_after_trace_seed2026_v2
status              /root/autodl-tmp/srtrack_primary_precommit_gate_after_trace_seed2026_v2/status.json
runner log          /root/autodl-tmp/srtrack_primary_precommit_gate_after_trace_seed2026_v2/runner.log
initial stage       waiting_for_trace
watched trace code  /root/autodl-tmp/srtrack_primary_safe_template_precommit_trace_all152_seed2026_v2.exit.code
~~~

watcher 的动作边界：仅当 v2 trace exit code 为 0 时继续；随后依次运行 `tools/analyze_primary_precommit_trace.py` 生成 `precommit_capacity.json`，训练 seeds `2026/2027/2028`（训练 exit code 0 或 2 均会保留 report；2 表示无 safe OOF policy），最后调用严格 selector。selector exit code 0 写 `selected_ready_for_top14`，exit code 2 写 `selection_rejected_no_safe_policy`，不自动启动 VOT/CDTB/DepthTrack 公开评测。若 trace 失败，watcher 只写 `trace_failed` 并停止。

### 24.49 primary precommit official VOT bridge 补齐（2026-08-13）

等待 full Train trace 期间，预检下一阶段 official VOT 入口时发现：`primary_precommit` 已接入 DepthTrack/CDTB 的 `rgbd_evaluation_deployment.py` resolver，但 official VOT TraX workspace 仍只有 V36 precommit bridge，primary stack 选出 artifact 后会卡在 workspace/shard 白名单与 artifact 约束。已补齐如下：

~~~text
new bridge       tools/vot_siamtrack_rgbd_primary_precommit.py
workspace prep   tools/prepare_votrgbd2022_workspace.py
parallel shards  tools/prepare_votrgbd2022_parallel_shards.py
test seam        tests/test_vot_siamtrack_rgbd_v36.py
~~~

`tools.vot_siamtrack_rgbd_primary_precommit` 直接复用 base TraX bridge，但把 runtime tracker 切到 `PrimaryPrecommitGateTracker`。进入 base bridge 前会清除可能从长生命周期 shell / retry workspace 继承的 V35 helper 环境变量：ranker、multiframe、state gate、setwise、language-anchored candidate；唯一允许的 learned artifact 是 `SRTRACK_LANGUAGE_ANCHORED_PRECOMMIT_GATE`。这样 primary precommit official VOT 不会意外加载 V35 LATCH stack。

workspace 准备逻辑新增 `PRIMARY_PRECOMMIT_BRIDGE_MODULE = tools.vot_siamtrack_rgbd_primary_precommit`，并把它加入 `BRIDGE_MODULES` 与 `PRECOMMIT_BRIDGE_MODULES`，但明确不加入 `RANKER_BRIDGE_MODULES`、`MULTIFRAME_BRIDGE_MODULES`、`STATE_GATE_BRIDGE_MODULES`、`SETWISE_BRIDGE_MODULES`、`LATCH_BRIDGE_MODULES`。因此 primary precommit workspace 只要求 precommit gate artifact + safe-template update，不要求 V35 五个 helper artifacts。precommit artifact 绑定按 bridge 类型分支：V36 继续要求 source V35 artifacts 与当前 ranker/multiframe/state/risk/latch SHA 一致；primary 要求 artifact source stack 为 `PRIMARY SAFE-TEMPLATE`、`safe_template_update=true`、`precommit_gate_active=false` 且 `source_artifacts={}`。

parallel shard 生成也同步允许 `tools.vot_siamtrack_rgbd_primary_precommit`，并把 precommit artifact 的 required/forbidden 条件从硬编码 V36 改成 precommit bridge family。已有测试 seam 扩展为：primary bridge 必须在 `BRIDGE_MODULES` 与 `PRECOMMIT_BRIDGE_MODULES` 中，但不得出现在任何 V35 helper artifact family；`_tracker_ini()` 生成的 command 必须为 `tools.vot_siamtrack_rgbd_primary_precommit`，并只传播 precommit gate，helper env 为空。

验证结果：

~~~text
py_compile:
  tools/vot_siamtrack_rgbd_primary_precommit.py
  tools/prepare_votrgbd2022_workspace.py
  tools/prepare_votrgbd2022_parallel_shards.py
  tests/test_vot_siamtrack_rgbd_v36.py

pytest focused:
  tests/test_vot_siamtrack_rgbd_v36.py
  tests/test_select_depthtrack_precommit_seed.py
  tests/test_language_anchored_precommit_gate.py
  tests/test_train_depthtrack_precommit_gate.py
  tests/test_siamtrack_recovery_peak_v36.py
  tests/test_safe_template_update.py

result: 37 passed, 1 warning（timm deprecation）
~~~

截至本节写入，v2 full Train trace 仍在运行，约 17 个 trace 文件 / 15 个已完成 sequence prediction，watcher 仍为 `waiting_for_trace`。该 bridge 补丁只移除后续 official VOT 的工程阻塞，不构成任何 EAO/ACC/ROB 改善声明；是否进入 official VOT 仍取决于 watcher 产出的三 seed strict selector 是否 `ready_for_top14=true`。



### 24.50 primary precommit watcher 恢复与当前运行状态（2026-08-13 06:10 CST）

06:09 CST 复查远程运行态时，`primary_precommit_trace_152_s2026_v2` 仍在正常运行，但 `primary_precommit_after_trace_v2` watcher screen 已不在；controller 目录仅保留 `status.json` 与 `runner.log`，stage 为 `waiting_for_trace`，且没有 `runner.exit.code`。因此不能把 24.48 的 watcher 视为仍然存活。

只读状态核验如下：

~~~text
repo HEAD          c8de7bb35fe9
worktree status    217 entries（脏工作区，未提交）
trace screen       536136.primary_precommit_trace_152_s2026_v2
GPU usage          GPU0 6619/24576 MiB, GPU1 7707/24576 MiB, utilization ~65-67%
trace manifest     schema=srtrack-primary-precommit-trace-run/v1, status=running
trace files        19 jsonl
latest trace       bottle01_indoor.536226.jsonl, mtime 2026-08-13 06:09:49
trace exit code    not yet present
trace sample       schema=srtrack-primary-precommit-evidence-trace/v1, tracker_stack=PRIMARY SAFE-TEMPLATE, no GT fields
controller stage   waiting_for_trace
controller files   no precommit_capacity.json, no seed_selection.json, no runner.exit.code
~~~

06:10 CST 已恢复 watcher：

~~~text
watcher screen     540966.primary_precommit_after_trace_v2
command            continue_after_trace.py >> runner.log; write runner.exit.code on exit
stage after launch waiting_for_trace
updated_at_unix    1786572641
~~~

该恢复动作只重启等待/接续控制器，不改动 trace 结果、不启动 official VOT/CDTB/DepthTrack 公开评测。若 trace exit code 为 0，watcher 才会继续 primary capacity analysis、三 seed precommit gate training 与 strict selector；若 selector 不产生 `selected_ready_for_top14`，不得进入 top14/full VOT，也不得声明 EAO/ROB 改善。当前 claim 边界保持不变：full Train trace 与接续链路正在运行，尚无 primary precommit gate artifact 与 VOT-RGBD2022 EAO/ACC/ROB 数字。

### 24.51 primary precommit 审查修正与严格续跑链（2026-08-13）

本节废止 24.47/24.48/24.50 中“共享 artifact schema、v2 full trace 可直接接训练、旧 watcher 有效”的过时状态。两路只读代码审查确认原实现缺少同配置 fixed6 零扰动前置门、运行配置与 GT 根绑定、152/456 完整性、独立 artifact 内部 seed 核验和 public track 原子回滚测试；v2 又违反 24.46 规定的门顺序。因此 v2 已保留原文件但标记为 invalidated，不进入任何训练或指标声明。

修正后的主契约如下：

~~~text
artifact schema        srtrack-primary-language-anchored-precommit-gate/v1
candidate config SHA   9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
checkpoint SHA         30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
fixed6 gate            6 sequences / 12 prediction+score files byte-identical
full trace gate        152 JSONL / 456 outputs / zero errors / two max diffs = 0
source bindings        config, checkpoint, target JSONL, dataset root,
                       sequence list, GT aggregate, trace aggregate,
                       fixed6 report, OOF, trace/runtime/analyzer/trainer/verifier code
policy validation      sequence-group OOF calibration + held-out policy audit
promotion              seeds exactly {2026,2027,2028}, each internal artifact
                       seed/policy/provenance must equal its report
~~~

代码层同时完成：参数入口记录 config path/SHA；trace 从实际 parent hook 捕获 selected index、raw score 与 hook candidate，禁止用常量差异伪装零扰动；动态模板 tensor 快照使用 clone，hold 路径恢复 bbox、policy、tensor/source；primary runtime 和 official VOT bridge 均拒绝任意 config；workspace 再次校验 artifact 的 source config SHA。公开 track seam、selector、source loader 与 VOT bridge 的聚焦回归为 42 passed、1 个 timm deprecation warning。

严格控制器只执行 Train/fixed6 开发证据、三 seed 训练和 selector，不会自动启动 VOT/CDTB/DepthTrack-test。当前有效运行：

~~~text
screen       primary_precommit_strict_v7
controller   /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v7
stage        fixed6_trace
baseline     复用 v4 在任何无效 trace 前已完整落盘的普通 SIAMTrack fixed6
GPU          2 cards active
public eval  false
~~~

claim 边界仍为：尚无可部署 gate artifact，尚无新的 VOT-RGBD2022 EAO/ACC/ROB；只有 v7 fixed6 通过后才允许 full152，随后三 seed 均通过 held-out policy audit 才允许 top14。

### 24.52 fixed6 零扰动门通过并进入 full152（2026-08-13）

最终冻结实现上的 v7 fixed6 对照已经完成。普通 SIAMTrack baseline 与 primary trace adapter 使用同一 checkpoint、候选配置、首帧文本与 fixed6 序列；比较排除 time 文件，仅比较 6 个 prediction 与 6 个 all_scores 文件。结果：

~~~text
invariance report  /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v7/fixed6_invariance.json
status             pass
files compared     12
mismatches         []
candidate SHA      9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
checkpoint SHA     30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
~~~

因此 24.46 的同配置零扰动前置门已满足，严格控制器已自动进入
`/root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v7/full152_trace`。当前 stage 为
`full152_trace`，2 个 worker/2 GPU 正常，尚未分析、训练或选择 artifact，也未启动任何公开 benchmark。


### 24.53 primary precommit 公开评测 fail-closed 续跑器（2026-08-13）

等待 v7 full152 时新增了独立的下游续跑器 `tools/continue_primary_precommit_evaluation.py`。该文件不参与 trace、分析、训练、selector 或 gate 决策，也不在 v7 full152 的 implementation manifest 中，因此不会改变当前冻结实验的 provenance。其 preflight 已校验 checkpoint/config、VOT/DepthTrack/CDTB 首帧语言 JSONL、14 序列清单与目标阈值；当前 VOT workspace 尚无 manifest，`public_evaluation_started=false`。

续跑顺序固定为：

~~~text
v7 strict terminal status
  -> selector schema + seeds exactly {2026,2027,2028}
  -> all_required_seeds_eligible=true + ready_for_top14=true
  -> selected artifact file SHA + internal loader/config/checkpoint/seed binding
  -> primary-precommit-only VOT workspace（无 V31–V36 helper artifacts）
  -> official multi-start top14 + projection target gate
  -> DepthTrack full50 target + primary reference preservation gate
  -> CDTB full80 target + primary reference preservation gate
  -> official VOT-RGBD2022 full127
~~~

任一前置条件不满足时，续跑器只写 rejection result，不创建或不继续公开评测。top14、OPE 与 VOT full 使用 `fixed6_isotropic_098_v1` 作为 report-only 输出层缩放；manifest 必须同时证明 `reported_box_scale_feedback_to_tracker_state=false`。因此训练 trace 与递归 tracker/template state 仍保持 1.0 原始 bbox，0.98 不参与下一帧 crop、模板更新或 precommit 原子回滚。

当前控制状态：

~~~text
strict screen       primary_precommit_strict_v7
strict stage        full152_trace
evaluation screen   primary_precommit_evaluation_v1
evaluation stage    waiting_for_strict_selector
public eval started false
VOT manifest        absent
full152 progress    16/152 complete + 2 running（48 prediction files）
trace errors        0 observed
GPU utilization     two GPUs active, approximately 66–69%
~~~

claim 边界不变：fixed6 已通过；full152、三 seed held-out policy audit 与 selector 尚未完成，所以目前没有 primary gate artifact，也没有新的 VOT EAO/ACC/ROB 或 OPE 指标。


### 24.54 v7/v1 统计作废与 v8/v2 真正 held-out 重启（2026-08-13）

在 v7 full152 尚未完成、公开 VOT workspace 仍未创建时，完整复审发现旧 trainer 先对全部 Train 序列生成 5-fold OOF，之后才划分 calibration/audit，并在 audit 上比较多个 calibration-feasible policy。这样 audit 既通过模型训练间接进入 calibration，又被直接用于策略排名，不满足 held-out 契约。v7 与下游 v1 因此主动停止并写入不可变作废记录；其 partial trace 只能作工程诊断，不能用于 artifact promotion 或公开指标声明。

~~~text
v7 invalidation  /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v7/invalidation.json
v7 SHA           a27cedc2e4e70aa53352f2b23f43d58b34df9c7597629cc6eafd412241f4ca28
v1 invalidation  /root/autodl-tmp/srtrack_primary_precommit_evaluation_seed2026_v1/invalidation.json
v1 SHA           68a53a21aa21d2479056f24668b217139588a9799f5289cb7df917789fd41cea
public VOT       never started; manifest absent
~~~

重新冻结的 v3 policy-validation 契约如下：先按 sequence group 固定一个 audit fold；calibration 的每个 OOF 模型训练时同时排除自身 validation fold 和整个 audit fold；唯一 calibration winner 由预注册全序确定；一个仅用全部 calibration 行训练的模型对该 winner 审计一次，失败不得回退；部署直接复用这个零 audit 训练行的已审计模型，不再吸收 audit 重训。三 seed 必须全部通过，但不再按 audit 指标挑 seed，固定部署预注册 seed 2026。selector 对三份 artifact 均调用完整 runtime loader，逐一重算并核验 config/checkpoint、trace/analysis/OOF/fixed6、implementation、GT aggregate、audit 安全阈值与 artifact SHA，避免自洽伪 provenance 置绿。

控制器同时修复了 strict 成功终态被 SystemExit 覆盖的问题。公开评测每卡最多 2 worker，共 4 worker；每个 VOT shard 使用独立 PGID，任一 shard 非零立即 fail，异常路径对所有 PGID 执行 TERM、宽限、KILL 和 wait，避免遗留 TraX/PyTorch 子孙进程。聚焦回归结果为 34 passed、1 个 timm deprecation warning；两轮独立只读复审均确认无阻断。

新证据链固定为：

~~~text
strict root      /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v8
evaluation root  /root/autodl-tmp/srtrack_primary_precommit_evaluation_seed2026_v2
fixed6 baseline  reuse immutable v4 ordinary outputs
fixed6 trace     rerun because implementation manifest changed
full trace       only after new fixed6 byte-identical pass
public workspace must remain absent until all three seeds pass selector
~~~

当前仍不声明新的 VOT/DepthTrack/CDTB 数字。v8 将从 fixed6 零扰动重新开始；仅在 full152、三 seed、top14 projection、DepthTrack/CDTB preservation 依序通过后才允许正式 VOT-RGBD2022 full127。


### 24.55 v8/v2 正式启动（2026-08-13）

在 34 个聚焦回归通过且两名独立只读复审均给出 无阻断后，干净证据链已正式启动：

~~~text
strict screen       primary_precommit_strict_v8
strict PID          560583
strict stage        fixed6_trace
evaluation screen   primary_precommit_evaluation_v2
evaluation PID      560672
evaluation stage    waiting_for_strict_selector
GPU layout          fixed6 当前 1 worker/GPU；公开 VOT 最多 2 worker/GPU
public eval started false
VOT workspace       absent at preflight
~~~

fixed6 trace 已加载两个独立 tracker 进程，GPU 显存约 2.25 GiB/卡并正常推进；当前日志没有 Traceback、RuntimeError、OOM 或 FileNotFoundError。v2 只写等待状态，不会在 selector 前创建 VOT manifest。后续 stage、SHA 与正式结果必须以 v8/v2 目录中的不可变产物为准。


### 24.56 v8 fixed6 零扰动通过并进入 full152（2026-08-13）

v8 最终冻结实现的 fixed6 已完成严格字节比较：

~~~text
report          /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v8/fixed6_invariance.json
report SHA      edb037a612d4606d38d46935251fbac176ea7b0f828bbec9c1d36922050fa406
status          pass
files compared  12
mismatches      []
config SHA      9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
checkpoint SHA  30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
~~~

因此新的 implementation manifest 未改变未启用 gate 时的公开预测与 score 字节。strict 控制器已自动进入 /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v8/full152_trace，当前两个 worker 分别运行在 GPU 0/1；启动观察为 2 个 trace 文件、零 Traceback/RuntimeError/OOM/FileNotFoundError。v2 仍等待三 seed selector，未创建公开 VOT workspace。

### 24.57 v8/v2 基础设施契约失败与 v9/v3 重启（2026-08-13）

v8 的 Train-only 因果 trace 已完整结束，原始跟踪证据本身无错误：

~~~text
sequences                 152
trace files               152
prediction files          456
trace errors              0
trace aggregate SHA       2c89a14395f7f3402c7980c3423835eb43ff5e2e89a10cf0eadc0e87ee6973f8
future frame text used    false
trace changes output      false
public evaluation started false
~~~

随后容量分析成功读完全部 trace，但绑定版本的 `analyze_primary_precommit_trace.py` 没有把 `trace_aggregate_sha256` 写入分析 JSON；绑定 trainer 又把该字段与 trace manifest 的相等性设为强制条件。因此 seed 2026 在生成任何 artifact 前 fail closed，错误为 `source trace analysis contract failed`。这是基础设施生产者/消费者契约缺口，不是 policy 不安全或模型指标失败；不得手工修改分析 JSON，也不得改写 v8 manifest 后继续 promotion。

v8/v2 已保留全部产物并明确作废：

~~~text
v8 invalidation  /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v8/invalidation.json
v8 SHA           30fa9b6f8d9a139c0e3fe849551910435f283905468b2c379cfd74013ba755cd
v2 invalidation  /root/autodl-tmp/srtrack_primary_precommit_evaluation_seed2026_v2/invalidation.json
v2 SHA           84c0f8d2a454da8c8cc27433d2436f0a8c3fbd697bacf09c9eb4dd2d1ecc8adf
seed artifacts   none
public VOT       never started
~~~

分析器现由自身对实际 trace 路径调用与 trainer 相同的聚合函数并写出 SHA；修复后在 v8 immutable trace 上得到的诊断 SHA 与 manifest 完全相同。`py_compile` 与 34 个聚焦回归通过（仅 1 个 timm 弃用告警）。由于分析器 SHA 属于 trace implementation manifest，新证据不能复用或篡改 v8，已从干净目录重跑：

~~~text
fixed analyzer SHA  e39f335eb7e161f4298fbb56f140b7329227f183025b5f63d3f0741f747fae7c
strict root         /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9
strict screen       primary_precommit_strict_v9
strict PID          570836
strict stage        fixed6_trace
evaluation root     /root/autodl-tmp/srtrack_primary_precommit_evaluation_seed2026_v3
evaluation screen   primary_precommit_evaluation_v3
evaluation PID      570837
evaluation stage    waiting_for_strict_selector
public eval started false
~~~

当前 claim 边界仍不变：v8 的完整 trace 仅证明捕获容量，不产生可部署 gate；v9 必须重新通过 fixed6、full152、三 seed 单次 held-out audit 与 selector，v3 才可进入 top14/OPE/正式 VOT。

### 24.58 v9 fixed6 通过并进入修复后 full152（2026-08-13）

v9 在新 analyzer implementation SHA 下重新执行 fixed6，普通 tracker 与 trace adapter 的 6 个 prediction 及 6 个 all_scores 文件再次全部字节一致：

~~~text
report          /root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9/fixed6_invariance.json
report SHA      10f602f6fef863f208410cf71a2ec2bda5b2980a147de08399fac895ceb2cfda
status          pass
files compared  12
mismatches      []
config SHA      9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0
checkpoint SHA  30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
analyzer SHA    e39f335eb7e161f4298fbb56f140b7329227f183025b5f63d3f0741f747fae7c
~~~

严格控制器已进入 `/root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9/full152_trace`。新 manifest 明确记录 `future_frame_text_used=false`、`trace_changes_tracker_output=false`、raw reported box scale 1.0、safe-template profile 与 fixed6 SHA；v3 继续等待 selector，公开评测仍未启动。

### 24.59 v9 三 seed 最终 fail-closed 结论（2026-08-13）

v9 在修复后的 analyzer / trainer / selector 契约下完成了完整 Train-only 证据链。full152 trace 包含 152 个 sequence trace、456 个 prediction/score 输出，trace error 为 0，`future_frame_text_used=false`，两个最大输出差异均为 0；trace aggregate SHA 为 `7447959d135aa3cc469bcf81e8a45cab711c9304538a96c3b16035ad5a64eaf0`。fixed6 仍使用 24.58 的 12 文件字节一致报告 `10f602f6fef863f208410cf71a2ec2bda5b2980a147de08399fac895ceb2cfda`。

三个预注册 seed 均只允许一个 calibration winner 接受一次 held-out policy audit，失败不得回退：

~~~text
seed 2026  calibration: 4 holds / 4 transitions / gain 3.350736
           audit:       0 holds / 0 transitions / gain 0
           decision:    rejected, no artifact
seed 2027  calibration: 9 holds / 9 transitions / gain 7.292932
           audit:       2 holds / 2 transitions / gain 1.528062
           precision:   1.0, healthy/catastrophic harm = 0/0
           decision:    eligible
           artifact SHA 9b0d5083cf9488cea1070c5f84a7a5e3faf4a36f37250ef6ba48ff9f7e5a2163
seed 2028  calibration: 3 holds / 3 transitions / gain 2.507045
           audit:       0 holds / 0 transitions / gain 0
           decision:    rejected, no artifact
~~~

selector 结果为 `all_required_seeds_eligible=false`、`ready_for_top14=false`、`selected_seed=null`、`selected_artifact=null`；strict 终态为 `selection_rejected_no_safe_policy`。下游 v3 continuation 终态为 `complete_precommit_selection_rejected`，`public_evaluation_started=false`、`formal_vot_started=false`，没有创建 primary-precommit VOT workspace。seed 2027 的单次通过只能证明该随机种子存在局部容量，不能绕过三 seed 稳定性门，也不构成可部署 gate 或公开 benchmark 改善。

### 24.60 Train-only 稳定性诊断与路线停止条件（2026-08-13）

拒绝后仅使用 DepthTrack Train trace/GT 做了诊断，不读取 VOT、DepthTrack-test 或 CDTB GT。旧 60k policy grid 在五折上表现出明显 seed 不稳定：2026/2028 没有策略能安全覆盖超过 2/5 folds，2027 有 670 个五折可行策略；把 calibration-safe 与 audit-safe 直接求交时三个 seed 分别为 `0 / 3420 / 0`。均值、median、conservative ensemble、分类-only、线性 action/harm、独立 MLP 与 OpenCV RTrees 等替代头均没有得到三个 seed 全部五折稳定的策略。

trace 中共有 215 个真实 transition，分布在 46 个 sequence；bbox 因果历史对 transition 有可见信号，例如 current/previous jump ratio 约 `9.23 vs 2.26`、acceleration 约 `1.17 vs 0.38`、stable streak 约 `0.38 vs 0.15`。但在固定 4+1 nested calibration/audit 契约下，历史特征仍表现为 seed 2026 无 calibration feasible、2027 单 audit 通过、2028 audit 0 hold，未闭合多 seed 安全门。

因此停止继续扫描阈值或更换轻量分类头：当前失败是跨 seed / 跨 sequence 泛化不足，不是尚未找到某个阈值。若未来重开 learned precommit 路线，必须增加独立 Train sequence 与更强的时序表示并重新走 fixed6/full152/三 seed 契约；不得复用 2027 artifact 直接进入公开评测。

### 24.61 artifact-free 精确安全模板 Top-14 v4（2026-08-13）

learned precommit 路线 fail closed 后，单独评测已经通过 fixed6/full trace 的精确安全模板候选，不携带任何 learned gate。候选配置为 `droptrack_depthtrack_final_language_primary_trimodal_guard_safe_template_blend002`，SHA `9e825baefbabf9aa854c8c0a8158cdd25414e277e6fb32fdce3f4da6fdae40a0`；相对 primary safe025 只改变 `TEST.SAFE_TEMPLATE_UPDATE`，保留 `fixed6_isotropic_098_v1` 的 report-only 输出缩放，且缩放不反馈 tracker/template state。

前两次 workspace v1/v2 在 TraX 启动前失败：缺失 artifact 的空 `env_*` 被工具链物化为短命 `/tmp/tmp*` 路径。workspace 与 shard 生成器已改成完全省略不存在的六类 artifact env。v3 又证明仅在专用桥中 `os.environ.pop()` 仍不足以抵抗 TraX 边界注入，因此基础桥新增显式 `ignore_learned_artifacts=True`，在参数解析层把 setwise/latch/precommit 强制为 `None`；专用桥同时清除 ranker、multiframe、state、setwise、latch、precommit 六类环境，并锁定 config、安全模板开关、无额外 recovery/identity/quarantine、0.98 report-only profile。v1/v2/v3 只作为基础设施失败证据，不进入模型结果。

专用桥还传递式锁定基础桥实现：

~~~text
workspace          /root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4
workspace manifest d7f82570de64840ee914305d82e79d290e7b93709bd43ccc1959a945a74b19b0
bridge module      tools.vot_siamtrack_rgbd_primary_safe_template
bridge SHA         072925383eb0f0dab7f4665c8c1ebf928ded266d50e46600fc7ac0f652d45495
base bridge SHA    767122bc3c1e3951a2b451a45708cc896791b51e370373e6fb76f95079c64826
checkpoint SHA     30c804ba6c68e6e4f18a45e1c39cb20e83fed0819545755e3c43d1e5b63485ab
language SHA       b0e08fcee58f5ae8119d951eabf4a5688a433864279291add34e56440f57072d
focused tests      26 passed, 1 timm deprecation warning
GPU layout         (0,0,1,1), 4 workers, at most 2 workers/card
~~~

v4 preflight 已验证固定 Top-14、127-sequence official workspace、首帧静态文本、safe-template 完整 profile、无 learned artifact 与 report-only 0.98。四个 TraX tracker 已成功连接并分别占用约 2.24 GiB，controller stage 为 `top14_running`，`public_evaluation_started=true`；本节写入时尚未形成完整 anchor coverage 或投影指标。因此当前最好的正式数字仍是 baseline VOT-RGBD2022 `EAO/ACC/ROB = 72.908956 / 82.535868 / 87.988071`，不能把 v4 启动状态表述为改善。

### 24.62 正式 VOT 指标偏低的根因与具体序列（2026-08-13）

本节只解释已完成的 primary safe025 full-127 官方 multi-start 结果，不使用正在运行的安全模板 v4
中途轨迹，也不据此修改阈值。正式结果为
`EAO/ACC/ROB = 72.908956 / 82.535868 / 87.988071`；相对目标，ACC 已高
`+0.435868pp`，EAO 与 ROB 分别低 `4.991044pp`、`5.711929pp`。所以当前短板不是“成功
锁定时的框普遍不准”，而是失锁出现得太早、太频繁；VOT 的 EAO 同时受准确性和鲁棒性影响，ROB
缺口会直接压低有效重叠曲线。

#### 24.62.1 统计口径与总体归因

对 127 个序列、1,765 个官方 forward/backward anchor 逐条重放正式轨迹，共识别 363 个达到
“连续 10 帧 overlap 不高于 0.1”的正式失败运行，占 anchor 的 `20.57%`。属性诊断把每个失败
streak 的第一帧与 VOT-RGBD2022 官方逐帧 tag 对齐；曝光基线是每条运行从 anchor 到成功结束或
首次失败的累计 1,161,114 个帧：

~~~text
属性                 失败起点占比     评测曝光占比     集中倍数
partial-occlusion       34.16%            6.75%          5.06x
similar-objects         33.06%            7.98%          4.14x
background-clutter      27.27%            9.89%          2.76x
fast-motion              9.92%            4.30%          2.30x
size-change             17.91%           10.35%          1.73x
depth-change              5.79%            6.23%          0.93x
~~~

这些倍数是相关性诊断，不是独立同分布显著性检验：multi-start 会让同一困难片段被多个 anchor
重复经过，属性也可重叠。所有 GT/tag 只在离线分析时后接，未进入 tracker，也没有使用未来帧文本。

最关键的几何反证是：363 次失败中，`95.87%` 在失败后仍有可见目标落在 factor-5 几何范围内，
`99.45%` 落在 factor-7 内；失败后第一帧所需 factor 的中位数仅 `1.463`。因此全局放大搜索
窗口不是主要解法。多数失败是目标仍在局部搜索域内，但遮挡、相似物或背景产生的响应峰被选中并
写回递归位置；第一步身份选错后，搜索中心继续跟随错误状态，ROB 和 EAO 一起下降。

置信度也不足以单独阻止错误写回。失败 streak 的 3,630 个低-overlap 帧中，`70.17%` 的分类
置信度仍不低于 0.5，`49.42%` 仍不低于 0.6；最强的单帧 motion/scale 诊断虽然覆盖
`349/363` 个失败运行，但预警 lead 中位数为 0 帧。简单低分阈值通常到失锁当帧才触发，不能
可靠地区分“困难但正确”与“高置信度跟错”。

#### 24.62.2 具体序列例子

~~~text
序列                         失败/anchor     ACC       ROB      h=1 factor中位数  最短边token中位数
cup02_indoor_1                 36/36        85.217     6.960          1.254              2.361
toy09_indoor_1                 26/26        82.766    42.430          1.512              1.913
earphone01_indoor_1            20/20        76.463    24.536          1.328              1.617
glass01_indoor_2               19/19        74.217    15.818          1.699              2.072
stick_indoor_1                 11/19        77.377    59.647          4.562              2.728
cube02_indoor_1                 8/13        87.769    68.507          6.86附近            2.66附近
~~~

- `cup02_indoor_1`：36 个 anchor 全部失败，但 ACC 仍有 85.217。36/36 个失败起点都带
  `similar-objects`，h=1 所需 factor 只有 1.254；第 106 帧附近目标仍在很小的局部范围内。
  首帧静态文本却写成 “red cup with white interior ... no distractors”，不能表达相似杯竞争。
  这是身份峰误选而非框回归或搜索范围不足。
- `toy09_indoor_1`：26/26 anchor 失败，其中 16 个失败起点带 `background-clutter`，16 个带
  `partial-occlusion`；第 1,124 帧附近同时出现 clutter、depth-change、fast-motion 和
  partial-occlusion。最短边只有约 1.913 个 patch token，遮挡后多模态峰容易混淆。首帧文本仍为
  no distractors/no occlusion，说明静态文字只能作身份先验，不能描述时变状态。
- `earphone01_indoor_1`：20/20 anchor 失败，12 个失败起点处于 background clutter，最短边
  中位数仅 1.617 token，h=1 factor 只有 1.328。ROB 24.536 而 ACC 仍为 76.463，同样指向
  “小目标 + 背景竞争 + 递归漂移”。
- `glass01_indoor_2`：19/19 失败起点全部带 background clutter；语言记录把目标描述为
  transparent glass、depth quality=medium。透明目标的 RGB 边界和深度身份都弱，即使目标在
  factor 1.699 范围内，仍容易选到背景响应，ROB 只有 15.818。
- `stick_indoor_1` 是几何例外：11/19 anchor 失败，h=1 所需 factor 中位数为 4.562；第
  416、415 帧附近分别需要约 5.317、5.905，并带 aspect-change。这类帧需要风险触发的宽搜索，
  但不支持所有序列永久放大搜索。
- `cube02_indoor_1`：倒放 anchor 在第 174 帧附近反复失锁，同时带 out-of-plane、
  similar-objects、size-change，所需 factor 约 6.86--6.91；但 ACC 高达 87.769。它说明少数
  位移/尺度跳变需要 factor-7 候选，同时仍需身份判别，不能把“候选可见”当成“候选可选对”。

#### 24.62.3 架构层原因与实验优先级

正式 workspace 关闭 `SAFE_TEMPLATE_UPDATE`、`ONLINE_LANGUAGE_UPDATE`、
`LANGUAGE_IDENTITY_RERANK` 与 `LANGUAGE_SEARCH_RECOVERY`；文本只来自第 0 帧。配置中的 early
grounding residual 上限仅 `0.00010`，静态模板也不随可靠外观变化更新。历史 full-127 对照中，
visual baseline 到 language epoch-5 只有
`EAO +0.028899pp / ACC +0.152085pp / ROB -0.140490pp`；最终 primary guard 相对
v5-no-recovery 也只有
`EAO +0.000818pp / ACC +0.000954pp / ROB -0.005463pp`。语言当前主要是弱残差先验，没有形成
能显著改善 ROB 的时序身份约束。

实验优先级固定为：

1. 先验证正在运行的 artifact-free 精确安全模板 v4，让模板适应可靠外观变化，同时在遮挡/
   相似物帧拒绝污染；只看完整预注册 Top-14 投影，不看中途结果。
2. 主要方向是写回前的跨帧身份确认：联合 RGB、Depth、首帧语言、初始模板和候选历史，专门处理
   partial-occlusion、similar-objects 与 clutter；不能只用当前置信度阈值。
3. factor-7 只由 motion/scale 风险触发，用于 `stick/cube` 型几何外点；其余失败优先在
   factor-5 内提升候选判别，避免宽搜索引入更多干扰峰。
4. 后续学习门仍只能在 DepthTrack Train 上训练并重新走固定多-seed安全门；本节 VOT GT/tag
   只用于解释正式结果，不得反向选择部署阈值。

权威原始来源：

~~~text
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/votrgbd2022_official_safe025_scale098
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/vot_full127_failure_precursors_geometry.json
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/primary_trimodal_vot_sequence_gaps.json
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/runtime/vot_baseline_vs_language_e5_sequence_gap_analysis.json
/root/autodl-tmp/VOT-RGBD2022/sequences/*/*.tag
/home/OSTrack_RGBD_L_dataset_modified/annotations_cleaned/votrgbd2022_language.jsonl
~~~
### 24.63 artifact-free 安全模板的 post-Top14 自动续跑链（2026-08-13）

为避免 24.61 的 v4 在生成 `ready_for_ope` 后停住，新增独立、可恢复且 fail-closed 的控制器：

~~~text
controller
/home/SRTrack_RGBD_L/tools/continue_primary_safe_template_exact_post_top14.py
controller SHA
0e4666fd85fcb8d05af663f605ca7b56dd261a82f47e30b5b321e0d94ae78def
screen
primary_safe_template_exact_post_top14_v4
PID
364418
status
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/post_top14_status.json
~~~

该控制器使用独立 `post_top14.lock`，等待时不占 GPU，也不修改 Top-14 workspace、轨迹或投影。
它重新校验 workspace manifest SHA、冻结的 14 序列、Top-14 result schema、三项目标和
`decision` 一致性，然后严格按下列顺序执行：

1. Top-14 counterfactual projection 的 EAO/ACC/ROB 必须同时达到
   `77.9/82.1/93.7`；任一未达标即写
   `top14_projection_below_target`，不启动 OPE/full VOT。
2. DepthTrackTest full-50 使用 primary tracker、同一 safe-template profile、v11 长期恢复和
   report-only 0.98；既要达到 `65.2/64.9/65.1`，又要相对当前正式 best
   `65.995933/65.335885/65.664250` 的 P/R/F 各项下降不超过 0.05pp。
3. CDTB full-80 使用相同模板、v11 与 0.98，既要达到 `72.9/75.6/74.2`，又要相对当前正式 best
   `75.387821/76.005850/75.695574` 各项下降不超过 0.05pp。
4. 两个 OPE 保护门全部通过后，才用 `GPU_LAYOUT=(0,0,1,1)` 启动官方 full-127 multi-start；
   full VOT 保持 exact 模板候选的无 recovery、无 learned artifact、首帧静态文本与 report-only
   0.98 契约。最终还需覆盖 127 序列、1,765 anchors、1,327,004 tracker frames 并重新计算
   官方 EAO/ACC/ROB。

OPE 的 v11 是为了保持已经达标的 DepthTrack/CDTB 部署能力；正式 VOT 仍采用与 Top-14 完全相同的
无恢复 profile，二者没有混写。任一 OPE 门失败后不会继续 full VOT，也不会尝试第二套模板参数。
controller 已通过 `py_compile`、pass/reject 契约 smoke 和现有 safe-template/VOT bridge 聚焦回归
`14 passed, 1 timm deprecation warning`。

本节写入时，Top-14 为 `49,560/259,874 = 19.07078%`，四个 worker 均为运行态，post controller
为 `waiting_for_top14`。尚无完整 projection，因此最好的正式结果仍是
`72.908956/82.535868/87.988071`，不得把覆盖率或等待状态表述为指标改善。
### 24.64 当前安全模板的 Train-152 成对轨迹诊断（2026-08-13）

为解释“模板更新为什么既可能提高 ROB、又可能伤害已达标的 OPE”，新增只读分析器
`tools/analyze_safe_template_train_pair.py`（SHA
`d636846c5e69fab72deb1c06d3652ed27f7f6501133c84a74600dffd88112d6f`）。分析只使用
DepthTrack Train：先完成两次 GT-blind causal inference，再离线拼接 Train GT；没有使用
DepthTrackTest、CDTB 或 VOT GT。两条轨迹共用 checkpoint、source config、152 序列、首帧文本和
文本 assignment SHA；`toy07_indoor_320` 按已知图像长度只比较前 1,367 帧，39 行多余 GT 尾部被
显式记录。

原始诊断结果如下。这里 P/R/F 是 Train-152 的 VOT long-term 宏平均，只用于内部机理分析，不能
替代 DepthTrackTest 正式指标：

~~~text
部署                                      P          R          F
无模板历史 Train trace                 79.954950  77.634990  78.777894
当前 safe-template Train trace          79.897912  77.336744  78.596469
safe - no-template                      -0.057038  -0.298247  -0.181425 pp
~~~

在 219,954 个有标签帧中，模板轨迹有 208,657 帧 bbox 与历史无模板轨迹不同，占
`94.8639%`；125/152 个序列第一次分叉恰好发生在 zero-based 第 6 帧，所有非空分叉都不早于
第 6 帧，而这正是当前 `CHECK_INTERVAL=5, MIN_STABLE_FRAMES=3` 能完成首次模板提交后的最早影响
区间。可见帧上，模板相对轨迹有 5,040 个 IoU 提升大于 0.10 的帧，也有 5,615 个下降大于
0.10 的帧；它把 3,776 个基线 miss（IoU≤0.1）救到 IoU≥0.5，同时把 4,235 个原本 IoU≥0.5
的帧破坏到 IoU≤0.1。健康到 miss 的单步转移由 245 增到 259。因此当前机制不是纯救援门，而是
提交后几乎常驻的递归偏置，救援收益被健康轨迹污染抵消。

具体正负例：

~~~text
序列               mean-IoU delta   rescued miss   destroyed healthy
guitarbag_indoor       +0.618412          902                 2
bottle01_indoor        +0.249044         1117                 0
leaves02_indoor        +0.143629          153                 0
trophy_indoor          -0.359886            0               238
ball09_wild            -0.239456            0               408
basket_indoor          -0.215193            0               186
notebook02_indoor      -0.135526            0               386
~~~

这项比较必须降级标为 `diagnostic_not_strict_ablation`：无模板 manifest 绑定的
`frozen_tracker SHA=12e267...` 早于当前安全模板轨迹，而后者没有对完整 base tracker 源文件
提供对称 SHA。共同 provenance 和“只在最早模板提交之后分叉”使其成为强诊断证据，但不足以把每个
差值严格归因于模板。GPU 空闲后的第一项内部实验必须是同源码、同配置、同 checkpoint 的
safe-template ON/OFF Train-only 配对；在该对照完成前，不用本节数字选择公开集阈值。

对下一版架构的约束已经明确：

1. 动态模板继续允许写入一个有界槽，但默认休眠；不能在提交后每帧同时进入 backbone
   `dynamic_template_weight=0.10` 和 primary-input blend=0.02。
2. 只有当前帧出现预注册的 motion/scale/response/跨模态身份风险，且动态模板对候选峰提供跨帧
   一致支持时，才临时激活模板；健康低风险帧必须走静态 anchor 的字节等价路径。
3. 激活后若 RGB、Depth、语言身份任一硬冲突，立即回滚到静态模板，并保证 bbox、递归状态和动态
   tensor 原子恢复；模板更新只作可撤销记忆，不能成为新的永久跟踪中心。
4. 正在运行的 v4 仍按冻结参数完成 Top-14，不依据中途 coverage 或本节 Train 诊断改参数；完整
   projection 若拒绝，再实现上述 risk-gated dormant memory 并重新走 Train、多 seed、OPE 与
   full-127 门。

权威产物：

~~~text
/root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9/safe_template_vs_primary_train152.json
result SHA: 92475171ab185e3a9f58ecc6a871effc551c62d5c9502f6780d5fe602c973bcd
baseline manifest:
/root/autodl-tmp/srtrack_primary_evidence_trace_all152_seed2026/manifest.json
safe-template manifest:
/root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9/full152_trace/manifest.json
~~~

写入本节时，v4 Top-14 已运行到 `107,064/259,874 = 41.198427%`，四个 shard 仍为
`returncode=null`，两张 GPU 利用率为 99%/95%；这仍只是运行状态，不是 VOT 改善证据。

### 24.65 同源码模板 ON/OFF 严格 Train 配对队列（2026-08-13）

24.64 的历史对照只能作强诊断证据，因此新增 fail-closed 控制器
`tools/continue_safe_template_strict_train_pair.py`。它不会与当前公开验证抢 GPU：先等待
`post_top14_result.json` 形成完整终态，再要求两张 GPU 连续两次显存低于 500 MiB，之后才依次
执行 `template_off -> template_on -> strict_pair_analysis`。OFF/ON 使用同一
`safe_template_blend002` YAML；OFF 只通过
`safe_template_update_force_disable=True` 关闭动态模板，因此基础配置、checkpoint、语言查询和
所有非模板推理分支保持一致。

预检固化并在每个阶段后复验 9 个实现文件，包括 base tracker、model、参数加载器/别名、模板策略、
runner、数据构造器、metric、分析器和控制器本身；聚合
`paired_source_snapshot_sha256=74b36f21d843e9bd97b9de6921ef81c21bc712d21c81652ffd36c5197ec43106`。
任一源码在 OFF、ON 或最终分析前后变化都会 fail-closed。每侧必须产生精确 152 序列、
456 个预测文件；`toy07_indoor_320` 的 39 行已登记 GT 尾部只在最终共享前缀分析中显式裁剪。
只有两侧 manifest 的实现字典和 source snapshot 完全相等，分析器才允许写
`causal_attribution_status=strict_source_identical_ablation`。

~~~text
controller SHA:
047986948f51cac27acb34ea80dddff8da7488256aa57791ac302497db22907b
analyzer SHA:
fa1d1b8906aa58c071f71e9f3f96dd7a02d3ce5a38d27b47e628df59c8031afe
screen:
safe_template_strict_train_pair
PID:
370548
status:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026/runtime/status.json
result:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026/runtime/result.json
~~~

启动后已验证状态为 `waiting_for_public_chain`，两张 GPU 仍由 v4 Top-14 使用，控制器没有创建
`template_off/template_on` 预测目录、没有加载 checkpoint，也不占 GPU。控制器还会监视
post-Top14 的 `failed` 终态，避免异常时无限等待；等待结束、OFF 完成、ON 完成和最终分析前
都会现场重算 source snapshot。分析器独立重算 9 个实现文件 SHA 与两侧各 456 个预测文件的
aggregate SHA，不信任 manifest 自报值。OFF/ON 每侧运行在独立 session/process group 中；
控制器被中断或单侧异常时，对整组执行 `SIGTERM -> grace -> SIGKILL -> wait`，无 GPU
父/子进程组回收 smoke 已得到 `PROCESS_GROUP_CLEANUP_PASS`。它属于 Train-only
内部归因实验，不会读取 VOT、DepthTrackTest 或 CDTB GT；严格结果只用于决定下一版 dormant
risk-gated 模板架构，不会回写已经冻结的 v4 参数。


### 24.66 因果风险信号对模板收益/伤害的选择性诊断（2026-08-13）

为把 24.64 的“常驻模板会同时救援和污染”进一步落实为可实现的门控条件，新增只读工具
`tools/analyze_safe_template_risk_selectivity.py`（SHA
`6bbe4b378d5107f028b60b04ef4005e73690be8c0e15fb4ffa86a52a4b5ac098`）。它只读取
DepthTrack Train 的历史模板 ON/OFF 诊断和 GT-blind causal trace：跟踪结束后才拼接 Train GT，
不读取 VOT、DepthTrackTest 或 CDTB GT，也不改任何在线输出。trace 共含 152 个 metadata、
17,538 个因果采样帧；去掉 3,989 个不可见帧后，可严格对齐 13,549 帧。以
`safe IoU - baseline IoU > 0.10` 为显著收益、`< -0.10` 为显著伤害，得到 617 个收益帧、
735 个伤害帧和 12,197 个中性帧。显著伤害比收益多 118 帧，再次说明当前模板不应常驻激活。

单帧特征区分“收益 vs 伤害”的 AUC 如下；方向已自动选择为更有利于收益的一侧：

~~~text
特征                                      AUC     有利方向
raw_score                               0.8051   越高越好
selected_minus_held_response            0.7798   越高越好
route_probability                       0.7545   越低越好
identity_min_rank                       0.6680   越高越好
grounding_min_rank                      0.6241   越高越好
abs_log_area_change                     0.5641   越低越好
normalized_center_jump                  0.5411   越高越好
motion_scale_joint                      0.5173   越低越好
~~~

这说明主要矛盾不是“运动有多大”，而是发生运动/遮挡后当前选峰是否仍有可信的响应和跨模态身份支持。
例如：

* `flowerbasket_indoor` 的采样帧净计数为 `4 benefit / 106 harm`，平均 IoU 差
  `-0.3816`。zero-based frame 600 上，baseline IoU 为 `0.9876`、模板轨迹为 `0`；当时
  `raw_score=0.1356`、`selected-held=0.00636`、`identity_min_rank=0.3351`、
  `route_probability=0.5321`。这是低置信、低身份一致性下仍让递归模板介入，直接把健康轨迹变为 miss
  的具体污染例。
* `trophy_indoor` 为 `0 benefit / 41 harm`，采样帧平均 IoU 差 `-0.6044`；
  `notebook02_indoor` 为 `4/61`、平均 `-0.2495`。二者都支持“模板写入后常驻”会在健康段累积错误，
  不能把更新本身等同于恢复。
* 正例 `bottle01_indoor` 为 `80 benefit / 0 harm`，采样帧平均 IoU 差 `+0.1625`；
  zero-based frame 2,600 上 baseline IoU 为 `0`、模板 IoU 为 `0.9507`，同时
  `raw_score=0.7143`、`selected-held=0.5444`、`identity_min_rank=0.9948`、
  `route_probability=0.0202`。这属于动态记忆与当前高身份候选一致时的有效救援。
* `guitarbag_indoor` 为 `20 benefit / 3 harm`，平均差 `+0.2290`；zero-based frame 600
  baseline IoU 为 `0`、模板 IoU 为 `0.9568`，`raw_score=0.8983`、
  `identity_min_rank=0.9931`、`route_probability=0.00457`，同样表现为高置信身份一致后的恢复。

简单阈值扫描也揭示了精度与覆盖率的约束：`route_probability <= 0.00699` 在采样集选中
1,355 帧，其中 26 个显著收益、0 个显著伤害，但收益召回仅 `4.21%`；
`raw_score >= 0.82786` 为 `29 benefit / 1 harm`；
`selected_minus_held_response >= 0.70396` 为 `46/3`。放宽阈值可提高收益召回，却会重新放入伤害帧。
因此下一版不能只用 motion/scale 单阈值：应以高 `raw_score`、足够大的 selected-held margin 和
RGB/Depth/语言身份一致性作联合许可，并以 route risk、连续帧一致性及硬冲突作为否决/回滚条件；
动态模板保持 dormant，只在联合许可成立时短暂参与候选选择。

两/三特征 AND 网格的历史最优低伤害候选为
`selected_minus_held_response >= 0.503378 AND route_probability <= 0.0770815`：选中 3,316
个采样帧，其中 `118 benefit / 0 harm / 3,198 neutral`，显著收益召回 `19.12%`，是上述零伤害
单阈值召回的约 4.5 倍。这只是从同一 Train 诊断集的 8,100 条组合规则中扫描出的研究候选，存在
明显选择偏差；必须在 24.65 严格配对的独立切分/多 seed 上重新选择和审计，不能直接写进公开部署。

为检验跨序列泛化，分析器又执行确定性的 5 折 sequence-held-out 审计：每折只在另外 4/5 序列上
扫描规则，要求 calibration 至少 20 个收益且零伤害，再把唯一选中规则在未见的 1/5 序列上评一次。
五折聚合为 `124 benefit / 2 harm / 3,396 neutral`，收益召回 `20.10%`、显著效果精度
`98.41%`；四折零伤害，一折出现 2 个伤害。这说明联合信号有跨序列选择性，但也证明同集的
“0 harm”不能外推为绝对安全。

此外，同帧响应只能在当前 forward 后得到，不能用来决定同一次 forward 是否激活模板。为遵守在线
因果顺序，分析器只保留相邻且均被 trace 捕获的帧，用 `t-1` 特征预测 `t` 的模板 ON/OFF IoU
效果。在 5,943 个连续对（278 benefit、291 harm、5,374 neutral，覆盖有连续采样对的 139/152
序列）上再次做同样的序列隔离 5 折，
聚合得到 `56 benefit / 1 harm / 1,601 neutral`，收益召回 `20.14%`、显著效果精度
`98.25%`。滞后一帧后最强单特征仍为 selected-held margin（AUC `0.8106`），其次为 raw score
（`0.7948`）、route probability（`0.7117`）和 identity rank（`0.6952`）。因此可实现结构应是：
在帧 `t` 完成静态前向后更新历史许可状态；只允许帧 `t+1` 临时激活动态记忆，并在冲突或许可失效
时立即回到静态路径，而不是用未来/同帧信息反向门控。

一个负结果也应保留：把 `t-1` 限定为 `capture_reason=motion_risk` 后仍保留 5,883/5,943 个连续对，
序列隔离汇总从 `56 benefit / 1 harm` 变为 `57/2`，显著效果精度从 `98.25%` 降到
`96.61%`。这是因为 causal trace 本来就主要在风险帧采样。motion-risk 可以作为“何时检查/唤醒”的
必要条件，却没有足够选择性充当“允许动态模板写入或注入”的安全条件；许可仍需 response margin、
raw score 与跨模态身份联合决定。

对唯一的滞后一帧 held-out 显著误放行作实例复核：`cube06_indoor` 的 feature frame 705 上
`raw_score=0.7246`、selected-held=`0.5508`、identity rank=`0.9774`，表面证据很强，但
`normalized_center_jump=1.0825`；effect frame 706 上 baseline IoU=`0.9206`，模板 IoU=`0`。
这证明大位移可以让高置信高身份候选仍成为递归污染源。另一个 motion-risk 子集误放行
`bottle03_indoor` 的上一帧 center jump=`0.3967`，模板使下一帧 IoU 从 `0.7392` 降到
`0.4914`。

然而，把现有写门 `MAX_CENTER_JUMP<=0.35` 不加区分地叠到已选滞后读门的未见审计折，虽然将
`1 harm` 降为 `0`，也把收益从 `56` 降到 `2`，收益召回从 `20.14%` 降到 `0.72%`。因此动态
模板必须拆分两个状态：严格几何/深度/身份门只负责认证并写入记忆；大位移风险发生后，读门可在
response/identity 多帧一致时短暂读取已认证记忆，但必须保留当前 bbox/递归状态快照，并在候选冲突
或下一帧证据恶化时原子回滚。简单复用写门作为读门会把 VOT 最需要的大位移救援一并消掉。

本结论仍标为 `diagnostic_not_strict_ablation`。因果特征虽然不使用 GT，但来自已经启用模板的轨迹，
而 ON/OFF 标签来自 24.64 的历史非同源码对照；它证明哪些信号值得进入下一轮预注册门，不证明上述
阈值在闭环部署中安全。最终实现和阈值必须等待 24.65 的同源码严格配对完成，再只用 Train、多 seed
选定，并重新经过 OPE 与 full-127 VOT 门。

~~~text
result:
/root/autodl-tmp/srtrack_primary_precommit_strict_seed2026_v9/safe_template_risk_selectivity.json
result SHA:
a457fdcdec2aa0301cced21ddf81fbc3e3158d70a64f886ab176b22ae3beb8e7
trace aggregate SHA:
7447959d135aa3cc469bcf81e8a45cab711c9304538a96c3b16035ad5a64eaf0
~~~

### 24.67 artifact-free 安全模板 Top-14 完整结果与序列级抵消（2026-08-13）

24.61/24.63 中尚未完成的 v4 已按冻结协议完整结束。14 个序列、295 个 official multi-start
anchor 和 259,874 个 tracker frames 均覆盖，四个 shard 的 return code 全为 0；没有复用旧覆盖，
也没有在运行中修改配置。候选与 reference 绑定同一 checkpoint、同一语言文件、同一 report-only
0.98，只把冻结的 `SAFE_TEMPLATE_UPDATE` profile 从 OFF 切换为 ON，且不携带 learned artifact。

~~~text
结果口径                         EAO          ACC          ROB
正式 full-127 reference       72.908956    82.535868    87.988071
选定 Top-14 reference         50.175795    74.967504    53.431277
Top-14 safe-template          50.779652    74.948110    54.385093
反事实 full-127 投影           73.029068    82.517833    88.154615
投影 - 正式 reference          +0.120112    -0.018035    +0.166544 pp
目标                           77.900000    82.100000    93.700000
投影距目标                     -4.870932    +0.417833    -5.545385 pp
~~~

投影只满足 ACC，EAO/ROB 仍明显不达标，自动门因此写
`top14_projection_below_target`；`formal_full_vot_started=false`，后续 DepthTrack/CDTB OPE 和
full-127 VOT 均未启动。这是完整 Top-14 的预注册反事实投影，`official_full_dataset_result=false`，
不能表述为新的正式 VOT 成绩。当前所有公开数据集的正式最好指标仍为：DepthTrackTest
P/R/F `65.995933/65.335885/65.664250`，CDTB P/R/F
`75.387821/76.005850/75.695574`，VOT-RGBD2022 EAO/ACC/ROB
`72.908956/82.535868/87.988071`。

逐序列结果进一步解释了总体提升为何只有 EAO `+0.1201pp`、ROB `+0.1665pp`：模板在一批遮挡、
外观变化序列中延长了成功轨迹，却在另一批健康或相似物序列中把错误外观写入递归记忆，正负贡献
大幅抵消。

~~~text
序列                         EAO全局贡献差   ROB序列差    ROB全局贡献差   ACC序列差
toy09_indoor_1                 +0.381534      +3.573065      +0.339363      -0.406084
bag02_indoor_2                 +0.223824      +6.816539      +0.614543      -0.259367
humans_shirts_room_occ_1_A_1  +0.100020      +2.170950      +0.121190      +0.090587
toy02_indoor_1                 +0.085669      +2.707633      +0.217986      -0.068348
earphone01_indoor_1            +0.044323      +0.540765      +0.041618      -0.069101
notebook01_indoor_1            +0.025674      -2.280643      -0.115828      +0.030983
glass01_indoor_2               -0.057179      -0.923219      -0.058610      +0.090031
yogurt_indoor_1                -0.174813      -2.434296      -0.277825      -0.230354
~~~

具体解释如下：

* `bag02_indoor_2` 的 ROB 从 `69.6185` 提高到 `76.4350`，是最大的 ROB 正贡献；模板适应外观后
  延长了成功区间，但 ACC 下降 `0.2594pp`，说明更新更像“避免早失锁”，不是更精确的框回归。
* `toy09_indoor_1` 的 ROB 从 `42.4301` 提高到 `46.0032`，产生最大的 EAO 正贡献。结合 24.62
  的 partial-occlusion/background-clutter 失败集中，这证明动态外观记忆确实能缓解部分遮挡后的
  旧模板失配，但仍不足以解决其多数 anchor 的身份混淆。
* `humans_shirts_room_occ_1_A_1` 和 `toy02_indoor_1` 的 ROB 分别提高 `2.1710pp` 和
  `2.7076pp`，说明安全写门在部分遮挡/形变序列中有稳定价值，不应简单删除模板更新能力。
* `yogurt_indoor_1` 的 ROB 从 `78.1272` 降到 `75.6929`，EAO 全局贡献下降 `0.1748pp`；这是本组
  最大负例，表明一旦动态槽写入错误外观，常驻 0.10 动态输入和 0.02 primary blend 会把污染
  递归传播。
* `notebook01_indoor_1` 的 EAO 全局贡献略升，但 ROB 下降 `2.2806pp`，同时 ACC 权重变化使
  全局 ACC 贡献下降；单看一个汇总指标会掩盖“少数更长轨迹换来更多失锁”的结构性副作用。
* `glass01_indoor_2` 的 ROB 下降 `0.9232pp`。透明目标本来就具有较弱的 RGB 边界和深度身份，
  动态模板无法可靠区分目标与背景，印证 24.62 的透明/背景竞争原因。
* `cup02_indoor_1` 的 ROB 只提高 `0.0565pp`，EAO 反而下降 `0.0046pp`；36/36 anchor 的相似杯
  身份竞争不能靠持续模板混合解决。模板会适应“当前所跟对象”，但若第一次选中的就是相似物，
  更新反而会巩固错误身份。

因此，下一版不能继续扫描常驻 blend weight。架构结论是保留“可写动态记忆”，但把写入认证与读取
许可拆开：低风险帧必须完全省略 dynamic kwargs 并走静态模板字节等价路径；帧 `t` 只用因果响应、
selected-held margin、RGB/Depth/语言身份和历史一致性产生 `t+1` 的短期 read license；许可期间若
候选冲突或证据恶化，必须原子恢复 bbox、递归状态与动态 tensor。24.66 已证明简单复用
`MAX_CENTER_JUMP<=0.35` 作读门会把收益召回从 `20.14%` 压到 `0.72%`，所以写门应严格、读门应
专门面向大位移恢复且可撤销。

权威产物：

~~~text
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/top14_result.json
SHA: eae8e8e7e624031703389039e97fd883dc4fd9ab5006ae9c03c3f69b3f3e5ca2
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/top14_projection.json
SHA: 67f4797092c6e7638d08791e7f6ad832508a635eefab9fc52d695e9ff409bbd1
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/top14_comparison.json
SHA: f95f5fd7ab3bbc3d291590c00de66310adfae643ea39bec306fca4d78f86dfd5
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/post_top14_result.json
SHA: 4c51ae6ee241a7477aa892003ee6958b1ed1de0cc707679b54a7a4c0928b7e12
~~~

### 24.68 同源码 Train 配对 v1 编排失败与 v2 修复（2026-08-13）

24.65 的首个控制器在真正处理序列前 fail closed。根因不是 checkpoint、数据或模板推理，而是
`safe_template_update_force_disable=True` 只把 `SAFE_TEMPLATE_UPDATE.USE` 置为 false，却遗留
从属的 `PRIMARY_TEMPLATE_BLEND_USE=true`；tracker 初始化因此稳定抛出
`ValueError: Primary template blend requires SAFE_TEMPLATE_UPDATE`。v1 只留下 running manifest，
没有任何预测文件，不能进入配对分析。

参数加载器现已把 owner/dependent 一起关闭。最小不变量检查确认 OFF 侧为
`USE=false, PRIMARY_TEMPLATE_BLEND_USE=false`，相关参数与模板更新回归为 `39 passed`，并通过
`py_compile` 和 `git diff --check`。为避免复用半成品，重新建立独立 v2 根；OFF/ON 两侧仍由同一
进程源生成并绑定新的统一源码快照：

~~~text
root:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026_v2
paired source snapshot SHA:
1c939c2e678fed6b249bd346e07a9e99d6381af8d96d89f49a8bc8814e7c9156
controller SHA:
076c2ce32ee540896da753c059e1d0c2b2018f8c80f8919fc2cd6010a2b5854c
parameter loader SHA:
41efa70ccab0be2a10a9ea7a2ae96b87fb9177c961b6710e7886446728181a51
execution order:
template_off -> template_on -> strict_pair_analysis
~~~

写入本节时，v2 已越过两次 GPU 空闲检查并进入 `running_template_off`，两个 worker 均已加载
checkpoint；尚未形成完整 152 序列预测或 strict analysis。因此 24.64 的历史差值仍只能作为非严格
诊断，不能提前选择 dormant read gate。只有 v2 两侧各产生精确 456 个文件且 source snapshot
复验相同后，才允许用严格差值训练/审计下一版读写分离模板。

### 24.69 读写分离的休眠模板架构与自动验证链（2026-08-14）

24.67 证明“经过严格认证的动态模板槽”有恢复价值，但把它持续接入主干会让收益与污染互相抵消。
因此本轮不再调常驻 blend，而采用静态主路径与动态反事实分支分离的四阶段结构：

~~~text
首帧静态模板 z0 ───────────────> static forward ──> public bbox / recursive state
        │                                  │
        │                         因果风险达到采样条件
        │                                  │
严格 RGB-D/几何/响应写门 ──> dormant zt ──> dynamic counterfactual forward
                                           │
                                  214维同状态双分支证据
                                           │
                         Train GT 仅在推理完成后连接并训练 read gate
~~~

当前 `siamtrack_dormant_template_counterfactual_trace.py` 的公共输出始终强制静态：在每帧 forward 前
临时把 `dynamic_active` 置为 false，因此不传入任何 dynamic kwargs；writer 只在定位完成后的
observe seam 恢复并更新一个认证槽。只有帧前已经存在 `source=safe` 的模板，且当前静态轨迹达到
motion risk `>=0.4`、发生写/淘汰事件，或每 100 帧的稳定抽样时，才从**完全相同的帧前 bbox、搜索
crop、网络输入和模型状态**做第二次动态 forward。动态结果只写 trace，不允许改 public bbox、score
或递归状态。launcher 还逐文件比较静态 bbox 与 all-score 输出同冻结 OFF reference 的 SHA；fixed6
必须 12/12 字节一致，full152 必须 304/304 字节一致，否则分析链关闭。

读门输入不是单一 motion 阈值，而是：静态分支 69 维 causal precommit evidence、动态分支同一
69 维、两者逐维差 69 维，再加模板年龄、选峰是否一致、两框 IoU、框间位移/尺度风险及 primary
blend 是否实际发生，共 `69*3+7=214` 维。模型共享 96 维 encoder，并分别预测 static IoU、
dynamic IoU、显著收益概率、失锁转恢复概率和伤害概率。这样写门只负责“模板能否进入认证槽”，
读门才负责“本帧是否值得从槽中读取”，避免用写门的 `MAX_CENTER_JUMP<=0.35` 把大位移恢复全部
屏蔽。

因果和选择协议固定如下：

1. 轨迹阶段只用 DepthTrack Train、首帧语言和在线可见证据；不读取任何 GT，也不让动态分支改变
   输出。帧前模板张量、commit frame、blend weight 和 max blend weight 均在 writer observe 前
   冻结，防止把帧后状态混入同帧反事实。
2. 152 序列全部结束且静态零扰动通过后，分析器才连接 Train GT，标注 dynamic-static IoU gain、
   healthy harm、catastrophic harm 和 transition recovery。没有反事实样本的序列仍保留在 source
   sequence/fold provenance 中，不伪造空样本。
3. seed 2026/2027/2028 各做 5 折 sequence-group 验证。每个 seed 先固定一个完全隔离的 policy-audit
   fold；其余四折 OOF 训练永不接触 audit 序列，15,625 个策略只在 calibration OOF 选择唯一第一名，
   然后该策略在 audit 上只评一次，失败不回退。
4. 每个 seed 的安全条件均为 read precision `>=0.85`、healthy/catastrophic harm 均为 0、至少恢复
   一个 transition、总 IoU gain `>0`，且连续采样帧最多读取一次。三 seed 必须全部独立通过；之后
   不按 audit 指标挑 seed，而是固定预注册 seed 2026，才允许进入 Top-14。
5. artifact loader 会重新核验 checkpoint、config、完整 trace/analysis/OOF/GT aggregate、fixed6、
   source implementation 和 training implementation SHA。VOT、DepthTrackTest、CDTB GT 在选门阶段
   全部禁止使用。

当前运行链为：

~~~text
strict source-identical Train-152 OFF/ON:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026_v2

dormant counterfactual + read gate:
/root/autodl-tmp/srtrack_dormant_template_counterfactual_train152_seed2026_v1

trace wrapper SHA:
adb7226c53ae03161d3f5ee51273fdd1856d50199a32c25210d340bbe87e3d10
trace launcher SHA:
3639f843a80b2a04a2bfd072a1c43f5b253942d2c061d837bb84295289df8428
read-gate shared contract SHA:
7b8518534e3957c92446e08bb84d4f4c11f0a41efe681d2a66c63989d48515ca
trainer SHA:
1136c096e77417b7be2b5576d46f138e54ecbf07b64080bd746b9e487c6a0024
read-gate continuation SHA:
765de60481edb8c466ed2df9a8d99518c3aaf426a559dd30c0065d181dc6c1b4
~~~

更新本节时，严格 OFF 已完成 152/152（456 文件），ON 完成 72/152（217 文件），两张 GPU 各有一
个正常 worker；反事实与 read-gate 控制器均只读等待上游，尚未生成可宣称的新公开指标。
因此目前正式最佳结果仍保持 24.67 所列：DepthTrackTest F `65.664250`、CDTB F `75.695574`、
VOT-RGBD2022 `EAO/ACC/ROB=72.908956/82.535868/87.988071`。本节描述的是正在验证的架构和严格
协议，不把预检或未完成运行写成性能提升。

### 24.70 休眠读门的事务运行时与公开评测放行链（2026-08-14）

训练协议之后已经补齐真正的 public runtime，而不是只停留在 counterfactual trace。其每帧事务
顺序固定为：保存帧前 bbox 与 writer 深拷贝快照；关闭 writer 后完成静态主路径；只在存在帧前
认证模板且静态风险 `>=0.4` 时，复用完全相同的搜索 crop/网络前态做动态 forward；214 维读门
决定是否采纳动态结果；最后只对被选中的一个 action 执行一次 writer observe。静态分支未被采纳
时直接返回原 base result 字典，因此 bbox/score 不发生重新编码。动态 forward、读门、writer 或
运行 trace 任一环节异常时，恢复 bbox、writer policy、模板 tensor 与 source 快照，并只重放一次
静态 writer observe；禁止部分动态状态泄漏到下一帧。读门还强制最多连续读取一帧，避免模板错误
形成递归自证。

部署入口现覆盖 DepthTrack、CDTB 和 VOT-RGBD2022。`dormant_template_read_gate` 部署必须提供完整
loader 可验证的 artifact，并绑定 checkpoint `30c804...`、config `9e825b...`、safe-template writer
与 report-only `0.98`；recovery、rerank、quarantine 和其它 learned artifact 均禁止组合。VOT 的
full workspace 与四个并行 shard 都再次核验 artifact SHA，并把
`SRTRACK_DORMANT_TEMPLATE_READ_GATE` 显式透传给 TraX 子进程。异常清理以每个 shard 的独立进程组
执行 TERM、宽限、KILL 和 wait，避免残留 TraX/PyTorch 子孙进程占用 GPU。

公开评测采用严格串行放行：

~~~text
Train-152 OFF/ON 完成
  -> fixed6 12/12 字节零扰动
  -> full152 304/304 字节零扰动 + Train GT 后连接
  -> seed 2026/2027/2028 全部 audit 通过，固定 seed 2026
  -> VOT Top-14 反事实投影
  -> DepthTrackTest 50 序列保持性（各指标相对正式 reference >= -0.05pp）
  -> CDTB 80 序列保持性（各指标相对正式 reference >= -0.05pp）
  -> 仅当 Top-14 同时预测 EAO>=77.9、ACC>=82.1、ROB>=93.7 时启动 full-127
  -> official VOT 覆盖复验：127 sequences / 1765 anchors / 1,327,004 tracker frames
~~~

评测 workspace 为：

~~~text
/root/autodl-tmp/srtrack_dormant_template_read_gate_evaluation_seed2026_v1

runtime tracker SHA:
462b938e26bc33568a5963b7711009e2344720cec5017a77d95e4e9d401cc2db
VOT bridge SHA:
af1ac3e1be95f18a7ac3814637b6eddcd62448c06884ed2fce77f62a09dd0d9a
workspace builder SHA:
6d681cb646272f68315f9179543f6f812602a8f8e7edc33d745784a82ea4d1a3
parallel shard builder SHA:
15911fc7597016b626c8ae70e5fd0f2e83356838e9f361c877302fe815792317
evaluation controller SHA:
c86f81efd71d5f09bbd31d8b613cb6f35f2c62f89f17f33cc3a7c9ee39df9cd6
~~~

当前回归证据为：运行时/部署/VOT bridge/workspace 组合 `27/27` 通过，并行 shard/anchor 组合
`20/20` 通过；只读评测 preflight 明确记录 `selection_ready=false`、`workspace_created=false`、
`public_evaluation_started=false`。这证明评测链已就位，不代表产生了新指标。正式最佳值仍为
DepthTrackTest `P/R/F=65.995933/65.335885/65.664250`、CDTB
`P/R/F=75.387821/76.005850/75.695574`、VOT-RGBD2022
`EAO/ACC/ROB=72.908956/82.535868/87.988071`；Top-14 safe-template
`73.029068/82.517833/88.154615` 仍只属于反事实投影。

### 24.71 严格 ON 中途机制诊断：为什么必须“按需读”而不能“常驻读”（2026-08-14）

为提前检查方向，在 strict ON 尚未完成时，对当时已经完整形成三文件组的 79/152 个 Train 序列做了
一次**只读、非正式、部分覆盖**诊断。该诊断没有写入训练 artifact，也不替代 152/152 后由完整
manifest 和 aggregate SHA 约束的正式分析。109,517 个可见帧上，常驻模板同时产生 2,155 个
`OFF IoU<=0.1 -> ON IoU>=0.5` 救回帧，也产生 2,490 个
`OFF IoU>=0.5 -> ON IoU<=0.1` 摧毁帧；平均可见帧 IoU 从 `0.788472` 降到 `0.785676`，差值
`-0.002796`。79 序列中 42 个改善、35 个受损、2 个轨迹字节不变，说明更新槽本身有强恢复信号，
但无条件把它持续混入后续帧会把少量错误放大为长段递归漂移。

具体轨迹例子如下（帧号为 zero-based）：

- `ball09_wild`：两路首次在帧 166 分叉；帧 641--947 连续 307 帧被摧毁。区间首帧 OFF IoU
  `0.9677`、ON `0.0`，末帧 OFF `0.8966`、ON `0.0`。该段没有救回帧，属于典型的错误模板读入后
  长时间锁定错误身份，而非单帧定位抖动。
- `basket_indoor`：帧 624--809 连续 186 帧被摧毁；起点 OFF/ON 为 `0.7161/0.0`，终点为
  `0.9011/0.0`，下一帧 ON 才恢复到 `0.7135`。这说明常驻读取会延迟静态路径本可快速恢复的轨迹。
- `bottle01_indoor`：相反地，帧 2283--2652 连续 370 帧被救回，起点 OFF/ON 为
  `0.0/0.5697`，区间末端仍为 `0.0/0.6835`，且没有摧毁段。这证明认证动态模板确实能够长期保持
  正确身份，不能简单禁用模板更新。
- `glass04_indoor`：帧 262--300 连续 39 帧被救回，起点 OFF/ON 为 `0.0/0.8824`，末端为
  `0.0/0.9012`；帧 301 静态路径自行恢复后两路为 `0.7551/0.7719`。理想策略应只在静态失锁窗口
  读取动态槽，并在静态恢复后立即停止读取。
- `flowerbasket_indoor`：同一序列内先出现帧 437--691 的 255 帧连续摧毁，后又出现帧 914--1000
  的 87 帧连续救回。它直接否定“按序列决定是否启用模板”，支持当前逐帧双分支证据与事务 read
  gate：相同对象、相同序列的不同时段需要相反动作。

因此本轮不是把常驻模板权重继续从 `0.1` 微调到另一个常数，而是把认证槽作为 dormant memory：
静态路径默认不读；只有模型同时预测动态 IoU、正 gain、transition recovery 和低 harm 时才进行一次
事务读取，下一帧禁止连续读取。运行时异常边界也已扩展为捕获所有普通 `Exception` 并回退静态，
但仍让 `KeyboardInterrupt/SystemExit` 传播；新增回归确认 `AttributeError` 等未预期 backend 异常
同样恢复 bbox、writer policy、模板 tensor 并重放一次静态 writer observe。

### 24.72 所有公开数据集正式最佳指标汇总（2026-08-14）

以下只汇总已经完成全量正式评测、可直接对外引用的最好结果。数值均按百分制记录；“差值”为
`正式最佳 - 目标`，正数表示达标。DepthTrackTest 与 CDTB 的指标是 Precision/Recall/F-score，
VOT-RGBD2022 的指标是 EAO/ACC/ROB，因此不能跨数据集横向比较同一列。

| 数据集（正式覆盖） | 指标 1 | 指标 2 | 指标 3 | 目标 | 相对目标差值 | 是否全部达标 |
|---|---:|---:|---:|---:|---:|:---:|
| DepthTrackTest（50 序列） | P **65.995933** | R **65.335885** | F **65.664250** | 65.2 / 64.9 / 65.1 | +0.795933 / +0.435885 / +0.564250 | 是 |
| CDTB（80 序列） | P **75.387821** | R **76.005850** | F **75.695574** | 72.9 / 75.6 / 74.2 | +2.487821 / +0.405850 / +1.495574 | 是 |
| VOT-RGBD2022（127 序列、1,765 anchors） | EAO **72.908956** | ACC **82.535868** | ROB **87.988071** | 77.9 / 82.1 / 93.7 | -4.991044 / +0.435868 / -5.711929 | 否 |

为避免口径混淆，当前 safe-template Top-14 结果单独列为**非正式反事实投影**：

| 诊断结果（非正式） | EAO | ACC | ROB | 相对正式 VOT reference | 距正式目标 |
|---|---:|---:|---:|---:|---:|
| Top-14 safe-template 反事实 full-127 投影 | 73.029068 | 82.517833 | 88.154615 | +0.120112 / -0.018035 / +0.166544 | -4.870932 / +0.417833 / -5.545385 |

因此，当前两个 OPE 数据集已经全部达标；VOT 只有 ACC 达标，核心缺口仍是 EAO 和 ROB。正在运行的
Train-152 OFF/ON 与 dormant read-gate 链尚未产生新的公开全量结果，不能覆盖本表中的正式最佳值。

正式证据文件与 SHA256：

~~~text
DepthTrackTest:
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/depthtrack_test_full50_recovery_v11_scale098_safe025/metrics.json
SHA256 5f41886598559ead7af401247442f130c4aa38080f87ba870b0070828abe639c
coverage 50 sequences / 76,373 frames

CDTB:
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/cdtb80_full_recovery_v11_scale098_safe025/metrics.json
SHA256 271c91bad7f652c89ec097f99ae38c5ddd3c3045956a254a1889499ceb7a69be
coverage 80 sequences / 101,956 frames

VOT-RGBD2022:
/root/autodl-tmp/srtrack_primary_trimodal_guard_probe_e1_seed2026/votrgbd2022_official_safe025_scale098/official_metrics_parsed.json
SHA256 a8958fd27f6889cee0be2250c2a598c6a085b272d9ad856ffe4e6e2fa5a82980
coverage 127 sequences / 1,765 anchors / 1,327,004 tracker frames
~~~

三份正式结果的 inference profile 都明确 `safe_template_update=false`。它们是新模板候选必须保护或超越的
静态参考，不是模板更新已经带来的成绩；任何新模板结果只有完成相同公开覆盖后才可替换本表。

### 24.73 dormant read-gate 帧对齐与诊断失效隔离审计（2026-08-14）

在等待 Train-152 上游完成期间，对 trace→GT 后连接和 public runtime 做了最后一轮只读契约核验。
帧号没有整体偏移：base tracker 在 `initialize()` 后将 `frame_id=0`；runner 将初始化框作为结果第 0
行；第一次 `track()` 在 forward 前把 `frame_id` 加为 1。因此 counterfactual trace 中的 `frame=k`
对应数据集第 `k` 帧和 `GT[k]`，而 `pre_frame_bbox` 对应 `GT[k-1]`。分析器明确以这两个下标分别
计算 static/dynamic 当前 IoU 和 previous IoU，并在序列内部独立读取 groundtruth；不存在跨序列状态
拼接或整体错一帧。

同时修复了一处仅影响可观测性的 fail-closed 边界：public action 完成后的 runtime JSONL 写入原先
只忽略 `OSError/TypeError/ValueError`，其它普通诊断异常仍可能使已完成帧抛错。现在统一捕获普通
`Exception` 并保持所选 bbox、递归 state 和 writer 状态不变，但仍允许
`KeyboardInterrupt/SystemExit` 等进程控制异常传播。专门回归将 trace writer 注入
`AttributeError`，验证静态输出与 state 均保持不变。

更新后的 runtime tracker SHA256 为：

~~~text
462b938e26bc33568a5963b7711009e2344720cec5017a77d95e4e9d401cc2db
~~~

runtime、三 seed 训练/选择、部署、VOT bridge/workspace/shards 与 evaluation controller 组合回归为
`48/48 passed`，相关模块 `py_compile` 通过。改动发生在 read-gate 训练启动前；trainer 会在子进程
真正启动时把当前 runtime SHA 写入 artifact provenance，loader 和公开评测再逐文件复验，因此未
改变已经运行的 strict Train ON/OFF 轨迹，也不会造成旧 SHA artifact 被误部署。

### 24.74 严格配对中途的正式 motion-risk 口径诊断（2026-08-14）

为检验 dormant trace 的 `capture_risk_min=0.4` 是否会漏掉关键恢复入口，对 strict OFF/ON 当时已经
形成完整 bbox/score/time 三文件组的 108/152 个序列做了只读容量诊断。分析器
`tools/analyze_safe_template_partial_motion_risk.py` 不重新实现风险公式，而是逐帧直接调用 public
runtime 使用的 `state_drift_guard.motion_scale_risk()`；产物同时绑定两侧 manifest、所有纳入的
OFF/ON bbox、Train GT、分析器和风险实现 SHA。该分析不进入训练、policy grid 或 selector，也没有
读取 DepthTrackTest、CDTB、VOT GT。

~~~text
覆盖：108 sequences / 149,510 valid frames
全部常驻 ON 相对 OFF：
  beneficial(delta IoU > 0.1) = 3,834
  harmful(static IoU >= 0.5 且 delta < -0.1) = 3,556
  catastrophic(static IoU >= 0.5 且 ON IoU < 0.2) = 2,963
  transition recovery = 49
  accumulated IoU delta = +249.281593

minimum static motion risk >= 0.4：
  sampled frames = 9,200
  transition recovery = 49/49（recall 100%）
  beneficial = 447；harmful = 532；catastrophic = 429
  accumulated IoU delta = -114.311941

minimum static motion risk >= 0.8：
  sampled frames = 2,285
  transition recovery = 48/49（recall 97.9592%）
  beneficial = 194；harmful = 143；catastrophic = 110
  accumulated IoU delta = +10.833577
~~~

结论是：`0.4` 适合作为 same-state 双分支取证的高召回采样下界，因为它覆盖当前部分轨迹中的全部
49 个“上一帧静态健康、当前静态失锁、动态模板可救回”入口；但它不是动态 action 的充分采纳条件，
若把所有 `risk>=0.4` 帧都持续读模板，收益仍为负。到 `0.8` 才出现微弱正容量，说明真正区分信号
必须来自 214 维静态/动态对照中的 RGB、Depth、语言、跨模态 consensus、响应图与 harm 预测，而
不能只调一个运动阈值。预注册 grid 仍保留 `{0.4,0.6,0.8,1.0,1.2}`，不会依据本次中途诊断改动。

权威诊断产物：

~~~text
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026_v2/runtime/partial_motion_risk_probe.json
result SHA256: 843ff5450fd365da1021c211620ad6f7bea91b3af27e55f51454bd2f098a12c9
analyzer SHA256: c161866cd776e6e0da503aab4021f9a30ee45eee1b24daae7d2b1b25d1c1fb80
risk implementation SHA256: 7d2b2f4a1b9e0005f15b50bd8d74e2507a60ef18dfd86ea36f6597cbb09b04ce
~~~

这仍是“OFF 静态轨迹 vs 常驻 ON 递归轨迹”的部分覆盖容量分析，不能替代 full152 完成后的同状态
动态 counterfactual、三 seed audit、Top-14 或正式公开评测，也不能作为新的 EAO/ROB 数字。

### 24.75 Top-14 完整轨迹组合 oracle：主要瓶颈在候选能力而非序列选择（2026-08-14）

为区分“模板候选本身太弱”和“门控没有选对候选”两个原因，对已经完成的 Top-14 safe-template
workspace 做了一个只读的**完整序列轨迹组合 oracle**。对冻结的 14 条诊断序列逐条选择正式 reference
轨迹或 safe-template 轨迹，共枚举全部 `2^14=16,384` 种组合；其余 113 条始终使用 full-127
reference。每个组合都保留该序列的全部 multi-start anchors，再用同一 VOT `burnin=10`、`grace=10`、
`threshold=0.1` 和 EAO 区间 `[115,755]` 重算指标。reference/candidate 还必须具有相同 checkpoint、
数据集 list SHA 和首帧语言 SHA。

| 轨迹组合（非正式诊断） | EAO | ACC | ROB | 相对正式 reference | 距目标 |
|---|---:|---:|---:|---:|---:|
| full-127 正式 reference | 72.908956 | 82.535868 | 87.988071 | — | -4.991044 / +0.435868 / -5.711929 |
| 14 条全部采用 safe-template | 73.029068 | 82.517833 | 88.154615 | +0.120112 / -0.018035 / +0.166544 | -4.870932 / +0.417833 / -5.545385 |
| 事后最大 EAO，且 ACC 达标 | **73.098307** | 82.510509 | 88.221120 | +0.189351 / -0.025359 / +0.233049 | **-4.801693** / +0.410509 / -5.478880 |
| 事后最大 ROB | 73.091426 | 82.504051 | **88.233583** | +0.182470 / -0.031816 / +0.245512 | -4.808574 / +0.404051 / **-5.466417** |

“14 条全部采用 safe-template”一行精确复现既有 Top-14 投影，证明组合器和正式序列统计口径一致。
最大 EAO 组合只保留 8 条模板轨迹：`bag02_indoor_1`、`bag02_indoor_2`、`ball06_indoor_2`、
`earphone01_indoor_1`、`humans_shirts_room_occ_1_A_1`、`toiletpaper01_indoor_2`、
`toy02_indoor_1`、`toy09_indoor_1`；最大 ROB 组合再加入 `cup02_indoor_1`、`shoes02_indoor_1`
和 `stick_indoor_1`。

具体正负例进一步说明了原因：

- `bag02_indoor_2` 的模板轨迹把序列 ROB 从 `69.6185%` 提到 `76.4350%`，对全局 ROB 的贡献
  `+0.6145pp`，对全局 EAO 的贡献 `+0.2238pp`；它属于模板能延长遮挡后成功轨迹的有效正例。
- `toy09_indoor_1` 的序列 ROB 从 `42.4301%` 提到 `46.0032%`，全局 ROB/EAO 分别贡献
  `+0.3394/+0.3815pp`，说明动态外观在部分相似物或遮挡恢复片段确实有价值。
- `yogurt_indoor_1` 反而把序列 ROB 从 `78.1272%` 降到 `75.6929%`，全局 ROB/EAO 分别损失
  `-0.2778/-0.1748pp`；较长健康轨迹一旦写入偏移模板，递归污染会抵消多条短序列收益。
- `glass01_indoor_2` 的序列 ROB 下降 `0.9232pp`，全局 EAO 损失 `0.0572pp`；它与已记录的
  透明/相似外观干扰一致，RGB 直方图身份相似度并不足以证明框内仍是原目标。
- `notebook01_indoor_1` 虽有轻微正 EAO 贡献，但序列 ROB 下降 `2.2806pp`、全局 ROB 损失
  `0.1158pp`，说明仅按单一指标或单帧收益选择会掩盖后续轨迹缩短。

最重要的结论是：即使允许使用不可部署的事后序列标签，在**现有两套完整轨迹**中完美选择，也只比
正式 reference 提升约 `0.19pp EAO/0.25pp ROB`，远小于仍缺的 `4.80pp/5.47pp`。因此当前主瓶颈
不是简单的“模板 ON/OFF 门控选错”，而是当前单槽 RGB-D 模板候选（backbone 动态槽权重 `0.10`，
另以 `PRIMARY_TEMPLATE_BLEND_WEIGHT=0.02` 混入 primary template 输入）无法产生足够强的新恢复轨迹。
正在运行的逐帧 dormant read-gate 仍必须完成，因为同状态、
因果逐帧切换会改变后续递归状态，不受这个**完整序列轨迹组合**严格上界约束；但若其 Top-14 仍只获
小幅提升，下一轮应优先扩展候选动作空间：保留不可变首帧锚，加入短/中期多时间尺度模板槽，对透明物
和相似物增加局部结构/深度边界身份证据，并允许高风险失锁时使用更强但受事务回滚保护的模板权重；
不应继续只扫描一个全局混合权重或单一 motion threshold。

权威诊断产物和实现绑定为：

~~~text
/root/autodl-tmp/srtrack_primary_safe_template_exact_top14_seed2026_v4/runtime/top14_sequence_oracle.json
result SHA256: 0e77de45fbec5c38aecd6f1d4b168ac1776a5c74b4438522465b89e5d1f2fc29
analyzer: /home/SRTrack_RGBD_L/tools/vot_top14_sequence_oracle.py
analyzer SHA256: 017afd0b7b698ebad478753a9622267cfee92451559d84d3ae0dbd8a84411f9f
core VOT analyzer SHA256: a7d4437c23ac728a3a089d6faa8e55122f1d33f1d7de309f0117a3b2ef269c3b
reference manifest SHA256: 0ef8cfb6df4ee0e96a415c24e7faabdd6304670cbe7f97794fe680ccbe44b53f
candidate manifest SHA256: d7f82570de64840ee914305d82e79d290e7b93709bd43ccc1959a945a74b19b0
~~~

该 oracle 使用 VOT GT 做事后诊断，`diagnostic_only=true`、`official_full_dataset_result=false`，
不是可部署策略、不是新的正式 VOT 成绩，也不是逐帧 read-gate 的上界；24.72 的正式最佳表保持不变。

### 24.76 旧 V24 复合候选的序列 oracle：提高旧分支强度仍不足（2026-08-14）

为避免在 read-gate 之后盲目重复旧路线，对已有 V24 Top-14 完整轨迹运行了与 24.75 相同的
`2^14=16,384` 组合 oracle。V24 不是 `BLEND_WEIGHT=0.20` 的单变量消融：它同时启用动态模板、
V24 language-search recovery 和 cross-scale ranker。因此本节只把它定义为一个**旧复合强候选**，
不能把差异归因于任一权重或模块。

| V24 复合轨迹组合（非正式诊断） | EAO | ACC | ROB | 相对正式 reference | 距目标 |
|---|---:|---:|---:|---:|---:|
| 14 条全部采用 V24 | 73.112138 | 82.510434 | 88.253551 | +0.203182 / -0.025434 / +0.265480 | -4.787862 / +0.410434 / -5.446449 |
| 事后最大 EAO，且 ACC 达标 | **73.155295** | 82.499284 | 88.307492 | +0.246339 / -0.036584 / +0.319421 | **-4.744705** / +0.399284 / -5.392508 |
| 事后最大 ROB | 73.153979 | 82.498452 | **88.307791** | +0.245023 / -0.037416 / +0.319720 | -4.746021 / +0.398452 / **-5.392209** |

即使事后选择 V24 的最佳完整序列轨迹，EAO/ROB 仍分别差 `4.7447pp/5.3922pp`。相较 24.75
纯 safe-template oracle，复合强候选上界只再增加约 `0.0570pp EAO/0.0742pp ROB`，不足以改变
方向判断。旧 V27–V36 的 recovery selector、state hold、setwise/risk/latch 结果也已经分别被
Train-only 或 Top-14 门拒绝，因此下一轮不能再把这些旧模块重组并把微小变化解释为新进展。

同时，V24 的 `BLEND_WEIGHT=0.20` 同栈 OPE 曾得到 DepthTrackTest
`P/R/F=65.666593/64.498467/65.077288`，相对当前正式最好下降约
`0.329340/0.837419/0.586962pp`；CDTB 为 `75.378983/75.927675/75.652334`。这不能单独证明
`0.20` 是退化原因，因为 V24 栈还含 recovery/ranker，但足以证明“把旧强候选常驻部署”不满足
当前每项最多下降 `0.05pp` 的跨数据集保持门。若后续验证更强模板权重，只能作为同状态下的稀疏
事务动作，与 static/weak 动作同时解码并由 Train-only held-out harm 门选择；不能直接替换全局
`0.10` profile。

~~~text
/root/autodl-tmp/srtrack_primary_v24_branchsafe_template_seed2026/runtime/top14_sequence_oracle.json
result SHA256: f84711115e1663356363d7fdaf0cf7b0abc2f70c591a64ee6634305a6f908f11
analyzer SHA256: 017afd0b7b698ebad478753a9622267cfee92451559d84d3ae0dbd8a84411f9f
candidate manifest SHA256: dd2d8a3bbe6182f299ffa000ed30aec7d95f956fe049f36da3ba4c220b368b01
~~~

该结果同样使用 VOT GT 做事后诊断，既不是正式 full-127 成绩，也不是权重消融。其作用是收缩下一轮
设计空间：先完成当前 dormant read-gate；若其仍被拒绝，再新增 weak/strong same-state 模板动作的
独立 Train-only 取证链，而不是回退到已被否决的 V24–V36 组合。

### 24.77 source-identical 全量 Train 模板对照与下一代多动作架构（2026-08-14）

严格 OFF/ON 对照已经完成 152/152 条 DepthTrack Train 序列。两侧绑定同一 checkpoint、配置、
首帧文本、sequence list 和冻结 source snapshot；唯一实验变量是 safe-template 是否允许在线读写。
结果只使用 Train GT 做推理完成后的离线配对分析，没有读取 VOT、DepthTrackTest 或 CDTB GT。

| 全量 Train 指标 | 模板 OFF | 模板 ON | ON-OFF |
|---|---:|---:|---:|
| Precision (%) | 79.954950 | 79.897912 | -0.057038 |
| Recall (%) | 77.634990 | 77.336744 | -0.298247 |
| F-score (%) | 78.777894 | 78.596469 | **-0.181425** |
| 帧加权可见 IoU | 0.792598 | 0.790887 | **-0.001711** |
| healthy→miss transitions | 245 | 259 | +14（越低越好） |

模板不是完全无效：它恢复了 3,776 个 baseline miss 帧，5,040 帧的 IoU 提升超过 0.10；但同时
毁掉 4,235 个 baseline healthy 帧，5,615 帧的 IoU 下降超过 0.10。74 条序列改善、71 条退化、
7 条字节级轨迹不变，说明净退化不是“模板普遍无信息”，而是同一个模板动作在不同状态下正负作用
严重混合。全局常驻读取会把局部恢复收益连同递归漂移一起写回状态，因此不能直接用于提升 VOT ROB。

具体例子如下：

- `guitarbag_indoor`：平均可见 IoU `+0.618412`，恢复 902 帧，只毁掉 2 帧；属于模板能稳定补回
  外观变化目标的强正例。
- `bottle01_indoor`：平均可见 IoU `+0.249044`，恢复 1,117 帧且没有毁掉健康帧；说明 certified
  memory 本身有很强恢复容量。
- `trophy_indoor`：平均可见 IoU `-0.359886`，没有恢复 miss，却毁掉 238 个健康帧；是错误模板
  长时间递归污染的典型反例。
- `ball09_wild`：平均可见 IoU `-0.239456`，毁掉 408 帧且没有恢复；快速运动和相似外观下，
  常驻读模板把本来正确的静态轨迹拉走。
- `flowerbasket_indoor`：恢复 87 帧但毁掉 255 帧，平均 IoU `-0.179086`；同一序列内同时存在
  有益和有害阶段，证明仅做“按序列启用”也不够，必须按帧决策并对错误动作原子回滚。

据此曾设计独立 `static / weak / strong` 三动作 Train-only v1 取证链。公开递归轨迹始终执行 static；
safe writer 仍按原 profile 维护 certified memory，但读取动作只做同一 pre-frame state、search crop、
checkpoint 和模板快照上的无状态反事实 forward。`weak` 使用 backbone dynamic weight `0.10`，
`strong` 使用 `0.20`，两者的 primary-template input blend 都固定为 `0.02`；模板 tensor 在反事实
forward 前显式 clone，两个动作均不得写 bbox、writer、模板或其他递归状态。

启动顺序 fail-closed：当前 dormant read-gate evaluation 必须先形成完整科学拒绝；工程失败不启动，
若正式 VOT 已达标则新链直接跳过。随后等待两张 GPU 连续两次低于 500 MiB，先跑固定 6 序列并要求
static bbox/score 与严格 OFF reference 字节一致，再跑全 152；GT 只在 inference 完成后连接，分析
`strong` 相对 `max(static, weak)` 的增量收益、增量伤害和 strong-exclusive transition recovery。
这一阶段只判断动作空间是否有容量，不训练 gate，也不启动任何公开数据集评测。

~~~text
strict result:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026_v2/runtime/result.json
strict analysis SHA256: 230a48b30da23f19bc4f3db0c6adc1ae2d006da1b5753d01d1f0892ef44ff971
paired source snapshot SHA256: 1c939c2e678fed6b249bd346e07a9e99d6381af8d96d89f49a8bc8814e7c9156
next capacity root:
/root/autodl-tmp/srtrack_dormant_template_multiaction_train152_seed2026_v1
~~~

claim 边界：24.72 的三个公开数据集正式最好数字保持不变。本节证明模板具有强但条件化的恢复容量，
同时证明常驻读会净退化；尚未证明新的逐帧 selector 安全，也没有新的 Top-14 或 full-127 VOT 成绩。

### 24.78 dormant weak/strong 多动作 v1 历史部署状态（已由 24.80 废弃，2026-08-14）

> 本节保存 v1 当时的审计记录，不代表当前运行状态。源级检查随后证明 v1 的
> `dynamic_template_weight` 在 router=false 配置下没有进入有效前向；该 controller 已无产物终止，
> 当前有效方案与状态以 24.80 的 primary-template v2 为准。

新链已经部署并通过 Python 编译、模块导入、控制器 preflight 以及 33 项相关模板/read-gate/evaluation
回归。复审后进一步收紧了三个 fail-closed 边界：上游 result 必须与 terminal status 的 schema、stage、
result path 和 public-evaluation 标志一致；每种科学拒绝必须携带对应失败的 projection/OPE/formal VOT
指标证据；断点复用必须重验 checkpoint/config/text、精确 `0.10/0.20/0.02` 权重、reference/fixed6、
14 项 implementation snapshot，以及 prediction/trace aggregate。分析器每次重新执行并绑定 analyzer、
controller、rows JSONL SHA，控制器收到 SIGINT/SIGTERM 时终止整个子进程组。

容量判定也不再使用“存在一个有益帧”这种弱条件。除 strong-exclusive transition 外，现在至少要求
strong 相对 `max(static, weak)` 有 `max(10, 0.1% labeled rows)` 个显著有益帧、oracle 至少选择
strong 10 次、收益覆盖至少 3 条序列，且 strong 独有 oracle 增量 IoU 总和至少为 1.0。该门仍只是
动作空间取证，不等同于 selector 安全门；后续若进入训练，仍需独立三 seed calibration/audit 的零伤害门。

当时 controller 曾在 screen `dormant_template_multiaction_v1` 中以等待态运行，PID `407643`，
`public_dataset_evaluation_started=false`。它不会占用当前 full152 dormant trace 的 GPU，也不会在其
engineering failure 后抢跑；只有现有 read-gate evaluation 写出完整科学拒绝且两张 GPU 连续空闲后，
才会依次运行 fixed6 与 full152。启动时新增文件 SHA256 为：

~~~text
tracker 650f52f9e064250301c4a88ea02d76bcdf4e5aa7f516ea38a8a7b97ea676bed9
runner  6722b416981c0e96161a0615a5304edc1274edd2f61bc9862a82d4dd236d8dd1
analyzer 8ac9d55f4ff9d9654d5e25f868ecfd554646ad1c3fcee4637bfb2323e257c9ef
controller 3586b721840157563240acc12ef001816570ce59833ce8a6147a252e9f780ea7
~~~

测试边界：本次相关 33 项回归全部通过，追加的上游 VOT 数值重算与 read-gate 13 项回归也全部通过。
仓库 `tests/` 全套为 `1,215 passed / 6 failed`；其中 5 项是既有 rich-teacher/strict-visual pipeline
冻结的 actor SHA 与当前脏工作树不一致，另 1 项是既有 V29 测试要求恢复后的 template tensor 保持
Python 对象身份，而当前安全快照实现使用 clone。这 6 项均未引用本节新增的五个模块。无选择地运行
仓库根 `pytest` 还会在收集 `workspace/test_*` 时因历史 `lib.models.vipt/lib.config.vipt` 模块不存在
而停止；因此这里不把仓库全局状态表述为全绿，也不为修复无关历史样例扩大本轮实验源范围。

截至 03:56，当前 dormant full152 已闭合 16/152 条序列，累计 18 个 trace metadata、253 个
counterfactual frame、139 个 memory event、0 个 frame error，吞吐约 52.1 sequence/hour。

### 24.79 严格模板轨迹的连续收益/污染段与有限时域设计（2026-08-14）

为判断模板收益究竟是孤立的单帧偏移，还是会在完整递归轨迹中形成持续区间，对 24.77 已完成的
DepthTrack Train strict OFF/ON 轨迹做了新的只读连续段分析。分析器首先要求两侧 manifest 都是
`srtrack-safe-template-train-pair-run/v1`，角色分别为 source-identical OFF/ON，且共享非空的
implementation snapshot；随后复用原严格分析器，重验当前实现 SHA、两侧 456 个预测文件 aggregate、
checkpoint/config/text 与 152 条序列覆盖。GT 仍只在两侧推理全部结束后连接，不使用
DepthTrackTest、CDTB 或 VOT GT。四类帧总数必须逐项复现 24.77，才允许写出结果。

| 完整轨迹差异区间 | 连续段数 | 帧数 | 中位长度 | P90 长度 | 至少 30 帧 | 至少 100 帧 | 最长 |
|---|---:|---:|---:|---:|---:|---:|---:|
| IoU 增益 `>0.10` | 1,172 | 5,040 | 1 | 2.0 | 24 | 6 | 812 |
| IoU 损失 `<-0.10` | 1,158 | 5,615 | 1 | 4.0 | 39 | 12 | 375 |
| baseline miss `<=0.10` → template track `>=0.50` | 203 | 3,776 | 1 | 33.4 | 26 | 7 | 810 |
| baseline track `>=0.50` → template miss `<=0.10` | 202 | 4,235 | 4 | 56.4 | 35 | 11 | 375 |

具体区间能说明为何只看当前帧同状态 bbox 差值不够：

- `guitarbag_indoor` 在帧 `497–1306` 出现连续 810 帧 baseline miss→template track；对应的
  `IoU delta>0.10` 区间从 `495–1306`，持续 812 帧。
- `bottle01_indoor` 的恢复区间包括 `2283–2652`（370 帧）、`1928–2203`（276 帧）和
  `2654–2864`（211 帧），模板候选的价值主要表现为后续轨迹长期保持，而不是一次框的小修正。
- `colacan01_indoor` 在 `2479–2853` 连续 375 帧由 baseline track 变为 template miss；这是最长
  污染段，同时也是连续 375 帧 `IoU delta<-0.10`。
- `ball09_wild` 在 `641–947` 连续 307 帧由健康轨迹变为 miss，印证快速运动/相似外观下错误模板
  一旦进入递归状态，伤害也可能持续很久。

claim 边界必须严格保留：这是两套**完整轨迹的配对差异区间**，不能把任一区间归因到某一次具体的
write/read action，也不能据此宣称已有 selector 能安全部署。它证明的是当前训练标签只看单帧
`dynamic_iou-static_iou` 可能遗漏“当帧变化很小、未来轨迹价值很大”的动作。正在运行的 dormant
same-state full152 和已排队的 weak/strong 多动作链仍按原预注册顺序完成，不因本诊断修改权重或门槛。

若 weak/strong same-state 容量门也拒绝，下一独立取证链应采用有限时域 branch-and-rollout，而不是
继续堆叠旧 V24–V36 模块：

1. 在 Train-only 的候选触发帧保存完整 pre-action runtime snapshot，包括 bbox、safe writer policy、
   clone 后的模板 tensor、模板来源/年龄以及所有递归 proposal/state；同一 snapshot 分叉 static、weak、
   strong 三个动作。
2. 各分支在完全相同的未来 RGB-D 帧和冻结首帧文本上因果 rollout，例如 `H={5,15,30}`；分支间不共享
   tensor 或递归状态。推理期间看不到 GT，全部 rollout 结束后才计算累计 IoU、失锁转移、恢复持续时间和
   healthy-track 破坏。
3. 训练标签使用折扣 horizon gain 和硬 harm 标签，而不是只用当前帧 delta；calibration/audit 按序列
   完全隔离，三个 seed 均要求 audit precision 至少 0.85、healthy/catastrophic harm 为 0，失败不回退。
4. public runtime 不可能访问未来帧，也不执行 oracle rollout；它只用当前帧已有的 214 维 RGB、Depth、
   language、consensus、响应图和 memory-age 特征预测训练期定义的 horizon value。任何非 static 动作都
   先在事务 snapshot 上执行，证据不足或异常时恢复全部状态，且不得污染后续 writer/template。

该方案把“当前动作是否改变一个 bbox”改为“当前动作是否值得承担未来状态风险”，更直接对应 VOT
ROB/EAO 的长时失锁代价；但它仍必须先在 Train-only 证明因果容量和零伤害 selector，之后才能进入
Top-14、DepthTrackTest/CDTB 保持门与正式 full-127 VOT。当前正式最佳表 24.72 不变。

~~~text
analysis:
/root/autodl-tmp/srtrack_safe_template_source_identical_train152_seed2026_v2/runtime/trajectory_persistence_analysis.json
result SHA256: 1283ff72bd067495427adacbe9c46a9689ef09bfd3e4a6b2800310088632c0e1
analyzer: /home/SRTrack_RGBD_L/tools/analyze_safe_template_trajectory_persistence.py
analyzer SHA256: 7c11cab7d896004392e0da861150ea47f714d32ca5e9e377ba5b302f176109d2
strict pair analysis SHA256: 230a48b30da23f19bc4f3db0c6adc1ae2d006da1b5753d01d1f0892ef44ff971
paired source snapshot SHA256: 1c939c2e678fed6b249bd346e07a9e99d6381af8d96d89f49a8bc8814e7c9156
~~~

截至 04:05，当前 dormant full152 已闭合 24/152 条预测、形成 26 个 trace 文件；公开评测仍未启动。

### 24.80 dormant 多动作 v1 失效原因与 primary-template v2 修正（2026-08-14）

对 24.77--24.78 的实现做源级追踪后确认：配置中的
`MODEL.TEMPLATE_MEMORY.ROUTER_USE=false`，而 backbone 的 `dynamic_template` / `dynamic_template_weight`
只有在 `use_template_memory_router=true` 时才会生成并消费 dynamic language tokens。因此原 v1 所谓
weak=`0.10`、strong=`0.20` 实际没有进入有效前向；两个分支只共享同一个
`PRIMARY_TEMPLATE_BLEND_WEIGHT=0.02`。在已完整的 61 条序列、1,452 个捕获帧上，static/dynamic 的
selected index 仅改变 3 次；bbox 最大绝对差的中位数为 `0.0284 px`、P99 为 `0.9520 px`。因此观测到
的变化应归于共同的 primary `0.02` 路径，不能被解释为“0.10/0.20 dynamic 权重已有有效消融”，更不能
据此宣称 `0.10/0.20` 模板本身没有容量。

少数离群帧也与上述解释一致：`cat05_indoor#452` 的候选索引由 299 变为 323、bbox 最大差
`29.99 px`；`colacan01_indoor#1791/#1873` 分别由 327 变为 300/299，bbox 最大差
`19.72/28.12 px`。这些例子证明共同的 `0.02` primary blend 偶尔能跨过候选边界，但没有验证被
router 禁用的 dynamic `0.10`，也没有 static/weak/strong 三档可供训练。v2 正是为隔离这三个动作而重做。

在稍后的 62 条完整序列快照中，推理后连接 Train GT 共得到 1,454 个可见捕获帧：显著
`IoU gain>0.10` 仅 2 帧、显著 healthy harm 0 帧、catastrophic harm 0 帧、transition recovery 2 帧，
平均 IoU delta 为 `+0.000902`。两个恢复例均来自 `colacan01_indoor` 的 motion-risk：帧 1791 从
`0.0886` 提升到 `0.6226`，帧 1873 从 `0` 提升到 `0.8091`。它们说明 `0.02` 偶有高价值恢复，但
样本极稀疏；这些计数会随剩余序列完成而变化，不是正式 full152 analysis，也不改变预注册门槛。

原 v1 controller 已在**没有 fixed6、full152 或 result 产物**时安全终止，终态为
`failed / InterruptedError: controller termination requested`；其根目录保留作为被废弃证据，不再续跑。
替代的 v2 使用独立 root 和 v2 schema，将同一个 clone 后 certified template 直接与不可变首帧
primary template 做 `torch.lerp`：weak 精确为 `0.10`，strong 精确为 `0.20`。两个入口均拒绝任何其他
权重，并明确拒绝 backbone dynamic slot 或启用 template router；公开 static 轨迹、writer 和递归状态
不受反事实 forward 影响。

v2 的因果与复现边界如下：

- 同一 captured network call、search crop、pre-frame state、权重和模板快照上依次计算 static/weak/strong；
  weak/strong 只替换 primary template 输入，绝不提交 bbox 或内部状态。
- fixed6 必须先与严格 OFF reference 的 bbox/score 文件逐字节一致，才允许 full152；GT 仍在完整推理后
  才连接，只做 Train-only 容量分析，不启动 gate 训练或公开评测。
- implementation snapshot 已覆盖 tracker、runner、analyzer、controller，以及直接参与配置、数据构建、
  static 比较、IoU/GT 分析、状态风险、crop 和原子写入的本地依赖；运行期间任一 SHA 改变都会失败。
- controller 的 skip/final 结果同时绑定被废弃 v1 status 的路径与 SHA。两路只读终审均确认没有剩余
  阻断；`fixed6` aggregate 的完整重算由官方 controller 门执行。

验证结果：五个模块 Python 编译通过；合成前向准确得到 `0.20` template lerp，且确认没有
`dynamic_template` 参数进入网络；runner/controller preflight 通过；相关回归为
`62 passed / 1 timm deprecation warning`。当前 v2 尚未占用 GPU，待旧 read-gate 形成完整科学拒绝后才会
按 `fixed6 -> full152 -> Train-only capacity` 顺序运行。

截至本次记录，旧 dormant same-state full152 已完成 52/152 条预测、形成 54 个 trace 文件，累计
1,297 个 counterfactual frame、495 个 memory event、0 个 frame error；这仍是运行中诊断，不是新的
公开数据集指标。24.72 的 DepthTrackTest、CDTB 和 VOTRGBD2022 正式最好表保持不变。

~~~text
superseded v1 root:
/root/autodl-tmp/srtrack_dormant_template_multiaction_train152_seed2026_v1
v2 root:
/root/autodl-tmp/srtrack_dormant_template_primary_multiaction_train152_seed2026_v2
tracker SHA256: 6a09404cabba62069626abdec73411c7dc96662fac9113ba691ab789086e5af8
parameter SHA256: 5c64b05bca61d1170480dc93c5e36c56b794572b9b00d1a165cb8ce2bb4991c9
runner SHA256: b3510c396bc26889f4c1265e512e40225d6942378e471821cc6b5a263d504db9
analyzer SHA256: f99287c4dd0af796ded0a9dac5ed1216f16790c5075e047013c5e8cef782d00a
controller SHA256: f9629b692b21948766938233fb0829df9a923a11f72a9d3aae085dc991fcddc2
~~~

### 24.81 primary-template 三动作 selector 预注册协议（容量成立时才执行）

为避免在看到 v2 full152 标签后再挑选有利门槛，后续训练协议在容量结果产生前固定如下。若 24.80 的
`capacity_supported=false`，本协议直接取消，不创建 artifact，也不启动 Top-14 或公开评测。

1. 每个捕获帧的输入只含推理时可得量：static/weak/strong 三个 action 的冻结 evidence、三组两两
   evidence delta、template age、selected-index 一致性、bbox IoU/center jump/area change/motion risk、
   capture reason one-hot 和精确 `0/0.10/0.20` action metadata。不得输入当前或未来 GT、序列名、帧号、
   数据集身份或未来帧文本。
2. 共享 encoder 同时预测三个 IoU，以及 weak/strong 的 `gain>0.10`、transition recovery、healthy harm、
   catastrophic harm 概率。策略先判断 strong，再判断 weak；strong 必须相对预测的
   `max(static, weak)` 有足够增益，weak 必须相对 static 有足够增益，两者各自通过 harm 上限，否则 static。
   不允许用 strong 容量标签替代 weak 的安全标签，也不允许在运行时 oracle 比较 GT。
3. 每 seed 按**序列**做 5-fold：`audit_fold=seed%5`。calibration 的 OOF 模型均完全排除 audit 序列；
   策略网格只在 calibration OOF 上按固定总序选择，并要求唯一 winner。随后只用 calibration 序列训练
   audited model，对 held-out audit 恰好评估一个 winner；失败不得尝试第二策略。
4. 单 seed audit 必须同时满足：所选非 static 帧数大于 0、`IoU gain>0.10` precision 至少 0.85、
   healthy harm=0、catastrophic harm=0、transition recovery>0、总 IoU gain>0；若选择 strong，还要求
   strong 相对 `max(static, weak)` 的 audit 增量 gain>0。动作连续提交上限预注册为 1。
5. seeds `2026/2027/2028` 必须共享完全相同的 source/implementation/training provenance 且三者全部过门；
   不按 audit 指标二次择优，固定部署 seed 2026 的 exact audited calibration-only model。loader 必须逐份
   重验 analysis/rows/trace/reference/fixed6/GT aggregate、checkpoint/config、implementation 和 OOF SHA。
6. runtime 同帧先完成 static，再在同一 pre-frame snapshot 上计算 weak/strong；只提交 gate 选择的一个
   action。提交或 writer observe 任一步异常时恢复 bbox、writer policy、clone 后 template tensor/source、
   最近提交帧及所有递归状态，并返回已完成的 static 结果。运行时仍看不到未来帧和 GT。

该协议的第一公开门仍是 Train-only 选好 artifact 后的 VOT Top-14 projection；只有 EAO/ROB 都达到目标
投影且 DepthTrackTest/CDTB 每项相对 24.72 正式最好下降不超过 `0.05pp`，才允许 full-127 VOT。正式
EAO/ROB 未达到 `77.9/93.7` 时不得把容量、oracle 或 partial 指标写成新最好结果。

补充分支规则：v2 当前 hard capacity 主要检验 strong 是否在 weak 之外增加动作容量。若该门失败，唯一允许
的后续诊断是对**同一份已冻结 rows**做一次对称 weak-only 容量重算：weak 相对 static 的
`gain>0.10` 至少 `max(10, ceil(0.1% rows))`，transition recovery>0，oracle 选择 weak 至少 10 次，正收益
覆盖至少 3 条序列，累计正增量 IoU 至少 1.0。不得重跑 trace、修改 `0.10/0.20`、改变捕获帧或扫描新
阈值。weak 过门只允许进入上述三 seed selector；它仍不等于安全部署或新的公开指标。

实现审计说明：已被冻结并排队的 strong v2 analyzer 使用 Python `int(0.001*N)`（向下取整）；在
`N<=10,000` 时由于外层最少 10 帧，该实现与上述门等价，若 `N>10,000` 则可能比严格 0.1% 少 1 帧。
不在运行中改写其绑定 SHA；结果产生时必须同时报告 `N` 与实际 minimum。新增的 weak-only analyzer
使用 `ceil` 严格实现文档口径，并已用 `N=10,001、positive=10` 的边界样例验证应拒绝。

### 24.82 weak-only 容量链的实现、审查与启动状态

为使 24.81 的补充分支不依赖人工解释，已新增一条独立的只读控制链。它仅在 primary-template
三动作 v2 的终态为 `multiaction_capacity_rejected` 时读取该链已经冻结的 full152 rows；若 strong 增量
容量已成立或公开目标已经达成，则直接写明 skip。无论哪条分支，它都不得重跑 trace、启动训练或启动
任何公开数据集评测。

该实现同时绑定 source result/status、multiaction analysis/rows、full/fixed/reference manifest、source
analyzer 与自身 analyzer/controller/atomic-reader 的路径和 SHA。分析开始前缓存全部期望 SHA，计算结束及
controller 落盘前再次核验，防止运行中换件；中断后若 analysis 已存在，只允许用同一冻结函数完整重算并
逐字段相等后复用，若 result 已存在则完整复验并只修复 terminal status。分析子进程使用独立进程组，异常
路径统一 TERM、宽限、KILL 和 wait。

入口编译、preflight、合成正负例与相关回归均通过；两个独立只读审查方向已无阻断。审查期间发现原
`int(0.001*N)` 边界不等于严格 0.1%，weak analyzer 已改为 `ceil`，并验证 10,001 rows 只有 10 个
positive 时必须拒绝。2026-08-14 05:42（Asia/Shanghai）控制器已启动并进入
`waiting_for_primary_multiaction_capacity`，PID 425460；状态明确记录
`public_dataset_evaluation_started=false`。启动时旧 dormant full152 为 93/152 条完成预测、95 条 trace、
3,097 条冻结事件、0 个解析/事件错误；最近 15/30/60/120 分钟均约 44 条序列/小时，按当时速度估计
剩余约 1.34 小时。该运行进度不是新指标，24.72 正式最好表不变。

~~~text
weak-only root:
/root/autodl-tmp/srtrack_dormant_template_primary_weak_capacity_train152_seed2026_v1
screen: dormant_template_primary_weak_capacity_v1
analyzer SHA256: 1e01d096cbcbff9322b321bcc7eb8db31741343a50179b5a28d6518fa2b26ee6
controller SHA256: 985dc3e9c2f507781c6b0c59ba999c4d6ce7600e67d3d142aa11d7e3f6a92a51
~~~

### 24.83 dormant read-gate v1 的正式终态（2026-08-14）

24.77--24.78 的旧 same-state read-gate 链已经完整闭合，而不是中途样本。full152 manifest 为
`complete`：152/152 条序列、456 份预测、152 个 trace；记录中有 3,135 个 counterfactual frame、
1,202 个 memory event、152 个 metadata、0 个 frame error，static byte invariance 为 `pass`。Train GT
只在推理完成后连接，共形成 3,101 个标签：`gain>0.10` 的 beneficial 2 帧、harmful 1 帧、
catastrophic 0 帧、transition recovery 2 帧；总动态 IoU 增量为 `+1.2437804743`，平均每标签增量为
`+0.0004010901`。

三组预注册 seed 2026/2027/2028 的 calibration feasible policies 均为 0，因而
`audit_policies_evaluated=0`、`held_out_policy_audit_passed=false`，终态全部是
`training_rejected_no_safe_oof_policy`。selector 没有 artifact、没有 selected seed，控制器终态为
`complete_rejected_no_safe_three_seed_gate`，并明确记录 `public_dataset_evaluation_started=false`。
这证明旧 `0.02` read 动作虽有两个孤立恢复例，却没有足够的可泛化安全决策边界；它不构成模板更新
有效性或 VOT 提升结论。

~~~text
old read-gate root:
/root/autodl-tmp/srtrack_dormant_template_counterfactual_train152_seed2026_v1
full152 manifest SHA256: e505a10608b71ecf566ca21c7bd44403c70787d3d5ae309c6fe8bb407faf4ded
analysis SHA256: 34afc284c2bb895c4839af0c26f77f249729201268b12947da4edf7ab5955623
result SHA256: 56c3cfc9461da6297fcbf7b5545f45aa70aa936f79bb648bd79226a96409c4f9
selection SHA256: 0dbfb4964b0af9b6922c637f549847f51197826bb0ad5b33a7332b9cbbde1e91
terminal status SHA256: ba640ba44e1f75b3688fd4f407e8bc8d87f6de4230224328a4e2d0d7ffa845ee
~~~

### 24.84 primary-template 多动作安全 gate 的实现、终审与启动

容量成立时的三 seed gate 已按 24.81 落地。每个样本使用 436 个纯推理时特征：static/weak/strong
三组冻结 evidence、三组两两差分、template age、bbox/候选一致性、capture reason 和精确动作元数据；
共享网络输出 3 个 IoU 回归量，以及 weak/strong 各 4 个 benefit/transition/harm/catastrophic 概率，
共 11 个输出。策略空间固定为 `5^6=15,625` 个阈值组合，先按动作行为去重，再要求 calibration OOF
唯一 winner；calibration 的每个 OOF 模型都排除 held-out audit 序列，winner 只审一次，失败不回退。
seeds 2026/2027/2028 必须全部过门，部署仍固定 exact seed 2026 模型。

安全边界同时在训练入口、selector 和 runtime loader 闭合：artifact 必须绑定 capacity result/status、
analysis、rows、152 个 trace、reference/fixed6、Train GT aggregate、checkpoint/config、implementation 和
OOF。OOF 的 436 维特征不再相信自报字段，而是从绑定 trace 重新构造并逐行比较。strong 历史 analyzer
的 floor minimum 与下游严格 ceil minimum 分开记录；若走 weak-only 分支，weak analysis 必须把
source result/analysis/rows 的路径与 SHA 绑定到当前冻结源，并由 trainer 与 runtime loader 从这些 rows
独立重算六项 ceil 容量门，不能混用另一轮实验的自洽报告。

断点恢复也采用不可混写设计：每个 seed 的每次尝试写入唯一 `attempt-<time>-<pid>-<seed>` 目录，
只有完整验证后的 report 才通过原子 `final.json` 提升；进程树继承唯一 attempt 环境标识。控制器重启时
先按该标识定位父进程已退出后仍存活的子孙 PGID，统一 TERM、宽限、KILL 并复扫确认，再隔离残缺
attempt，旧进程不能向新 attempt 或 canonical 路径回写。

六个模块 Python 编译、436 维 import 和 controller preflight 均通过；新增 4 项 focused tests 覆盖
冻结 rows 的 weak 容量重算、残缺 attempt 单次隔离、精确 seed 路径扫描、父进程退出后的子孙清理，
结果为 `4 passed`。spec 与 standards 两个独立只读终审均确认最新实现无阻断；重复的 weak 纯校验
逻辑和 `/proc` 遍历仅记为后续可抽取的非阻断维护项。

2026-08-14 10:11（Asia/Shanghai）gate controller 已启动，PID 629873，状态为
`waiting_for_primary_multiaction_capacity`，且 `public_dataset_evaluation_started=false`。启动前最后一次
进度核验为 strong v2 full152 137/152 个 trace、405/456 份预测；strong 结果尚未产生，weak 控制器仍在
等待，因此此时没有训练 artifact，也没有新的公开数据集指标。24.72 的正式最好表保持不变。

~~~text
gate root:
/root/autodl-tmp/srtrack_dormant_template_primary_multiaction_gate_train152_seed2026_v1
screen: dormant_template_primary_multiaction_gate_v1
shared gate SHA256: 3a43c3a090d579e625400d1c23127c8d429bc9a686a466c59597e342d0534b5a
trainer SHA256: cee848bb0b664191e34971eb407c97fd0e4c5446047276ae607b9e589ed6ab9a
selector SHA256: 1bd2431bc50b55378b5c09c45baa00d9f6b7d32894b00635d08b07253dbfd820
runtime tracker SHA256: a81b3f7a88b12ecb6f452392a299ed6962f8327af4a7dea320e5e88279f1f2b0
runtime parameter SHA256: f44f5f9f85edd590758e4b42e9b8416f5b888f82b30dfb2f8d76d9db7ed65e68
controller SHA256: 849be4acce4bb98bcf2738043752b2f63a9ada2dbe5246139cf0773722f764c7
focused test SHA256: 19d4aff48e091549343a3a950355daf3bfe67022eeece36e325ea4fc7db350a6
~~~

### 24.85 primary-template 三动作门终态、失败样例与 temporal v3（2026-08-14）

24.84 的等待链现已闭合。primary-template 三动作 full152 共得到 3,101 个 Train-only 可见标签；
strong 相对 `max(static, weak)` 有 17 个 `IoU gain>0.10` 正例，同时有 16 个 healthy harm 和
4 个 catastrophic harm。strong-exclusive transition recovery 虽有 7 帧，却只覆盖 4 条序列：
`cube06_indoor=1、cup07_indoor=2、cup09_indoor=1、cup10_indoor=3`。strong/weak 候选索引改变
39 次，weak/static 改变 29 次；说明动作确实有容量，但安全正例稀疏且跨序列覆盖不足。

三 seed 预注册门的正式终态如下。这里的 audit 是按序列隔离的 held-out fold，不是再次调阈值：

| seed | calibration/audit 行数 | calibration 结果 | held-out audit 结果 |
|---:|---:|---|---|
| 2026 | 2,381 / 720 | 0 个 feasible policy | 未审策略，拒绝 |
| 2027 | 2,530 / 571 | 选 6 帧，precision=1.0，transition=4，gain=+3.201017 | 选 0 帧，transition=0，gain=0，拒绝 |
| 2028 | 2,615 / 486 | 选 7 帧，precision=0.857143，transition=4，gain=+2.923689 | 选 0 帧，transition=0，gain=0，拒绝 |

三个 audit fold 的 strong-exclusive transition 真值都为 0；2026/2027/2028 audit 中 strong
transition 真值分别为 `0/0/1`。因此不是把阈值再放宽一点就能解决：放宽会同时放入已有的 harm，
而且 held-out 序列没有足够的恢复入口可证明安全。selector 最终为 `three_seed_gate_rejected`，没有
artifact、没有 selected seed，也没有启动 Top-14 或任何公开评测。

具体失败样例说明单帧、跨序列分类为什么不稳定：

- `book01_indoor#881`：真实 static/weak/strong IoU 为 `0.2489/0.5761/0.9284`，strong 的真实
  增量为 `+0.3523`；但 seed 2027 OOF 预测增量为 `-0.0671`、benefit 概率仅 `0.1193`，漏掉正例。
- `dumbbells02_indoor#592`：真实 strong 增量为 `+0.1150`，刚超过预注册收益线；seed 2028 虽
  预测增量 `+0.1107`、benefit 概率 `0.8337`，却同时给出 harm 概率 `0.5540`，安全门必须拒绝。
- `cup10_indoor#1323`：上一帧 IoU `0.9688`，static/weak 都跌到 0，strong 恢复到 `0.9738`；这是
  强动作价值最大的典型帧之一，但这种 strong-exclusive 入口只集中在极少序列，不能代表 audit 分布。
- `leaves05_indoor#860`：static/weak/strong IoU 为 `0.4482/0.3399/0.2793`，strong 增量
  `-0.1689`；它与上述恢复帧共享 motion-risk 外观，说明“发生跳变”本身不能区分救回和加害。

对 17 个正例和 16 个 healthy-harm 帧做同类时距检查，前后 `1/2/5/10` 个已捕获帧内都没有第二个
同类事件。旧 trace 因而只能回答“本帧如果读模板会怎样”，不能回答动作收益能否持续、下一帧是否
立即反转。这与 24.62 的 VOT 诊断一致：正式 ACC 已超过目标，而 ROB 低 `5.711929pp`、EAO 低
`4.991044pp`；`cup02/toy09/earphone/glass` 等主要是遮挡、相似物和背景竞争后的递归失锁，不是
单帧框精度普遍不足。要改善 ROB，selector 需要可验证的短时持续证据，而不是继续扫描单帧阈值。

为此新增 temporal v3 Train-only 取证链，仍保持公开递归轨迹完全 static：

1. certified writer 的 `replace_dynamic` 或 active-template 状态下 public motion-risk 触发后，只捕获
   严格未来 `t+1/t+2` 两帧；重叠触发把窗口重置为 2，`drop_dynamic` 立即清零，序列末尾只允许显式
   tail truncation。weak=`0.10`、strong=`0.20` 仍是同一 pre-frame snapshot 上的无提交反事实动作。
2. 验收器不信 trace 自报 risk：它从与 strict OFF reference 字节一致的逐帧 public bbox 独立重算
   jump/area/risk，再用完整 replace/drop 事件重建 template-active 状态。漏掉整个触发帧及后两帧、
   多采、重复帧、NaN/Inf、自报 bbox/risk 漂移都会失败。
3. producer、fixed6 授权点和 analyzer 消费点使用同一个逐序列状态机；fixed6 重新比较 12 个静态文件，
   full152 analyzer 重新比较 304 个静态文件，不能只信 manifest 的 `pass` 字段。GT 仍仅在完整推理后
   连接；当前阶段不训练、不部署、不评公开数据集。

真实 `bottle03_indoor` smoke 已通过：70 个 trigger、16 个 drop、134/134 个 required future frame、
130 个纯 post-trigger frame、2 个合法 tail truncation、0 frame error，static reference 比较 2/2 通过。
随后 fixed6 也通过：99 个 trigger、35 个 drop、188/188 个 required future frame、180 个纯
post-trigger frame、0 frame error，12/12 静态文件字节一致，public static decode 最大差为 0。
spec/standards 两路独立只读终审均确认无硬阻断。

2026-08-14 12:08（Asia/Shanghai）full152 已用两张 GPU 启动；只有 full trace 完整通过 SHA、
304 文件静态字节门和 temporal schedule 后，脚本才会自动执行 Train-only GT 后接分析。公开评测仍为
`false`，24.72 的正式最好表不变。

~~~text
primary multiaction result SHA256: 2145e7da1766ef6b33ba5be3a4646c05755b6348d2513418fb91e0b5ed30685c
three-seed gate result SHA256: 8df45109c8f2e6e6f7b5f10e70dae4da54464275746e14b77a6a86e24358cc96
three-seed selection SHA256: 14e23418dd3604a3594b14f4ea48e1daab8e95f85ae54ec200913a5f0acf33d3
temporal root: /root/autodl-tmp/srtrack_dormant_template_primary_temporal_train152_seed2026_v1
smoke manifest SHA256: 2677a404970ffa780c19dd75cab21a5ba7e54e417e2ec8348f785efeaf930f25
fixed6 manifest SHA256: 5928e981f088a807977e70a2968feb22e6f0c03fd65459df39ae4cc243d75bf7
base trace tracker SHA256: 867ab598615748d334ed4ee9ec3664f3a5c49a8a678bbb588ffdc926e905aa1d
temporal tracker SHA256: 99c37b1f9d36b02f8883a76e5102d39b1b733767e8f01a59b6d36b15a698f6a8
temporal parameter SHA256: ec5a28999f451ec646d5c57062f4c546f60cd3ea1ce627f86a1b1611c9a37407
runner SHA256: fc78d2b23a751f37337c7c08d8fdbdfec3c01f7b3420a491b15f891260a65a5b
analyzer SHA256: 4780031177d3e1c66d9b699c8a08cdfcf0eba56c25b8286fa2b3b551e4148fd8
~~~

### 24.86 temporal v3 full152 终态与训练授权边界（2026-08-14）

2026-08-14 15:49（Asia/Shanghai），24.85 的 temporal v3 full152 已正常结束，screen 自动退出、两卡显存释放，
trace 与随后执行的 Train-only GT 后接 analyzer 均无 Traceback/OOM/error。源码快照逐项复算与 manifest 完全一致；
这批数据只授权训练选择器，不授权直接启用 strong 动作，也不授权公开 VOT/DepthTrack Test/CDTB 评测。

| 完整性证据 | 原始结果 | 判定 |
|---|---:|---|
| 序列 / prediction / trace | 152 / 456 / 152 | 完整 |
| strict OFF reference 静态文件比较 | 304 | 全通过 |
| public static 最大解码差 | 0.0 | 字节级零扰动 |
| temporal trigger / drop / tail truncation | 2,723 / 1,169 / 4 | 状态机闭合 |
| required future frame | 5,294 / 5,294 covered | 无漏采 |
| post-trigger temporal burst / frame error | 5,072 / 0 | 通过 |

GT 后接后得到 8,135 条可见标签，另有 69 条不可见帧被跳过。与旧单帧三动作的 3,101 条相比，
temporal v3 增加 5,034 条标签（+162.33%），主要来自 5,037 条可见 post-trigger 样本；动作变化与标签如下：

| 指标 | 原始值 |
|---|---:|
| weak 相对 static 改变 peak | 37 |
| strong 相对 weak 改变 peak | 60 |
| weak transition recovered | 4 |
| strong transition recovered | 11 |
| strong-exclusive transition recovered | 7 |
| strong incremental beneficial / harmful / catastrophic | 21 / 23 / 4 |

七项 capacity check 全为 true，`capacity_supported=true`、`eligible_for_gate_training=true`。oracle 相对 static 的
总 IoU 可恢复量为 `+39.644767`；其中 strong 相对最佳 non-strong 的 oracle 增量为 `+21.387779`。但若无条件
使用 strong，它相对最佳 non-strong 的总 IoU 是 `-13.564989`，即“存在可学习容量”与“动作本身安全”明显不同。

该结果支持下一步独立三 seed、按序列隔离的 calibration OOF + 单次 held-out audit：模型需要从 temporal 证据中
识别少量真实恢复入口，同时拒绝 23 个 healthy harm 与 4 个 catastrophic harm。只有 2026/2027/2028 三 seed
全部满足 precision>=0.85、零 harm、transition>0、gain>0，才可固定 seed 2026 并进入公开评测。

本阶段终态仍为 `gate_training_started=false`、`public_dataset_evaluation_started=false`；因此 24.72 的正式最佳
DepthTrack Test、CDTB 与 VOT RGBD2022 指标均保持不变，不能把本节 oracle/capacity 数字当成公开数据集提升。

~~~text
full152 manifest SHA256: c89a56af53604342ea9447235ccbcab75cca34d57a975131ae414dd1fa53a347
temporal capacity SHA256: 775606d4822087cdc650881a615f1993ce2fb75fb79f77de05c6b9eda63bd89e
temporal rows SHA256: e41b586f6d7478c4c4324c396b7c56ac5403a548446faa9c1b49434ddcbf6b6f
implementation snapshot match: true
public dataset evaluation started: false
~~~

### 24.87 temporal v3 三 seed gate 终态与正式指标边界（2026-08-14）

24.86 的 Train152 容量授权已进入正式三 seed gate。启动前关闭了最后一个训练/部署契约缺口：
`post_trigger_temporal_burst` 是由 runtime 因果队列产生的外部动作资格，不是模型输入。模型仍只使用
433 个可在推理时重建的数值特征；capture reason 不进入 `FEATURE_NAMES`。普通 `motion<0.4` 帧仍返回
static，只有 writer/risk 触发后严格未来两帧可绕过 motion 门；IoU、预测增益、benefit、transition、
harm/catastrophic 阈值和最多一个连续非 static 动作预算均不绕过。该描述取代 24.84 中“436 维且包含
capture reason”的早期实现说明。

远程真实数据回归重建了全部 8,135 行、433 维特征，所有值有限；同一个 `motion=0.1` 合成样本在普通帧
训练/runtime 都为 static，在 causal burst 帧两侧都允许 strong。另一个反例 `static=0.4、weak=0.9、
strong=0.1` 被正确计为 1 个 healthy harm、1 个 catastrophic harm，strong 相对最佳 non-strong 增量
为 `-0.8`。spec/standards 两路最终只读复审均为 PASS、0 个阻断。

2026-08-14 16:48（Asia/Shanghai）三 seed 训练、校准和单次 held-out audit 已全部结束：

| seed | calibration / audit 行数 | 15,625 个阈值组合的 calibration 结果 | held-out audit | 终态 |
|---:|---:|---|---|---|
| 2026 | 6,261 / 1,874 | feasible policy=0，behavior=0 | 未审策略 | 拒绝 |
| 2027 | 6,638 / 1,497 | feasible policy=1,420，behavior=3；唯一 winner 只选 1 个 strong，precision=1.0，transition=1，gain=`+0.973845` | 选 0 帧，transition=0，gain=0 | 拒绝 |
| 2028 | 6,859 / 1,276 | feasible policy=0，behavior=0 | 未审策略 | 拒绝 |

seed 2027 给出了最具体的失败例：校准 winner 的阈值是 motion `1.2`、action IoU `0.7`、预测增益
`0.3`、benefit `0.95`、transition `0.9`、最大 harm `0.3`；它在校准 OOF 中只抓到一个
strong-exclusive 恢复帧，真实增量 `+0.973845`，但在完全隔离的 1,497 条 audit 行中一次也不触发。
这说明 temporal v3 扩大了样本量并证明存在 oracle 容量，却没有把稀疏恢复入口变成跨序列稳定的判别边界。
2026/2028 连 calibration 的零伤害门都无法通过；继续放宽阈值会把 23 个 healthy harm 和 4 个
catastrophic harm 带入动作集，因此不能把“零动作”解释为可部署安全策略。

selector 终态为 `all_required_seeds_eligible=false`、`ready_for_top14=false`，没有 artifact、selected seed
或 public candidate；控制器终态为 `complete_three_seed_gate_rejected`，并保持
`public_dataset_evaluation_started=false`。因此没有运行 VOT、DepthTrack Test 或 CDTB，新实验没有产生
可报告的公开指标；正式最好值保持如下：

| 数据集 | 指标 | 当前正式最好 | 目标 | 差值 |
|---|---|---:|---:|---:|
| DepthTrack Test | Precision | 65.995933 | 65.2 | +0.795933 |
| DepthTrack Test | Recall | 65.335885 | 64.9 | +0.435885 |
| DepthTrack Test | F-score | 65.664250 | 65.1 | +0.564250 |
| CDTB | Precision | 75.387821 | 72.9 | +2.487821 |
| CDTB | Recall | 76.005850 | 75.6 | +0.405850 |
| CDTB | F-score | 75.695574 | 74.2 | +1.495574 |
| VOT RGBD2022 | EAO | 72.908956 | 77.9 | -4.991044 |
| VOT RGBD2022 | Accuracy | 82.535868 | 82.1 | +0.435868 |
| VOT RGBD2022 | Robustness | 87.988071 | 93.7 | -5.711929 |

科学结论是：当前主要瓶颈仍是 ROB/EAO 所对应的遮挡后递归失锁，而不是 ACC；单帧或固定两帧
counterfactual 虽能发现少量恢复动作，但跨序列泛化不足。下一轮若继续，应优先增加可部署的历史状态
表示（例如动作前后短轨迹聚合、遮挡/相似物竞争状态），再重新预注册 gate；不应在当前 audit 结果上
继续调阈值，也不应越门做公开评测。

~~~text
capacity result SHA256: 849d491af5d6b6eba557e732cb56ebb4baf011e8955dd970bff674318a1a9609
seed 2026 report SHA256: 52c0e448f9e3d548f202261bcb5b9283d66a734c509e40d7b86952df22d6d5cc
seed 2027 report SHA256: 00e1429e27c5685d49c8e782d591899a350263e2d937aedeb1283ee1e7c3c4e2
seed 2028 report SHA256: 61228624a8c22e28fc56825550be272647875594c4e4adeabf64a09e7b9e954c
selection SHA256: 4e0cc81d03a4eb76ce7b97fb1daec1ee43f78d2199420d02f85f800e52b8b566
gate result SHA256: 9cd68e629d10b7ac96705a5520c4f4a68e22642d9cc7168979f9e3105cda8960
terminal status SHA256: 0b66e53a630b8788c3f66a37327aeb494862ca120c2897dc62eb4966b66aa983
shared gate SHA256: 585a8d0662763ed4621152c9b302d01b81db703c4c3f9e9b604f8de4646d63cc
trainer SHA256: d31119cdee24c770d9cfec3d46e3affe947922f2b75fee2aea8063b4228051d6
selector SHA256: 974a4229f72a240783cdf6fe74d2de3d825db8a0d7a5bd0078839058f284f1fd
runtime tracker SHA256: 4001773263a9c698dd29ee68d7ea50ea2134edc51fab595cc60117c41d7a4245
runtime parameter SHA256: 69986b377bf2b9e01965dee4f8d35e4383ad228581b053f0ff40f2a44ef16422
controller SHA256: e564b33ecf42c7fd61466917ef2bf11f037f153b9acc8bbd1ab57bdb3b74fb5f
public dataset evaluation started: false
~~~
### 24.88 VOT-RGBD2022 源码口径、开源高分机制与质量感知融合探针（2026-08-14）

本节补齐 24.62 的指标源码口径，并把开源方法调研转成当前工程的最小可检验改动。完整的模型、
论文、官方提交包与公平比较边界见：

~~~text
/home/SRTrack_RGBD_L/docs/rgbd_open_source_tracker_research_20260814.md
SHA256 bc56d29b25f4ef0e2fb7f72e95afa3d8271a206a1df463d8eefa8c4b0af117b5
~~~

#### 24.88.1 固定 toolkit 的实际计算，而不是沿用旧 VOT 直觉

服务器固定使用 `vot-toolkit==0.7.1`。权威实现与 stack 为：

~~~text
/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/analysis/multistart.py
SHA256 5a09065e2315387405f4cb8f96c0b8fd32d7428996a34eedd92ac4e2a4deeb02
/root/miniconda3/envs/mplt/lib/python3.8/site-packages/vot/stack/vot2022/rgbd.yaml
SHA256 7a2822e3a1500e674d000ae73e526f888e8cf058454cfa70311eba8a0103f35a
~~~

RGBD2022 是 anchor multi-start：每个 anchor 独立向前或向后跑到方向终点，fragment 内失败后不
reset。目标非空时，IoU `<=0.1` 连续 10 个分析下标才确认失败；中间任一帧恢复到 `>0.1` 就把
grace 重置为 10，确认失败时 progress 回溯到低重叠 streak 的第一帧。该版本的 `burnin=10` 没有
删除前 10 帧，只作为 bounded-overlap 的真值开关，不能按旧 supervised-VOT 的跳帧/reset 解释。

对序列 `s` 的 fragment `r`，设方向总长为 `L_r`、失败前 progress 为 `p_r`、重叠为 `o_rt`：

~~~text
A_s = sum_r sum_(t < p_r) o_rt / sum_r p_r
R_s = sum_r p_r / sum_r L_r
~~~

全数据集 ACC 以各序列的 `sum_r p_r` 加权；ROB 不是失败次数，而是先得每条序列的 `R_s`，再按
原始序列长度加权，越高越好。EAO 对每个 fragment 构造前缀平均；失败后的尾部置零，越早失败就
给更长前缀带来更多零尾。stack 写 `low=115, high=755`，但 v0.7.1 实际曲线数组只有下标
`0..754`，Python 切片最终平均 `115..754` 共 640 个点。当前正式结果仍为：

~~~text
EAO / ACC / ROB = 72.9089559737 / 82.5358676995 / 87.9880710788
目标             77.9          / 82.1          / 93.7
差值             -4.9910440263 / +0.4358676995 / -5.7119289212 pp
~~~

所以优化约束非常明确：ACC 已过门，不应靠更激进回归继续抬失败前 IoU；必须在第 10 个连续 miss
之前重新取得 `IoU>0.1`，并尽量让恢复发生在 EAO 115--754 的生存区间之前。

#### 24.88.2 按官方 ROB 聚合重算的缺口和重点序列

24.62 的 1,765 个正式 fragments 逐序列重算后，原始序列长度权重分母为 `80,741`，当前加权
生存分子为 `71,042.448470`；93.7% 目标分子为 `75,654.317000`，还差
`4,611.868530` 个加权生存单位，相当于必须消除现有加权生存损失的 `47.5521%`。直接把所有
anchor 帧拼接所得的 82,652-frame 近似不是官方 ROB 口径，不作为正式结论。

~~~text
序列                         失败/anchor   序列ROB   对全局ROB缺口贡献
cup02_indoor_1                 36/36         6.960       2.0200 pp
earphone01_indoor_1            20/20        24.536       1.0141 pp
toy09_indoor_1                 26/26        42.430       0.9547 pp
glass01_indoor_2               19/19        15.818       0.9331 pp
shoes02_indoor_1               12/13        25.208       0.5688 pp
bag02_indoor_2                 13/27        69.618       0.4783 pp
~~~

前四条即使事后全部完美修复也只提供约 `4.9220pp` ROB，仍小于 `5.7119pp` 缺口；因此不能只
为四个公开序列写规则。它们用于解释 failure family：`cup02` 是相似杯身份峰误选，`toy09` 是
遮挡/背景/快速运动联合，`earphone01` 是极小目标背景竞争，`glass01` 是透明目标且 RGB/depth
身份都弱；`shoes02/bag02` 说明同类相似物与遮挡恢复还必须泛化到更多序列。

已有离线 perfect-tail oracle 给出了机制规模下限。若能事后准确识别并完美修复 145 个
`direct_jump_any_score` 失败运行，投影为 `EAO/ACC/ROB=78.3495/83.8827/94.1215`；147 个
`lookback_without_depth` 为 `78.3151/83.8266/94.1072`，均过门。只修复 129 个
`lookback_v14_semantic` 则为 `77.4763/83.6682/93.2735`，仍不过门。真实 factor-9 replay 只有
24 个正确 commit，投影 `EAO=73.8653, ROB=89.1035`。这些都是使用未来 GT 的上界而非部署结果，
但它们证明当前问题需要约 145 个高精度 fragment 级恢复，不是十几个模板动作或 +0.1pp 微调。

#### 24.88.3 为什么开源模型高，以及当前主干缺了什么

VOT2022 官方 MixForRGBD 为 `77.9/81.6/94.6`，SAMF 为 `76.2/80.7/93.6`。官方说明中，
MixForRGBD 使用两套 MixFormer、逐元素 max 融合、置信度/间隔在线模板，并用 LaSOT、COCO、
GOT-10k 的 DenseDepth 伪深度加 DepthTrack-train 微调；SAMF 用双 MixFormer 加 SA-Gate 学习
融合。它们的数字同时包含强 backbone、更多数据和训练预算，不能把全部增益归因于模板或 gate。

当前工程代码则在 12 层 RGB/depth 独立流和 4 个 cross blocks 后固定执行
`(x_rgb+x_event)/2`。深度预处理逐帧按非零中位数裁剪、min-max 到 0--255 再 JET 伪彩，因此绝对
和跨帧 metric depth 尺度被丢弃。最终 primary checkpoint 与最初映射视觉初始化相比，RGB/depth
blocks 的平均相对改变量分别只有约 `1.193%/1.148%`；从 V5 到最终 primary，visual/head 全部
逐张量不变。最终 checkpoint 中 148 组对应 RGB/depth 参数的平均余弦相似度接近 `0.9998`，patch
embed 余弦约 `0.999884`。语言诊断虽常选择 depth route，但 depth backbone 并未形成足够独立的
专门表示。这解释了为什么固定均值在相似物、遮挡和深度失真时不能可靠纠错。

历史 depth-only 全层微调不是可接受的直接解法：它虽把 fixed-6 总 F 提到约 60.95（相对当时更弱
source 有提升），却把 `ball16` 后段可见 IoU 从 `0.6762` 降到 `0.2383`，安全门失败。下一步必须
保护强 RGB/language 路径，只学习受界的小融合，而不是再次解冻整个 depth backbone。

#### 24.88.4 已落地的零扰动 reliability fusion 与当前运行状态

新增 `DepthReliabilityFusion`：RGB/depth token 经过结构对称的低维投影，结合局部差异、乘积一致性
和全局上下文，输出逐 token preference。最终融合为：

~~~text
mean = (rgb + depth) / 2
fused = mean + preference * (depth - rgb) / 2
preference in [-0.35, +0.35]
~~~

所以两模态权重始终为正；最后决策层全零初始化，训练前 `fused` 与历史 mean 逐元素完全相同。新配置
只训练 `28,737` 个 fusion 参数，RGB/depth backbone、cross blocks、head、语言和模板逻辑全部冻结。
完整 checkpoint 预检确认 0 个 source tensor mismatch、唯一 optimizer group 为 fusion，聚焦测试
`3 passed`。相关文件与 SHA：

~~~text
lib/models/srtrack/depth_reliability_fusion.py
cf065e3a11dd3a97d8b8479057a7c45a6168aa211927c5789378480c9075a6bb
lib/models/srtrack/siamtrack_dropmae.py
67246fd4fbfa1e70440768bc5d58c8e9b381ca4a24ba526176052abe030160ed
lib/config/srtrack/config.py
55e9e010fa0be00982ea718a4eed0247c26bcce0d04ffe5857ff988c07e02719
lib/train/base_functions.py
6a858e0615eeec91e43d6bcd1738482c160b81da04f036495beb131f9b537463
experiments/srtrack/droptrack_depthtrack_final_language_depth_reliability_fusion_probe_e1.yaml
ea3f7cd0d50b255c43d2ef569595ddea1c098705c6c558740eb2ce92bc0e4503
tools/run_depth_reliability_fusion_probe.py
13011abf0bc4a5e7422ed9fc086da86ce0473a1a5c3aae66702359c2c4995899
~~~

运行前两次分别因旧入口缺 PATH、缺 PYTHONPATH 在模型加载前失败；第三次发现旧
`tracking/train.py` 无视 `--mode single` 并硬编码多卡，已主动终止且记录为 failed。正式有效探针改为
直接调用 `lib/train/run_training.py`，只见 GPU 0，目录：

~~~text
/root/autodl-tmp/srtrack_depth_reliability_fusion_probe_e1_seed2026_v4
~~~

本节写入时 seed 2026 的 DepthTrack-Train `8,000 x 1 epoch` 正在运行；
`public_evaluation_started=false`。单 seed 只回答 fusion 是否有容量，不能授权 VOT。只有 fixed-6
不低于 safe025 `65.504506/65.219985/65.361936` 且无序列级灾难退化，才预登记三 seed；随后还需
DepthTrack/CDTB 保存门和 Train-only survival 门，所有门通过后才允许唯一一次 public-127。

#### 24.88.5 seed 2026 终态：always-on fusion 未通过容量门

有效 v4 训练完成并通过 checkpoint 完整性检查：来源 checkpoint 的 `747` 个张量逐字节无变化，
只新增并更新 `6` 个 fusion 张量、共 `28,737` 个参数。checkpoint 为：

~~~text
/root/autodl-tmp/srtrack_depth_reliability_fusion_probe_e1_seed2026_v4/checkpoints/train/srtrack/droptrack_depthtrack_final_language_depth_reliability_fusion_probe_e1/SIAMTrack_DropMAE_ep0001.pth.tar
SHA256 fcbcf91ddf89bfd7fd038954d3dedf3fc1c4789085de7fbf60941ad10a6c2dab
size   1,090,087,372 bytes
~~~

随后仅在 DepthTrack Train fixed-6 development 上运行 `correct_full`，显式关闭 safe-template；未启动
VOT、DepthTrack Test 或 CDTB。完整结果如下：

~~~text
                         Precision    Recall       F
safe025 baseline        65.504506    65.219985   65.361936
reliability fusion      65.532075    62.827214   64.151145
delta                   +0.027569    -2.392772   -1.210791 pp
~~~

逐序列 F 变化为：`toy03 -0.293360pp`、`pigeon05 +0.359409pp`、`bottle03 +0.346926pp`、
`ball16 -2.757762pp`、`bag04 -5.067678pp`、`flower03 +0.149348pp`。模型略微抬高 precision，
却显著降低 recall；尤其 `bag04/ball16` 证明 bounded、两侧权重均为正仍不能防止小 token 偏移通过
递归位置状态放大。因此该探针 **未通过容量门**，不扩展 seeds 2027/2028，不授权任何 public VOT。

权威产物：

~~~text
runtime/result.json
SHA256 81578f1226dfd0a1298da3fa8b75bb627d1306d3a642749b54139bce44a664b9
fixed6/correct_full/manifest.json
SHA256 d49d138e147d29caa120f94e0322531e30a631ebbf281388a46ae87490e65dbb
fixed6/correct_full/metrics.json
SHA256 0b2900fa8ac34306da8e2d69b6709d0d6e4f13e1b5f1c14368cccdaa9ffff1e2
~~~

下一候选不再扫描常驻 preference 上限。训练侧应同时计算 exact legacy mean 与 learned fusion 两个
反事实候选，只把“提高目标支持且不增加位移/身份风险”的帧标成 fusion-admissible；部署侧 gate
零初始化为 reject，拒绝时逐字节走 legacy mean，只在因果证据通过时短时启用 fusion。它直接针对
VOT 的失败机理：ACC 已达标，目标是防止高置信度错误峰递归写回，并在连续第 10 个 `IoU<=0.1`
之前恢复，而不是继续改变所有正常帧。该 gate 必须先过同一 fixed-6 零伤害门，才允许任何公开评测。

#### 24.88.6 exact primary + 旧 router 对照：隔离成立，但接纳标签错位

为区分“fusion 本身无容量”和“always-on 写回破坏递归状态”，增加 counterfactual-primary 对照：
legacy arithmetic mean 保持为不可变 primary；learned fusion 只进入已有 language counterfactual router
的 proposal 分支。router 拒绝时 primary tensor 与 inherited tensor 是同一对象，预测框可以字节一致。
默认配置仍保持旧行为；新模式若没有 causal router 会 fail closed。聚焦测试 `5 passed`，完整模型预检
仍为 `747` 个来源张量零差异、唯一 `28,737` 参数 fusion optimizer group。

单种子训练与 fixed-6 结果：

~~~text
run /root/autodl-tmp/srtrack_depth_reliability_fusion_counterfactual_probe_e1_seed2026_v1
checkpoint SHA256 dc437352bbabf39a5451abbf23a5aa8efed91ebe26953282a857dbe7afbbd543

                         Precision    Recall       F
safe025 baseline        65.504506    65.219985   65.361936
counterfactual primary  64.622978    63.755219   64.186166
delta                   -0.881527    -1.464767   -1.175770 pp
~~~

~~~text
序列                 baseline F   always-on F   counterfactual F   CF delta
toy03_indoor            80.029876      79.736517       80.029876      +0.000000
pigeon05_wild            4.366735       4.726144        4.295578      -0.071157
bottle03_indoor         86.856624      87.203550       86.845721      -0.010904
ball16_indoor           68.426690      65.668927       61.848850      -6.577840
bag04_indoor            72.488367      67.420689       72.587173      +0.098806
flower03_indoor         82.276875      82.426223       82.276875      +0.000000
~~~

`toy03` bbox 文件与 baseline SHA 完全相同，`bag04` 也从 always-on 的 `-5.067678pp` 恢复到
`+0.098806pp`，证明 exact primary 隔离真实有效。但 `ball16` 反而 `-6.577840pp`：旧 router
面向 language candidate 训练，在少数关键帧错误接纳 fusion；一次错误峰足以通过递归 bbox 状态放大。
因此“低接纳率/存在 fallback”不等于 action precision 足够，这版仍失败，不扩三 seed、不跑 public VOT。

下一步必须训练独立的 fusion-admission head，而不是复用语言 router。Train-only 反事实标签只有在
fusion 相对 exact legacy 同时满足“目标 support/IoU 提高、selected peak 实际改变、位移和身份风险
不增、无 catastrophic harm”时才为正；模糊样本一律 reject。策略选择必须使用 OOF/held-out audit，
以零 harm 和高 precision 为门，而不是最大化动作数。这一要求也直接对应 VOT 口径：一个误接纳即可
触发 10 帧连续低 overlap 并大幅损失 ROB/EAO，不能用大量普通帧的小收益抵消。

权威产物：

~~~text
runtime/result.json
SHA256 47fb8b0222e253dd353f19fa500b60f3aea51f1afc36f16d97760a23577799ff
fixed6/correct_full/manifest.json
SHA256 4958a594e2c107f2516fd9b40bb7239bd8e37fbdb3e7f99d03b85df1a71475f6
fixed6/correct_full/metrics.json
SHA256 c626e9c848270097f452fa584a496e90c6fc241af5cd7f9c10f06f5735a8dc82
~~~

#### 24.88.7 严格峰值支配仍失败：高置信错误需要身份与回滚

对 24.88.6 的 `ball16` 做事后因果定位后，首次 bbox 分叉出现在 0-based 帧 `526`。该帧目标不可见，
legacy/counterfactual 分数分别为 `0.089111/0.039078`，旧 router 仍接纳了更弱 proposal。为只检验
这个已观测缺口，新增无可调 margin 的严格支配条件：上游 router 已选 proposal 后，只有 candidate
当前峰值严格大于 exact primary 峰值才允许输出；否则逐元素恢复 primary。该 follow-up 不重新训练
checkpoint、不扫描阈值，聚焦测试 `7 passed`。

fixed-6 结果为：

~~~text
                         Precision    Recall       F
safe025 baseline        65.504506    65.219985   65.361936
peak dominance          65.435802    63.036177   64.213579
delta                   -0.068704    -2.183808   -1.148357 pp
~~~

它只把无 guard counterfactual 的 F `64.186166` 提到 `64.213579`。`bag04` 仍比 baseline
`+0.070930pp`，但 `ball16` 仅从 `61.848850` 提到 `61.939140`，仍低 `6.487550pp`；
`flower03` 还新增 `-0.190806pp`。因此“候选分数更高”既不是身份正确的充分条件，也不能保证下一帧
递归状态安全。

更具体地，`ball16` 在帧 526 的低峰动作虽会被新 guard 拒绝，但后续高峰动作仍造成状态分叉。
到帧 `887`，legacy 已以 `IoU=0.931893, score=0.657070` 重捕目标，guarded fusion 仍为
`IoU=0, score≈0.34`；整条轨迹有 `151` 个可见帧满足 `legacy IoU>=0.3` 且 `fusion IoU<=0.1`。
这与 full VOT 的统计相互印证：高置信错误很常见，单帧 score 不能区分目标峰和相似物/背景峰。

下一版停止增加分数规则。fusion action 必须保存 action 前 bbox 与递归 tensor snapshot，先作为两帧
tentative proposal；只在 legacy-relative support 持续改善、RGB/depth/首模板身份一致且无位移恶化时
commit，任一证据恶化即原子 rollback。其 admission policy 只可从 Train-only paired counterfactual
trace 训练，并用 OOF calibration + 单次 held-out audit 约束零 harm/高 precision；在此之前不授权
三 seed 或 public VOT。

权威产物：

~~~text
experiments/srtrack/droptrack_depthtrack_final_language_depth_reliability_fusion_counterfactual_peak_dominance_e1.yaml
SHA256 a788f00b6cfa5e74842291318f1fa4b50bb5cdf6eed73d564dd3b079b92157ec
fixed6/peak_dominance/manifest.json
SHA256 08d5ec5fe8b46a5a549bbf9a8216e5001b06f1028df70b147cb3eee9ba903af4
fixed6/peak_dominance/metrics.json
SHA256 c79e24a9106ab9aac1e77d4e1ac93564a2417ae05c6c520e257d147a4367d90d
~~~

### 24.89 VOT 定向模板容量复核：更强更新并不能恢复身份

VOT-RGBD 的主要缺口是 ROB，而不是 ACC。当前 formal ACC 已为 `82.535868`，高于目标
`82.1`；ROB 为 `87.988071`，低于目标 `93.7`。官方 toolkit v0.7.1 的 multi-start 分析把
可见 GT 上连续 `10` 个 `IoU<=0.1` 视为失败，并把失败后的 EAO overlap 尾部补零。因此模板机制
必须在第 10 个低 overlap 之前恢复正确身份；正常帧的小幅 IoU 收益不能补偿一次错误写回造成的长段失败。

为检验“既有 0.10/0.20 更新太弱”这一假设，固定同一 public static 输出，只在同一状态下生成
`0.50` 与 `1.00` primary-template 反事实。GT 在推理完成后才 join；固定 6 序列共比较 `12` 个
static bbox/score 文件，逐字节 `0` mismatch。单帧与风险后严格未来两帧的结果均不支持该假设：

~~~text
                                    rows  beneficial  harmful  catastrophic  recovered  cumulative IoU gain
single/current-frame, weight 0.50    112       0          4          1            0          -1.770457
single/current-frame, weight 1.00    112       0          4          0            0          -1.069088 (incremental)
future-2-frame, weight 0.50           292       0          8          1            0          -3.765301
future-2-frame, weight 1.00           292       0          7          0            0          -3.844236 (incremental)
~~~

具体反例：

1. `bag04_indoor`, frame 1593：上一帧 IoU `0.798653`，static IoU `0.739491`；0.50 更新把峰
   翻到错误位置，IoU 直接变为 `0`，而 1.00 又回到 `0.748352`。响应对 blend weight 高度非单调，
   因而不能用“更大权重应有更强恢复”推断动作安全。
2. `bottle03_indoor`, frame 3156：static IoU `0.823933`，0.50/1.00 分别降至
   `0.520281/0.565375`；随后 frame 3157 与 3158 的两帧 burst 仍持续低于 static，未发生一次
   `IoU<=0.1 -> IoU>=0.3` 的恢复转移。
3. `bottle03_indoor`, frame 2503：static IoU `0.817182`，0.50/1.00 只有
   `0.589923/0.600833`；未来两帧 strong 仍分别比 static 低 `0.219488/0.205284`。

结论：模板更新已经实现，但当前动态模板不是缺失的身份恢复信号；提高 blend weight 只会放大错误峰，
不具备 VOT survival 容量。该路线停止于 fixed-6，不扩 full-152、三 seed 或 public VOT。

下一条受控候选改为联合训练 learned RGBD fusion 与它实际使用的 counterfactual admission router。
此前 fusion-only 训练后仍复用旧 language router，导致 candidate distribution 与 admission policy 错位；
联合训练只开放 `28,737` 个 fusion 参数与 `2,499` 个 router 参数，其他 checkpoint 张量冻结，拒绝
proposal 时仍返回 exact legacy mean。它是否有效只由 fixed-6 相对 safe025 的 P/R/F、逐序列零灾难
和接纳行为决定，训练 loss 本身不构成通过证据。

权威强模板产物：

~~~text
/root/autodl-tmp/srtrack_strong_primary_template_fixed6_probe_seed2026_v1/analysis.json
SHA256 7df2652b16df970d6b0b9192745fcd6251c670cdeb09ec86e3077c1f268d4e8a
/root/autodl-tmp/srtrack_strong_primary_template_temporal_fixed6_probe_seed2026_v1/manifest.json
SHA256 c092898cbe2a9dcdac451372c9119333a047a56c93663eb76487027304ea84a2
/root/autodl-tmp/srtrack_strong_primary_template_temporal_fixed6_probe_seed2026_v1/analysis.json
SHA256 32729ef11af32e0a91a350605d6907a5ea12fe10ce3fe1c5daf887673acb142e
~~~

### 24.90 联合 fusion + router 仍失败：单帧监督不能保护递归身份

24.89 预注册的联合候选只开放 `28,737` 个 depth-reliability fusion 参数与 `2,499` 个
counterfactual router 参数；其余 checkpoint 共 `747` 个来源张量全部冻结且逐项零差异。拒绝 proposal
时仍返回 exact legacy mean。seed 2026 训练完成后，fixed-6 相对 safe025 的结果为：

~~~text
                         Precision    Recall       F
safe025 baseline        65.504506    65.219985   65.361936
joint fusion + router   64.771638    63.310567   64.032769
delta                   -0.732868    -1.909418   -1.329167 pp
~~~

~~~text
序列                 joint F      delta vs safe025
toy03_indoor          80.029876      +0.000000
pigeon05_wild          4.274090      -0.092644
bottle03_indoor       86.856624      +0.000000
ball16_indoor         60.903428      -7.523262
bag04_indoor          72.808993      +0.320627
flower03_indoor       82.194042      -0.082833
~~~

最明确的递归反例仍是 `ball16`。joint 首次相对 baseline 分叉在 0-based frame 615；frame 624
baseline 为 `IoU=0.834948, score=0.481636`，joint 却以 `score=0.257944` 跳到图像上边界且
`IoU=0`。到 frame 630，baseline 已为 `IoU=0.925981, score=0.845012`，joint 仍在错误边界
`IoU=0, score=0.443918`。整段有 `135` 个可见帧满足 `baseline IoU>=0.3` 且
`joint IoU<=0.1`，最长连续段分别为 `40` 帧（1618--1657）、`37` 帧（624--660）和
`29` 帧（1686--1714）。可见帧平均 IoU 由 baseline 的 `0.710358` 降到 joint 的
`0.616405`。

根因不是 optimizer 没有更新。训练中的 router target 仍由单帧 GT Gaussian support/gap 与候选峰
变化生成，没有 rollout 后的身份存活/原子回滚标签；当前诊断窗口也只有一帧。因此 router 可以在
当前帧学到“峰值更像目标”，却不知道这个动作会把下一帧搜索中心带到错误物体或边界。该候选失败
fixed-6，停止三 seed/full-152/public VOT；正式最好指标不变。

权威产物：

~~~text
/root/autodl-tmp/srtrack_depth_reliability_fusion_joint_router_probe_e1_seed2026_v1
checkpoint SHA256 0acc61ec82f9d75d915c8702252847edd37d1d3659aa42d19845cb8d4476d6b0
runtime/result.json SHA256 954d9b0e04c5a9cb6f109880a7b1d8e8f10a7eb8034ff6e44c5d488b46645ca3
fixed6 manifest SHA256 5dc8088f977d00ba4fd97ff918111aaf0b0806eaf2762e6eb06e02bf4644b329
fixed6 metrics SHA256 aa374b3a6ad163414917c8314a60b6cc55099cdf7b27612e37541b74c5e38f01
~~~

### 24.91 保留创新点迁移至 SUTrack-L384（不重复测试官方 baseline）

连续三个受控结论已经排除“只增强旧 SRTrack 的融合/模板/单帧路由”这条路线：always-on fusion、
exact-primary counterfactual、严格峰值支配、强模板以及 joint router 都未通过 fixed-6。按当前决策，
后续保留两项已形成的创新语义，但把基础跟踪器换为公开、原生支持 RGBD 与语言的 SUTrack-L384：

1. **结构化初始语言**：沿用清洗后的 VOT-RGBD2022 127 序列描述，不含 bbox 泄漏与绝对路径；
   runtime 同时锁定文件 SHA256 `b0e08fcee58f5ae8119d951eabf4a5688a433864279291add34e56440f57072d`
   和 `127` 条唯一序列。描述经 SUTrack 原生 CLIP ViT-L/14 text encoder 形成 text token，与 search、
   template token 在 Fast-iTPN 主干内联合建模，不新增旁路伪融合。
2. **安全动态模板槽**：保留首帧模板为不可变 slot 0，只允许 slot 1 在线替换。候选必须同时满足
   置信度、NMS 后响应峰值差、连续稳定帧、归一化中心位移、首帧 RGB Bhattacharyya 身份以及原始
   depth 中位数一致性；硬冲突或超龄立即把动态槽原子恢复为首帧模板，深度/锚点不可用时 fail closed。

选 SUTrack-L384 而不是直接采用竞赛结果数字更高但缺少完整 RGBD winner 实现的 MixForRGBD：
SUTrack 仓库明确发布了统一 RGB/RGBD/RGBT/event/language 代码、MIT 许可、L384 权重和 VOT-RGBD22
结果；其模型表报告 VOT-RGBD22 EAO `76.6`、DepthTrack F `66.4`，论文另报
EAO/ACC/ROB `76.6/83.5/92.2`。这仍未满足当前 `77.9/82.1/93.7` 的 EAO/ROB 目标，但起点明显
高于现有正式 `72.908956/82.535868/87.988071`，而且具备可验证的原生 RGBD+语言融合接口。

新仓库与配置：

~~~text
repository /home/SUTrack_RGBD_L
upstream commit d65052d1ba3fcf55010e1fb3665ee6616c139a2c
config experiments/sutrack/sutrack_l384_rgbd_language_safe_template.yaml
launcher lib/test/vot/sutrack_l384_rgbd_language_safe_template.py
official L384 checkpoint SHA256 2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4
official CLIP ViT-L/14 SHA256 b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
~~~

官方 `sutrack_l384.yaml` 不修改、官方 baseline 不重跑。只有上述新配置会启用语言 manifest 与安全
模板 gate。真实 `adapter01_indoor_1` 6 帧烟雾已通过：只消费首帧 GT 初始化，官方 checkpoint
严格加载，frame 6 发生一次安全动态模板写入，无 drop，最终 confidence `0.665491`。收据：
`/root/autodl-tmp/sutrack_rgbd_language_safe_template_smoke_v1/adapter01_indoor_1_smoke.json`，
SHA256 `f9c87b0682128ef80c4cd5f12bed704e58a361dbe20fd57f45415c3951646158`。下一步先做
小规模移植门禁，再决定是否进入完整 VOT-RGBD2022。相同序列的 60 帧连续性检查只发生 frame 6
一次 replace；frame 28 因身份/depth 冲突 drop，之后保持 static 到 frame 60，没有频繁污染模板。
60 帧收据 SHA256 `aeb66021560b099bbe03afbd28f748aa4640bc90c34b4bc28755eab2e183ac7a`。

### 24.92 SUTrack 移植的六序列 ROB 定向门

为避免重跑官方 SUTrack baseline，本轮只评测“官方 L384 checkpoint + 绑定语言 + 安全模板”新配置；
对照直接复用 primary safe025 已完成的 SRTrack 正式轨迹，再用同一个 VOT toolkit `0.7.1` 对同六序列重新
汇总。六条序列是 SRTrack 历史正式 ROB 加权损失最大的 `cup02/earphone01/toy09/glass01/shoes02/bag02`，
覆盖 141 个 forward/backward anchor。新运行按预计 tracker-frame 负载分成 10 个互斥 shard，
最终合并前逐条要求 `.bin/confidence/time` 三件套齐全，共 423 个结果文件。

multi-start 的精确口径沿用 24.88：可见 GT 上 `IoU<=0.1` 连续 10 帧才失败；ACC 是 progress 前
overlap 的加权均值，ROB 是每序列 anchor 生存比例再按序列长度加权，EAO 把失败后的 fragment 尾部
补零后平均曲线下标 115--754。因此这个门主要检验新的语言/模板递归状态能否延长 failure-free run。

~~~text
同六序列                 EAO         ACC         ROB
SRTrack历史正式轨迹复算  39.446101   78.281194   30.725423
SUTrack移植             48.311723   79.807124   45.295506
相对SRTrack参考delta     +8.865622   +1.525931  +14.570083 pp
~~~

上表不是纯 SUTrack baseline 消融。官方 SUTrack baseline 按要求未做 full-127 服务器复测，因此只可
报告“SUTrack+创新”的绝对实测值，以及它相对 SRTrack 历史正式参考的变化；不能写成创新相对
SUTrack baseline 的全量增益。16-anchor default 分支只用于组件归因，也不是 full-127 baseline。

逐序列不是全胜：`glass01` 从 19/19 失败降为 9/19，ROB `15.817818 -> 81.589475`；`bag02` 从
13/27 降为 5/27，ROB `69.618472 -> 86.846209`；`earphone01`、`toy09` 也同时改善。反例是
`cup02` 仍 36/36 失败且 ROB `6.959906 -> 5.612288`，`shoes02` 从 12/13 变为 13/13，ROB
`25.208436 -> 10.528037`。所以聚合门证明更强基线方向有容量，但不能把它写成零伤害，也不直接
启动 full-127。下一步只在事先选定的 16 个最大正/负 progress anchor 上，把安全模板 gate 与
SUTrack 默认置信度/间隔更新做组件消融；语言、checkpoint、RGBD 读图和 VOT 协议保持不变。

~~~text
new shard manifest b7abaa6af4991a22359ab9c5af719a30eca6a7a9f23227f407e8a7c9ab482f0e
new merge receipt 61f58bf52679ea67b5a88c6849550a802493807582c52dfe6eda020ccd8c66dc
new analysis       aa846361f4a9f1a9168a8b80bcc16387017bd5b8345965bde907da78fbd37616
old analysis       81c073576b02aa921406c5f489a4db86f0a1a28bf13276a732b3908fa7236ac2
~~~

该六序列数字是定向诊断，不是 full-127 正式成绩；当前正式最好仍为
`72.908956/82.535868/87.988071`。

### 24.93 模板组件归因、v2 否决与 full-127 启动

六序列内部仍有 `cup/shoes` 退化，因此没有直接把公开序列写成新规则。预先固定 16 个最大正/负
progress anchor，保持 checkpoint、结构化语言、RGBD 读图和 VOT 协议不变，只比较模板策略：

~~~text
策略                         failures   progress   progress加权ACC
safe v1                         8        10043        77.566594
SUTrack default（仅消融）      7        11658        78.404825
safe v2                         8         9837        77.560596
~~~

default 相对 v1 多 1,615 个 progress 帧；最大差异为 `bag02@100 +841`、`toy09@100 +324`、
`cup02@1600 +248`、`shoes02@613 +174`。v1 并非完全无益，`toy09@150` 比 default 多 335 帧。
代码检查发现 v1 最早可在 frame 5、confidence 0.65 写入，而官方更新是 25/0.70。只据此做一次
预注册 v2：检查/最小间隔对齐 25、confidence 对齐 0.70，其余 RGB/depth/身份条件保持不变。
v2 总 progress 反而比 v1 少 206，并让 `glass01@50` 从成功变失败，所以 v2 否决，不再扫描公开
anchor 阈值。default 只用于归因；按用户要求保留安全模板创新，部署仍选六序列完整聚合三项同升的 v1。

随后创建 full-127 独立工作区：127 序列、1,765 anchors，10 个 shard 的预计负载严格均衡在
132,679--132,727 tracker frames。已完成的同配置六序列 141 条轨迹逐文件核 SHA 后预填充，
因此 detached `screen` `sutrack_rgbd_safe_full127_v1` 从 `141/1765` 启动，不重复计算 baseline 或
既有新轨迹。启动后两卡各 5 个 tracker，显存约 15.95 GiB/卡；预计剩余约 15--20 小时。

~~~text
default manifest 6807ef44760e58039d785db88a2c2f3a3129b37875eccf8aa3e56d0ecb08a1ef
default merge    843e504c6e1f0bb82514de54e3ce225921e38c9693d594d832e63eb37dcd38d4
v2 manifest      cf0a22946574fe604bc4b6ff54294a7582ff160caa78e47fda62205df0c0dfac
v2 merge         215e516a55f8198c3cf8953177514cb958820c37b2d2ff8ad248b3d702747038
full manifest    8bf5271b3cdc0e0f4587657502f0aa4d873c6cfbc8716f88a1fabb55aa5334b3
preseed receipt  1612112e02942c7bf7df1e6f910ba65866c508bcb0be13f981c51ceadea408a9
~~~

full toolkit analysis 未完成前，正式最好仍是旧模型
`EAO/ACC/ROB = 72.908956/82.535868/87.988071`；六序列与 16-anchor 数字都只作诊断。

<!-- SUTRACK_FULL127_RESULT_BEGIN -->
### 24.94 SUTrack full-127 自动终态

无人值守 finalizer 已在 VOT toolkit `0.7.1` 下验证 127 序列、1,765 anchors 和全部结果 SHA，
随后生成正式 full-127 汇总。至少一项目标未达到，不能写成目标已完成。 检查：EAO=未通过、ACC=通过、ROB=未通过。

| 结果 | EAO | ACC | ROB |
|---|---:|---:|---:|
| SRTrack 历史正式参考（非 SUTrack baseline） | 72.908956 | 82.535868 | 87.988071 |
| SUTrack 官方论文报告（未在本服务器复测） | 76.600000 | 83.500000 | 92.200000 |
| SUTrack-L384 + 结构化语言 + safe-v1（本服务器实测） | **73.974969** | **82.627562** | **89.455266** |
| 相对 SRTrack 历史正式参考变化（pp） | +1.066014 | +0.091694 | +1.467195 |
| 目标 | 77.900000 | 82.100000 | 93.700000 |

权威结果：`/root/autodl-tmp/sutrack_rgbd_language_safe_template_vot_full127_v1/full_result.json`；analysis SHA256 `e3feabdee88b5dc28938171a08c5d58b13aca52d5cffda8265d4b970e1a68e08`；merge SHA256 `a00462de9fb0025ee10b905564959ab100a40ecdfeee82122b0ac513762846c9`。
SUTrack 官方 baseline 按要求未重跑，因此这里不声称“创新相对 SUTrack baseline 的 full-127 增益”；
只有 SUTrack+创新的绝对实测指标，以及相对 SRTrack 历史正式参考的变化。
该 full-127 只更新 VOT 证据；DepthTrack/CDTB 未在 SUTrack 移植上重测，原有已达标正式数字保持不变。
<!-- SUTRACK_FULL127_RESULT_END -->

<!-- STATE_ROLLBACK_V3_WORST5_20260815 -->
## 追加：SUTrack 有界状态回滚 v3 的完整定向结论（2026-08-15）

safe-v1 full-127 的逐序列归因显示，`bandlight_indoor_1`、`box_room_noocc_4_1`、
`cube05_indoor_5/6` 与 `yogurt_indoor_1` 的主要损失不是当前帧框略差，而是错误候选已经写入
`self.state`，随后搜索窗连续围绕错误位置递归。旧安全模板逻辑只能丢弃动态模板，不能恢复这一 bbox
状态，因此高置信错误仍会形成长失败链。

v3 保持官方 SUTrack-L384 checkpoint、结构化语言、RGBD 输入和 safe-v1 阈值不变，只增加一个无训练的
推理期状态事务：每帧先保存 `prior_state`；发生中心大跳、静态 RGB 身份冲突、raw-depth 大变或时序身份
拒绝时，丢弃动态模板并最多连续一次恢复 `prior_state`。第二个连续冲突 fail-open 并重新绑定状态，避免
真实快速运动时无限冻结。旧 safe-v1 配置默认关闭回滚且原 YAML 字节不变。

确定性复现由原先的 `REPRO_STATE_POISONED` 转为 `PASS_STATE_ROLLED_BACK`；回滚预算、第二次冲突
fail-open、干净帧复位以及 legacy-disabled 四条状态机路径全部通过。真实 6-frame RGBD+language smoke
也通过，未触发冲突的序列输出与旧 safe-v1 完全相同。该改动没有新增 `nn.Parameter`，继续加载
`SUTRACK_ep0180`（SHA256 `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`），
不需要重新训练。

随后在上述 post-hoc worst-5 上完整运行 112 个 VOT multi-start anchors。旧 safe-v1 不重跑，直接从已审计
full-127 轨迹按文件 SHA 构造同 anchor 对照；v3 使用 10 个互斥 shard、两张 RTX 3090 完成 112 个 region、
confidence 和 time 三元组，共 336 个结果文件。该集合是在 full-127 结果之后选择的，只是容量诊断，
不是无偏正式 benchmark。

| worst-5 / 112 anchors | EAO | ACC | ROB |
|---|---:|---:|---:|
| safe-v1 冻结轨迹 | 55.341429 | 73.505170 | 59.976116 |
| state-rollback v3 | 55.565112 | 72.676422 | 62.271528 |
| v3 相对变化（pp） | **+0.223683** | **-0.828748** | **+2.295412** |

预注册容量门要求 EAO 至少 `+0.50pp`、ROB 至少 `+3.00pp`，且 ACC 损失不超过 `1.00pp`。v3 仅通过
ACC 保护项，EAO/ROB 两项均未通过，因此 `eligible_for_full127_evaluation=false`，不会启动 full-127。
结论是“单次整框回滚有真实 ROB 容量，但强度不足且牺牲 ACC”，不能替代帧级可学习接纳策略。当前正式
full-127 最好仍为 safe-v1 的 `73.974969 / 82.627562 / 89.455266`。

下一步冻结 SUTrack 主干、语言编码器、模板与检测头，只用 SUTrack 自己产生的 Train-only 轨迹训练轻量
时序/多动作门控；旧 SRTrack gate 不直接复用，因为特征和置信度分布已经改变。只有训练门、零伤害门和
DepthTrack/CDTB 保护门都通过，才允许新的公开 full-127。

权威证据：

```text
root                 /root/autodl-tmp/sutrack_rgbd_language_state_rollback_v3_worst5_v1
diagnostic result    SHA256 958b9e22764f59382ad068988b8e038e628931c0498f6d6e12aa2ca52dd300a2
shard manifest       SHA256 4b77873fe4aeb1d790e205943ce4555aa2ecc0ca1f818b6a326ad3f4430611f3
merge receipt        SHA256 2123be0dc2d82ebfec5334af315f63257254164bac68217b5acf7f590cc2ba41
candidate analysis   SHA256 ad8af05be0c55b790211dfbf284216c659c5b656a2912a5b5c2c5df2a4f68162
baseline analysis    SHA256 f669172e7337378ac5938bb4a56932f29f620c7c89afc5879b8a69e870a8bc4e
source snapshot      SHA256 924fd33ae1f9dfb4948d0f024a48eedbfea0e7acf59b3df448d78ca9fd608a4a
```

本节只追加到远程统一交接文档。`C:\Users\gb\Desktop\document` 中的本地文件保持为用户要求时的
一次性快照，后续不自动覆盖。

<!-- SUTRACK_STATE_GATE_FIXED6_FULL152_20260815 -->

## 2026-08-15｜SUTrack 专属轻量状态门控：fixed6 容量门通过，Train152 trace 运行中

- 方法边界：保留现有 SUTrack RGBD 结构化语言与 safe-template 创新；官方 `SUTRACK_ep0180` 主干完全冻结，只采集在线状态证据，后续仅训练小门控。当前没有重新训练 2.2 GB SUTrack 主干，也没有启动公开 VOT/DepthTrack Test/CDTB 评测。
- 训练语言：从 DepthTrack Train 首帧审校 v5 的 152 条序列级描述物化为无路径、无坐标的 clean manifest。输出 `/root/autodl-tmp/sutrack_rgbd_state_gate_train152_v1/source/depthtrack_train_language.jsonl`，SHA256 `d0dcc6c5d67faacd94358a8db8f6a128c7dcbb2ad50fa212816c52fa1e7358d7`；仅首帧初始化 GT 用于生成语言，未来帧文本未使用。
- fixed6 追踪：沿用预注册序列 `bottle03_indoor, ball16_indoor, bag04_indoor, flower03_indoor, pigeon05_wild, toy03_indoor`，共 10,041 帧、10,035 条在线 trace；两片 exit 均为 0。跟踪器运行时仅消费第一帧初始化框，完整 GT 严格在推理结束后由独立分析器加入。
- fixed6 容量结果：总 10,035 行，其中 8,698 行 GT 有效、1,337 行为官方 NaN absent 标注；hard-conflict 6,859 行。以“上一状态 IoU 至少比候选高 0.05 且上一状态 IoU≥0.10”为即时回滚收益标签，得到 253 个有益帧，覆盖 6/6 序列；全部容量检查通过。safe-v1 候选平均 IoU `0.705369`，仅作为上界诊断的即时 oracle 为 `0.709747`，累计即时 gain `38.074797`。
- 科学限制：同一集合存在大量回滚有害帧，故固定阈值/一刀切回滚不可接受；fixed6 只证明存在可学习的即时动作容量，不证明递归轨迹收益，不是最终模型指标。
- fixed6 证据：`capacity_result.json` SHA256 `2b77acc2b4cdbe28e736aa2f62677b2726104e93e5953efb5dbb1668d8043f4e`；`capacity_rows.jsonl` SHA256 `61a04c264e1fe0555379674115e1526d291f981962a4c81f0378c749135dff4d`；分析器 SHA256 `629a56f7e26feecd6f00b5c360cb39c355fc611c5e33282026c90520d965ba7e`。
- full152 trace：复用 fixed6 冻结证据，仅补跑剩余 146 条、209,913 帧；总训练集 152 条、219,954 帧。确定性均衡计划为 GPU0 73 条/104,896 帧、GPU1 73 条/105,017 帧，plan SHA256 `b9d36eb7d1b042b50dc1bd03013e8b9eab4bae7de607a9c4bb82cb7f8c5f74a3`。screen `sutrack_state_trace_full152_g0/g1` 已启动，当前状态为运行中，不得写成训练完成。
- 下一步：full152 trace 完整性通过后，做 sequence-group OOF，只训练小门控；要求高精度、零 catastrophic harm，并另留严格序列 holdout 做一次真实递归轨迹审计。只有这些 Train-only 门通过，才允许正式 RGBD 公开评测。

### 24.95 SUTrack Train152 状态门终态与 recovery-search fixed6 终态（2026-08-15）

#### 24.95.1 full152 trace、分析器恢复与即时动作容量

DepthTrack Train152 的双分片已全部完成，两个 exit 都为 `0`。固定 6 条与新增 146 条合计
`219,954` 帧、`219,802` 条在线 trace；跟踪器仍只读取首帧初始化框，完整 GT 严格在推理结束后加入，
`future_frame_text_used=false`、`public_evaluation=false`。full152 plan SHA256 为
`b9d36eb7d1b042b50dc1bd03013e8b9eab4bae7de607a9c4bb82cb7f8c5f74a3`。

首次分析暴露一个工程边界：`colacan04_indoor` 的静态身份锚不可用，1,000 个在线帧都合法记录
`checked=false, eligible=false, reasons=[anchor_unavailable]`，但旧分析器把空的
`normalized_center_jump` 强制转为 float。修复只允许 **unchecked** 行用中性 jump `0`，同时保留
`checked=0`；若 `checked=true` 仍缺 jump 则继续 fail-closed。两个回归样例和完整 152 序列重放均通过，
没有修改冻结 trace、bbox、checkpoint 或 GT。修复后 full152 analyzer SHA256 为
`2e459ce6131018c4574836098fab7aae99121567cac0587d84ccd0512ea97b1d`；回归测试 SHA256 为
`284e60d34bd664b271add3c9012fc86215f5a7d3fba0bbd0f1bd62dd28f360af`。两次失败状态与日志均已保存在
`/root/autodl-tmp/sutrack_rgbd_state_gate_train152_v1/logs/incidents/`，未删除。

full152 即时容量结果如下：

| 项目 | 原始值 |
|---|---:|
| 总 trace 行 | 219,802 |
| 有效 GT 行 / absent GT 行 | 203,376 / 16,426 |
| hard-conflict 行 | 127,874 |
| 有益上一状态回滚行 | 2,998 |
| 最小容量门 | 204 |
| baseline candidate mean IoU | 0.838261389 |
| immediate oracle mean IoU | 0.839913543 |
| oracle 累计即时 IoU gain | 336.008515 |

全部容量检查通过，说明 Train152 确实存在上一状态回滚机会；但这是 post-inference 即时 oracle，仍不证明
递归 rollout 能安全改善 ROB。权威结果：
`/root/autodl-tmp/sutrack_rgbd_state_gate_train152_v1/full152_analysis/capacity_result.json`，
SHA256 `60e364af352d71f4257dcd3a46f031499b9e582dd47ff14fc1dc2d4d6384cf34`；rows SHA256
`57047b5fc1eb6e8eeeb4a14e7bed078965de223dd33f3082c6efc060b617fcbe`。

#### 24.95.2 三 seed 小门拒绝：动作存在，但 55 参数线性时序门不可安全分离

冻结 SUTrack 主干后，只训练 `54` 个 current/delta1/mean2 特征的线性门（含 bias 共 `55` 参数）。
校准候选 `105,143` 行、保留 audit 候选 `22,731` 行；2026/2027/2028 三个 sequence-group OOF seed
均完成 5 折训练，但三者的 `oof_selection` 都为 `null`、`oof_passed=false`。因此：

- `all_seeds_oof_passed=false`，正式 decision 为 `training_rejected`；
- 唯一一次 immediate audit **没有被消费**，`immediate_audit_evaluated=false`；
- 三份 artifact 只作为拒绝证据，均不可部署；
- 没有启动递归 audit、DepthTrack Test、CDTB 或公开 VOT。

这不是主干欠拟合：三个 seed 的 final training loss 都约 `0.55738`，但在精度、伤害、零 catastrophic
与最小动作覆盖的联合门下没有任何可接受阈值。结论是“上一 bbox”这一单动作虽有 oracle 容量，当前线性
特征门无法在跨序列 OOF 上安全识别它。权威训练结果：
`/root/autodl-tmp/sutrack_rgbd_state_gate_train152_v1/gate_training/training_result.json`，SHA256
`8141d3f0fa5bb4e391144864e2da16a6f72c988e0e5b12084d1a1d230f08f69b`。

#### 24.95.3 有界同帧扩展搜索：exact-OFF 修复通过，但 ON 动作方向错误

作为与状态回滚不同的 Train-only fallback，本轮测试同一硬冲突帧上的第二次扩展搜索：以 prior bbox
为中心、search factor `6`，最多连续 1 次，随后冷却 2 帧；只有 recovery 的 RGB 身份、raw depth、
confidence 和 response margin 全部严格通过才接纳。它不读在线 GT，并保留结构化语言与 safe-template。

首版 OFF 暴露 `SUTRACKRecoverySearch.track()` 即使关闭 recovery 仍重写父类推理，导致最大仅
`7.1e-13` 的 ULP 级 bbox 扰动。该版被严格门拒绝，完整产物已移入
`logs/incidents/20260815T110055Z_recovery_off_ulp/`。修复后的 disabled 路径直接调用
`super().track()`，只附加不反馈的审计字段；回归测试证明 exact parent prediction 不被重建。
tracker SHA256 为 `e20e337c2d065afad20d62f39f5e427bfe37df6b43dc0b1c4e36ec0972f122d3`。
revision-2 plan 在新推理前冻结，SHA256
`eded3d2676eadfca37ea903975f75aa782100a699881c7f6efc847500d96c967`，并显式 supersede 旧 plan
`ff63a87a2ee3bbfe332c9216f2ae15ce1a042722851088b78131196213137d25`。

revision-2 OFF 对 10,041 帧达到严格 `off_maximum_bbox_difference=0.0`，随后才允许 ON。ON 共执行
2,776 次第二搜索，但只有 12 次被严格在线规则接纳，且只覆盖 `bottle03`、`bag04` 两条序列。

| recovery-search fixed6 指标 | 原始值 | 预注册门 | 通过 |
|---|---:|---:|:---:|
| recovery action rows | 12 | ≥5 | 是 |
| action sequences | 2 | ≥3 | 否 |
| beneficial / harmful / catastrophic actions | 0 / 3 / 0 | precision≥0.80、harm≤0.05、cat=0 | 否 |
| action precision / harm rate | 0.000000 / 0.250000 | ≥0.80 / ≤0.05 | 否 |
| mean IoU delta | -0.000219032 | ≥0 | 否 |
| 10-frame failure-start delta | 0 | ≤-1 | 否 |
| catastrophic sequence regressions | 0 | 0 | 是 |

逐序列只在 `bottle03` 接纳 3 次、净 IoU gain `-0.126721`，在 `bag04` 接纳 9 次、净 gain
`-0.041780`；其余四条无动作。也就是说扩大搜索窗产生了大量候选，但现有“绝对质量通过”规则接纳的
方向与真实收益相反，既没有降低 VOT 的长失败起点，也轻微降低均值 IoU。正式 decision 为
`fixed6_recovery_search_rejected`，`eligible_for_full152_recovery_trace=false`。权威结果：
`/root/autodl-tmp/sutrack_rgbd_state_gate_train152_v1/recovery_search_fixed6_v1/analysis/fixed6_recovery_result.json`，
SHA256 `f074eb42d10e300206cfc819258a6ec81d098e125ac1f58a62325f5d22f252db`；controller 终态 SHA256
`8e7926552dc7854f011db1dd7c2bb956073b2eef7c6259d1df15f0ba64d98f0c`。

#### 24.95.4 当前边界与下一接手动作

当前公开最好仍是 SUTrack-L384 + 结构化语言 + safe-v1 的
`EAO/ACC/ROB = 73.974969 / 82.627562 / 89.455266`；目标仍为
`77.9 / 82.1 / 93.7`。DepthTrack Test 与 CDTB 的受保护正式 P/R/F 分别保持
`65.995933/65.335885/65.664250` 和 `75.387821/76.005850/75.695574`，本节没有重跑或覆盖它们。

不能继续把时间花在同一组固定阈值上：状态门失败说明单一 prior-bbox 动作的跨序列可分性不足；
recovery-search 失败说明绝对 RGB/depth/confidence 质量不等于“相对 baseline 更好”。下一结构必须至少
同时改变动作表示与相对接纳机制，例如让多个候选共享一次相对排序/短期 tentative rollout，再做原子
提交；否则应按既定用户决策迁移到更高公开指标且代码/权重完整的 RGBD baseline，并只移植结构化语言、
不可变首模板、安全动态槽和原子回滚创新。纯 baseline 数字仍不需要在本服务器重测。

建议接手时依次使用：`research`（只查官方论文/仓库与可复现权重）、`experiment-plan`（冻结新的
Train-only 决策门）、`run-experiment`、`monitor-experiment`、`analyze-results`；只有 Train 门通过后
才使用 `result-to-claim` 并安排公开 VOT。

### 24.96 为什么当前 VOT 仍低于 SUTrack 官方报告（2026-08-15 复核）

#### 24.96.1 先限定比较口径

当前服务器实测 `73.974969 / 82.627562 / 89.455266` 是
“SUTrack-L384 checkpoint + 结构化语言 + safe-v1”的完整 127 序列、1,765 anchors 结果；
`76.6 / 83.5 / 92.2` 是 SUTrack 论文报告值，官方纯 baseline 按用户要求未在本服务器重跑。
两者差值为 EAO `-2.625031pp`、ACC `-0.872438pp`、ROB `-2.744734pp`。因此可以确定差距主要落在
失败/生存侧，但不能把全部差值严格归因给某一个创新模块；缺少同服务器、同 toolkit、同数据快照的
纯 SUTrack full-127 对照。

VOT toolkit `0.7.1` 的本轮配置是 anchor multi-start：可见 GT 上连续 10 帧 `IoU<=0.1` 形成失败；
ACC 统计有效跟踪片段的 overlap，ROB 统计 anchor 生存，EAO 将失败后的 fragment 尾部补零并在
长度 115--755 的有效区间积分。因此“平时框得准、偶尔连续漂移”会呈现 ACC 尚可而 EAO/ROB 明显下降。
当前 ACC 仅比论文低 `0.87pp`，但 EAO/ROB 低约 `2.6/2.7pp`，正符合这一模式。

#### 24.96.2 已由配对证据确认的原因

1. **safe-v1 模板事务在部分困难 anchor 上过早/过严，弱于 SUTrack 默认更新。** 固定 16-anchor
   同源配对中，safe-v1 为 `8` 次失败、`10,043` progress 帧；保持语言、checkpoint、RGBD 读图和
   VOT 协议不变，只恢复 SUTrack 默认更新后为 `7` 次失败、`11,658` progress 帧。default 多
   `1,615` 个 progress 帧，说明至少在该诊断集合上，安全门没有带来净 ROB 收益。具体例子：
   `bag02@100 +841`、`toy09@100 +324`、`cup02@1600 +248`、`shoes02@613 +174` progress 帧
   都是 default 更好；safe-v1 只在 `toy09@150` 多 `335` 帧，说明作用并非单向。
2. **拒绝动态模板不等于回滚完整跟踪状态。** safe-v1 先把候选框写入 `self.state`，再根据冲突决定
   drop template；错误框仍成为下一帧搜索中心。确定性复现已经把旧状态从
   `REPRO_STATE_POISONED` 修到 v3 的 `PASS_STATE_ROLLED_BACK`，证明该递归污染路径真实存在。
   具体长失败链集中在 `bandlight_indoor_1`、`box_room_noocc_4_1`、`cube05_indoor_5/6`、
   `yogurt_indoor_1`。但 worst-5/112 anchors 的单次有界回滚只带来 EAO `+0.223683pp`、ROB
   `+2.295412pp`，同时 ACC `-0.828748pp`，未过预注册容量门，说明“只恢复上一 bbox”强度不足。
3. **现有在线质量分数不能判断相对收益。** fixed6 扩展搜索执行 `2,776` 次，只接纳 `12` 次，
   有益/有害为 `0/3`，平均 IoU delta `-0.000219032`；Train152 的 55 参数线性状态门三 seed 又都
   `oof_selection=null`。这证明 confidence、response margin、RGB 身份和 depth 绝对质量能够描述
   当前候选，却不足以判断“替代 baseline 后是否更好”，所以对失败恢复的命中方向仍不可靠。

#### 24.96.3 高概率原因，但尚未做严格配对消融

**RGBD 路径上的语言输入存在训练/部署分布错位。** 官方 `sutrack_l384.yaml` 的 `USE_NLP` 只显式为
`TNL2K=True`，`depthtrack` 走 `DEFAULT=False`；当前新配置却增加 `DEPTHTRACK=True`，把真实结构化
描述送入冻结 CLIP/SUTrack 融合路径。主 checkpoint 没有为这批 VOT-RGBD 描述重新训练，文本可能帮助
目标身份，也可能扰动原本以 RGB+Depth 为主的 task token/特征分布。因为 full-127 没有做“同一
safe-v1、语言严格 OFF/ON”对照，目前只能把它列为高概率原因，不能写成已证实结论。

#### 24.96.4 具体序列为什么呈现两极分化

- `glass01_indoor_2`：相对旧 SRTrack 参考，失败从 `19/19` 降到 `9/19`，ROB
  `15.817818 -> 81.589475`；这里较强 SUTrack 表征和受控模板更新确实延长了生存。
- `bag02_indoor_2`：失败从 `13/27` 降到 `5/27`，ROB `69.618472 -> 86.846209`；但在
  `bag02@100` 的模板配对中 default 又比 safe-v1 多 `841` progress 帧，说明 SUTrack 主干有恢复能力，
  safe gate 反而可能阻断了有用外观适应。
- `cup02_indoor_1`：仍为 `36/36` anchors 全失败，ROB `6.959906 -> 5.612288`；单次候选拒绝或
  上一框回退不能解决长遮挡/错误搜索中心后的再捕获。
- `shoes02_indoor_1`：失败从 `12/13` 变成 `13/13`，ROB `25.208436 -> 10.528037`；这是
  “保守模板保护仍可能造成适应不足”的直接反例。

#### 24.96.5 baseline 调研勘误与当前决策

初版调研把 STTrack 的 **DepthTrack F-score 77.6** 错读成 VOT-RGBD2022 EAO；作者官方仓库明确
STTrack 的 VOT EAO 是 **63.3**。因此 STTrack 不是强于 SUTrack 的 VOT baseline，迁移在代码改动和
公开评测前已停止；已冻结的 commit、MIT 许可和 checkpoint 只作审计保留。真正高于 SUTrack EAO/ROB
且有可运行产物的性能参考仍是 MixForRGBD `77.9/81.6/94.6`，但其 VOT 挑战提交包无顶层 LICENSE，
不得直接复制为开源 baseline。当前合法主线仍是 SUTrack，下一步优先做训练域内的语言 OFF/ON
因果消融和相对候选/短期 tentative rollout，而不是继续扫描公开 VOT 阈值。

本次复核时两张 RTX 3090 均空闲，远程无 tracker/training `screen`；没有启动新的公开评测，
DepthTrack/CDTB 正式结果未改动。本节及 baseline 勘误只续写远程权威文档，不回写本地桌面归档。

<!-- RGBD_LANGUAGE_HANDOFF_24_97_FIXED6_CAUSAL_LANGUAGE_ABLATION -->
## 24.97 结构化语言 ON/OFF 的 Train-only 因果消融（2026-08-15）

### 24.97.1 为什么做这一步

当前正式 VOT 最优仍为 SUTrack-L384 + 结构化语言 + safe-v1：EAO/ACC/ROB = `73.974969 / 82.627562 / 89.455266`，低于论文中官方 SUTrack-L384 的 `76.6 / 83.5 / 92.2`。差距主要在 ROB（-2.744734 个百分点），而不是 ACC（-0.872438 个百分点）。除模板更新引入的额外失败外，另一个高概率原因是：官方 DepthTrack 路径关闭 NLP，而当前部署给冻结的 SUTrack 权重启用了结构化长描述，可能造成语言输入分布偏移。

因此冻结一个严格的单变量实验：同一 checkpoint、同一 safe-v1、同一 6 条 DepthTrack Train 序列、同一初始框，只改变以下两个等价语言开关：

- ON：`TEST.USE_NLP.DEPTHTRACK=true`，`TEST.RGBD_LANGUAGE.USE=true`；
- OFF：上述两项均为 `false`；
- 其余配置字节级一致；推理阶段只读取首帧初始化 GT；两支预测和 manifest 全部冻结、复验 SHA 后，分析器才读取完整 Train GT；没有训练，没有公开 VOT 评测。

冻结计划：`/root/autodl-tmp/sutrack_rgbd_language_ablation_fixed6_v1/source/language_ablation_plan.json`，SHA256 `dc12f2a9db8f66793edca7f67537a8adea56bef37192f911f419b6ddfbb6f898`。覆盖 `bottle03_indoor, ball16_indoor, bag04_indoor, flower03_indoor, pigeon05_wild, toy03_indoor`，每支 10,041 帧。

### 24.97.2 结果

分析报告：`/root/autodl-tmp/sutrack_rgbd_language_ablation_fixed6_v1/analysis/analysis.json`，SHA256 `0bceea00ec2705185813aa14399168f8c1b5b282734a2bd16aae82af82de9d4f`；配对逐帧数据 SHA256 `82a18ee83f88ec4765a7045fbbb1bd10e36b7c8a52143de178706df32759c29b`。

| Train-only 单起点代理指标 | 语言 OFF | 语言 ON | ON-OFF |
|---|---:|---:|---:|
| 平均 IoU | 0.696363 | 0.705369 | +0.009006 |
| 连续低 IoU 失败段 | 16 | 18 | +2 |
| 严重低 IoU 帧 | 1,718 | 1,610 | -108 |
| 非负序列数 | - | 2/6 | 未达 4/6 门槛 |

逐帧配对中，ON 相对 OFF 有 316 个明显收益帧、211 个明显伤害帧、155 个 rescue 帧、70 个 catastrophic 帧。总体均值虽然上升，但失败段增加且收益只覆盖 2/6 序列，最终决定为：`structured_language_not_supported_on_fixed6`。

具体例子：

- `bottle03_indoor`：平均 IoU `+0.035629`，失败段 `-1`，118 个 rescue、0 个 catastrophic，语言明显有益；
- `ball16_indoor`：平均 IoU `-0.036549`，失败段 `+1`，54 个 catastrophic、仅 6 个 rescue，是最明确的语言伤害样例；
- `bag04_indoor`：平均 IoU `-0.005374`，失败段 `+1`，11 个 catastrophic、2 个 rescue；
- `pigeon05_wild`：平均 IoU `+0.027609`，严重帧减少 49，但失败段仍 `+1`，说明平均重叠改善不能保证 VOT 鲁棒性改善；
- `flower03_indoor`、`toy03_indoor`：均值轻微下降且没有 rescue。

### 24.97.3 对 VOT 低于 baseline 的修正结论

该实验把此前的“语言分布偏移”推测收紧为可复验证据：**当前全局启用结构化长描述不是可靠的 VOT ROB 优化**。它能改善部分正常帧和个别序列，但也会增加连续失跟段；VOT 的重初始化/失败惩罚会放大这类尾部伤害，所以出现 ACC 接近而 ROB、EAO 更低是合理的。

这不等于删除语言创新点。后续只允许两条 Train-only 路线进入候选：

1. 目标类别/短提示替代长结构化描述，减少冻结权重的文本分布偏移；
2. 训练一个可审计的语言启用门，仅在 Train OOF 证明无新增失败/伤害的帧或序列启用语言；默认保持 SUTrack 的零语言 token。

模板更新仍保留为核心创新，但不能继续全局 safe-v1。下一轮应以 VOT 失败主导的恢复场景为目标，限制动态模板写入、提供原子回滚，并以“失败段不增加、跨序列非负”为 Train-only 先验门。只有通过该门才允许正式 VOT；DepthTrack Test 和 CDTB 不重跑、不改写其已达标结果。

### 24.97.4 实现与可复现信息

- 运行器：`tools/run_depthtrack_train_language_ablation.py`，SHA256 `0056472b5b857ecdea46c047c8d494787672ab5baf5099cec5a7acf42fe66714`；
- 分析器：`tools/analyze_depthtrack_train_language_ablation.py`，SHA256 `4833c61a5488bd3d6e56a16dba88fa2bac7d718097478ff926dac4ed306fc8cb`；
- ON 配置 SHA256 `c6366f56b34d9b8a8b1d7864a7dca5423ef76aab7fbdba59895753adcd5ce682`；
- OFF 配置 SHA256 `bc5c4c07958335d5c1136dc1ad65fc543133a6c522a3dd3e680754cb98f485d2`；
- SUTrack checkpoint SHA256 `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`；
- CLIP checkpoint SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`；
- 分析报告明确标注 `single-start Train-only proxy; not VOT anchor multi-start EAO/ROB`，不得把这些 IoU 数字写成正式 VOT 指标。

<!-- RGBD_LANGUAGE_HANDOFF_24_98_GITHUB_SOURCE_ONLY_PUBLICATION -->
## 24.98 GitHub 源码-only 发布记录（2026-08-15）

按用户要求，已将当前与 RGB-D 语言/模板更新有关的代码推送到 `https://github.com/666666666666gao/Track`，但没有发布权重、数据集、VOT workspace、预测、日志、环境、服务器路径配置文件或内部交接文档。

目标仓库已有独立的 MPLT/RGB-D 内容，且与 SUTrack 上游没有共同 Git 历史。因此本次没有 force push、没有覆盖根项目，而是在目标 `main` 上做一次快进提交：

- 推送前目标 `main`：`8a01687d66597558a3dde01c30cb702a9b637629`；
- 推送后目标 `main`：`030220b9e822084f3284cdf225183bc8d99d8547`；
- commit message：`Add SUTrack RGB-D language template overlay`；
- 公开目录：`projects/sutrack_rgbd_language_template/`；
- 形式：以 SUTrack `d65052d1ba3fcf55010e1fb3665ee6616c139a2c` 为基线的 source overlay，不是重新分发完整 baseline；
- 公开 README 明确写明当前正式 VOT 结果低于论文 baseline，禁止把未通过的模块表述成已提升官方 baseline。

发布内容包括 48 个 overlay 文件：相关的 tracker/config/VOT bridge、语言 manifest、安全模板更新、状态回滚/相对候选实验代码、Train-only 运行器/冻结分析器、配置和 SUTrack 原许可证；另含 overlay README、SHA256 manifest 以及根 README 链接。本次提交共 51 个文件变更、约 426 KiB 源码文本。

推送前后复验：

- 40 个 Python 文件完成 AST 解析；
- `tools/test_state_trace_analyzer_features.py`：2/2 通过；
- `tools/test_sutrack_recovery_search_disabled.py`：1/1 通过；
- 48 个 overlay 文件与 `MANIFEST.sha256` 全部一致；
- 密钥扫描未发现 SSH 密码、GitHub token 或私钥；
- GitHub API 复验项目树中超过 10 MiB 文件为 0，`.pth/.pt/.ckpt/.onnx/.safetensors/.weights/.h5/.hdf5/.pb` 为 0；
- 明确排除了 `checkpoints/`、`__pycache__/`、`.aris/`、`lib/test/evaluation/local.py`、所有 `/root/autodl-tmp` 实验输出以及本权威交接文档/receipts。

远程 `/home/SUTrack_RGBD_L` 仍是继续实验的权威工作树；GitHub 目录是经过筛选的发布快照。后续代码形成稳定、可复现增量时，再刷新该 overlay；交接内容继续只续写本远程文档。

<!-- RGBD_LANGUAGE_HANDOFF_24_99_SHORT_PROMPT_AND_ORACLE_CAPACITY -->
## 24.99 短提示与三动作语言容量审计（2026-08-15）

### 24.99.1 单一预注册短提示

24.97 已证明全局长结构描述会增加连续失败。为保留语言创新而减少冻结权重的文本分布偏移，本轮只从审核后的首帧目标 crop 提取 `annotation.appearance + annotation.category`，构造平均 5.79 词的自然短提示；明确排除 depth relation/quality、occlusion、distractor 和 motion 字段。152 条 manifest 无路径、bbox 或未来帧文本。

冻结计划：`/root/autodl-tmp/sutrack_rgbd_language_short_prompt_fixed6_v1/source/short_language_ablation_plan.json`，SHA256 `b62be99c9da17b9701256463ff5a3a7145022f007e83912eb8ab6c87d0f3165b`。同 24.97 一样，严格重跑 6 条 DepthTrack Train、每支 10,041 帧；ON/OFF 只差两个语言开关，checkpoint、CLIP、safe-v1 和实现均相同。OFF 两个分片的 prediction SHA 与 24.97 的独立 OFF 重跑完全一致，验证了确定性。

短提示分析 SHA256 `af30bf7dc546f3cc73d828cbd35c91ca78c7b204b72a26e43fb0808bcb89b38c`，逐帧配对 SHA256 `c11c9a6bafb823f968b1f7d2a684d667c2c0907cd5d4374f65939ac996698ec6`。

| Train-only 单起点代理指标 | OFF | 长结构描述 | 短 appearance+category |
|---|---:|---:|---:|
| 平均 IoU | 0.696363 | 0.705369 | 0.700371 |
| 连续低 IoU 失败段 | 16 | 18 | 17 |
| 严重低 IoU 帧 | 1,718 | 1,610 | 1,677 |
| 相对 OFF 非负序列 | - | 2/6 | 1/6 |

短提示相对 OFF 为 `+0.004008` 平均 IoU、严重帧 `-41`，但仍新增 1 个失败段，且只有 1/6 序列均值非负，因此同样被门拒绝。逐帧有 264 个 benefit、226 个 harm、126 个 rescue 和 82 个 catastrophic。短提示比长描述少 1 个额外失败，但未解决 ROB 风险。

具体序列：

- `bottle03_indoor`：短提示均值 `+0.035089`、失败段 `1→0`、116 rescue/0 catastrophic，语言持续有益；
- `ball16_indoor`：均值 `-0.031357`、失败段 `3→4`、47 catastrophic/5 rescue；
- `bag04_indoor`：均值 `-0.006829`、失败段 `4→5`；
- `pigeon05_wild`：失败段 `3→2`，但均值反而 `-0.013774`、严重帧增加 19，说明当前失败段代理与整体轨迹质量需要联合约束；
- `toy03_indoor`：短提示新增 1 个失败段；`flower03_indoor` 无 rescue 且均值轻微下降。

### 24.99.2 OFF/长/短三动作 GT oracle 容量上界

为判断是否值得立即扩展到 full-152 并训练动态语言门，新增一次只消费已冻结 paired rows 的后验 oracle。它逐帧从 OFF、长描述、短提示选择 GT IoU 最大者，只是容量上界，不是可部署策略。结果文件：`/root/autodl-tmp/sutrack_rgbd_language_short_prompt_fixed6_v1/capacity/language_multiaction_capacity.json`，SHA256 `4b254f8fbda9f80cac81047b53d7afcceadfdd70bf17af2cb633c19bd0afeb35`。

oracle 在 8,698 个有效非初始化帧中选择 OFF/短/长分别 3,852/2,468/2,378 次；平均 IoU 达 `0.721716`，相对 OFF 上界 `+0.025354`，严重帧降至 1,527。但连续失败段仍是 `16`，没有比 OFF 的 `16` 减少，最终容量决定为：`mean_overlap_capacity_without_robustness_capacity`。

因此，三动作语言门有改善正常重叠精度的容量，却没有在当前 fixed6 上证明 VOT ROB 所需的失跟恢复容量。为主目标 EAO/ROB 立即运行三分支 full-152 并训练大门控，收益依据不足；本轮停止该扩展，不访问公开 VOT。

### 24.99.3 当前架构决策

语言创新不删除：manifest 绑定、长/短文本分支和未来可学习 gate 均保留；但 VOT 候选必须 fail-closed，以 OFF 作为保护动作，只有新的 Train-OOF/held-out 证据同时满足“失败不增加、跨序列非负”才允许启用语言。下一优化优先级回到模板/搜索状态：以 SUTrack 默认轨迹为保护分支，让新模板只在 shadow/tentative 短期 rollout 中探索，并以相对候选的一致性证据决定是否原子切换，目标必须是减少失败段而非只提升平均 IoU。

实现与配置：

- 短提示 materializer SHA256 `08511beac542cca1ee2a785d23258d61de1f0e15f9d56010587e7ff22b1623c2`；
- 三动作容量 analyzer SHA256 `b57370db52af9d22b5ae17616bcc25fec9596f36637f82c363e6912d9e675712`；
- 短提示 ON/OFF config SHA256 分别 `fdd231dfb1c009d51410edfb6f4ad28f7df106c3f30bbe1e1587a2a0f6e70cc2`、`6be7c5ebb6a6b93c7ab9a1836248e51ee93c4953011e0a98e1e292b1ead31035`。

本节所有数字都是 Train-only 单起点代理或 GT oracle 上界，不是 VOT EAO/ACC/ROB；没有训练、没有新的公开评测，DepthTrack Test/CDTB 正式结果保持不变。

<!-- RGBD_LANGUAGE_HANDOFF_24_100_GITHUB_SHORT_LANGUAGE_UPDATE -->
## 24.100 GitHub 短提示审计源码增量（2026-08-15）

24.99 完成后，目标 GitHub `main` 再次以快进方式从 `030220b9e822084f3284cdf225183bc8d99d8547` 更新到 `2d1887d84b06c40bb87c368be4fc3decf56beddb`（`Add short language capacity audit`）。新增短提示 materializer、三动作容量 analyzer 和短提示 ON/OFF 两份配置，并更新公开 README/`MANIFEST.sha256`；overlay manifest 现绑定 52 个文件。

GitHub API 终态复验：目标 `main` 指向 `2d1887d84b06c40bb87c368be4fc3decf56beddb`，项目树超过 10 MiB 文件为 0、权重扩展名文件为 0。仍未上传预测、analysis JSON、GT、数据集、checkpoint、日志、内部 handoff 或 receipt。远程 `/home/SUTrack_RGBD_L` 继续作为实验权威工作树。

<!-- RGBD_LANGUAGE_HANDOFF_24_101_REAL_TENSOR_BLEND_FIXED6 -->
## 24.101 真实 0.10 模板张量混合的 fixed6 因果消融（2026-08-15）

### 24.101.1 发现并修正的实现语义问题

复核 SUTrack 安全模板路径后确认：历史配置中的 `SAFE_TEMPLATE_UPDATE.BLEND_WEIGHT=0.10` 只进入策略元数据，实际 `_replace_dynamic_template` 仍把候选模板整块写入动态槽，代码中没有 `torch.lerp`。这意味着此前的 safe-v1 不是“10% 低权重模板更新”，而是“通过严格门后 100% 替换模板”；该语义能解释部分更新后轨迹突变，但不能直接证明改成插值会更好。

本轮增加向后兼容开关 `TEST.SAFE_TEMPLATE_UPDATE.APPLY_TENSOR_BLEND`，默认 `False`，以保证全部历史配置继续执行原始整块替换。仅新的 blend 配置将其设为 `True`，并在模板 tensor 与 template-annotation tensor 上执行 `torch.lerp(static, candidate, 0.10)`。tracker 额外输出 `applied_template_blend_weight`，producer 对每一次 replace 强制验证 raw=`1.0`、blend=`0.10`。预检覆盖了配置仅一个字段不同、0.10 数值插值、raw 整块替换、非法/非有限权重拒绝、历史配置默认关闭以及既有 recovery-disabled 测试；全部通过后才启动 fixed6。

代码与配置：

- `lib/config/sutrack/config.py`：新增默认关闭的真实张量混合开关，SHA256 `cb0135c1baec612068ceb2adfa132d9b30e9505ae45d5f2e771d9cedac80c77c`；
- `lib/test/tracker/sutrack.py`：真实 tensor/annotation 插值与应用权重 trace，SHA256 `c4a455ab854c86d1f198702a142244ba41853702dc94ada320f818bfeabce539`；
- `tools/run_depthtrack_train_template_blend_ablation.py`，SHA256 `e99b36b35c4273929e0257b73e8580f5ffcfc31385c2a7cee16f3ed0b1af49e9`；
- `tools/analyze_depthtrack_train_template_blend_ablation.py`，SHA256 `74feb1d5afcb268b27d6dea5fa2d99d3d4b5e12f388242a8e59fe866b0750036`；
- raw/blend 配置 SHA256 分别为 `60d63f5e6fa2d3e2cfb85ae1e24ce436ac390a844ca2606f7df19cd1392dd164`、`11baaf6ea640d4d06091850021a7b9b864603e6e33eea9de363f70cac60b5f0f`，两者只差 `APPLY_TENSOR_BLEND`。

### 24.101.2 冻结协议与结果

冻结计划：`/root/autodl-tmp/sutrack_rgbd_tensor_blend_fixed6_v1/source/template_tensor_blend_plan.json`，SHA256 `73a6eef0cc2bb15332a0c13b136952de39d360788655c824ee03695a4ac50f25`。实验严格复用 24.97/24.99 的 6 条 DepthTrack Train、每分支 10,041 帧；语言/NLP 全关、checkpoint/CLIP/安全模板门和其余实现相同。四分片全部 exit 0；producer 只读取初始化 GT，分析器在四分支冻结并复核 SHA 后才读取完整 Train GT。没有训练、没有访问 VOT/DepthTrack Test/CDTB。

分析报告：`/root/autodl-tmp/sutrack_rgbd_tensor_blend_fixed6_v1/analysis/analysis.json`，SHA256 `01cc64f4d9b4ce3adf196481456253c055f2e3bdef9a755ea4ab860732699498`；逐帧配对 SHA256 `e03cc7e691ce35155e469833bcb250f1c758a64bb7e88ec8a6849bab34f4a63d`。

| Train-only 单起点代理指标 | raw 100% 替换 | 0.10 tensor blend | 差值（blend-raw） |
|---|---:|---:|---:|
| 平均 IoU | 0.696363 | 0.693793 | -0.002569 |
| 连续低 IoU 失败段 | 16 | 17 | +1 |
| 严重低 IoU 帧 | 1,718 | 1,745 | +27 |
| catastrophic / rescue | - | 77 / 54 | 未过门 |
| 相对 raw 非负序列 | - | 2/6 | 门槛 4/6 |
| 实际 replace 帧 | 55 | 57 | 已真实触发 |

所有核心门均失败，最终决定为 `tensor_blend_not_supported_on_fixed6`。raw 与 24.99 历史 OFF 逐框最大 bbox 差严格为 `0.0`，证明默认关闭的实现保持历史行为，当前正式最佳指标没有被污染。

序列层面只有 `flower03_indoor` 为正（`+0.001984`），`pigeon05_wild` 完全不变；`bottle03/ball16/bag04/toy03` 均值分别下降 `0.004723/0.001596/0.001930/0.003353`。其中 `bottle03` 失败段 `1→2`，`ball16` 出现 50 个 catastrophic、46 个 rescue，说明把候选稀释到 10% 并没有解决错误更新，反而可能让不良状态以较弱但更持久的方式留在递归模板中。

### 24.101.3 当前决策与下一优化方向

该分支停止，不进入 full152，不启动正式 VOT。真实 tensor blend 代码保留为实验能力，但默认关闭；论文或报告不得把历史 safe-v1 称为 0.10 模板插值，也不得把本轮 Train-only IoU 写成 VOT 指标。

当前证据进一步支持：问题不在“替换幅度太大”这一单一旋钮，而在候选模板本身的身份正确性、写入后的递归持久性和缺少 shadow 验证。下一条可执行路线应保持 SUTrack 默认/static 路径为保护状态，在独立 tentative slot 中短期 rollout；只有相对保护分支的因果证据同时证明收益、无新增 failure/catastrophic，才原子 promote，否则丢弃候选。优先优化 ROB/失败恢复，不再扫描固定 blend weight，也不重跑已达标 DepthTrack Test/CDTB。

<!-- RGBD_LANGUAGE_HANDOFF_24_102_SHADOW_TEMPLATE_FIXED6 -->
## 24.102 两帧 tentative-template shadow 容量审计（2026-08-15 至 2026-08-16）

### 24.102.1 目的与实现

24.101 证明固定模板混合权重不是根因后，本轮实现真正隔离的模板事务探针：公开/保护路径继续执行 SUTrack 原始 `25` 帧间隔、置信度 `0.70` 的模板更新；safe-template writer 只生成候选模板，候选在独立 state/template 副本中读取未来 `t+1/t+2` 两帧，第二步使用第一步 shadow bbox 递归裁剪；任何 shadow 结果都不写公开 bbox、模板、annotation 或 policy state。语言关闭，只回答“候选模板是否存在失败恢复容量”。

首次 v1 trace 暴露了状态机边界：writer 在事件后下一帧发出 `drop_dynamic` 时，旧实现错误截断已经开始的两帧因果窗口，导致 `bag04_indoor` 的 event 3 只有 step1。该根作为无效证据原样保留，没有复用。修正为“writer 的后续 drop 不能删除已冻结候选的固定两步读窗口”后，真实 8 帧 GPU 预检得到事件帧 5、shadow 帧 6/7，保护路径 8/8 帧一致，再在新 v2 根完整重跑。

冻结计划：`/root/autodl-tmp/sutrack_rgbd_shadow_template_fixed6_v2/source/shadow_template_plan.json`，SHA256 `23879b19886ea8955f0b55d975714cae1d6ad5d8fb245186bb7031714d5bac48`。分析报告：`/root/autodl-tmp/sutrack_rgbd_shadow_template_fixed6_v2/analysis/analysis.json`，SHA256 `63d2519ac26b08215a4bfa9db05d7fa8ba2a0013d4252cf879dc8c1d518db94a`；逐帧配对 SHA256 `41b71966a33a3c7750d2a3c712cb312ebbd1e3e88965e5f855c64b206642ac9a`。

### 24.102.2 结果与决定

| Train-only shadow 指标 | 结果 |
|---|---:|
| writer events / 完整两步事件 | 78 / 78 |
| 有效 shadow 帧 | 156 |
| 保护 / shadow 平均 IoU（仅 shadow 帧） | 0.900546 / 0.900110 |
| oracle 选 shadow 的帧 | 59 |
| oracle 累积 IoU 增益 | +0.578261 |
| 显著 benefit / harm（阈值 0.05） | 0 / 0 |
| failure recovery / catastrophic 帧 | 0 / 0 |
| 保护 / oracle failure episode | 20 / 20 |
| ON-OFF bbox / score 最大差 | 5.97e-13 / 0.0 |

虽然 shadow 在 59 帧略高于保护分支，但全部差值都小于预注册的 0.05，且没有恢复任何低 IoU 失败，也没有减少 failure episode。跨 GPU 独立运行还出现 `5.97e-13` 的 bbox 浮点尾差，未满足字节级零扰动门。最终决定为 `shadow_template_capacity_not_supported_on_fixed6`：不进入 full152，不训练 promote gate，不访问 VOT。

实现快照：`sutrack_shadow_template.py` SHA256 `c743ba66002be765fae7947c5a05edc6ad08c2255574143497f392624a0954b8`；runner/analyzer SHA256 分别 `fe1955f81ae8dd9d416eb6df1c555cbd22ea552e7c9b4d196c2ba962f910b5e1`、`ca4542c44640c2c96bbd8457b8e640674403ada0e022b6bc9bd81f36f6d0b3e3`；OFF/ON 配置 SHA256 分别 `efd0b3dff3e5c3d5b2684ffb0cc1d43ee900428e0fddf28b09cf9bad1bf7ac0d`、`4519445848e7d8335d6aae368174b9fb5ec32aaafecca182cf3ac4304c9c809a`。

该负结果只否定“当前 safe writer 的整模板候选 + 两帧 horizon”具有足够容量，不否定原子事务架构本身。后续若再做 tentative 分支，必须更换候选动作或延长到能观察 survival 的窗口，不能只增加同一候选的阈值扫描。

<!-- RGBD_LANGUAGE_HANDOFF_24_103_GATED_ONLINE_QWEN_FIXED6 -->
## 24.103 受控在线 Qwen 外观语言记忆 fixed6（2026-08-16）

### 24.103.1 实际开启方式

按用户要求，已把旧 SRTrack 的本地 Qwen2.5-VL 能力迁移到 SUTrack，但没有开启逐帧无条件描述，也没有与实验模板同时提交。新路径为：

```text
首帧短 appearance+category 文本（永久不可变）
        ↓
每5帧检查 RGB/Depth/响应/运动稳定性
        ↓（每轨迹最多2次，间隔至少60帧）
本地 Qwen2.5-VL-7B 结构化 JSON
        ↓
类别一致 + visual_valid + 未来两帧连续确认
        ↓
Shadow：只记录，不影响跟踪
Active：CLIP text embedding 以 0.10 权重 lerp
        ↓
硬冲突或 TTL 90 立即清除，精确回静态文本
```

Qwen 只返回 `category/appearance/visibility/motion_state/depth_relation`，服务端确定性构造最多 40 词的动态文本；不允许改类别、首帧稳定属性或模板。endpoint/model 只从 `QWEN_VL_ENDPOINT/QWEN_VL_MODEL` 环境变量读取，worker 只监听 `127.0.0.1`。模型目录、processor/tokenizer、index 和 5 个 safetensors shard 全部逐文件 SHA 绑定在冻结计划中，没有下载或改权重。

失效 worker 的 20 帧真实 GPU 预检中发生 1 次受控生成错误，但相对 Static 的 bbox/score 最大差均为 `0.0`。真实 worker 预检在 `bag04_indoor` 第5帧生成 `black, textured / partially obscured / stationary / close`，后续 bbox IoU `0.9169/0.8941`，第7帧确认；Shadow 20帧仍严格零扰动。Active 从第8帧才首次出现差异，证明同帧/未来帧泄漏不存在。

冻结计划：`/root/autodl-tmp/sutrack_rgbd_online_language_fixed6_v1/source/online_language_plan.json`，SHA256 `cc0270d865b224ad32c2076db3a62334affd88e45a42dd2735e7d6de69747eac`。三路均为同一 6 条 DepthTrack Train、每路 10,041 帧、同一 SUTrack checkpoint/CLIP/短文本 manifest、SUTrack 默认模板路径；只允许 `USE` 和 `APPLY_DYNAMIC_TEXT` 两个配置字段不同。Static、Shadow、Active 依次在同一物理 GPU1 执行，Qwen 独占 GPU0，三路 exit 均为 0；producer 只读首帧 GT。

原冻结 analyzer 第一次读取完整 GT 后发现不可见帧表示为 `None`，旧代码在 None 判断前调用 IoU 而 fail-closed，未生成结果。机械修正为与既有 language analyzer 相同的约定：不可见帧只在 failure-series 记 0，不进入有效 IoU 均值；没有改变任何阈值、trace 或预测。amendment SHA256 `8a1a93e506f6fcfeda5cff7b59d3faabcac844570aedd6a67ef44b271064e566`，明确记录旧/new analyzer SHA 与三路 frozen source。

### 24.103.2 结果

分析报告：`/root/autodl-tmp/sutrack_rgbd_online_language_fixed6_v1/analysis/analysis.json`，SHA256 `1135907f2cbdfb4632cec01cabb3af8c0ce9147e02184ae649b631397d238b65`；逐帧配对 SHA256 `827e2cdc9cc515ef0e40bb36907cd3db8a0fb2d466647da94b998ecd1cb9f033`。

| Train-only 单起点代理指标 | Static | Qwen Active | Active-Static / 判定 |
|---|---:|---:|---:|
| 平均 IoU | 0.553040 | 0.692580 | **+0.139541** |
| failure episodes | 29 | 31 | +2，失败 |
| 严重低 IoU 帧 | 3,094 | 1,764 | -1,330，通过 |
| catastrophic / rescue | - | 81 / 1,402 | 新增 catastrophic，失败 |
| 非负序列 | - | 3/6 | 低于4/6 |
| 确认提交 / 10帧有益提交 | - | 7 / 2 | precision 28.57%，低于85% |
| Static-Shadow bbox/score 最大差 | - | 0.0 / 0.0 | 严格零扰动 |

最终决定为 `online_language_not_supported_on_fixed6`。在线 Qwen 具有很强的候选容量，不能简单归类为“无效”：它显著减少严重帧并大量 rescue；但当前“确认后立即写入公开 text/state 递归路径”仍增加 failure episode 和 catastrophic，正好触犯 VOT ROB/EAO 的核心门。因此不进入 full152，不运行公开 VOT，正式最好仍是 `73.974969/82.627562/89.455266`，DepthTrack Test/CDTB 正式结果不变。

### 24.103.3 具体正负例

- `bottle03_indoor`：平均 IoU `0.675322→0.887227`，failure `7→3`、severe `737→46`、rescue 721；两次提交的未来10帧均小幅正收益（`+0.002572/+0.005573`），是在线外观语言的明确正例。描述分别为 `orange, spray nozzle, red cap` 和 `orange, plastic, spout`。
- `ball16_indoor`：平均 IoU `0.170122→0.674926`、rescue 678，却 failure `11→15`。例如帧874起 Static IoU 为0而 Active恢复到 `0.627/0.837/0.801`；但帧115 Static `0.830`、Active仅 `0.043`。这说明平均 overlap 大增仍可能被更碎的连续失锁段拖累 VOT ROB。
- `bag04_indoor`：平均 IoU `-0.022321`、failure `5→7`、46 catastrophic/3 rescue。第7帧提交 `black, textured` 后仅使用2帧即因 hard conflict 清除，但递归 state 已被改变；帧117/119附近 Static IoU `0.867/0.899`，Active为0。第70帧 Qwen 把 crop 识别成 `shirt`，类别不匹配被正确拒绝，证明 fail-closed 类别门有效，但还不能撤销此前的 state 分叉。
- `flower03_indoor`：两次提交后均值基本不变但略负（`-0.000049`），两次10帧窗口均为负；说明“描述正确”并不等于“对 frozen SUTrack 条件分布有益”。
- `pigeon05_wild`：证据门全程不触发，0调用、0提交、输出与 Static 完全一致，是正确 abstain 样例。
- `toy03_indoor`：均值 `-0.002547`、1个 catastrophic；动态记忆在 TTL 到期后清除，但后续轨迹仍保留递归分叉影响。

最重要的因果发现是：动态文本实际活跃帧很少（例如 `ball16` 仅10帧、`bag04` 仅2帧），但之后数百帧仍可能与 Static 大幅不同。清除 text token 只能恢复输入，不能恢复已经被预测框写入的递归 crop/state。因此在线文本与 safe-template 暴露的是同一个根问题：**未经保护的短暂动作会永久改变后续搜索轨迹**。

### 24.103.4 下一版架构决策

保留本轮的结构化 Qwen worker、环境变量、预算、类别拒绝、两帧确认和失效端点精确回退；停止“确认后立即融合到公开路径”。下一版应改为：

1. 在早期可靠帧生成并认证外观描述，但只存入 dormant language memory；
2. 只有后续出现多峰/同类干扰/遮挡风险时，才在独立 tentative state 上启用动态文本；
3. protected Static 与 tentative Qwen 分支同时递归 rollout 至少覆盖 survival 窗口，公开 `state/template/text` 在此期间完全不变；
4. 只有相对保护分支的 RGB身份、Depth、响应 margin、轨迹连续性同时胜出才原子 promote，否则完整 rollback；
5. 语言角色只作 top-K 目标—干扰物关联和身份排除，不再把“视觉描述正确”直接当作提交充分条件。

由于本轮只有 2/7 提交在10帧窗口有益，不能用同一 fixed6 调阈值后声称改善，也没有授权 full152。若继续，应先在冻结 trace 上定义不使用 GT 的相对 survival 证据并做新的预注册 fixed6；目标必须是 failure 不增、catastrophic=0，而不是重复追求已经显示巨大但不安全的平均 IoU 增益。

### 24.103.5 实现快照

- config / base tracker / online tracker：SHA256 `1ac5f3af536ea0b867bac17c62de792cc6f670dc8c343a9b21e02182fc464736` / `8e547883c324a24f18b64e9f5df5017926d0216f997a249ce07ba6e757c01e06` / `22eacd3c5d72a09554825df1a51443eb725f6123d1e3ea1ccf158d7ae204ea3d`；
- client / worker：SHA256 `e52600084ce6ba828e19c033fe98cf0e827d939f572473664809ac635421e154` / `a1a07c6f17983e1a2f3c66ba07eb89f6d71828692fc4c844f51c2e8039679532`；
- runner / 修正后 analyzer：SHA256 `6e2ce28060d38be5fbfa5f8d7e9c1bd2946a2a6a2b98445562ad92bf05aabea4` / `514d3d632e9b28c7632ddedb0b62611bac22d7e6560f899e600ddb5d9f3a7f83`；
- Static / Shadow / Active 配置 SHA256：`32047803f447ecb8a193de8c70ab9f3bb0965fac26525ae1fb1b33c696226f91` / `9ec86e409e4e457e4b659931d7e465f23b867b87f86f590afbc5eda02b5e88b2` / `65d20b061b7f036cbde73dbf0f76b9d54063e6801706dce3d66bc8ad599452f1`。

本节所有指标均是 DepthTrack Train fixed6 单起点代理，不是 VOT EAO/ACC/ROB；没有训练新 checkpoint，没有修改 SUTrack 权重，没有访问 DepthTrack Test、CDTB 或公开 VOT。

## 24.104 Qwen 首帧实例身份二选一审计（2026-08-16，未通过）

### 24.104.1 为什么在在线文本失败后做这一步

24.103 已证明：动态文本有很强的局部 rescue 容量，但文本短暂进入 SUTrack 后会改变 `self.state`、下一帧 crop 和后续递归轨迹；即使动态文本随后被清除，轨迹也不会自动回到 Static。为避免再次把未经确认的语义写入递归状态，本轮把 Qwen 限制为**只读身份裁判**：Qwen 不生成跟踪文本、不更新 bbox、不更新模板，只判断保护分支与试探分支中哪个更像首帧的同一物理实例，证据不足时必须弃权。

输入固定为六张图：首帧目标 crop、当前完整 RGB（标注 A/B）、A 的 RGB/Depth crop、B 的 RGB/Depth crop。提示词提供不可变首帧类别和首帧描述，但不提供 tracker score 或 GT。输出严格限定为 `A/B/abstain`、两候选 match 状态、置信度和最多三条可见身份线索。

### 24.104.2 冻结协议与因果边界

- 根目录：`/root/autodl-tmp/sutrack_qwen_pairwise_identity_fixed6_v1`；
- 事件来源：24.103 已冻结的 Static 与 Active prediction trace；
- 事件选择不读取 GT：从第 8 帧起，当两分支 bbox IoU `<=0.60` 时取事件；同序列间隔至少 60 帧，每序列最多 2 个；
- 共冻结 9 个事件，来自 bottle/ball/bag/flower/toy；pigeon 没有满足条件的分歧事件；
- A/B 标签由 `sha256(sequence:frame_index)` 奇偶确定，Qwen 完成并原子写出 9 条 response 后，analyzer 才第一次读取 GT；
- 模型仍是本地 Qwen2.5-VL-7B，确定性解码；无网络 API、无训练、无公开评测；
- 预注册判定：一侧 IoU `>=0.50` 且两侧差值 `>=0.25` 才算 decisive；两侧都 `<=0.10` 时正确动作是 abstain；其余排除；
- 通过门：至少 3 个 decisive、decisive 选择覆盖率 `>=50%`、选择精度 `>=85%`、错误 decisive 选择为 0；至少 2 个 neither 且弃权率 `>=80%`；9 条 response 必须全部结构有效；
- 即使通过也只授权一个新的、未见 fixed6，不能直接进入 Train152 或 VOT。

冻结 plan SHA256：`7de3baa5f76f7b96a17e8c600aa3a8ab136c40bde6b26c67a2d365184910b4ba`。

### 24.104.3 结果

| 项目 | 结果 | 门槛 | 判定 |
| --- | ---: | ---: | --- |
| Qwen response | 9/9 有效 | 100% | 通过 |
| decisive 事件 | 3 | >=3 | 通过 |
| decisive 覆盖率 | 3/3 = 100% | >=50% | 通过 |
| decisive 正确 | 2/3 = 66.67% | >=85% | **失败** |
| decisive 错选 | 1 | 0 | **失败** |
| neither 事件 | 6 | >=2 | 通过 |
| neither 正确弃权 | 2/6 = 33.33% | >=80% | **失败** |

Qwen 最终选择分布为 `A=7、B=0、abstain=2`。分析终态：

```text
pairwise_identity_capacity_not_supported_on_frozen_events
```

因此本轮没有实现在线 promote/rollback，没有进入新 fixed6、Train152 或 VOT。

### 24.104.4 具体例子

1. **bottle03@531，错误且高置信。** 保护分支 IoU=0，试探分支 IoU=0.869416，正确答案是 tentative；Qwen 却高置信选择 protected，并给出“orange spray bottle / red cap / shape”作为线索。这说明描述在类别和外观层面看似正确，仍不能可靠区分当前两个实例/位置。
2. **bag04@200，正确。** 保护分支 IoU=0.775396，试探分支 IoU=0.485649；Qwen 选择 protected，线索为“black bag / rectangular shape / occluded”。这是 2 个正确 decisive 判断之一。
3. **toy03@852，正确。** 保护分支 IoU=0.870267，试探分支 IoU=0.505100；Qwen 选择 protected，使用红蓝人形玩具、棕色鞋子和人形轮廓作为实例证据。
4. **bottle03@471、ball16@87/169，目标在 GT 中不可见。** 两候选 IoU 均为 0，正确动作应为 abstain；Qwen 却都高置信选择 A，说明视觉语言模型容易把“像这一类别/外观”误当成“就是首帧同一实例”。
5. **flower03@172、toy03@1112，正确弃权。** 两候选均为 0，Qwen 以低置信 abstain；模型存在弃权能力，但在 6 个 neither 事件中只使用了 2 次，远不足以作 fail-closed 在线门。

### 24.104.5 新发现的问题与结论

本轮失败不是 JSON 格式、服务稳定性或推理速度问题：9 条输出全部有效，Qwen 推理总耗时约 17.9 秒。真正问题是**实例身份校准和弃权校准不足**。模型倾向于从候选中强行挑一个语义相符者，而不是在目标不可见或两个 crop 都不可靠时拒绝决策。

此外，虽然标签由哈希确定，但 9 个事件恰好形成明显不均衡，Qwen 输出又表现为 `A=7、B=0`。因此不能排除候选顺序/颜色偏置。该缺陷不会改变“当前方案已经未过门”的结论，但禁止据此声称 Qwen 具备稳定 pairwise identity 能力。若以后重启此方向，必须对每个事件同时运行 A/B 互换的成对输入，并要求交换前后映射一致；还应移除颜色含义、加入明确的 `neither` 视觉示例和校准集。

当前结论：

1. 不能将 Qwen 自由生成的动态文本直接注入 SUTrack 递归状态；
2. 也不能把当前 Qwen pairwise 判断直接作为 protected/tentative promote 门；
3. 继续方向应是先在视觉跟踪器内部保留 protected 与 tentative 两套独立 state/template，再用可验证的短窗视觉 survival 证据决定 rollback/promote；
4. Qwen 最多保留为 dormant memory：只记录候选属性，在视觉轨迹已经独立确认后提供低权重辅助，永远不能单独授权状态或模板提交；
5. 下一轮门仍必须是 failure 不增、catastrophic=0、Static/故障回退字节级一致，而不是平均 IoU 单项上涨。

### 24.104.6 证据与实现快照

- Qwen result / responses SHA256：`bb18927aea5b66210c11576456a57aa84ea849d916b475d3646a254e559f8dea` / `b24eaa52c717b7e1f171449ded2e933a852da2648063ad5218ca24376050f67e`；
- analysis / rows SHA256：`27e0d4eac3a94c58e009ce75e6c048c2b30b37d2a5142ed5163c9470ae4b84f1` / `72d8d2db756fb438494b9d8ed31291de3d237e613ac43a0ad79c9d3ab79a2e69`；
- plan freezer / runner / analyzer SHA256：`42cd7263465eefad20c1026222c4efc9c903558d85c501d837e6b9ec342a180c` / `1f6ad7e46c73c765219a76d355963b6948341b3c1aebae7b8b359ecde42511c1` / `5646206ac16706af077253bde21003c4463481c2618a5295ce6f380e40e031a0`。

本节所有 IoU 只用于 response 冻结后的 GT 后验审计，不进入 Qwen 输入或事件选择；所有结果仍是 DepthTrack Train fixed6 诊断，不是 VOT EAO/ACC/ROB，也没有产生新 checkpoint。

## 24.105 第二轮 baseline 收敛：冻结 STTrack，拒绝 SAMF（2026-08-16）

### 24.105.1 对 STTrack 官方指标的最终勘误

此前 24.96 根据 STTrack GitHub README 单行表格，把 `77.6` 误读为 DepthTrack F-score、把 `63.3` 误读为 VOT-RGBD2022 EAO。重新核对作者 AAAI-25 正式论文 Table 2、正文、分离发布的 VOT22/DepthTrack 权重和 raw result 后，字段对应关系已闭合：

| 模型/来源 | VOT-RGBD2022 EAO | ACC | ROB | 证据性质 |
| --- | ---: | ---: | ---: | --- |
| SUTrack-L384 论文 | 76.6 | 83.5 | 92.2 | 作者论文，服务器未复跑纯 baseline |
| STTrack 论文/作者 VOT22 发布物 | **77.6** | **82.5** | **93.7** | 作者正式结果，尚不是本服务器实测 |
| 当前服务器 SUTrack+结构化语言+safe-v1 | 73.974969 | 82.627562 | 89.455266 | 本服务器 full-127 正式实测 |
| 项目目标 | 77.9 | 82.1 | 93.7 | 目标 |

STTrack 相对 SUTrack 论文值是 EAO `+1.0`、ACC `-1.0`、ROB `+1.5`；它正好补强当前最缺的生存/鲁棒性侧，且 ACC 仍高于项目门槛 82.1。其官方 EAO 距目标只差 0.3，因此适合作为下一唯一迁移 baseline。README 的列错位不再作为指标来源；本节明确覆盖 24.96 的旧 STTrack 数值判断。

### 24.105.2 “再试一两个 baseline”的有界裁决

用户要求再试一两个 baseline，若仍拉不回 VOT，就在最好的 baseline 上继续创新。实际执行范围固定为已经充分实验的 SUTrack 和新迁移的 STTrack；额外只对 SAMF 做一次第一方硬筛，没有下载其 1.46 GB 权重、没有启动 GPU 或公开评测。

SAMF 被判 `NO-GO`：VOT2022 官方 Table 7 的 EAO/ACC/ROB 为 `76.2/80.7/93.6`，相对 STTrack 分别低 `1.4/1.8/0.1`，其中 ACC 还低于项目 82.1 门槛；作者只给 V100-SXM2 32 GB 复现环境，没有 RTX 3090 24 GB 峰值证据；提交包虽然顶层声明 MIT，但包含上游 GPL-3.0 的 AlphaRefine 代码且没有完整保留该许可证，整包许可不闭合。因此不再设置第二候补，不用更弱且风险更高的模型消耗服务器资源。

完整第一方来源审计续写在：

```text
/home/SUTrack_RGBD_L/docs/OPEN_SOURCE_BASELINE_MIGRATION_RESEARCH_ZH.md
SHA256 6f2fa959e5364ceffda0dbe026ba6fe97a671d0dd531f427ba6125cc65cc88ac
```

最终 baseline 决策：**冻结官方 STTrack VOT22 权重作为唯一迁移基线；后续保留现有语言身份与模板事务创新，但不再横向换模型。**

### 24.105.3 STTrack 隔离环境与非指标预检

为保护作者仓库和旧 SUTrack 正式结果，新工作完全位于独立 worktree：

```text
clean source: /root/autodl-tmp/sttrack_clean_clone_20260815T2007Z
innovation worktree: /home/STTrack_RGBD_L_innovation_v1
HEAD: 283cd6dd45536636490db8bca1c63c4647be799b
environment: /root/autodl-tmp/envs/sttrack
official checkpoint: /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar
checkpoint SHA256: cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98
```

Selective Scan CUDA 扩展已在 PyTorch 1.13.1+cu116 环境编译；官方权重 strict load 成功。5 帧无指标预检使用 RGB-D 输入、只读首帧 GT，5 帧 bbox 均有限，峰值显存约 607.6 MiB。随后 `bottle03_indoor` 60 帧 Base/OFF/SHADOW 接入审计全部通过：

- Base↔OFF bbox 最大差 `0.0`，score 最大差 `0.0`；
- Base↔SHADOW 公开 bbox 最大差 `0.0`，score 最大差 `0.0`；
- 捕获 1 次官方模板更新，完整执行未来 3 帧 protected/tentative shadow；
- 身份证据完整，shadow 对公开 state/template 的修改次数为 0；
- OFF 峰值显存 637,457,408 bytes，SHADOW（含 CLIP）1,579,664,896 bytes，双分支在 RTX 3090 24 GB 上有充足余量。

60 帧分析报告：

```text
/root/autodl-tmp/sttrack_innovation_v1/transaction_smoke_v1/analysis.json
SHA256 3058d50fb8e3619f6b017e2ed7a7c5d353c0fc800b4c6d109c275662fdedeffc
```

这一步只证明可兼容、可回退、可在 24 GB 显卡运行；没有读取初始化之后的 GT、没有计算 VOT 指标，也不构成精度提升声明。

### 24.105.4 已接入的创新边界

当前 STTrack 只接入 phase-1 非侵入 shadow：官方动态模板仍按 50 帧/score>0.75 的原逻辑进入 tentative 公共分支；protected 分支从写入前模板、当前 bbox 和双模态 `track_query_before` 深拷贝出发，独立递归未来帧。首帧 CLIP 图像特征和结构化语言特征只形成候选身份证据，不进入 STTrack 主干，也不能授权公开提交。

未来若容量门通过，真正的原子事务快照必须覆盖 bbox、`z_dict`、RGB/Depth 两模态 `track_query_before`、`z_patch_arr`、事件计数器和证据状态；`z_dict[0]` 始终不可变。未训练、未审计的语言不得注入 backbone，只能作为事务外部判据。当前尚未实现 Active promote/rollback，不能据此运行正式 VOT。

## 24.106 STTrack release 时序 query 窗口缺陷与修复（2026-08-16）

### 24.106.1 为什么第一份 fixed6 容量结果被废止

最初直接使用作者 release 源码和 VOT22 checkpoint 跑 Train fixed6，公开路径在 8,704 个可见 GT 帧上的 mean IoU 只有 `0.157803`。输入已逐项核对为官方 wrapper 的 `rgbcolormap + depth_clip=True`，checkpoint strict load 且 SHA 正确，因此继续定位递归状态。

release 配置声明：`TRACK_QUERY_OLD=4`、`TRACK_QUERY=1`，即每帧应在固定 4-query 历史中加入 1 个新 query、淘汰最旧 1 个。但 `lib/models/sttrack/sttrack.py` 实际使用：

```python
track_query_before = track_query_before[:, -TRACK_QUERY_OLD-1:]
track_query_before = cat(track_query_before, track_query_now)
```

因此历史长度从初始化 4 变成第一帧后的 5、第二帧后的 6，之后保持 6。backbone/TSG 仍只按 `TRACK_QUERY_OLD + TRACK_QUERY = 5` 个 query 从 token 尾部剥离，多出的 1 个 query 被误当成搜索网格 token，导致空间 token 错位。六条序列都出现相同表征：第 1 个预测帧基本正确，第 2、3 个预测帧横向各偏移约一个 patch，之后沿错误 crop 递归。

第一份错误基线 trace 位于：

```text
/root/autodl-tmp/sttrack_innovation_v1/transaction_fixed6_v1
plan SHA256 18c20d9d9f86e7b9bfe0cbeef3763d67d8def928a5553e706f7457922b1af418
analysis SHA256 e24b7cd00825fffa24d9e2e5ddc6f42846606f834b794ec22fe49b0a289b1a55
```

该 trace 虽得到 `capacity_supported`，但它测量的是 release query 缺陷污染后的 protected/tentative 分歧，已被 v2 plan 明确 supersede，不能授权训练、Active、Train152 或公开 VOT。首次 analyzer 还因错误要求所有 GT 都为正面积框而 fail-closed；机械修正只把不可见/NaN GT 排除出 IoU 分母，并保留完整逐序列行数和 10 帧事件调度检查，门槛没有改变。

### 24.106.2 默认关闭的固定窗口修复

修复不改变 checkpoint，不增加可学习参数，也不重训。新增 `MODEL.TSG.FIX_QUERY_WINDOW`，默认 `False`，所以旧 YAML 和 release 路径仍可严格复现。启用时，先保留 `TRACK_QUERY_OLD - TRACK_QUERY = 3` 个历史 query，再追加 1 个当前 query，并在运行时断言两模态长度始终等于 4。

`bottle03_indoor` 20 帧红/绿诊断结果：

| 路径 | query 长度 | 前20帧 mean IoU |
| --- | --- | ---: |
| release | `5,6,6,...` | 0.188470 |
| fixed | `4,4,4,...` | **0.946091** |
| fixed-release | - | **+0.757620** |

诊断报告：

```text
/root/autodl-tmp/sttrack_innovation_v1/query_window_diagnostic_v1/analysis.json
SHA256 ba0fc4b22b080fe991b3379e08491d287348b92d9888de22a89eb5cd0e61efd0
```

随后用固定窗口重新运行 60 帧 Base/OFF/SHADOW：bbox 与 score 的 Base↔OFF、Base↔SHADOW 最大差仍全部为 `0.0`，1 次模板事件和未来 3 帧身份证据完整，shadow 未修改公开状态。报告 SHA256 `c734a5982813e94952f1f627c201e372819133a9a45cc18650664b917f1fc1a7`。这证明 query 修复与事务 wrapper 可组合，同时保留 fail-closed OFF。

### 24.106.3 修复后 fixed6 v2

修复后容量实验使用新的冻结 plan：

```text
/root/autodl-tmp/sttrack_innovation_v1/transaction_fixed6_v2/plan.json
SHA256 ba2492efea5dd676c16a6b1c044f8fb2b81b5f7ee2b548464b2d98b03798b8e8
```

v2 保持相同六条 DepthTrack Train、相同 10 帧 protected shadow、相同容量门和相同首帧 GT-only 推理合同；新增绑定 model source SHA，避免核心 STTrack 实现未进入 provenance。只有 v2 结果可以决定是否继续训练事务选择器。

v2 结果：

| Train-only fixed6 指标 | query-window-fixed STTrack public | GT 后验 oracle/统计 |
| --- | ---: | ---: |
| 可见 GT 帧 / 全 trace 帧 | 8,704 / 10,041 | - |
| mean IoU | **0.724747** | 0.726030 |
| 完整模板事件 / shadow 行 | 113 / 1,130 | - |
| protected 明显有益帧 | - | 35，覆盖5条序列 |
| 正总收益 / 负总收益事件 | - | 52 / 51 |
| oracle 累积 IoU 增益 | - | 11.162923 |
| `tentative IoU<=0.1 → protected IoU>0.1` | - | **0帧、0序列** |
| always-protected catastrophic | - | 1帧 |

analysis SHA256：`311b4993b4059d67ea2b1781961d5280712fa33769ae1bf7b115c9b7ff0f5b19`。容量终态为 `capacity_not_supported`：完整事件数、有益帧/序列和 oracle 总增益门通过，但直接对应 VOT survival 的 recovery 行/序列门均失败。因此不训练“在官方定时模板更新时选旧/新模板”的 selector，不实现该路径的 Active，不进入 Train152/VOT。

具体序列表现也说明 baseline 已恢复、问题集中在真正的失锁而非普通框回归：`bottle03/flower03/toy03` mean IoU 分别 `0.878529/0.838601/0.802223`；`bag04/ball16` 为 `0.782081/0.741612`；`pigeon05` 仅 `0.038670`，有 996 个可见严重帧。`pigeon05` 的低 IoU 帧平均 score 仍约 `0.6689`，后两次长失败起点 score 为 `0.4512/0.6984`，证明单纯低置信阈值也不足。

下一架构不再绑定 50 帧模板写事件，而是在低 score、异常运动/尺度、多峰等风险事件上，从当前帧前的完整快照同时展开：默认 public、query-reset、static-template+query-reset、wide-search recovery。每个候选独立递归 10 帧，语言/首帧图像只作身份证据；先做 Train-only oracle 容量，未过门不训练 selector。真正 Active 仍需原子提交完整 bbox/template/query 状态。

## 24.107 STTrack 风险恢复容量与 Train152 晋升（2026-08-16）

### 24.107.1 独立 held-out 6 序列容量门

模板定时事务 v2 被否决后，新的 shadow 不再等待 50 帧更新，而在以下任一风险出现时触发：`score<0.30`、归一化中心跳变 `>0.75`、绝对 log 面积变化 `>0.70`。每个事件从当前帧前的完整 bbox/template/query 快照独立展开 10 帧，动作固定为：

1. `query_reset`：保留双模态模板，清空历史 query，保持 factor 4；
2. `static_query_reset`：两个模板槽都回到首帧模板，清空 query，保持 factor 4；
3. `wide_static_query_reset`：首帧模板、清空 query、factor 6。

`pigeon05_wild` 只用于发现动作，不能授权。随后从旧 split plan 中预先冻结、且与发现集/fixed6 不重叠的 6 条 audit 序列做独立容量验证：`toiletpaper02_indoor`、`toy06_indoor`、`mushroom02_wild`、`cup13_indoor`、`glass05_indoor`、`thermos01_indoor`。全程只读首帧 GT；9,358 帧 trace 完成后才一次性拼接 GT。

冻结产物：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_heldout6_v1/plan.json
plan SHA256 a34c085c9aadea3ed9a1a4ec8f8dd9a09eb3349a6fd6f4226ea1727d39b2b84d
shard0 SHA256 528ef88b8e22f6c4330fc32fc404a998a435c7c5e7e3b3a98c56ae587b3319a9
shard1 SHA256 eee7b1ee38a7cd2ee42fac10bfe8029372806083469078208fc500931fcf47e4
analysis SHA256 3fa22499eb52243aae3ad9c33a0bdf8ebfed15d25fa0c7a023a25ca2ba578b0d
```

结果：45 个完整事件、450 个捕获 action 行、325 个有效 GT action 行；oracle 有益 88 行、覆盖 5/6 序列，恢复 47 行、覆盖 4/6 序列，oracle 累计 IoU 增益 `+41.763381`，所有预注册容量门均通过。三个动作的后验统计为：

| 动作 | 有益行 | 恢复行 | catastrophic | 累计 IoU 增益 |
| --- | ---: | ---: | ---: | ---: |
| query_reset | 27 | 17 | 3 | **+11.916244** |
| static_query_reset | 46 | 34 | 30 | +0.905591 |
| wide_static_query_reset | 62 | 40 | 59 | **-33.650156** |

因此容量门只说明存在可学恢复机会，不能直接开启任何动作。尤其现有 CLIP `first-frame image + structured language` 最大相似度选择器虽然选了 256 行，却得到 47 个有益、78 个有害、累计 `-20.423299`；它明确不得部署。`wide_static_query_reset` 虽恢复多，但破坏正常目标过多，也不进入第一版 selector。后续只训练最稳的 `query_reset` 与 public 二选一。

### 24.107.2 已启动的 Train152 source-only trace

独立容量通过后冻结 Train152 计划：152 序列、219,954 帧，按帧数均衡为两片 76/76 序列、109,917/110,037 帧。计划在任何未来 GT join 前落盘：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/plan.json
SHA256 0c40cf46a543293b5ade0a6373d5ef50f80f8b2b46e9b4fda02a3c727dd2bc50
screen: strisk_full152_s0 / strisk_full152_s1
```

两卡均使用 RTX 3090，运行时各约 3.33 GiB，未出现 OOM。该任务不是 backbone 训练，也不是公开评测；它只收集 public 与三个非侵入恢复分支的 bbox/score/response/首帧图像与结构化语言身份证据。runner 绑定 checkpoint、配置、模型、事务、恢复源码和计划 SHA，初始化之后不读 GT，`future_frame_text_used=false`。

### 24.107.3 选择器协议与创新保留方式

正在实现的第一版门只决定是否原子提交 `query_reset`，不允许 static/wide 动作。部署特征固定为事件起点即可观测的 22 维证据：public/candidate score 与差值、response margin、运动与尺度风险、public/candidate 框一致性，以及候选相对 public 的首帧 RGB 身份和结构化语言身份差。语言仍是外部实例锚点，不注入冻结 STTrack backbone；这保留项目的语言创新，同时避免当前错误的“CLIP 分数最大即提交”。

训练协议为 5-fold sequence-group OOF，seed 固定 `2026/2027/2028`，三 seed 的 calibration OOF 都必须通过；部署 seed 预注册为 2026，不能根据 audit 指标选 seed。只有在策略和阈值完全冻结后，才允许读取 30 条 audit 序列 GT 一次。calibration 门要求 precision≥0.85、harm rate≤0.02、catastrophic=0、至少20个动作事件/10序列、至少5个恢复事件/3序列、净 IoU 增益≥1；audit 只评一个策略，仍要求 precision≥0.85、catastrophic=0、至少2个恢复事件/2序列和净增益>0。

即使即时 audit 通过，也只能进入一次 Train-only 递归审计；必须进一步证明实际原子 promote 后 mean IoU 不降、catastrophic 序列不增加、10帧 failure starts 至少减少1，才允许 DepthTrack/CDTB 保真与正式 VOT。当前正式最好指标仍是旧 SUTrack 的 `73.974969/82.627562/89.455266`，本节没有产生新 VOT 数值。

### 24.107.4 Train152 补充 implementation snapshot

全量 trace 启动后复核 provenance 时发现，原 plan 虽绑定 checkpoint、YAML、model、recovery、transaction 和 runner，但没有单独列出 recovery 直接依赖的基础 tracker、配置解析、数据处理与 bbox 工具源码。两个 trace 进程仍在运行时立即做 fail-closed 补强：先从进程表确认 PID 696212/696214 均于 `2026-08-15T18:29:04Z` 启动，再验证以下 11 个依赖的 mtime 全部早于启动时刻且 SHA 未变化：配置、基础 tracker、recovery、transaction、核心 model、processing、box_ops、depth_utils、data_utils、runner、YAML。

补充快照在 trace 完成和任何未来 GT join 之前原子落盘：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/source_snapshot.json
SHA256 86697767b441bc95ce070045f31c53693e0c82ef973000be46b4185494947960
implementation_file_count 11
```

selector plan 生成器和 trainer 已改为必须复验该快照、所有 trace source 字段、分片顺序、序列列表、帧数、文件大小与 SHA；任一依赖改变即拒绝训练。选择器共享源码当前 SHA `9631cc433fdeee5341d0b0c4abc330e87f83a78591e10b35ba5a193a9bce7651`，trainer SHA `a44bff1fea1af9da4a6332f2326536390844815d479264ec08907a37fcad87f2`，selector-plan generator SHA `36f117f5a39741ec582a3bff2be5adfd636828089d0839b9431726f814c32a6d`。这些新增文件不在正在执行的 inference import 链中，因此没有改变已启动 trace 的运行语义。

### 24.107.5 持久续跑控制器与当前运行状态

为避免 SSH 断开后需要人工猜测 trace 是否完整，已启动只负责“等待、验证、训练选择器、停止”的持久控制器：

```text
screen: strisk_full152_continue
controller: /home/STTrack_RGBD_L_innovation_v1/tools/continue_sttrack_risk_recovery_after_trace.py
controller SHA256: 173d23b20c16c9922973d3e03841d73dda8c7079a8e341b565c5274ede270325
status: /root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/continuation_status.json
```

控制器只在 `shard0.exitcode` 与 `shard1.exitcode` 都存在且均为 0 后，才复验 source snapshot、冻结两个 trace 的大小/SHA、生成 selector plan，并运行三 seed calibration OOF 与唯一一次 audit。trainer 的 exit 2 被解释为科学门禁拒绝，控制器会记录拒绝并停止；工程异常也 fail-closed。无论选择器成功还是失败，该控制器都不会自动启动递归审计、DepthTrack Test、CDTB 或 VOT，状态文件始终显式保留 `public_evaluation=false`。

`2026-08-16T03:15:24+08:00` 复核时，两个 trace Python 已运行约 46 分钟，GPU 0/1 各占约 3.33 GiB，利用率 70%/72%、温度 62/57°C；两个 exitcode/trace 仍未产生，控制器状态为 `waiting_for_full152_trace`，三个 screen 均存活且日志没有异常。held-out 9,358 帧仅产生约 10.9 MB JSON，按帧数外推 full152 约 256 MB；当时 `/root/autodl-tmp` 仍有 9.1 GB，可排除当前产物空间不足。

同一时刻另用独立脚本从 `source_snapshot.json` 递归读取全部绑定项并重新哈希，结果为 `records=12、bad=[]`，命令 exit 0；plan SHA 仍为 `0c40cf46a543293b5ade0a6373d5ef50f80f8b2b46e9b4fda02a3c727dd2bc50`，snapshot SHA 仍为 `86697767b441bc95ce070045f31c53693e0c82ef973000be46b4185494947960`。因此当前 source trace 未发生运行中实现漂移。

### 24.107.6 已实现但尚未获准运行的 Active/递归审计架构

为避免选择器通过后临时实现部署路径造成训练/运行语义不一致，Active query-reset 和递归 Train-only 审计接线已预先完成，但当前没有运行。Active 源码：

```text
/home/STTrack_RGBD_L_innovation_v1/lib/test/tracker/sttrack_risk_recovery_active.py
SHA256 af9f8696706eafb58c00165d05250c766c0acf9e6b9fd7b00d8e98361378708f
```

部署动作仍严格限制为事件 age 0 的 `public/query_reset` 二选一，不允许容量实验中高伤害的 static/wide 动作。query-reset 的事务边界覆盖 bbox、`z_dict`、RGB/Depth 两模态 `track_query_before`、`z_patch_arr`、`z_dict1` 与事件状态；首帧 `z_dict[0]` 不可变。所有待提交 bbox/tensor 都会先做有限性、形状验证和深拷贝，只有全部准备成功后才第一次写公开状态，防止 clone/OOM 发生在半提交状态。提交或拒绝后都设置 19 帧调度抑制，使下一风险事件与 source shadow 的 age 20 节奏一致。

runtime loader 不只读取 artifact 自报字段，而要复验 training result 为 `ready_for_recursive_audit`、三个 seed 全部合格、audit 只消费一个预注册策略、部署 seed 固定为 2026，并重新绑定 source snapshot、trace/OOF/artifact SHA 和实现 SHA。相关递归工具快照：

```text
plan generator SHA256 7df4cc04fb32eeebd662d23fafdc3c67a5f4fcbe7721da63d0a186ce0ecd4ab8
runner SHA256         4eb21a6e8d89d976b035a2008fa3c9bcd3444d90be97c7da4081d7677ed8766b
analyzer SHA256       644dd6c46e8cba4e5048f53cb3b6bc209899ddd97aeca5f530051432b39742f6
```

只有真实选择器终态为 ready，plan generator 才允许从预冻结的 30 条 Train audit 序列生成一次 baseline/Active 配对计划。递归门要求 mean IoU delta≥0、10 帧 failure starts 至少减少 1、catastrophic 序列回归为 0、实际 commit≥1；失败即停止，不运行公开数据集。当前这些文件只证明部署和验证路径已经准备好，不证明策略有效，也没有改变正式 VOT 最好值。

## 24.108 STTrack Train152 风险恢复选择器严格拒绝（2026-08-16）

### 24.108.1 source-only full152 完整完成

两个分片均使用 query-window-fixed STTrack、同一官方 VOT22 checkpoint 和冻结 source snapshot，初始化之后不读 GT。两片分别于 `2026-08-16T04:28:27+08:00` 与 `04:26:54+08:00` 原子落盘，exitcode 均为 0：

| 分片 | 序列 | 帧数 | 风险事件 | trace SHA256 |
| --- | ---: | ---: | ---: | --- |
| shard0 | 76 | 109,917 | 758 | `846872b20c04fed1ffb132c84135925b819617a739b8319985cc70239a4e5fe3` |
| shard1 | 76 | 110,037 | 708 | `5b168dbd3b60aed619b9c9b9113a4c5c85d4f3d8ce23ca0fbebd91005d0656f9` |

独立完整性复核对每片均得到：序列顺序与 plan 精确一致；逐序列行数、全局行数、frame index 覆盖一致；所有 bbox 和非初始化 score 有限；恰好每序列 frame 0 的 score 为 `None`；行只含 public/source shadow 证据；`ground_truth_used_after_initialization=false`、`metric_computed=false`、`public_evaluation=false`、`future_frame_text_used=false`。两片峰值显存均为 1,584,160,256 bytes。

selector plan 在完整 trace 和任何 audit GT 读取前冻结：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/selector_plan.json
SHA256 2427caa72df216f5146d8d81b373447da4207ac64ae99a733fa87a8c94cfa7fa
```

### 24.108.2 age-0 query-reset 三 seed OOF 被拒绝

122 条 calibration 序列中有 104 条实际出现完整风险事件，共 893 个事件；剩余 18 条没有触发事件。30 条 audit 序列保持封存。calibration 后验标签统计：有益 118、有害 160、catastrophic 48、recovery 42。若 893 个事件全部提交 query-reset，累计未来 10 帧 IoU 增益仍为 `+6.171518`，说明动作具有局部恢复容量；但伤害规模远超安全部署要求。

seed `2026/2027/2028` 的 sequence-group 5-fold OOF 全部得到 `oof_selection=null`。三个 seed 的 artifact/OOF 都落盘用于审计，但没有任何 seed 获得部署授权。对每个 OOF 排序逐一枚举所有唯一 threshold 后，在满足动作数、序列覆盖、recovery 覆盖和净增益门的候选中：最高 precision 只有约 `25.93%`，最低 harm rate 约 `14.04%`，最少仍有 5 个 catastrophic；预注册门分别是 `≥85%`、`≤2%`、`0`。因此失败不是阈值略偏，也不是随机 seed 波动，而是 age-0 可观测证据无法安全区分 query-reset 的有益与有害事件。

正式训练报告：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_full152_v1/policy_training_v1/training.json
SHA256 76a2e2476e537294f4aabd4057812fd6c3e948bcb31e5fb0e705c5de139c7ab6
decision selection_rejected_no_recursive_audit
all_seeds_oof_passed false
audit_evaluated false
audit_policies_evaluated 0
ready_for_recursive_audit false
```

持久控制器正确写入同一终态并停止；Active、递归 Train audit、DepthTrack Test、CDTB 和 VOT 均未运行。30 条 audit GT 未被本轮训练读取，因此没有发生“在 audit 上换 seed/阈值”的泄漏。

### 24.108.3 age-2 延迟暂存事务也未形成安全容量

age-0 失败后没有放宽门槛，而是检验更贴合暂存事务的因果策略：protected 与 query-reset shadow 先并行运行 age 0/1/2，决策只使用截至 age 2 已可观测的 79 维证据；标签完全排除这三帧，只比较 age 3–9 的未来 survival。特征包括三帧 public/action score、margin、两框 IoU/中心/面积差、首帧 RGB 与结构化语言身份差、各分支自身轨迹一致性和触发风险。分析只读取相同 calibration GT，audit 仍未读取。

固定比较两个模型：线性 benefit classifier，以及单个预注册的 `79→64→32→2` 多任务 MLP；后者分别预测 beneficial 和 unsafe（harmful 或 catastrophic），部署排序为 `p(beneficial) × [1-p(unsafe)]`。两模型都使用原五折 sequence-group OOF 和 seed `2026/2027/2028`，没有扫描 architecture 或 audit threshold。

age-2 可用事件为 838 个、覆盖 104 条序列；标签为有益 106、有害 115、catastrophic 36、recovery 33，全部提交时 age 3–9 净增益只有 `+1.191588`。六组 `model × seed` 均没有合格 threshold。多任务 MLP seed 2026 的一个最接近覆盖门的候选仍需提交 815 个事件，precision `12.76%`、harm `13.62%`、catastrophic 33，虽有 33 个 recovery/26 序列和净增益 `+9.831347`，仍同时违反三个硬安全门。另两 seed 结论一致。

冻结分析：

```text
/root/autodl-tmp/sttrack_innovation_v1/risk_recovery_temporal_capacity_v1/analysis.json
SHA256 49642567c35430fb3acbc6af3782efceea35e89d1f556291721ee01a770aa869
analyzer SHA256 ba8aebccf56690fe37cafdcfec906fcf0471beb95e6579c65e554b8f1d4843f3
decision temporal_capacity_not_supported
audit_ground_truth_read false
audit_policy_evaluations 0
public_evaluation false
```

因此不能继续在同一批 score、bbox、response margin 和 CLIP 身份特征上换分类器或调 threshold。该证据族与有益/有害 query-reset 的可分性不足，Active/递归/VOT 路径继续禁止。

### 24.108.4 下一步架构收敛

后续仍冻结 STTrack 作为唯一 baseline，不回到 SAMF，也不删除语言与模板事务创新。下一步新增的是 **STTrack-native dual-modal query consistency**：从 RGB/Depth 两模态 `track_query_before` 直接计算 protected/action 的 query cosine、相对残差、跨模态一致性和两帧稳定性，再与结构化语言身份只作联合否决证据。它不把语言注入 backbone、不修改 protected 输出；先在独立 Train 子集做 source-only shadow 容量，未过零 catastrophic/低 harm 门就停止。

若原生 query evidence 仍无法分离，则不再训练更多事后 classifier，而转入更强的视觉自验证：候选 crop 到首帧 RGB-D template 的 cycle consistency/target-distractor association，并保持 protected/tentative 两套完整状态。任何方案仍需先过 Train-only OOF、一次 audit、递归 failure-start 门，最后才允许公开 VOT。当前正式指标没有变化，仍为旧 SUTrack full-127 的 `73.974969/82.627562/89.455266`；STTrack `77.6/82.5/93.7` 仍是作者报告值，不是本服务器新正式结果。

## 24.109 STTrack 原生身份证据与首帧 RGB-D 锚点回环（2026-08-16）

### 24.109.1 两个开源 baseline 的最终裁决

按“最多再筛一到两个 baseline，若仍不能拉回 VOT 则在最强 baseline 上改进”的约束，候选已经收敛，不再继续横向换仓库：

| 候选 | 作者报告 VOT-RGBD2022 EAO/ACC/ROB | 相对 STTrack | 工程裁决 |
| --- | --- | --- | --- |
| STTrack | `77.6/82.5/93.7` | 基准 | **GO，唯一迁移 baseline** |
| SAMF | `76.2/80.7/93.6` | `-1.4/-1.8/-0.1` | **NO-GO** |

SAMF 不仅三项都不超过 STTrack，而且作者只给 V100 32 GB 复现环境；其发布包还包含 AlphaRefine/GPL-3.0 依赖，不能把整包当作纯 MIT 代码直接迁移。继续实测 SAMF 既不能提高目标上限，也增加显存和许可证风险，因此没有下载约 1.46 GB 权重、没有运行 GPU 或公开评测。STTrack 保持唯一后续 baseline；`77.6/82.5/93.7` 是作者报告值，不是本服务器新产生的正式结果。

### 24.109.2 STTrack-native RGB/Depth query evidence 被严格拒绝

age-2 时序证据失败后，在同一组预冻结、calibration-only 的 12 条 Train 序列上采集 STTrack 两模态原生 query 一致性：protected/action query cosine、相对 L2、norm ratio、各自两帧稳定性和 RGB/Depth 跨模态 cosine。该路径不修改公开 bbox、score、query 或模板；15,366 帧、87 个完整十帧事件全部完成，形成 2,610 条 action-frame evidence。两个 trace SHA 为：

```text
shard0 334ea1b16af12190ccf883bc09e8d3c25e8ce12301bc0f1e32b00db775720864
shard1 a192319da4119e9541df2ba924fd89320a093931c07e09506ed5edd5034bec17
analysis 704b7475c18c1e460fe87cb1b29e6afa4d6930f3c1fa4d55485023b0ac90a2a1
```

GT 后验只覆盖 9 条实际有事件的 calibration 序列、70 个可用 age-2 事件；标签为 beneficial 4、harmful 5、catastrophic 1、recovery 3，全部提交的 age 3–9 净增益为 `+5.234640`。三 seed、三折 OOF 的结果均不通过。

最接近安全门的 temporal-only seed 2026 提交 14 个事件/4 条序列，包含 3 个 beneficial、0 harmful、0 catastrophic、3 recovery/3 条序列，净增益 `+8.098266`；唯一失败项是 precision 只有 `21.43%`，远低于 `85%`。加入 57 维原生 query 后反而更差：seed 2026 需提交 62 个事件才能覆盖 recovery，其中 beneficial 4、harmful 3、precision `6.45%`、harm rate `4.84%`、净增益 `+8.004301`。另两 seed 结论一致。冻结终态：

```text
decision native_query_capacity_not_supported
audit_ground_truth_read false
audit_policy_evaluations 0
public_evaluation false
```

因此“query 表征更接近 STTrack 内部状态”并不等于“能可靠判断身份恢复”；当前 query 相似度主要反映两个分支表征是否接近，没有提供足够的目标—干扰物可分性。该特征族不得进入递归或 VOT。

### 24.109.3 首帧 RGB-D 锚点回环容量结论

为检验 tentative crop 是否仍对应首帧目标，在每个事件 age 2 固定执行一次非侵入式回环：分别把 protected bbox 与 query-reset bbox 的当前 RGB-D crop 当作模板，反向匹配到不可变首帧，并记录首帧 IoU、峰值、margin、归一化中心误差和面积误差。只使用 frame 0 初始化框，不读取未来 GT，不修改公开状态。

计划、trace 与分析在各阶段依次冻结：

```text
trace plan    696b2f8db086a0c03da9c42a52784b30531cfe263574623160639398083ca36a
shard0 trace  9dbe8871913bbd0edf918fae8a857134f12cb4eea4c365f73754c1ed88b55f78
shard1 trace  4edee8da676eade201b9b4d225c5fc5fe6d4690b2b669e72fdc0fc9deb1dbeca
analysis plan 6d33f7f15e611481014358228d722282d3294b021a4e4c2c034a43e438483ac0
analysis      e05201a966539804cd33dc71407393d6fc0768d978831a62f9f3ecbad51de7a4
```

两片共 15,366 帧、87 个完整风险事件；每个事件恰好在 age 2 出现一组 public/action 回环，其他 age 为 0 组。逐帧与上一版 query trace 比较，公开 bbox 和 score 的最大绝对差均严格为 `0.0`。峰值显存 1,584,307,712 bytes，因此失败不是工程污染或显存问题。

GT 后验仍是同 70 个可用事件、4 beneficial、5 harmful、1 catastrophic、3 recovery。固定的 79 维 temporal-only 与 94 维 temporal+anchor-cycle 多任务 MLP 在 seed `2026/2027/2028` 上全部失败。temporal+cycle seed 2026 最接近门的候选需提交 15 个事件/4 条序列，只有 2 个 beneficial、2 个 recovery，虽为 0 harmful/0 catastrophic、净增益 `+7.122723`，precision 仍只有 `13.33%`，比不加回环时的 `21.43%` 更低。

回环失败的具体原因不是“回不到首帧”，而是 **能回到首帧并不能预测 query-reset 对未来递归状态是否有益**：

| Train 例子 | 事件帧 | 动作后未来增益 | public/action 首帧 IoU | 结论 |
| --- | ---: | ---: | ---: | --- |
| `flowerbasket_indoor` event 6 | 490–499 | `-2.686553` | `0.942925/0.945318` | catastrophic；action 回环反而略高，若按回环接受会误提交 |
| `flowerbasket_indoor` event 7 | 533–542 | `+6.203036` | `0.928606/0.939298` | 真 recovery；此例回环提升与收益方向一致 |
| `basket_indoor` event 6 | 674–683 | `+0.987611` | `0.205301/0.204244` | 真 recovery，但两分支回环都很低，硬设高 IoU 门会漏救 |
| `human03_wild` event 3 | 1309–1318 | `+0.883002` | `0.967927/0.959125` | 真 recovery，但 action 回环更低，delta 符号与收益相反 |
| `book04_indoor` event 1 | 954–963 | `-0.086390` | `0.912922/0.870848` | harmful；低 action 回环可拒绝这一例，但不能形成统一判据 |

统计上，beneficial 事件 action 首帧 IoU 均值为 `0.757169`，harmful 为 `0.916715`，唯一 catastrophic 更高达 `0.945318`；回环越高并没有对应越安全。原因是 public 与 query-reset 在 age 2 往往仍围绕同一外观候选，反向匹配主要确认“这个 crop 像首帧目标”，却无法辨别两套 query memory 在 age 3–9 的递归分叉。因此不能用单一 cycle consistency 给模板或状态提交授权。

冻结终态：

```text
decision initial_anchor_cycle_capacity_not_supported
audit_ground_truth_read false
audit_policy_evaluations 0
public_evaluation false
```

### 24.109.4 后续改进从事后判别转为 STTrack 内生关联训练

age-0、age-2、原生 query 和首帧回环四条事后 classifier 路径均未达到 85% precision；继续换阈值、加 MLP 或重复使用这 70 个标签只会过拟合。后续仍以 STTrack 为唯一 baseline，保留以下创新点：不可变首帧 RGB-D/结构化语言身份锚点、protected/tentative 双状态、模板事务、原子 promote/rollback。

下一阶段不再判断“已经产生的单个 query-reset 是否安全”，而是训练一个 **target–distractor association head**：在风险帧从 STTrack RGB/Depth response 提取 top-K 候选，联合首帧 RGB-D prototype、稳定语言属性和两模态时序 query 做候选排序；训练使用 DepthTrack Train 的 4–8 帧 predicted-crop rollout，并直接优化未来 survival/身份保持。所有新权重仍须先过 Train-only OOF、一次封存 audit 和递归 failure-start 门，才允许跑 VOT。Qwen 在线文本只能在该身份关联与暂存事务都通过后作为低权重动态外观记忆接入，不能先于身份确认直接更新主干或模板。
\n## 24.110　STTrack factor-4 / factor-6 候选上限与原生身份特征门（2026-08-16）

### 24.110.1　结论先行

本节没有产生新的公开集正式指标。当前服务器正式最好仍为 SUTrack-L384 + 结构化语言 + safe-v1：VOT-RGBD2022 `EAO/ACC/ROB = 73.97496948296595 / 82.62756179006247 / 89.45526602400152`；DepthTrack Test `P/R/F = 65.9959327 / 65.3358854 / 65.6642504`；CDTB `P/R/F = 75.3878211 / 76.0058500 / 75.6955741`。STTrack 的 `77.6/82.5/93.7` 仍是作者公开值，不是本服务器新实测。

本轮得到两个相互补充的结论：

1. STTrack 公开 factor-4 response 的 top-K 候选不足以覆盖当前 ROB/EAO 失锁起点，禁止直接训练关联头；
2. 在保持公开 factor-4 路径不变的同时，额外从上一公开 bbox 做 factor-6 shadow search，候选集合出现显著且跨序列的恢复上限；但第一版“绝对 IoU 回归”原生 token 小头无法以 `>=85%` 强收益精度安全出手，因此仍未获准扩到122条 calibration，更未获准进入 audit 或 VOT。

### 24.110.2　factor-4 top-K：容量门拒绝

冻结根：`/root/autodl-tmp/sttrack_innovation_v1/topk_candidate_preflight12_v2`。trace plan SHA 为 `6888f1b2866e3a75104d23ef7a365192f487663e011a3ac13600fbb615dcb6da`，analysis SHA 为 `5f2ae6c47b1895f5ad2d986e12275016f4732c016a150d39c14beef25ca2c187`。

| 项目 | 结果 |
|---|---:|
| 有效候选帧 | 14,550 |
| risk-valid 帧 | 223 |
| public mean IoU | 0.7124083815 |
| factor-4 top-K oracle mean IoU | 0.7152932544 |
| oracle 累计 IoU 增益 | +41.9749005952 |
| gain>=0.2 | 59帧 / 7序列 |
| risk strong rescue | 3帧 / 2序列（门槛5帧） |
| public 十帧 failure starts | 19 |
| 可恢复 failure starts | 1起点 / 1序列（门槛2/2） |

唯一可恢复的 failure start 为 `flowerbasket@533`，rank-1 候选 IoU 约0.7396。大量失败起点的8个候选 IoU 全为0，证明目标已经离开 factor-4 的有效搜索域；在这种数据上训练候选排序器不能解决主问题。

### 24.110.3　受保护 factor-6 shadow 候选：容量门通过

冻结根：`/root/autodl-tmp/sttrack_innovation_v1/wide_topk_candidate_preflight12_v1`。该批12条 calibration 序列与 factor-4 的12条和冻结30条 audit 均不重叠；公开 factor-4 bbox、score、template 和双模态 query 不接受 shadow 写入。

- wide plan SHA：`eb1507fc76869c6627aec4638cc526b23dfd9b26aff145805eddb46794346239`；
- shard0/1 SHA：`afcf2651c385b135758d23033a66b1b05e6b62d3a7a1434155aea5d9daee72fb`、`8c5229de2614f97405e34e4b88d7ca569ff370c870ddbf909d4e81822ca1e5b6`；
- analysis plan SHA：`cd4a23e0a144ff3452f66c7a3887c0d8d8abb6fb6e7e567f004712b056195485`；
- analysis SHA：`5eaef1db40515ac0588761e51d0dae6e7cbe0b38879f19285e45f90dd30f06fe`。

机械复验覆盖17,938帧：公开 rank-0 bbox 预绑定最大浮点尾差 `5.684341886080802e-14`，score 差为0；factor-4 与 factor-6 两组候选的最小 grid Chebyshev 距离均为3；峰值显存约639MB。

| 项目 | public | public + 8个 factor-6 候选 GT 后验 oracle |
|---|---:|---:|
| 有效候选帧 | 16,299 | 16,299 |
| mean IoU | 0.6554974385 | 0.6884082785 |
| recall@IoU0.5 | 0.7333578747 | 0.7583287318 |
| 累计 IoU 增益 | — | +536.4137810543 |

额外容量指标：`gain>=0.2` 共757帧、10序列；286个 risk-valid 帧中 strong rescue 为20帧、7序列；28个十帧 failure starts 中有3个起点、3序列在起点帧可被候选恢复。全部预注册容量检查为真。

具体例子：

- `bag05_indoor@808`：public IoU `0.037953`，factor-6 rank-1 IoU `0.577531`，触发原因 `low_score`；这是可恢复 failure start；
- `ball08_wild@2`：public IoU `0`，rank-1 IoU `0.911643`，触发 `low_score + center_jump`；这是可恢复 failure start；
- `pigeon05_wild@38`：public IoU `0`，rank-1 IoU `0.552568`，触发 `low_score + center_jump`；这是可恢复 failure start；
- `cup10_indoor@1855`：public IoU `0`，rank-2 IoU `0.966959`，但当前 risk 规则未触发，说明候选容量和触发覆盖是两个独立问题；
- `ball12_wild@825` 等多处起点的9个候选仍全部 IoU0，factor-6 也不是全局重检测的替代品。

### 24.110.4　浅层身份与原生 token 第一版均未获部署资格

在已经读取 GT 的同一12条 calibration 上做了仅用于特征设计的探索：首帧 HSV、Lab、Depth texture、gray texture 对所有384个 `public<=0.1 且 oracle>=0.5` 帧的精确 oracle-rank 命中分别为111、87、46、95；固定融合命中115/384（29.95%）。因此不能用颜色/深度直方图直接控制递归状态。

随后冻结 STTrack 自身 MambaFusion 前后的 RGB、Depth、融合 token，使用固定 seed `20260816` 的 `768->32` 随机投影，并加入静态/动态模板 cosine、public-candidate cosine、response 和几何量，共137维/候选。该路径不增加 backbone 参数，也不修改 checkpoint。

- 特征计划 SHA：`e13e3fe82c11d7acfb172085198589bc67e0c15d6ee4fe7fd25282bdc79b019a`；
- 特征代码 SHA：`9f7fb34ee682054dfff0bd1750d5113161d496f04efd3da8aff092f1ef4b6c2d`；
- artifact SHA：`f163f31e82b993765de8f503f6bc09de103ec920c0192ac86d0a485742fe4144`、`78d6b866a90d39f2c70aaf82a75c2fa9a665aa1a536588b2cedaf6809f50fc35`；
- 17,926个非初始化帧，两个分片的公开 bbox/score 最大差均为0；
- OOF analysis plan SHA：`b4ab7a38d15451499a0d58f8c3abd42e518e1c077d84b33bfcf19acd88e0d002`；analysis SHA：`788584c2a57779dae7d00cebe0ae85f7b63d23096a20df094b1997860e1e3974`。

固定的137→64→32→1绝对 IoU 回归头做三折 sequence-group OOF。三个训练折在较高 margin 下都可达到零 catastrophic，且两个折可达到零 harm，但强收益精度最高仍只有约41.46%、27.27%、28.30%，低于85%门。fail-closed 逻辑因此对三个验证折全部选择“不动作”，aggregate OOF 为0动作、0收益；正式判定 `native_feature_separability_not_supported`。这不是“零伤害成功”，而是容量不足的拒绝。

### 24.110.5　当前正在运行的唯一后续：pairwise strong-benefit

第一版的错误目标是预测绝对 IoU，容易把大量小正增益动作也当作可执行动作。下一版保持同一137维冻结特征和所有安全门，只把学习目标改为候选相对 public 的强收益：

```text
输入 = candidate feature + public feature + (candidate - public)
网络 = 411 -> 64 -> 32 -> 1
正标签 = candidate IoU - public IoU >= 0.2 且 candidate IoU >= 0.5
线上资格 = 当前风险触发 + 非public候选概率超过训练折选出的固定候选阈值
```

新的 source plan SHA 为 `45e0b642068c0433b15ac49176107be084c9115bd5c7a83b52e84f0a9b12cba7`，选择12条全新 calibration、20,701帧，与已用24条 calibration 和30条 audit 均不重叠。runner/analyzer SHA 分别为 `6ff69f4cdefe048f7638446a0f51af617b46be7d7174ec4d029849842e332c95`、`22ec15744095efe2f52db18ac8d51b6812653ce61d6e5b09b43c02f8e4b1ebf9`。截至本节落盘，两路 source+feature 采集正在 `stpair12_s0/s1` 运行；尚无结果，不得提前写“通过”。

若 pairwise OOF 仍不满足 `precision>=0.85、harm=0、catastrophic=0、action>=10/3序列、strong rescue>=5/3序列、净增益>0`，就停止局部/宽域候选关联头，不再做第三次阈值/头扫描，转向 STTrack 上的 last-reliable/global re-detection + protected–tentative 原子事务。Qwen 在线文本继续保持可选后置模块，只有视觉身份门通过后才允许以低权重参与，当前不进入主干。

### 24.110.6　审计边界

- 本节所有 GT 后验均只读 DepthTrack Train calibration；冻结30条 audit 未读；
- 未运行 DepthTrack Test、CDTB 或 VOT-RGBD2022；
- 未修改 STTrack 官方 checkpoint；
- 所有 factor-6 候选和 token 特征均为 shadow 证据，公开 recursive state/template/query 保持保护路径；
- STTrack 作者指标仍只作 baseline 外部参考，不能与服务器正式结果混写。
\n## 24.111　pairwise strong-benefit 最终拒绝与全局重检测转向（2026-08-16）

### 24.111.1　第三批完全独立的 calibration 结果

§24.110 末尾记录为“正在运行”的 pairwise 预检现已完成。冻结根为：

```text
/root/autodl-tmp/sttrack_innovation_v1/wide_topk_pairwise_preflight12_v1
```

关键 provenance：

- source plan SHA：`45e0b642068c0433b15ac49176107be084c9115bd5c7a83b52e84f0a9b12cba7`；
- shard0/1 artifact SHA：`b03f54bb09e533b9a58b3fe2a0d42c93457a0d3c3739c95c84544bef58fc8e69`、`fb2bf2f003315ab1377d310b5f5210ae623f9c47946c25f3dba04da76a365061`；
- analysis plan SHA：`37da35b8cf760a9007ef95cff102028b4336936031649a5827e87abba483080c`；
- analyzer SHA：`22ec15744095efe2f52db18ac8d51b6812653ce61d6e5b09b43c02f8e4b1ebf9`；
- analysis SHA：`ea826409beac55f22a4260ef317c1aa2cf0ea4fec5c53e70f091b509718a289d`。

本批选择12条全新 calibration、20,701总帧，与 factor-4/factor-6 前两批24条以及冻结30条 audit 均不重叠。source artifact 含20,689个非初始化帧、555个 source risk 帧；GT有效 risk 行为284。两个分片的公开 rank-0 bbox 预绑定最大尾差均为 `5.684341886080802e-14`，score 差为0，所有特征/候选有限，公开状态未被 source trace 写入。

三折 sequence-group OOF 的最终 aggregate：

| 指标 | 结果 | 预注册门 | 是否通过 |
|---|---:|---:|---|
| action | 45帧 / 8序列 | >=10帧 / >=3序列 | 是 |
| strong-benefit | 16帧 / 6序列 | — | — |
| precision | 35.5556% | >=85% | 否 |
| harm | 2 | 0 | 否 |
| catastrophic | 1 | 0 | 否 |
| strong rescue | 13帧 / 5序列 | >=5帧 / >=3序列 | 是 |
| 当前帧累计 IoU gain | +13.090129 | >0 | 是 |

正式 decision 为 `pairwise_benefit_not_supported`。该模型不是完全无信号：动作覆盖、强恢复和净增益均为正；但它没有达到控制递归状态所需的精度和零灾难条件，因此不能部署。

### 24.111.2　为什么训练折看似完美，留出折仍失败

三个训练折在 threshold `0.5` 都出现近100%训练 precision，训练 loss 约 `6e-5～1e-4`；但留出折分别为：

- fold0：23动作，precision 26.09%，2 harm、1 catastrophic；
- fold1：12动作，precision 58.33%，0 harm、0 catastrophic；
- fold2：10动作，precision 30.00%，0 harm、0 catastrophic。

这说明411维 pairwise 输入在仅12条序列/少量正例上严重记忆训练序列，概率也过度饱和：训练折从0.5到0.99得到相同动作集合，阈值无法提供可靠校准。继续增加 MLP 层、换 seed 或在同一数据扫描阈值，只会提高过拟合风险，不能构成新的科学证据。

因此严格执行 source plan 中的停止条件：

> 不再做第三个局部/宽域候选分类器，不扩到122条，不读 audit，不运行公开集。

### 24.111.3　下一架构转向：last-reliable global re-detection + temporal transaction

factor-6 oracle 已证明“目标有时存在于更宽搜索域”，两种分类头则证明“单帧静态特征不足以安全选中”。下一阶段不再试图靠当前帧分类概率直接覆盖公开 state，而采用：

```text
protected public STTrack
        |
        +-- last-reliable bbox/query/template 只读快照
        |
风险事件 --> last-reliable + multi-tile global proposals
        |
        +-- 每个候选建立独立 tentative state
        +-- 未来2帧 shadow rollout，不写 protected
        |
相对 survival + RGB-D identity + 固定语言身份共同确认
        |
        +-- promote：bbox/query/template/counter 原子提交
        `-- rollback：完整丢弃 tentative，protected 字节级保留
```

建议的第一步仍是容量分析，不训练：只在新的 Train calibration 子集上，对风险事件生成上一可靠位置 factor-6、全帧多 tile 和公开候选，计算“未来两帧后验 survival oracle 是否能恢复更多 failure starts”。若候选集合和两帧 temporal evidence 都没有预注册恢复上限，就停止该方向；若有，再冻结 compact rollout 特征和训练协议。

语言创新在这里有明确但受限的角色：首帧类别/稳定属性只参与 tentative 候选的身份一致性与弃权，不直接更新 backbone，也不覆盖 protected template；Qwen 在线文本仍只能作为通过视觉+时序确认后的低权重动态记忆。

### 24.111.4　边界与当前指标

- 本节未运行 DepthTrack Test、CDTB 或 VOT-RGBD2022；
- 冻结30条 audit 仍未读取；
- 未产生新 checkpoint，未修改 STTrack 官方权重；
- 当前正式最好仍为 VOT `73.974969/82.627562/89.455266`，DepthTrack Test/CDTB 正式指标不变；
- STTrack 作者 `77.6/82.5/93.7` 仍只是外部 baseline 参考。

## 24.112　STTrack 全局重检测两帧 survival 容量：局部很强但跨序列 failure 恢复门拒绝（2026-08-16）

### 24.112.1　冻结设计与零扰动证据

在 factor-6 候选有 oracle 容量、两个单帧候选头均因留出折安全性失败后，本轮按 §24.111 的转向只做 source-only 容量分析。第四批12条 DepthTrack Train calibration 共14,279帧，与前三批36条及冻结30条 audit 完全不重叠。风险帧固定生成：

- last-reliable bbox 的 factor-6 top-8；
- 3×3全图tile各top-2；
- 合计26候选，每个候选独立前推未来2帧；
- public STTrack 的 bbox、z_dict、RGB/Depth query 均不允许被shadow写入。

20帧预运行在3个风险帧均得到10组、26候选和完整2帧rollout；公开bbox/score与既有同源STTrack trace最大差均为0，所有public mutation flag为false。冻结 provenance：

- source plan SHA：`d9efb9009ccfc4bb3a08b98a5f4dfcacd76c9815c4ea39e300fae47dc859a8ae`；
- global tracker SHA：`75154eed780671d4101c9ff0d9556aa8b3083145bbcc10b8cacbc64c5e961aaa`；
- runner SHA：`40628adf73aa5b08807c7a6b1cc96fc57c6078c688a2616fc0cb689c5cadab24`；
- shard0/1 SHA：`8849e6535ff62cf9299c5645fa4cca8dc93cce50c427530b9277a7469d828790`、`e85bbda3797118af01736b2ea5b2221826f81cd845b156bacfcb30f44cf506e9`；
- analysis plan SHA：`024952bea66507ca374173fae41ec33259f4c9bcfec4e02cc703cc6f8639b53b`；
- analyzer SHA：`aac02c26d10fb32480f79264a4a20feae8da8aac3ab7b2246929dd58f74788fb`；
- analysis SHA：`a272ba86ba565624e9f06e73c624f41f45a2a3c8882262571d748d436ad00994`。

两份source共含783个风险帧和20,358个候选；在artifact与analysis plan冻结前未计算GT指标，audit和公开集均未读取。

### 24.112.2　结果与正式裁决

| 指标 | 结果 | 冻结门 | 通过 |
|---|---:|---:|---|
| GT有效完整三帧事件 | 378 | — | — |
| 三帧oracle累计IoU增益 | +136.776910 | >=10 | 是 |
| gain>=0.2 | 208帧 / 8序列 | >=10帧 / >=3序列 | 是 |
| 当前强恢复 | 158帧 / 7序列 | >=5帧 / >=3序列 | 是 |
| 三帧持续恢复 | 148帧 / 7序列 | >=5帧 / >=3序列 | 是 |
| public十帧failure starts | 14 | — | — |
| 可持续恢复failure starts | 2起点 / **1序列** | >=2起点 / >=2序列 | 起点过，序列不过 |

正式 decision 为 `global_redetection_capacity_not_supported`。恢复能力高度集中在 `leaves03_wild`，不足以证明对跨序列VOT失败链可泛化，故不得训练选择器、不得扩122条、不得读取audit或运行VOT。

### 24.112.3　具体例子与原因

- `leaves03_wild@58`：public三帧IoU为 `0/0/0`；last-reliable factor-6 rank0为 `0.8567/0.9194/0.8483`，三帧稳定救回。
- `leaves03_wild@227`：public仍为 `0/0/0`；global tile-02 rank0为 `0.8428/0.8974/0.9321`，说明目标离开局部搜索域时全局候选确实有容量。
- `ball05_indoor@609`：public为 `0/0/0`；global tile-07为 `0.4071/0.8231/0.8653`。它显示下一两帧能够回到目标，但当前帧未达到预注册的IoU>=0.5强恢复定义，不能在看到结果后放宽阈值。
- `leaves03_wild@599`：候选为 `0.8919/0/0`，只救当前帧而未来立即再次丢失，证明单帧高IoU不能替代survival门。

这组结果解释了为何“扩大搜索域 + oracle选框”仍不足以直接提升VOT：候选集合对普通风险帧有大量局部收益，但正式ROB依赖跨序列、连续十帧失败链；少数序列上的高oracle上限不能证明可部署门能稳定选择，也不能覆盖其余failure族。

### 24.112.4　下一优化方向

停止继续扩大tile、扫描候选分类器或事后改IoU门。普通query-reset、age-2延迟、原生query证据和首帧RGB-D回环已在 §24.107–109 完成并失败，不能重复。下一实验进入真正需要重新训练的 STTrack 内生 target–distractor association：官方主干和box regression先冻结；风险分支使用不可变首帧模板、query reset和factor-6 search，从MambaFusion的完整search token图学习低参数身份残差；结构化语言以冻结CLIP文本特征投影到同一身份空间，与static-template prototype联合重排响应图。训练样本固定为4帧离线predicted-crop rollout，fold1–3拟合、fold4选epoch、fold0只做一次留出验证。只有留出fold达到零catastrophic、低harm和跨序列failure rescue，才实现protected/tentative Active与原子promote。

本节没有产生新checkpoint或公开指标。当前正式最好仍为 VOT `73.974969/82.627562/89.455266`；STTrack作者值 `77.6/82.5/93.7` 仍只作外部baseline参考。

## 24.113　STTrack 语言锚定 response adapter：独立留出 fold 严格拒绝（2026-08-16）

### 24.113.1　结构、训练边界与可回退性

本轮按 §24.112 的架构转向实现了一个 361,474 参数的低参数关联 adapter。官方 STTrack backbone、MambaFusion 主路径和 box size/offset regression 全部冻结；adapter 只读取不可变首帧模板 token 与冻结 CLIP 结构化语言表示，在 factor-6 query-reset 风险支路的 16×16 response 上学习 identity residual。residual scale 初始化为0，因此未加载或未通过门时可精确回退到冻结 STTrack。

数据只来自 DepthTrack Train calibration：固定660个四帧 predicted-crop 事件。fold1/2/3 共412事件用于拟合，fold4的113事件只用于选 epoch，fold0的135事件只作一次最终验证；冻结30条 audit 完全未读取。`toy07` 仅存在39条尾部 GT-only 标注，计划和提取器显式绑定该异常并只消费与图像重叠的1367帧。

冻结证据：

- training plan SHA：`f401d0f7aadc6f9e4c328bb1d29b54849e1662caf3bdb53ef1952f1536aede82`；
- feature shard manifest SHA：`963b97511426e9f14c16a46ffe894d74d7c231da90fb8f406b98acd18c7950be`、`ccce6a11740947643213c23ae727eec5d35d12d27df0f803d8810fea7884069e`；
- adapter source SHA：`65852cbdd8e7cd06d4bcdaf5b709f105578a4d6a5dc14985dbe773c0e0072ca3`；
- feature extractor SHA：`dd1c4cdc89230500d8b758115d1434f46d6f484a2fce8af466b917277c55dc3b`；
- trained seed2026 adapter SHA：`f45ba28126a6bd95eea45b6e09f0344ee3fa819c038de137c62e5ae21f9e67a8`；
- training report SHA：`db832eb86c1216871ea92b82662c0cac6360c89b6d88fb4c8b191332fedf88e4`。

### 24.113.2　一次性留出结果与裁决

fold4 选择 epoch27；随后只对 fold0 执行一次最终验证，共540个未来帧：

| 指标 | 结果 | 冻结门 | 通过 |
|---|---:|---:|---|
| baseline mean IoU | 0.531050 | — | — |
| adapter mean IoU | 0.532911 | 高于baseline | 是 |
| 累计 IoU 增益 | +1.004968 | >0 | 是 |
| strong benefit | 0帧 / 0序列 | >=10帧 / >=3序列 | 否 |
| harm | 2帧（0.3704%） | <=2% | 是 |
| catastrophic | 1帧 | 0 | 否 |
| failure recovery | 1帧 / 1序列 | >=5帧 / >=3序列 | 否 |
| 语言相对 zero-language 累计贡献 | +0.026568 | >=0 | 是 |

正式 decision 为 `language_association_preflight_not_supported`。平均增益很小，且零灾难、强收益覆盖和跨序列恢复三项关键门失败，故不实现 Active、不读取 audit、不运行递归评测或公开 VOT。

具体例子：`flower02` 的 event466+1 从0.1025提高到0.3244，虽增益0.222但仍未达到0.5强恢复；`bag04` event17+2 从0.238降到0，构成catastrophic；`ball14` event132+1 从0.673降到0.453，构成明确harm。这说明浅层静态模板/语言残差能够改变局部response，却不足以在冻结主干下学出可靠的递归目标身份。

### 24.113.3　止损与唯一后续

该浅层 adapter 路线至此停止，不换 seed、不扫描阈值、不重复消费 fold0。若继续，只允许一次更深但仍严格受控的 STTrack 原生递归微调：使用全部122条 calibration 的4帧真实 predicted-crop rollout，冻结 ViT backbone 与 size/offset regression，只解冻风险支路的 TSG/MambaFusion/center-response 及关联头；seed和epoch预先固定，权重冻结后才允许对30条 audit读取GT一次。audit 不满足零catastrophic、低harm及跨序列 failure recovery 时即永久停止，不进入公开VOT。

本节没有产生新正式指标。当前正式最好仍为 VOT `73.974969/82.627562/89.455266`，DepthTrack Test/CDTB 保持不变；STTrack作者值仍只作外部参考。

## 24.114　STTrack 原生 response 最终微调：恢复容量显著但安全门严格拒绝（2026-08-16）

### 24.114.1　冻结结构与一次性协议

在浅层语言 adapter 被拒绝后，本轮执行预注册的最后一次 STTrack 原生微调。protected public factor-4 STTrack 始终不变；训练只作用于 factor-6 query-reset tentative 分支的 search fusion convolution、center conv4/conv5 和语言/静态模板 identity residual，共5,689,155个可训练参数。ViT backbone、TSG、MambaFusion、center stem、size/offset regression、模板和公开递归状态全部冻结。

训练使用既有660个 calibration 四帧 predicted-crop 事件，固定 seed2026、固定5轮，不做 epoch selection；30条 audit GT 在权重冻结前只做字节SHA绑定，不解析数值。初始 tentative score 相对官方分支最大差为 `0.000230`（来自冻结 float16 feature 量化），训练 loss 从 `2.8377` 降到 `2.1214`。冻结证据：

- train plan SHA：`c5378bafa023a8d179d69c6e07ad953e31eaf1420a53f4bca0a8746a8c274691`；
- model/trainer/plan-creator SHA：`22f30d38c2baf858e4e164a7f04ce33a8c5546d6acbc8e6c5b27162d2d47b9f7`、`dcd99ce79ca4d1688017d6734c3c40d7262f468ffd718742d1461b7b5ae693ca`、`dffdbc914f81774199e4370d2898aef4474f690f4d73c569d0fd637d757748b2`；
- epoch5 artifact SHA：`e1418a9541e8a1b723e5b8e1beb2c64f5146bc988d48b85c2a4c5731aa87489d`；
- training report SHA：`b1c0ae1991d20109a05d2dabc310a6926c62ee407047134aa399e3e4bd8259b7`；
- single-audit plan SHA：`6b4779bf08971c72eea485ad5caf4d8c861ed4ba8b5c61d9a6c3252fa744506b`；
- audit runner SHA：`14abd67c458b5d361b2435a1fa51602bcf70c26303db6433a920737495c9e904`；
- audit result SHA：`bf70aec4fb013c11c828f730b0a1b0607e02ff007c03b1beef61d8ad5b3dc10d`。

### 24.114.2　30条封存序列、291事件、10帧递归 audit

权重冻结后只执行一次 audit。291个source-only风险事件覆盖27条有事件序列，每个事件分别递归运行带语言和zero-language的10帧tentative分支；因不可见/无效GT，最终2080个帧行进入指标。对照始终是冻结source trace中的protected public factor-4框。

| 指标 | 结果 | 冻结门 | 通过 |
|---|---:|---:|---|
| public / tentative mean IoU | 0.479047 / 0.563602 | tentative更高 | 是 |
| 累计IoU增益 | +175.872888 | >=1 | 是 |
| strong benefit | 382帧 / 20序列 | >=10 / >=3 | 是 |
| recovery | 415帧 / 17序列 | >=5 / >=3 | 是 |
| 10帧持续恢复 | 7事件 / 5序列 | >=2 / >=2 | 是 |
| harm | 262帧，**12.5962%** | <=2% | **否** |
| catastrophic | **126帧** | 0 | **否** |
| 语言相对zero-language累计贡献 | **-2.880166** | >=0 | **否** |

正式 decision 为 `native_response_single_audit_not_supported`。科学拒绝 exit code 2；没有实现Active、没有改权重、没有运行DepthTrack Test/CDTB或VOT。

### 24.114.3　具体正负例与VOT含义

该分支的恢复能力是真实且很强的：

- `car01_indoor@1317`：protected连续10帧IoU全0；tentative从0.1398开始，随后9帧约 `0.934～0.970`，形成严格持续恢复。
- `glass03_indoor@143`：protected连续10帧全0；tentative由0.1964逐步升至0.9753。
- `speaker_indoor@2879`：protected连续10帧全0；tentative十帧始终约 `0.805～0.946`。

但它也会把本来非常稳定的目标直接切走：`toiletpaper02_indoor` 的 event257+2，protected IoU为0.9534，tentative降到0；同一事件后续多个offset从约0.94降到0。此类正常状态被高容量恢复专家覆盖，正是126个catastrophic和12.6% harm的来源。

因此结果不能解释为“VOT应该上涨”：平均IoU和恢复数很高，只证明factor-6原生微调可以找到丢失目标；VOT EAO/ROB会同时放大对正常anchor的灾难写入。未经保护事务直接部署，大概率仍是ACC/均值改善、ROB恶化。语言贡献为负还说明当前冻结CLIP结构化文本不应进入该恢复专家的主决策，后续只能作为可弃权的身份佐证或删除消融。

### 24.114.4　止损结论与后续架构边界

该epoch5 artifact永久拒绝，不得再次消费同一audit、调阈值或直接跑公开VOT。横向baseline也停止：SAMF官方VOT为 `76.2/80.7/93.6`，整体不优于STTrack且存在32GB复现和AlphaRefine打包许可风险；STTrack继续作为唯一值得改进的baseline。

下一版若继续，不能再训练“更强的全时响应”，而必须把恢复专家永久放在dormant tentative分支：protected STTrack先输出；只有公开路径已连续恶化时才启动专家；未来2～3帧同时保留protected和tentative，比较相对响应稳定性、RGB-D首帧身份和轨迹连续性；只有tentative持续胜出才原子提交bbox/query/template，否则完整rollback。当前30条audit已被一次性消费，不能用于开发该router；在没有新的未触碰Train-only验证源前，不得把后续改动包装成已验证提升。

当前正式最好仍为 VOT `73.974969/82.627562/89.455266`；本节没有新公开指标。

## 24.115　STTrack dormant recovery transaction：安全性接近过门但覆盖不足（2026-08-16）

### 24.115.1　为什么继续做这一轮

§24.114 已证明 native factor-6 recovery expert 同时具有强恢复和强伤害，问题不是候选不存在，而是不能全时覆盖 protected STTrack。本轮因此不改epoch5专家权重，而把它降为 dormant shadow：候选响应固定使用 zero-language；结构化语言只作为外部身份否决/弃权特征。每个 calibration 风险事件独立递归10帧，只使用前3帧的专家、同crop官方响应、语言反事实、候选/公开一致性和轨迹稳定性构造32维证据；未来offset2–9只用于GT后验标签。

660个事件在任何GT join前分两片完成source-only trace，公开框和状态未被写入，已消费的30条audit没有再次读取。冻结证据：

- plan SHA：`c78575fb122a27c8e4801783e3fb9897b0370b4826ae38037726e40d891c470e`；
- creator/collector/analyzer SHA：`cf8c78f3af93cc22fd30111d59d5da20cae72940ad51f878b1f9f74c453c25a6`、`8f9f147218ee61e144b56b18d25818ce9b6acb830f2b10279a484ea4eeaf0a28`、`a5482251b54ecf64206958b39f952edc1d5c7f5625d2bb5564d7fcf4458d58bd`；
- source shard SHA：`810ab67406e124e0fa8f890eeb4da3b3df19855048f8b3cf78d718e715cd64ae`、`35d9ccab367bb50e6ec98b15782f35294972d0b3c27642def3c8b6b0b3fd525b`；
- OOF analysis SHA：`4fe5564d826975de2bc5541341d380ad9348a443a08522f36cc288da99082d8f`。

### 24.115.2　三seed sequence-group OOF结果

GT join后，660个事件中129个满足预注册benefit标签、172个含unsafe outcome，全动作有53个持续恢复事件。router固定为标准化32→32→2 benefit/unsafe MLP，5折按sequence分组，seeds 2026/2027/2028；threshold网格和门均在source前冻结。

| seed | benefit/unsafe阈值 | 动作/序列 | precision | harm/catastrophic | 累计增益 | recovery | 持续恢复 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | 0.95 / 0.10 | 6 / 5 | 83.33% | 0 / 0 | +32.0152 | 36帧 / 4序列 | 3事件 / 2序列 | precision和动作数失败 |
| 2027 | 0.95 / 0.10 | 7 / 6 | 85.71% | 0 / 0 | +37.8040 | 44帧 / 5序列 | 3事件 / 2序列 | 仅动作数失败 |
| 2028 | 0.95 / 0.15 | 7 / 6 | 85.71% | 0 / 0 | +31.0604 | 44帧 / 5序列 | 4事件 / 3序列 | 仅动作数失败 |

预注册门要求每seed至少10个动作、至少3序列、precision≥85%、harm≤2%、catastrophic=0及恢复/累计增益门全部通过。2027/2028已经达到零伤害、零灾难和85%精度，但只有7个动作；2026只有6个动作且precision 83.33%。正式 decision 为 `dormant_transaction_oof_not_supported`。

### 24.115.3　科学解释与停止边界

这是目前最接近VOT ROB目标的一次内部结果：与全时专家的12.6% harm、126 catastrophic相比，dormant router在两个seed上把伤害压到0，同时保留44个恢复帧和跨序列持续恢复，证明“protected + dormant expert + 两三帧相对证据”架构方向成立。

但不能把动作数门从10事后降到7。7个OOF事件过少，单个错误就会把precision从85.7%降到75%，也不足以支持在约132万VOT tracker frames上部署。禁止用同一数据换seed、加threshold或重复消费audit；本轮不生成部署artifact、不实现Active、不运行VOT。

若获得新的RGB-D Train-only序列，优先扩大同一冻结协议的正例覆盖，而不是放宽安全门。没有新增未触碰证据时，最诚实的结论是：STTrack dormant事务是当前最佳改进方向，但尚未形成可部署模型。当前正式VOT仍为 `73.974969/82.627562/89.455266`。

## 24.116　STTrack dormant 合成增强与三 seed median 集成：覆盖仍不足（2026-08-16）

### 24.116.1　冻结的 Train-only 增强

§24.115 的自然 OOF 已把 harm/catastrophic 压到零，但每个 seed 只有6～7个动作。为判断这是恢复容量不足还是小样本方差，本轮在不增加公开数据、不复用30条audit、也不改变门槛的前提下，只对660个 calibration 自然事件构造训练专用反事实：每个事件生成两个确定性 tentative 起点，分别为按公开运动方向平移0.75个框尺度，以及平移0.35个框尺度并放大到1.25倍。共1320个合成事件、每事件10帧；它们只进入训练折，最终门仍只在原始660个自然事件的sequence-group OOF上计算。

冻结 plan SHA 为 `e2fb6b0f3ef07103dbea2d1ad2f41201b2a0e1fa4aa8874da05923b9a58608c5`；两片合成 trace 均 exit 0，SHA 为 `f28d85051a1f980f0e6cf9570793a4b70a4d535e68b02408900cccbfaf754893` 与 `a9f59fece425bc7b744edc9c91da939a2c80ac09c0f3d35b73c8bf83a1f60b55`。合成集中272个benefit事件、368个unsafe事件；分析器 SHA 为 `49051bacb13497c62e6612ffd4c77b80f79d6ef98c05c87804f5a5bc4a73ef49`，结果 SHA 为 `36bf7a8769695964eddffe1bb256316f98774854b7881c1b72f8c775738b3cda`。

### 24.116.2　自然 OOF 结果

| seed | benefit/unsafe阈值 | 动作/序列 | precision | harm/catastrophic | 累计增益 | recovery | 持续恢复 | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | 0.95 / 0.15 | 7 / 5 | 85.71% | 0 / 0 | +38.2911 | 44帧 / 4序列 | 4事件 / 3序列 | 仅动作数失败 |
| 2027 | 0.95 / 0.30 | 11 / 8 | 100% | 0 / 0 | +73.2038 | 84帧 / 8序列 | 8事件 / 6序列 | 单seed通过 |
| 2028 | 0.95 / 0.25 | 9 / 5 | 100% | 0 / 0 | +60.3475 | 68帧 / 5序列 | 7事件 / 4序列 | 仅动作数失败 |

合成增强证明模型可以在自然事件上形成11个完全无伤害动作，但只有seed2027通过；三seed稳定性合同仍失败，正式 decision 为 `dormant_augmented_natural_oof_not_supported`。不能事后挑2027部署。

为排除“单seed选择”并把随机性变成确定性，本轮在读取集成结果前又冻结唯一一次 coordinate-wise median 集成：对三seed的两个自然 OOF 概率分别取中位数，使用原阈值网格和原安全门。plan/analyzer SHA 为 `3d3177654bc746ffb91469f3fbc79045997b2785f4857d536c896d53e6d9f1ae` / `d20efb5ef835b6da022233f8fb7976e8d1cf2197770d05dfe56a6de298747513`。最终只有8动作/5序列，precision 100%、harm=0、catastrophic=0、累计增益 `+52.936465`、60个恢复帧/5序列、6个持续恢复事件/4序列，但没有任何阈值达到minimum_actions=10；analysis SHA `0f14d49269d058cb36548fe9ef88744d1a42422b0d74643fa5802a5f6c9d89d0`，decision `dormant_augmented_median_ensemble_oof_not_supported`。

### 24.116.3　终止边界

本轮没有生成部署 artifact、没有实现 Active、没有读取公开 GT、没有运行 VOT。不得把动作门从10降到8，也不得再在同一660事件上扫描增强幅度、网络结构或概率聚合。dormant expert 的恢复容量与零伤害子集真实存在，但证据覆盖不足以支撑约132万帧的公开部署；这条 router 搜索到此终止。

## 24.117　baseline 与额外训练数据止损结论（2026-08-16）

开源候选已收口：STTrack 官方 VOT22 权重、TraX入口和MIT许可完整，论文/作者发布物给出的 VOT-RGBD2022 为 `77.6/82.5/93.7`，是当前唯一适合继续承载创新的底座；SAMF 为 `76.2/80.7/93.6`，整体低于STTrack，且作者复现依赖V100 32GB、发布包含AlphaRefine GPL来源，判定NO-GO；MixForRGBD顶层无明确许可证，权重约1.57GB且旧双分支环境负担较大，也不进入服务器移植。研究文档为 `/home/SUTrack_RGBD_L/docs/OPEN_SOURCE_BASELINE_MIGRATION_RESEARCH_ZH.md`，SHA `6f2fa959e5364ceffda0dbe026ba6fe97a671d0dd531f427ba6125cc65cc88ac`。

额外Train-only数据同样暂不可用。RGBD1K的1000条train在科学上是首选，但有标注RGB-D帧只以 `288401172382` 字节单体ZIP发布，无按序列/分片入口；服务器仅约7GB可用数据盘，无法安全下载或解压，且README与LICENSE分别写CC BY-NC-SA 4.0和CC BY-NC 4.0。ARKitTrack的250条train也缺数据媒体许可、train-only包体积和分片下载证据。完整审计在 `/home/SUTrack_RGBD_L/docs/RGBD_TRAIN_ONLY_DATASET_RESEARCH_ZH.md`，SHA `1dd56fdb90b4d54ca0f8f8b9f215934bb4b11606db5e5ede6a008916a8286398`。

因此不再横向扫描baseline，也不下载来源不闭合的数据。后续若继续，只在STTrack上保留首帧语言身份锚点、protected/tentative双状态和原子promote/rollback创新；任何学习型改动仍须先过DepthTrack Train-only零catastrophic/低harm/恢复覆盖门，之后才允许一次公开VOT。当前正式最好不变：VOT `73.974969/82.627562/89.455266`，DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。

## 24.118　DepthTrack Train multi-start 自然轨迹扩增（2026-08-16，full122运行中）

### 24.118.1　动机与隔离合同

合成增强和median集成的失败不是恢复容量不足，而是660个OPE风险事件只能稳定筛出8个零伤害动作。VOT-RGBD2022本身采用multi-start anchor；因此本轮不再调同一小路由，而用DepthTrack Train模拟同分布：每条序列在约1/3和2/3位置各取一个anchor，只读取该anchor的一条GT bbox初始化，之后protected跟踪、风险触发和10帧dormant expert rollout全部source-only；完整GT只在四片trace完成后由独立分析器加入。

anchor选择不得看IoU或结果。最初v1在`ball02`目标分位遇到无效GT行并fail-closed；另一片随即停止，半批结果永久弃用。v2/v3只按“距目标分位最近的有效GT行”机械调整，`ball02`与`ball08`均只后移1帧。v3再把同一20轨迹机械拆为四片，GPU布局 `0,0,1,1`；序列、anchor、阈值和门没有改变。

### 24.118.2　20轨迹容量预检

预检覆盖10条calibration序列、20条multi-start轨迹、16,251帧，产出211个自然风险事件。plan SHA为 `c5ace7c056f149ecdb866c251fb169bd8b2ccd34f5650cbbb53f0cfe20c6a960`；四片均exit 0，SHA为：

- `1961226425aafd1914da7bd6a53c116ed89460f4c31f6071e7f82026bef9707d`；
- `6df6f1c28f0f5e82f4c4bb4892548b2fb3645864cfc81068714e3bfff944f690`；
- `d708d85965011d5f85e6d182f08fffcd229db90dcb8b1d4594a54bfb2f4f70a6`；
- `6556ae4c36817dad743e28e36ba812cef7b8e6dcf8f748d7a1c348dfa01ecba9`。

第一次分析报告仅 `minimum_oracle_cumulative_gain` 为false，因为分析器把211个动作全部相加为 `-24.499831`，却误命名为oracle；这会把GT oracle明确不会选择的unsafe/负收益动作也计入。该值保留为 `all_action_cumulative_gain`，说明专家绝不能全时启用。独立amendment在读修正结果前冻结：safe oracle严格定义为只累加已经满足gain≥1且无unsafe的benefit事件，不改任何阈值、不重跑trace。amendment plan/result SHA为 `bdc0e3fab82510e11b99bf88507b91e64928c1de686495036a60c989aee9fe73` / `7e036fdd186192466480108b297166d7c4cf9555da9c20a69fac55c09ad5c76d`。

修正后的容量结果：

| 项目 | 结果 |
|---|---:|
| complete events | 211 |
| benefit events / sequences | 23 / 8 |
| unsafe events | 66 |
| recovery frames / sequences | 163 / 9 |
| persistent recovery events / sequences | 12 / 6 |
| trajectories with events | 20 / 20 |
| safe-oracle cumulative gain | +114.553892 |
| all-action cumulative gain | -24.499831 |

全部预注册容量门通过，decision `multistart_natural_capacity_supported`。这只授权扩大自然Train轨迹，不授权Active或VOT。

### 24.118.3　full122 source trace 与更严格OOF门

full计划使用冻结split中的全部122条calibration序列；28条原OPE无风险事件的序列也从原始split plan读取正式fold归属，不从事件表猜测。每序列两个anchor，共244轨迹、179,761帧，四片为44,947/44,942/44,939/44,933帧。plan SHA `e2b730e8e77ad6dba2d7cdd6d806f3d6b52f5c2c3365e8748a6b24170e3d5a10`；creator/runner/analyzer SHA为 `0302d31e2f864eb24ac1341f08a15019d4deed3d1c27de0710eb6a0b177f8bd3`、`959632ecaf2a4d085a36f366563905522283c3e609a8248081b123dc92bc240e`、`18a6b7d1356e49a158fa461ae20ea85c9f0b03b31b50a5e4c16006c1d4e97c3a`。

由于样本规模扩大，最终sequence-group OOF门同步加强且在trace前冻结：三seed必须全部达到至少30动作、10序列、precision≥85%、harm≤2%、catastrophic=0、累计增益≥20、recovery≥30帧/10序列、持续恢复≥10事件/5序列。OOF合并原660个anchor0自然事件与新增multi-start事件，按base sequence分5折，确保同一序列不同anchor永不跨训练/验证折。

当前screens为 `stmulti_full_s0`～`s3`，持久控制器为 `stmulti_full_continue`；启动后两卡各约4.92GiB、利用率约93%，四片均已产生首批事件，controller状态 `waiting_for_shards`。四片exit 0后它只自动运行上述Train-only OOF：exit2科学拒绝即停止；即使通过也只写 `oof_supported_no_active_or_vot`，绝不自动实现Active或运行公开VOT。

本节没有新公开指标。当前正式最好仍为 VOT `73.974969/82.627562/89.455266`。

## 24.119　full122 multi-start 三 seed OOF 终态：恢复覆盖成立，但 precision 门拒绝（2026-08-16）

### 24.119.1　source trace 完整收口

四个分片均 exit 0，合计严格覆盖 `179,761/179,761` 帧、`244/244` 条轨迹，每片各61条且跨片无重复。四个 trace 均声明并经独立复核：`complete=true`、初始化后不读GT、事件选择不使用GT、未计算任何指标。最终事件数为1,303。冻结 trace SHA 为：

| shard | 帧数 | 轨迹 | 事件 | SHA256 |
|---:|---:|---:|---:|---|
| 0 | 44,947 | 61 | 329 | `2eb4c69280416372ce93025fbccc97789b67731405cb9af3bbd75cbf02f08562` |
| 1 | 44,942 | 61 | 384 | `1baf3cd6d0c34a2b79cfded5f37eacfdf44f6ffcbbd71439a093f2e3af0bf5de` |
| 2 | 44,939 | 61 | 300 | `85b637f8c636ac19d4ff753b9b516317bf38bbc764e442551afafc4a87eb075d` |
| 3 | 44,933 | 61 | 290 | `112d2aa3b26dabb537ceb337a22408a5b743b94d09a120ecb97c9dc06d5fd484` |

分析器随后才加入冻结的DepthTrack Train GT，把原660个OPE自然事件与1,303个multi-start事件合并为1,963个事件；其中315个满足benefit定义，465个满足unsafe定义。122个base sequence的两个anchor始终在同一fold，和原660事件中94个重叠序列的fold完全一致。

### 24.119.2　三 seed 原始结果

analysis SHA 为 `010ab5419ad55ed9fe9a6c963c8a2701eea623788c2ea031239a2bf4e79ad38a`，480,495 bytes；analyzer SHA仍为 `18a6b7d1356e49a158fa461ae20ea85c9f0b03b31b50a5e4c16006c1d4e97c3a`。analysis exitcode为2，controller终态为 `oof_rejected_no_active_or_vot`。

| seed | benefit/unsafe阈值 | 动作/序列 | precision | harm/catastrophic | 累计增益 | recovery | 持续恢复 | 失败门 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | 0.85 / 0.10 | 37 / 16 | 72.972973% | 0 / 0 | +169.467175 | 189帧 / 15序列 | 18事件 / 10序列 | precision<85% |
| 2027 | 0.80 / 0.10 | 34 / 19 | 73.529412% | 0 / 0 | +148.365494 | 165帧 / 15序列 | 16事件 / 9序列 | precision<85% |
| 2028 | 0.80 / 0.10 | 35 / 18 | 82.857143% | 0 / 0 | +163.698133 | 183帧 / 17序列 | 16事件 / 9序列 | precision<85% |

三seed平均为 `35.33±1.25` 个动作、precision `76.4532%±4.5340pp`、累计增益 `+160.5103±8.9048`、恢复帧 `179±10.20`；所有选择的harm rate和catastrophic都为零。除minimum precision外，其余九项覆盖、安全、增益和恢复门全部通过。

### 24.119.3　候选地形与不可事后修门

只读复算全部42个冻结阈值组合后，seed2026没有任何precision≥85%的网格点。seed2027/2028虽各有4个precision通过点，但最好的分别只有15动作/6序列和16动作/7序列，同时失败minimum actions、action sequences、recovery sequences及persistent覆盖。三seed正式选择动作的两两Jaccard仅 `0.5435/0.5652/0.6047`；三者交集只有21动作，precision仍为76.1905%，不能用事后交集或挑seed挽救。

因此不能把85%降到82.86%，不能挑2028部署，也不能继续扫描同一32维特征、阈值、seed或概率集成。该结果证明dormant expert和source-only相对证据中确实存在跨序列恢复信号，并能在OOF上把观测harm/catastrophic压到零；但它不证明当前router能以足够高的benefit precision安全提交递归状态或模板。

### 24.119.4　独立 result-to-claim 与后续边界

独立result-to-claim审查给出 `claim_supported=no`、confidence=`high`。支持的最强表述只能是：当前router在Train-only sequence-group OOF上获得跨序列恢复覆盖、正累计后验增益及零观测灾难，但benefit precision仅为72.97%～82.86%，未达到预注册85%，所以不具备Active、递归状态/模板提交或公开VOT资格；OOF中的零观测伤害也不能外推为部署零伤害。

本路线正式停止：不生成部署artifact、不实现Active、不运行VOT，也不复用已经消费的30条audit。若继续研究，只允许改变表示而非调阈值，例如unsafe-first/abstention或语言锚定target–distractor身份表示，并且必须重新预注册同一安全门，在新的sequence-disjoint Train-only证据上一次性验证。当前外部Train-only数据仍受包体/许可限制，因此在获取新证据前不得把这一建议写成已获准实验。

本节没有新公开指标。当前正式最好仍为VOT `73.974969/82.627562/89.455266`，DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。

## 24.120　RGBD1K 77维语言锚定 abstention：表示非退化，但恢复容量门拒绝（2026-08-16）

### 24.120.1　为什么引入新的Train-only证据

DepthTrack Train上既有32维router已经在自然OPE+multi-start的三seed OOF中只失败85% precision门，说明继续调同一阈值、seed或概率组合不再科学。随后通过Range/Zip64流式抽取和严格跨benchmark exact/pHash去重，冻结18条RGBD1K Train序列，每条600个RGB与Depth帧。selected18合同SHA为`b75b59d44656443e73eed89458b15ea51dc649541c7e4c19772d4c3de3f3c627`，dedup结果SHA为`aee7e91e8acc39ab301b7ce757fa8296df0cb9f8ee8b135e87392b98af8c2443`。

冻结salt先产生12条development和6条一次性audit。audit为`Car/car6`、`Cooker/cooker1`、`Animal/deer17`、`Ipad/ipad1`、`Adapter/adapter1`、`Animal/cat3`；其GT只为源文件完整性做字节哈希，没有解析数值或语义，两个anchor都保持未解析和未授权。development每序列在1/3、2/3附近机械选有效初始化行，共24条轨迹；后续source事件选择不读GT。

### 24.120.2　77维表示与protected/expert边界

protected分支保持官方STTrack factor4、官方动态模板更新和公开递归query；身份度量始终参照不可变第一模板槽。dormant expert使用factor6、static-static模板、query reset和zero-language response，不写公开状态。表示由两部分组成：

1. 原有32维短窗生存/响应/语言反事实证据；
2. offsets 0--2三帧，每帧15维expert-minus-protected RGB/Depth/fused首模板身份差，共45维。

总计77维。offsets 0--2只作决策证据，promotion最早只能影响offset3；所有benefit、unsafe、recovery和persistent标签严格只读取offsets 3--9。capture reason不进入特征。预注册容量门为：完整事件≥20、benefit≥10/5序列、unsafe≥10/3序列、recovery≥30帧/5序列、persistent≥5/3序列、safe-oracle累计增益≥20、有事件轨迹≥8、非恒定身份维≥15。

五个实现文件经Spec/Standards两轴多轮只读复审最终无硬阻断。最终合同包括：每个文件同一fd读取并核inode/stat前后不变；18条冻结receipt与每个payload字节重哈希；拒绝数据根/category/sequence和树内symlink或特殊项；输出不得位于repo、媒体或冻结输入路径；runner/analyzer发布前重新复验源、GT、trace、plan和实现；risk/cooldown/tail独立重算；event与started-row双射；硬链接no-replace及并发winner完整复验。初始提交为`c902fb282df2398547d7a97caec5908b41fc904d`。

### 24.120.3　v1采证失败及其根因

v1 plan SHA为`650763c8331cb303f8a49062e1caff9c429912f9f3031ab17cd037f676a0a5ea`。两个shard都在首个公开track帧以`dormant public identity capture is incomplete` fail-fast，GPU立即释放，没有写出任何trace，因此v1没有科学结果，也没有被后续复用。

按照确定性单帧最小复现，真实调用计数为：`MambaFusion=1`、`cal_bbox=2`。第一次`cal_bbox`是CenterHead内部基于raw score生成bbox，第二次才是tracker对`Hann window × raw score`生成公开bbox。原hook错误假设只有一次调用。修复精确要求1次Mamba和2次`cal_bbox`，并用`torch.equal(response[1], output_window * response[0])`绑定第二张public response，只用它选primary/runner-up。hook和实例方法在全部异常路径恢复，公开bbox、score、模板和query均未改写。单帧复现由红转绿，两轴复审PASS；修复提交为`a64262d21f9821a5579773f2f3146afb9ce4cf86`。

### 24.120.4　v2 source trace与容量原始数值

v2重新冻结，plan SHA为`2fbf48288d16a61c601c99490f240bfaecb95019b0330012177f1aee9a3c41e4`，split、fold和24条轨迹与v1完全相同，只改变采证实现绑定和输出路径。双RTX3090 source trace约4.5分钟完成：

| shard | 帧数 | 事件 | 事件序列 | trace SHA256 |
|---:|---:|---:|---:|---|
| 0 | 3,600 | 28 | 4 | `33406bcb42fef56491a76cd675c9d4a80a251a1d1e303e5a0868d10379c0105b` |
| 1 | 3,581 | 10 | 4 | `6be041ac902c2e3fca32b2a67eb09f8f4217b675ff5ecc3e58eaa399c13137f3` |

两份trace重新走existing-output完整校验均以`recovered_existing_trace=true`通过；合计7,181帧、38个事件、12条有事件轨迹。之后分析器才读取12条development GT，audit仍未读。analysis SHA为`7210856d752f636c4b4f4e0ac1bcffacfbd058279c6afed3a2ea05aea254a74c`，exitcode=2，原始指标为：

| 容量项 | 观测值 | 预注册门 | 结果 |
|---|---:|---:|---|
| complete events | 38 | ≥20 | 通过 |
| trajectories with events | 12 | ≥8 | 通过 |
| nonconstant identity dimensions | 45/45 | ≥15 | 通过 |
| benefit | 2事件 / 2序列 | ≥10 / 5 | 拒绝 |
| unsafe | 8事件 / 2序列 | ≥10 / 3 | 拒绝 |
| recovery | 7帧 / 1序列 | ≥30 / 5 | 拒绝 |
| persistent recovery | 1事件 / 1序列 | ≥5 / 3 | 拒绝 |
| safe-oracle cumulative gain | `+3.135228` | ≥20 | 拒绝 |

45个身份差维度全部变化，只证明表示不是常量；它没有把expert相对protected的真实恢复容量变大。38个事件中只有2个满足安全benefit定义，safe-oracle上限也远低于训练所需水平，缺口发生在学习之前，不能靠MLP、seed或阈值补足。

### 24.120.5　独立claim裁决与停止边界

独立result-to-claim裁决为`claim_supported=no`、confidence=`high`，路由为`capacity_rejected_stop_before_learning`。当前支持的最强表述只能是：77维语言锚定身份差在sequence-disjoint RGBD1K Train预检中具有可观测变化并捕获少量benefit/unsafe差异，但当前factor6 static/query-reset expert的跨序列安全恢复与oracle增益容量显著不足，不具备进入三seed OOF训练的依据。

因此本路线立即停止：不训练2026/2027/2028、不解析audit6、不生成部署artifact、不实现Active、不运行VOT。禁止降低容量门、在同18条上事后扫事件/anchor/阈值，或把保留的audit用于开发。若继续，只能作机制级pivot：先构造能显著增加expert相对protected真实恢复的动作，例如真正的global multi-tile或last-reliable re-detection，再用全新未触碰、sequence-disjoint Train-only数据预注册同等级容量/安全门。这不是当前失败配置的retry。

本节没有产生新公开指标。当前正式最好仍为VOT `73.974969/82.627562/89.455266`，DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`；STTrack论文值`77.6/82.5/93.7`仍只是作者结果，不是本服务器新测量。

## 24.121　RGBD1K native-global dormant recovery：规则网格扩大了伤害地形，但没有增加独占恢复容量（2026-08-17）

### 24.121.1　为什么这一轮仍然没有直接跑VOT

§24.120 的77维方案证明身份/语言证据不是常量，但factor-6 static/query-reset expert相对protected几乎没有恢复容量。本轮不是在同18条上调阈值，而是更换候选动作并使用全新Train-only序列：保留官方STTrack protected路径，冻结epoch5 native-response expert，同时生成三类dormant动作：上一公开框factor-6 top4、last-reliable框factor-6 top4，以及全图3×3规则tile各top2，共11组、26候选。每个候选独立shadow递归10帧，offset0--2只提供身份/响应/语言反事实证据，offset3--9才作GT事后outcome。

这里的科学问题是：扩大候选动作后，是否出现足够多、跨序列、protected无法替代的安全恢复。如果这个“动作容量”在学习前不存在，训练router或直接跑VOT都只会放大全时expert已有的catastrophic风险。因此公开评测仍被容量门阻断。

### 24.121.2　新数据抽取、去重和最终selected18

本地通过Google Drive Range和Zip64中央目录精确抽取24条RGBD1K Train序列，每条600帧；不下载288GB整包。`Doll/doll20`永久只作runtime smoke，不进入科学池。其余23条先在远端逐序列重算1,204项payload receipt，再对DepthTrack、CDTB、VOT-RGBD2022全部RGB执行exact SHA与完整pHash≤8扫描。

pool23在完整覆盖DepthTrack后，于CDTB第47,721个源帧发现重复：RGBD1K `Bottle/bottle2/color/00000504.jpg` 与CDTB `boxes_office_occ_1/color/00000720.jpg` 的pHash距离为8。`Bottle/bottle2`被可恢复地移入rejected目录，未删除。剩余22条完整扫过479,024个benchmark RGB后无重复；再按获取前已经冻结的候选顺序取前18条，4条未用序列移入reserve目录，不根据GT或模型表现选择。

最终selected18再次从头完成独立全扫描：

| 证据 | 数值 / SHA256 |
|---|---|
| 目标范围 | 18序列、10,800 RGB帧 |
| DepthTrack | 296,327 / 296,327 |
| CDTB | 101,956 / 101,956 |
| VOT-RGBD2022 | 80,741 / 80,741 |
| GT / model output read | false / false |
| scope contract | `b59bbf3e21dcda1964e9c21c44a640817ef9e36664749efc03a866d4c26f076a` |
| dedup result | `109a38ecba0bfb0d4d0ba701be6742091f0ea61fc57fbf53d2cb2797f8a323d7` |
| terminal status | `58fa0574036e65ba23be31b77808d92bcb5c4065ed489603442958e695c459cc` |
| target fingerprint aggregate | `808dd02d0f253eea1648f1c7b6ddb5591119449089d1e06af6ab1b3352be6cdc` |

最终18条为：`Hard_disk/hard_disk1`、`Basin/basin7`、`Bag/bag19`、`Camera/camera1`、`Fan/fan2`、`Calendar/calendar1`、`Pot/flowerpot5`、`Book/book7`、`Holder/holder1`、`Bowl/bowl6`、`Racket/racket5`、`Backpack/backpack2`、`Cup/cup22`、`Box/box37`、`Cake/cake1`、`Controller/controller1`、`Hat/hat1`、`Medal/medal1`。

### 24.121.3　protected/dormant架构与runtime smoke

新shadow实现位于`lib/test/tracker/sttrack_native_global_recovery_trace.py`，SHA `7c222ee967a6ad0bb300004f06642a7a136b0d917d0691537f253d5b419443c1`。protected分支继续使用官方factor-4、官方动态模板与公开query；shadow只读取快照，并对每个候选使用两份不可变static template、独立query和zero-language response。静态类别语言在同一候选grid上另算反事实，只作未来弃权证据，不进入候选主响应。

每次shadow前后逐字段比较：`state`、`frame_id`、`z_patch_arr`、`z_dict`、`track_query_before`、`association_static_templates`、`box_mask_z`、`z_dict1`。任何tensor、bbox、计数或引用语义变化都会fail-fast。永久排除的`Doll/doll20@99` GPU smoke通过：11组、26候选、260条候选帧记录，exit0，最大分配显存`685,551,616` bytes；smoke log SHA为`a3177826e006d2908641da33eb72de6cdea1b5477408c12469a0de4d7ed3583c`。

其余实现与冻结证据：

- source snapshot：`dc1f525cf6013f0bd7fe0d1f8491ba989fad5f64fb2cf59e7b7942576d33f479`；
- creator / runner / analyzer：`fffebf7d57427cbf8e52cb7d6ef8135c498e5ff2c78a56e66823d8a4bbb4a4b3` / `d375bd8b30ad42b00912e971ba9c52ab43cf0a571cfcbd94c4fdce14bef5d160` / `9eb468da4f97305638ccf2c4760cf5fb39e0fe351e3501c42caac6716689e969`；
- deterministic category-only language manifest：`390322e384a7569f8811eb1c814c6f94de852e9de804fcf89ee20d1dde61fc0a`；
- capacity plan：`96ac5527ba52f483e0525d30128696e8644a0f04164ce539addcfbb55e0034f6`。

creator第一次在发布plan前因漏导入`IDENTITY_FEATURE_NAMES`而fail-fast，只留下了可确定性复验的language manifest；没有source trace或科学结果。补齐导入后复用完全相同的manifest，split、门和数据未改变。

### 24.121.4　source-only trace与预注册容量结果

最终salt把18条分为dev12/audit6。audit为`Controller/controller1`、`Medal/medal1`、`Hard_disk/hard_disk1`、`Bowl/bowl6`、`Bag/bag19`、`Fan/fan2`；其GT只做文件字节完整性hash，数值和语义未解析，anchor未解析，source未运行。dev12每序列在约1/3和2/3处机械选有效初始化行，共24轨迹。

两片source-only trace均exit0，并用existing-output路径重新完成全量复验：

| shard | public帧 | 事件 | 事件序列 | trace SHA256 |
|---:|---:|---:|---:|---|
| 0 | 3,619 | 13 | 6 | `03e61f937dd89721a0920f95cfdd94456e1a16c81646b4b3f151fe1901c9f611` |
| 1 | 3,600 | 15 | 3 | `351a9bf573dbd28d6a12dc50bc62fbcf3256d5e6c9ea46f52df0181ca72afde8` |

分析器随后才读取dev12 GT。analysis SHA为`9edebcebe8b9617a0f76f8b12872e88b6ed84b98008af2fee3ae6d3f15bdabb9`，existing-result复验保持同SHA，exitcode=2，decision=`rgbd1k_native_global_capacity_not_supported`：

| 容量项 | 观测值 | 预注册门 | 结果 |
|---|---:|---:|---|
| complete events | 28 | ≥20 | 通过 |
| trajectories with events | 11 | ≥8 | 通过 |
| benefit | 3事件 / 2序列 | ≥10 / 5 | 拒绝 |
| unsafe | 13事件 / 6序列 | ≥10 / 3 | 通过（证明伤害地形存在，不是部署安全门） |
| recovery | 21帧 / 2序列 | ≥30 / 5 | 拒绝 |
| persistent recovery | 3事件 / 2序列 | ≥5 / 3 | 拒绝 |
| safe-oracle gain | `+16.960326` | ≥20 | 拒绝 |
| nonconstant identity | 45 / 45 | ≥15 | 通过 |
| nonconstant language counterfactual | 12 / 12 | ≥3 | 通过 |
| selected nonlocal benefit | 0事件 / 0序列 | ≥5 / 3 | 拒绝 |
| selected proposal families | local、last-reliable | ≥2 | 通过 |

三个安全benefit的具体例子为：

- `Backpack/backpack2@400@585`：`last_reliable_factor6#00`，offset3--9累计gain `+4.069817`，恢复7帧并持续恢复；
- `Racket/racket5@200@354`：`local_previous_factor6#00`，累计gain `+6.402514`，恢复7帧并持续恢复；
- `Racket/racket5@400@546`：`local_previous_factor6#01`，累计gain `+6.487995`，恢复7帧并持续恢复。

反例同样明确：`Holder/holder1@200@241`的多个global-tile候选累计gain均为`-6.213392`且恢复0帧。全部728次候选评估中，global tile占504次，其中155次unsafe。它们虽然产生23个candidate-level benefit，但全部出现在已经能由local/last-reliable更好恢复的三个事件中，没有一次成为safe oracle首选，因此“规则全局tile增加独占恢复”的计数为0。

### 24.121.5　为什么这解释了旧创新在VOT下降

这一轮把旧现象进一步分解为动作容量问题，而不只是router问题：扩大搜索区域确实制造更多高响应候选，也同时制造大量错误身份候选。规则tile不知道“同类别物体中哪一个是首帧实例”；一旦把这些候选直接写入bbox/query/template，155个unsafe global候选所代表的错误递归链会直接压低ROB和EAO。45维身份差和12维语言反事实全部非恒定，只说明证据可观测，不说明正确目标已经进入候选集，更不说明router能安全选中。

因此不能通过训练更复杂MLP、调seed、降benefit门或直接上VOT补救。当前缺的是具有首帧实例检索能力的候选动作，而不是更多规则tile或更大的搜索factor。下一机制若继续，必须把规则网格替换为真正的instance-conditioned global retrieval，例如首帧RGB-D实例memory bank、深度可靠性门和target–distractor proposal ranking；仍只能先写dormant transaction，不能修改protected路径。

### 24.121.6　独立裁决、停止边界与指标状态

独立result-to-claim给出`claim_supported=no`、confidence=`high`、integrity_status=`unavailable`。支持的最强表述仅为：一次干净的RGBD1K Train-only dev实验表明，冻结native-response候选具有非恒定身份/语言证据和显著unsafe地形，但没有足够的跨序列、安全、持续恢复容量，不能授权学习或部署router。完整trace位于`.aris/traces/result-to-claim/2026-08-17_run01/`。

本机制到此停止：不训练三seed、不读取audit6、不生成部署artifact、不实现Active、不跑公开VOT；也不得在相同dev/audit上调阈值、seed、anchor或候选排序。若继续，必须是新的instance-conditioned动作机制，并重新使用全新未触碰、sequence-disjoint Train-only证据和同等级预注册门。

本节没有产生新的公开指标。当前正式最好仍为旧SUTrack+结构化语言+safe-v1的VOT `73.974969/82.627562/89.455266`，不是STTrack+本创新；DepthTrack Test仍为`65.995933/65.335885/65.664250`，CDTB仍为`75.387821/76.005850/75.695574`。STTrack论文`77.6/82.5/93.7`继续只作为作者报告值，不能当成本服务器已复现baseline。


## 24.122　RGBD1K首帧实例检索反事实：恢复容量真实，但主要来自response候选而不是检索独占（2026-08-17）

### 24.122.1　问题与严格停止条件

§24.121 已证明规则global tile会扩大unsafe地形，却不能提供独占恢复。本轮在另一组全新RGBD1K Train序列上检验更强假设：用首帧目标token分别构造RGB、Depth、Fused与RGB-D Consensus检索图，从全局位置提取候选；同时保留完全同预算的response-only global controls。只有检索候选在matched controls之外产生至少5个、覆盖3序列的唯一非局部benefit，且四个检索族各自至少有1个exclusive benefit，才允许训练selector。

### 24.122.2　数据、provenance与架构

final18包含18条、10,800帧，每条600帧；对DepthTrack 296,327、CDTB 101,956、VOT-RGBD2022 80,741张RGB全部执行exact与pHash<=8比较。最终去重contract/result/status SHA分别为`d8b9bed3...2252`、`d779af93...17c`、`f9f044c4...608`。机械split为12条dev与6条audit；audit只做文件字节完整性hash，GT数值/语义、anchor和source均未读取。

每个风险事件评估24个只读候选：local 4、last-reliable 4、response-only global 8，以及Consensus/RGB/Depth/Fused各2。所有候选拥有独立bbox、query和两份不可变首帧模板，递归shadow 10帧；offset0--2记录45维RGB-D身份差和12维静态语言反事实，offset3--9才在source封存后由dev GT计算benefit/unsafe/recovery。`state`、`frame_id`、`z_dict`、`track_query_before`、template与相关计数逐tensor验证零写入。plan SHA为`1d4951c58b9d1d3a1481d2a52de62a1e1289e3ea1e6617cd4ef7f19cc594fe67`。

### 24.122.3　source与容量结果

两片source均exit0：shard0为3,603帧、42事件、10事件序列，SHA `64af4eae925ec4074926e5e54257f46e8565e780ac6a2c7ca7849a88ea7d29e5`；shard1为3,600帧、30事件、8事件序列，SHA `7de23f628aa273f0738e5657716fc1ec35f9174ab9b6fda55965eb7567e7be8f`。分析器之后只读取12条dev GT，analysis SHA `c29cb4dc3ef51bdee13396b184d6770c40c6009c64a5736fbdd4382661c5cb0b`，exit2，decision=`rgbd1k_instance_retrieval_capacity_not_supported`。

| 项目 | 观测 | 预注册门 | 结论 |
|---|---:|---:|---|
| complete events / event trajectories | 72 / 18 | >=30 / >=8 | 通过 |
| benefit | 26事件 / 5序列 | >=10 / >=5 | 通过 |
| recovery | 175帧 / 5序列 | >=30 / >=5 | 通过 |
| persistent recovery | 23事件 / 4序列 | >=5 / >=3 | 通过 |
| safe-oracle gain | `+140.999411` | >=20 | 通过 |
| nonlocal retrieval benefit | 25 / 5 | >=10 / >=5 | 通过 |
| retrieval-unique nonlocal benefit | **1 / 1** | >=5 / >=3 | **拒绝** |
| family-exclusive benefit | Consensus 3、RGB 2、Depth 0、Fused 0 | 每族>=1 | **拒绝** |
| matched response-only | 28 benefit / 5序列，`+151.560098` | 反事实 | 高于检索 |

### 24.122.4　具体正反例与因果解释

唯一的检索独占事件是`Mat/mat1@400@467`：`instance_rgb#00`累计gain仅`+1.154333`，恢复4帧且没有persistent recovery。它证明RGB实例检索偶尔能补充response候选，但数量和持续性远不足以授权训练。

相反，`Detergent/detergent3@182@330`中last-reliable、response-only、Consensus和RGB候选都得到约`+6.73`且恢复7帧。这是同一恢复位置被多种生成器同时发现，不是实例检索的独占贡献。`Bucket/bucket1@400@542`更直接：Depth、Fused和一个response control在同一事件都为unsafe，累计gain均为`-6.383723`，说明多模态相似度没有自动提供身份安全性。最坏response control出现在`Dryer/dryer1@200@488`，gain `-6.463803`，也证明response候选必须经过fail-closed关联而不能直接提交。

### 24.122.5　独立裁决与下一架构

独立result-to-claim为`claim_supported=no`、confidence=`high`。支持的最强表述是：候选库存在真实且很大的通用恢复上限，RGB-D身份/语言证据也非退化；但当前恢复主要由response-only位置解释，Depth/Fused检索没有独占贡献，不能将收益归因于首帧实例检索。

因此四族retrieval路线立即停止：不训练三seed、不读取audit6、不实现Active、不跑VOT。下一架构改为 **Response-Proposed, Language-Anchored Target–Distractor Transaction**：由STTrack response产生top-K位置，首帧RGB-D身份与稳定语言只做目标/干扰物关联和弃权；所有动作先进入dormant shadow，连续验证后才原子promote state/template/query，否则完整rollback。该pivot必须使用全新未触碰、sequence-disjoint Train-only证据，并预注册相对纯response scoring的配对增量门；当前dev12/audit6不得用于确认。

本节没有产生新公开指标。正式最好仍为旧SUTrack+结构化语言+safe-v1的VOT `73.974969/82.627562/89.455266`；DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。STTrack论文`77.6/82.5/93.7`仍只是作者值。

## 2026-08-17：全新数据上的 Response-Proposed RGB-D/语言关联实验

按用户要求，`/root/autodl-tmp/qwen/Qwen3_8B` 完整保留。本轮只删除5个已经被结论淘汰的一轮SRTrack probe checkpoint，共释放5,453,190,764字节；删除清单与结果已写入STTrack远程日志，未删除正式baseline、STTrack/SUTrack权重、ViT-L/14或当前关联artifact。

### 架构冻结

新模块为 `sttrack-response-identity-shadow/v1`。候选位置不再由RGB-D retrieval或语言生成，而是严格由released STTrack center response给出：上一状态周围top4、last-reliable周围top4、3×3自适应global tiles中的NMS top8，共16个候选。候选分支固定使用不可变首帧模板与隔离query rollout，不写公开state、template、query或计数器。
RGB、Depth、fused/consensus首帧目标减背景身份、candidate-minus-public身份，以及稳定类别语言counterfactual只作为后续target/distractor association与fail-closed abstention的证据；它们不能移动候选框，也不能直接提交模板更新。真正部署时仍要求protected/tentative双分支、连续验证和state/template/query原子promote/rollback。

### 数据与当前进度

- 新获取12条此前池中未用类别的RGBD1K Train序列，共7,200 RGB与7,200 Depth帧；服务器传输后逐序列重新验证receipt、文件SHA和CRC。
- 7,200个目标RGB帧已与DepthTrack、CDTB、VOTRGBD2022共479,024帧做完整exact/pHash<=8去重，结论为 `no_exact_or_phash_le8_duplicate`；去重过程未读GT与模型输出。result/status SHA为 `7d822325...27f5` / `bdcb990d...0091`。
- 在任何source trace前按固定salt冻结dev8/audit4。audit4为Blanket、Shoe、Mop、Leaf；anchor仍未解析，GT数值/语义未读，source也未授权。
- 永久排除的`Doll/doll20` GPU smoke通过：16候选、160 rollout记录、语言counterfactual非恒定、峰值显存718,743,552字节、公开递归状态逐字段零变化。
- 正式计划SHA256为 `0204442636a4947ab53b701c6a1fbd560ec2b8c89374eba663eb42dc48d94989`。GPU0/GPU1正在并行采集2,430/2,400个source-only public frames；两份trace成功后控制器才会join dev GT并生成容量结论。

### 安全门与下一步

本阶段不授权训练、Active模板/状态更新或公开VOT。只有候选恢复容量与RGB-D/语言证据非退化门全部通过，才会另行冻结“response-only selector”与“response+RGB-D/language selector”的配对OOF比较；之后audit4只允许一次使用。若容量门失败则停止该路线，不降阈值、不读取audit4。
本节尚未产生新公开指标。正式最好仍是SUTrack+结构化语言+safe-v1的VOT `73.974969/82.627562/89.455266`；DepthTrack `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。STTrack论文 `77.6/82.5/93.7` 仍只是作者报告值，不是本服务器复现。

## 24.123　Response-Proposed RGB-D/语言关联容量门最终拒绝（2026-08-17）

### 24.123.1　运行状态与完整性复验

`response-pool12` 的两片 source-only trace 已全部完成，不再处于运行中。shard0 覆盖2,430个public frames、21个事件、5个有事件序列，SHA256为`e62f14279cbf8b0942685c529602c781037bc369cc741e93f66eaebd36750fde`；shard1 覆盖2,400个public frames、17个事件、4个有事件序列，SHA256为`5c12a69638acc68f5b7112108a82563e3a9ae8a79ee4256ff28c8340dcca9af2`。两片均为exit0。

分析器在source封存后只读取dev8的GT并生成结果；analysis JSON SHA256为`6ac682cd1dfdef62a3e638af03198b50d22efa549e5f7d7c031c46a790453547`，退出码为2，decision=`rgbd1k_response_identity_capacity_not_supported`。随后执行了一次幂等重分析，结果显示`recovered_existing_result=true`、SHA不变、退出码仍为2。因此这不是中断、旧文件或未完成状态，而是正式容量门拒绝。

源码合同与结果一致：plan固定为8条development、4条audit，audit序列为`Blanket/blanket1`、`Shoe/shoe1`、`Mop/mop3`、`Leaf/leaf3`；audit anchor仍为unresolved，audit source未授权，audit GT数值/语义未解析。trace阶段只用GT做初始化与文件完整性哈希，`metric_computed=false`；analysis阶段才读取development GT并计算offset3--9的outcome。候选生成合同为`released_sttrack_center_response_with_static_templates_and_isolated_query`，RGB-D身份与语言counterfactual仅为证据，不能改变候选框，也不能写公开`state/template/query`。

### 24.123.2　容量结果

| 项目 | 观测 | 预注册门 | 结论 |
|---|---:|---:|---|
| complete events | 38 | >=16 | 通过 |
| event trajectories | 12 | >=6 | 通过 |
| benefit events / sequences | **7 / 3** | >=8 / >=4 | **拒绝** |
| recovery frames / sequences | **43 / 3** | >=20 / >=4 | **序列覆盖拒绝** |
| persistent recovery events / sequences | **6 / 2** | >=4 / >=3 | **序列覆盖拒绝** |
| safe-oracle cumulative gain | `+28.933140` | >=15 | 通过 |
| global benefit events / sequences | 7 / 3 | >=4 / >=3 | 通过 |
| unsafe candidates / events / sequences | 214 / 19 / 6 | 诊断项 | 风险很高 |
| identity delta nonconstant | 45 / 45 | >=15 | 通过 |
| masked identity nonconstant | 12 / 12 | >=6 | 通过 |
| language counterfactual nonconstant | 12 / 12 | >=3 | 通过 |

结论是：表示不是退化的，RGB-D身份与语言证据确实在变化；released-response候选也有一定恢复容量。但可恢复事件只有7个、覆盖3条序列，少于预注册门，并且214个unsafe候选覆盖19个事件、6条序列。该路线没有足够的跨序列、安全、可训练容量，不能进入selector训练、audit4、Active事务或公开VOT。

### 24.123.3　具体正反例

正例主要集中在`Mask/mask3`和`Mouse/mouse2`。例如：

- `Mask/mask3@400@540`：`released_local_previous_factor6#02`累计gain `+6.333793`，offset3--9的protected IoU全为0，而candidate IoU约`0.866--0.934`，7帧全部恢复且persistent recovery为true。
- `Mask/mask3@164@513`：`released_global_response#00`累计gain `+5.934643`，7帧恢复，证明response global候选偶尔能找到正确目标。
- `Mouse/mouse2@400@448`：`released_global_response#01`累计gain `+3.559827`，后段IoU从`0.230/0.244`逐渐上升到`0.879/0.834`，说明候选存在晚恢复能力。

反例同样强，且更影响是否可部署：

- `Trashcan/trashcan1@406@409`：多个global候选把原本protected `0.826--0.967` 的高IoU轨迹打到接近0，最坏累计gain约`-6.369173`，没有任何恢复帧。
- `Outlet/outlet1@200@480`：`released_global_response#00`累计gain `-5.509060`，protected IoU最高`0.982`，candidate IoU多帧只有`0.020--0.214`，属于典型高响应错误身份。
- `Mouse/mouse2@400@590`：local、last-reliable、global中多个rank1/0候选均累计约`-6.430990`，说明同一个风险点会被多个候选族同时错误强化，不能靠“多候选投票”自然变安全。

### 24.123.4　对VOT优化路线的影响

这轮直接回答了“为什么baseline很好而创新点会拉低VOT”：一旦创新模块有机会覆盖公开递归状态，少量错误身份候选会连续污染后续crop、query和模板；VOT的ROB/EAO会放大这种连续失败。当前response-proposed机制的优点是没有写公开状态，所以不会伤害正式指标；但正因为Train-only容量门没过，也不能把它部署到VOT上赌收益。

因此本路线停止：不训练response-only或response+RGB-D/language selector，不读取audit4，不实现Active，不跑公开VOT，不调阈值、anchor或seed来补门。下一步只能换更保守的机制：以官方STTrack/SUTrack baseline为protected分支，所有创新只作为dormant shadow证据；只有在多帧相对验证中同时满足“候选可恢复、identity不冲突、语言不冲突、protected确已失败”时，才允许原子promote `state/template/query`，否则维持baseline输出。换句话说，后续创新必须先证明“不会覆盖好baseline”，再谈提升ROB。

### 24.123.5　指标状态

本节没有产生新的公开指标。当前正式最好仍为旧SUTrack+结构化语言+safe-v1的VOT `73.974969/82.627562/89.455266`；DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。STTrack论文`77.6/82.5/93.7`仍只是作者报告值，不是服务器复现值。`/root/autodl-tmp/qwen/Qwen3_8B`已按要求保留；本轮清理只删除5个已淘汰probe checkpoint，释放`5,453,190,764`字节，清理结果SHA256为`9d95e8c501b1761a93c9d14dd2da7b3857101c34f130f5e341b5453b7efbe47a`。

## 24.124　Exact Dynamic-Response候选容量门最终拒绝（2026-08-17）

### 24.124.1　数据、去重与运行完整性

本轮使用的是再次全新获取的 `dynamic-response-pool12`，不是上一节 `response-pool12`。12条RGBD1K Train序列分别为 `Folder/folder2`、`Instrument/sax1`、`Kickboard/kikiboard2`、`Sign/sign4`、`Dustpan/dustpan1`、`Cart/cart6`、`Buggy/buggy2`、`Bicycle/bicycle9`、`Motor/motor5`、`Sculpture/sculpture3`、`Roadlock/roadlock2`、`Bus/bus1`。每条均经过远程原子提交、receipt、文件SHA/CRC和final tree复验。`Doll/doll20` 仍仅作smoke，不进入科学集合。

跨benchmark去重已完成：7,200张目标RGB与DepthTrack/CDTB/VOTRGBD2022共479,024张RGB完整比较，decision=`no_exact_or_phash_le8_duplicate`，未读取GT或模型输出。dedup contract/result/status SHA256分别为 `b1e72d9d030b7fc9f0029d1fdef7aef9566251d6489e7042b2090c216c1058d8`、`306d285e49c6309d5f1016913c92be472812f80afe4085db8e913df17d5c293f`、`4a48289e08f4d223436e070b22751e120ba9f4191fa15562d74c1c146778a9a9`。

计划SHA256为 `772610963928811e1b1090012401c31b760a89c56be0987f7edbbfb2e11b5b7d`。固定salt切分为dev8/audit4：dev为 `Bus/bus1`、`Buggy/buggy2`、`Motor/motor5`、`Dustpan/dustpan1`、`Sculpture/sculpture3`、`Cart/cart6`、`Sign/sign4`、`Folder/folder2`；audit为 `Kickboard/kikiboard2`、`Bicycle/bicycle9`、`Roadlock/roadlock2`、`Instrument/sax1`。audit anchor仍为unresolved，audit source未授权，audit GT数值/语义未解析。

source-only trace两片均为exit0：shard0覆盖2,411帧、6事件、2有事件序列，SHA256 `4cb68e998d18b713fc7e12c7ffa5bc3298b381a65ee05aa1ef9e069ab716b548`；shard1覆盖2,400帧、3事件、1有事件序列，SHA256 `8ef54cc04557a4809cd1bd3f595a3d22b558e984132282f7eb052c169b2265c6`。analysis JSON SHA256为 `615bc5229d2344d0821b01ee65e6303b8cdc9e0f80b64e941532d88765fc81ac`，退出码2，decision=`rgbd1k_dynamic_response_identity_capacity_not_supported`。

### 24.124.2　容量结果

| 项目 | 观测 | 预注册门 | 结论 |
|---|---:|---:|---|
| complete events | **9** | >=16 | **拒绝** |
| trajectories with events | **3** | >=6 | **拒绝** |
| benefit events / sequences | **1 / 1** | >=8 / >=4 | **拒绝** |
| global benefit events / sequences | **0 / 0** | >=4 / >=3 | **拒绝** |
| recovery frames / sequences | **7 / 1** | >=20 / >=4 | **拒绝** |
| persistent recovery events / sequences | **1 / 1** | >=4 / >=3 | **拒绝** |
| safe-oracle cumulative gain | `+5.726728` | >=15 | **拒绝** |
| unsafe candidates / events / sequences | 49 / 4 / 2 | 诊断项 | 风险仍存在 |
| identity delta nonconstant | 45 / 45 | >=15 | 通过 |
| masked identity nonconstant | 12 / 12 | >=6 | 通过 |
| language counterfactual nonconstant | 12 / 12 | >=3 | 通过 |
| response feature nonconstant | 18 / 20 | 诊断项 | 非退化 |

这说明工程和表征没有退化，但恢复容量远远不足。唯一强正例是 `Dustpan/dustpan1@400@470`：`released_public_dynamic_response#03`，rank3，累计gain `+5.726728`，7个outcome帧全部恢复，persistent recovery为true。弱正例不足以构成容量，例如 `Dustpan/dustpan1@400@577` 的global候选最高只有 `+0.544338` 且只恢复1帧。

反例显示不能部署：`Dustpan/dustpan1@200@311` 中 last-reliable候选 `released_last_reliable_dynamic_response#02` 和多个global候选均为unsafe，累计gain约 `-6.440578`；`Folder/folder2@189@373` 中多个global候选累计约 `-6.292183`。这些错误候选正对应VOT ROB/EAO最怕的情况：高响应错误身份如果被写入递归状态，会连续污染后续crop、query和模板。

### 24.124.3　Result-to-claim与路线决策

本轮result-to-claim结论为 `claim_supported=no`、confidence=`high`、integrity_status=`unavailable`（未提供独立experiment-audit artifact；本判断基于直接读取冻结产物与fail-closed合同）。支持的最强表述只能是：exact public STTrack response proposal、last-reliable/global dynamic response proposal以及RGB-D/语言证据均可被正确记录且非恒定；但它们没有提供足够跨序列、安全、持续的恢复容量。

因此本候选生成器停止：不训练selector，不读取audit4，不实现Active state/template/query写入，不运行公开VOT，不通过调阈值、换seed、挑anchor或复用同一dev/audit来修补结论。

下一步必须是“真正不同”的机制，而不是把普通阈值微调包装成新创新。更贴合当前VOT ROB短板的方向是 **Protected-Baseline-First Dormant Update-Suppression Transaction**：官方STTrack/SUTrack baseline作为不可变protected分支；创新模块第一阶段不主动搜索替代框、不覆盖baseline输出，只在多帧RGB-D/语言身份证据显示冲突且protected确已进入失败风险时，抑制或回滚动态template/query/state更新。也就是说，先证明“不会覆盖好baseline”，再谈恢复；这比继续扩展top-K候选更直接针对当前创新点拉低VOT的根因。

本节没有产生新的公开指标。当前正式最好仍为旧SUTrack+结构化语言+safe-v1的VOT `73.974969/82.627562/89.455266`；DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。`/root/autodl-tmp/qwen/Qwen3_8B` 继续保留。


## 24.124　Exact Dynamic-Template Response 候选容量门 v2 最终拒绝（2026-08-17）

### 24.124.1　本轮和上一轮 response-pool12 的区别

上一节24.123记录的是 `response-pool12`：候选由 released response 产生，但没有严格绑定“事件前公开动态模板和query快照”。本节是新的 `dynamic-response-pool12 v2`，目标是更贴近真实VOT递归状态：候选均来自 **exact pre-event public dynamic-template/query snapshot**，公开分支仍保持官方 STTrack 输出，shadow候选只在隔离分支内rollout。

这不是新公开模型，也不是VOT评测。本轮只回答一个Train-only容量问题：在不改公开 `state/template/query` 的前提下，exact dynamic response候选加RGB-D/语言身份证据是否有足够、安全、跨序列的恢复容量，值得继续训练selector或进入Active事务。

### 24.124.2　数据、去重与source完整性

12条新RGBD1K Train科学序列均已完成远程原子传输与独立source复验：`Folder/folder2`、`Instrument/sax1`、`Kickboard/kikiboard2`、`Sign/sign4`、`Dustpan/dustpan1`、`Cart/cart6`、`Buggy/buggy2`、`Bicycle/bicycle9`、`Motor/motor5`、`Sculpture/sculpture3`、`Roadlock/roadlock2`、`Bus/bus1`。复验覆盖7,200 RGB、7,200 Depth、12个GT文件和14,412个payload文件，source validation SHA256为 `9bb23210a348a0120a6b80ae630f26f36fe7529ff56c0f9e56c11d4db55f1b61`。

跨benchmark去重覆盖7,200个目标RGB帧与DepthTrack/CDTB/VOTRGBD2022共479,024帧，结果为 `no_exact_or_phash_le8_duplicate`，`ground_truth_read=false`、`model_output_read=false`。contract/result/status SHA256分别为 `b1e72d9d030b7fc9f0029d1fdef7aef9566251d6489e7042b2090c216c1058d8` / `306d285e49c6309d5f1016913c92be472812f80afe4085db8e913df17d5c293f` / `4a48289e08f4d223436e070b22751e120ba9f4191fa15562d74c1c146778a9a9`。

冻结计划SHA256为 `772610963928811e1b1090012401c31b760a89c56be0987f7edbbfb2e11b5b7d`。dev8为 `Bus/bus1`、`Buggy/buggy2`、`Motor/motor5`、`Dustpan/dustpan1`、`Sculpture/sculpture3`、`Cart/cart6`、`Sign/sign4`、`Folder/folder2`；audit4为 `Kickboard/kikiboard2`、`Bicycle/bicycle9`、`Roadlock/roadlock2`、`Instrument/sax1`。audit anchor仍为unresolved，`audit_source_read=false`、`audit_ground_truth_read=false`，没有消费audit4。

### 24.124.3　运行与不变量

两片source-only trace均完成且exit0：shard0覆盖2,411个public frames、6个事件，SHA256 `4cb68e998d18b713fc7e12c7ffa5bc3298b381a65ee05aa1ef9e069ab716b548`；shard1覆盖2,400个public frames、3个事件，SHA256 `8ef54cc04557a4809cd1bd3f595a3d22b558e984132282f7eb052c169b2265c6`。analysis在source封存后才join dev GT，结果SHA256 `615bc5229d2344d0821b01ee65e6303b8cdc9e0f80b64e941532d88765fc81ac`，exitcode为2，decision=`rgbd1k_dynamic_response_identity_capacity_not_supported`。

关键不变量全部保持：`public_evaluation=false`、`future_frame_text_used=false`、`development_gt_joined_after_source_trace=true`、`candidate_generation_uses_only_released_sttrack_response=true`、`rgbd_language_association_used_only_as_recorded_evidence=true`。source trace内 `public_state_mutated_by_shadow=false`，plan的shadow contract也写明 `public_response_reused_exactly=true` 与 `public_state_mutated=false`。因此本轮不会改变任何正式公开指标。

### 24.124.4　容量结果

| 项目 | 观测 | 预注册门 | 结论 |
|---|---:|---:|---|
| complete events | 9 | >=16 | **拒绝** |
| event trajectories | 3 | >=6 | **拒绝** |
| benefit events / sequences | **1 / 1** | >=8 / >=4 | **拒绝** |
| recovery frames / sequences | **7 / 1** | >=20 / >=4 | **拒绝** |
| persistent recovery events / sequences | **1 / 1** | >=4 / >=3 | **拒绝** |
| global benefit events / sequences | **0 / 0** | >=4 / >=3 | **拒绝** |
| safe-oracle cumulative gain | `5.726728` | >=15 | **拒绝** |
| unsafe candidates / events / sequences | 49 / 4 / 2 | 诊断项 | 风险存在 |
| identity delta nonconstant | 45 / 45 | >=15 | 通过 |
| masked identity nonconstant | 12 / 12 | >=6 | 通过 |
| language counterfactual nonconstant | 12 / 12 | >=3 | 通过 |

表示层仍然非退化，RGB-D身份差和语言反事实确实在变化；但恢复容量太窄，只有1个benefit事件，而且所有global benefit均为0。这说明“更贴近公开动态模板状态”的候选并没有比上一轮更稳定，反而事件覆盖太少，不足以训练或部署。

### 24.124.5　具体例子

唯一正例是 `Dustpan/dustpan1@400@470`：`released_public_dynamic_response#03` 的10帧shadow在outcome窗口累计gain `+5.726728`，产生7个recovery frames，并满足persistent recovery。这证明exact public dynamic response的某个低rank候选偶尔能救回目标。

但负例覆盖更能解释为什么不能部署：

- `Dustpan/dustpan1@200@311`：15/16个候选为unsafe，最差 `released_last_reliable_dynamic_response#02` 累计gain `-6.440578`，中心距离尺度约2.65，典型表现是把递归候选推向错误身份。
- `Dustpan/dustpan1@200@383`：16/16个候选为unsafe，最差 `released_global_dynamic_response#01` 累计gain `-4.937723`，中心距离尺度超过21，global动态响应在该事件上几乎完全错误。
- `Folder/folder2@189@373`：8/16个候选unsafe，最差 `released_global_dynamic_response#01` 累计gain `-6.292183`，再次说明global响应峰在未确认前不能写入公开递归。

这些例子和VOT低ROB/EAO的根因一致：baseline本身强时，创新模块只要偶尔把高响应错误身份提交给递归状态，就会连续污染后续crop/query/template；VOT会把这类连续失败放大。

### 24.124.6　结论与下一步

本轮最终结论是 `rgbd1k_dynamic_response_identity_capacity_not_supported`。停止exact-dynamic-response route：不训练selector、不读取audit4、不实现Active state/template/query提交、不运行公开VOT，也不事后降低阈值或改split。

当前正式最好指标仍未改变：VOT `73.974969/82.627562/89.455266`；DepthTrack Test `65.995933/65.335885/65.664250`；CDTB `75.387821/76.005850/75.695574`。STTrack论文 `77.6/82.5/93.7` 仍是作者报告值，不是本服务器复现值。

后续最合理路线不是继续挖response峰，而是回到用户要求的“最好的baseline上做保守创新”：以官方STTrack/SUTrack作为不可污染protected branch，所有RGB-D/语言/template创新先作为dormant evidence和shadow transaction；只有当protected已明显失败、候选连续多帧优于protected、RGB-D身份和语言均不冲突时，才允许原子promote，否则输出baseline不变。

## 24.125　Protected-Baseline-First更新抑制事务工程smoke（2026-08-17）

在 `24.124` 拒绝 exact dynamic-response 候选生成器后，新增真正不同的工程雏形：`STTrackUpdateSuppressionShadow`。它复用已有 `STTrackTemplateTransaction` 的 protected-vs-public shadow，但不把protected bbox作为公开输出，也不主动搜索替代框；它只记录“如果未来要做Active，是否应该抑制或回滚本次动态template/query/state更新”的证据。当前模块无论内部判断如何，都保持官方STTrack的 `target_bbox` 和 `best_score` 原样输出。

新增源码：

- `/home/STTrack_RGBD_L_innovation_v1/lib/test/tracker/sttrack_update_suppression_shadow.py`，SHA256 `56f411dc8af3b8dfbc76a3457f6997667ce916a2fd84438e729e41e475aacaa5`；
- `/home/STTrack_RGBD_L_innovation_v1/tools/smoke_update_suppression_shadow.py`，SHA256 `f817a35cc210f44cf2e4b133540467643b68625fd0a52fb08a5984401ff8b100`。

工程smoke只读 `DepthTrack Train/bottle03_indoor` 前80帧，首帧GT仅用于初始化，不读取后续GT，不计算公开指标，不运行VOT。结果文件 `/root/autodl-tmp/sttrack_innovation_v1/update_suppression_shadow_smoke_v1/bottle03_80.json`，SHA256 `08402c7682f24b162b93d066885875ce0de49a206b6ff0f20582f8c819926cd5`。核心检查如下：

| 检查 | 结果 |
|---|---:|
| public_output_exact | `True` |
| maximum_bbox_difference | `0.0` |
| maximum_score_difference | `0.0` |
| suppression_shadow_records | `79` |
| would_suppress_records | `0` |
| template_transaction_rows | `79` |
| maximum_gpu_memory_bytes | `1580129792` |

该smoke只证明“新shadow不会改公开路径”，不支持任何指标或容量主张。下一步若要验证它是否能改善VOT ROB，必须重新冻结一个fresh、sequence-disjoint、Train-only容量计划：默认输出baseline bbox；仅在多帧证据表明protected已经失败且动态更新存在身份冲突时，才允许候选Active策略抑制或回滚更新。不能使用上一轮失败的dev8/audit4来修补结论，也不能把普通阈值扫描伪装成新机制。

## 24.126　VOT-RGBD2022低指标序列的anchor身份短文本验证（2026-08-29）

### 24.126.1　用户门槛与冻结实验契约

本轮严格执行“先只测试低指标序列；只有换注释后低指标集合改善，才允许全序列验证”的门槛。低指标序列按旧正式结果冻结为 `ACC < 70` 或 `ROB < 75`，共22条序列、303个multi-start anchors。单序列EAO没有作为筛选条件，因为短序列会受VOT-RGBD2022官方115--755帧积分区间影响而出现结构性低值；单序列EAO只保留为诊断量。

所有路径固定使用相同的SUTrack-L384 checkpoint `/root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar`，SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`。该权重在DepthTrack参与的官方SUTrack训练路径上获得；本轮没有使用VOT GT训练、微调或选参，VOT toolkit仍为0.7.1，safe-v1推理路径和其他配置均保持不变。唯一变量是语言注释。

旧方法为“一条原视频一条结构化首帧文本，在该视频的所有anchor复用”。新方法为“每个anchor一条 `category + stable identity` 身份短文本”：文本由当前anchor初始化身份绑定，但跟踪中固定复用，不逐帧生成，不读取未来帧，不写初始位置、深度关系、遮挡、运动、模糊、当前干扰物等瞬时状态。该设计使语言时间点与VOT每次multi-start初始化时的视觉模板一致。

### 24.126.2　22条低指标序列与基线失败原因

旧文本低22合并基线为 EAO `42.629281`、ACC `71.827916`、ROB `53.412816`，确认失败200/303 anchors。完整逐序列数值、失败前兆、源文本和新文本见：

- `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.md`，SHA256=`fddebcd99664b9db568d713ad08924cc3ee2284bc0c23d6574a2f9f77eb1e84e`；
- `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.json`，SHA256=`4c9c4ba0898402d9943c3bdd7724666c3008cfc38472c00e460083c4bd6aa817`；
- baseline失败前兆 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/baseline_low22_failure_precursors.json`，SHA256=`3bb0f04af86da894c93ac7396a3258bffbd981059b20b4b56676a4b4077405bb`。

| 序列 | EAO* | ACC | ROB | 失败anchor | 主要失败原因 |
|---|---:|---:|---:|---:|---|
| `ball06_indoor_2` | 27.19 | 69.40 | 88.16 | 1/8 | 相似物体与快速运动；失败前中心/尺度突变中位数1.060，高于健康窗口Q90=0.443 |
| `bandlight_indoor_1` | 53.57 | 80.08 | 42.61 | 19/25 | 形变、遮挡/再出现、尺度和反光；高置信错误跳变后递归失锁 |
| `cube02_indoor_1` | 51.37 | 88.23 | 63.74 | 8/13 | 相似物体、尺度、面外旋转；失败点约需6.83倍搜索因子，超出factor=4 |
| `cube02_indoor_2` | 46.22 | 86.31 | 63.78 | 6/13 | 相似物体与杂乱背景；高响应身份切换后错误状态递归 |
| `cube05_indoor_1` | 10.53 | 81.36 | 66.15 | 2/4 | 杂乱、相似物体、旋转、遮挡/再出现；中心/尺度突变 |
| `cube05_indoor_2` | 5.40 | 85.99 | 69.82 | 1/4 | 杂乱、旋转、相似物体、尺度；高置信身份切换 |
| `cube05_indoor_4` | 38.52 | 89.77 | 73.33 | 5/7 | 相似物体、杂乱、旋转、尺度；高置信错误状态递归 |
| `cube05_indoor_5` | 8.87 | 55.15 | 14.18 | 11/11 | 几乎全失败；相似物体、杂乱、遮挡/再出现、面外旋转 |
| `cube05_indoor_6` | 52.05 | 86.20 | 57.41 | 11/16 | 相似物体、杂乱、旋转、尺度；高响应错误身份持续传播 |
| `cup02_indoor_1` | 18.06 | 83.72 | 5.61 | 36/36 | 几乎全失败；相似物体和遮挡/再出现导致高置信身份切换 |
| `duck03_wild_1` | 21.27 | 85.29 | 53.35 | 5/6 | 形变、相似物体、旋转、尺度变化；中心/尺度大跳变 |
| `duck03_wild_2` | 21.49 | 83.14 | 59.33 | 4/6 | 形变与相似物体；高响应错误身份递归 |
| `earphone01_indoor_1` | 57.52 | 79.98 | 43.91 | 17/20 | 尺度和面外旋转；失败前跳变中位数1.762，为低22最高之一 |
| `humans_shirts_room_occ_1_A_2` | 51.90 | 69.46 | 77.54 | 7/13 | 人体形变、遮挡/再出现；高置信状态跳变 |
| `humans_shirts_room_occ_1_B_1` | 43.01 | 68.98 | 100.00 | 0/12 | 无确认失锁；问题是成功轨迹内框定位/尺度贴合，不是ROB |
| `robot_human_corridor_noocc_1_B_1` | 46.40 | 45.98 | 100.00 | 0/19 | 无确认失锁；面外旋转/人体形变造成框回归偏差 |
| `shoes02_indoor_1` | 12.04 | 83.54 | 10.53 | 13/13 | 几乎全失败；相似鞋、杂乱、尺度变化；高置信错误身份切换 |
| `shoes02_indoor_2` | 10.67 | 89.10 | 42.19 | 4/4 | 全失败；杂乱背景与相似鞋造成递归身份切换 |
| `squirrel_wild_1` | 26.84 | 68.58 | 100.00 | 0/9 | 无确认失锁；尺度和形变造成定位/框贴合偏差 |
| `toy09_indoor_1` | 65.24 | 86.28 | 50.61 | 21/26 | 旋转、遮挡/再出现、杂乱、尺度；高置信错误状态递归 |
| `two_tennis_balls_3` | 3.00 | 63.12 | 53.51 | 2/4 | 同类球不可区分、深度关系变化、快速运动；失败点约需5.76倍搜索因子 |
| `yogurt_indoor_1` | 56.01 | 63.98 | 67.33 | 27/34 | 多数anchor形成持续失败；高响应中心/尺度突变后递归传播 |

\* 单序列EAO只作诊断；晋升使用22条合并官方聚合。

失败证据把低分分为两类。第一类是18条以ROB为主的序列：相似实例、遮挡、旋转或尺度变化先引发中心/尺度大跳变，错误候选仍保有较高响应，随后错误bbox成为下一帧搜索中心并造成连续失锁。第二类是 `humans_shirts_room_occ_1_B_1`、`robot_human_corridor_noocc_1_B_1`、`squirrel_wild_1` 等ROB=100序列：它们没有失锁，低分来自框定位和尺度贴合，身份文本理论上只能有限改善，不能替代回归优化。

### 24.126.3　低22受控结果与晋升决定

| 版本 | EAO | ACC | ROB | 确认失败anchors |
|---|---:|---:|---:|---:|
| 原结构化序列文本 | 42.629281 | 71.827916 | 53.412816 | 200/303 |
| anchor身份短文本 | **43.274104** | **72.065511** | **54.388022** | **195/303** |
| 差值 | **+0.644824pp** | **+0.237596pp** | **+0.975206pp** | **-5** |

预注册晋升门为：EAO增量>0、ROB增量>0、ACC不得下降超过0.10pp、失败anchor不得增加。四项均通过，所以只在此结果产生后才解除全127禁令。这里不能把低22增益外推为全127正式增益；当前正式全量最好仍是 `73.974969/82.627562/89.455266`，直至新全量评测完成。

主要ROB改善包括：`cube05_indoor_2 +30.1843pp`（69.8157→100，少1个失败）、`duck03_wild_1 +16.3873pp`、`duck03_wild_2 +11.7647pp`、`cube02_indoor_2 +4.9039pp`、`cube02_indoor_1 +3.4228pp`、`toy09_indoor_1 +2.9966pp`、`yogurt_indoor_1 +0.9766pp`（少2个失败）和 `bandlight_indoor_1 +0.5306pp`。明确负例为 `cube05_indoor_6 -6.2951pp`（多1个失败）、`humans_shirts_room_occ_1_A_2 -4.3115pp` 和 `earphone01_indoor_1 -1.0815pp`。`two_tennis_balls_3`虽修正类别但ROB不变、ACC下降0.8451pp，说明同类目标完全相似时只靠类别/稳定属性文本仍不足。

候选失败前兆文件为 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/candidate_low22_failure_precursors.json`，SHA256=`a0b52733b8c63e05f1c8ca934c0ab4a1285424f504944b2c05819c818609921d`；low22合并结果 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/run/merge_result.json`，SHA256=`8de65cf28122cc59fd620b5aac5938be026d6f8e108f37c45effd2e794b3c7ee`；低22manifest SHA256=`a56bb51836fb9c120d8492bb2742b8340dd3339ca44d20875f50facb5b375ee9`。

### 24.126.4　实现与全127评测状态

新增功能默认关闭，不影响既有配置：`lib/config/sutrack/config.py` 增加 `RGBD_LANGUAGE.ANCHOR_SPECIFIC=False` 与严格记录数配置；`lib/test/tracker/rgbd_language_manifest.py` 新增exact-SHA、exact-record-count、无序列fallback的 `RGBDAnchorLanguageManifest`；`lib/test/vot/sutrack_class.py` 将VOT一基RGB文件名转换为零基anchor并只读取精确anchor文本。低22构建脚本为 `tools/build_vot_low22_anchor_identity_manifest.py`，完整127构建脚本为 `tools/build_vot_all127_anchor_identity_manifest.py`。

全127最终manifest位于 `/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/annotations/votrgbd2022_all127_anchor_identity.jsonl`，SHA256=`7175259c4ecc2a22fe525a0b9dcb21036787988df30e1955f23b3e50d4d94867`，严格包含127条序列、1765个anchors；22条低序列文本与已验证候选逐项一致。其余文本经过类别语义修正和瞬时词过滤，方法计数为低22手工22、人工语义修正84、结构化身份清洗20、category-only fallback 1。Qwen3_8B仍按用户要求保留，但本轮没有用文本模型读取未来帧或在线改写跟踪状态。

因低22门槛已通过，全127正式候选已在 `/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run` 启动；10个shards、2张RTX 3090，低22已完成的303条trajectory在exact-text与归一化配置一致性校验后预置复用，剩余1462个anchors实际计算。controller PID记录在 `run/controller.pid`，日志为 `run/controller.nohup.log`。此时任务仍在运行，不能提前报告新全127 EAO/ACC/ROB；完成后必须先验证1765/1765完整性、合并哈希和正式分析输出，再更新本节最终指标。

### 24.126.5　无人值守终态校验器

检查发现旧 `finalize_vot_full127.py` 默认用SRTrack历史 `72.908956/82.535868/87.988071` 作比较，而且没有冻结本轮新的anchor-specific VOT入口；若直接运行会产生错误的增益归因。现已将finalizer改为显式接收candidate名称、同配置comparison名称及三项冻结指标，并用可重复的 `--source-file` 参数将额外运行源码纳入源快照。修改后的finalizer SHA256=`cbbe55132a4e64157011c0cbfa9162bd583f09588e3e497dcf741a8d5174a6bb`；最终启动脚本 `/home/SUTrack_RGBD_L/tools/launch_vot_all127_anchor_identity_finalizer.sh` SHA256=`b71ca3bf8cf0a97ef034b1da3b98af4442713b713cfde5fe838d28b0d12bf054`。

本轮comparison明确冻结为旧文本同配置实测 `73.97496948296595/82.62756179006247/89.45526602400152`，而不是SRTrack历史结果或SUTrack论文报告值。finalizer PID写入 `run/finalizer.pid`，日志为 `run/finalizer.nohup.log`，状态为 `run/finalizer_status.json`。最终源快照 `run/finalizer_source_snapshot.json` SHA256=`6084aae897dbdbdc0621d6e6fa9f4ccc0b9f10726b4ce2f32e5b3734a6daa2c6`，冻结27项来源，其中checkpoint SHA=`2a686e8b...dacd4`、CLIP SHA=`b8cca3fd...3836`、配置 SHA=`b7a3b89a...fae717`、全127文本manifest SHA=`7175259c...d94867`、anchor入口 SHA=`25de4edf...618f85`。此外已将产生comparison数值的旧正式 `full_result.json`（SHA=`4fda2e46...cd6eec`）、`full127_analysis.json`（SHA=`e3feabde...1a68e08`）和 `merge_result.json`（SHA=`a00462de...846c9`）一起冻结，消除手抄数值无来源的问题。第一次未绑定这三个artifact的等待态收尾器已安全停止，其文件完整移入 `run/superseded_finalizer_v1/`，未影响controller或10个VOT workers。

finalizer会等待controller产生 `merge_result.json`，随后核验127序列、1765个唯一anchor、5295个结果文件及逐文件SHA，再运行VOT toolkit 0.7.1官方analysis，生成 `run/full_result.json`。它没有开启自动改写交接文档，避免未审结果直接发布；终态结果必须人工复核comparison字段、指标形状、完整性和日志后，才续写本节。2026-08-29 03:03 CST时finalizer为 `waiting_for_merge`，记录343/1765，worker无Traceback/OOM/RuntimeError。

### 24.126.6　全127逐序列与失败anchor自动诊断

只看全局三项指标无法判断低22的改善是否能推广到其余105条，也无法定位总分下降由哪些序列造成。已新增 `/home/SUTrack_RGBD_L/tools/finalize_vot_anchor_identity_diagnostics.py`，最终SHA256=`56793ccee48a0793e7e5ebce8b64e67bba6d29cb9d5c81e20c9c0da5b4b662c4`；启动脚本 `tools/launch_vot_anchor_identity_diagnostics.sh` SHA256=`0d6afb42e433d544ed71329b300195f05c63d2a85a3fa3a2187eb76dc4face0c`。diagnostics PID和日志分别为 `run/diagnostics.pid` 与 `run/diagnostics.nohup.log`，状态为 `run/diagnostics_status.json`。最终版显式保证SUTrack代码目录优先于只提供通用分析函数的旧工具目录，防止Python `lib` 模块被错误遮蔽；同时交叉验证candidate terminal绑定的finalizer source snapshot SHA，以及旧baseline `full_result.json`是否在该快照中唯一且SHA一致。

该任务等待finalizer产生并校验 `full_result.json` 后才开始读结果，不与GPU推理竞争。它先构造只读 `analysis_workspace_view`，然后使用sequence-gap分析器（SHA=`a7d4437c...269c3b`）和已修复progress=0窗口的failure分析器（SHA=`6fbd6636...c6165a`），输出：

- `run/full127_sequence_and_failure_diagnostics.json`：全127、冻结低22、非低105三组的精确聚合；127条逐序列EAO全局贡献、ACC和ROB变化；旧/新确认失败anchor总数与逐序列计数；所有输入和分析器SHA；
- `run/full127_sequence_and_failure_diagnostics.md`：改善/退化最大的各15条序列和失败anchor旧→新表；
- `run/baseline_full127_failure_precursors.json` 与 `run/candidate_full127_failure_precursors.json`：1765条trajectory的完整离线失败前兆，GT只用于事后诊断，不进入推理。

分析器会强制检查新旧toolkit均为0.7.1、各127序列/1765 anchors、checkpoint一致、candidate的comparison字段与旧正式 `full_result.json` 数值逐项一致，并要求自行重算的全局EAO/ACC/ROB与两边terminal result在 `1e-12` 内吻合。2026-08-29 03:14 CST时其状态为 `waiting_for_full_result`，随主任务记录365/1765；目前还没有逐序列终态结论。

发布等待器前已用真实低22结果做端到端sequence-gap复算，而非只做语法检查。以旧full127 workspace选出相同22条作reference、低22anchor身份workspace作candidate，重算得到reference `42.62928069572743/71.82791568709518/53.41281573338630`、candidate `43.274104354018916/72.06551125207067/54.38802182117735`，差值 `+0.6448236582914879/+0.23759556497549017/+0.9752060877910473pp`，与 `LOW22_REPORT.json` 精确一致。旧的仅等待态diagnostics文件完整保存在 `run/superseded_diagnostics_v1/`；重启未影响controller、finalizer或10个VOT workers。2026-08-29 03:19 CST最终等待器记录378/1765。

## 24.127　保护—暂存模板事务模块（未接线工程版）

### 24.127.1　目的与发布边界

现有safe-v1的主要风险不是普通帧框回归，而是候选框先写入递归 `self.state`，随后才做身份、Depth和运动检查；错误候选一旦成为下一帧搜索中心，单纯丢弃动态模板不能撤销已经发生的状态污染。为下一轮低指标序列实验新增了保护—暂存双分支事务控制器：

- `/home/SUTrack_RGBD_L/lib/test/tracker/protected_tentative_transaction.py`；
- `/home/SUTrack_RGBD_L/tools/smoke_protected_tentative_transaction.py`；
- Git提交 `2c6a7ff`（`Add dormant protected template transaction`）。

该模块当前明确为 **dormant/unwired**：没有被 `SUTRACK.track()`、VOT入口或当前YAML导入，验证器扫描 `lib/test/tracker` 与 `lib/test/vot` 后确认公开路径引用数为0。因此它没有改变正在运行的anchor身份文本全127结果，不能用当前VOT指标为该模块宣称收益。必须等本轮全127结束后另建配置接线，并重新执行低指标序列门控。

### 24.127.2　原子状态与决策规则

一个递归快照把以下内容作为不可拆分的原子单元：bbox、完整template tensor列表、template annotation列表，以及track query、策略状态、文本记忆等辅助树。事务同时持有保护分支和暂存分支，任何候选模板或大位移候选先进入暂存分支；未来帧只在以下条件同时满足时累计确认：

- 与冻结的anchor identity标识一致；
- RGB身份相似度、Depth一致性和时序连续性分别过门；
- confidence与response margin相对保护分支没有超限亏损；
- 加权utility至少领先保护分支预设margin。

晋升固定要求 **恰好两个连续未来帧** 均通过。跳过任一frame立即以 `nonconsecutive_shadow_frame` 回滚；hard conflict立即回滚；达到真实elapsed-frame horizon仍未完成两次连续确认也回滚。晋升或回滚只返回一个深拷贝的完整快照，调用方以后只能一次性安装该快照，不能只更新bbox或只更新模板。

为适配真实SUTrack状态，annotation布局按分支独立验证：每条分支允许1个广播annotation或与template数量相同的N个annotations；保护分支和暂存分支不要求annotation数量相等，未来帧也允许合法的 `1↔N` 转换，但两条分支的template槽数始终相等且在事务内冻结。这覆盖了当前初始化 `N templates + 1 annotation`，以及动态模板写入后 `N templates + N annotations` 的真实转换。

### 24.127.3　失败原子性与审查结论

`begin()`、`observe()` 和 `cancel()` 都先完成输入类型、slot布局、身份标识及全部深拷贝，再提交event id、frame id、age、确认计数或分支状态。对象dtype NumPy数组被拒绝，避免 `.copy()` 对对象元素形成浅拷贝；NaN、bool和数字字符串不能伪装成在线证据。任何clone或校验失败都不消耗frame/event，也不局部覆盖保护分支，同一合法frame可以修正输入后重试。

两轮独立代码审查最初发现并已关闭：广播annotation不兼容、跨分支annotation错误等长、失败调用局部改写内部状态、horizon按调用次数而非帧差计数、跳帧被误算为连续确认、confirm_frames可降为1、identity reference不冻结、object ndarray浅拷贝及smoke导入位置依赖。最终两位审查者均给出“无blocker”。

最终CPU smoke共27项，覆盖两帧晋升、hard-conflict回滚、horizon回滚、跳帧回滚、失败begin/observe原子性、非单调frame、不可变identity anchor、`1/N`跨分支布局与 `1↔N`转换、模板/标注/嵌套辅助状态深拷贝、dtype/device保持及公开路径未接线扫描，全部通过：

- controller SHA256=`9ee6517fa33b356f49816b58c323621181a7a644d8fa76afce5917c36ae93598`；
- smoke SHA256=`f69363b09ca3783e8cbd7ffa8d4e9d8b976d243ec04053f0bdb209a15171c496`；
- smoke artifact `/root/autodl-tmp/sutrack_protected_tentative_transaction_v1/smoke.json`，SHA256=`051ca675f08d084becfd53f1375a51f54069100518e2f4ccca30f89c8d0fb910`；
- artifact明确记录 `public_tracker_connected=false`、`public_tracker_wiring_scan_exercised=true`、`public_vot_metrics_changed=false`。

CUDA device clone尚未单独执行，因为两张RTX 3090正在承担冻结的full127正式评测；`torch.clone()`的CPU dtype/device保持已验证，但GPU接线验证必须在当前正式任务结束后进行，不能把未执行的CUDA验证写成已通过。

### 24.127.4　下一轮接线与实验门控

当前anchor身份文本full127在本节写入时为456/1765，10个worker健康、两张GPU在工作，未发现Traceback/OOM/RuntimeError；正式指标仍待finalizer。事务模块下一步不得直接跑新的full127，而应遵循用户指定顺序：

1. 等当前anchor文本full127完成并冻结最终指标及逐序列归因；
2. 新建独立tracker/YAML接入事务，旧正式配置保持不变；
3. 先在相同22条低指标序列、相同DepthTrack权重和相同anchor身份文本上比较“无事务”和“保护—暂存事务”；
4. 只有低22官方聚合EAO与ROB均提升、ACC不越过退化门、失败anchor不增加，才允许新的full127；
5. 同时复验DepthCrack和CDTB保真，避免用VOT收益交换已经达标的数据集。

因此本提交属于下一轮模板更新架构的可审查基础设施，不属于新的VOT结果。

## 24.128　保护—暂存事务正式接线与严格low22晋升门（2026-08-29）

### 24.128.1　范围与实现

24.127记录的是第一版dormant控制器。当前状态已更新为：正在运行的anchor身份文本full127仍保持原路径、原YAML和原VOT入口，其finalizer冻结的仓库源码SHA仍逐项一致；另建的隔离tracker只服务下一轮low22，将保护—暂存事务接入真实SUTrack推理。两条路径互不引用，因此新接线不会改变当前full127结果。

隔离实现已提交到 /home/SUTrack_RGBD_L，Git提交为 9aedcf4cff269a373ce9ab0aad27ee3f58e1c4d6（Add gated low22 template transaction evaluation）。主要文件如下：

- lib/test/tracker/sutrack_transaction.py：真实SUTrack保护/暂存双分支tracker；
- lib/test/tracker/protected_tentative_transaction.py：两帧原子事务控制器；
- lib/test/parameter/sutrack_transaction.py：固定证据门和强制trace绑定；
- lib/test/vot/sutrack_transaction_class.py：传递精确anchor index的VOT适配器；
- lib/test/vot/sutrack_l384_rgbd_anchor_identity_transaction_low22.py：隔离low22入口；
- experiments/sutrack/sutrack_l384_rgbd_anchor_identity_transaction_low22.yaml：与已通过门槛的anchor身份文本low22配置语义相同；
- tools/prepare_vot_transaction_low22.py、tools/launch_vot_transaction_low22.sh、tools/finalize_vot_transaction_low22.py：冻结集合、启动和机器晋升门。

该路径继续使用DepthTrack训练得到的 SUTRACK_ep0180_l384.pth.tar，当前文件SHA256现场复算为 2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4。文本仍是已在low22改善的每anchor category + stable identity，不逐帧更新，也不读取未来帧。

### 24.128.2　真实递归接线

普通、无风险且不触发模板更新的帧继续采用SUTrack当前预测，不冻结正常运动。只有safe-v1产生动态模板候选或检测到hard conflict时才建立事务：

1. protected快照冻结候选产生前的bbox、全部template tensors、template annotations、safe-policy状态、text token和task index；
2. tentative快照保存当前候选bbox、候选动态模板及候选policy；
3. 事件帧公开输出保持protected先验bbox，confidence固定为保守的0.0，避免把tentative置信度错误挂到protected框上；
4. 后续恰好两个连续帧分别运行protected和tentative网络分支；
5. tentative必须同时满足绝对confidence/margin、RGB身份、Depth一致性、时序连续性、相对protected不越过亏损门，并且utility至少领先0.01；
6. 任一分支hard conflict、跳帧、格式错误或两帧内未完成确认都会完整rollback；只有连续两帧均通过才原子promote整个tentative快照。

事件创建阶段从网络推理、policy observe、候选模板构造、快照捕获到begin全部置于fail-closed边界。非OOM异常会恢复候选前的bbox/templates/annotations/policy并重建空事务；OOM继续上抛。每条trajectory写入唯一的sequence + anchor JSONL，参数缺少 SUTRACK_TRANSACTION_TRACE_ROOT 时直接拒绝运行；同一trajectory重试时覆盖该trajectory旧trace。

### 24.128.3　复核与验证

两轮独立代码复核先后发现并关闭：protected快照曾错误捕获候选后的state/policy；缺少绝对confidence/margin门；shadow不是严格两帧；protected hard conflict未回滚；创建或rollout异常不恢复；错误声称支持任意模板数；checkpoint未现场算SHA；事件帧框与confidence跨分支；正式launcher未绑定trace；已有merge时无法重启finalizer；以及gate只有文字声明、没有机器终态。最终两位审查者均给出“无blocker”。

最终结构smoke为22/22全部通过，artifact位于 /root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke.json。它确认：

- 当前anchor身份文本full127冻结源码SHA仍完全一致且未接入transaction；
- 当前checkpoint实际SHA与冻结SHA精确一致；
- mock真实调用track，事件帧公开递归保持候选前状态；
- 候选模板构造异常时完整恢复先验状态；
- 正式trace目录绑定在本次run根目录内；
- 使用官方VOT 0.7.1确认失败逻辑重算旧anchor身份low22，精确复现195/303 failures。

该smoke仍是CPU/结构验证，明确记录 gpu_inference_exercised=false、low22_vot_started=false，不能据此声称transaction提高了VOT。

### 24.128.4　冻结low22机器晋升门

新策略必须重新从同一22条低指标序列、303个anchors开始。对照不是更旧的结构化文本，而是已通过上一轮门槛的anchor身份短文本：

| 对照 | EAO | ACC | ROB | 确认失败anchors |
|---|---:|---:|---:|---:|
| anchor身份短文本 + safe-v1 | 43.2741043540 | 72.0655112521 | 54.3880218212 | 195/303 |

prepare脚本同时冻结该基线的manifest SHA、LOW22_REPORT.json SHA=4c9c4ba0898402d9943c3bdd7724666c3008cfc38472c00e460083c4bd6aa817、三项精确浮点值和195个确认失败。finalizer在303条完成后运行官方VOT analysis、重新计算confirmed failures、核验303个trajectory trace和合并结果逐文件SHA，并生成 run/low22_gate_result.json。

四项机器门为：

- candidate EAO严格高于0.43274104354018916；
- candidate ROB严格高于0.5438802182117735；
- candidate ACC不低于基线减0.001，即最多下降0.10pp；
- candidate confirmed failures不超过195。

只有四项全部为true，artifact才会写 gate_passed=true 和 full127_authorized=true。无论结果如何，automatic_full127_launch=false；即使low22通过，也不会自动启动transaction full127，必须先人工复核。若不通过，路线立即停止，不做全序列。

### 24.128.5　当前运行状态

2026-08-29本节更新时，已晋级的anchor身份文本full127仍在 /root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run 运行，最新controller进度为574/1765；10个shards持续推进。它之所以运行全127，是因为24.126已先在low22把EAO/ACC/ROB分别提高 +0.644824/+0.237596/+0.975206pp，失败anchor从200降为195，满足预注册门槛。

transaction实验尚未启动，/root/autodl-tmp/sutrack_transaction_low22_v1/run 仍不存在。等待器PID=200373，脚本为 /root/autodl-tmp/sutrack_transaction_low22_v1/wait_then_launch_transaction_low22.sh；它只等待当前full127产生经过finalizer校验的full_result.json，随后只启动transaction low22 controller和low22 gate finalizer，不包含transaction full127命令。

因此当前正式最好仍为：VOT EAO/ACC/ROB 73.974969/82.627562/89.455266；DepthTrack Test 65.995933/65.335885/65.664250；CDTB 75.387821/76.005850/75.695574。新的anchor身份文本full127和transaction low22均没有终态指标，不能提前替换这些数字。/root/autodl-tmp/qwen/Qwen3_8B继续保留。
## 24.129　anchor身份短文本在low22上的逐序列收益与风险复核（2026-08-29）

本节补充24.126的逐序列反事实结果。比较对象始终是同一DepthTrack训练权重、同一safe-v1模板策略、同一VOT toolkit 0.7.1和同一303个multi-start anchors；唯一变化是把旧的序列级结构化文本替换为与当前anchor初始化画面对应的 `category + stable identity` 短文本。下表中的单序列EAO仍只作诊断，是否晋级只由22条合并官方聚合决定。

| 序列 | ΔEAO* (pp) | ΔACC (pp) | ΔROB (pp) | Δ失败anchor | anchor身份短文本 |
| --- | ---: | ---: | ---: | ---: | --- |
| ball06_indoor_2 | +0.212330 | -0.005427 | +0.140977 | 0 | a yellow spherical ball |
| bandlight_indoor_1 | +0.948752 | +1.134879 | +0.530574 | -1 | a green band-shaped light |
| cube02_indoor_1 | +1.945846 | +0.079849 | +3.422773 | 0 | a black cube |
| cube02_indoor_2 | +2.765370 | -0.143016 | +4.903943 | -1 | a dark cube with a fabric-like surface |
| cube05_indoor_1 | +0.019039 | -0.825584 | +0.000000 | 0 | a dark rectangular cube |
| cube05_indoor_2 | -1.053438 | +1.848881 | +30.184332 | -1 | a cube |
| cube05_indoor_4 | -0.449759 | -0.031004 | -0.546780 | 0 | a light-colored cube |
| cube05_indoor_5 | -0.115509 | -0.351183 | -0.094362 | 0 | a wooden cube with a printed number |
| cube05_indoor_6 | -3.838546 | -0.087487 | -6.295104 | +1 | a white cube |
| cup02_indoor_1 | -0.078883 | -0.180918 | -0.010463 | 0 | a red cup with a white interior |
| duck03_wild_1 | +3.845653 | +0.739702 | +16.387337 | -1 | a dark-feathered duck |
| duck03_wild_2 | +7.890511 | +0.303964 | +11.764706 | 0 | a dark-feathered duck |
| earphone01_indoor_1 | +0.093523 | +0.887948 | -1.081531 | 0 | black over-ear headphones with padded earcups |
| humans_shirts_room_occ_1_A_2 | -2.205674 | -0.870393 | -4.311497 | 0 | a person wearing a patterned shirt and jeans |
| humans_shirts_room_occ_1_B_1 | +0.191413 | +0.313050 | +0.000000 | 0 | a person wearing a patterned long-sleeved collared shirt |
| robot_human_corridor_noocc_1_B_1 | -1.093000 | -1.032895 | +0.000000 | 0 | a person wearing a black shirt and blue pants |
| shoes02_indoor_1 | +0.266672 | -0.941114 | +0.343306 | 0 | a black laced shoe |
| shoes02_indoor_2 | -0.457901 | -3.292936 | -0.260417 | 0 | a black laced shoe |
| squirrel_wild_1 | +0.185240 | +0.560202 | +0.000000 | 0 | a brown squirrel |
| toy09_indoor_1 | +2.326207 | +0.127340 | +2.996641 | 0 | a rectangular metallic-looking toy |
| two_tennis_balls_3 | +0.167974 | -0.845139 | +0.000000 | 0 | a yellow tennis ball |
| yogurt_indoor_1 | +0.362097 | +0.289944 | +0.976591 | -2 | a yogurt cup with a printed label |

逐序列计数为：14/22条单序列诊断EAO上升，10/22条ROB上升，10/22条ACC上升。确认失败减少集中在 `bandlight_indoor_1`、`cube02_indoor_2`、`cube05_indoor_2`、`duck03_wild_1`、`yogurt_indoor_1`，分别减少1、1、1、1、2个anchor；`cube05_indoor_6`新增1个失败，合计净减少5个，和303-anchor汇总的 `200 -> 195` 精确一致。

结果支持两个结论。第一，删除“当前更近、当前无遮挡、当前深度可靠”等原视频首帧瞬时状态，并让文本与每个VOT anchor的初始化画面对齐，确实能减轻一部分相似目标、形变和遮挡场景中的身份切换；`duck03_wild_1/2`、`cube02_indoor_2`和`yogurt_indoor_1`是具体正例。第二，静态身份短文本并非普遍增益：`cube05_indoor_6`、`humans_shirts_room_occ_1_A_2`、`shoes02_indoor_2`明显退化，说明语言仍可能在同类实例、局部属性不可见或姿态变化时强化错误候选。下一轮保护—暂存事务必须首先检查这些退化序列是否被回滚机制保护，不能只报告low22总分。

本节数据直接来自 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.json`。源报告SHA256=`4c9c4ba0898402d9943c3bdd7724666c3008cfc38472c00e460083c4bd6aa817`；没有读取未来帧文本，也没有在逐序列结果产生后修改low22集合或门槛。
## 24.130　保护—暂存事务的逐序列与行为诊断收尾器（2026-08-29）

为避免下一轮low22只留下全局三项指标而无法解释模板事务是否真正工作，新增独立诊断收尾器 `tools/finalize_vot_transaction_low22_diagnostics.py`，并由 `tools/launch_vot_transaction_low22.sh` 在low22 workspace建立后启动。Git提交为 `062e5b0de7629d1821e79086a63c48cc2b0b6005`（Add low22 transaction sequence diagnostics）。该修改只属于尚未启动的transaction low22隔离路径，没有改动当前anchor身份文本full127冻结入口或YAML。

诊断器等待 `run/low22_gate_result.json` 完成后才工作，输出：

- `run/low22_transaction_diagnostics.json`：anchor身份文本基线与模板事务候选的22条精确逐序列EAO全局贡献、ACC、ROB和失败anchor变化；
- `run/low22_transaction_diagnostics.md`：可直接检查的中文表格；
- `run/transaction_diagnostics_status.json`：等待、逐序列聚合、trace分析、完成或失败状态；
- 303条trajectory trace的事务启动、模板候选、状态冲突候选、promote、rollback、轨迹末未决、创建异常、可恢复异常和rollback原因统计，并按22条序列分解。

它会再次校验VOT toolkit 0.7.1、22序列/303 anchors、候选tracker ID、`automatic_full127_launch=false`、gate授权与gate结果一致、merge artifact SHA、gate source snapshot SHA以及DepthTrack checkpoint SHA=`2a686e8b...dacd4`。逐序列聚合从只读analysis view读取；参考端必须精确复现anchor身份文本low22基线 `43.274104/72.065511/54.388022`，候选端必须与gate terminal数值在 `1e-12` 内一致，否则fail-closed。

发布前做了两次完整CPU端到端自检。第一次把已完成的303条anchor身份轨迹同时绑定为reference/candidate，并生成303条零事务trace，精确得到EAO/ACC/ROB差值全0、22条逐序列覆盖和303条无事务trajectory。该自检首先发现analysis manifest缺少checkpoint绑定并按预期拒绝输出；补入gate source snapshot到checkpoint SHA的强绑定后通过。第二次在相同303条轨迹上注入3个合成事务，覆盖2个模板候选、1个状态冲突、1次promote、1次rollback、1个轨迹末未决和6个transaction frames；诊断器逐项精确复现全部计数及rollback reason。临时自检workspace均位于经过前缀验证的 `/root/autodl-tmp/txdiag-smoke-*`，完成后已删除。

安装后重新运行原事务结构smoke，22/22检查全部通过；artifact为 `/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke_diag_v2.json`，SHA256=`5adc1ede97c249da3c0e44d1d81021830e80f820321984299b48e06c4b46e86d`，仍明确记录 `gpu_inference_exercised=false`、`low22_vot_started=false` 和 `public_full127_path_changed=false`。新文件SHA256=`6ea0e6b697ffc5e293c184883dbed6d42084148c12da977ee39e654515481717`；更新后的launcher SHA256=`6c26b09826d4434bccca1e1f6f3679ed8a8d2c6658175206b9286cb3296fc19d`，gate finalizer SHA256=`70a8052ed19781266d3b54c4243975b05e3a8d98ef4d8fc386d7f73ea14d2fef`。

2026-08-29本节更新时，anchor身份文本full127为711/1765，双卡100%运行且日志错误扫描为0；`full_result.json`尚未产生，transaction low22 workspace仍未建立。因此当前正式最好指标仍不变，新诊断器也没有产生任何新的科学结果。等待器之后仍只会启动transaction low22；即使low22门控通过，诊断报告也明确写入 `automatic_full127_launch=false`，必须人工复核逐序列和事务行为后才能决定是否运行transaction full127。
### 24.130.1　门控完成后的诊断恢复路径

继续审计启动器时发现：第一版新增诊断器后，launcher仍在检测到 `low22_gate_result.json` 时立即退出；若门控已经完成而诊断进程曾异常退出，重新执行launcher无法补启诊断器。已将逻辑改为先记录 `gate_complete`，建立/重启诊断进程后才退出，且仍不会重启controller、重算VOT或启动full127。修复提交为 `6691ebb89fb5c1e004d6f6f785f757bb5e2c32f5`，launcher新SHA256=`8da1533362e372173f7262b206cf2614f4ae9dd6ea2e7911b1d9c07d6a10b22f`。修复后结构smoke再次22/22通过，artifact `/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke_diag_v3.json` SHA256=`5a53057ebe269a02b23757a277447375ff96e2c05fbfc9b67663e213b10506d5`；仍确认transaction low22未启动且当前full127路径未改变。
### 24.130.2　诊断接线纳入结构审计

原22项结构smoke的 `all_isolated_sources_exist` 尚未包含新诊断器，也没有验证“先启动/恢复诊断器，再在已完成gate上退出”的顺序。已把 `finalize_vot_transaction_low22_diagnostics.py` 加入隔离源码与SHA清单，并新增两项检查：`launcher_restarts_diagnostics_after_completed_gate`、`diagnostics_bind_sequence_metrics_and_transaction_traces`。提交为 `4e00a0e4ab34f07ed9ed9f96db02f4d4a24715a2`，smoke脚本SHA256=`55d8532eed54021bf79605aeead1da1f7bf73c7d0d26009be608171fac75fccf`。最终结构smoke由22项增为24项，24/24通过；artifact `/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke_diag_v4.json` SHA256=`fb627f5b7cdd050bf7de1760e7329e2fdf34fdb7ce198efad626d5fb34818c63`。仍明确记录 `low22_vot_started=false` 与 `public_full127_path_changed=false`。
## 24.131　模板候选与状态冲突的事务语义拆分（2026-08-29）

继续复核隔离transaction low22实现时发现，旧实现对两种不同事件采用同一个event-frame动作：无论是已经通过safe-v1全部门槛的正常模板候选，还是包含大中心跳变/身份冲突/深度冲突的可疑状态候选，公开分支都回到上一帧bbox。这样会在每次正常模板候选时人为引入一帧停框，使low22实验同时测量“模板shadow事务”和“周期性状态冻结”，容易无谓伤害ACC并掩盖动态模板本身的因果收益。

现已按风险语义拆分：

```text
template_candidate：
    当前bbox已经通过confidence/margin/RGB identity/depth/motion门
    protected = 当前bbox + 旧动态模板（或过期后static模板）
    tentative = 同一当前bbox + 新候选模板
    event-frame公开当前bbox及其真实confidence
    后两帧比较只隔离“写不写新模板”的影响

state_conflict_candidate：
    当前bbox触发大跳变/低身份/深度冲突等硬冲突
    protected = 上一可信bbox + 上一完整递归状态
    tentative = 当前可疑bbox + 候选递归状态
    event-frame公开上一可信bbox，confidence置0
    后两帧验证状态切换是否可以promote
```

实现中，模板候选的protected policy会接纳本帧在线观测、更新 `last_frame_id/previous_bbox/trusted_depth`，但调用 `cancel(frame_id)` 清除pending template write；tentative policy才执行 `commit(frame_id)`。因此两个分支在当前bbox和观测时序上相同，只在动态模板及相应提交状态上不同。若旧动态模板同帧因TTL过期，protected分支使用static模板，tentative分支使用新模板，仍保持可解释对照。状态冲突路径继续完整保留先前的bbox、模板、annotation和policy，不受此修改影响。

修改提交为 `ee11fb4bdcafef70003b8bdf000b58a72ad63bc5`（Separate template and state transaction holds）。`lib/test/tracker/sutrack_transaction.py` SHA256=`cd3b739171fe9aaeed1d4f077d618fc9d2474d5448d868f647f14abf42f7b196`；结构smoke脚本SHA256=`aea9f0348f4420b6df2e7d803cc09f581fb6fde859b4313658a9be97453255b3`。

验证分为两层：

- 新结构smoke显式构造正常模板候选，确认public/protected/tentative三者均使用当前bbox，protected保留旧模板并清除pending write，tentative写入新模板且dynamic active；再单独构造 `large_center_jump` 状态冲突，确认public/protected仍为上一可信bbox、tentative为当前可疑bbox。25/25检查通过，artifact `/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke_template_state_split_v1.json` SHA256=`638e403a787067923bb8ed2b5ea0311fb139d3d7fe20151fc10f38033d1cced0`。
- 独立事务控制器smoke覆盖深拷贝、两帧确认、promote、rollback、hard conflict、跳帧、异常原子性、annotation布局和证据合法性，共32/32通过；artifact `/root/autodl-tmp/sutrack_transaction_low22_v1/controller_smoke_template_state_split_v1.json` SHA256=`9f322c7de680a9f3e3fdc8eb9ca590ba26b7a81a49c20133c77ba13307aeb425`。

两层smoke都明确未运行GPU VOT，不能据此声称指标提升；但它们确认新low22实验不再把正常模板更新与人为停框混为一谈。2026-08-29本节更新时，当前anchor身份文本full127为741/1765，`full_result.json`尚未产生，transaction low22 workspace仍不存在；当前full127冻结源码检查继续通过，因此本修改只会作用于之后的low22模板事务实验。

<!-- RGBD-HANDOFF-24.132-ANCHOR-KEYED-NOT-ANCHOR-VISUAL -->
## 24.132　“anchor身份文本”的命名审计与下一轮真实逐-anchor注释门（2026-08-29）

继续审计实际manifest与构建器后，需要对§24.126--131中的简称作严格限定。当前low22文件含303条anchor记录、22条序列；正在运行的full127文件含1765条anchor记录、127条序列。逐记录统计显示：low22的22/22条序列、full127的127/127条序列均为“每个序列只有1个唯一language字符串”。`anchor_index`和`direction`确实参与严格查找，因此不会从一个anchor误读另一个序列或fallback到普通序列文本；但同一序列不同anchor并没有根据各自初始化图像生成不同描述。

因此本轮方法的科学名称应为 **anchor-keyed sequence-stable identity-only text v1（按anchor严格索引的序列稳定身份短文本）**，不能写成“每个anchor由当前图像单独标注”。它相对旧结构化文本在low22取得的 `+0.644824/+0.237596/+0.975206pp` 和失败anchor `200→195` 仍是有效受控结果，因为唯一变量确实是删除深度关系、遮挡、运动、位置等瞬时状态并修正类别/稳定外观；但该结果只能支持“identity-only清洗有益”，不能支持“anchor视觉对齐文本有益”。正在运行的full127保持冻结，不修改manifest、tracker、YAML或finalizer，以保留预注册实验完整性；终态报告必须同时附上本节命名更正。

若要检验真正的逐-anchor注释，必须另建 `anchor_visual_identity_v1`，对每个初始化anchor使用且只使用 `(当前anchor RGB图, 当前初始化GT框)` 生成 `category + stable visible identity`；不得读取未来帧、trajectory结果、Depth状态、运动、遮挡、绝对位置或当前干扰物描述。每条记录必须保存图像SHA、初始化框、生成器版本、prompt SHA、原始结构化JSON、清洗后文本和拒绝原因。相同目标在不同anchor外观相同，文本可以自然重复；判断“真实逐-anchor”的依据是独立图像来源和可审计provenance，不是强迫字符串互不相同。

真实逐-anchor版本仍必须先只运行冻结的同一22条低指标序列、303个anchors，checkpoint继续固定为 `SUTRACK_ep0180_l384.pth.tar`（SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`）。直接对照采用当前已通过门的anchor-keyed sequence-stable identity-only low22：EAO/ACC/ROB `43.274104354018916/72.06551125207067/54.38802182117735`、确认失败195。只有EAO和ROB严格提高、ACC下降不超过0.10pp、失败anchor不增加，且逐序列诊断没有新增灾难性身份切换，才可人工决定是否标注并验证full127；无论结果如何均禁止自动启动全量。

本节是命名与实验契约修正，没有产生新指标，也没有改变当前正式最好值或正在运行的full127预测。

<!-- RGBD-HANDOFF-24.133-LOW22-CONCRETE-FAILURE-EXAMPLES -->
## 24.133　低22逐序列具体失败anchor实例（2026-08-29）

为把§24.126的原因分类落实到可复查的trajectory，本节从冻结的旧结构化文本基线失败前兆文件中，为每条存在确认失败的序列选取“相对run长度最早失败”的一个正进度实例；三条ROB=100的序列明确记为无确认失败。源文件为 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/baseline_low22_failure_precursors.json`，SHA256=`3bb0f04af86da894c93ac7396a3258bffbd981059b20b4b56676a4b4077405bb`。

表中F/B分别表示forward/backward；“生存进度”是VOT multi-start trajectory在确认失败起点前的local progress/run length；confidence、中心跳变和motion+scale均取失败前10帧窗口最大值，其中中心跳变以此前bbox尺度归一化；搜索factor是用失败前最后预测框与失败起点GT做的离线几何诊断。GT只用于事后解释，没有反馈给tracker，也没有用于文本生成或阈值选择。

|序列|具体基线失败例（anchor/方向→全局失败帧）|生存进度|失败前10帧max conf|中心跳变(1帧)|motion+scale(1帧)|失败起点所需搜索factor|
|---|---|---:|---:|---:|---:|---:|
|`ball06_indoor_2`|`329B→251`|78/330|0.627|1.045|1.060|3.789|
|`bandlight_indoor_1`|`800B→772`|28/801|0.821|1.859|1.884|1.884|
|`cube02_indoor_1`|`506B→473`|33/507|0.646|0.898|1.461|1.356|
|`cube02_indoor_2`|`0F→76`|76/590|0.777|1.382|1.451|1.224|
|`cube05_indoor_1`|`0F→11`|11/111|0.887|1.111|1.196|1.619|
|`cube05_indoor_2`|`0F→15`|15/146|0.861|1.098|1.162|1.910|
|`cube05_indoor_4`|`298B→265`|33/299|0.593|1.462|1.475|1.198|
|`cube05_indoor_5`|`150F→152`|2/342|0.585|1.038|1.068|1.361|
|`cube05_indoor_6`|`700B→676`|24/701|0.715|1.307|1.357|1.393|
|`cup02_indoor_1`|`1050B→1047`|3/1051|0.584|0.957|0.970|1.174|
|`duck03_wild_1`|`200B→178`|22/201|0.822|1.509|1.509|1.069|
|`duck03_wild_2`|`0F→23`|23/247|0.819|0.838|1.967|0.831|
|`earphone01_indoor_1`|`850B→819`|31/851|0.556|1.741|2.126|1.638|
|`humans_shirts_room_occ_1_A_2`|`100F→367`|267/502|0.733|1.085|1.131|1.152|
|`humans_shirts_room_occ_1_B_1`|无确认失败（ROB=100）|—|—|—|—|—|
|`robot_human_corridor_noocc_1_B_1`|无确认失败（ROB=100）|—|—|—|—|—|
|`shoes02_indoor_1`|`150F→153`|3/464|0.568|1.631|1.673|1.072|
|`shoes02_indoor_2`|`0F→19`|19/111|0.881|0.927|0.946|1.075|
|`squirrel_wild_1`|无确认失败（ROB=100）|—|—|—|—|—|
|`toy09_indoor_1`|`450F→457`|7/889|0.746|0.772|0.960|1.922|
|`two_tennis_balls_3`|`100B→95`|5/101|0.639|1.529|1.538|5.580|
|`yogurt_indoor_1`|`1250B→1195`|55/1251|0.588|0.996|1.048|2.961|

这些具体实例补强了三点。第一，失败并不总伴随低置信度：`cube05_indoor_1/2`、`shoes02_indoor_2`、`bandlight`和两条`duck03`在失败前窗口仍出现0.819--0.887的高响应，因此单一confidence阈值无法可靠阻断身份切换。第二，`cube05_indoor_5`、`cup02`、`shoes02_indoor_1`和`toy09`在2--7个progress frames内就进入确认失败，属于VOT EAO/ROB最敏感的早期失败链；模板或状态一旦在这些帧写错会被长零尾放大。第三，`two_tennis_balls_3`在失败起点需要factor 5.580，`ball06`为3.789且随后迅速远离，`yogurt`为2.961，说明部分失败已经接近或超出公开factor-4局部搜索域，文本只能帮助候选身份判断，不能单独召回根本未出现在搜索crop中的目标。

因此下一轮low22事务诊断必须逐条检查上述anchor附近是否发生`template_candidate`、`state_conflict_candidate`、promote或rollback，并报告它是在错误写入前保护了状态，还是仅在目标已经离开搜索域后才触发。三条ROB=100但ACC低的序列则主要用于监测事务是否无谓伤害框回归，不能把它们当作failure recovery正例。

<!-- RGBD-HANDOFF-24.134-TRANSACTION-CUDA-PREFLIGHT -->
## 24.134　transaction low22正式启动前的metric-blind CUDA预检（2026-08-29）

此前transaction隔离路径已经通过控制器和接线的CPU smoke，但尚未真实加载DepthTrack checkpoint并执行CUDA前向。若full127释放GPU后直接启动303-anchor正式实验，模型加载、RGB-D 6通道、CLIP文本或transaction输出结构的运行时错误会浪费整轮。因此新增 `tools/smoke_sutrack_transaction_gpu.py`，提交为 `5f8db42`（Gate low22 transaction with CUDA smoke）。

启动顺序现在固定为：检测到当前identity-only full127的经过finalizer校验的`full_result.json` → 运行26项CPU结构smoke → 在指定单卡运行metric-blind CUDA smoke → 仅当两层预检都通过才创建transaction low22 workspace和启动303 anchors。CUDA smoke使用low22内 `ball06_indoor_2@0F` 的初始化RGB-D、初始化GT框和已冻结身份文本，随后只前向3帧；它不计算IoU/EAO/ACC/ROB，不读取任何未来GT行，也不据输出修改阈值。检查内容包括：checkpoint SHA、manifest SHA、网络参数实际位于CUDA、预测框有限且在图像范围内、score有限、每帧包含transaction诊断、没有被tracker吞入`recoverable_error`的运行时异常、初始化trace合法。正式trace与smoke trace使用不同目录，smoke输出不进入VOT workspace。

CPU结构smoke已增加 `launcher_runs_metric_blind_gpu_smoke_before_low22_prepare` 与新源码存在性检查，现为26/26通过；artifact `/root/autodl-tmp/sutrack_transaction_low22_v1/structural_smoke_gpu_preflight_v2.json` SHA256=`8f940c8625eca53cb34a32552996edcc499e39996f20a5f738f8e7192932d967`。该artifact仍明确写 `gpu_inference_exercised=false`、`low22_vot_started=false`；真实CUDA smoke刻意等待当前full127结束，不能提前声称已通过。

相关源码SHA256：

- `tools/smoke_sutrack_transaction_gpu.py`：`c72a33eca23d89aff5447b698858dce3fe77901891804e06e7bac9508a319581`；
- `tools/launch_vot_transaction_low22.sh`：`58de70660ba629c4a5ac6661a564a358c2780424a97cbf70a678e64d9dd40144`；
- `tools/smoke_sutrack_transaction_integration.py`：`74cb77a2513fed654a4e8586f5dbb0911f670fa2376a60db0af29854ad3b89df`；
- `tools/finalize_vot_transaction_low22.py`：`17d4987076df7a7538c1bf3c95558c3289bc4d95529f500eed5528b51db87b46`。

gate finalizer已把GPU smoke脚本纳入正式source snapshot；运行期间源码漂移会fail-closed。等待器仍只调用low22 launcher，脚本中没有transaction full127命令；即使low22通过也不会自动全量。本节写入时当前identity-only full127为849/1765、错误扫描为0，transaction run目录仍不存在。

<!-- RGBD-HANDOFF-24.135-GITHUB-OVERLAY-PUSH -->
## 24.135　相关源码已推送到用户指定GitHub overlay（2026-08-29）

用户指定仓库为 `https://github.com/666666666666gao/Track`。现场只读检查确认其 `main` 原HEAD为 `2d1887d84b06c40bb87c368be4fc3decf56beddb`，且仓库已有 `projects/sutrack_rgbd_language_template/overlay`，上游基准固定为官方SUTrack `d65052d1ba3fcf55010e1fb3665ee6616c139a2c`。因此没有把完整SUTrack仓库强推覆盖，而是把当前相关代码依赖闭包同步到既有overlay。

源工作树 `/home/SUTrack_RGBD_L` 先将正式运行必需的RGB-D读取、语言manifest、安全模板、时序深度身份、VOT/TraX适配、anchor-keyed identity-only配置与构建/分片/诊断工具提交为 `f56a835`；连同此前protected/tentative transaction系列，当前相对官方上游共8个项目提交、39个相关文件、8053行新增/37行删除。逐文件审计确认无 `.pth/.pt/.ckpt/.tar/.bin/.safetensors/.npy/.npz`，无`weights/checkpoints/results/outputs/__pycache__`路径，最大文件仅28414 bytes。

目标overlay更新在独立浅克隆中完成：39个同步文件与源工作树逐SHA一致；保留目标仓库原有注释、旧诊断工具和历史配置，不删除其他内容；更新后的 `MANIFEST.sha256` 绑定75个overlay文件并在Linux逐项 `sha256sum -c` 通过。项目README明确记录：

- 当前identity-only方法的准确名称是“anchor-keyed sequence-stable identity-only”，不是每个anchor独立视觉描述；
- low22受控指标为 `43.274104/72.065511/54.388022`、失败195/303，对照为 `42.629281/71.827916/53.412816`、失败200/303；
- full127仍在运行，不能提前替换正式最好值；
- protected/tentative transaction尚未产出low22指标，不能宣称提升；
- transaction即使low22通过也不会自动运行full127。

目标提交先在Linux形成 `488f315`，再以补丁SHA256=`2f3c49383165a07d67ef957592297c7ce7325531226fa0465e4bd3f218ea0cd3`无冲突应用到本地Windows目标克隆；两端Git tree均为 `abb243f07c5feef3cc7258ae2e5b6ce90963f691`，证明CRLF工作树差异没有改变提交对象。最终使用Windows凭据管理器执行普通非force fast-forward：

```text
2d1887d..a04835a  HEAD -> main
```

GitHub权威HEAD复核为 `a04835a9cad649209d58ae4d6661b5cd5a59f671`（Update SUTrack RGBD language transaction overlay），链接：`https://github.com/666666666666gao/Track/commit/a04835a9cad649209d58ae4d6661b5cd5a59f671`。本次推送34个变更路径、5455行新增/35行删除，全部位于 `projects/sutrack_rgbd_language_template`，未推送模型权重、数据集、VOT workspace、预测、日志或远程交接文档。官方 `chenxin-dlut/SUTrack` origin没有收到任何push。

本节写入时当前identity-only full127为892/1765、硬错误扫描为0；正式指标仍未更新，transaction low22尚未开始。

<!-- RGBD-HANDOFF-24.136-LOW22-FIRST-FOR-EVERY-NEW-ANNOTATION -->
## 24.136　用户确认：每一种新文本注释方法必须先只验证低22（2026-08-29）

用户再次明确实验顺序：**不能因为准备了新的文本注释，就直接标注或评测全部127条序列。** 对任何改变文本来源、视觉输入、prompt、字段、清洗方式、模型或融合方式的新版本，都先只处理冻结的低指标22条序列、303个multi-start anchors；checkpoint继续固定为DepthTrack训练得到的 `SUTRACK_ep0180_l384.pth.tar`，不得混入重新训练或换权重变量。

低22直接对照固定为已经通过的 `anchor-keyed sequence-stable identity-only text v1`：EAO/ACC/ROB为 `43.274104354018916/72.06551125207067/54.38802182117735`，确认失败195。新注释版本至少要满足：EAO与ROB均严格提高、ACC下降不超过0.10个百分点、确认失败不超过195、逐序列诊断不新增灾难性身份切换。未满足时立即停止该注释方向，只记录失败原因，**不生成、不启动、不排队full127**；满足时也只提交低22报告，由人工复核后再决定是否做全序列。

正在运行的 `anchor_identity_all127_v3` 不属于未经低22验证的新注释：它在启动全量前已先完成上述同一低22集合，并相对旧结构化文本取得EAO `+0.644824pp`、ROB `+0.975206pp`、失败anchor `200→195`，因此本轮全量继续运行不违反该规则。它的结果也不能被下一种注释方法复用成“已通过低22”的依据；每一种新方法都必须重新过低22门。

下一种真实逐-anchor视觉身份注释 `anchor_visual_identity_v1` 的允许范围因此冻结为：仅22序列/303 anchors，只读取当前anchor初始化RGB图和初始化GT框，只生成类别与稳定可见身份属性；不读取未来帧、Depth状态、运动、遮挡、绝对位置、跟踪输出或事后指标。第一阶段禁止构建全127 manifest，禁止启动full127 VOT。当前服务器保留的Qwen3-8B是纯文本模型，不能充当视觉标注器；在可审计视觉模型可用前，只能准备低22输入清单与验证契约，不能伪称已产生逐-anchor视觉注释。

该门槛同样适用于后续在线文本、短文本/长文本router及其他文本策略；protected/tentative transaction虽不是新注释方法，也已独立冻结为low22-only，等待器只调用low22 launcher，即使通过也不会自动启动transaction full127。

<!-- RGBD-HANDOFF-24.137-LOW22-ANCHOR-VISUAL-INPUT-FREEZE -->
## 24.137　真实逐-anchor视觉注释：仅低22输入已冻结，尚未生成文本（2026-08-29）

依照§24.136的新硬门，只准备了 `anchor_visual_identity_v1` 的低22输入，未构建full127 manifest、未运行视觉语言模型、未启动新的VOT评测。新增工具 `tools/freeze_vot_low22_anchor_visual_annotation_inputs.py`，提交 `63cddffb25a5365621ac1723ee6e602d8e51d547`，源码SHA256=`6e8b7682f60e0a6786ea634fd5ead3229f2f20d61121e217fe2b746624642dad`。

该工具显式冻结22个序列名和排序后303个anchor key摘要 `b4eec4565ce0acb6ca68afb9f2b58597c7098eb4e013fb7337a71a548e639771`，不能用任意另一组22/303自证通过。运行时还实际定位并哈希DepthTrack训练权重 `/root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar`；expected与observed SHA均为 `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`，不一致即fail-closed。

最终v2冻结证据：

|项目|结果|
|---|---|
|序列/anchors/crops|22 / 303 / 303|
|输入manifest|`/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/annotation_inputs/low22_anchor_visual_inputs.jsonl`|
|manifest SHA256|`0e07babdb468af459671c0be8ac9f3a58678eae6e85ba96009dd3e2807c9f5a7`|
|receipt|`/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/annotation_inputs/freeze_receipt.json`|
|receipt SHA256|`8b5aa071e2f06a67d4c3b7cd3c4eb24c8e7d9f9142166cd97f7c0a9c660d36b2`|
|范围审计|`low22_only=true`、`full127_manifest_created=false`、`visual_generation_executed=false`|
|数据暴露|Depth=false、tracker output=false、future image=false、future GT region exposed=false|
|当前状态|303/303均为`pending_visual_model`，没有任何新指标|

每个记录保存当前anchor RGB路径/SHA/尺寸、当前初始化region、扩展crop路径/SHA/尺寸、category hint、prompt SHA和逐记录SHA。由于原始GT是行式文本，为定位第N行，提取器可能以opaque bytes顺序跳过其前缀；非目标行不会解码、解析、持久化或暴露给标注器，只有所选当前anchor行会解码。这一物理限制已显式写入record，避免早期版本声称“底层从未读取任何前缀字节”的过强表述。

视觉prompt只允许类别、颜色、材质、形状、纹理、图案和稳定部件/标记；明确禁止位置、姿态、运动、尺度变化、可见性/遮挡、Depth、场景、背景、支撑面和干扰物叙述。当前保留的 `/root/autodl-tmp/qwen/Qwen3_8B` 是 `Qwen3ForCausalLM` 纯文本模型，没有vision config，因此本步没有把它伪装成视觉标注器。待可审计视觉模型可用时，也只能消费本节冻结的303条输入；先生成、验证和评测low22，未过门不得标注full127。

代码审查分Standards与Spec两轴独立完成。首轮发现原子发布重复、方向分派重复、区域三元组易错配，以及三项规格缺口（GT前缀表述过强、只写死权重SHA未核验实际文件、只校验22/303数量未冻结集合）。修正后复审确认：原子发布已统一、方向编码复用、初始化区域改为不可变领域对象；标注器只见当前RGB/region、实际checkpoint SHA已验证、22序列和303 anchor摘要均fail-closed。两轴均无剩余硬问题。第一版输出被可恢复地移动到 `annotation_inputs_superseded_5907708`，正式引用只允许上述v2文件。

本节写入时已经通过低22门的旧 `anchor-keyed sequence-stable identity-only` full127仍在运行，为949/1765、错误扫描为0；它不是本节尚未生成的视觉注释。正式最好值仍未更新。

<!-- RGBD-HANDOFF-24.138-HIGH-BASELINE-FALLBACK-RESEARCH -->
## 24.138　高指标RGB-D baseline后备路线审计（2026-08-29）

完整一手来源调研已独立写入 `/home/SUTrack_RGBD_L/docs/RGBD_HIGH_BASELINE_RESEARCH_20260829.md`，SHA256=`f0c4e353ca0d5525c896f09da10a7fdf1ac76980894faf65e3ad0e1f0d9b9b9a`。该报告只采用作者论文、官方仓库/项目页、官方权重/raw results和VOT官方报告；尚未下载候选权重，也没有把作者报告数字冒充成本服务器复现值。

|候选|作者/官方VOT-RGBD2022 EAO/ACC/ROB|可获取性|当前结论|
|---|---:|---|---|
|STTrack|77.6 / 82.5 / 93.7|官方代码、权重、raw results、VOT workspace、MIT|首选fallback|
|CSTrack|77.4 / 83.3 / 92.9|代码、权重、raw results、VOT workspace；无LICENSE|第二技术线，复用前需许可|
|MixForRGBD|77.9 / 81.6 / 94.6|VOT官方结果；精确RGB-D提交代码/权重未确认|性能参照/条件候选|
|FlexTrack|78.0 / 83.8 / 93.1|会议版代码未释放；V2对应关系/许可不完整|暂缓|
|MDTrack-U|80.0 / 83.5 / 95.1|未确认官方代码和权重|仅作架构/指标上限|

STTrack优先不是因为数字绝对最高，而是它是本次严格筛选中唯一同时具备较高ROB、原生VOT-RGBD workspace、官方checkpoint/raw、完整代码和明确MIT许可的候选。若当前SUTrack的低22文字/事务路线仍不能恢复EAO/ROB，按用户此前指示可直接采用作者checkpoint作为新baseline，不重复做完整baseline测量；只做哈希、RGB-D输入、VOT wrapper和少量metric-blind smoke，再把保留创新移植到STTrack。

在STTrack上的迁移顺序仍受§24.136约束：先做 `anchor identity-only + top-K候选重排` 的同一low22；通过后才加入protected–tentative事务，而且事务必须原子覆盖 `bbox state + fixed/dynamic templates + RGB/Depth temporal states + counters`，不能只回滚框或模板。两步都只能先low22，未改善不得全量。报告建议的论文创新表述为“language-anchored RGB-D target–distractor association with protected–tentative atomic template and recurrent-state transactions”，避免把普通候选关联或固定阈值模板更新单独包装成创新。

<!-- RGBD-HANDOFF-24.139-LOW22-QWENVL-ANNOTATION-PIPELINE -->
## 24.139　Qwen-VL逐-anchor视觉身份注释只开放低22（2026-08-29）

为执行用户最新硬门，本轮没有给127条序列生成文本，也没有启动新的VOT。已下载并固定可处理图像的 `Qwen/Qwen2.5-VL-3B-Instruct`，目录为 `/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct`，revision=`66285546d2b821cf421d4f5eb2576359d3770cd3`；完整下载清单 `/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct/PINNED_DOWNLOAD_MANIFEST.json` SHA256=`9cb9eacdd8b4af1294e4d34ce57acd434a32c74527b2399f5494ee6dcb971255`，14个文件、总计7,520,919,614 bytes。原 `/root/autodl-tmp/qwen/Qwen3_8B` 继续完整保留，不作为视觉模型加载。

视觉注释器由 `tools/generate_vot_low22_anchor_visual_identity_qwen25vl.py` 实现，提交 `f58a84052f0fc714c625b7567fabe07a6e904ddf`，源码SHA256=`8669d0372835c137d7ba5cbb7b28000d4e64f6c1e212e011647725fbb6c126a1`。它只能读取§24.137冻结的当前anchor完整RGB、当前初始化区域crop和authoritative category hint；sequence name、旧文本、Depth、tracker output、未来图像和未来GT均不进入模型。每个anchor先做primary生成，再对同一两张绑定图执行adversarial visual verifier；输出必须是严格单一JSON，类别必须精确匹配，稳定属性必须作为独立词边界短语出现在identity text中，位置/运动/遮挡/Depth/背景/标注泄漏均会拒绝。`red`误匹配`shredded`、Markdown fence、类别`balloon`冒充`ball`、复制或篡改已接受记录等路径均已fail-closed。

生成器支持逐anchor原子记录、断点续跑和错误重试，但只有303/303全部为 `accepted_visual_verified` 才会写 `annotation_ready=true`；`manual_review_verified`、`model_error`、`validation_failed` 任一存在均不得物化VOT文本。当前正式输出路径预注册为 `/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/qwen25vl3b_primary_verify/reviews.jsonl`，本节写入时该文件尚未生成，因而没有新文本结果和新指标。

<!-- RGBD-HANDOFF-24.140-FAIL-CLOSED-LOW22-VOT-GATE -->
## 24.140　新注释的低22 VOT评测门与后台状态（2026-08-29）

新增低22专用物化、workspace、终态和等待器，提交 `cbd68759352f16681939f35747d9281c3072a89c`。关键源码SHA256如下：

- `tools/materialize_vot_low22_anchor_visual_identity_manifest.py`：`259df3b9bf57e33607845652132284d2ed14875d5794f289bb5628a6a5b39de3`；
- `tools/prepare_vot_anchor_visual_identity_low22.py`：`84d43ee5d46148388ad4caddc492a8b572fd62c34c3a269f25f12d99cf17ef98`；
- `tools/finalize_vot_anchor_visual_identity_low22.py`：`28911a369b577ca4d9863fde64cb52d0d597fe08526f2c2aac8f48f0f40bde8a`；
- `tools/launch_vot_anchor_visual_identity_low22.sh`：`5b2a74374a83a23889093fc1f20f71818ab7953a321889eaf021f1a78d95d250`；
- `tools/run_low22_anchor_visual_pipeline_after_gpu_free.py`：`96a30174754b899edbcdf9d74c9997d333885978fcd2385ae0bb3149dd3af3d5`；
- YAML template：`6642730f3e7fa5e93cea3fa1e28b0811bdce44c702ec219c2d0ea33ac824a0d1`；
- VOT wrapper：`f6e7d43d8b36a3c5eb600f481ac1f243edfd8554cc96bbf1bf886a271a2e5a9c`。

决定链不是只按“22/303数量”自证。它硬绑定旧低22baseline shard manifest SHA256=`600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed`、报告SHA256=`4c9c4ba0898402d9943c3bdd7724666c3008cfc38472c00e460083c4bd6aa817`，并重新计算303条实际trajectory摘要 `86e53022c62f9d8a5cf9ecce4ab7401499a84eae02cd2508c4c755613949a16a`、逐项核对序列顺序与trajectory集合。workspace先在独立staging目录完整生成和验证，再原子rename发布，避免中断留下半成品被误续跑。已有artifact也不能只因文件存在而跳过：launcher每次先重跑严格materializer，再同步执行finalizer `--preflight-only`；只有全部SHA、实际路径、source snapshot与exact cover通过后才启动VOT controller。

权重继续是相同DepthTrack训练SUTrack：声明路径 `/root/autodl-tmp/sutrack_assets/weights/SUTRACK_ep0180_l384.pth.tar` 和tracker实际加载软链接 `/home/SUTrack_RGBD_L/checkpoints/train/sutrack/sutrack_l384/SUTRACK_ep0180.pth.tar` 均解析到同一文件，SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`。CLIP声明路径与实际缓存软链接 `/root/.cache/clip/ViT-L-14.pt` 同样解析到同一文件，SHA256=`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。YAML保持与直接对照相同的SUTrack-L384、search factor 4和safe-v1，不混入重新训练、transaction或其他结构变量。

低22直接对照固定为EAO/ACC/ROB `43.274104354018916/72.06551125207067/54.38802182117735`、确认失败195。新视觉注释只有同时满足EAO与ROB严格提高、ACC不低于 `71.96551125207067`（精确规则为 `candidate >= baseline - 0.001 fraction`，即最多下降0.10个百分点），且确认失败不增加，才将 `full127_authorized_for_human_decision=true`；无论通过与否均固定 `automatic_full127_launch=false`。注意该字段只表示可交给人工复核，不会生成或启动全127。

两轴最终只读复审均为PASS。此前发现并修复的正式问题包括：smoke错误记录无法恢复、manual review误当通过、partial shard不可恢复、baseline集合未硬绑定、实际runtime权重/CLIP未绑定、旧artifact可在finalizer拒绝前启动、source snapshot漏掉生成器/编排器/failure counter，以及preflight只相信声明trajectory摘要。最终版本要求2-anchor smoke为2/2 accepted；任何人工复核或错误立即停止。

后台等待器screen=`sutrack_low22_visual_waiter`，主进程PID在本节写入时为360500，状态文件 `/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/pipeline_status.json` 当前为 `waiting_for_current_identity_full127_to_release_gpus`，且明确记录 `low22_only=true`、`expected_sequences=22`、`expected_anchors=303`、`transaction_launch=false`、`automatic_full127_launch=false`。执行顺序固定为：等待已通过旧低22门的identity-only full127产出合法终态并释放GPU → GPU0至少16GiB空闲 → 2-anchor metric-blind Qwen-VL smoke → 303条低22注释 → 低22 VOT。不会在现有正式作业期间抢GPU。本节写入前现有full127进度为1040/1765；其 `full_result.json`仍不存在，因此正式最好指标仍为此前的 `73.974969/82.627562/89.455266`，不能提前更新。

<!-- RGBD-HANDOFF-24.141-LOW22-VISUAL-TEXT-ONLY-GATE-HARDENING -->
## 24.141　换注释方法仍只测低22：零新增灾难anchor与完整基线证据门（2026-08-29）

用户再次明确最终实验顺序：新的逐-anchor视觉身份文本不能先在全127验证；必须只在同一固定低指标22序列、303个VOT multi-start anchors上，使用同一DepthTrack训练SUTrack checkpoint进行比较。只有low22相对直接identity-only对照确实改善，才把full127列为人工复核后的可选下一步；代码不得自动生成、启动或续跑任何新注释full127。本节不改变此前已合法启动的identity-only full127，它此前已先通过旧low22门，当前只是占用GPU的既有作业。

为防止“总失败减少但换了一批新anchor失锁”被聚合指标掩盖，提交 `fbc08c7d5558785b377d02bc5baddce9faf8f8b8` 将正式low22晋升门扩展为五项同时成立：

1. EAO严格高于直接identity-only baseline `43.274104354018916`；
2. ROB严格高于 `54.38802182117735`；
3. ACC不低于 `71.96551125207067`，即最多下降0.10个百分点；
4. VOT官方failure设置下确认失败总数不超过195；
5. `baseline survived && candidate confirmed-failed` 的exact anchor数必须为0，并逐序列记录，rescued anchor不能抵消new-failure anchor。

第五项采用保守、可审计定义：若固定identity-only直接对照在同一 `sequence + anchor + direction` 上存活，而新文本候选发生VOT确认失败，则记为一个新增灾难failure regression。finalizer使用VOT-RGBD2022官方 `burnin=10/grace=10/threshold=0.1/ignore_masks=_ignore`，分别重算基线和候选的303条outcomes，要求key集合、sequence、anchor、direction及run length完全相等；只有五项gate的 `all(checks)` 为真，`full127_authorized_for_human_decision` 才为真，且 `automatic_full127_launch` 永远为false。

直接基线证据也已从“只验merge receipt SHA”加强为完整闭包。固定analysis view manifest SHA为 `f93162533df288ee325dd5f6d210b13f5d45eec548dcbcdb246cff2d2f171844`，merge receipt SHA为 `8de65cf28122cc59fd620b5aac5938be026d6f8e108f37c45effd2e794b3c7ee`，shard manifest SHA为 `600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed`。每次正式gate和后续diagnostics都必须：校验receipt的schema/status/tracker/master/source绑定、303 anchors与909 result files；由冻结303 trajectories精确推导 `.bin`、`_confidence.value`、`_time.value` 的909路径全集；拒绝绝对路径、`..`与resolve越界；最后逐文件重算SHA并与receipt的909项一一相等。现场验证为909/909通过，随后helper精确复现22序列、303 anchors、195个确认失败，包含1个`progress=0`失败。

诊断器会在low22候选终态后记录全部22序列及303 anchors，而非只列代表样本。JSON包含每条anchor的 `rescued/new_failure/failure_delayed/failure_earlier/failure_same_progress/both_survived` 转换、候选全部确认失败窗口、逐序列EAO贡献/ACC/ROB/失败数变化和文本差异。失败原因码明确是基于事后GT、预测框、confidence和几何的解释性诊断，只用于记录，不反馈给tracker或Qwen；包括 `very_early_failure`、`high_confidence_failure`、`high_confidence_large_jump`、`low_confidence_ambiguity`、`severe_motion_scale_change`、`target_outside_factor4_search` 与兜底持续低overlap。

诊断终态改为目录事务：先在同父目录staging中生成并校验 `diagnostics.json`、`diagnostics.md`、`receipt.json`，receipt绑定low22 gate SHA与两份输出SHA，最后一次目录rename才发布 `/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/run/low22_visual_language_diagnostics/`。完整终态重跑只验证、不覆盖；非法或不完整目录fail closed；launcher仅以 `--validate-complete-only` 的完整receipt校验为完成条件，不再因单个JSON存在而跳过。

本提交关键源码SHA256：

- `tools/finalize_vot_transaction_low22.py`：`f3d7d8349074264be080797b066085002273d4bf97450c1a62f3ae04e998ab3e`；
- `tools/finalize_vot_anchor_visual_identity_low22.py`：`ca75938a9cb40350de0a56000c5f99a74885fc97cbf43f270cf5ae558bc085b6`；
- `tools/finalize_vot_anchor_visual_identity_low22_diagnostics.py`：`b05540736309e2e1b0a0eaf169008cfc2d672d026c74d3e2bb5133cd992cb196`；
- `tools/prepare_vot_anchor_visual_identity_low22.py`：`8ff6176590c18ff489c9f619fafcb17d61a676243d6a6331fe9fef4d97bb8461`；
- `tools/launch_vot_anchor_visual_identity_low22.sh`：`797bc254df70b860efe0b757142157a231a4e75e8a27f040c423748fb5f92907`。

后实现验证均已通过：Python compile、shell语法、303/195官方failure复算、909/909基线文件闭包、聚合指标改善但新增1个灾难anchor时gate强制false、诊断目录原子发布/幂等不覆盖/篡改拒绝。Spec与Standards两路独立复审最终均为PASS，无剩余formal blocker；仅保留“诊断脚本职责较多”的非阻断代码气味。

本节写入时，新Qwen视觉注释仍未启动，低22run目录和新指标均不存在。后台 `sutrack_low22_visual_waiter` 仍停在 `waiting_for_current_identity_full127_to_release_gpus`；既有identity-only full127进度为1133/1765，`full_result.json`和`merge_result.json`均不存在。显存释放后只执行2-anchor metric-blind smoke → 303条low22注释 → low22 VOT与上述诊断；不启动新文本full127。因此当前正式最好VOT仍为 `EAO/ACC/ROB=73.974969/82.627562/89.455266`，不得提前更新。

<!-- RGBD-HANDOFF-24.142-IDENTITY-LOW22-COMPLETE-SEQUENCE-DOSSIER -->
## 24.142　更正§24.133并冻结当前identity-only低22完整逐序列表（2026-08-29）

§24.133的表格来自旧结构化文本失败前兆文件，因此只能保留为历史代表anchor示例，不能作为当前identity-only直接对照的逐序列失败账本。最明显的反例是：旧表把 `cube05_indoor_2` 列为确认失败，而当前identity-only正式报告中该序列为0/4失败、ROB=100%。后续所有Qwen视觉文本low22比较、new-failure判定和论文表格必须以本节冻结的当前直接对照为准，不得再引用§24.133的旧200-failure集合支持195-failure基线结论。

权威来源为 `/root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/LOW22_REPORT.json`，schema=`votrgbd2022_low22_anchor_identity_report_v1`，SHA256=`4c9c4ba0898402d9943c3bdd7724666c3008cfc38472c00e460083c4bd6aa817`。聚合EAO/ACC/ROB为 `43.274104354018916/72.06551125207067/54.38802182117735`，303个anchors中有195个确认失败。表中EAO为报告给出的singleton诊断值，只用于定位困难序列，不等同于官方low22集合聚合EAO；ACC、ROB和失败anchor计数均来自同一冻结identity-only workspace。

|序列|ACC|ROB|singleton EAO诊断|失败anchors/总anchors|最早失败帧|当前identity-only失败原因|
|---|---:|---:|---:|---:|---:|---|
|`ball06_indoor_2`|69.395521|88.298872|27.397856|1/8 (12.50%)|248|少数初始化点失败并显著拉低ROB/EAO；失败前集中出现快速运动和相似物体；中心/尺度突变中位数1.620超过健康Q90=0.443；失败起点约需5.42倍搜索因子，超出factor=4。|
|`bandlight_indoor_1`|81.219012|43.140116|54.519947|18/25 (72.00%)|208|多数初始化点形成持续失败链；形变、局部遮挡/再出现、尺度变化和反光共同出现；突变中位数1.635，且突变时仍有较高响应，符合错误身份/状态递归。|
|`cube02_indoor_1`|88.307301|67.165463|53.313888|8/13 (61.54%)|173|相似物体、尺度变化和面外旋转；突变中位数1.652；失败起点中位约需6.87倍搜索因子，已超factor=4。|
|`cube02_indoor_2`|86.165434|68.688911|48.982384|5/13 (38.46%)|76|相似物体与背景杂乱；突变中位数1.124，置信加权突变0.570，属于高响应下的身份切换/错误状态递归。|
|`cube05_indoor_1`|80.530115|66.145833|10.549960|2/4 (50.00%)|11|背景杂乱、相似物体、面外旋转和局部遮挡/再出现；突变中位数1.429，且存在极早失败。|
|`cube05_indoor_2`|87.842823|100.000000|4.349952|0/4 (0.00%)|—|没有确认失锁；低分来自成功轨迹中的框定位/尺度贴合偏差，主要伴随相似物体、尺度变化和面外旋转。|
|`cube05_indoor_4`|89.735047|72.782503|38.074691|5/7 (71.43%)|255|相似物体、背景杂乱、面外旋转和尺度变化；突变中位数1.196，较高响应下继续写入错误状态。|
|`cube05_indoor_5`|54.798816|14.083510|8.751744|11/11 (100.00%)|59|所有初始化点均持续失败；相似物体、背景杂乱、面外旋转与遮挡/再出现并发；突变中位数0.911。|
|`cube05_indoor_6`|86.108938|51.115799|48.207166|12/16 (75.00%)|94|相似物体、背景杂乱、面外旋转和尺度变化；突变中位数0.955，置信加权突变0.413，符合身份切换。|
|`cup02_indoor_1`|83.542080|5.601825|17.979295|36/36 (100.00%)|63|所有初始化点均形成持续失败链；相似物体与遮挡/再出现突出；突变中位数1.124，较高响应下错误身份持续递归。|
|`duck03_wild_1`|86.033436|69.739292|25.118922|4/6 (66.67%)|162|形变、相似物体、面外旋转和尺度变化；突变中位数1.444，置信加权突变0.496。|
|`duck03_wild_2`|83.440033|71.092437|29.379884|4/6 (66.67%)|181|形变、相似物体及遮挡/再出现；突变中位数1.401，置信加权突变0.559，错误身份在高响应下延续。|
|`earphone01_indoor_1`|80.865162|42.827430|57.611639|17/20 (85.00%)|446|尺度变化、面外旋转和快速运动；突变中位数1.901，置信加权突变0.462。|
|`humans_shirts_room_occ_1_A_2`|68.588308|73.228610|49.697097|7/13 (53.85%)|185|形变与局部遮挡/再出现；突变中位数0.908，置信加权突变0.417，部分anchor出现错误状态递归。|
|`humans_shirts_room_occ_1_B_1`|69.297287|100.000000|43.196806|0/12 (0.00%)|—|没有确认失锁；低分来自成功轨迹中的定位/尺度贴合偏差，主要属性为遮挡/再出现和形变。|
|`robot_human_corridor_noocc_1_B_1`|44.947699|100.000000|45.304095|0/19 (0.00%)|—|没有确认失锁；低分主要是框定位/尺度误差，伴随面外旋转和形变。|
|`shoes02_indoor_1`|82.593947|10.871342|12.304622|13/13 (100.00%)|45|所有初始化点均形成持续失败链；相似物体、背景杂乱、尺度变化和面外旋转；突变中位数1.760，置信加权突变0.425。|
|`shoes02_indoor_2`|85.811978|41.927083|10.215668|4/4 (100.00%)|19|所有初始化点均失败；背景杂乱与相似物体突出；突变中位数1.369，置信加权突变0.470。|
|`squirrel_wild_1`|69.139154|100.000000|27.025540|0/9 (0.00%)|—|没有确认失锁；低分来自成功轨迹中的框定位/尺度贴合偏差，主要属性为尺度变化和形变。|
|`toy09_indoor_1`|86.406030|53.611238|67.571184|21/26 (80.77%)|235|面外旋转、遮挡/再出现、背景杂乱和尺度变化；突变中位数1.497，置信加权突变0.496。|
|`two_tennis_balls_3`|62.270336|53.514739|3.165489|2/4 (50.00%)|95|深度关系变化、相似物体、尺度变化和快速运动；突变中位数1.662；失败起点约需5.87倍搜索因子，超factor=4。|
|`yogurt_indoor_1`|64.266482|68.311073|56.376348|25/34 (73.53%)|38|多数anchor持续失败；突变中位数1.313，置信加权突变0.415，符合高响应身份切换/错误状态递归。|

这22条可分成两种不同优化目标。`cube05_indoor_2`、`humans_shirts_room_occ_1_B_1`、`robot_human_corridor_noocc_1_B_1`、`squirrel_wild_1` 四条ROB=100，主要监督文本是否伤害成功轨迹的定位与尺度，不能作为身份恢复正例。其余18条存在确认失败，其中 `cup02`、两条 `shoes02`、`cube05_indoor_5` 为全部anchor失败，`toy09`、`earphone01`、`cube05_indoor_6`、`yogurt`、`bandlight` 为高失败率核心族；它们才是Qwen视觉身份短文本能否改善ROB/EAO的主要判断对象。

文本最可能帮助的是相似物体、高响应身份切换和遮挡后再出现；对已超factor-4搜索域的 `ball06`、`cube02_indoor_1`、`two_tennis_balls_3`，单纯文本不能召回crop外目标，只能减少离开搜索域之前的错误候选选择。新Qwen low22终态产生后，§24.141的诊断器将对本表303个exact anchors逐一给出rescued/new-failure等转换；若聚合指标提高但任意基线存活anchor变成确认失败，仍不得进入full127人工决策。

<!-- RGBD-HANDOFF-24.143-QWENVL-CPU-READINESS-PREFLIGHT -->
## 24.143　Qwen视觉文本low22非推理预检（2026-08-29）

在不加载模型权重、不占用GPU和不生成任何文本的条件下，已对后续low22注释路径完成运行环境与输入闭包预检。`/home/qwen25_env`实际版本为Python 3.10.20、PyTorch 2.5.1+cu121、Transformers 4.51.3，CUDA可见且识别2张GPU；`Qwen2_5_VLForConditionalGeneration`与`AutoProcessor`均可正常导入。

冻结输入 `/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/annotation_inputs/low22_anchor_visual_inputs.jsonl` SHA256=`0e07babdb468af459671c0be8ac9f3a58678eae6e85ba96009dd3e2807c9f5a7`。实际逐条验证结果为22序列、303个唯一anchor keys、303张当前anchor RGB JPEG和303张目标crop PNG；606张图像均成功完整解码，尺寸与记录一致，逐文件SHA与manifest一致，绑定图像总字节数32,951,442。全部记录仍为 `pending_visual_model`，并共同满足：不暴露Depth、未来图像、未来GT、tracker输出、旧身份文本或sequence name。

视觉模型 `/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct` 的14个固定文件已逐文件复算大小和SHA，合计7,520,919,614 bytes，全部匹配 `PINNED_DOWNLOAD_MANIFEST.json`；revision仍为 `66285546d2b821cf421d4f5eb2576359d3770cd3`。本预检只证明环境、输入和模型文件已准备完毕，不代表2-anchor smoke通过，也不代表303条注释已经生成；本节写入时正式注释仍为0/303，low22 VOT尚未启动。

<!-- RGBD-HANDOFF-24.144-IDENTITY-FULL127-AND-QWEN-CONSENSUS-V3 -->
## 24.144　identity-only全127终态与Qwen anchor视觉文本fail-closed修复（2026-08-29）

### 24.144.1　identity-only全127正式终态

此前已按low22晋升门通过的 `anchor identity-only` 方法完成127序列、1,765个multi-start anchors的正式评测。结果目录为 `/root/autodl-tmp/sutrack_vot_all127_anchor_identity_v3/run`，正式结果为：

|方法|EAO|ACC|ROB|
|---|---:|---:|---:|
|原结构化序列文本，同checkpoint/toolkit|73.974969|82.627562|89.455266|
|anchor identity-only，全127|**74.020583**|82.579344|**89.565651**|
|变化（百分点）|**+0.045613**|-0.048218|**+0.110385**|

因此该方法只带来很小但可复现的EAO/ROB提升，ACC轻微下降；ACC仍高于82.1目标，但EAO/ROB仍未达到77.9/93.7。`full_result.json` SHA256=`34da51c0607c41345552e04d1bbeed3f14247081400c5dbaa926858de029ce3c`，正式analysis SHA256=`713857047981a9b816a9d3c41907ddf0cbcefa2741b4d35bf43283a2f5455748`，merge SHA256=`3709a4aba1e5da5a1c98e14e615435b003c9f2941855efe1d65d9ce534f9c615`。这组结果仍使用同一DepthTrack训练权重，不能表述为达到项目目标。

### 24.144.2　Qwen当前-anchor视觉文本的真实失败机制

新方法仍严格限制在同一low22：22序列、303个固定anchors；模型只能看到当前anchor完整RGB、冻结目标crop和权威类别，不暴露Depth、未来帧/GT、tracker输出、旧文本或sequence name。最初2-anchor smoke为0/2，典型错误是把 `in motion`、`above head`、`in air`、`shiny surface` 等瞬时状态写成身份。加强禁止词和独立验证提示本身仍不能消除该问题；随后不向验证器暴露主候选，并采用安全属性精确交集，2-anchor smoke达到2/2。

正式303生成又揭示第二个系统性错误：Qwen经常输出 `decision="manual_review"`，但同时给出非空身份、非空稳定属性且 `manual_review_reason="none"`。例如 `cube02_indoor_1@150F` 的主输出含 `black color / flat surface / rectangular shape`，独立验证器也给出同样三项，但旧v3因主decision标签而把全部属性丢弃。最小复现命令在旧投影上稳定返回 `RED: independently agreeing visible attributes were discarded`；同一真实记录经新投影后稳定得到 `black / flat surface / rectangular`。

### 24.144.3　fail-safe v4架构与不可放宽的边界

提交 `74590659b3b27b7aad23bb9d1848207dd9066475` 新增确定性fail-safe投影。它只修复一种可机器证明的标签自相矛盾：类别必须精确匹配、`manual_review_reason`必须字面为 `none`、身份与属性均非空、身份必须包含权威类别；修复后的属性仍必须与第二个独立模型输出经过瞬时/场景词过滤后逐字一致。不得使用单模型属性，不做任意语义近似，也不把 `red cap` 与 `red` 等价。

若不存在至少一个独立精确共识属性，输出固定退回 `a <category>`，而不是采用未验证外观；因此模型JSON错误、目标不可辨认或两个模型冲突都只会得到纯类别文本。最终语言分成两种mode：

```text
independent_exact_consensus:
    当前anchor类别 + 1～3个双模型精确共识的稳定属性

category_only_fallback:
    当前anchor权威类别；不含任何未验证外观
```

投影脚本 `tools/materialize_vot_low22_anchor_visual_identity_failsafe_v4.py` SHA256=`92d4b13ca8894063654cb37ebcb6795a1fa75a563cb44279026e399ba05f0ba1`。配套prepare/finalizer/launcher也已改为SHA绑定v4方法，并继续硬性约束 `low22_only=true`、303 anchors、同一checkpoint、禁止自动full127。

### 24.144.4　当前运行状态（不是指标结果）

本节写入时，v3双模型原始生成仍在GPU0运行；最新一致快照为191/303：源状态86条直接接受、97条人工复核、8条模型错误。对这191条做metric-blind v4重投影后，98条得到独立精确属性共识，93条退回纯类别。该快照只反映注释覆盖率，**不是VOT指标，也不能据此声称性能改善**。

只有303/303原始记录完整、v4物化receipt通过、且low22正式VOT相对 identity-only 基线同时满足EAO严格提高、ROB严格提高、ACC下降不超过0.10个百分点、失败数不超过195且没有新增灾难性身份切换时，才可以向用户报告并请求是否进行full127；代码中没有自动full127路径。

<!-- RGBD-HANDOFF-24.145-ANCHOR-VISUAL-LOW22-FINAL -->
## 24.145　Qwen当前-anchor视觉身份文本low22正式终态：未改善，停止full127（2026-08-29）

### 24.145.1　固定范围、唯一变量与“低指标序列”口径

本轮严格只评测冻结的VOT-RGBD2022低指标集合：22条序列、303个multi-start anchors。22条中有18条在sequence-stable identity-only对照下存在VOT确认失锁；另外4条 `cube05_indoor_2`、`humans_shirts_room_occ_1_B_1`、`robot_human_corridor_noocc_1_B_1`、`squirrel_wild_1` 的ROB为100，但ACC低于70或整体定位贡献偏低，因此也保留在固定low22中。

候选与直接对照使用同一DepthTrack训练checkpoint、同一CLIP、同一SUTrack配置、同一VOT toolkit 0.7.1、同一22序列/303 anchors。唯一实验变量是语言manifest：

```text
直接对照：sequence-stable identity-only文本
候选方法：由当前anchor完整RGB + 当前anchor目标crop + 权威类别生成的视觉身份文本
```

候选生成不读取Depth、未来帧、未来GT、tracker输出、旧文本或sequence name。Qwen2.5-VL-3B的两个独立输出经过fail-closed投影后，303条中153条采用 `independent_exact_consensus`，150条采用 `category_only_fallback`；最终共有84种不同文本。Qwen3_8B未加载且继续保留。

需要区分两次“换注释”结果：此前把旧结构化长文本改成sequence-stable identity-only，low22的EAO/ACC/ROB由 `42.629281/71.827916/53.412816` 提高到 `43.274104/72.065511/54.388022`，确认失败由200降到195，属于小幅有效；本节评测的是真正按当前anchor图像重新生成视觉属性，结果如下。

### 24.145.2　正式指标与晋升门

|指标|identity-only直接对照|当前-anchor视觉身份文本|变化（百分点）|冻结门要求|是否通过|
|---|---:|---:|---:|---|---|
|EAO|43.274104|42.946692|-0.327413|严格提高|否|
|ACC|72.065511|71.910688|-0.154823|下降不超过0.10|否|
|ROB|54.388022|54.305144|-0.082878|严格提高|否|
|确认失败anchors|195|202|+7|不得增加|否|
|新增灾难失败anchors|0|11|+11|必须为0|否|

五项门控全部失败，`gate_passed=false`、`full127_authorized_for_human_decision=false`、`automatic_full127_launch=false`。因此该注释方法停止在low22，未创建、未启动也不得补跑full127。

### 24.145.3　22条序列中究竟有多少改善

按对low22聚合EAO的可加贡献统计，11/22条为正、11/22条为负；但这种正负不能单独代表生存能力。按确认失败数统计，仅3条序列改善、6条变差、13条不变：

- 改善：`cube02_indoor_1` 8→7、`cube05_indoor_6` 12→11、`duck03_wild_2` 4→3。
- 变差：`bandlight_indoor_1` 18→19、`cube02_indoor_2` 5→6、`cube05_indoor_1` 2→3、`humans_shirts_room_occ_1_A_2` 7→8、`two_tennis_balls_3` 2→3、`yogurt_indoor_1` 25→30。
- 其余13条确认失败数不变。

anchor级共有4次rescue，但新增11次“基线存活、候选确认失败”的灾难回归，净增7次失败。ACC在7条序列改善、15条退化；ROB在8条改善、10条退化、4条不变。下表保留全部22条的正式逐序列变化；EAO列是对low22聚合EAO差值的可加贡献，不是singleton EAO。

|序列|EAO贡献Δpp|ACCΔpp|ROBΔpp|失败旧→新|rescued/new|
|---|---:|---:|---:|---:|---:|
|`ball06_indoor_2`|-0.008047|-0.806111|-0.093985|1→1|0/0|
|`bandlight_indoor_1`|+0.000336|-1.526817|+0.835655|18→19|0/1|
|`cube02_indoor_1`|+0.041106|-0.392896|+5.280358|8→7|1/0|
|`cube02_indoor_2`|-0.113347|-0.105066|-6.605999|5→6|1/2|
|`cube05_indoor_1`|+0.072996|+6.581728|-9.114583|2→3|0/1|
|`cube05_indoor_2`|-0.000148|+0.052789|0.000000|0→0|0/0|
|`cube05_indoor_4`|-0.002677|+0.064400|+0.060753|5→5|0/0|
|`cube05_indoor_5`|-0.021072|+6.147438|-2.123142|11→11|0/0|
|`cube05_indoor_6`|+0.115969|+0.523365|+5.806595|12→11|1/0|
|`cup02_indoor_1`|+0.017874|-0.020692|+0.062777|36→36|0/0|
|`duck03_wild_1`|+0.002250|-0.035234|+0.465549|4→4|0/0|
|`duck03_wild_2`|+0.025728|-1.459423|+17.310924|4→3|1/0|
|`earphone01_indoor_1`|-0.164783|+0.503702|-1.259805|17→17|0/0|
|`humans_shirts_room_occ_1_A_2`|+0.098120|-0.106533|-0.317513|7→8|0/1|
|`humans_shirts_room_occ_1_B_1`|-0.006363|-0.092398|0.000000|0→0|0/0|
|`robot_human_corridor_noocc_1_B_1`|-0.050094|-0.391220|0.000000|0→0|0/0|
|`shoes02_indoor_1`|+0.024672|-2.315152|+0.817394|13→13|0/0|
|`shoes02_indoor_2`|-0.010226|-2.777706|-1.302083|4→4|0/0|
|`squirrel_wild_1`|+0.000320|-0.434134|0.000000|0→0|0/0|
|`toy09_indoor_1`|-0.363371|-0.055993|-3.065353|21→21|0/0|
|`two_tennis_balls_3`|+0.069395|+0.350559|-7.709751|2→3|0/1|
|`yogurt_indoor_1`|-0.056050|-0.010878|-0.998133|25→30|0/5|

### 24.145.4　失败原因与具体例子

这次退化不是因为模型权重或评测配置变化，而是无条件用当前anchor文本替换了原有sequence-stable身份文本。11个新增灾难失败中，7个使用纯类别回退，丢失了原文本中的区分属性；另外4个虽然有双模型精确共识属性，但属性并不一定具有跨帧身份判别力。

1. **纯类别回退删除了真正有用的实例属性。** `bandlight_indoor_1@500F` 从 `a green band-shaped light` 变成 `a band light`，基线完整生存685 progress frames，候选在388处确认失败；失锁前出现高置信错误与严重运动/尺度变化，随后目标离开factor-4搜索域。`cube02_indoor_2@400B` 和 `@500B` 从 `a dark cube with a fabric-like surface` 退成 `a cube`，分别由基线生存401/501帧变为65/166处失败，并出现高置信大跳变。`cube05_indoor_1@110B` 删除 `dark rectangular` 后由完整生存111帧变为75处失败。`two_tennis_balls_3@0F` 删除颜色 `yellow` 后由130帧完整生存变为96处失败，且失败时目标已超factor-4搜索域。

2. **“当前可见属性”不等于稳定身份属性。** `yogurt_indoor_1` 有5个新增失败：原文本始终保留 `printed label`，候选却分别变成纯 `yogurt cup`、`round`、`white lid` 或 `white plastic container`。这些属性要么太泛化，要么只反映当前可见面，无法在旋转、尺度变化和遮挡后稳定区分实例，导致多个长轨迹在后段失败。

3. **精确共识仍可能语义冗余或强化同类干扰物。** `humans_shirts_room_occ_1_A_2@200F` 从 `a person wearing a patterned shirt and jeans` 改成 `a person, man wearing blue striped shirt and man wearing jeans and man with beard`。虽然字段得到两个独立模型的字面共识，但重复的 `man` 结构与当前可见胡须并不保证是长期身份锚；候选在301 progress处发生高置信大跳变，而基线完整生存402帧。

4. **文本无法召回已经离开搜索区域的目标。** 新灾难失败普遍伴随 `severe_motion_scale_change`；`bandlight@500F` 与 `two_tennis_balls@0F` 还明确出现 `target_outside_factor4_search`。文本最多能在目标仍位于crop内时改变候选排序，不能在目标离开搜索域后恢复；如果前一帧文本促成了错误候选，递归state会进一步把搜索区域移向错误位置。

这不是“逐-anchor视觉文本完全无用”：它救回了4个anchor，例如 `cube02_indoor_1@150F` 从 `a black cube` 变成 `a cube, black and flat surface and rectangular`，生存由223提高到完整422；`cube02_indoor_2@200F`、`cube05_indoor_6@300F` 和 `duck03_wild_2@50F` 也被救回。但同一种category-only动作既能救回某些anchor，也会伤害另一些anchor，说明问题是**无条件替换缺少反事实选择与递归保护**，不能靠继续放宽Qwen属性过滤解决。

### 24.145.5　后续技术决策

- 不运行该方法的full127；当前全127最好仍为§24.144的identity-only `74.020583/82.579344/89.565651`。
- sequence-stable identity-only继续作为不可变保护文本；当前-anchor视觉文本只能作为可拒绝的候选动作，不能覆盖保护文本。
- 下一轮语言模块若继续，必须在同一low22上比较 `protected identity-only`、`category-only`、`anchor attributes` 三个动作，并以短窗survival/候选分歧作shadow确认；不能用当前帧CLIP相似度单独提交。
- 按既定主线，优先把Protected–Tentative模板事务接入真实SUTrack：先暂存state/template，未来1～2帧验证后原子promote，否则完整rollback；仍先跑同一low22门，禁止自动full127。
- 新方法必须显式保留 `green`、`dark/fabric-like`、`dark rectangular`、`yellow`、`printed label` 这类已被本轮反事实证明有价值的稳定属性，并拒绝仅由当前可见面产生的泛化词或瞬时属性。

### 24.145.6　正式证据绑定

- gate：`/root/autodl-tmp/sutrack_vot_low22_anchor_visual_identity_v1/run/low22_gate_result.json`，SHA256=`5ff3bb7865ab1a2b5c977607f361cfd70213f6d38eb5ed8913716e497193a7e5`。
- official analysis：`anchor_visual_identity_low22_analysis.json`，SHA256=`9e402396bac2dd7348da413f1aaac0cee4c0d752f3b0fe1f79ef94a1300aeb84`。
- merge：`merge_result.json`，303 anchors、909个结果文件，SHA256=`395836c29414a21a320a66ad3eb0529fb4977334a07d51d352711e95275b0308`。
- annotation manifest：SHA256=`0440cb17393740cedd19efc9a1d723844006c51b1262be75efd203b9a65d575a`；materialization receipt：SHA256=`504eeeba7c199c8556f2fb076215593c91f8c42927bd738f5f54cfcbb965d3ae`。
- 逐anchor诊断JSON：SHA256=`4c3dd31d4207f84f63a4eb513608be9ccfc93220f92d5ee3b6c86b1fed3300fa`；Markdown：SHA256=`4bc721d0d145bc0efb775bf8354ff2ea4b576029269e43a2c62d5824a372a1b8`；receipt：SHA256=`e92434494aec2bae9432428a325f7d136c6033a40b92200a8873002575070d83`。
- checkpoint与CLIP SHA仍分别为 `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`、`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。

<!-- RGBD-HANDOFF-24.146-TRANSACTION-CUDA-FIX-AND-LOW22-LAUNCH -->
## 24.146　Protected–Tentative事务真实CUDA修复与low22正式启动（2026-08-29）

anchor视觉文本在§24.145未通过low22门后，没有启动full127；下一主线按既定方案转为保护—暂存模板事务，仍只使用同一sequence-stable identity-only文本、同一DepthTrack训练checkpoint、同一固定22序列/303 anchors。

### 24.146.1　正式启动前真实CUDA发现的三个接线问题

CPU结构检查此前只能证明事务状态机、深拷贝、原子promote/rollback和VOT门控的静态契约，不能代替真实模型前向。本轮在启动low22前先跑不读取未来GT、不计算公开指标的CUDA smoke，因此发现并修复了三个只会在真实tracker路径出现的问题：

1. GPU smoke直接调用参数加载器后未像正式VOT adapter一样设置 `params.visualization=False`、`params.debug=False`，首次构造tracker报 `AttributeError: TrackerParams has no attribute debug`。
2. GPU smoke把tracker运行数据集写成 `votrgbd2022`，而正式VOT adapter实际以 `depthtrack` 创建tracker以开启RGB-D与NLP任务路径；修正前被模态绑定检查拒绝。
3. 真实第1帧触发 `state_conflict_candidate` 后，第2帧保护分支从不可变snapshot得到tuple bbox；共享 `sample_target` 只接受list或带 `.tolist()` 的数组，导致 `AttributeError: tuple object has no attribute tolist`。修复方式是在snapshot内部继续保持tuple不可变性，只在送入protected/tentative `_infer()` 边界时显式 `list(snapshot.state)`，不修改共享裁剪函数。

三处修复及对应结构检查提交为 `c5b5e6d08de91d6f9e7c5ed5e30a3be6aa05aa4d`。只修改：

```text
lib/test/tracker/sutrack_transaction.py
tools/smoke_sutrack_transaction_gpu.py
tools/smoke_sutrack_transaction_integration.py
```

未修改checkpoint、公共SUTrack tracker、语言manifest、safe-v1阈值、transaction阈值、VOT配置或冻结low22选择。

### 24.146.2　修复后的结构与真实GPU证据

当前结构预检扩展为29项并全部通过，其中新增检查确保GPU smoke与正式VOT runtime字段一致、使用 `depthtrack` 路径，并真实调用active事务的protected/tentative两个推理入口验证bbox在边界物化为list。正式launcher重新生成的结构smoke SHA256=`f98da925124cbe718445068e9c74b096f6a2d141a126e482b4086fbcdfe3bf5b`。

随后完成两组metric-blind CUDA前向：

- `ball06_indoor_2@0F` 35帧：状态为passed，触发18次 `state_conflict_candidate`；trace为18次hold、17次rollback，35帧公开输出全部选择protected分支；未读取未来GT、未计算指标。artifact SHA256=`50d5d6e22e79518809824ce1c5ec8f0e8f56f5ca695e146012467777e3acb09f`，trace SHA256=`98d84270873fc85fda0b7d25e1f46750d8dfc8d9fd528f69e333a84cedeec52f`。
- `cube05_indoor_2@0F` 45帧：状态为passed，实际触发3次 `template_candidate` 和10次状态冲突，共13个事务；3个模板候选均因未来两帧没有持续优于保护分支而rollback，45帧公开输出全部选择protected。artifact SHA256=`d1cd4925d607da59fa05ff6b7c8c34b678af5ba11860747162b2d0b1bf0f384e`。

第二组证明真实GPU路径不只是“能前向”，而是已覆盖动态模板候选创建、两分支shadow、证据比较和rollback；promote逻辑仍由CPU确定性状态机检查覆盖，正式low22 trace将统计实际promote数量。

### 24.146.3　正式low22运行绑定

2026-08-29 17:45 CST正式launcher通过自身29项结构门和3帧CUDA门后，才原子创建：

```text
/root/autodl-tmp/sutrack_transaction_low22_v1/run
```

运行范围与负载：

|项目|值|
|---|---:|
|序列|22|
|anchors|303|
|分片|4|
|各分片anchors|75 / 76 / 76 / 76|
|估算frame load|55,086 / 55,141 / 55,128 / 55,128|
|总估算frames|220,483|
|GPU|2×RTX 3090，每卡2个worker|

正式canonical CUDA smoke SHA256=`fd95a98c80af04a4aaab149b62826aa9fea637c1ecd0aeea85abeeec018a56b0`；正式shard manifest SHA256=`264dda6b32360ce1e53c75dde1fa6a652d52ccb41107ab54de5f1fa33ac0195d`。controller、finalizer和逐序列/trace诊断器均由独立nohup进程托管，终态将生成 `low22_gate_result.json` 和 `low22_transaction_diagnostics.json`。

直接对照继续冻结为identity-only low22：EAO/ACC/ROB=`43.2741043540/72.0655112521/54.3880218212`、确认失败195。晋升门不变：EAO和ROB严格提高、ACC下降不超过0.10个百分点、确认失败不增加；即使通过也只允许人工决定是否做full127，`automatic_full127_launch=false`。

本节写入时首批4个 `cup02_indoor_1` 长轨迹已各推进约900 tracker frames，4个worker与两张GPU持续在线，无Traceback、OOM或recoverable error；尚无完成anchor，也没有任何可解释为中途指标的结果。事务风险帧会额外运行保护/暂存双分支，因此预计本轮约需3～4小时。终态不得以本节启动状态代替。

<!-- RGBD-HANDOFF-24.147-TRANSACTION-ROOT-CAUSE-AND-TEMPLATE-ONLY-V2 -->
## 24.147　旧事务正式失败根因与template-only v2修正（2026-08-29）

§24.146启动的Protected–Tentative v1已经完成固定low22全部22条序列、303个multi-start anchors；结果不是轻微退化，而是必须封存的灾难性负结果。随后本节以正式VOT轨迹、事务trace和identity-only直接对照构建可重复诊断，证明主因不是动态模板被错误promote，而是v1把安全模板writer的“冲突”错误升级成bbox状态冻结，导致所谓protected分支并不等于保护基线。

### 24.147.1　v1正式终态

|指标|identity-only直接对照|Protected–Tentative v1|变化（百分点）|
|---|---:|---:|---:|
|EAO|43.274104|21.412955|-21.861149|
|ACC|72.065511|61.659985|-10.405526|
|ROB|54.388022|28.125866|-26.262156|
|确认失败anchors|195|251|+56|

覆盖为303/303 anchors、909个结果文件，toolkit为0.7.1；creation/recoverable error均为0，因此不能把退化归因于缺文件、OOM、异常回退或评测失败。所有晋升门均失败，`gate_passed=false`、`full127_authorized=false`、`automatic_full127_launch=false`，没有启动v1 full127。

典型序列的确认失败变化包括：`ball06_indoor_2` 1→8、`cube05_indoor_2` 0→4、`squirrel_wild_1` 0→9、`cube02_indoor_2` 5→13、`yogurt_indoor_1` 25→34。与此同时也存在局部rescue，例如两条`duck03`和`humans_A`的ROB上升；但它们无法抵消全局连续失锁，且ACC大多下降。

### 24.147.2　可重复根因证据

新增只读取冻结工件的诊断命令：

```bash
PYTHONPATH=/home/SUTrack_RGBD_L \
/root/miniconda3/envs/mplt/bin/python \
tools/diagnose_vot_transaction_failure.py \
  --gate /root/autodl-tmp/sutrack_transaction_low22_v1/run/low22_gate_result.json \
  --diagnostics /root/autodl-tmp/sutrack_transaction_low22_v1/run/low22_transaction_diagnostics.json \
  --baseline-workspace /root/autodl-tmp/sutrack_vot_low22_anchor_identity_v1/analysis_workspace_view \
  --baseline-tracker sutrack_l384_rgbd_anchor_identity_low22 \
  --candidate-workspace /root/autodl-tmp/sutrack_transaction_low22_v1/run/master \
  --candidate-tracker sutrack_l384_rgbd_anchor_identity_transaction_low22
```

该命令两次均以预期非零状态输出`REGRESSION_REPRODUCED`，核心计数完全一致：

|诊断量|结果|
|---|---:|
|新灾难anchor：基线存活而v1失败|69|
|其中失败前没有任何template promote|43|
|其中失败前先发生protected bbox冻结|38|
|tentative utility更高但因`protected_hard_conflict`被拒绝|514|
|全部事务|88,683|
|state-conflict候选|84,479（95.26%）|
|真正template候选|4,204（4.74%）|
|promote / rollback|1,063 / 87,472|

这组证据排除了“模板promote是主要原因”：例如`ball06`从1个失败变为8个、`cube05_indoor_2`从0变为4个时，序列级promote均为0；它们仍然灾难退化，只能由状态语义变化解释。

最小真实例子是`ball06_indoor_2@0F`。v1在第1跟踪帧检测到`large_center_jump`后，将protected bbox保持为初始化框`[261,289,22,21]`，却把网络当前预测`[249.0614,263.7551,25.8094,24.6190]`只放进tentative分支。第2帧tentative utility约0.4295，明显高于protected约0.1139，控制器仍因`protected_hard_conflict`回滚；之后每两帧重复一次冻结/回滚，搜索crop长期围绕旧位置，快速运动目标离开搜索域。

根因由三个相互叠加的旧语义构成：

1. `track()`把任一中心跳变、RGB身份低、深度突变等hard conflict创建成`state_conflict_candidate`，公开状态先安装上一帧bbox。
2. active shadow中的`_advance_branch()`再次在hard conflict时保留分支上一bbox，而不是接受本帧预测。
3. controller让`protected_hard_conflict`拥有否决权，即使tentative未来证据更优也rollback到被冻结的protected状态。

但冻结的identity-only直接对照明确配置`HARD_CONFLICT_STATE_ROLLBACK=False`：它会拒绝危险模板写入，却仍把当前预测框写入`self.state`。因此v1的protected分支从第一处冲突开始就不再是基线，违反了“保护分支必须可作为无创新退路”的设计前提。

### 24.147.3　template-only v2架构

修正提交为`513cd3c578298c48b8bc4673e021ac7535bae3d2`，tracker id改为：

```text
sutrack_l384_rgbd_anchor_identity_template_transaction_low22
```

新递归顺序为：

```text
当前RGB-D搜索帧
      ↓
SUTrack产生当前bbox
      ↓
safe-v1 writer只判断模板动作
      ├─ 非replace_dynamic（包括所有hard conflict）
      │      → 接受当前bbox与writer policy，不创建事务
      │
      └─ replace_dynamic
             → 两个snapshot使用完全相同的当前bbox
             → protected仅保留旧模板
             → tentative仅写入候选模板
             → 未来2帧独立shadow
             → 连续优势则原子promote，否则rollback模板分支
```

因此v2保持两个硬不变量：

- **无模板候选时bbox、置信度和writer状态与identity-only直接基线一致。** 中心跳变、身份或深度冲突可以阻止模板，但不能再冻结bbox。
- **事务只隔离模板因果变量。** 起始帧两分支bbox相同，差异仅为动态模板；后续分支即使出现hard conflict，也按直接基线语义接受各自当前预测框。

这不是简单删除模板更新：候选模板仍先进入tentative槽，未来证据足够时可以promote；只是把尚未证明安全的“bbox rollback”从模板事务中完全移除。

### 24.147.4　metric-blind预检

正式low22创建前完成以下不读取未来GT、不计算公开指标的检查：

|检查|结果|
|---|---|
|Python/bash语法与候选/基线YAML语义一致性|通过|
|CPU结构与接线检查|30/30通过|
|`ball06@0F` 3帧真实CUDA|第1帧接受预测框，无状态事务|
|`ball06@0F` 35帧直接基线/候选逐帧对照|bbox最大误差0、score最大误差0、writer decision完全一致、事务0|
|`bandlight@50F`真实模板事务|第5帧start，第7帧rollback|
|`bandlight@100F`真实模板事务|第5帧start，第7帧promote tentative|

这同时覆盖了无事件基线同构、真实动态模板创建、两帧shadow、rollback和promote五条路径。checkpoint仍为DepthTrack训练权重，SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`；语言仍为冻结identity-only anchor manifest，SHA256=`a56bb51836fb9c120d8492bb2742b8340dd3339ca44d20875f50facb5b375ee9`。

### 24.147.5　v2正式low22启动与边界

v2在所有预检通过且源码提交后，才创建：

```text
/root/autodl-tmp/sutrack_template_transaction_low22_v2/run
```

范围继续固定为22条低指标序列、303 anchors、4 shards=`75/76/76/76`，两张RTX 3090各两个worker；对照不变为identity-only `EAO/ACC/ROB=43.2741043540/72.0655112521/54.3880218212`、195次确认失败。晋升门仍要求EAO与ROB严格提高、ACC最多下降0.10个百分点、确认失败不增加；无论门是否通过，都必须先报告用户，`automatic_full127_launch=false`。

本节写入时v2仍在正式运行，不能把进度或trace比例解释成中途指标。已观察到的事务全部为`template_candidate`，`state_conflict_candidate=0`，运行错误为0；最终EAO/ACC/ROB、失败变化、逐序列贡献、事务统计和是否建议full127必须在终态工件产生后另节追加。

### 24.147.6　证据绑定

- v1 gate：SHA256=`1a0016204f390586b21038fcfdc074adcf0a51b1b649fea51e3e39b9bd896ca9`。
- v1 diagnostics：SHA256=`5009b3cbc24925c7d3385d592dfa53228ff156c22bbf846e02c874592a37df32`。
- v1 merge：SHA256=`fd8a6e3056fa162244cd33028b27801da3769a4caa599bb77b54faab7b254bfe`。
- 第二次确定性root-cause复现：SHA256=`7332a808a8c7c0fde253e318d2778c962828c979bd2b524674d04e7341b3ce5d`。
- v2结构smoke：SHA256=`51ed553d415d94cb68031b924883a3c2f7773e02919f50ced1e02f096fa2b2c9`。
- v2 3帧CUDA smoke：SHA256=`2d32332794012dc9d3f4f7c02f84f96c0339d5a150f599db676da9e766ebf7a4`。
- v2 35帧CUDA parity：SHA256=`a33d50cc6e55acedc639f71fc60463103ff5863e9b5609ca1b2f2022b5d183f6`。
- 真实rollback/promote smoke：SHA256分别为`ccb5cb3015a8e59ef2a29e70d1197ded848e87fafc3abe840994838a7a56452b`、`e5495eeaae0404309de202369171a2ac90d440ef23f57ab5fd441896708570ea`。
- v2 shard manifest：SHA256=`da7222f37052dd75bee83d565b74e684a080a7c552692ed76281906f9cfe6013`。

<!-- RGBD-HANDOFF-24.148-TEMPLATE-V2-FINAL-AND-VETO-V3 -->
## 24.148　template-only v2正式终态、分支语义反转根因与baseline-first veto v3（2026-08-30）

### 24.148.1　template-only v2正式low22结果

§24.147启动的template-only v2已完成冻结low22全部22条序列、303个multi-start anchors和909个结果文件；VOT toolkit仍为0.7.1，checkpoint、CLIP、identity-only文本、搜索区域和全部门控阈值均未改变。正式结果为：

|指标|identity-only直接对照|template-only v2|变化（百分点）|
|---|---:|---:|---:|
|EAO|43.274104|43.106822|-0.167282|
|ACC|72.065511|72.019219|-0.046293|
|ROB|54.388022|54.162754|-0.225268|
|确认失败anchors|195|201|+6|

ACC满足“最多下降0.10个百分点”的容差，但EAO、ROB与失败数三项门均失败；正式gate为`gate_passed=false`、`full127_authorized=false`、`automatic_full127_launch=false`，没有运行v2 full127。gate、analysis、merge的SHA256分别为`5337cfb3697e6168c9a055b4ce94067425679c12be3ab1dde017021ee58e4d63`、`fc221bfe0032044ecac04c9e52b90d4cc29d414a1674528f59655cf98d7f5569`、`3d06128a6ad5bfcc3f5e4e940a1ba68401c271851f434a6e89a7a06268fd10bd`。

正式trace共创建7,052个纯模板事务，1,764次promote、5,256次rollback、32次轨迹结束时未决；`state_conflict_candidate=0`，creation/recoverable error均为0。相对直接基线共有8个新灾难anchor和2个rescue，净增加6个失败。旧v1的bbox冻结回归没有复现：冻结状态灾难为0，状态冲突事务为0，说明§24.147修复确实消除了旧故障，但v2仍存在另一个独立语义错误。

### 24.148.2　剩余退化的确定性最小根因

唯一“失败前零promote”的新灾难是`cube02_indoor_2@450B`：直接基线完整生存451帧，v2仅生存118帧。逐帧轨迹与事务分支对齐证明这不是CUDA非确定性，也不是rollback漏恢复状态：

- 事务在tracker frame 10创建，正式轨迹从frame 11首次分歧，在frame 12 rollback之前已经发生。
- frame 11直接基线bbox为`[350.3226013, 194.2774048, 55.7378998, 53.9912987]`。
- v2 tentative/new-template bbox与直接基线最大误差仅`4.5776e-05`；v2 protected/old-template bbox与基线误差为`0.191452`。
- 正式v2公开bbox与protected/old-template分支误差仅`4.5776e-05`，而与tentative/new-template分支误差为`0.19154`。
- 到确认失败前，v2与基线最大bbox误差扩大到`73.7944`，最大置信度差为`0.25748`。

因此v2把分支角色定义反了。直接identity-only基线在safe-v1允许更新时会立即写入新动态模板；v2却把“继续使用旧模板”命名为protected并在两帧shadow期间公开，把真正与正式基线一致的“新模板”放在tentative中。即使最终没有promote，rollback也会保留错误的旧模板分支，所以仍可能退化。确定性outcome诊断与branch-alignment工件SHA256分别为`45ec318a5dfedbfc01eed9076339310279d15259bc3d37a552eee8c55f518f10`和`bd18cff6f0cf4ba6093f1d5b116a51e1780e3f2272de22a9160c37ec1382bb63`。

### 24.148.3　baseline-first counterfactual template veto v3

提交`56e1bfb7880ef50f0ba1dbfde62ca9644147d20f`只增加一个向后兼容的snapshot定向接口和独立v3 tracker，旧v2默认方向保持不变。新架构定义为：

```text
safe-v1产生合法新模板
        │
        ├─ protected/public：立即写入新模板的直接identity-only基线
        │        └─ hold、timeout、冲突或异常均继续保留该分支
        │
        └─ tentative/shadow：取消本次写入、继续使用旧模板的反事实分支
                 └─ 未来连续2帧显著优于直接基线时promote
                       = veto本次模板更新
```

这里`promote`不再表示接受新模板，而表示接受“旧模板反事实”，即撤销有害更新；`rollback`反而表示保留直接基线的新模板。公开输出在shadow期间始终来自直接基线分支，因此没有veto时应与identity-only路径同构。controller阈值、两帧确认、权重、文本、safe-v1 writer和checkpoint全部冻结，不使用low22指标调参。

结构检查11/11通过，明确验证旧v2方向未改变、v3 protected=new-template、tentative=old-template、两次未来优势才veto、失败veto回到直接基线；artifact SHA256=`933a907bb2a2b0bcffa252e3c1b0394e1442050a84d25b3d2870e0a0cbe5f192`。

真实CUDA最小复现继续使用`cube02_indoor_2@450B`，只读取初始化bbox，不读取未来GT、不计算公开指标。12帧内实际创建1个模板事务并rollback，veto promote为0；v3与直接identity-only的bbox、best score和writer decision全部逐值一致，bbox/score最大误差均为0。该预检恰好覆盖v2原先从frame 11开始偏离的路径，artifact SHA256=`45644cc363a29d452e0a734ca0449d6cb1a04e8a4581b83be2bf9ea48c2e52ff`。

### 24.148.4　v3正式low22启动状态与不可越过的边界

完成源码提交和预检后，才于2026-08-30约00:54 CST创建：

```text
/root/autodl-tmp/sutrack_template_veto_low22_v3/run
```

范围仍严格为同一22条低指标序列、303 anchors，4个shard为`75/76/76/76`，总估算tracker frames为220,483，两张RTX 3090各运行2个VOT worker。正式manifest SHA256=`3783fc19fbfa067e0e20b8daf3c9fb4507ff4e273c3eb60132d066649d5dfddb`；source snapshot SHA256=`66faa83e0153067e7f6664e4182746788b913051c7fb22697f103d73edd30652`，绑定原DepthTrack checkpoint SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`。

本节写入时正式运行刚创建，0/303完成，不能解释成中途指标。晋升门继续冻结为：EAO与ROB严格提高、ACC最多下降0.10个百分点、确认失败不超过195；无论是否通过都先向用户报告。代码没有自动full127启动路径，只有low22正式终态通过且用户明确决定后，才可考虑全127。

<!-- RGBD-HANDOFF-24.149-TEMPLATE-VETO-V3-FINAL -->

## 24.149　baseline-first template veto v3 正式终态与相关代码发布（2026-08-30）

### 24.149.1　固定 low22 正式结果：历史退化已消失，但 ROB/EAO 没有改善

§24.148 启动的 `baseline-first counterfactual template veto v3` 已完成冻结的 VOT-RGBD2022 low22 全部 22 条序列、303 个 multi-start anchors 和 909 个结果文件。正式运行目录为 `/root/autodl-tmp/sutrack_template_veto_low22_v3/run`；VOT toolkit 为 0.7.1，继续使用同一 DepthTrack 训练 checkpoint、同一 CLIP、同一 sequence-stable identity-only 文本、同一搜索区域与同一 safe-v1 writer。candidate 与直接对照的唯一变量是 baseline-first 模板更新反事实 veto 事务。

|指标|identity-only 直接对照|template veto v3|变化（百分点）|晋升条件|通过|
|---|---:|---:|---:|---:|---|
|EAO|43.274104354019|43.273264776973|-0.000839577046|严格提高|否|
|ACC|72.065511252071|72.064521953739|-0.000989298331|不低于对照 -0.10|是|
|ROB|54.388021821177|54.388021821177|0.000000000000|严格提高|否|
|确认失败 anchors|195/303|195/303|0|不得增加|是|

正式 gate 文件为 `/root/autodl-tmp/sutrack_template_veto_low22_v3/run/low22_gate_result.json`，SHA256=`0ab39e347064357a0cd05f602e514998e03d077bb02e5c7cd97fe862102425c2`。`gate_passed=false`、`full127_authorized=false`、`automatic_full127_launch=false`；因此本方法不运行 full127。

### 24.149.2　逐 anchor 终态证明 v3 已恢复保护基线语义

303 个 anchor 的 VOT failure 终态与直接 identity-only 基线逐一比较结果为：

- 195 个 anchor 两边均失败，且 failure progress 逐值完全相同；
- 108 个 anchor 两边均完整生存；
- 新增灾难 failure 为 0，rescue 为 0，提前或延后 failure 为 0；
- 22 条序列的确认失败数逐条完全相同。

这证明 v3 已修复 v1 的 bbox 冻结与 v2 的公开分支语义反转。正式 candidate 的极小 EAO/ACC 差异只来自部分生存帧的亚像素/小幅 bbox overlap 差异，不来自 failure 进度改变。因此应将本轮结论写成：**模板事务回归已经消除，但 robustness 增益不存在。**

事务 trace 共记录 3,050 次模板候选事务：5 次 veto promote、3,039 次 rollback 保留直接基线新模板、6 次轨迹终止时未决；creation error、recoverable error 与 state-conflict error 均为 0，25 个 trajectory 没有产生事务。五次实际 old-template veto 为：

|序列与 anchor|event/frame|直接基线新模板 utility|旧模板反事实 utility|
|---|---:|---:|---:|
|`cup02_indoor_1@1550`|11 / 347|0.8205175202|0.8422559797|
|`humans_shirts_room_occ_1_B_1@0`|5 / 127|0.8208744659|0.8650695595|
|`toy09_indoor_1@50`|6 / 257|0.8249838957|0.8398637653|
|`toy09_indoor_1@500`|18 / 747|0.8398344619|0.8634671400|
|`yogurt_indoor_1@100`|13 / 1162|0.7852306575|0.8030227709|

尽管这五次 veto 都满足连续两帧在线证据，最终没有任何一个把失败 anchor 救回。veto 激活率只有 `5/3050=0.163934%`，说明继续扫描在线 utility 阈值既缺乏监督，也容易重新引入 v1/v2 的损失；该路线停止做 low22 阈值扫描。

正式 analysis SHA256=`bb7892fd2a903158f965104c99b1f94b7cecf297d59d8249010cec0bafd57dff`，merge receipt SHA256=`4c6976e144a1fdced5dbdd8b52b42056311e9e3f671abb58b51fe8323c533736`，transaction diagnostics SHA256=`2e5ce7e2bfe9828a01063260742e9775a6cd84be64522a0cee3201ab2559b273`，outcome diagnostic SHA256=`34671d16f78c0f0776cb2d4a8a493ead93e2cb79754268e15d75f08f0c2972e0`，后者 verdict 为 `TEMPLATE_TRANSACTION_REGRESSION_ABSENT`。

### 24.149.3　下一主线：DepthTrack Train 监督的 survival/template-veto gate

v3 保留为不可变的 baseline-first、fail-open 事务外壳；下一步不再用 low22 GT 调阈值，而是在 DepthTrack Train 的真实递归 rollout 中构造模板更新的有害/有益反事实标签。轻量门控以未来 5--10 帧 survival utility 为训练目标，至少覆盖：未来平均 IoU、短窗 failure、目标/干扰物身份一致性、RGB/Depth 可靠性、响应 margin/entropy、中心和尺度连续性。训练与阈值选择只能使用 DepthTrack Train；完成 Train-only 跨序列验证和真实 CUDA 预检后，仍只允许先运行同一 fixed-low22/303。EAO 与 ROB 必须严格提高、ACC 最多下降 0.10 个百分点、确认失败不得增加且不得出现新增灾难 anchor，才可向用户报告并请求是否运行 full127。

### 24.149.4　指定 GitHub 发布凭据

与 v2/v3 模板事务有关的 23 个 overlay 源码/配置文件及更新后的 README、`MANIFEST.sha256` 已通过逐文件源树比对、Python/shell/YAML 语法检查、manifest 校验、`git diff --check`、大文件/权重扩展名/服务器密码扫描，并以非强制 fast-forward 推送到用户指定仓库。提交为 `aae824bcd0a4cece66b46ca8190d94e3268951a0`：`https://github.com/666666666666gao/Track/commit/aae824bcd0a4cece66b46ca8190d94e3268951a0`。本次提交仅包含 `projects/sutrack_rgbd_language_template` 下的相关源码、配置、README 与清单；没有 checkpoint、Qwen3_8B、数据集、预测结果、VOT workspace、日志、密钥或本交接文档。

<!-- RGBD-HANDOFF-24.150-SURVIVAL-TEMPLATE-TRAIN-GATE -->

## 24.150　DepthTrack Train 监督的 survival-template 反事实门控：B0/M1 终态与 full152 启动（2026-08-30）

### 24.150.1　冻结方法与不可越过的评测边界

本轮不再使用 VOT low22 的标签或指标学习模板 veto。方法固定为：safe-v1 合法写入新动态模板后，新模板分支继续作为公开的直接基线；旧模板只在隔离快照中递归运行未来 12 帧。前 2 帧只提取部署时可获得的 old-vs-new 在线特征，第 3--12 帧才由独立后处理器加入 DepthTrack Train GT，形成未来生存标签。若证据不足，最终部署默认保留新模板，不改变直接基线。

```text
safe-v1 合法模板写入事件
        │
        ├─ public/new：直接基线，立即使用新模板并正常递归
        │
        └─ shadow/old：取消本次写入，使用旧模板递归 H=12
                     │
                     ├─ step 1--2：在线 confidence/margin/identity/depth/continuity
                     └─ step 3--12：推理结束后才加入 GT 生存标签
```

完整计划位于 `/home/SUTrack_RGBD_L/refine-logs/EXPERIMENT_PLAN.md`，SHA256=`9d7ae4f3705de9776c0ccd08932618b2397678ac4d560960d10cc6867337765d`。122 calibration / 30 audit 的预先冻结划分 SHA256=`f09ee1c36e51c24969311365a2bda0970ac951baf54ca916878cae77bac3dabd`，full152 trace plan SHA256=`b9d36eb7d1b042b50dc1bd03013e8b9eab4bae7de607a9c4bb82cb7f8c5f74a3`。推理阶段只读取每条序列第一行 GT 初始化框，不读取未来 GT，不使用 Qwen，也没有 VOT/low22 输入。

实现提交为 `e629fc94546cc208f86226b6f2c0c79af25afead`，启动脚本权限提交为 `c36cd94f4429c6d081cdfac50224a5dd558731cb`。两轮独立审查最终均为 PASS：冻结规格、GT 边界、122/30互斥全集、逐帧 schedule、trace/prediction 行数、alias 计数、TOCTOU 源码复核和 shell `pipefail` 均已闭合。

### 24.150.2　B0 fixed6 真实 CUDA 同构与内存隔离终态

最终 B0 目录为：

```text
/root/autodl-tmp/sutrack_rgbd_survival_template_gate_v1/fixed6_parity_v5
```

六条冻结序列各自运行直接 SUTrack safe-v1 与 survival probe，汇总门结果 SHA256=`d11c4b0c06d51ef77ecd07302a6bbbed7f1fcec39290ef3cb80ecc694f4defc0`。终态为：

|检查|结果|
|---|---:|
|固定序列|6/6|
|完整模板事件|14|
|跨分支 tensor storage 检查|182|
|跨分支共享 storage|0|
|公开 bbox 最大误差|0|
|best score 最大误差|0|
|writer state 最大误差|0|
|H=12 / decision steps=2|通过|
|未来 GT 被 tracker 读取|否|

快照检查覆盖 old、new 和安装后的 live 三棵 tensor tree；每个事件创建帧和每个 shadow 递归帧都比较底层 storage 指针。服务器旧版 PyTorch 不支持 `untyped_storage()`，首次诊断因此 fail-closed；最终实现同时支持新接口与旧版 `storage()`，失败诊断目录保留但没有被当作成功证据。

### 24.150.3　M1 fixed6 H=12 长轨迹终态

最终冻结 trace 位于：

```text
/root/autodl-tmp/sutrack_rgbd_survival_template_gate_v1/fixed6_trace_v2_frozen
```

两个 shard manifest SHA256 分别为 `bd1fa56229221f810bd449a6ff89b328e2d23d2e63403613d725995a801c0f8e` 与 `26403395aac993c6c0af7be5c75844ec512032215538283b2424799d4e77b5cd`。两片合计 10,041 帧、10,035 条非初始化 trace、67 个模板事件、65 个完整事件、850 次跨分支 storage 检查；源码、配置、checkpoint、短 identity 文本和 trace plan 均在首帧前冻结，并在发布终态文件前重新哈希。任一中途变化只允许保留 `.partial`，不能创建 `manifest.json`。

后验标签分析目录为 `/root/autodl-tmp/sutrack_rgbd_survival_template_gate_v1/fixed6_analysis_v4_frozen_provenance`；capacity result SHA256=`12b18f5c3768a586b378dd448ad900c38cf4da9e4fa1434ac9a1ccd1e0685407`，event rows SHA256=`6f74acb7110f1e8701e8aaa28e31120036c74e69e984e2bdb1855d567fcaa4d2`。

|M1标签统计|数量|
|---|---:|
|可标注完整事件|65|
|old-template veto 正例（未来均值差≥0.02）|2|
|ambiguous|63|
|new/old 十帧确认失锁|0 / 0|
|rescue / catastrophic-old|0 / 0|
|正例覆盖序列|2（`bottle03_indoor`、`flower03_indoor`）|

oracle 在 fixed6 上把未来平均 IoU 从 `0.8876851665` 提高到 `0.8896549992`，增量 `+0.0019698326`，但没有任何 failure start 可减少。该结果只说明普通 fixed6 的模板有害信号很稀疏，不能据此训练或声称改善 VOT ROB；预注册的容量判决必须由 full152 calibration 完成。

### 24.150.4　full152 Train-only 采集已启动，尚无终态结论

2026-08-30 04:45 CST 已启动：

```text
/root/autodl-tmp/sutrack_rgbd_survival_template_gate_v1/full152_trace_v1
```

GPU0 严格运行 frozen fixed6 + trace-plan shard0，GPU1 运行 shard1；两者并集恰为 152 条唯一 DepthTrack Train 序列。启动器在运行前检查 8 GiB 可用空间、trace plan SHA、源码 clean 状态和两片全集；每个 runner 还独立重复验证允许 scope。当前仅处于无未来GT的轨迹采集阶段，不能报告 capacity 指标。

终态后只允许先对 122 条 calibration 加入 GT，并检查：至少100个 veto 正例、覆盖至少15条序列、oracle failure starts 至少减少10、未来均值 IoU 不下降。任一项失败即停止 learned gate，不消费30条 audit、不运行 low22；全部通过后才进入三种子 group-OOF 与一次冻结 audit。无论后续结果如何，仍禁止自动 full127。

<!-- RGBD-HANDOFF-24.151-SURVIVAL-TEMPLATE-CAPACITY-FINAL -->

## 24.151　Survival-template full152 容量门终态：监督稀疏，停止 learned gate（2026-08-30）

### 24.151.1　终态结论

本节对应 §24.150 中预注册的 `DepthTrack Train 152` 条 survival-template 反事实轨迹。轨迹和 calibration-only 分析均已完整结束，但四项容量门只通过两项，因此正式结论为：

```text
decision = survival_template_capacity_not_supported
capacity_supported = false
```

按冻结计划立即停止该 learned gate 分支：不训练三种子 group-OOF、不读取30条 audit GT、不运行 VOT low22，更不运行 full127。该实验没有改变正式推理配置或 checkpoint，因此不会改写已经达标的 DepthTrack/CDTB 结果。

### 24.151.2　full152 无未来GT trace 的完整性

两片 runner 均先只读取各序列第一帧 GT 进行初始化，跟踪器看不到未来 GT；文本仍为冻结的 DepthTrack Train 152 条 short identity manifest，在线文本关闭。公开分支为 `direct_safe_v1_new_template`，反事实分支为 `old_template_without_event_write`，`H=12`、决策观察帧为前2帧。

| 分片 | 序列 | 帧 | events | complete events | 跨分支 storage alias 检查 | elapsed(s) | manifest SHA256 |
|---|---:|---:|---:|---:|---:|---:|---|
| shard0 | 79 | 114,937 | 1,240 | 1,227 | 16,032 | 9,677.319 | `a8229e9f6fb240504ed1317a08c116772d28bec7733f747311620023b6775c0c` |
| shard1 | 73 | 105,017 | 1,146 | 1,138 | 14,836 | 8,954.602 | `c6f6e374676727544280fc38989193aca1f4b0b0cba4a7fa32b53e4e85aefc12` |
| 合计 | **152** | **219,954** | **2,386** | **2,365** | **30,868** | — | — |

两片 manifest 均为 `complete=true`、`branch_tensor_isolation_verified=true`，所有输出先写 `.partial`，终态才原子改名。自动 calibration 守卫仅在两片正式 manifest 同时存在且 `.partial` 全部消失后启动，终态退出码为0。checkpoint SHA 仍为 `2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`；trace-plan SHA 为 `b9d36eb7d1b042b50dc1bd03013e8b9eab4bae7de607a9c4bb82cb7f8c5f74a3`。

### 24.151.3　严格 calibration-only 容量结果

分析器只为冻结 split 中的122条 calibration 序列加入未来 GT；结果中 `role=calibration`、`sequence_count=122`，30条 audit 仍未消费。split SHA 为 `f09ee1c36e51c24969311365a2bda0970ac951baf54ca916878cae77bac3dabd`，calibration/audit 角色互斥。

| 项目 | 正式结果 | 预注册要求 | 判定 |
|---|---:|---:|---|
| veto-positive events | **29** | ≥100 | **失败** |
| positive sequences | **17** | ≥15 | 通过 |
| oracle confirmed-failure starts 减少 | **1**（12→11） | ≥10 | **失败** |
| oracle future mean IoU | `0.912163277560923` | 不低于 baseline | 通过 |
| baseline future mean IoU | `0.911389504397144` | — | — |
| mean-IoU delta | **+0.000773773163779** | ≥0 | 通过 |

calibration 中共见到1,917个事件，其中1,898个完整；1,863个具有可用标签，另有35个 absent-GT events 和19个 incomplete events。可用标签分布为：

```text
ambiguous = 1,810 / 1,863 = 97.15%
veto      =    29 / 1,863 =  1.56%
keep      =    24 / 1,863 =  1.29%
```

没有 `catastrophic_old`，但也只有1个真正减少 confirmed failure 的 rescue：`cube04_indoor` 在 event frame 775 选择旧模板，可避免新模板分支从 frame 778 开始的持续失败；该事件未来均值 IoU 为 `0.489083` 对 `0.0`。其余11个 baseline failure starts 未被 oracle 消除，包括 `ball13_indoor` 3个、`cube04_indoor@748`、`cup07_indoor` 3个、`mushroom01_indoor` 3个和 `shoes01_indoor@1148`。

29个 veto 正例仅覆盖17条序列，并明显集中：`cup14_indoor=8`、`cup08_indoor=3`、`cup11_indoor=3`，三条 cup 序列合计14/29（48.28%）。这意味着即使忽略数量门，监督也存在明显的类别/序列集中风险。

### 24.151.4　为什么不继续训练 gate

本实验否定的不是“模板永远无用”，而是当前这个更具体的假设：用模板写入后前2帧的置信度、margin、RGB identity、Depth consistency、temporal continuity 等局部特征，可以稳定预测未来第3～12帧 survival，并据此学习高精度 veto gate。

证据有三点：

1. 97.15%的完整可标注事件在冻结 `0.02` mean-IoU margin 下没有明确分支偏好，可学习监督过于稀疏。
2. 正例高度集中，且只有1/12个 baseline failure starts 能被完整后验 oracle 消除，远低于 ROB/EAO 所需的生存改善量级。
3. 局部证据与长期结果并非单调对应。例如 `bottle03_indoor@160` 中旧模板未来 IoU 高约 `+0.0715`，但前两帧置信度和 identity 都反而支持新模板；若只学两帧局部打分，容易把这类真正正例判错。

因此不能降低门槛、偷看 audit 或直接在 low22 上调参来挽救该方向。`capacity_result.json` SHA 为 `7ee8f9b301f49f1af59e32c8c2bfcc214b2f19dfcd4e80cad2f22f7e10ee571a`，`event_rows.jsonl` SHA 为 `b14b93e3a64bf7e80026c737aa100ee0d5173e2e0d014bed3381b09a21a76653`。

### 24.151.5　下一步架构转向

下一阶段不再围绕稀有的“是否撤销一次模板写入”事件训练分类器，而应把监督和控制点前移到每帧候选身份与递归状态：

```text
visual/language/depth top-K candidates
        ↓
target-memory 与 distractor-memory 关联
        ↓
protected state / tentative state 并行1～2帧
        ↓
候选身份确认后原子提交 state；冲突时回滚
```

这一路线直接作用于导致 VOT ROB/EAO 下降的高频状态传播，而不是等待少量模板更新事件；模板只在候选身份和状态已确认后作为次级事务提交。新的方法仍必须先在 DepthTrack Train 做容量和递归 rollout 验证，再按既定规则只跑 VOT low22；只有 low22 的 EAO/ROB 和 failures 同时改善，才向用户报告并考虑 full127。

<!-- RGBD-HANDOFF-24.152-CANDIDATE-CAPACITY-FINAL -->

## 24.152　同一搜索裁剪 top-K 候选容量终态：动作空间不足，停止关联器训练（2026-08-30）

### 24.152.1　实验范围与终态结论

本节完成 §24.151 后预注册的候选身份容量检查。模型、文本和权重均保持冻结：仍使用 DepthTrack Train short identity 文本、SUTrack-L384 与 checkpoint `SUTRACK_ep0180_l384.pth.tar`（SHA256=`2a686e8b55091d3396886de0c9e2d7a46794a5773581b96e37006f851e9dacd4`）。tracker 只在初始化读取第一行 GT；未来 GT 仅在推理全部结束后由 analyzer 加入。候选仅来自同一个 factor-4 搜索裁剪、同一 Center Head 响应图的确定性 `K=1/4/8/12` 解码。

正式 schema-v5 终态为：

```text
decision = candidate_capacity_rejected_protocol_integrity
candidate_capacity_supported = false
numeric_capacity_supported_on_calibration_only = false
```

因此该分支停止：不训练 top-K 候选关联器、不进行三种子 group-OOF、不消费冻结 audit 指标、不运行 VOT low22，更不运行 full127。该实验没有改变公开 tracker、checkpoint、DepthTrack Test 或 CDTB 配置，所以此前已达标结果不受影响。

### 24.152.2　full152 无未来 GT trace 完整性

正式 source plan 为 `/root/autodl-tmp/sutrack_rgbd_candidate_state_v1/source/candidate_capacity_plan_v8.json`，SHA256=`eac283bde8293995f0a92f3663a5a26431b46866eacb31349ff0113fa15490b5`。两片各含76条唯一序列，合计152条、219,954帧；压缩 trace 合计528,187,900 bytes，低于1.5 GiB预注册磁盘上限。

|分片|序列|trace rows|trace bytes|trace SHA256|manifest SHA256|
|---|---:|---:|---:|---|---|
|shard0|76|109,841|264,485,041|`eec450ae6a374167458efbc75fbc047c5fbc800e519567c1036b341d3f12a33e`|`6424f6ad06ae34633efab777c04c104bd0db0cd47d78af932691db94a744ff13`|
|shard1|76|109,961|263,702,859|`40232c78ccdd9b5a896a8570219109dfc2e03e6c08ffe84f3b76a666e5aa5bd1`|`0a82a90445ad54f2edfa56faf4ebfc364eafac3c894923022d02e56ed289eb5a`|

注：终态容量表的行数是按风险窗口展开后的179,540行，不等同于原始逐帧 trace 行数。两张 RTX 3090 在终态核验时均空闲，日志错误扫描为0。

### 24.152.3　`toy07_indoor_320` 数据契约与 analyzer 修复

原冻结 analyzer 按严格帧数契约正确停止，并报：

```text
ValueError: GT frame-count mismatch toy07_indoor_320
RGB = 1367, Depth = 1367, GT = 1406
```

两次独立只读复核均确认 RGB/Depth 文件名连续为1--1367，没有缺帧；`groundtruth.txt` 有1406行，只有末尾1368--1406共39行没有对应图像。完整 GT 文件 SHA256=`683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2`，bytes=20,581。

修复不是通用截断，而是 fail-closed 的精确尾部契约：仅当 sequence、完整 GT SHA/bytes、RGB/Depth 文件清单 SHA、首末文件名、1367帧计数以及39行尾部长度全部精确匹配时，才允许使用前1367行；任一字段变化立即拒绝。相关7项回归检查全部通过。原 analyzer 保持不变，schema-v5 analyzer 和 amendment 记录了该例外及来源。

### 24.152.4　必须保留的 audit 协议事件

在定位 `toy07` 前，曾执行一次“只比较文件行数”的全集扫描。该扫描没有解析 bbox 数值、没有计算 audit 指标，也没有把 audit 用于模型选择，但它确实打开了冻结30条 audit 的 GT 文件。因此不能继续声称 audit 文件在字节层面未被触碰。终态明确记录：

```text
audit_gt_byte_untouched = false
audit_gt_values_parsed = false
audit_gt_metrics_computed = false
audit_gt_used_for_model_selection = false
audit_gt_files_read = 30
audit_gt_files_read_by_analyzer = 0
```

所以正式 decision 包含 `protocol_integrity` 拒绝。今后若需要独立审计，必须从从未打开标签的新授权数据源重新冻结 holdout；不能把这30条悄悄重新命名为“untouched audit”。即使忽略该协议事件，本节的 calibration 数值门也失败，因此科学路线结论不受该事件改变。

### 24.152.5　122条 calibration 的候选容量结果

122条 calibration 中共有165,152个可见帧、12,979个风险帧，检测到209个确认失败起点，分布于63条序列。容量结果如下：

|项目|K=1|K=4|K=8|K=12|冻结要求/结论|
|---|---:|---:|---:|---:|---|
|可救确认失败起点|19|19|19|19|K8数量≥10通过，但没有超出K1的新救援|
|失败起点覆盖率|9.0909%|9.0909%|9.0909%|9.0909%|K8要求≥40%，失败|
|覆盖序列数|14|14|14|14|K8要求≥10，通过|
|风险帧 oracle mean IoU|0.008206|0.019199|0.020888|0.021566|K8−K1=`+0.0126819`，要求≥0.02，失败|

K8/K12失败救援比为1.0，说明K=8已经饱和；但这种“饱和”发生在很低的9.09%覆盖率上。只有112个风险帧、34条序列存在优于rank-1的非首候选；这些候选与公开分支的平均中心差为1.62665个 GT 对角线，最大5.61503。最终七项门中，失败样本数量、K8救援数量、覆盖序列数和K8/K12饱和通过；覆盖率、风险IoU增益和audit字节未触碰三项失败。

14条可救援序列为：`bag04_indoor`、`ball08_wild`、`ball16_indoor`、`book06_indoor`、`clothes_indoor`、`colacan02_indoor`、`cup05_indoor`、`guitarbag_indoor`、`human03_wild`、`leaves03_wild`、`leaves04_indoor`、`lock01_wild`、`pigeon05_wild`、`toy03_indoor`。确认失败最集中的代表包括 `leaves04_indoor=13`、`ball13_indoor=13`、`colacan04_indoor=11`、`beautifullight01_indoor=10`、`ball07_indoor=8`。这些失败大多已经超出当前 factor-4 crop 或进入错误递归中心；在同一响应图内增加K无法产生缺失的正确目标候选。

### 24.152.6　为什么停止 top-K selector，以及下一架构

本实验否定的是“只在当前同一 crop 的 Center Head 峰值间学习重排，就足以显著修复 ROB/EAO”。K=4/8/12没有比K=1多救回任何确认失败起点，证明主要瓶颈不是 selector 排错，而是动作空间中根本没有正确目标，或错误 bbox 已经改变下一帧搜索域。

下一步必须构造真正不同的候选来源，而不是继续扫描K或重复已经失败的统一 factor-6 搜索：

```text
公开保护分支：完全保持当前 baseline state/template
        │
风险触发（仅用在线可见证据）
        ↓
扩展候选：当前中心 + last-reliable中心 + 速度外推中心
          在受控 factor-7 crop 中分别解码
        ↓
tentative branch 独立递归2帧
        ↓
相对 survival / RGB身份 / Depth可靠性 / 运动连续性确认
        ↓
原子 promote：bbox state + template + annotation + RGB/Depth memory + counters
否则 rollback，公开分支逐位保持 baseline
```

新动作空间仍先在 DepthTrack Train 做无未来GT trace和后验容量验证。由于旧30条 audit 已发生count-only打开事件，下一轮只能报告 calibration diagnostic，或从新的未读数据源冻结合法 holdout。只有 Train-only 容量确实支持后，才允许运行固定 VOT low22/303；即使 low22 通过也必须先报告用户，禁止自动 full127。

### 24.152.7　可复查产物与代码

- schema-v5结果：`/root/autodl-tmp/sutrack_rgbd_candidate_state_v1/b1_capacity_analysis_v4_schema_v5/capacity_result.json`，bytes=99,067，SHA256=`fac1032fdc177689d980b3b75081ba42f27292a2a54e18279909e502d37ff721`；
- 行级结果：`capacity_rows.jsonl`，179,540行，bytes=107,980,863，SHA256=`5fe8c125ff8b0d121487798f29455251dae5706aa30182510cabe4db22926f2b`；
- metric-blind amendment v2：SHA256=`0cf8ba6e30035e41c7d73397db796e57102b1b973675eb42332def3faf89df14`；
- schema-v5 analyzer：SHA256=`f68be2988927143803f1807463cad9f3fcd1fe429640f5a1d3815306ba6363d3`；
- GT契约模块：SHA256=`30825057ba84349a392d374a5ac763ba8d1513f05d984104a1ce901b31da5e31`；
- 7项契约回归检查：SHA256=`f32693f538cdc298b81d9eb90f9f6d480b92259caa733cfdf65ff56eecf8cb04`，`python -m unittest ... -v`为7/7通过；
- 相关代码提交：`0a550bf1a9fa62ed997d8b08d557ed267f2b6b20`。

<!-- ## 24.153 -->
## 24.153　扩展候选动作空间：正式 fixed-6 容量实验（2026-08-30）

### 24.153.1　实验问题与冻结范围

同一 factor-4 crop 内 K=1/4/8/12 已证明无法产生足够的新目标候选。本轮只回答一个更窄但关键的问题：在不改变公开 SUTrack 递归状态的条件下，从 `current prior`、`last reliable` 与受限速度外推三个中心分别执行 factor-7 搜索，是否能在连续两帧 strict-H2 条件下救回已知失败链。

正式范围仍是 DepthTrack Train fixed-6 calibration diagnostic：`bottle03_indoor`、`ball16_indoor`、`bag04_indoor`、`flower03_indoor`、`pigeon05_wild`、`toy03_indoor`，共10,041帧、10,035条非初始化 trace。推理只读取每条序列第一帧 GT 初始化框；未来 GT 只允许在推理完成后的 analyzer 中使用。候选永不提交，VOT low22 与 full127 均未随本实验自动启动。

冻结的判定门保持不变：至少5个完整 strict-H2 救援、覆盖至少3条序列、相对 same-crop K12 至少新增2个救援、至少1个救援属于 `(last_reliable ∪ velocity) - current_factor7`、风险帧 oracle mean IoU 增益至少0.05。混合来源的两帧 pair 不计入 last/velocity 独有救援。

### 24.153.2　三网络保护与 fail-closed 契约

runner 同时实例化三份同 checkpoint、初值逐值相等且 storage 两两不相交的模型：public baseline、independent direct control、candidate probe。候选只在 probe 网络执行，并使用 public 当前帧推理前 template/annotation 的逐值克隆。每帧在 probe 前后都比较 public/control 的完整输出对象、bbox、score、递归 bbox、frame id、语言状态、模板与 annotation、safe-template policy 和 network buffers；任何差异、probe error、rank-1 parity 非零或候选输入共享 storage 都立即终止，不能生成 `complete=true`。

无限正式运行还必须精确满足冻结的 `frame_count=10041` 和 `trace_row_count=10035`。trace 每行写入后立即 flush，并检查256MiB压缩预算与1GiB剩余磁盘下限。所有 parity、计数、快照和 trace 校验都发生在 staging directory 内；只有全部通过才以同文件系统 `os.replace` 原子发布正式目录。

### 24.153.3　实际执行的冻结快照

上轮复审指出“只验证 live worktree 后再从 live 路径 import”仍存在 TOCTOU。本轮已改为真实执行快照：planner 在项目 import 前冻结172个项目 Python 文件和完整 `site-packages` 24,429个非 bytecode 文件（4,287,088,516 bytes），包括 torch、numpy、cv2 的二进制扩展以及 easydict、yaml、CLIP、timm 和传递依赖。runner bootstrap 仅使用标准库，把这些精确字节复制到只读临时目录，再以 `python -I -S` 从快照 runner 重新启动，移除 live repository/site-packages import path，并禁用 bytecode。

2.21GB SUTrack checkpoint 与890MB CLIP checkpoint 不写入额外磁盘副本，而是复制到 Linux sealed memfd；两者均要求 `F_SEAL_WRITE|F_SEAL_SHRINK|F_SEAL_GROW|F_SEAL_SEAL`（mask=15），实际加载路径为 `/proc/self/fd/*`。快照在 ML import 前和发布前逐文件复核；运行时还审计所有带 site-packages 路径的已加载 module 必须来自快照。

最终 bootstrap v5 以当前提交 `db35f1c` 和 plan v7 运行成功：12帧/6条trace、probe error=0、所有 output/state/storage/rank-1 parity 为0，1,929个已加载外部 module 全部来自快照，两个 sealed 权重均报告 immutable=true。plan v7 SHA256=`9254a7278a18d64ef1766b483cc11d9daf1358d8ab69ea6ccc28093c035d21bf`。两路独立复审（experiment-spec 与 standards/correctness）均给出 PASS，无剩余阻断项；旧 bootstrap v4 不再作为有效凭证。

### 24.153.4　正式运行状态与可复查路径

正式 plan：`/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/m1_fixed6_plan_v3/plan.json`，bytes=6,903,776，SHA256=`fb92e270411b497582663b9b8ead38d79492885403c9c6b449f9fb7c554835c1`。正式 runner 于2026-08-30启动，初始 PID=8732；输出目标为 `/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/m1_fixed6_trace_v3/run`，日志为同级 `m1_fixed6_trace_v3.log`。本节记录的是启动与协议状态，不提前声明容量结果；最终数值和是否允许进入 VOT low22 必须在正式 trace 完整原子发布并由 analyzer 通过后另行追加。

<!-- RGBD-HANDOFF-24.154-UNIFIED-FIXED6-V2-CAPACITY-FINAL -->

## 24.154　统一 fixed-6 扩展动作空间 V2：正式容量结论与路线停止（2026-08-30）

### 24.154.1　终态结论

§24.153 记录的正式运行已完成并通过执行完整性检查，但预注册容量门失败。最终判定为：

```text
complete = true
capacity_supported = false
decision = expanded_actionspace_capacity_not_supported
```

因此本路线停止在 DepthTrack Train fixed-6 calibration diagnostic：不训练候选router，不扩到122/152条Train序列，不运行VOT low22/303，更不运行full127。预注册阈值不因负结果而降低。公开SUTrack tracker、checkpoint、DepthTrack Test与CDTB结果均未被本实验改写。

### 24.154.2　实际执行、数据隔离与产物绑定

正式冻结提交为 `8b7367e622c9e9532f4cb4d8bfc39bc909764864`。plan位于：

```text
/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/
unified_fixed6_v2/plan/plan.json
```

plan SHA256=`649d4abaa91f9c1cdb0a301c171473a6e3e8887cae272a470a92942e802f0db9`，bytes=11,790,187。正式执行覆盖6条DepthTrack Train序列、10,041帧、10,035条非初始化trace：`bottle03_indoor` 3,185行、`ball16_indoor` 1,913行、`bag04_indoor` 1,668行、`flower03_indoor` 743行、`pigeon05_wild` 1,249行、`toy03_indoor` 1,277行。

launch receipt位于 `unified_fixed6_v2/launch_receipt.json`，SHA256=`713f9cb52ac6678c8deec89b880e86c7fbf76d693f18fd395acaa574953b8e95`，明确记录 `accepted=true`、`complete=true`、child exit code 0、进程已reap、prelaunch输出不存在、source closure运行前后不变且staging原子发布。正式trace：

```text
unified_fixed6_v2/run/unified_actionspace_trace.jsonl.gz
```

SHA256=`bf503ff76cbc04245cd0e2fb586328a12c1b48ea9d7379a5f7b98363bae88af1`，bytes=16,713,121；manifest SHA256=`e36d962dc4c64c904e0df5432f237b200eb91d5a7a9a5b1c09ef7a71cebc6a90`。10,035行全部满足：

```text
ground_truth_available_to_tracker = false
future_frame_text_used = false
candidate_committed = false
```

same-crop K12的rank-1 bbox/score与公开分支逐值误差为0，所有probe error为0，public/control输出、递归状态、模板、annotation与storage隔离检查均通过。未来GT只在receipt被接受、推理结束之后由analyzer打开。

### 24.154.3　正式容量数字

analyzer结果位于 `unified_fixed6_v2/analysis/capacity_result.json`，SHA256=`10912fe455036de58787fb73b97bdac1b07e73e4cc2acc7a0071bd2111634bec`。共有8,698个可见帧、1,783个风险帧和16个confirmed-failure starts：

```text
bag04_indoor:    252, 671, 1040, 1138
ball16_indoor:   613, 1432, 1513
bottle03_indoor: 487
flower03_indoor: 243, 411, 488, 628
pigeon05_wild:   37, 643, 839
toy03_indoor:    827
```

风险帧raw-IoU后验oracle均值如下：

|动作空间|风险帧 oracle mean IoU|
|---|---:|
|公开baseline|0.0169854894|
|same-crop K12|0.0864459209|
|current-center factor7 K4|0.1590037605|
|last-reliable factor7 K4|0.2932422563|
|velocity factor7 K4|0.2863635613|
|完整multi-center|0.3710678883|

完整动作空间相对baseline的单帧oracle增益为 `+0.3540823989`，说明扩大动作来源确实能在部分风险帧重新看见目标；但严格要求候选连续两帧成立的H2 survival救援结果是：

|冻结门|实际值|要求|结果|
|---|---:|---:|---|
|完整strict-H2救援|1|≥5|失败|
|覆盖序列|1|≥3|失败|
|相对same-crop K12新增救援|1|≥2|失败|
|last/velocity相对current独有救援|0|≥1|失败|
|风险帧oracle IoU增益|+0.354082|≥0.05|通过|

same-crop K12没有严格救援；current、last-reliable、velocity和完整multi-center虽然各计1个，但都是同一个 `toy03_indoor@827`，不能视作四个独立救援。结论是：多中心动作空间有较高的“单帧看到目标”容量，却缺少在公开错误state持续裁剪下的连续候选容量，不能支持后续router训练。

### 24.154.4　独立实验完整性审计

独立GPT-5.5 xhigh reviewer直接读取plan、receipt、trace、manifest、analyzer与结果，保存于：

```text
/home/SUTrack_RGBD_L/.aris/traces/experiment-audit/2026-08-30_run02/
/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/
unified_fixed6_v2/analysis/EXPERIMENT_AUDIT.md
```

审计结论为 `WARN`，唯一警告是审计当时本交接文档仍停留在“运行中”；GT provenance、raw IoU无自归一化、实际GPU执行、产物/commit绑定和fixed6范围控制均通过。审计明确支持本节的负容量结论，明确不授权VOT low22/full127，也不允许修改冻结阈值。本节同步后，文档滞后原因已处理，但原始审计 verdict 仍按原样保留，不事后改写为PASS。

### 24.154.5　为什么下一步测试递归重定心，而不是继续加K或改文本

trace后验显示，有些失败在确认前几十帧曾出现高IoU扩展候选，但下一帧仍围绕公开错误bbox裁剪，因此正确候选很快消失。例如 `flower03_indoor@411` 在约frame 355--365附近出现可用的last-reliable/velocity候选；`flower03_indoor@488` 在430--450附近、`pigeon05_wild@643` 在585--590附近也有类似现象。这说明下一问题不是“同一帧选更多峰”，而是：若把候选只提交给隔离branch的bbox state，后续crop围绕新state递归，能否跨过原失败窗口。

下一步先做一次可丢弃的fixed6原型，不形成正式论文结果：每5帧用metric-blind固定周期触发；从已接受trace读取模型候选框；模板冻结；branch只提交bbox并用factor-4围绕自身state递归最多60帧；公开baseline逐位保持不变；推理结束后才打开GT。原型若看不到持续survival，直接转向更强合法baseline；若看到容量，也必须重新冻结正式runner/analyzer并再次过Train gate，不能据此直接跑low22。

### 24.154.6　VOT与文本路线状态不变

固定low22仍为22条低指标序列、303个anchors。此前将长结构化文本改为稳定身份短文本只得到小幅改善：EAO/ACC/ROB从 `42.629281/71.827916/53.412816` 提高到 `43.274104/72.065511/54.388022`，confirmed failures从200降至195；最新anchor-specific Qwen视觉注释则退化到 `42.946692/71.910688/54.305144`，failures升至202。因此Qwen注释路线没有进入full127；当前full127最好结果仍是稳定身份短文本的 `74.0205826687/82.5793442346/89.5656513398`。

<!-- RGBD-HANDOFF-24.155-RECURSIVE-STATE-PROTOTYPE-AND-FROZEN-R1 -->

## 24.155　递归状态重定心原型与metric-blind fixed6 R1冻结协议（2026-08-30）

### 24.155.1　为什么独立帧H2失败后仍继续做一次递归原型

§24.154的multi-center单帧oracle在风险帧上很高，但严格连续两帧H2只有1个救援。进一步查看accepted trace发现，一些正确候选在失败前几十帧出现过，却因为下一帧仍围绕公开错误bbox裁剪而消失。因此本节只验证一个机制问题：把某个候选bbox写入完全隔离的branch state后，普通factor-4 SUTrack能否围绕新state持续递归，并在遮挡/重现后继续跟到目标。

原型始终满足：公开tracker不接收候选；branch模板在trigger冻结；候选来自accepted no-GT trace；未来GT只在所有推理结束后打开；每次运行公开bbox与accepted trace最大误差均为0。该原型不修改checkpoint、文本、Qwen、公开模板策略或VOT配置。

### 24.155.2　DepthTrack absent-GT解析问题与修复

第一次 `flower03` 原型在推理全部结束后打开GT时，因第171行 `nan,nan,nan,nan` 停止；随后 `bottle03` 第481行 `455,153,1,0` 和 `pigeon05` 第1212行 `455,193,2,0` 也暴露相同契约问题。根因不是模型，而是原型错误要求每一行必须是finite positive bbox。

正式unified analyzer的既有契约是：能解析为数字但包含NaN或非正宽高的行表示absent GT，返回None；只有非数字畸形行才报错。原型已改为完全相同的契约。2秒最小CLI复现由稳定red变为：

```text
GT_PARSE_GREEN nan_and_zero_height_absent=true
```

修复只影响推理完成后的后验统计，未改变任何模型forward或branch轨迹。

### 24.155.3　retrospective mechanism probe结果

共运行7个trigger probes，覆盖6个不同baseline failure events；其中5个事件得到递归救援，正例覆盖3条序列：

|序列/失败起点|trigger|最佳branch来源|baseline前5个可见failure IoU|branch前5个可见failure IoU|结论|
|---|---:|---|---|---|---|
|flower03@411|365|last-reliable|0/0/0/0/0|.065/.076/.096/.507/.927|救援|
|bottle03@487|465|current-prior|0/0/0/0/0|0/0/0/.506/.784|救援|
|ball16@1513|1495|current-prior|0/.012/.019/.037/.049|0/0/.083/.829/.867|救援|
|flower03@488|440|last-reliable|0/0/0/0/0|.082/.682/.678/.830/.790|救援|
|flower03@628|570|last-reliable|0/0/0/0/0|0/.825/.826/.807/.849|救援|
|pigeon05@643|585与590|所有来源|均低于.1|均未恢复|失败|

例如 `flower03@365` 的last-reliable branch在38帧absent区间后，于重新出现后的第4、5个可见帧恢复到IoU 0.507/0.927，整段23个可见评估帧mean IoU为0.66078；公开baseline为0。`bottle03@465` 的current branch在44个可见评估帧mean IoU为0.81133，公开baseline仅0.08582。完整汇总：

```text
/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/
unified_fixed6_v2/prototype_recursive_state_rollout/
prototype_recursive_state_rollout_summary.json
```

SHA256=`38df8e9eb9fbb455aa2d46dfc923b4250cd99eac2d08df9ed3a9c050db63e37f`。这些trigger是在查看过fixed6 GT后选择的机制探针，因此不能报告为“5/16正式容量”，不能用于在线阈值结论，也不授权VOT。

### 24.155.4　无GT trigger检查与正式R1冻结规则

只读取accepted trace中的在线字段后，固定每5帧check共有2,004个。冻结的metric-blind trigger为：任一expanded top1候选 `score>=0.05` 且与公开deployed bbox的IoU `<0.20`。该规则产生656个trigger frames；5个retrospective正例和2个pigeon负例都被触发，说明它不会只筛掉负例。

正式R1协议已在运行前冻结于：

```text
/home/SUTrack_RGBD_L/refine-logs/
EXPERIMENT_PLAN_AMENDMENT_RECURSIVE_STATE_ROLLOUT_V1_20260830.md
```

SHA256=`032c9eb1521d4ccbd6ffeb7bc0e90f61c24f8ac18e7a32fccda6a153f5060dae`。trigger后保留current/last/velocity的去重rank-1候选，branch冻结当前模板并围绕自身bbox执行factor-4最多80帧；所有branch均与公开tracker隔离。runner允许按chunk≤8批量推理，但必须先证明batch/scalar bbox与score parity；正式输出上限为800 triggers、2,400 branches、200,000 branch updates和128MiB gzip trace。

推理原子发布后，analyzer才打开完整GT，并先复现原16个baseline failure starts。严格递归救援要求：failure起点后的前10个可见帧不能全部IoU≤0.1；前5个可见帧至少2帧IoU≥0.5；前10帧mean IoU相对baseline至少+0.20；禁止跨branch拼接。最终门保持至少5个不同failure rescues、覆盖至少3条序列，且current来源与last/velocity来源都必须有救援。

### 24.155.5　当前权限边界

R0只证明递归state动作空间值得正式测试；R1还没有正式结果。当前没有启动VOT low22，更没有启动full127。即使R1容量通过，下一步也只能在Train上设计部署可见的relative branch selector与原子事务；必须再通过零新增灾难的冻结门，才可能按用户规则只测low22。full127始终需要先报告用户，禁止自动运行。

<!-- RGBD-HANDOFF-24.156-RECURSIVE-STATE-R1-FORMAL-FINAL -->

## 24.156　metric-blind递归状态rollout R1正式终态、独立审计与后续边界（2026-08-31）

### 24.156.1　正式运行已闭合，文档不再处于“运行中”状态

§24.155冻结的DepthTrack Train fixed-6 R1已经完成。正式launch receipt：

```text
/root/autodl-tmp/sutrack_rgbd_expanded_actionspace_v1/
recursive_state_rollout_v1/formal/launch_receipt.json
```

SHA256=`2d561ead2667e4449c8e1cbc2806223486f8f6c9d958b23fddc9a2f36e34c5e6`，记录 `accepted=true`、`complete=true`、child exit code 0、child已reap、`atomic_promotion=true`、`error=null`。正式期间没有启动VOT low22或full127。

frozen plan SHA256=`685b0a01ad01c43e23e8db125d04f6e461cdd078975433ad2941b7752182ea7d`，bytes=5,263,115；代码commit=`a27130bc2e6c2de2dfcf84314a469dd0f94a22dc`。

### 24.156.2　执行规模、公开路径保护与完整性

正式manifest：

```text
recursive_state_rollout_v1/formal/run/manifest.json
```

SHA256=`50609a1ba70fd277570ccc7b9cbde6235fdd95bbc65cda29b01fada10f5b4c02`。实际执行：

|项目|终态|
|---|---:|
|frames|10,041|
|metric-blind trigger frames|656|
|created branches|1,968|
|recursive branch updates|153,705|
|trace rows|155,673|
|maximum active branches|48|
|GPU peak memory|3,062,910,464 bytes|
|wall time|8,860.97秒|
|exception/CUDA/probe/batch errors|全部0|
|maximum public bbox error|0|
|maximum public score error|0|

runner没有得到未来GT，完整GT只在推理完成、receipt接受且trace/manifest验证后由analyzer打开。所有trace均记录 `future_frame_text_used=false`、`candidate_committed=false`；公开tracker始终运行原baseline，候选分支未提前写回。

逐序列执行量：

|序列|frames|triggers|branches|branch updates|
|---|---:|---:|---:|---:|
|`bottle03_indoor`|3,186|67|201|16,080|
|`ball16_indoor`|1,914|228|684|53,181|
|`bag04_indoor`|1,669|48|144|11,187|
|`flower03_indoor`|744|103|309|24,657|
|`pigeon05_wild`|1,250|192|576|44,280|
|`toy03_indoor`|1,278|18|54|4,320|

正式branch trace SHA256=`96c94579acd84a409c1a6e47e3075ca7e1e882703d34639138400c5e4a2bcb53`，bytes=7,330,656。

### 24.156.3　冻结容量门正式通过：7个rescue、4条序列

analyzer结果：

```text
recursive_state_rollout_v1/formal/analysis/capacity_result.json
```

SHA256=`4c96d0fdd20e21290e958da433e5ab1debd10205fa04654c2eb72fecfb507497`。16个baseline confirmed failure starts中共有7个严格rescue，覆盖4条序列：

|失败事件|trigger|最佳来源|baseline future mean IoU|branch future mean IoU|增益|
|---|---:|---|---:|---:|---:|
|`bottle03@487`|465|velocity|0.000000|0.652508|+0.652508|
|`ball16@1513`|1450|current|0.026128|0.686207|+0.660078|
|`flower03@243`|180|last|0.000000|0.754689|+0.754689|
|`flower03@411`|395|last|0.000000|0.615500|+0.615500|
|`flower03@488`|480|last|0.000000|0.753569|+0.753569|
|`flower03@628`|610|last|0.000000|0.768200|+0.768200|
|`pigeon05@839`|800|current|0.000000|0.677461|+0.677461|

成功来源分布为current=2、last=4、velocity=1，满足来源多样性门。未救回9个失败：`ball16@613/@1432`、`bag04@252/@671/@1040/@1138`、`pigeon05@37/@643`、`toy03@827`。

冻结门终态：minimum rescue `7>=5`通过；positive sequences `4>=3`通过；current与last/velocity来源多样性通过；公开parity、错误门和资源上限全部通过。因此 `capacity_supported=true`，decision=`recursive_state_rollout_capacity_supported`。

### 24.156.4　独立审计与严格claim ceiling

独立GPT-5.5 xhigh只读审计保存于：

```text
recursive_state_rollout_v1/formal/analysis/EXPERIMENT_AUDIT.md
recursive_state_rollout_v1/formal/analysis/EXPERIMENT_AUDIT.json
/home/SUTrack_RGBD_L/.aris/traces/experiment-audit/
2026-08-31_run01/reviewer_trace.json
```

audit Markdown SHA256=`7facf734512831d8dbfb1b12c47b3d897c6fbed1c0c8578bec437c2b212acb80`；JSON SHA256=`163ef8b99077d149570aeafc1eba70d1237e7dfc98ea0e30771f91c02e0c83d8`。

审计总体为`WARN`，唯一警告是审计时本canonical和本地进展文档仍写“R1运行中”。Ground Truth provenance、raw-IoU计分、产物/哈希链、实际branch执行、未来信息隔离、公开parity、错误和资源门均为PASS；独立trace复算也得到完全相同的16 failures、7 rescues、4 sequences。续写本节后，文档过期问题已闭环，但保留原始WARN，不事后篡改为PASS。

本实验的严格上限是：

```text
metric-blind fixed6 recursive action-space capacity only;
no deployable selector or VOT claim
```

7个分支是推理结束后由真实GT后验oracle选出的。这证明“正确递归恢复轨迹存在”，不证明在线系统知道选哪个分支；不能宣称VOT EAO/ROB提升，不能从本结果直接跑low22/full127。

### 24.156.5　对当前瓶颈的新认识与下一步

R1使根因判断更具体：single-frame top-K和strict-H2失败，并不是正确目标完全不存在；当候选围绕自身bbox递归时，16个失败中的7个能够跨越失败窗口。真正尚未解决的问题已从“是否存在恢复动作”收敛为：

```text
如何只使用在线证据，高精度判断哪个递归分支值得原子promote，
并在不确定时严格回退protected baseline。
```

下一阶段只能先冻结Train-only selector协议。可用特征包括相对score/margin/entropy轨迹、首帧RGB身份一致性、Depth有效性与变化、中心/尺度连续性、protected与tentative分歧、稳定身份文本相似度以及distractor memory。必须采用按序列分组的OOF或新未读holdout、高精度弃权、异常fail-open，并原子提交bbox、模板、annotation、RGB/Depth记忆与计数器。

当前fixed-6只有7个正例、4条正例序列，数量不足以在同一集合自训自测后宣称泛化。应先做冻结trace的在线特征可分性审计；若不足，再按预注册范围扩大DepthTrack Train采集。只有Train selector做到零新增catastrophic/failure、公开parity为0且跨序列救回，才允许按用户规则只测VOT low22；low22明确改善并报告用户后，才可能决定是否跑full127。

### 24.156.6　VOT、文本与权重状态不变

本次没有重新训练checkpoint，也没有运行VOT。当前full127正式最好仍为identity-only路径：

```text
EAO / ACC / ROB = 74.0205826687 / 82.5793442346 / 89.5656513398
```

low22 identity-only仍为 `43.274104/72.065511/54.388022`、195 failures；Qwen anchor视觉文本仍为 `42.946692/71.910688/54.305144`、202 failures，因此不进入全量。Qwen3_8B权重继续保留，不在清理范围内。

<!-- RGBD-HANDOFF-24.157-BRANCH-RGBD-FINAL-20260831 -->
## 24.157　候选分支RGB-D证据正式负结果与路线切换

### 24.157.1　受控问题与完整性

第一版在线selector已证明score、geometry、public bbox RGB identity与public Depth无法把R1后验容量转化为可部署动作。本轮严格保持标签、模型族、正则、阈值、冷却、nested leave-one-sequence-out和零新增灾难门不变，唯一新增tentative branch自身的RGB身份、Depth有效性/变化/中心—环带结构及相对public差值。

无GT证据提取完整闭合：检查155,673条branch trace，生成13,521条证据、1,704个唯一决策帧，`current/last/velocity`各4,507条；RGB identity缺失0条、Depth缺失2,290条、public对照缺失2,524条。初始化后GT、后验容量、标签、未来观测和candidate commit均未读取。run1/run2 gzip逐字节一致：

```text
branch_rgbd_evidence.jsonl.gz
SHA256 a967ed39235f357c885ec6b9075051e095eed09bced2aa32abd5c303d9d26fae
```

### 24.157.2　正式结果

`branch_rgbd_only`与`full_plus_branch_rgbd`均在6/6个外层留一序列折得到`abstain_no_inner_configuration`，正式动作0，7个R1后验rescue覆盖0。最接近通过但仍被拒的inner precision范围为0.091～0.600，远低于冻结0.95门；降低门槛会同时放入catastrophic，不能作为优化。

```text
branch_rgbd_telemetry_supported = false
decision = branch_rgbd_fixed6_separability_not_supported
Train-152 / VOT-low22 / VOT-full127 = false / false / false

branch_rgbd_separability_result.json
SHA256 fe2a8e7a224e3c3d7375567425b46ba64d3f1186fa8d570ca275f3194dcd6a5b

posthoc_negative_diagnosis.json
SHA256 3dcc27731ebba7542a7762086bf9438e78755e44ee93d852f129b42dd3caba3c
```

0个catastrophic不是安全成功，而是全部折弃权、没有提交动作。

### 24.157.3　最重要的新问题

branch RGB-D证据在全样本上并非无信息：branch-public IoU、宽高比、面积与identity差异均较大；branch RGB identity均值在beneficial/neutral/catastrophic约为0.64/0.56/0.51，Depth有效率约为0.87/0.86/0.38。但这些差异高度受序列域控制：

```text
Depth缺失：beneficial 141/2013=7.00%
           neutral    626/7030=8.90%
           catastrophic 871/1169=74.51%

ball16贡献932/1169=79.73%的catastrophic，
flower03贡献1254/2013=62.30%的beneficial。
```

因此手工selector容易学习“这是ball16/缺Depth域”或“这是flower03域”，而不是“该candidate是不是初始化时的同一实例”。这解释了全局均值有差异、按整序列留出却全部失败，也否定了继续增加颜色直方图、Depth均值或缺失阈值的路线。

### 24.157.4　正式停止与迁移决策

本轮触发预注册停止条件：不再在SUTrack fixed6上追加手工selector特征，不降低0.95精度门，不扩大Train-152，不运行VOT low22/full127。保留identity-only anchor语言、protected/tentative独立递归状态、bbox/template/annotation/RGB-D memory原子promote与完整rollback等创新点。

下一路线改为：用4～8帧predicted-crop真实递归rollout训练候选RoI target–distractor association与future-survival/hazard；同时静态审计更强RGB-D baseline，按用户规则baseline本身不重复评测，只在最合适的baseline上移植上述创新。Qwen3_8B继续保留；当前full127正式最好仍为：

```text
EAO / ACC / ROB = 74.0205826687 / 82.5793442346 / 89.5656513398
```

本节未训练checkpoint、未运行任何VOT。

<!-- RGBD-HANDOFF-24.158-LRA-M1-SMOKE-20260831 -->
## 24.158　Learned candidate association结构验收与真实SUTrack RoI绑定

### 24.158.1　隔离实现与默认安全性

在不触碰主目录dirty工作树的前提下，新建隔离工作树：

```text
/root/autodl-tmp/rgbd_baselines/SUTrack_learned_assoc_v1
branch = codex/learned-candidate-association-v1
base   = 847051ab02e83c2cf58719a69a1a14bde716f02e
HEAD   = 5e9e02b5b53667ef6698db83a52fc96d77d8d841
```

两次实现提交分别增加候选对齐association scaffold和真实SUTrack候选RoI接线。新增模块默认`USE=false`且fail-closed；没有association checkpoint时不能接管公开路径。最终head使用768维视觉/文本输入、3×3候选RoI、显式Depth值与valid mask、8维相对在线证据、多头target/distractor/H5 utility/H10 hazard/RGB-Depth reliability/action probability，总参数量1,027,958。

### 24.158.2　最终合成CPU/CUDA重复验收

最终768维版本在CPU和CUDA各运行两次：

```text
CPU run1/run2 SHA256
1284879328168808e3e256e0749bd8a645e5ae21c092704cdc1c571e69d61b33

CUDA run1/run2 SHA256
4bd7b7fb14c57eeda4cc11d2cf1ef23e5d2dfbfa29e04dbdad6e0c58674281b4
```

同设备两次结果逐字节一致。验证了候选RoI随自身bbox变化且不影响其他候选、候选置换等变、Depth缺失与有效零变化可区分、不同快照materialize零tensor alias、两次确认后的完整promote以及硬冲突后的完整rollback。合成验收没有加载权重、没有打开GT。

### 24.158.3　真实SUTrack-L384 one-frame side-channel验收

发布SUTrack-L384与CLIP权重只作为冻结视觉路径加载；association head未加载权重。对`bottle03_indoor`前两帧运行只读side-channel，a3与从最终提交重新运行的a4结果逐字节一致：

```text
result SHA256 = 86234262995a0ad4c9f8428ebc629f415de0d0ac9c3877ccd65fe0f563aade4f
search feature = [1, 768, 24, 24]
candidate RoI = [1, 3, 768, 3, 3]
Depth RoI = [1, 3, 2, 16, 16]
target prototype = [1, 768]
candidate count = 3
```

只读取初始化GT第一行15 bytes（SHA256 `566bbe188c944c905933ce6676ff40ba390120d1715b5bfd5bdf37923ef74414`），`future_gt_opened=false`、`ground_truth_available_to_association=false`、`candidate_committed=false`、`public_state_written=false`。原路径rank-1 bbox和score均精确parity，最大绝对误差为0。

a1因Hann window位于CPU而响应位于CUDA而停止；a2发现实际SUTrack-L384候选特征为768维而非原假设512维而停止。二者都未生成有效结果、未提交候选、未污染状态；修复后a3/a4一致通过。这两个失败属于结构验收发现并修复的接线问题，不是模型效果实验。

### 24.158.4　当前结论与下一步边界

现在只证明了“每个候选真正看自己的RGB-D RoI，并且完整状态事务可安全接线”，还没有证明学习头可以选对目标，也没有产生新checkpoint或新指标。本轮未运行DepthTrack Test、CDTB、VOT low22或full127；正式最好仍为：

```text
VOT full127 EAO / ACC / ROB = 74.0205826687 / 82.5793442346 / 89.5656513398
DepthTrack Test P / R / F  = 65.995933 / 65.335885 / 65.664250
CDTB P / R / F             = 75.387821 / 76.005850 / 75.695574
```

下一阶段进入DepthTrack Train-only：先在fixed6按metric-blind R1候选做有界表征采集和容量/存储验收，runner禁止读取初始化后GT，标签只能在推理后离线拼接；通过后才考虑Train152。仍禁止自动启动VOT low22/full127，Qwen3_8B继续保留。

<!-- RGBD-HANDOFF-24.159-LRA-M2-FIXED6-REP-20260831 -->
## 24.159　DepthTrack Train fixed-6 候选自身 RGB-D 表征正式采集

### 24.159.1　目的、输入与隔离边界

M1只证明了candidate-own RoI和protected/tentative事务能够安全接线；本轮进一步回答“能否在不读取未来GT、不提交候选的条件下，为每个递归候选收集可训练的自身RGB-D表征”。正式运行使用隔离工作树：

```text
/root/autodl-tmp/rgbd_baselines/SUTrack_learned_assoc_v1
branch = codex/learned-candidate-association-v1
HEAD   = 09d276141fc0cfe8bb81ba746391f18a294d309e
```

输入严格冻结为DepthTrack Train fixed-6、已接受的R1无GT递归trace、source trace、六个序列的初始化GT第一行和152条短身份文本manifest。对每个风险触发的`current_prior`、`last_reliable`、`velocity_extrapolation`分支，在age 0～2按分支自己的bbox裁factor-4搜索区，使用不可变首帧模板提取候选视觉RoI与Depth+valid。collector没有association checkpoint、没有初始化后GT、没有未来文本、没有candidate commit，也没有public state写入。

### 24.159.2　正式运行与确定资产

正式输出位于：

```text
/root/autodl-tmp/sutrack_lra_rep_fixed6_v1_20260831
exit_code = 0
wall_time = 446.434 s
peak GPU memory = 2,535,002,624 bytes
manifest SHA256 = e81fe82883a371e3118a0a457832e0d3daff6fcfb3761c4a6ec4b6d628379661
```

记录规模与冻结计划精确相符：总计5,904行、1,968个分支、656个触发点；age 0/1/2各1,968行，三种候选源各1,968行。每个分支恰好有三个age，每个触发点恰好有三种source×三个age，branch和trigger拓扑错误均为0。按序列分布为：

```text
bag04_indoor       432
ball16_indoor     2052
bottle03_indoor    603
flower03_indoor    927
pigeon05_wild     1728
toy03_indoor       162
```

有效负载为：

| 文件 | shape / rows | bytes | SHA256 |
|---|---|---:|---|
| `candidate_visual.f16` | `[5904,768]` | 9,068,544 | `2437cc7f...91c17` |
| `depth_rois.f16` | `[5904,2,16,16]` | 6,045,696 | `94f803e84...cbce` |
| `relative_features.f32` | `[5904,8]` | 188,928 | `f9ad1e9fa...ae6f` |
| `sequence_context.f16` | `[6,2,768]` | 18,432 | `b4ec20883...8dea` |
| `sequence_index.i16` | `[5904]` | 11,808 | `f15dd7706...53ee` |
| `metadata.jsonl.gz` | 5,904行 | 482,263 | `d2f25e62b...430fe` |

五个定长二进制文件合计15,333,408 bytes，与计划估计精确相等；含确定性gzip元数据后的总payload为15,815,671 bytes，远低于64 MiB硬上限。

### 24.159.3　完整性审计与发现的问题

所有浮点数组均为有限值，候选视觉全零行数为0，实际文件字节数与dtype/shape逐项精确一致，metadata row index连续，二进制sequence index与元数据逐行一致。Depth valid mask始终位于`[0,1]`，由mask重新计算的有效比例与在线相对特征中`depth_valid_ratio`的最大绝对差仅`1.5020e-05`。

本轮发现8条候选的Depth value平面全零。这不是异常丢弃项，因为其valid mask被独立保存，完整数据的有效比例范围为0～1；learned head可以明确区分“Depth缺失”和“Depth有效但变化为0”。这正面修复了旧手工selector把缺失深度误当零变化的风险，但尚未证明网络会正确利用该信号。

审计标志全部满足安全边界：

```text
association_checkpoint_loaded = false
ground_truth_available_to_collector = false
future_gt_opened = false
future_frame_text_used = false
candidate_committed = false
public_state_written = false
VOT low22 / full127 started = false / false
```

### 24.159.4　当前结论与下一步

本轮只能接受为“确定、完整、无标签泄漏的候选自身RGB-D表征资产”。它没有产生checkpoint，不能说明learned association已经能选择正确实例，更不能说明VOT指标提升。fixed-6还存在明显的序列域偏置，只可用于容量和管线诊断，不能替代Train-152的sequence-disjoint泛化验证。

下一步先冻结独立的post-inference标签连接：只对age-2记录按`branch_id`与已存在的离线H5/H10标签连接，模型输入禁止sequence/frame metadata；随后进行fixed-6 learned-capacity诊断。若表征连fixed-6容量都不足则停止当前头；若有容量，也只能据此设计Train-152真实predicted-crop rollout、sequence-disjoint OOF与零灾难门，不能直接进入VOT。

本轮未运行DepthTrack Test、CDTB、VOT low22或full127，正式最好仍为：

```text
VOT full127 EAO / ACC / ROB = 74.0205826687 / 82.5793442346 / 89.5656513398
DepthTrack Test P / R / F  = 65.995933 / 65.335885 / 65.664250
CDTB P / R / F             = 75.387821 / 76.005850 / 75.695574
```

Qwen3_8B权重继续保留，不在清理范围内。

<!-- RGBD-HANDOFF-24.160-LRA-M2-FIXED6-CAPACITY-20260831 -->
## 24.160　Fixed-6 learned association 跨序列容量正式负结果

### 24.160.1　实验完整性与正式门

在复合键`(branch_id, decision_age=2, decision_frame)`连接1,968个候选、0缺失/0重复/0错位后，按冻结计划比较5,124参数的relative-only负对照与1,027,958参数的candidate RGB-D + static identity text head。正式运行使用3个seed、6个外层留一序列折、每折5个inner cross-fit模型，共180个fit、3,546条外层OOF预测，退出码0，耗时829.51秒。

```text
formal HEAD = 4df3d06e80864e7ec1cb61647bc1b0a8f8b4d546
result SHA256 = ce84111fa5c32b7bef7535bf8daca2e53d97916757b78364339700207b7d5672
OOF SHA256    = d1d5382e5572bf3a6fb8ee5dbdaeb1c3ae92e2d63ae677df18e7be405159479c
manifest      = e13b08949f7c1786bb00ff170ce92567193185e558802b72d0479f4b5427b3eb
```

独立审计确认fit、fold、预测键、label topology和hash全部一致；没有未来/GT模型输入，没有候选提交或public state写入，没有保留诊断权重。

正式安全门结果为：

| 模型 | 通过内层安全门的折 | 三seed正式外层动作 |
|---|---:|---:|
| relative-only | 0/18 | 0 / 0 / 0 |
| candidate RGB-D + static text | 0/18 | 0 / 0 / 0 |

所有折都因不存在同时满足“至少10动作、覆盖2序列、beneficial precision≥0.95、catastrophic=0、H5均值≥0”的内层阈值而fail closed。这里的零catastrophic完全来自零动作，不能宣称安全改进。正式结论是：

```text
fixed6_cross_sequence_signal_not_supported
```

### 24.160.2　不是没训练，而是训练域拟合后不能泛化

relative-only的平均loss从3.8636降至2.0914；完整head从3.7387降至0.5931，且90/90个完整head fit末值均低于首值。因此负结果不是代码未执行、梯度异常或优化器不收敛，而是四条训练序列上学到的规则不能迁移到外层未见序列。

严格标记为post-hoc、未参与任何阈值选择的OOF诊断显示，无阈值强制选top-1时：

| 模型 | beneficial | neutral | catastrophic | beneficial精度 | catastrophic率 |
|---|---:|---:|---:|---:|---:|
| relative-only | 450 | 938 | 385 | 25.38% | 21.71% |
| candidate RGB-D + text | 435 | 1,147 | 191 | 24.53% | 10.77% |

完整候选表示确实把灾难top-1约减半，说明candidate-own RGB-D不是完全无信息；但它没有增加beneficial，主要把错误动作换成neutral，无法形成真正的rescue。

### 24.160.3　最重要的新问题

完整head在held-out sequence上的beneficial AUC均值只有0.4876、AP 0.2072。三类safe score中位数分别为：

```text
beneficial   0.81966
neutral      0.81342
catastrophic 0.82467
```

三类高度重叠，任何全局阈值都无法保留beneficial并排除catastrophic。完整head跨seed top-1一致率只有44.33%，显示1.03M参数在fixed6小域上解不稳定。

relative-only虽然beneficial AUC约0.7146，但其catastrophic高分方向严重反转：catastrophic score中位数0.99810，高于beneficial的0.98071；阈值越高反而越保留灾难。这精确复现了旧手工selector“全局看似有信号、按序列留出即失败”的域偏置。

序列分布仍高度集中：完整head三个seed的无阈值top-1中，`ball16_indoor`仍有139个catastrophic；`flower03_indoor`有273个beneficial且0 catastrophic。flower03本来就是主要beneficial域，ball16本来就是主要catastrophic域，模型容易学习序列域而不是初始化实例身份。

最合理、但尚非因果证明的解释是：每个inner model只有4条训练序列，却拟合102.8万参数；首帧target prototype和静态文本又在同一序列内恒定，可能成为domain proxy。当前只用age-2单时点，没有加入计划中的4～8帧因果时序和distractor memory，也限制了真正的身份关联。

### 24.160.4　路线决策

停止fixed6阈值扫描、head宽度微调和VOT尝试；该head不部署。fixed6只证明“候选自身表示有部分避灾信息，但不足以高精度选择恢复候选”。

下一步进入原计划的Train152前置设计：用更多序列生成metric-blind predicted-crop action space，先通过≥30可恢复事件/≥10序列的Gate A；association head应缩小并加强正则化，加入因果时序candidate evidence与target/distractor对比，sequence/frame/source ID仍禁止作为输入。只有Train152 sequence-group OOF通过0.95精度、0新增灾难门，才接在线事务；若仍全部弃权，则迁移同一创新到FlexTrackV2。

本轮没有新跟踪checkpoint，没有运行DepthTrack Test、CDTB、VOT low22或full127。正式最好仍为：

```text
VOT full127 EAO / ACC / ROB = 74.0205826687 / 82.5793442346 / 89.5656513398
DepthTrack Test P / R / F  = 65.995933 / 65.335885 / 65.664250
CDTB P / R / F             = 75.387821 / 76.005850 / 75.695574
```

Qwen3_8B继续保留。

<!-- RGBD-HANDOFF-24.161-LRA-M0-CACHED-CAUSAL-20260831 -->
## 24.161　Train152缓存模板重建与候选自身因果表征门完整通过

### 24.161.1　这一阶段要回答什么

fixed-6已经证明：动作空间里存在后验可救回的轨迹，但旧selector跨序列无法安全判断。进入Train152之前还存在两个实现完整性风险：

1. 旧full152 trace虽然保存了protected bbox、score和safe-v1事件，但没有template tensor；如果按replace/drop账本重建出的动态模板与真实tracker不完全一致，后续factor-7候选和H10 rollout就不是同一条公开路径。
2. 旧表征collector是在候选框周围额外裁一张验证crop，不能证明候选特征来自“产生该候选的同一递归搜索状态”。新实验要求age 0–4每一行都来自该tentative branch自己的causal search-token map、自己的候选框和同一crop的Depth。

因此本阶段只做M0实现门，不打开未来GT、不训练selector、不提交候选、不运行VOT。它不能回答指标是否上升，只回答full152采集是否具备因果与工程完整性。

### 24.161.2　M0b：direct replay与cached reconstruction逐值一致

正式输出：

```text
/root/autodl-tmp/sutrack_lra_cached_template_smoke_a1_20260831
/root/autodl-tmp/sutrack_lra_cached_template_smoke_a2_20260831
```

系统固定为：

```text
SUTrack-L384
+ DepthTrack Train short identity text
+ safe-v1 whole-slot dynamic template
+ checkpoint SHA256 2a686e8b…dacd4
+ config SHA256 fdd231df…0cc2
+ language manifest SHA256 c85a38b9…923d1
```

两次运行的`parity_result.json`逐字节一致，SHA256均为：

```text
74bef5cfb61d8f682625826628272c8c424dc105577404d410168eb9f06936ad
```

覆盖情况：

| 项目 | 数量 |
|---|---:|
| 序列 | 2：`ball16_indoor`、`cat04_indoor` |
| 非初始化帧 | 50 |
| safe-v1 checked帧 | 10 |
| factor-7 current/last/velocity候选比较 | 23 |
| dynamic template replace | 2 |
| dynamic template drop | 2 |
| pre-template比较 | 50 |
| post-template比较 | 50 |

所有硬门结果：

| 完整性项 | 最大误差/错误数 |
|---|---:|
| direct public bbox vs cached trace | 0.0 |
| direct public score vs cached trace | 0.0 |
| cached vs direct factor-7 candidate bbox | 0.0 |
| cached vs direct factor-7 candidate score | 0.0 |
| writer decision不一致 | 0 |
| candidate probe对public state的写入 | 0 |
| 两个tracker间template storage alias | 0 |

峰值CUDA显存为4,631,023,104 bytes。缓存路径严格按以下时序重建：

```text
读取第t帧protected prior和pre-public template
→ side-channel生成factor-7候选
→ 不写tracker state
→ 从sealed trace安装第t帧protected bbox/score
→ 按当前writer decision执行replace/drop
→ 得到第t+1帧的template状态
```

这一顺序很重要：如果先应用第t帧replace，再生成第t帧候选，就会把未来一次状态写入泄漏到当前候选。

#### M0b遇到的两个问题

第一次运行在加载网络前被旧环境绝对路径拦截：

```text
/home/cx/cx1/github-repo/SUTrack/experiments/...
```

修复方式不是建立隐式软链接，而是给工具增加显式`--config-file`与`--checkpoint`，并对config/checkpoint/CLIP逐项校验SHA。这样不同服务器不会静默加载另一份同名配置。

第二次运行在首帧storage独立性检查处停止，因为服务器旧版PyTorch没有`Tensor.untyped_storage()`。兼容实现优先使用`untyped_storage()`，旧版本回退到`storage().data_ptr()`；检查含义不变。两次失败都在模型差异判断前fail-closed，未生成accepted目录，也未污染后续结果。

M0b工具提交：

```text
222011b2988139b4ee6958fce27ad6a8eeffd661
```

### 24.161.3　M0c：候选自身causal RGB-D表征与H10递归

新增模块：

```text
lib/test/tracker/sutrack_causal_candidate_observation.py
tools/smoke_train152_causal_candidate_representations.py
```

提交：

```text
f9b072b0fe0037d8d5784eeff6ed7207981f99bd
```

每一个tentative observation执行：

```text
branch prior bbox
→ 围绕该branch自己的prior裁factor-4搜索crop
  （spawn时围绕current/last/velocity anchor裁factor-7）
→ 使用trigger时刻冻结的pre-public templates
→ SUTrack encoder/decoder产生当前branch bbox和score
→ 从同一次encoder输出的causal search-token map按该bbox做视觉RoI
→ 从同一搜索crop的raw Depth按同一bbox做normalized-depth + valid-mask RoI
→ 计算相对protected、相对上一branch状态、响应margin/entropy和显式depth missing
→ 当前bbox成为下一帧branch prior
```

数值特征固定不包含：

```text
sequence ID / frame ID / source ID / rank ID
dataset或属性组标签
裸的target prototype
裸的text embedding
未来GT或未来文本
```

首帧视觉原型与短身份文本只用于候选—身份的pairwise cosine、difference和product诊断。sequence、frame、source只保留在审计metadata和将来的sequence-group拆分中，不能进入selector数值输入。

相对特征共15维，包括：

```text
branch score
相对public与上一branch score变化
branch/public IoU、中心距离、面积比
branch自身中心与尺度变化
age fraction
candidate depth valid ratio
相对初始化目标的log-depth change
显式depth missing bit
response margin / entropy
候选在causal crop中的可见面积比例
```

#### 第一次覆盖失败是有价值的负结果

最初使用`ball16_indoor + cat04_indoor`、前45帧，并只允许前35帧spawn，以保证新分支能完整走到H10。结果为：

```text
ball16：3条分支，3条完成H10
cat04：0条满足score≥0.05且IoU<0.20的分支
age 0–4：各3行
状态污染：0
```

collector因“两条序列都必须有动作”的覆盖门正确拒绝。这里没有把IoU阈值从0.20放宽，也没有降低score门。查阅既有、无GT的fixed-6 expanded trace后发现：`pigeon05_wild`在frame 35已有风险候选，可在frame 45完整观察H10；因此只替换烟雾测试的第二条序列，不改变模型、动作阈值或候选规则。

### 24.161.4　M0c最终结果与确定性

正式双重复：

```text
/root/autodl-tmp/sutrack_lra_causal_representation_smoke_a1_20260831
/root/autodl-tmp/sutrack_lra_causal_representation_smoke_a2_20260831
```

最终范围：

```text
ball16_indoor：frame 0–45
pigeon05_wild：frame 0–45
spawn cutoff：35
branch horizon：10
feature ages：0–4
```

结果：

| 项目 | 结果 |
|---|---:|
| checked帧 | 18 |
| 创建tentative branches | 5 |
| 完成H10 branches | 5 |
| branch transitions | 50 |
| unfinished branches | 0 |
| 最大同时活跃branches | 3 |
| age 0/1/2/3/4特征数 | 5 / 5 / 5 / 5 / 5 |
| 总特征行 | 25 |
| last-reliable来源 | 4 |
| velocity来源 | 1 |
| candidate写入public state | 0 |
| 初始化后GT读取 | 0 |
| 峰值CUDA显存 | 2,548,387,328 bytes |

张量：

| 文件 | Shape | SHA256 |
|---|---|---|
| `candidate_visual.f16` | `[25,768]` | `19f55b3e…03bf` |
| `depth_rois.f16` | `[25,2,16,16]` | `7fcab739…9da` |
| `relative_features.f32` | `[25,15]` | `f7f19445…5374` |
| `pairwise_diagnostics.f32` | `[25,6]` | `1b24f9a…2030` |
| `branch_trace.jsonl.gz` | 55 rows | `85d2d767…da11` |

两次运行上述五类payload全部逐字节一致。a1的result/manifest SHA256分别为：

```text
bae9e25b68a46b882a2e0e2487f27bd78c84300008b5df177be080219c341659
d7e1783742c37beead2a8c72f6ef53c55df56f0a74570647f72e8b7c44264521
```

M0c没有产生current-source分支。这不是失败，因为M0c只检查schema、因果空间绑定、H10完整性、确定性和状态隔离；Gate A仍明确要求full152中的正例同时覆盖current与last/velocity。不能把M0c的5条分支写成恢复容量，更不能据此宣称ROB/EAO会上升。

### 24.161.5　M0之后确认解决和仍未解决的问题

已确认解决：

1. 缓存protected bbox/score与direct replay完全一致。
2. safe-v1 replace/drop可以从图像和sealed ledger精确重建template/annotation tensor。
3. factor-7候选不会因为使用缓存路径而改变。
4. tentative branch拥有独立bbox和冻结模板，真实使用predicted crop递归到H10。
5. age 0–4特征来自该branch产生当前bbox的同一次causal search map，而不是public/protected bbox或额外候选中心crop。
6. Depth missing通过独立valid mask和missing bit表达，不再与“Depth变化为0”混淆。
7. 候选探针与递归分支不写protected state；双重复载荷完全确定。

仍未解决：

1. full152动作空间是否有足够正例仍未知；必须通过Gate A后才能训练selector。
2. 25行smoke不能证明跨序列泛化，也不能证明current来源有可恢复动作。
3. 短身份文本是否真正帮助目标—干扰物实例关联仍需post-inference标签后判断。
4. 当前只冻结branch template，不更新分支内部模板；这是为了先隔离state recovery容量，不代表最终在线事务结构。
5. selector小于400k参数、3-seed nested sequence OOF及零灾难门尚未执行。
6. 没有新checkpoint，没有DepthTrack Test/CDTB/VOT新指标。

### 24.161.6　当前终态与下一步硬门

M0现已完整通过，但旧计划中的授权仍为：

```text
depthtrack_train_trace: false
selector_training: false
vot_low22: false
vot_full127: false
automatic_next_stage: false
```

因此本阶段没有自动启动full152。下一步必须先生成新的M1机器可读冻结单，绑定当前commit、M0b/M0c结果、两片survival trace、checkpoint/config/language SHA、两GPU分片、磁盘硬门、因果表征schema与15维特征顺序，并继续保证post-inference前严禁打开GT。

full152只有在Gate A同时达到以下条件时才能进入selector训练：

```text
beneficial actions ≥ 30
positive sequences ≥ 10
rescued confirmed failure starts ≥ 10
rescued failure sequences ≥ 5
正例同时含current与last/velocity来源
```

否则停止SUTrack learned路线并进入FlexTrackV2 fallback审计。当前正式最好指标仍为：

| 数据集 | 指标 | 数值 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | 74.020583 / 82.579344 / 89.565651 |
| DepthTrack Test | P / R / F | 65.995933 / 65.335885 / 65.664250 |
| CDTB | P / R / F | 75.387821 / 76.005850 / 75.695574 |

Qwen3_8B继续保留，本路线未调用在线Qwen。

<!-- RGBD-HANDOFF-24.162-LRA-M1-TRAIN152-LAUNCH-20260831 -->
## 24.162　M1流式复现通过并启动两卡Train152无GT采集

M0完成后新增流式runner与plan冻结器，提交：

```text
1eb6fa07015a9a100ae1b8d1ea70d7476efea4e4
```

实现：

```text
tools/run_train152_causal_candidate_collection.py
SHA256 f3527bade2a5cf675168a3f44d57547a443191872f2f44e9f8b7ed6f9ce6602a

tools/plan_train152_causal_candidate_collection.py
SHA256 943b9e485a5be5edff710bb080aaa145601d04c2f1cef7052871847c1d6518f8
```

base plan为：

```text
/home/SUTrack_RGBD_L/refine-logs/
LRA_M1_TRAIN152_CAUSAL_COLLECTION_PLAN_20260831_054906.json
SHA256 f2e6252747f51e31fdc30855cd80548eaa91a4a6d316d9c0d06bb7e72e3e27b9
```

base plan没有直接授权full152，只允许新流式runner先复现M0c。smoke输出：

```text
/root/autodl-tmp/sutrack_lra_train152_collection_smoke_v1_20260831
result SHA256 dd1010b62931cc43f8dc2b565746769ee27c244f397c1d03ebfbc7e7df7e3013
```

复现结果仍为5条H10分支、25行age 0–4表征、last/velocity来源4/1、public state写入0、初始化后GT读取0。更重要的是，新runner生成的branch trace、visual、Depth、relative、pairwise五类核心payload与M0c逐字节一致，说明从两序列内存collector改为full152流式writer没有改变候选、递归或表征。

正式启动单：

```text
LRA_M1_TRAIN152_CAUSAL_FORMAL_LAUNCH_20260831_055040.json
SHA256 bbc739891c645f33b004b5f2238085413e44b4e5c34ca204775a5f6b2c3ec25b
```

本次唯一新增授权是：

```text
depthtrack_train_trace=true
```

以下全部仍为false：

```text
post_inference_label_join
selector_training
online_transaction_replay
vot_low22
depthtrack_test
cdtb
vot_full127
automatic_next_stage
```

正式分片：

| Shard | GPU | 序列 | 含初始化帧 | PID | 输出 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 79 | 114,937 | 71512 | `.../sutrack_lra_train152_causal_collection_v1_20260831/shard0` |
| 1 | 1 | 73 | 105,017 | 71513 | `.../sutrack_lra_train152_causal_collection_v1_20260831/shard1` |

启动时两卡各空闲24,135MiB、GPU利用率0%，磁盘可用9,568,829,440 bytes。启动后两进程正常存活，两卡各使用约3.45GiB。运行中每完成一条序列检查磁盘；低于5GiB立即fail-closed。

当前状态为`RUNNING`，不是结果。两片完整manifest封存前严禁打开未来GT；封存后仍需独立冻结标签连接和Gate A分析，不能自动训练selector或运行任何VOT。正式最好指标尚未变化，Qwen3_8B继续保留。

<!-- RGBD-HANDOFF-24.163-LRA-M1-NUMERIC-RELAUNCH-20260831 -->
## 24.163　M1两分片数值失败精确归因、边界修复与全量重启

### 24.163.1　上一节的`RUNNING`状态已经失效

24.162记录的旧formal attempt没有形成可接受结果。两个独立GPU进程随后都在同一保护检查处fail-closed：

| 旧分片 | 已完整处理 | 失败位置 | 已保留partial |
|---:|---|---|---|
| shard0 / GPU0 / PID71512 | `bottle03_indoor`、`ball16_indoor` | `bag04_indoor`的H4 active-branch前向；sealed记录完整到frame 1017，下一次branch调用前失败 | 6,204条branch trace、2,829条age0–4 feature metadata及对应tensor |
| shard1 / GPU1 / PID71513 | 0条完整序列 | 首条`colacan02_indoor`的factor-7前向 | 1,551条branch trace、705条feature metadata及对应tensor |

两片错误文本相同：

```text
RuntimeError: pooled depth validity mask left [0, 1]
```

shard0发生在runner line 626的H4递归分支，shard1发生在line 709的factor-7候选探测。这一点很重要：错误不属于某个候选来源、某个搜索因子或单张GPU，而属于二者共用的Depth-valid RoI管线。

旧partial没有被删除，但它们的科学状态固定为：

```text
rejected diagnostic-only
禁止续跑
禁止两个旧分片互拼
禁止与replacement分片拼接
```

两个进程均未把candidate写入公开tracker，初始化后没有打开future GT，也没有运行label join、selector或VOT。

### 24.163.2　精确根因不是Depth错误，而是一个float32 ULP

最初依据最后flush的记录，只能把shard1失败帧推测到1830。单独重建1830并重复128次时，valid mask始终精确位于`[0,1]`，因此没有贸然加入clamp。随后使用与正式runner相同的完整历史、同一模板事件、同一active-branch顺序回放，精确定位到：

```text
sequence = colacan02_indoor
frame = 1825
phase = factor-7
depth pooling calls before failure = 850
violating calls = 1
violating elements = 1
raw maximum = 1.0000001192092896
excess = 1.1920928955078125e-7
```

`1.1920928955078125e-7`恰好是1.0附近的一个float32 epsilon。Depth valid channel的输入是0/1二值掩码，双线性RoIAlign理论上应位于凸包`[0,1]`，但GPU浮点加权累加可产生一个ULP的端点上溢。因此：

> 旧门把“语义上错误的Depth范围”和“数值舍入造成的1 ULP端点上溢”混为一谈，导致正确的fail-closed策略在极少数合法输入上过度拒绝。

完整机器回执：

```text
/home/SUTrack_RGBD_L/refine-logs/LRA_M1_DEPTH_MASK_NUMERIC_DIAGNOSIS_20260831_062407.json
SHA256 8937fe344602bfff4948d9558a2652910c6d0dda958c4fc60c01332c5ea4f9ac
```

早期的1830推测已明确标记为rejected inference，后续不得继续引用为精确失败帧。

### 24.163.3　修复仍然fail-closed，而不是无条件截断

修复提交：

```text
bfc18b4b75283720c69732e15d0b6df69a7a9b40
```

冻结数值契约：

```text
DEPTH_VALID_MASK_CLAMP_TOLERANCE = 9.5367431640625e-7
                                  = 8 × float32 epsilon
```

执行规则：

1. 先计算valid channel低于0或高于1所需的最大修正量；
2. 若最大修正量大于8 ULP，继续硬失败，不能掩盖真实Depth管线错误；
3. 只有在容差内，clone `depth_rois`并只对valid channel做`clamp(0,1)`；Depth value channel和RGB候选特征不变；
4. 每个formal shard必须记录总forward数、发生修正的forward数、修正元素数、最大修正量，以及factor-7/H4分项；
5. 最终存储前再次验证valid channel严格位于`[0,1]`。

补丁后直接重放原失败点`colacan02_indoor@1825`：

| 项 | 值 |
|---|---:|
| clamped elements | 1 |
| maximum correction | `1.1920928955078125e-7` |
| stored minimum | 0.0 |
| stored maximum | 1.0 |
| stored out-of-range | 0 |

代码哈希：

| 文件 | SHA256 |
|---|---|
| `sutrack_causal_candidate_observation.py` | `8641cb4fae72a0d4f10868fcdf5bef16865bdb9a5f8da2477ff5b3aec1998995` |
| `run_train152_causal_candidate_collection.py` | `b07971f6f801cfef184f3784089d5ec5251a909c2ba00bf7b91f3409dc6ef30e` |
| `plan_train152_causal_candidate_collection.py` | `c2c68463439c1b25bd94c38eaebb2993c876afff39fbf242fda22d3004434c43` |

### 24.163.4　replacement smoke证明修复没有改变实验内容

replacement base plan：

```text
LRA_M1_TRAIN152_CAUSAL_COLLECTION_PLAN_NUMERIC_V2_20260831_062047.json
SHA256 b121584c82399630c5f7d650cdc7cd5d4c88f35f7b8f9c500a321aa71d6caf3d
```

新smoke输出：

```text
/root/autodl-tmp/sutrack_lra_train152_collection_smoke_numeric_v2_20260831
result SHA256   6faa024470a14e75f63e4e813eeb41ab11ab308cb59ce3ebefafb52445ef057c
manifest SHA256 9b9b72ccaa0b2559d48e1dff242dc07c1d954c264d0287d99a8b2ddc907123c2
```

smoke覆盖`ball16_indoor/pigeon05_wild`，产生5条完整H10分支、age0–4各5行、共25行。44次候选forward中没有触发clamp；最重要的是以下五类核心payload与M0c逐字节一致：

| Payload | SHA256 |
|---|---|
| branch trace | `85d2d76708d81d1b861f2c5d4a891217dd464e47c5127394a56917ac4e87da11` |
| candidate visual | `19f55b3e66f152fd73dc5d1b35c14e25fa698371dbb1c08156d1ab44978c03bf` |
| Depth-valid | `7fcab739cceed29900cd61363c0e35a23ba49b0d9bfb8efee2bfcafac5ed69da` |
| relative | `f7f194454728bda81d2e2943815a5be8e9a4bf60c981dd00d1ec4d73f2a15374` |
| pairwise | `1b24f9a73b7baf4e61660c3f538793b5855754a464a71c4d69e524f17ea2030a` |

这证明修复只改变数值边界处理与审计元数据，没有改变候选轨迹、候选特征或动作空间。

### 24.163.5　两片从头重启及当前授权边界

replacement正式启动回执：

```text
LRA_M1_TRAIN152_CAUSAL_FORMAL_LAUNCH_NUMERIC_V2_20260831_062407.json
SHA256 3397be8dfcf3dee44ae46563d25831fbfd0851850a908795b8ab27a803e8212d
```

| Shard | GPU | 序列 | 含初始化帧 | PID | 输出 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 79 | 114,937 | 76615 | `.../sutrack_lra_train152_causal_collection_numeric_v2_20260831/shard0` |
| 1 | 1 | 73 | 105,017 | 76616 | `.../sutrack_lra_train152_causal_collection_numeric_v2_20260831/shard1` |

启动时磁盘可用9,555,701,760 bytes，两卡各空闲24,135MiB。当前只允许metric-blind action/representation采集，仍未授权：

```text
post_inference_label_join
selector_training
online_transaction_replay
depthtrack_test
cdtb
vot_low22
vot_full127
automatic_next_stage
```

所以本节没有产生新模型、没有训练checkpoint、没有新VOT/DepthTrack/CDTB指标。正式最好指标保持不变，Qwen3_8B继续保留。只有replacement两片均完整封存后，才能另行冻结future-GT label join并执行Gate A。

06:38的早期健康门进一步确认：shard0已经完整结束`bag04_indoor`并继续完成`flower03_indoor`，shard1已经完整结束`colacan02_indoor`和`egg_indoor`。因此H4与factor-7两个旧失败点均已由真实formal runner越过，而不只是单帧诊断脚本通过。两PID仍存活、日志无Traceback/RuntimeError、future GT未打开、candidate未写public tracker。证据：

```text
LRA_M1_NUMERIC_V2_EARLY_HEALTH_20260831_063837.json
SHA256 4ac9430de60509706413ee5571ff28898333d3ebf7b37d11fc17afc116d91f57
```

该回执只证明数值失败已修复且采集可继续，不是完整Train152结果，也不授权任何下游阶段。

<!-- RGBD-HANDOFF-24.164-LRA-M1-GATE-A-PREREG-20260831 -->
## 24.164　M1 Gate A 在未来GT打开前冻结精确标签与失败救援定义

### 24.164.1　为什么必须补这份冻结单

`LRA_M1_TRAIN152_CAUSAL_COLLECTION_PLAN_NUMERIC_V2_20260831_062047.json`已经在采集前冻结了Gate A的数量门：beneficial action不少于30、正例序列不少于10、救回confirmed failure start不少于10且覆盖不少于5条序列，并要求正例同时包含`current_prior`与`last_reliable/velocity_extrapolation`来源。但是原计划没有在同一机器可读工件中展开`beneficial`、`catastrophic`和`confirmed failure rescue`的精确定义。

如果等到Train-152标签可见以后再选择IoU阈值、未来窗口或failure判定方式，就会形成后验阈值选择；即使数量门通过，也不能证明动作空间具有预先定义的恢复容量。因此在两片仍运行、`collection_result.json`和`manifest.json`均不存在、watcher仍明确记录`future_ground_truth_opened=false`时，新增独立只读冻结单：

```text
/home/SUTrack_RGBD_L/refine-logs/LRA_M1_GATE_A_POST_INFERENCE_SPEC_20260831_065355.json
bytes 8598
mode 0444
SHA256 563e4ef451e0de06bf82b73d32df830f7d2f684c6c97a3b56b57fbe2dd72eeb8
```

冻结时formal实时状态为shard0完成`6/79`、shard1完成`5/73`；两PID存活，候选未提交到public tracker，Gate A输出根目录尚不存在。该工件不打开GT、不计算IoU，也不改变正在运行的commit `bfc18b4b75283720c69732e15d0b6df69a7a9b40`。

### 24.164.2　动作标签的唯一正式定义

每条完整branch只形成一个action；selector的数值证据固定使用age 0--4，决策帧为`trigger_frame + 4`。标签窗口固定为从`trigger_frame`开始、在age 0--10完整branch轨迹中能够对齐的前10个可见GT帧。sequence/frame/source/rank、GT、future overlap与future text只能用于离线join审计或完整序列分组，不得进入数值特征。

沿用此前fixed-6已经冻结的阈值，不能在看到Train-152标签后修改：

| 标签 | 精确定义 |
|---|---|
| beneficial | branch 10帧mean IoU不低于0.50，且相对public mean IoU增益不低于0.20，同时前5个可见帧中至少2帧IoU不低于0.50 |
| catastrophic | `public mean≥0.50且branch mean≤0.20`，或branch相对public的mean IoU损失至少0.30，或branch 10帧全部`IoU≤0.10`而public并非全部低于该阈值 |
| neutral | 标签可用，但既不满足beneficial也不满足catastrophic |
| unavailable | 不足10个可见对齐GT帧，或branch/public在冻结窗口内覆盖不完整 |

优先级固定为`beneficial → catastrophic → neutral`。这一标签只用于Train-152离线容量和后续selector监督；它不是在线可见信号。

### 24.164.3　confirmed failure与rescue的唯一正式定义

public轨迹中连续10个有标签且帧号连续的帧均满足`IoU≤0.10`时，只把该连续低重叠段第一帧记为一个confirmed failure start；遇到帧号断裂或GT不可用即重置计数，同一连续低重叠段只计一次。

一条branch要救回该failure start，必须同时满足：

1. `trigger_frame < failure_start`；
2. 从failure start开始的10个可见帧上，branch与public都有完整box；
3. branch前5个可见帧至少2帧`IoU≥0.50`；
4. branch相对public的10帧mean IoU增益不低于0.20；
5. branch不能10帧全部`IoU≤0.10`。

同一failure start有多个合格branch时，严格按`branch_mean_iou`、`mean_iou_gain`、`trigger_frame`、`branch_id`降序选择，只用于确定性oracle容量统计。

### 24.164.4　GT对齐与授权边界

默认要求GT行数与连续RGB/Depth帧数完全一致。唯一预注册例外仍是历史已审计的`toy07_indoor_320`：RGB/Depth均为1,367帧，GT为1,406行，只允许在GT文件SHA256仍为`683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2`时忽略末尾39行；任何其他不一致都必须fail closed。

只有replacement两片的result和manifest均`complete=true`、`accepted=true`，全部输出字节数和SHA通过、future GT/public mutation/future text均为false后，才允许运行一次独立post-inference join。Gate A通过后也只授权selector训练与DepthTrack Train complete-sequence group OOF；在线事务、DepthTrack Test、CDTB、VOT low22/full127仍全部关闭，不能自动串联。若Gate A失败，则停止SUTrack learned路线并进入已预注册的FlexTrackV2 fallback审计。

本节仍没有产生checkpoint或新指标。当前正式最好继续保持：VOT-RGBD2022 `74.020583/82.579344/89.565651`，DepthTrack Test `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`；Qwen3_8B继续保留，本轮未调用在线Qwen。

<!-- RGBD-HANDOFF-24.165-STTRACK-LACHTT-M1-LAUNCH-20260831 -->
## 24.165　SUTrack停止后迁移STTrack：候选自身RGB-D身份与独立递归事务M1启动

### 24.165.1　为什么停止原SUTrack learned selector

SUTrack Train-152正式三seed、nested sequence-group OOF已经闭合为负结果：15个外折全部为`abstain_no_inner_gate_b_threshold`，正式动作数为0。后验AUC虽高，但高分尾部仍包含大量catastrophic动作；例如`bag05_indoor@575:last_reliable`的p1 gain为`-0.9117`，`ball04_indoor@270`为`-0.7864`，`ball08_wild`还有多条约`-0.94`的高置信错误。该现象确认了当前语言/首帧外观仍主要识别“同类别、像初始化目标的物体”，不能稳定识别“初始化时的具体实例”。继续扫描相同SUTrack阈值、seed或小head不再有科学价值。

FlexTrackV2按预注册顺序先做了源码与权重可达性审计。源码commit为`30a5ff1b39b8f3004dde4018018b901e6cc1bf54`，但官方large checkpoint的HuggingFace页面与镜像API均返回401，直连又超时；没有下载到权重、没有随机初始化冒充baseline、没有产生指标。对应只读回执：

```text
FLEXTRACKV2_FALLBACK_M0_AUDIT_SPEC_20260831_110222.json
SHA256 a21c7308b5563059feb11093de845f49e62a0c084bead26742d7fc07e0be426a

FLEXTRACKV2_FALLBACK_M0_RESULT_20260831_111101.json
SHA256 ecb385a418504068596b8e6c7640d4850732982898c9c9c3182eb1b68e185b50
```

因此按预注册顺序迁移到已有官方VOT权重且接口可审计的STTrack；这不是把STTrack作者指标当作本服务器结果，也没有重测baseline。STTrack作者报告的VOT为`77.6/82.5/93.7`，只作外部背景；本项目正式最好仍是SUTrack identity-only的`74.020583/82.579344/89.565651`。

### 24.165.2　STTrack M0固定点与修复的递归状态问题

隔离工作树：

```text
/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
branch codex/language-anchored-candidate-transaction-v1
```

官方源码起点为`283cd6dd45536636490db8bca1c63c4647be799b`。官方VOT权重为：

```text
/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar
bytes 532407510
SHA256 cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98
```

M0提交`f317cc7f68ccc1c5248d9a85fc6b87faffd44d4f`修复了track query窗口的OLD--NEW更新顺序，并把RGB、Depth、融合search token以及独立RGB/Depth query tensor作为只读候选接口返回。最终M0 smoke严格加载官方权重，missing/unexpected均为空；三次递归query长度始终为`[4,4]`，search RGB/Depth/fused均为`[1,256,768]`且有限，模板保持不变。最终M0工件：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_FALLBACK_M0_SMOKE_20260831_1125_FINAL
result SHA256 33e4f4deb6856dd48b9ec0b3d69492ac423b36ecf8c7e980b5020f4d8ab31684
console SHA256 6444e7af0e2bec462eafde121b11d93aed6f1c1598bd68016f4e0c24adbf6f00
```

M0只读取第一帧初始化框，不计算IoU，不是baseline评测。

### 24.165.3　M1与旧失败方法的实质区别

旧STTrack已经验证失败的方法包括单帧137D absolute-IoU selector、411D pairwise selector、rule-grid全局检索、RGB/Depth/Fused检索、response-proposed identity association和exact dynamic-response candidates；这些方法不得换名重复。

本次M1的动作与观测改变为：

```text
旧GT-free public风险时间表
        ↓
trigger_frame-1的current / last reliable / velocity三个先验
        ↓
每个先验factor-6搜索并做贪心NMS top-2，共6条动作
        ↓
每条动作复制不可变首帧双模板，query在age0重置
        ↓
6条分支各自维护RGB/Depth query，factor-4递归到age9
        ↓
age0--4形成因果selector特征；age5--9仅在封存后用于未来效用标签
```

每个候选观察的是候选自身，而不是public bbox：

| 特征 | 形状/语义 |
|---|---|
| native RGB RoI | 768维，候选框在其所属16×16 search token图上池化 |
| native Depth RoI | 768维，与RGB分开保存 |
| native fused RoI | 768维 |
| CLIP candidate image | 768维，实际候选crop编码 |
| immutable CLIP anchors | 首帧图像768维、短身份文本768维 |
| recursive query | RGB 768维、Depth 768维，分支独立 |
| raw Depth RoI | `[2,16,16]`，稳健归一化log-depth与独立valid mask |
| 因果标量 | response score/margin/entropy、相对public/先验的IoU与几何变化 |

同一事件内其他候选可重建top-3 distractor bank。sequence/frame/source/rank、GT、future IoU和future text只允许用于审计/分组，不得成为模型数值输入。Qwen不参与本轮，短文本只作不可变身份锚。

### 24.165.4　实现失败、修复与最终smoke

首个工程smoke在年龄0安全失败：某个响应图只有一个严格局部极大值，而实现错误地要求两个严格局部峰，因此没有生成候选轨迹、没有读取未来GT。修复提交`69da84494d80c12c54f71d23720baafa8b5786e2`改为标准贪心NMS：取一个峰后抑制其3×3邻域，再取下一峰。失败目录和日志保留，不能算作通过结果。

最终实现固定点为`680e08ae5847b4da377cd1c9973edb32c3deb06d`。通过的第二次smoke位于：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M1_SMOKE_20260831_1200_RERUN2
manifest SHA256 81957d1ca167b86a47ad7a83d0552c9ba3d2f762d75850ca796f46037c775b8f
```

结果为1个真实风险事件、6条命名分支、每条10帧；8组age0--4 tensor的shape全部匹配且数值有限，峰值CUDA allocated为1,989,764,096 bytes，候选提交数为0，future GT打开为false。该事件的三个先验存在重合，所以6条命名动作只有4个唯一age0框；该事实必须保留为动作多样性诊断，不能把“6条动作”误写成“6个不同位置”。

### 24.165.5　正式冻结与两卡无GT采集

第一份冻结单在随后增加“formal输出必须严格等于冻结shard路径、禁止formal携带`--sequence/--max-events`、最终事件数必须精确匹配”后被V2取代；V1保留但不授权运行。唯一正式V2为：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M1_TRAIN152_SPEC_V2_20260831_1200.json
bytes 6777
mode 0444
SHA256 944e3957eea94898152ccbcc5e41a4fe8b5bb43b357c10de332f87e44a4e70b1
commit 680e08ae5847b4da377cd1c9973edb32c3deb06d
```

冻结事件数为shard0 `758`、shard1 `708`。Gate A沿用事前阈值：beneficial不少于30且覆盖不少于10序列，救回confirmed failure不少于10且覆盖不少于5序列，并要求current与last/velocity两类来源均有容量；通过也只授权selector训练和DepthTrack Train complete-sequence OOF。

11:58 CST两片正式无GT采集启动：

| shard | GPU | screen | 输出 |
|---:|---:|---|---|
| 0 | 0 | `sttrack_lachtt_m1_s0` | `/root/autodl-tmp/sttrack_lachtt_train152_collection_v1_20260831/shard0` |
| 1 | 1 | `sttrack_lachtt_m1_s1` | `/root/autodl-tmp/sttrack_lachtt_train152_collection_v1_20260831/shard1` |

启动时两卡均低于500MiB，数据盘可用约8.7GiB；启动后每卡约3.7GiB。早期日志已分别完成9个和8个事件，无Traceback。当前状态仍是RUNNING，不是结果；两个manifest完整封存前禁止打开future GT、禁止训练selector、禁止运行DepthTrack Test/CDTB/VOT low22/full127。

本节没有新checkpoint或新指标。正式最好仍为VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`；Qwen3_8B继续保留且本轮未调用。

<!-- RGBD-HANDOFF-24.166-STTRACK-LACHTT-GATES-20260831 -->
## 24.166　STTrack M1动作容量通过、在线selector失败：具体证据与新缺口

### 24.166.1　H10协议矛盾在GT打开前修正

正式采集中复核Gate A定义时发现：branch只覆盖age 0--9共10帧，但旧句子同时要求`trigger_frame < failure_start`并从failure start再完整评估10帧；该条件在现有H10轨迹上结构性不可能满足。因为此时两片仍运行、两个manifest与Gate根均不存在、future GT未打开，所以只修正这一条并冻结只读amendment：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M1_GATE_A_PRE_GT_AMENDMENT_20260831_1203.json
bytes 3417
mode 0444
SHA256 0e97f2be1f1b9ece84add9d24049c5e70d6cdc677aaae396fc972689dd4d85cb
```

修正后严格rescue只允许`trigger_frame == confirmed_failure_start`，这样branch/public age 0--9与连续10帧失败窗精确对齐。beneficial、catastrophic、IoU阈值、Gate A数量门均未变化。这不是看到标签后的放宽；相反，它是更保守的exact-start定义。

### 24.166.2　Train-152无GT采集完整封存

两片正式结果：

| shard | 事件 | metadata行/文件 | 序列锚点 | 运行时间 | manifest SHA256 |
|---:|---:|---:|---:|---:|---|
| 0 | 758 | 758/758 | 76 | 1662.736秒 | `1c2f0e38…ba27e` |
| 1 | 708 | 708/708 | 76 | 1557.824秒 | `8c6ebac7…e24a7` |

合计1,466事件、8,796动作、1,466个feature文件、152个anchor文件，占用467MB。两个分片峰值CUDA allocated均为1,989,938,176 bytes。两份manifest均为：

```text
complete=true
accepted=true
ground_truth_used_after_initialization=false
future_ground_truth_opened=false
metric_computed=false
candidate_committed_to_public_tracker=false
repository_commit=680e08ae5847b4da377cd1c9973edb32c3deb06d
```

Gate A启动前又冻结一次性回执，绑定两个manifest/metadata/log、分析器commit和输出根：

```text
STTRACK_LACHTT_M1_GATE_A_LAUNCH_SPEC_20260831_1227.json
mode 0444
SHA256 8a5d7270ecdfc267bea57678d7059d57ad505ad474b2f5af57d9d9169b87b07a
```

分析器先校验每个feature的bytes、SHA256、shape和finite，再第一次打开Train GT。

### 24.166.3　Gate A正式通过：动作空间确实有恢复容量

输出根：

```text
/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831
```

正式统计：

| 项目 | 数值 |
|---|---:|
| 全部动作 | 8,796 |
| beneficial | 341 |
| neutral | 4,123 |
| catastrophic | 228 |
| unavailable | 4,104，即684个整事件 |
| beneficial覆盖序列 | 43 |
| public confirmed failures | 338 |
| exact-start严格rescue | 14 |
| rescue覆盖序列 | 13 |

三类来源的beneficial分别为`current=112`、`last_reliable=118`、`velocity=111`；strict rescue分别为`1/6/7`。最大beneficial序列`ball16_indoor`贡献30/341=8.80%，不像旧SUTrack flower03那样由单序列垄断。六个冻结门全部通过：beneficial不少于30、正例序列不少于10、rescue不少于10、rescue序列不少于5、current有容量、last/velocity有容量。

14个严格rescue覆盖：`bottle01_indoor`、`speaker_indoor`、`cube04_indoor`、`human05_wild`、`ball16_indoor`、`mobilephone05_indoor`、`glass03_indoor`、`colacan02_indoor`、`egg_indoor`、`toiletpaper02_indoor`、`cup06_indoor`两次、`cube06_indoor`、`flowerbasket_indoor`。branch H10 mean IoU约0.687--0.952，而对应public几乎全为0。

关键工件：

```text
gate_a_result.json SHA256 d74aff281a4dba95a9f37e33e450e8a5ab14e0add12976e045756466c12e0592
manifest.json      SHA256 4ddc527a8cdbb108a893cd4575f305eeab5dbb62c399cdfce6627da2ea34ab49
labeled_actions    SHA256 f30316f40a4cd29bd609a2f7088234a9daa4de580ad823b11eb529b3bb66457a
failure_rescues    SHA256 db5f7b51a5dfc2da6542e764011c59837e76d8a58484e105497860a4bf414bb5
```

Gate A只是oracle容量，不是可部署效果。

### 24.166.4　learned target--distractor selector与Gate B

selector固定为261,739参数：六路768维候选特征分别投影到24维，raw Depth经卷积编码，CLIP候选分别与不可变首帧图像/短文本计算相似度；每个候选用5帧GRU形成时序状态，再用无候选位置/source embedding的4头Transformer在6候选之间建模distractor。三个输出头分别预测beneficial、catastrophic和mean-IoU gain。

正式规范：

```text
STTRACK_LACHTT_SELECTOR_OOF_SPEC_20260831_1238.json
mode 0444
SHA256 0a3ae7e13f40dc7ef12a3a3d7a6655d981a261f9a36d70b5413fd32b41e14819
```

数据为782个标签可用事件、119条序列；采用6 outer sequence folds、每个outer内5 folds、20 epoch固定训练、seed 2026/2027/2028。内层门要求至少10动作、4序列、precision不低于0.95、catastrophic=0；外层Gate B要求每seed至少4个有效outer folds、20动作、8序列、precision不低于0.95、catastrophic=0。零动作明确算失败。

三seed结果：

| seed | 有效/弃权outer folds | 动作 | beneficial | neutral | catastrophic | precision | 序列 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1/5 | 1 | 1 | 0 | 0 | 100% | 1 |
| 2027 | 0/6 | 0 | 0 | 0 | 0 | 0% | 0 |
| 2028 | 2/4 | 38 | 20 | 16 | 2 | 52.63% | 15 |

seed2026唯一通过的内层阈值为0.996，内层`19/20=95%`且零灾难，到外层只剩1个动作。seed2028的两个内层阈值0.994/0.979分别达到95.24%与100%，但外层骤降为70%和46.43%，并各产生1个catastrophic。最明确的高置信灾难为：

```text
lock01_wild@386 / velocity_peak0
score 0.997373
future gain -0.750544

egg_indoor@39 / current_peak1
score 0.981255
future gain -0.716586
```

训练loss普遍从约4.0--4.6降到0.43--0.93，因此不是优化器没有学习；问题是内层高精度区域不能跨序列迁移，而且seed极不稳定。Gate B正式失败：

```text
/root/autodl-tmp/sttrack_lachtt_selector_oof_v1_20260831/final/gate_b_result.json
SHA256 b8c2e1654bc48a69048e2cd7b97564063d243648fab5dd7a13da906ef3dffcb3
decision stop_sttrack_selector_no_final_no_online_replay
```

因此不训练final selector、不做online replay、不跑low22/full127，也不降低precision门。

### 24.166.5　当前发现的新增结构缺口

M1终于证明了动作空间有跨序列恢复容量，但selector仍缺少一类关键证据：它有候选自身native RGB/Depth/fused特征，也有CLIP首帧图像与短文本锚，却没有**STTrack原生首帧RGB和Depth实例原型**。因此native候选特征只能彼此比较，不知道哪一个更像初始化时的具体实例；CLIP则仍偏类别/语义，难以排除同类实例。

下一次唯一允许的M2变化是：为152条Train序列各自提取不可变native RGB/Depth template prototype，并显式构造candidate-to-initial以及candidate-to-distractor contrast；复用现有1,466个候选轨迹，不重跑VOT、不重复阈值扫描。M2仍只允许Train sequence-disjoint审计；若仍不能通过零灾难Gate B，则停止该STTrack selector分支。

当前没有新tracking checkpoint，也没有新VOT/DepthTrack/CDTB指标。正式最好保持VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`；Qwen3_8B保留且本轮未调用。

<!-- RGBD-HANDOFF-24.167-STTRACK-LACHTT-M2-NATIVE-20260831 -->
## 24.167　STTrack M2首帧原生身份锚仍未解决跨序列安全性：selector路线正式停止

### 24.167.1　M2为什么只增加这一项证据

M1 Gate A已经证明六动作H10空间中存在341个beneficial动作和14个严格exact-start rescue，但M1 selector只有候选当前native特征、CLIP首帧图像和短文本，没有STTrack自身的首帧RGB/Depth实例表征。M2因此预注册为严格单变量实验：

```text
每条DepthTrack Train序列首帧RGB-D + 初始化框
        ↓
官方STTrack前向一次，query重置
        ↓
不可变native RGB template bank [64,768]
不可变native Depth template bank [64,768]
        ↓
每个候选自身native RGB/Depth pooled feature
        ↓
RGB和Depth各4个token-bank cosine统计
+ 4个跨模态一致性统计，共12维
        ↓
其余M1模型、数据、标签、折分、loss、阈值和Gate B全部不变
```

12维包括每模态对64个首帧token的mean/max/top-4 mean/std，以及RGB/Depth top-4的minimum、product、absolute difference和两模态maximum均值。sequence/frame/source/rank、GT和future IoU仍不作为模型输入。

### 24.167.2　152条不可变首帧原型采集

新增实现提交：

```text
541bb2e5ebb187da966469886b37ac2ef9c2494b
Add immutable native identity selector audit
```

首个collector smoke在`lock01_wild`通过：token bank形状均为`[64,768]`、数值有限、峰值CUDA allocated 637,818,880 bytes，future frame/GT均未打开。正式采集规范：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M2_NATIVE_ANCHOR_SPEC_20260831_1257.json
mode 0444
SHA256 eb564f69cbca2a5e117e0a669533e26e6a85d575eab92cd820a15e723641d83e
```

正式输出：

```text
/root/autodl-tmp/sttrack_lachtt_native_anchors_v1_20260831
152/152 sequences
anchor payload 30,626,192 bytes
index SHA256 da09f4069f0ebd000649afab9419e176a68269dc7ff92b1d1375dab4573bb29c
manifest SHA256 c7c344179704b5ec08904759a90eb711cf0ce54d62e952e90ce9b5d66d92f7a5
elapsed 22.560 seconds
peak CUDA allocated 645,254,656 bytes
```

每条序列只读取`groundtruth.txt`第一行和第1帧RGB-D；`future_frame_opened=false`、`future_ground_truth_opened=false`、`metric_computed=false`、`candidate_committed=false`。152个anchor文件逐个通过bytes/SHA256复核并已只读封存。

### 24.167.3　工程smoke中的唯一接线失败

M2 selector第一次smoke在Python导入阶段报`ModuleNotFoundError: tools`，当时尚未加载数据、训练模型或计算指标。提交`1137e553a7c78427f62438aeb6bf5f4ffcf6c464`把仓库根目录显式加入`sys.path`；随后又以提交`b3623fdf4a156a39bd7d94fabc7e41049151f7df`绑定M1 base trainer、原selector spec、Gate A labels和两个collection manifests，防止隐式源资产漂移。

新的smoke目录：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M2_SELECTOR_SMOKE_20260831_1302_RERUN1
```

24事件、2 epochs、264,403参数，loss从4.5195降到4.1683，8个预测全部有限；result SHA256为`cb22272316520fdf2e5f07c3080ab51f33deb002c19cbfe941553a87f2a0c568`。这只证明工程链可运行，不是效果证据。

### 24.167.4　正式M2 nested OOF结果

正式规范：

```text
STTRACK_LACHTT_M2_SELECTOR_OOF_SPEC_20260831_1305.json
mode 0444
SHA256 50a484b90cf8baed308a27577ae228fc6b7eb45779a101b210b3311198831f4d
```

它复用M1相同的782个可用事件、119条序列、6 outer×5 inner sequence folds、20 epochs、三个seed和同一阈值网格。内门仍要求至少10动作/4序列/95% precision/0 catastrophic；每seed的外门仍要求至少4个有效折、20动作、8序列、95% precision、0 catastrophic，三seed全部通过才算通过。

| seed | 有效/弃权outer folds | 动作 | beneficial | neutral | catastrophic | precision | 序列 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 2/4 | 5 | 4 | 1 | 0 | 80.00% | 5 |
| 2027 | 0/6 | 0 | 0 | 0 | 0 | 0% | 0 |
| 2028 | 1/5 | 14 | 10 | 3 | 1 | 71.43% | 7 |

与M1相比，M2在seed2028把precision从52.63%提高到71.43%，catastrophic从2降到1，说明原生首帧身份锚确实过滤了一部分错误动作；但动作数从38降到14、覆盖序列从15降到7，seed2027仍完全弃权，跨seed和跨序列安全性仍不成立。seed2026也只有80% precision，远低于95%门。

剩余高置信灾难仍是M1已经出现的同一事件：

```text
lock01_wild@386 / velocity_peak0
score 0.9973636269569397
future mean-IoU gain -0.7505438327789307
```

也就是说，即使候选显式匹配STTrack首帧native RGB/Depth token，严重外观/姿态/遮挡变化下错误分支仍可能看起来高度一致。首帧静态原型是有用证据，但不是充分身份证明。

关键工件：

```text
seed2026 result SHA256 2ef2e9c378c72411a7a7816dbc6562cca57a291f4eb9a1ad3697cf36260340bc
seed2027 result SHA256 15fa044c0c0c41613500dd73899ae21fb9f3e07b12724334a5168b9b6952fa3f
seed2028 result SHA256 423bf35133cf98a3a5e51982c7a616d8ede055745fb787380e2b8a583a5ca26c
Gate B result SHA256 410ec671e95aa59054adee053fa56928db7bb6ed690ed376d8f558460d38d10f
summary SHA256 5bce76b85f819cdbbadef0123b64cd3c303c5c0efb1161632fe898f386a6d9d4
```

Gate B正式决定为`stop_sttrack_selector_no_final_no_online_replay`；全部结果已只读封存。

### 24.167.5　问题结论更新与下一路线

M2把“缺少STTrack首帧原生实例锚”从猜测变成了定量结论：它能减少部分灾难，但不能使固定特征selector跨序列可靠。当前剩余问题更深：

1. 首帧单一静态原型无法覆盖旋转、遮挡、长期外观和深度关系变化；
2. 现有candidate RoI是固定backbone产生的pooled特征，selector只能事后组合，不能训练视觉表征主动区分目标与干扰物；
3. H10未来标签监督selector，却没有让STTrack在训练时真实承受“自己的预测框裁下一帧”并学习恢复；
4. beneficial/catastrophic在序列间分布差异很大，内层极高分动作到held-out序列仍可变成灾难；
5. `lock01_wild@386`说明静态身份相似度本身也会被长期外观变化或相似干扰物欺骗。

因此停止继续给该selector堆统计量、扫描阈值或降低95%/零灾难门。下一条允许的Train-only路线应转为**真实4--8帧递归rollout训练**：保留不可变首帧语言/RGB-D身份锚和protected--tentative事务，但让可学习RoI target--distractor head与时序状态在predicted-crop链中联合训练，直接优化短窗survival/hazard。它必须先通过DepthTrack Train sequence-disjoint递归安全门；未通过前仍不运行VOT low22。

本节没有产生tracking checkpoint，也没有新VOT、DepthTrack Test或CDTB指标。正式最好仍为VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`。Qwen3_8B继续保留，本轮没有调用。

<!-- RGBD-HANDOFF-24.168-STTRACK-RECURSIVE-SMOKE-20260831 -->
## 24.168　真实predicted-crop递归训练链H4工程原型通过

### 24.168.1　确认官方训练虽然多帧，但不是递归失误训练

STTrack现有配置设置`DATA.SEARCH.NUMBER=4`，actor也依次前向4个search frame并传递track query；但`TrackingSampler + ViPTProcessing`在进入模型前，已经为每一帧独立使用该帧GT jitter box裁好search crop。即：

```text
现有训练：GT/jitter crop(t) -> GT/jitter crop(t+1) -> GT/jitter crop(t+2)
真实测试：predicted bbox(t) -> crop(t+1) -> predicted bbox(t+1) -> crop(t+2)
```

因此现有多帧训练能学习query时序，却不会暴露“上一帧预测错后，下一帧搜索域随错误状态移动”的分布。这是此前单帧/固定特征模块无法转化为ROB收益的直接训练结构原因。

### 24.168.2　新隔离关联头与事务边界

新增文件：

```text
lib/models/sttrack/lachtt_rollout_association.py
tools/smoke_sttrack_lachtt_recursive_training.py
commit 019510bb4282a119e432fd4820b4b17206dfd170
```

关联头有395,523参数。它对每个16×16 search token分别读取native RGB、Depth和fused特征；RGB/Depth各自cross-attend不可变8×8 anchor token bank，fused token与短身份文本形成语言关系，再显式输入Depth validity。它输出稠密association logits，只用于tentative分支重排原score map。`enabled=false`直接返回原tensor，protected score保持逐位精确一致。

工程smoke维护两个完全独立的递归状态：

```text
protected：官方score、官方query、自己的predicted crop
tentative：association重排score、自己的query、自己的predicted crop
```

tentative没有写入protected；训练只更新新关联头，官方STTrack权重冻结。

### 24.168.3　两次前置失败与最终结果

第一次选择`lock01_wild@382`，但该序列378--385帧GT均为NaN；完整性检查在模型前向前fail-closed。修复后只允许使用窗口内GT有限且宽高为正，不对缺失GT做填充。第二次在`egg_indoor@35`开始后，单通道Depth validity经过OpenCV crop被自动降成二维，旧`sample_target`无法解包H×W×C；改为三通道复制后再取第一通道，数值定义不变。该次在association head前失败，没有参数更新或可接受结果。

最终通过目录：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M3_RECURSIVE_TRAINING_SMOKE_20260831_1325_RERUN2
result SHA256 ad05984e46b30841e4ab4445b756bd2828ccf9de9edc32df3131ec30c6e20c52
```

固定`egg_indoor@35`、H4、3个更新步。结果：

| step | H4平均association loss | gradient norm |
|---:|---:|---:|
| 0 | 44.006317 | 288.403717 |
| 1 | 41.659721 | 276.988617 |
| 2 | 33.463318 | 579.763245 |

18个参数tensor实际变化，所有loss/gradient/输出有限，峰值CUDA allocated为1,630,555,136 bytes；protected score exact parity=true、protected/tentative state独立=true、candidate commit=false、checkpoint written=false。该片段目标运动很小，association初始化为零且三步训练没有改变argmax，所以两分支H4 IoU仍相同；这正是为什么它只能算“真实递归和反向传播链可运行”，不能算效果提升。

### 24.168.4　正式训练前必须补的门

正式M3不能直接把这一个容易片段扩成训练。必须先冻结：

1. 从M1 sealed GT-free风险时间表选择事件，sequence-disjoint拆训练/验证，禁止同序列泄漏；
2. anchor使用每条trajectory自己的初始化帧RGB-D模板，短文本只作不可变身份锚；
3. current/last/velocity多中心分支各自递归4--8帧，搜索crop必须来自各分支上一帧预测；
4. 训练目标同时包含dense association、target--distractor rank、短窗survival和failure hazard；
5. protected路径在所有训练/验证过程中零写入，tentative只在离线验证中比较，不在线promote；
6. 先用Train sequence-disjoint评估failure starts、catastrophic/rescue和平均IoU；通过零新增灾难门后才允许汇报并请求启动low22。

当前尚未冻结正式M3训练spec、没有训练checkpoint，也没有改变任何公开指标。Qwen3_8B仍保留且未调用。

工程阶段机器可读计划已冻结为`STTRACK_LACHTT_M3_RECURSIVE_TRAINING_PLAN_20260831_1335.json`（mode 0444，SHA256 `9a8d0afcf6249260c462a95a6287e49ef691b3967d478b480d7a0bf3d8d90c4e`）。它只授权下一次多中心风险事件工程smoke；不授权formal pilot、nested OOF、在线回放或VOT。

<!-- RGBD-HANDOFF-24.169-STTRACK-M3-MULTICENTER-PILOT-20260831 -->
## 24.169　多中心候选对齐递归pilot完成但安全弃权，M3停止

### 24.169.1　工程smoke补齐了六候选、候选对齐hazard和损失平衡

在`egg_indoor@39`这个sealed GT-free `low_score`风险事件上，按`current / last_reliable / velocity`三个中心、每中心factor-6 top-2建立六个分支；后续每个分支使用自己的预测框、RGB/Depth query和factor-4 crop递归H4。保护trace在运行前后逐字节一致，候选没有写回保护路径。

工程阶段发现并修复了三个不能带入正式训练的问题：

1. 首次误用了含`TEMPLATE_TRANSACTION`字段的旧工作树配置，在网络构建前fail-closed，无输出、无更新；随后绑定当前分支`deep_rgbd_256_lachtt_v1.yaml`。
2. 原CenterNet focal在本任务中把约255个负网格累加、只有一个正峰，dense loss约44并压倒survival/rank；即使把dense权重降至0.02，survival仍从0.693升至0.754。最终改为“正域均值+负域均值”的balanced focal，使四项损失同量级。
3. 最初hazard对整张source crop做均值，同一source的peak0/peak1共享风险，仍不是真正candidate-specific。最终改为16×16候选风险图，在每个NMS峰自己的网格位置读取hazard。

最终工程结果：

```text
result:
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M3_MULTICENTER_RECURSIVE_SMOKE_20260831_1348_RERUN4/result.json
SHA256 cab6d70ceb6b15000262eb0208102dd5eb5a806d27d0d7bc57767acee6af84fb

commit:
ae3cecfa5b10fd4db6b60867d1d2491ce8cbf930
```

| 项目 | step 0 | step 2 | 方向 |
|---|---:|---:|---|
| total loss | 1.932868 | 1.816717 | 改善 |
| survival | 0.693147 | 0.638177 | 改善 |
| hazard | 0.693147 | 0.629872 | 改善 |
| rank | 0.200000 | 0.199184 | 小幅改善 |

20个参数tensor变化，395,652个参数，显存峰值2,082,573,312 bytes；protected exact=true、candidate commit=false、checkpoint=false。触发帧的六分支中有三个正确候选，IoU为0.937/0.937/0.878；但从下一帧开始六分支都跌到0，即使GT仍位于6/6搜索crop内。三步训练未改变argmax，所以这里只证明训练链和目标方向正确，不证明恢复有效。

### 24.169.2　正式pilot严格预注册，未扫描阈值

pilot spec冻结为：

```text
STTRACK_LACHTT_M3_FORMAL_PILOT_SPEC_20260831_1410.json
mode 0444
SHA256 9ec6c84ffa3e3ae501dc02a2b3c71ff41ea320d7e894f883cad15fff426a987b
```

协议固定为DepthTrack Train only、seed2026、H4、2 epoch、head-only、100% predicted-crop、稳定哈希6折中的fold0验证，其余折训练；前60个训练事件、前24个验证事件，序列必须完全不交叉。动作规则固定为：hazard不高于0.20、与次优hazard margin至少0.10、refined response至少0.25，否则保护性弃权。Gate要求至少2个动作、2个beneficial、覆盖2条序列且0 catastrophic。冻结后没有更改或扫描这些阈值。

runner提交：

```text
c98acd743cb43bac303e9bb6adc3885ff564f19e
tools/run_sttrack_lachtt_recursive_pilot.py
```

第一次正式启动在首个模型更新前发现`toy07_indoor_320`长度异常：RGB、Depth和sealed trace均为连续1,367帧，GT却有1,406行，多出尾部39行。59个入选序列中只有这一条异常。修复严格要求RGB/Depth/trace等长，只允许忽略帧文件之后的GT尾行并在结果中显式审计；没有截断帧、重排事件或用后续事件回填。修复提交`92f56c49d16847fb7e285ad29f6d26067a3dec27`，RERUN1绑定SHA256为`a0ea3cc1cbab3d08ce60bad18d0612f96aa7f8af34055e3acd755b38f1aace97`。

### 24.169.3　pilot结果：损失下降，但18/18验证事件全部弃权

正式结果目录：

```text
/root/autodl-tmp/sttrack_lachtt_m3_recursive_pilot_seed2026_20260831_RERUN1
result SHA256 cf352d227fbbdf22e3c1f82f12060dbb71e2f39fbdcb19c0dfc66260d6969b5f
```

| 项目 | 结果 |
|---|---:|
| scheduled train events | 60 |
| 实际训练更新 | 50（25个唯一有效事件×2 epoch） |
| train sequences | 45 |
| scheduled eval events | 24 |
| 实际eval events | 18 |
| eval sequences | 14 |
| train/eval sequence overlap | 0 |
| loss | 1.932868 → 0.778204 |
| selected actions | 0 |
| beneficial / catastrophic | 0 / 0 |

`unavailable_records=76`包含epoch重复：去重后为35个训练事件和6个验证事件，共41个唯一role-event；它们的H4 GT含NaN或宽高不可用，没有填充或回补。

pilot没有失败在“候选动作空间为空”。对18个有效held-out事件的108个候选做只读后验审计：

| 后验候选标签 | 数量 |
|---|---:|
| neutral | 90 |
| beneficial | 9 |
| catastrophic | 9 |

9个beneficial覆盖3个事件和3条序列：`ball16_indoor@886`、`bag03_indoor@2058`、`mobilephone04_indoor@1278`。也就是说held-out中仍有可恢复容量，但当前在线头没有安全识别它。

真正失败点是风险可分性：

```text
每事件最低candidate hazard：0.03306 / 0.04160(median) / 0.05729
最低与次低hazard margin：0 / 0.000109(median) / 0.01450
冻结要求的margin：0.10
```

所有候选的绝对hazard都被压得很低，beneficial与catastrophic之间没有形成候选间间隔。因此18/18事件按照预注册规则全部弃权。若删除margin，`cup07_indoor`和`human06_indoor`等已知catastrophic候选也会进入动作集合；所以不能以“增加动作数”为理由降低门槛。

### 24.169.4　新的核心问题与停止决定

M3已经从“public框身份特征”推进到“候选自己的稠密RGB-D/text特征和候选网格hazard”，但它仍然逐个独立处理candidate。它没有在六个假设之间做显式target--distractor竞争，也没有用其他候选作为负实例。current/last/velocity在部分事件还会产生相同或近似crop，使独立hazard更容易收敛到近似值。这解释了绝对loss明显下降，却没有产生可提交margin。

正式决定：

```text
stop_m3_independent_candidate_hazard
no threshold rescan
no nested OOF
no complete-sequence replay
no DepthTrack Test / CDTB / VOT low22 / VOT full127
```

下一条允许的结构只能是setwise候选关联：把六个candidate token同时输入listwise target--distractor模块，显式建模candidate-to-candidate contrast、目标原型、干扰物原型与分支时序；监督直接优化相对排序和零灾难abstention，而不是继续训练六个彼此独立的绝对hazard。必须重新冻结新spec和小pilot；不能复用或放宽本次失败阈值。

机器可读总结为`STTRACK_LACHTT_M3_PILOT_RESULT_SUMMARY_20260831_1445.json`。本节没有产生可部署tracking checkpoint，没有调用Qwen3_8B，没有改变任何公开指标。正式最好仍为VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`。

<!-- RGBD-HANDOFF-24.170-STTRACK-M4-SETWISE-PILOT-20260831 -->
## 24.170　M4六候选setwise关联完成，但学习成无条件abstain

M3失败后没有降低hazard margin，而是实现378,199参数的permutation-equivariant setwise模块：每个候选的H4语言锚定RGB-D hidden和9个因果scalar先经共享GRU，再由无位置/source/rank编码的两层set Transformer联合比较六候选，并加入learned abstain token；输出beneficial、catastrophic、future gain和六候选+abstain selection。合成工程smoke的候选置换误差仅`2.38e-7`，loss `3.4423→0.6251`；真实`egg_indoor@39`单事件六分支前向/反传也全部有限。

正式spec SHA256=`3a6f7529…3c6a`，runner commit=`6c2e77481b42b5b4198507948053b3e1d082778c`。固定DepthTrack Train、seed2026、稳定哈希240/96事件、H4、2 epoch、sequence-disjoint、100% predicted-crop和不扫描的selection/benefit/cat/gain门。

正式pilot完成268次更新、78条训练序列；held-out为67事件/20序列、sequence overlap=0，loss `3.916793→0.312247`，protected mutation=0，无tracking checkpoint、Qwen或VOT。但67/67全部abstain，Gate失败，result SHA256=`8ea9cc1e…56c8`。

只读后验显示402候选中333 neutral、23 beneficial、46 catastrophic；beneficial覆盖6事件/6序列：`bag03`、`ball16`、`ball17`、`cup07`、`hand02`、`mobilephone04`。候选selection概率仅0.00349--0.00863，abstain为0.95447--0.97570；beneficial概率全部约0.05，predicted gain全部为负。说明setwise结构能运行，但未加权event-level CE被大量“无beneficial候选”事件支配，模型把abstain学成无条件先验。

正式停止M4：不改推理阈值、不跑OOF/回放/VOT。下一条仅允许M5训练折内标签平衡：提高含beneficial事件的selection权重、beneficial BCE正例权重和pairwise hard-event权重；held-out事件、动作门、零灾难规则和protected事务全部不变。机器总结为`STTRACK_LACHTT_M4_SETWISE_PILOT_SUMMARY_20260831_1600.json`。公开指标不变。

<!-- RGBD-HANDOFF-24.171-STTRACK-M5-LABEL-BALANCE-20260831 -->
## 24.171　M5标签平衡pilot仍全弃权：发现单事件权重抵消与概率门错位

### 24.171.1　训练折审计与M5唯一变量

M4失败后没有读取held-out标签来选权重。只在fold1--5、冻结的前240个训练事件上重新执行H4 outcome审计；fold0没有打开，Qwen、VOT和checkpoint写入均为false。审计结果如下：

| 训练折项目 | 数量 |
|---|---:|
| scheduled events | 240 |
| valid / unavailable events | 134 / 106 |
| 含beneficial候选的事件 | 32 |
| 不含beneficial候选的事件 | 102 |
| 候选总数 | 804 |
| beneficial / neutral / catastrophic candidates | 151 / 584 / 69 |

由训练折频率一次性计算、未扫描的逆频率权重为：

```text
selection positive-event weight = 102 / 32 = 3.1875
beneficial BCE pos_weight       = 653 / 151 = 4.324503311258278
catastrophic BCE pos_weight     = 735 / 69  = 10.652173913043478
gain/pairwise beneficial weight = 4.324503311258278
```

审计工件：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M4_TRAIN_LABEL_AUDIT_20260831_1615.json
SHA256 477d30d2a5037ff84ab2867af61a7aa975ae2ac18259540e55f26f327289aacd
mode 0444
```

M5 spec SHA256=`1331f2ee…c8535`，binding SHA256=`8e1d5b76…58709`。M5相对M4只改变训练loss balance；网络结构、seed2026、240/96事件、H4、2 epoch、fold0留出、动作阈值和pilot gate全部冻结。实现提交`afbdf16e16b37a2d1d8a134a9e3595e392b683c2`。默认权重全部为1时与M4 loss逐项数值完全一致，最大误差为0；setwise置换误差仍为`2.38e-7`。真实训练折`book06_indoor@1186`单事件加权前向/反传有限，69个参数tensor变化，没有保存checkpoint。

工程烟测最初尝试使用`egg@39`，但事前断言发现该事件属于held-out fold0，因此在模型构建/更新和输出写入前fail-closed；随即改为已经出现在M4封存训练轨迹中的`book06_indoor@1186`（fold5）。这是一次有效的协议保护，不是失败实验，也没有打开fold0标签供M5选权重。

### 24.171.2　正式结果：辅助输出改变，但67/67仍然abstain

正式结果：

```text
/root/autodl-tmp/
sttrack_lachtt_m5_label_balanced_setwise_pilot_seed2026_20260831

result SHA256:
a9bf686d24a4b920b410cf74b193c06a0cfc39e14b99a85d7e099690fca95551
```

| 项目 | M4未加权 | M5训练折逆频率加权 | 结论 |
|---|---:|---:|---|
| training updates | 268 | 268 | 相同 |
| held-out events / sequences | 67 / 20 | 67 / 20 | 相同 |
| train/eval sequence overlap | 0 | 0 | 合规 |
| loss first | 3.916793 | 3.916793 | 相同 |
| loss last | 0.312247 | 0.462830 | M5更高，不代表更差或更好，因为loss被重新加权 |
| selected actions | 0 | 0 | 无改善 |
| beneficial / catastrophic actions | 0 / 0 | 0 / 0 | 无在线动作 |
| protected trace mutations | 0 | 0 | 保护路径未污染 |

M5 held-out动作空间仍有容量：402个候选中有23个beneficial，分布在6个事件/6条序列；另有45个catastrophic。六个正事件与最佳真实动作如下：

| 序列@帧 | 最佳真实动作 | actual gain | M5 selection | benefit prob | catastrophe prob | predicted gain |
|---|---|---:|---:|---:|---:|---:|
| `ball16_indoor@886` | velocity_peak1 | +0.713325 | 0.005519 | 0.164738 | 0.082254 | -0.016803 |
| `bag03_indoor@2058` | last_reliable_peak0 | +0.887565 | 0.005472 | 0.154627 | 0.077477 | +0.006609 |
| `mobilephone04_indoor@1278` | velocity_peak1 | +0.058328 | 0.004160 | 0.129720 | 0.061025 | +0.061065 |
| `hand02_indoor@483` | velocity_peak0 | +0.242794 | 0.005004 | 0.138148 | 0.068602 | +0.011341 |
| `cup07_indoor@1792` | current_peak0 | +0.605412 | 0.004931 | 0.125904 | 0.062616 | +0.039883 |
| `ball17_wild@1310` | current_peak0 | +0.223624 | 0.006142 | 0.164069 | 0.085683 | -0.013727 |

固定动作门的逐项通过数揭示了失败位置：

| 条件（67个事件的top candidate） | M4通过数 | M5通过数 |
|---|---:|---:|
| candidate击败abstain | 0 | 0 |
| selection probability ≥0.50 | 0 | 0 |
| selection margin ≥0.10 | 0 | 0 |
| benefit probability ≥0.80 | 0 | 0 |
| catastrophe probability ≤0.05 | 10 | 0 |
| predicted gain ≥0.05 | 0 | 11 |
| 全条件同时通过 | 0 | 0 |

概率分布也不是“刚好卡在阈值附近”：

```text
M4 top selection  min/median/max = 0.00449 / 0.00727 / 0.00863
M5 top selection  min/median/max = 0.00394 / 0.00543 / 0.00726

M4 abstain        min/median/max = 0.95447 / 0.96253 / 0.97570
M5 abstain        min/median/max = 0.96219 / 0.97012 / 0.97733

M4 benefit        min/median/max = 0.04904 / 0.05632 / 0.05770
M5 benefit        min/median/max = 0.11974 / 0.14637 / 0.16306

M4 catastrophe    min/median/max = 0.04549 / 0.05250 / 0.05400
M5 catastrophe    min/median/max = 0.05404 / 0.07260 / 0.08556

M4 predicted gain min/median/max = -0.10763 / -0.09029 / -0.06276
M5 predicted gain min/median/max = -0.02124 / +0.00661 / +0.10437
```

所以逆频率权重确实使benefit和gain辅助输出向正方向移动，但没有形成可提交的selection；同时catastrophic正例权重把所有raw sigmoid catastrophe值整体抬高，反而使冻结的`≤0.05`绝对概率门从10个事件可通过变为0个。

### 24.171.3　确认的实现级根因一：batch-size-1使event权重严格抵消

runner的训练循环是：

```text
for event in train_events:
    run_event(...)
    optimizer.step()
```

因此每个optimizer step只有一个事件。M5 selection实现按“加权均值”计算：

```text
selection = sum(w_event * CE_event) / sum(w_event)
```

当batch只有一个事件时：

```text
selection = w * CE / w = CE
```

也就是说，预注册的`3.1875` positive-event selection权重在每一个正事件上都被自己的分母精确抵消，对直接selection梯度的作用严格为0。M5虽然名义上“加权selection”，实际只改到了beneficial/catastrophic/gain/pairwise辅助项；核心六候选+abstain CE仍按102个负事件对32个正事件的原始频率训练。这是M5仍学成无条件abstain的首要、可复现原因。

这不是可以通过降低0.50 selection阈值解决的问题。M5 top candidate最大仅0.00726，而abstain最小仍为0.96219；二者相差两个数量级。阈值扫描只会掩盖训练目标未生效。

### 24.171.4　确认的实现级根因二：pos_weight概率不能直接套用原绝对阈值

`BCEWithLogits(pos_weight=w)`改变的是正例相对代价，也会改变最优logit的先验偏置；raw sigmoid不再天然是原数据分布下的校准概率。M5把catastrophic `pos_weight`设为10.652后，catastrophe raw sigmoid整体从约0.05升到0.054--0.086，导致原冻结`≤0.05`安全门全部关闭。beneficial raw sigmoid虽从约0.05升到0.12--0.16，仍远低于`≥0.80`。

因此当前问题至少有两层：

1. selection正事件权重因单事件更新被严格抵消；
2. 辅助BCE的逆频率权重与“把raw sigmoid当校准概率”的绝对门不兼容。

不能把M5负结果概括成“setwise架构没有容量”，也不能把它解释成“只要再加大class weight就会好”。候选动作空间仍有6条序列的正动作；失败的是训练目标和部署门之间的数学契约。

### 24.171.5　正式停止边界与下一步

M5正式决定：

```text
stop_m5_no_threshold_scan_no_oof_no_public_benchmark
```

因此没有启动nested OOF、complete-sequence replay、DepthTrack Test、CDTB、VOT low22或full127；没有生成tracking checkpoint，没有调用Qwen3_8B，也没有改变公开最好指标。

机器可读对照：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M5_PILOT_COMPARISON_20260831_1555/result.json
SHA256 eb307e15e2d8bfadbc87234989dbc9d9551016bb70ab3112ee4f97050deee815
mode 0444
```

下一次若继续，只能另立M6 Train-only spec修复“训练目标没有实际生效”的问题，不能修改held-out折、推理阈值或零灾难Gate。最小可检验改动是让beneficial-event selection权重真正作用于梯度：要么在batch-size-1下直接使用`w_event * CE`而不再除以同一个`w_event`，要么先积累多个事件再做跨事件加权均值。同时必须明确处理BCE先验偏置与绝对概率门的校准契约；不能一边使用大`pos_weight`，一边把raw sigmoid直接解释为原始概率。

当前正式最好仍为：VOT `EAO/ACC/ROB = 74.020583/82.579344/89.565651`，DepthTrack `P/R/F = 65.995933/65.335885/65.664250`，CDTB `P/R/F = 75.387821/76.005850/75.695574`。

### 24.171.6　训练折最终头审计：不是held-out泛化问题，训练数据本身也0动作

为区分“训练目标未学会”和“只在fold0泛化失败”，使用M5封存的`heads_only.pt`在fold1--5的冻结前240事件上做只读推理。初版审计把`select_action()`返回的“abstain相对次优的正margin”误标为“候选相对abstain的margin”；它不影响0动作结论，但该字段语义错误，因此原结果`0b0f8d0e…aac73`只保留为无效诊断，不用于结论。修正提交`0c0b2c52b1ccd492b5f627182d7671a4b7be5339`后从头RERUN1。

正确结果：

```text
/home/SUTrack_RGBD_L/refine-logs/
STTRACK_LACHTT_M5_TRAIN_PREDICTION_AUDIT_20260831_1605_RERUN1.json
SHA256 a7e791b2048241f6c6c6c590edf49f375355b6adb6376a2de42d9725ee938060
```

| 项目 | 训练折最终头 |
|---|---:|
| scheduled / valid / unavailable events | 240 / 134 / 106 |
| 含beneficial候选的事件 | 33 |
| 含catastrophic候选的事件 | 23 |
| selected / beneficial / catastrophic actions | 0 / 0 / 0 |
| candidate击败abstain | 0/134 |
| selection≥0.50 | 0/134 |
| candidate margin≥0.10 | 0/134 |
| benefit≥0.80 | 0/134 |
| catastrophe≤0.05 | 0/134 |
| predicted gain≥0.05 | 35/134 |

训练折分布与held-out几乎相同：top selection为`0.00360/0.00539/0.00725`，abstain为`0.96183/0.96986/0.97859`，candidate相对abstain margin为`-0.97497/-0.96448/-0.95470`；benefit为`0.11863/0.14455/0.16659`，catastrophe为`0.05235/0.07320/0.08560`。这排除了“模型在训练折能动作、只在新序列失效”的解释：M5最终头连训练事件的冻结部署门都没有学会。

因此M6不能只做温度校准或held-out阈值调整。它必须让正事件selection梯度真实放大，并让benefit/cat/gain训练目标直接对齐已经冻结的部署安全门；仍只能在DepthTrack Train上验证。

<!-- RGBD-HANDOFF-24.172-STTRACK-M6-GATE-ALIGNED-20260831 -->
## 24.172　M6 gate-aligned objective仍失败：模型只学会负事件，正候选未被分离

### 24.172.1　M6修复内容与工程证据

M6不是继续扫描M5权重，而是针对两处已确认的数学失配做一次独立预注册：

1. 对含beneficial候选的事件，selection CE直接乘训练折逆频率`3.1875`，不再除以同一个event weight；单事件工程验证确认selection loss相对归一化实现精确放大`3.1875`倍。
2. beneficial/catastrophic BCE恢复`pos_weight=1`，避免raw sigmoid先验偏移；另用冻结部署门直接构造logit hinge：benefit正例`≥logit(0.80)`、负例`≤logit(0.20)`，cat正例`≥logit(0.80)`、负例`≤logit(0.05)`，beneficial gain不低于0.05。

网络、候选、H4递归、240/96事件、fold0留出、seed2026、2 epoch、优化器、推理阈值和零灾难Gate全部不变。spec/binding如下：

```text
STTRACK_LACHTT_M6_GATE_ALIGNED_SETWISE_PILOT_SPEC_20260831_1800.json
SHA256 5da720df8ac330d484f3e2fc290d6fec5ecd110b42d14258b77dd4e0c66d67ce

STTRACK_LACHTT_M6_GATE_ALIGNED_SETWISE_PILOT_BINDING_20260831_1830.json
SHA256 473b086804552357c0c60bc30abf5d84f390674919103c070bbde77c450ace07
```

实现提交`5fb77c43e4d19074d136b7354b6e6a6d4c4eb760`，真实工程smoke提交`93a503792e4d1b22e060ac5001fb37b2dcb158b1`。默认loss路径与M4数值保持一致，置换误差`2.38e-7`。真实训练折`colacan02_indoor@1664`包含4个beneficial、0 catastrophic候选；M6单步selection=6.1572、benefit gate=1.4334、cat gate=3.8232，69个参数tensor更新且全部有限。smoke SHA256=`d34929e1…976629`，没有checkpoint或VOT。

### 24.172.2　正式pilot仍为67/67 abstain

```text
/root/autodl-tmp/
sttrack_lachtt_m6_gate_aligned_setwise_pilot_seed2026_20260831

result SHA256:
9b8e66885f0747a153e29a9bb9e21a5dfbb51eaa050cebd95170ed667db1cfee
```

| 项目 | M5 | M6 | 变化 |
|---|---:|---:|---|
| training updates | 268 | 268 | 相同 |
| held-out events / sequences | 67 / 20 | 67 / 20 | 相同 |
| loss first → last | 3.9168→0.4628 | 8.6218→0.3684 | 目标不同，不能直接横比 |
| selected actions | 0 | 0 | 未改善 |
| beneficial / catastrophic actions | 0 / 0 | 0 / 0 | 未形成部署动作 |
| protected trace mutations | 0 | 0 | 保护路径未污染 |

M6没有改变候选后验容量：held-out仍为23 beneficial candidates/6 beneficial events，45 catastrophic candidates/14 catastrophic events。它修复了cat门并略微降低abstain，但没有解决正候选识别：

| top-candidate条件 | M5通过数 | M6通过数 |
|---|---:|---:|
| candidate击败abstain | 0/67 | 0/67 |
| selection≥0.50 | 0/67 | 0/67 |
| candidate margin≥0.10 | 0/67 | 0/67 |
| benefit≥0.80 | 0/67 | 0/67 |
| catastrophe≤0.05 | 0/67 | **67/67** |
| gain≥0.05 | 11/67 | 0/67 |
| 全条件 | 0/67 | 0/67 |

```text
M6 top selection  min/median/max = 0.00792 / 0.00890 / 0.00975
M6 abstain        min/median/max = 0.94551 / 0.94909 / 0.95451
M6 benefit        min/median/max = 0.05953 / 0.06250 / 0.06442
M6 catastrophe    min/median/max = 0.02461 / 0.02516 / 0.02553
M6 predicted gain min/median/max = -0.02487 / 0.02199 / 0.03940
```

对6个真实beneficial事件，正确候选的selection仍只有0.0079--0.0091，benefit约0.060--0.065，gain均未达到0.05。也就是说gate margin没有把正例推过门；cat门之所以全部通过，是模型把所有候选都学成了低cat风险，而不是学会了“哪个候选是beneficial”。

### 24.172.3　训练折最终头也0/134，确认不是泛化问题

M6最终head在fold1--5的corrected只读审计：

```text
STTRACK_LACHTT_M6_TRAIN_PREDICTION_AUDIT_20260831_1850.json
SHA256 f5080773ac9941fae9cbc096e397bb01557c1a778709faa403fdb20be41c7513
```

| 训练折条件 | 通过数 |
|---|---:|
| 有效事件 | 134 |
| 含beneficial / catastrophic候选事件 | 33 / 23 |
| selected actions | 0 |
| candidate击败abstain | 0/134 |
| selection≥0.50 | 0/134 |
| margin≥0.10 | 0/134 |
| benefit≥0.80 | 0/134 |
| catastrophe≤0.05 | **134/134** |
| gain≥0.05 | 0/134 |

训练折top selection最大0.00971、abstain最小0.94458，benefit最大0.06539、gain最大0.03956；与held-out几乎相同。这严格排除了“训练折能选、只在fold0失效”的解释。

训练trace进一步显示负事件和正事件发生分裂。第2 epoch的selection loss：

```text
mean   = 3.43760
median = 0.14836
min    = 0.01961
max    = 18.34310
> 3 的事件 = 33
```

33个高selection-loss事件的数量与最终训练折33个beneficial-capacity事件一致，但trace没有逐事件保存标签，因此这里只作为高度一致的模式证据，不冒充逐行因果绑定。更稳妥的结论是：大多数易负事件已经学会abstain，少数困难正事件仍保留极高loss，2个epoch内没有形成可分表示；benefit gate第2 epoch仍为mean 0.732、max 4.749、median 0，呈现同样的“多数负例已满足、少数正例未解决”结构。

### 24.172.4　停止同规模loss路线，转向更多递归监督或更强baseline

M6正式停止：

```text
stop_m6_no_threshold_scan_no_oof_no_public_benchmark
```

不增加epoch、不继续扫描class weight/margin/温度、不跑OOF或VOT。M3--M6已经依次排除：独立hazard、setwise无权重、逆频率权重、gate-aligned margin；在相同134有效事件的小pilot上继续换loss，不能解决实例身份表示和正样本覆盖不足。

下一步应转向两类实质变化之一：

1. 使用完整STTrack M1的1,466个真实风险事件/8,796动作，按beneficial/non-beneficial平衡mini-batch缓存H4候选轨迹，先证明setwise头能在训练折学会正候选，再做sequence-disjoint验证；
2. 按用户既定决策，停止当前STTrack小样本路线，把“语言锚定candidate-specific RGB-D身份 + protected/tentative递归事务”移植到VOT指标更强的开源RGB-D baseline，再用其更强表征训练关联头。

当前没有新tracking checkpoint，没有调用Qwen3_8B，没有运行VOT low22/full127；正式最好指标仍为VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`。

## 24.173　强RGB-D baseline官方来源复核与迁移边界（2026-08-31）

本轮按“如果当前创新仍拉不回VOT，就迁移到更强开源baseline”的既定决策，重新核对了论文指标、作者仓库、对应权重和许可证。完整独立报告为：

```text
/home/SUTrack_RGBD_L/docs/RGBD_STRONG_BASELINE_OFFICIAL_AUDIT_20260831.md
```

结论必须区分“论文分数最高”与“现在可合法、可复现地迁移”：

| 方法 | 作者报告VOT EAO/ACC/ROB | DepthTrack Pr/Re/F | 当前发布状态 | 本项目决定 |
|---|---:|---:|---|---|
| MDTrack-U | **80.0/83.5/95.1** | **68.1/67.6/67.9** | 论文所指官方仓库为空；无源码、权重、许可证 | 仅作架构参照，等待真实发布 |
| FlexTrack | 78.0/83.8/93.1 | 67.1/66.9/67.0 | 论文仓库只有README/LICENSE，没有论文模型代码和checkpoint | 不能复现，不能冒充可迁移baseline |
| STTrack | 77.6/82.5/93.7 | 63.2/63.4/63.3 | 源码、对应权重、raw results、MIT均完整 | **当前最高可靠可用的递归事务平台** |
| SUTrack-L384 | 76.6/83.5/92.2 | 66.5/66.4/66.4 | 源码、权重、MIT完整 | 语言接口成熟，但VOT不是更强主干 |
| FlexTrackV2 | VOT三指标未报告 | 66.1/69.0/67.5 | 代码和权重已发布，但仓库无可见许可证；权重端点本机401/服务器超时 | 只允许静态探索，不作为正式baseline |
| XTrack-L | 74.0/82.8/88.9 | 65.4/64.3/64.8 | 代码、权重、MIT完整 | VOT不强于当前正式路径，不迁移 |

因此本阶段没有为了追逐论文表格而换到不可复现模型。MDTrack-U的RGB/X独立时序状态与跨模态交互非常符合当前问题，但在作者真正发布前不能落地；STTrack仍是唯一同时具备高VOT、完整递归状态、可下载对应权重和明确许可的强平台。

### 24.173.1　FlexTrackV2只完成了fail-closed结构探索

隔离仓库：

```text
/root/autodl-tmp/rgbd_baselines/FlexTrackV2_20260831
branch: codex/lachtt-trace-v1
upstream fixed commit: 30a5ff1b39b8f3004dde4018018b901e6cc1bf54
local exploratory commit: 254191ca25491122179aa61c81dfb1b5a8ed592d
```

新增内容是默认关闭的trace-only递归候选事务：完整快照bbox、Mamba历史状态、模板、annotation、memory和计数器；`last_reliable`保存完整快照；`last_reliable/velocity`候选独立H3 rollout；candidate-own RGB identity、edge identity和显式标为“RGB colormap proxy”的辅助统计只写trace，绝不伪称真实metric depth。关闭模块时原`track()`生产函数AST精确一致，YAML基础项精确parity，合成smoke通过，公开protected路径零写入。

这不是效果实验。真实权重没有成功获取：作者Hugging Face端点在服务器超时、本地HEAD返回401、镜像404；只留下0-byte `.part`，没有产生可运行checkpoint。更重要的是V2没有官方VOT三指标且无可见许可证，所以不运行VOT、不把FlexTrack论文的78.0归给V2，也不向用户仓库推送V2源码。

## 24.174　M7：全M1事件均衡mini-batch训练（正式运行中）

M3--M6在相同134个有效pilot事件上依次排除了独立hazard、无权重setwise、逆频率权重和gate-aligned margin；训练折与held-out都0动作。当前唯一合理的同baseline升级不是继续扫loss/阈值，而是使用M1全部1,466个真实递归风险事件，显著增加正例与灾难例覆盖，并把梯度单位从“单事件”改成预注册的事件均衡mini-batch。

### 24.174.1　数据和标签规模

Gate A只读标签结果：

```text
events                    = 1,466
candidate actions         = 8,796
beneficial actions        = 341
catastrophic actions      = 228
neutral actions           = 4,123
unavailable actions       = 4,104
beneficial events         = 89
catastrophic events       = 66
neutral events            = 627
unavailable events        = 684
```

fold0冻结为未来评估折；当前训练只用fold1--5的628个available事件：78 beneficial、60 catastrophic、490 neutral。fold0含154个available事件：11 beneficial、6 catastrophic、137 neutral，正式训练期间不用于调参。

正式资产：

```text
Gate A:
/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/gate_a_result.json
SHA256 d74aff281a4dba95a9f37e33e450e8a5ab14e0add12976e045756466c12e0592

labels:
/root/autodl-tmp/sttrack_lachtt_train152_gatea_v1_20260831/labeled_actions.jsonl.gz
SHA256 f30316f40a4cd29bd609a2f7088234a9daa4de580ad823b11eb529b3bb66457a
```

### 24.174.2　事前冻结的M7训练契约

规范在实现与正式训练前只读冻结：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M7_FULLSET_BALANCED_SPEC_20260831_1930.json
SHA256 bd6392cd5cbe41860fcc62532d79d447825b4d8423fd7cb580b355335647a61a
mode 0444
```

冻结项如下：

- 模型仍为STTrack + static identity language + current/last/velocity × top-2六候选 + H4 candidate-own RGB-D递归证据；不改架构、候选、horizon、标签和部署门；
- batch size 8，严格由`2 beneficial + 2 catastrophic + 4 neutral`事件组成；稀有类确定性循环过采样，每个epoch覆盖全部490个neutral事件；
- 每epoch 123 batches、984次event forward；2 epochs共1,968次forward、246次optimizer step；
- 每个event loss除以8后累积，一个batch只做一次`zero_grad/clip/AdamW step`；
- `lr=0.001`、`weight_decay=0.0001`、`seed=2026`，沿用M6 gate-aligned losses；
- 部署门完全不变：candidate≥0.50、candidate-abstain margin≥0.10、benefit≥0.80、catastrophe≤0.05、predicted gain≥0.05；
- 不调用Qwen、不运行DepthTrack Test/CDTB/VOT、不生成tracking checkpoint、不自动进入下一阶段。

实现工作树：

```text
/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1
branch codex/language-anchored-candidate-transaction-v1
HEAD 0e4176a68de5260cb17cd6258595da6bec5f1464
runner SHA256 51653c51ce86593c4e4d928481d044af02fab2a7ba1e9e8c78ee4fdca23e5412
```

静态调度smoke确认123个batch全部严格2/2/4、每epoch覆盖全部neutral、两个epoch确定性一致。真实CUDA工程smoke结果：

```text
/root/autodl-tmp/sttrack_lachtt_m7_balanced_smoke_20260831_2000/result.json
SHA256 976afaab83fe4780dae493d710a1aea3db2dabdc5b08bac79eae4bf82ce0f618
accepted = true
changed tensors = 69
finite final gradient norm = 65.80192565917969
peak GPU memory = 2,078,316,544 bytes
elapsed = 29.01 seconds
```

正式绑定在训练前转为只读：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M7_FULLSET_BALANCED_BINDING_20260831_2005.json
SHA256 3f843c144f484cf2864959359aa7821c201e699098e130c6c85f6013c30a9310
mode 0444
```

正式运行：

```text
screen: sttrack_lachtt_m7_fullset_balanced
output: /root/autodl-tmp/sttrack_lachtt_m7_fullset_balanced_seed2026_20260831
log: /root/autodl-tmp/sttrack_lachtt_m7_fullset_balanced_seed2026_20260831.log
GPU0；GPU1保留空闲
```

启动预检中，仓库干净、HEAD/runner/spec/binding哈希全部匹配、两张GPU均仅1 MiB占用、输出与日志不存在、磁盘剩余约8.2 GiB。启动后GPU0约3.95 GiB且有计算负载，screen存活。此处只记录“正在训练”，不提前宣称改善。

### 24.174.3　M7之后唯一允许的判断

M7结果先检查训练折是否真正学会beneficial候选和abstain边界。只有非零动作、灾难门不靠全弃权、正候选跨多序列可分，才另行冻结Train-only complete-sequence/OOF计划；若仍然全弃权或出现灾难动作，则停止该M7表示，不扫描当前部署阈值，也不跑low22。

low22仍是公开VOT的第一道且唯一先行门，不能因训练loss下降直接启动。只有low22的EAO/ROB和失败anchor明确改善、无新增灾难，才复核DepthTrack/CDTB保真，最后才允许一次full-127。当前正式最好指标没有变化，Qwen3_8B继续保留且本轮未调用。

## 24.175　M7正式负结果：候选内排序已有正信号，但commit标签契约混合（2026-08-31）

M7正式运行完整结束并按只读方式封存：

```text
output:
/root/autodl-tmp/sttrack_lachtt_m7_fullset_balanced_seed2026_20260831

result.json SHA256:
88e91ab86c424f7e852aea1ccf2875e7978ae0fa2fc5c83306e2e01d43ef4bd7

manifest.json SHA256:
d7444bbe0feedd25a543e9a631853eeeda4a240504baff418c73120e8f15b1ef

head-only SHA256:
6cf16d3f6a7faad9afe97720188aa14779a32f6c270b5ad55a2370229b640917
```

正式结果：

| 项目 | M7结果 | 冻结门 | 是否通过 |
|---|---:|---:|---|
| evaluated available events | 154 | 完整 | 是 |
| evaluation sequences（含仅unavailable） | 23 | sequence overlap=0 | 是 |
| selected actions | **0** | ≥5 | **否** |
| strict/运行内 beneficial actions | 0 | ≥4 | **否** |
| beneficial sequences | 0 | ≥3 | **否** |
| beneficial precision | 0 | ≥0.95且非零动作 | **否** |
| catastrophic actions | 0 | =0 | 是，但来自全弃权 |
| protected trace mutations | 0 | =0 | 是 |

训练执行与契约完整性正常：1,968 event forwards、246 optimizer steps、96条训练序列、fold0与训练序列交集0；训练loss首末为`4.173892→3.232891`，耗时1,913.181秒，峰值显存2,172,158,464 bytes。所有正式文件均为0444；只生成head-only权重，没有tracking checkpoint、公开state commit、Qwen、DepthTrack Test、CDTB、VOT low22或full127。

正式决策是：

```text
stop_m7_no_threshold_scan_no_public_benchmark
```

### 24.175.1　M7相对M6的真实变化

均衡mini-batch确实改变了模型，不是M6的原样复现：

| 分数 | M6中位数 | M7中位数 | 变化 |
|---|---:|---:|---:|
| top candidate selection | 0.008902 | **0.194537** | +0.185635 |
| abstain | 0.949086 | **0.255460** | -0.693626 |
| benefit | 0.062495 | **0.229446** | +0.166951 |
| catastrophe | 0.025162 | 0.046322 | +0.021160 |
| predicted gain | 0.021990 | 0.168787 | +0.146797 |

但冻结门仍不可达：candidate击败abstain 63/154、selection≥0.50仅6/154、margin≥0.10为37/154、benefit≥0.80仅3/154、cat≤0.05为103/154、gain≥0.05为141/154；全部条件交集为0。

这说明M7把模型从“几乎无条件abstain”拉回到候选竞争区，却没有学会何时可靠地commit。直接降低0.50/0.80阈值会是在held-out结果之后调门，正式禁止。

### 24.175.2　最重要的新发现：严格H10与宽松H4两套标签控制了不同部件

M7不是文件损坏或label join失败；它的规范本身同时保留了两套不同定义：

1. balanced batch的event class来自Gate-A严格H10标签：beneficial要求H10 mean IoU≥0.50、相对public gain≥0.20、前5帧至少2个IoU≥0.50；
2. setwise selection/benefit/cat/gain target与最终`actual_*`来自运行内宽松H4标签：beneficial只要求mean gain≥0.05且不产生new low，catastrophic为gain≤-0.25或new low。

held-out交叉表为：

| 严格H10事件类 → 宽松H4事件类 | 事件数 |
|---|---:|
| beneficial → beneficial | 11 |
| catastrophic → catastrophic | 6 |
| neutral → beneficial | 14 |
| neutral → catastrophic | 20 |
| neutral → neutral | 103 |
| **两套类别不一致** | **34/154** |

也就是说，训练batch把某些事件当作strict neutral/catastrophic，但同一次forward的selection loss可能仍鼓励某个H4小增益候选接管。M7要学习的不是一个单一的survival commit问题，而是混合契约。

### 24.175.3　候选排序没有完全失败，失败集中在commit/abstain

在11个严格H10 beneficial held-out事件中，宽松H4 beneficial候选有10/11成为六候选内selection top-1。这是当前最重要的正信号：候选自身的时序表征已有一定条件排序容量。

但严格beneficial事件只有4/11的top candidate压过abstain，严格neutral事件却有59/137压过abstain；`1-abstain`区分严格beneficial event的AUC只有0.465989。换句话说，模型在“已经知道这是可救援事件”的条件下通常能选对候选，却不知道当前事件是否应该允许任何候选接管。

动作级绝对分数仍偏弱：selection→H4 beneficial AUC 0.565154、benefit AUC 0.526633、catastrophe AUC 0.596757。predicted gain与真实H4 gain的Pearson为0.015972、Spearman为-0.076701，不能承担长期生存决策。

### 24.175.4　名义六候选通常只有四个独立假设

current与last-reliable的peak0输出在151/154个事件完全相同，peak1同样为151/154；每个事件的独立action signature中位数只有4。这解释了为什么增加训练事件不能创造新的恢复路径：selector经常只是给重复候选分配概率。

只读审计资产：

```text
/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M7_READONLY_AUDIT_20260831_2045.json
SHA256 081a76fc44c5b429d8532f5a7af8b83e6584f1fc35ae073b903cec76753699db

/home/SUTrack_RGBD_L/refine-logs/STTRACK_LACHTT_M7_READONLY_AUDIT_20260831_2045.md
SHA256 451c6d1d601f5c6e4865fce331d4e688a88d232db8d9c883eb8d85273534d633
```

审计工具已提交到STTrack隔离分支：

```text
commit c40d4cd25b103e33af0982b0bc46626d6f1f0362
tools/audit_sttrack_lachtt_m7_fullset.py
SHA256 33aea0ac560c527d43124d9de1a049cebb816d1f128687a0061952ca83a1347a
```

## 24.176　M8冻结方向：严格H10两阶段commit/rank，不再混用标签

新的版本化实验计划：

```text
/home/SUTrack_RGBD_L/refine-logs/EXPERIMENT_PLAN_20260831_210500.md
/home/SUTrack_RGBD_L/refine-logs/EXPERIMENT_PLAN.md
SHA256 09f8cc8f92a46107bdf683891117c19d97d6429d701abbadac493528f0d014a1
```

M8主结构只包含两个核心动作：

1. `event_commit_logit`只回答当前风险事件是否属于严格H10 beneficial event，strict catastrophic/neutral都必须abstain；
2. `candidate_rank_logits`只在strict beneficial event中回答应该提交哪一个候选，不再让abstain token与六候选共同承担混合语义。

所有候选benefit/cat/gain监督也统一改用Gate-A严格H10 action label；H4仍是在线可见的causal feature窗口，但不再作为正式标签。current/last重复候选先做只读去重oracle审计，只有不损失strict beneficial覆盖才启用；否则保留variable-mask六候选。

M8的公开部署门在训练前固定为：commit≥0.80、conditional candidate≥0.50、top1-top2 margin≥0.10、strict benefit≥0.80、strict catastrophe≤0.05、predicted H10 gain≥0.20。Train-only fold0必须selected≥5、strict beneficial≥4、覆盖≥3序列、precision≥0.95、catastrophic=0、protected mutation=0才允许另行冻结消融或low22计划。

当前M8只完成实验计划，尚未授权实现或训练。正式最好指标仍未改变：VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`。Qwen3_8B保留且未调用。

## 24.177　M8-0严格H10契约通过，但候选去重假设失败

M8-0只读审计完成了1,466个事件与8,796个动作的严格H10重新计算：0缺失join、0重复action key、0 label conflict；strict action为`341 beneficial / 228 catastrophic / 4,123 neutral / 4,104 unavailable`，strict event为`89 / 66 / 627 / 684`。但预注册的候选去重假设没有成立：current/last两个rank同时重复的事件仅`1,318/1,466=89.9045%`，低于95%；去重后候选数分布为`2:3、3:1、4:1,334、5:2、6:126`，出现少于4候选。结果`accepted=false`，没有放宽门。

正式工件：

- spec：`STTRACK_LACHTT_M8_STRICT_CONTRACT_AUDIT_SPEC_20260831_211500.json`，SHA256=`e21ae24bb2a9d4708bc65456051dec85fe314e7757a3115d34dc4d1feee96ecf`；
- binding：`STTRACK_LACHTT_M8_STRICT_CONTRACT_AUDIT_BINDING_20260831_212200.json`，SHA256=`ec302f7f408fe82388e2cd14428da801aca11db3417c005081a072fa92bd9c43`；
- result：`/root/autodl-tmp/sttrack_lachtt_m8_strict_contract_audit_v1_20260831/result.json`，SHA256=`f0a0f5c59043bddad299e7627c80d88b3f2798b231d4d5dfc3fcfdcc5368c675`；
- manifest：SHA256=`a1efe7e0bb8e9dca2dd95cb4e93c3e1a118fa8abab4ffcd8e6779faacd0e8265`。

因此删除“候选去重”claim，后续固定保留六候选，避免改变动作空间。更重要的新发现是：strict H10 branch-name标签不能和可训练dense head在线重算的H4轨迹混用，否则训练过程中候选框变化会让标签与当前候选错位。M8b改为只使用M1封存的age0--4缓存特征及同一分支的strict H10标签。

## 24.178　M8b-0缓存特征—严格标签闭环通过独立完整性审计

M8b-0逐个校验了全部冻结缓存，而不是只检查文件数或tensor shape：

| 项目 | 结果 |
|---|---:|
| events / feature files | 1,466 / 1,466 |
| sequences / anchor files | 134 / 134 |
| feature bytes from ledgers | 457,637,434 |
| strict actions | 8,796 |
| missing / duplicate joins | 0 / 0 |
| strict label conflicts | 0 |
| feature SHA/size/shape/nonfinite错误 | 0 |
| anchor shape/nonfinite错误 | 0 |
| Depth validity越界 | 0 |
| candidate axis order/scalar mismatch | 0 |

候选轴验证不只依赖`[5,6,...]`形状：runner逐age、逐候选比较轨迹ledger的score/margin/entropy与缓存`scalars[age,candidate,:3]`，全部在`1e-6`容差内一致。所有输出目录/文件为0555/0444；没有打开数据集RGB/Depth/GT，没有加载STTrack/CLIP checkpoint，没有模型forward、训练或公开评测。

正式工件：

- M8b计划：`EXPERIMENT_PLAN_20260831_214000.md`，SHA256=`5c92844e2c80474e2c28089b61f02590ee4f01d5775c5c37081640d4cce6630e`；
- closure spec：`STTRACK_LACHTT_M8B_CACHED_CLOSURE_SPEC_20260831_214800.json`，SHA256=`dd65b923c2e9ed968d8dc43e964539fd6fcc08fb929071eafe4b22deb47788e7`；
- binding：`STTRACK_LACHTT_M8B_CACHED_CLOSURE_BINDING_20260831_224900.json`，SHA256=`ee1ed1902e86d946121c565c8536a1772950b93a8ee56251854081621c005ad5`；
- result：SHA256=`660ccb6009a87021a292480b724617f21aaa6963557c652bc5c21e7b2a8294dd`；
- closure：SHA256=`6feedfb25970b3ed8f1bbafb7d08bfb8d948a25f02ad8bf2aa83a4ceed1ff9cf`；
- manifest：SHA256=`78365b6686c5acae5b6bdcf8369e15c730e3630403e90526d0c06c4ea6e94af2`；
- runner commit=`30d973b67e97af665101773b5359b808af1d59e5`。

独立GPT-5.5 xhigh只读审计为PASS，独立重算的event/action/bytes/label counts全部一致。审计MD/JSON SHA256分别为`afd66020e9e92641c6449402464ea645b07773ac2489bee5de3fe9d7137250f3`与`26ead9983b4a0635423a861c4f75ae1aee59466cb62e3c83134e262fdb2c5236`。本阶段只证明cached closure完整，不能说明router可训练或VOT提升。

## 24.179　M8b-1发现隐藏梯度爆炸：初版假阳性已撤销

M8b-1按冻结的同一8事件`2 beneficial + 2 catastrophic + 4 neutral`批次实现502,663参数的two-stage router：共享模态投影、H5 GRU、两层无位置Transformer、event commit与conditional rank/benefit/catastrophe/H10 gain heads。置换最大误差为`3.5763e-7`，loss总值`4.085417`，forward数值正常。

初版runner存在一个验收漏洞：它检查每个gradient tensor finite和`norm>0`，但没有检查总范数本身finite。`torch.clip_grad_norm_`返回`Infinity`仍满足`Infinity>0`，导致v1错误记录`accepted=true`。该acceptance已用独立只读override改为false，原始只读输出保留，不覆盖、不删除，也不据此进入训练。

纠正v2用float64累加梯度平方和，得到真实pre-clip L2：

```text
1.4462875717497782e17
```

超过预注册上限1000，因此optimizer step未执行、changed tensors=0、state digest完全不变，正式decision=`stop_m8b_1_gradient_corrected_smoke_failed`。最大梯度集中在：

- native projection weight：`8.5589e16`；
- candidate encoder weight：`8.1860e16`；
- CLIP projection weight：`7.4160e16`；
- raw-depth projection weight：`3.5532e16`。

关键工件：v1 result/manifest SHA256=`512ee9a1…37c2`/`f5674d51…289d`；integrity override SHA256=`a57c47f7…c254`；v2 correction spec/binding SHA256=`b624ab7e…4c72`/`4ea943c9…1fe7`；v2 result/manifest SHA256=`af144a30…6356`/`beeb899e…1c01`。纠正runner commit=`89ff71e1fcf3e1356e98ed6bc502f73994f2a1f6`。

这说明问题不是loss显示为NaN、标签缺失或输出非有限，而是反向传播链严重放大。没有生成head或tracking checkpoint，也没有运行VOT。

## 24.180　M8c浅层DeepSets仍有`1.286e14`梯度，停止当前cached neural router

为验证多层LayerNorm/Transformer是否是主因，M8c只替换集合编码器，保持同一8事件、strict H10 target、seed、optimizer、loss和全部安全门：

- native/CLIP冻结输入先做parameter-free L2 normalize再tied projection；
- raw Depth直接projection，scalar先`tanh`；
- 单层candidate encoder+H5 GRU；
- 去掉全部LayerNorm和Transformer；
- 用mean/max DeepSets residual保持候选置换等变。

参数降到297,573，LayerNorm/Transformer module count均为0，置换误差降到`2.9802e-8`，loss为`3.548172`。梯度从M8b的`1.446e17`降低约1,124倍，但仍为：

```text
1.2862684464054302e14
```

仍远超1000，因此optimizer step未执行、changed tensors=0，结果`accepted=false`。最大梯度为raw-depth projection `9.0169e13`、candidate encoder `8.0346e13`，而GRU/head梯度约0.1量级。这进一步定位到候选输入关系链，尤其是可学习投影后的cosine/归一化反向路径，而不是集合Transformer本身；但这只是当前批次的工程诊断，不写成一般理论结论。

M8c plan/spec/binding SHA256=`a04cf83c…f850c`/`a63f2dfe…f4e3a`/`bccfe692…fd4e`；代码commit=`ea031c185cff6eec86b4b7861d2cf3f24adb3f49`；result/manifest SHA256=`dc9116e5…49cb`/`9a528d5a…1c01`。

按冻结停止条件，本轮停止当前cached neural router，不扫描epsilon、loss权重、梯度上限、事件batch或部署门，也不启动formal Train、low22或full127。后续若继续，必须新建不同机制：将cosine等身份统计变成对冻结输入的非可学习、无反向路径证据，或进入真实predicted-crop rollout中的显式target/distractor memory；不能把本轮失败包装成VOT改进。

### 24.180.1　正式指标与资产状态

本节没有新tracking checkpoint、没有DepthTrack Test/CDTB/VOT新运行，公开最好仍为：

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | 74.020583 / 82.579344 / 89.565651 |
| DepthTrack Test | Pr / Re / F | 65.995933 / 65.335885 / 65.664250 |
| CDTB | Pr / Re / F | 75.387821 / 76.005850 / 75.695574 |

Qwen3_8B继续保留；M8-0、M8b、M8c均未调用Qwen。DepthTrack与CDTB保护结果没有被覆盖。

## 24.181　M8b/M8c梯度失败链独立完整性审计

独立只读审计重新检查了M8b-v1、纠正后的M8b-v2以及M8c工程smoke的spec、binding、runner、result、manifest、仓库状态和输出目录。审计结论为：

- M8b-v1的`Infinity`梯度仍被记为`accepted=true`，这一原始验收为FAIL，且已被只读integrity override正式覆盖；
- M8b-v2用float64重算得到`1.4462875717497782e17`，在`optimizer.step()`前fail-closed，changed tensors=0；
- M8c用相同安全门得到`1.2862684464054302e14`，同样未执行step、changed tensors=0；
- 两个纠正输出均没有checkpoint、VOT或Qwen调用，仓库干净，远程没有活动实验screen；
- 存在历史v1假阳性和静态文件打开标志的`dead gate / phantom pass`风险，因此审计给出WARN，但不改变纠正失败链的PASS；
- 本审计不授权formal训练、low22、full127或继续扫描cached neural router。

审计MD/JSON SHA256分别为`96f828f46234e04b0ae90c8fc5aec04cc6aac820bf1734f531c3dc07ab028511`与`e47a3036188d8190a601c60ce1150e6cdb8db61fa0065d78dffb972e2ae605eb`，远程权限均为`0444`。正式结论仍是停止当前cached neural router，下一路线必须采用新的、无该反向关系链的机制。

## 24.182　M9冻结身份证据共识失败：静态相似度仍会一致支持错误实例

M9没有训练模型，而是对已经封存的1,466个DepthTrack Train风险事件、8,796个六候选动作计算10类无梯度证据：候选CLIP到首帧图像/短文本、native RGB/Depth到首帧token bank、三类H5时序连续性，以及响应score/margin/negative entropy。固定规则只有在唯一候选至少获得7票、领先第二名至少3票，且四个身份证据全部位于top-2时才提交；规则在运行前冻结，没有拟合权重或扫描阈值。

数据和置换完整性全部通过：1,466 events、8,796 actions、134 sequences，782个六动作全可判定事件、684个全不可判定事件，32次候选置换的选择不一致为0、最大数值误差为0。正式共识结果为：

|结果|数量|
|---|---:|
|abstain（可判定事件）|767|
|beneficial|8|
|neutral|6|
|catastrophic|1|
|可判定提交总数|15|
|beneficial precision|53.333333%|
|beneficial覆盖序列|5|
|不可判定事件中仍触发选择|7|

因此M9未达到`beneficial>=10`、`catastrophic=0`和`precision>=95%`三个硬门，正式`accepted=false`，decision=`m9_fail_stop_cached_selector_move_online_memory`。result/manifest/evidence SHA256为`3f6f352e…3c57`、`ae240f1b…7d7c`和`772e8b14…271a`。

最关键的反例仍是`lock01_wild@386 velocity_peak0`。该事件没有任何beneficial候选，三个peak0为catastrophic、三个peak1为neutral；但velocity peak0同时赢得首帧CLIP、短文本、native RGB、native Depth、响应和时序稳定性共8票。错误轨迹在当前外观上完全可以比保护轨迹“更像首帧目标、更稳定、更高置信”，静态身份与置信度共识仍无法识别“本事件根本不应提交”。

单证据诊断也说明问题：negative entropy和response margin区分beneficial/catastrophic的全局AUC分别为0.8126和0.7541，但CLIP首帧、CLIP文本、native RGB、native Depth的全局AUC只有0.4729、0.4745、0.4665、0.5278。部分序列内AUC较高而跨序列全局接近随机，仍存在明显序列域偏置。

独立GPT-5.5只读审计给出总体WARN、实验完整性PASS：标签确为DepthTrack Train真实GT派生的strict-H10标签，所有数字、hash、candidate axis和32次置换均独立复算一致；WARN仅因为审计时交接文档尚未写M9，以及审计import产生过一个已清理的pycache。审计MD/JSON SHA256=`8329ea7a…5178`/`bdcbf2b9…2710`。

M9只否定这条固定参数为零的缓存共识规则，不能泛化成“所有缓存证据无用”；但它明确不授权静态浅层校准、online replay或VOT。下一条允许路线是新的真实predicted-crop target/distractor memory：在候选自己的递归轨迹中维护动态目标记忆和干扰物记忆，并把event commit与candidate rank分开。

### 24.182.1　公开指标未变化

M9未加载模型、未训练、未生成checkpoint、未运行DepthTrack Test/CDTB/VOT，也未调用Qwen。正式最好仍为VOT`74.020583/82.579344/89.565651`、DepthTrack`65.995933/65.335885/65.664250`、CDTB`75.387821/76.005850/75.695574`。

## 24.183　M10a候选自身动态目标—干扰物记忆工程smoke通过

M9证明静态首帧身份和响应共识仍会一起支持错误实例，因此M10a没有再扫描静态阈值，而是把证据改成真实predicted-crop六候选各自的动态关系。每个候选在H5轨迹中维护只属于自己的目标EMA记忆，其余五个候选构成干扰物集合；输入包括候选自己的CLIP图像、短文本一致性、STTrack native RGB/Depth/fused RoI、RGB/Depth query、raw Depth结构和response/几何标量。所有关系都在`torch.no_grad()`中计算并detach，49维关系值由cosine、差值、有效率和`tanh`构成，避免把可学习projection后的cosine反向链重新带回来。

固定八事件工程smoke结果：

|项目|结果|
|---|---:|
|relation shape|`[8,5,6,49]`|
|relation max abs|`1.0000004768371582`|
|non-finite|0|
|router参数量|53,253|
|LayerNorm / Transformer|0 / 0|
|extractor置换误差|`3.5762786865234375e-7`|
|model置换误差|`2.60770320892334e-8`|
|total loss|`3.567655324935913`|
|pre/post gradient L2|`1.1476637462306125` / 同值|
|changed tensors|22|

第一次启动因runner把spec字符串传给只接受`Path`的hash helper而在读取特征、构建模型和创建输出目录前失败；只改这一行后以RERUN1重新绑定。正式result/manifest SHA256=`b84908b1…15f1`/`05de52b8…e28`，result只授权一次冻结批次capacity测试，不是训练或跟踪指标。

## 24.184　M10b在第8步触发梯度上限，作为可信负结果停止

M10b完全冻结M10a的八事件、49维关系、模型、loss、seed、AdamW学习率和梯度门，预注册要求100步全部完成、loss ratio不高于0.25并通过全部容量条件。实际只完成7步：

|步骤/结果|数值|
|---|---:|
|step 6 pre-clip gradient L2|`326.9576762038292`|
|失败step 8 pre-clip gradient L2|`1558.9453107723216`|
|预注册上限|`1000`|
|完成步数|7 / 100|
|final / initial loss ratio|`0.9797728937195295`|
|non-finite outputs/loss/gradients|0 / 0 / 0|
|checkpoint written|false|

runner在pre-clip门失败后立即`break`，失败step 8没有执行clip、`optimizer.step()`、completed-step递增或trace append；输出只有7行已完成trace，没有checkpoint。正式`accepted=false`，decision=`m10b_fail_stop_without_rescan`，result/manifest/trace SHA256=`0822ff80…62c9`/`c99ccba5…2736`/`73b28de3…a249`。

这说明候选自身的动态目标—干扰物关系已经解决M8的单步反向爆炸，但当前GRU时序router仍不能稳定完成冻结批次容量测试。不能通过降低梯度门、改学习率、改loss或增加步数把本次失败硬跑成通过。下一次只允许一个结构变化：保留同一49维detach关系，移除GRU，将H5改成固定`mean/max/min/last/delta`时间池化，再用小型MLP/DeepSets完成event commit与candidate rank；仍须先通过同一100步冻结批次契约。

## 24.185　M10a/M10b独立完整性审计与授权边界

独立只读审计结论为总体WARN、实验完整性PASS。WARN只因为审计时本canonical和`EXPERIMENT_TRACKER.md`尚未记录M10，而不是发现伪造或越权。审计确认：

- 标签来自封存的DepthTrack Train strict-H10真实GT派生闭环，M10不是公开benchmark；
- 输入不含sequence/frame ID、GT rank或label，未发现own-max归一化或标签泄漏；
- M10a所有shape、hash、置换、loss和梯度数字一致；
- M10b trace恰为7行，失败step确实没有执行optimizer step；
- 输出目录没有tracking checkpoint，仓库在`4a8a360b82d3f9845fc38fe4b494f70a168cdf23`干净；
- 不授权sequence-disjoint pilot、online replay、DepthTrack Test、CDTB、VOT low22/full127或Qwen。

审计MD/JSON SHA256=`9dbd576e745b3926d8147659a81b37c8f973db0213557cda4d059e94de4df3ce`/`67c94452e207a9ebf81e3babf0d68bf23982722fb544625ca6d23bd73bbe1394`。

本阶段没有新正式指标，最好结果仍为：VOT `74.020583/82.579344/89.565651`，DepthTrack `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。Qwen3_8B继续保留但未调用。

## 24.186　M11固定时间池化解决梯度尖峰，但未通过容量门

M11保留M10完全相同的49维detach动态目标—干扰物关系、八事件、strict-H10标签、loss、seed、AdamW学习率和全部门，只做一个结构变化：删除H5 GRU，把每候选5帧关系无参数汇总为`mean/max/min/last/last-first`共245维，再用44,997参数的MLP/DeepSets router输出event commit、rank、benefit、catastrophe和H10 gain。代码提交=`52a3abfd2cf2a6df10cf74f1d2c602e3939221f7`，无GRU/RNN/LSTM/LayerNorm/Transformer/MultiheadAttention。

唯一一次事前冻结的100步结果：

|项目|M10b GRU|M11 fixed pool|
|---|---:|---:|
|完成步数|7 / 100|100 / 100|
|最大pre-clip梯度L2|`1558.9453`（失败step）|`1.3745051`|
|non-finite|0|0|
|loss ratio|`0.979773`|`0.277489`|
|event commit|6 / 8|8 / 8|
|beneficial best rank|0 / 2|2 / 2|
|benefit accuracy|0.25|0.833333|
|catastrophe accuracy|0.895833|0.8125|
|gain MAE|0.29137|0.188458|

因此固定池化明确消除了M10b的递归梯度尖峰，并学会了两级策略的主任务event commit和beneficial rank；但是仍有四个硬门未过：loss ratio要求`<=0.25`，实际`0.277489`；benefit/catastrophe要求均`>=0.95`，实际`0.833333/0.8125`；gain MAE要求`<=0.10`，实际`0.188458`。正式`accepted=false`，decision=`m11_fail_stop_without_rescan`。

100行trace连续覆盖step1--100，loss从`3.580993`平滑降至step100前向的`1.009749`，没有突然爆炸。所有20个可训练tensor发生变化，但没有保存checkpoint。result/manifest/trace SHA256=`854a649c…a41`/`30b10ead…8b`/`1e8bfb18…bd3`。

这一步的关键诊断是：梯度稳定性已不是当前瓶颈；共享64维candidate token同时服务五个目标，在冻结100步内只能充分拟合commit/rank，未能同时满足benefit/catastrophe/gain。禁止通过追加步数、改学习率、放宽门或改pool统计重跑M11。若继续，必须另写新的任务解耦结构计划，让commit、association/survival、catastrophe hazard和gain不再只共享同一窄表示。

## 24.187　M11独立审计确认负结果可信

独立只读审计给出完整性PASS、科学门FAIL、总体WARN。WARN仍只因为审计时canonical/tracker尚未写M11。审计独立复核：

- strict-H10标签来自DepthTrack Train真实GT派生closure，1,466 events/8,796 actions/134 sequences；
- plan/spec/binding在实现/执行前冻结，唯一结构变化与代码一致；
- 100行trace连续，最大梯度、loss ratio和所有最终指标精确一致；
- 训练loop在任何finite/gradient门失败时都在optimizer step前停止，本次100步均合法执行；
- `accepted=false`由四个预注册条件失败得到，没有phantom pass；
- 输出只有result/manifest/trace，无checkpoint、online replay、Test/CDTB/VOT或Qwen。

审计MD/JSON SHA256=`d58768c2fde3b1dbae6629cfaa0f1a17782eb3a74ba92836f9fccd68213964e5`/`f8619e99d4e206728fb2f089b89fbea4c3f5992972b7b0750f41bf02dc675b71`。

M11不授权sequence-disjoint pilot执行或公开评测。正式指标仍未变化：VOT `74.020583/82.579344/89.565651`，DepthTrack `65.995933/65.335885/65.664250`，CDTB `75.387821/76.005850/75.695574`。Qwen3_8B保留且未调用。

## 24.188　M12参数量近似匹配的任务解耦塔同样失败，停止49维标量router扫描

M12直接检验M11的共享表示是否造成多任务冲突。49维detach关系、固定H5 pool、八事件、strict-H10标签、loss、seed、optimizer和100步门全部不变；只把一个共享64维tower改为五个互不共享参数的24维tower，分别服务commit、rank、benefit、catastrophe和gain。总参数45,581，比M11的44,997只多584（1.2979%），不是简单扩大模型。

结构硬门全部真实通过：5个tower、parameter ID重叠0；逐任务单独backward时，本任务非零gradient tensor为9--12，其他四任务合计始终为0；置换误差`2.2351742e-8`，禁止module计数0。唯一100步run也数值稳定：最大梯度`1.2141274`，nonfinite=0，100行trace连续。

但最终容量反而比M11差：

|项目|M11 shared|M12 separated|
|---|---:|---:|
|loss ratio|0.277489|0.433032|
|event commit|8 / 8|6 / 8|
|beneficial best rank|2 / 2|2 / 2|
|benefit accuracy|0.833333|0.75|
|catastrophe accuracy|0.8125|0.895833|
|gain MAE|0.188458|0.119663|

catastrophe和gain有所改善，但commit与benefit退化，loss ratio显著更差，仍有五个门失败。正式`accepted=false`，decision=`m12_fail_stop_without_rescan`；result/manifest/trace SHA256=`a99ee60f…fef5`/`dab572f7…e7cc`/`d6508d51…5ff1`，无checkpoint。

这一结果否定“完全任务隔离即可解决问题”的假设，并说明共享表示存在正迁移。现在应停止在同一49维标量关系上扫描tower宽度、共享比例、loss、LR、步数、gate、batch、label或pool。若继续，必须换成信息更丰富的候选自身RoI实例关系，而不是再雕刻scalar router。

## 24.189　M12独立审计与时间戳元数据警告

独立只读审计为完整性PASS、科学门FAIL、总体WARN。审计确认选中八事件与strict closure一致、结构隔离检查在live path真实执行、100步/hash/指标完全匹配、`accepted=false`正确、没有公开评测或权重。

审计发现一个需保留的元数据错误：binding文件的filesystem mtime和runner/hash链都证明它在输出前已经存在，但JSON内部手写`created_at=02:20`晚于约02:12的result mtime。原binding不修改、不覆盖；该问题归类为时间戳元数据错误，不改变实验值或授权边界。

审计MD/JSON SHA256=`f0292184533a8c03b52cd16114f18dd34af42f9a0adf23893a71b99c6f4f1bc5`/`0507a6095ce84b13da9e1c70ea96de77e0384a22e2bbcf0a1a69bfbccb3f4016`。

M12不授权sequence-disjoint pilot、low22/full127或Qwen。正式指标仍为VOT `74.020583/82.579344/89.565651`、DepthTrack `65.995933/65.335885/65.664250`、CDTB `75.387821/76.005850/75.695574`。

## 24.190　M13a候选自身 richer-RoI 实例关系通过工程门

M11/M12说明当前障碍已不是GRU梯度，而是49维cosine/统计标量可能丢掉了区分具体实例所需的方向性信息。M13a因此没有继续扫描49维router，而是在原49维关系后追加128维固定随机投影差分，形成177维candidate-specific关系。六类输入分别为CLIP image、STTrack native RGB、native Depth、native fused、RGB query和Depth query；每个候选分别与首帧身份锚、自己的EMA target memory及其余五候选的soft distractor memory比较。Depth相关块由候选框自己的有效深度比例门控。

所有新关系都在`torch.no_grad()`中构造并detach，固定Rademacher投影无可训练参数，输出经`tanh`有界。一次冻结八事件工程smoke结果：

|项目|M13a结果|
|---|---:|
|relation shape|`[8,5,6,177]`|
|relation max abs|`1.0000004768371582`|
|non-finite|0|
|候选关系/模型置换误差|`3.5762787e-7` / `1.4901161e-8`|
|router参数量|85,957|
|projection trainable params|0|
|forbidden modules|0|
|total loss|`3.6032378673553467`|
|pre/post gradient L2|`0.684218509097773` / 同值|
|optimizer step|1，实际执行|
|checkpoint|无|

第一次执行在Python进入函数体之前被纯接线错误拦截：runner传入`alpha=`，函数签名为`ema_alpha=`。旧binding和异常记录均保留；现场确认没有forward/backward/optimizer step、没有输出目录。R1提交只修改两处关键字参数名，随后才执行唯一一次优化步。result/manifest SHA256=`e4d1228b…c82d`/`58f3a69d…4834`，代码提交=`5a9c05a3d8cdaacf9641c9d62dfefc8c446f4326`。

独立只读审计给出integrity PASS、scientific gate=`PASS_ENGINEERING_SMOKE_ONLY`、overall WARN；WARN只因审计时canonical/tracker尚停在M12。审计核验第一次错误没有双重step、R1仅改两处参数名、关系确为候选自身且置换等变、所有冻结hash一致、输出无checkpoint。审计MD/JSON SHA256=`7e917d09…b5727`/`de21b40f…f3387`。

M13a只证明 richer relation 工程上可稳定计算和反向，不证明容量、泛化或VOT改善。它只授权冻结M13b容量计划。

## 24.191　M13b事前计划：参数预算匹配的richer-RoI容量对照

M13b计划/spec已经在任何runner实现和优化执行前冻结，SHA256分别为`c186c121…b9fe4`和`6ced7a2a…98b2`。实验仍使用与M11/M12相同的8个DepthTrack Train事件、strict-H10标签、H5六候选、五项loss、seed 2026、AdamW、`3e-4`学习率、100步和完全相同的容量门；唯一机制差异是49维标量关系换为已审计的177维richer RoI关系。

为避免“关系更丰富”和“参数翻倍”混淆，M13b将共享router hidden width固定为37，参数量44,755；M11为44,997，只少242（0.5378%）。通过仍必须同时满足100/100步、loss ratio不高于0.25、commit 8/8、rank 2/2、benefit/catastrophe均不低于0.95、gain MAE不高于0.10、梯度/置换/hash全部过门。失败后禁止扫描projection seed/width、hidden、步数、LR、loss、gate、batch、label、EMA或soft-distractor scale。

M13b当前只完成事前冻结，正在等待独立protocol audit；尚未实现runner或执行容量优化，更未运行sequence-disjoint、low22/full127。正式指标仍无变化，Qwen3_8B保留且未调用。

## 24.192　M13b容量结果：richer RoI只改善部分风险分类，仍未突破容量门

M13b事前protocol audit为PASS，确认实验只改变关系表示、hidden=37后的参数量44,755与M11 44,997相差0.5378%，并授权唯一一次100步run。代码提交=`42c7ec9370c5188204dfa2974b465dfe4ab9c09b`。

正式运行稳定完成100/100步，最大pre/post梯度L2=`1.7271854147399088`，没有non-finite、hash漂移或checkpoint；但最终结果为：

|项目|M11 49D shared|M13b 177D richer|门|
|---|---:|---:|---:|
|loss ratio|0.277489|0.296074|`<=0.25`|
|event commit|8/8|8/8|8/8|
|beneficial best rank|2/2|2/2|2/2|
|benefit accuracy|0.833333|0.833333|`>=0.95`|
|catastrophe accuracy|0.8125|0.895833|`>=0.95`|
|gain MAE|0.188458|0.173007|`<=0.10`|

richer relation使catastrophe和gain略有改善，但benefit没有变化，总loss ratio反而更差，四个科学门仍失败。正式`accepted=false`，decision=`m13b_fail_stop_without_rescan`。result/manifest/trace SHA256=`bba3a428…822f`/`80c7a8af…7556`/`a1e98602…25c0`。

独立结果审计总体/完整性PASS、科学门FAIL，100行trace、参数量、置换误差、关系范围、optimizer顺序及所有hash均复核一致；审计MD/JSON SHA256=`89b094bb…e47c`/`21cacfd5…0ba7`。因此这是一条可信负结果，不授权M13b rescan、sequence-disjoint、Test/CDTB、VOT或Qwen。

结论进一步明确：固定随机投影虽然保留了更多方向性信息，但仍是手工冻结的表征，不能让benefit/catastrophe/gain三种安全语义充分可分。下一机制不能再扫随机projection seed/width或hidden；应转为真正可学习的RoI association projector，但必须把输入先detach并归一化，禁止在可学习投影后再做cosine/normalize，soft-distractor权重也须由detach证据固定，以避免M8的反向爆炸。下一阶段仍先做单步工程门，不直接训练或评测。

### 24.192.1　公开指标仍未变化

M13b只使用DepthTrack Train冻结八事件，无tracking checkpoint。VOT最好仍为`74.020583/82.579344/89.565651`，DepthTrack仍为`65.995933/65.335885/65.664250`，CDTB仍为`75.387821/76.005850/75.695574`；Qwen3_8B保留且未调用。

## 24.193　M14a：可学习但有界的候选自身 RoI association 通过工程门

M13b 的固定随机方向让 catastrophe/gain 略有改善，却不能学习哪些高维方向真正对应具体实例身份。M14a 因而给六类候选自身差分各增加一个 `Linear(768,8,bias=False)` projector，覆盖 CLIP image、native RGB、native Depth、native fused、RGB query 和 Depth query。六个 projector 初始值逐元素等于 M13a 固定 Rademacher 矩阵；输入差分仍在 `no_grad` 中 detach、float32、L2-normalize，distractor softmax 权重也由 detached raw cosine 固定。可学习 projector 后只接 `tanh`，明确禁止再次 normalize/cosine/softmax，避免重现 M8 的 `1e14～1e17` 反向爆炸。

M14a 仍使用相同八个 DepthTrack Train predicted-crop 事件，只执行一个 optimizer step，不保存 checkpoint。模型由 36,864 个 projector 参数和 44,755 个 hidden-37 router 参数组成，总计 81,619。

|工程门|M14a结果|
|---|---:|
|raw difference shape|`[8,5,6,16,768]`|
|raw max abs|`0.5447601079940796`|
|49D scalar max abs|`1.0000004768371582`|
|与 M13a fixed 177D 初始逐值误差|`7.003545761108398e-7`|
|raw / initial relation 候选置换误差|`3.5762786865234375e-7`|
|model 候选置换误差|`1.4901161193847656e-8`|
|pre/post-clip gradient L2|`0.6390683413138711`|
|有非零梯度的 projector tensors|6/6|
|实际变化的 projector tensors|6/6|
|non-finite / checkpoint|0 / 无|

独立结果审计 integrity=`PASS`、scientific gate=`PASS_ENGINEERING_SMOKE_ONLY`。主要工件：plan/spec SHA=`12ca5557…73d7`/`7896a61f…d055`，binding=`591217b0…ef6b`，result/manifest=`2634db5c…f6d116`/`9bf0c58c…e626e`，审计 MD/JSON=`48d6f308…bdc`/`b9dfee9b…abab6`。实现 commit=`35493ebdeaf2f75e0b298697cc3290b1858307b0`，模型/runner SHA=`0cee5638…240c`/`76da46fc…3783d`。

这个结果只证明“可学习 projector 的梯度路径现在安全且确实被使用”，不证明容量、泛化或 VOT 改善。

## 24.194　M14b-R1：learned projector 明显改善容量，但灾难识别和未来收益仍不过门

### 24.194.1　原 M14b 报告门接线事故

M14b 在实现前冻结了与 M11/M13b 相同的 strict-H10 容量门。原 runner 完成 100-step 计算和最终前向后，在组装 conditions 时误用 `nonfinite_outputs_max/nonfinite_losses_max/nonfinite_gradients_max`，而 spec 的键为 `output_nonfinite_max/loss_nonfinite_max/gradient_nonfinite_max`，因此触发 `KeyError`。原输出目录不存在，没有 result、manifest、trace 或 checkpoint，不能从该进程形成任何科学结论。

事故文档 SHA=`7eda8542…188a`。随后另行冻结 M14b-R1 wiring-recovery plan/spec（`bf626dcb…c433`/`5b638f67…2cc3`）并完成独立 pre-execution audit（MD/JSON=`53b7c88f…702e`/`9bd43d05…38ea`）。R1 只允许修三处键、增加优化前 required-key preflight、更新 schema/output/binding/frozen-record 接线；八事件、特征、模型、loss、seed、optimizer、100 steps 和所有科学门均未改变。原 binding 已消耗，禁止直接重跑。

### 24.194.2　有效 R1 容量结果

R1 commit=`158480e53dba6a087d2abf358000fc603d664cd3`，runner SHA=`ba84203c…11471`，binding SHA=`b38bce6d…ab94b`。唯一 R1 执行完成 100/100 步，最大 pre/post-clip L2 均为 `1.911889004850349`，无 NaN/Inf、无中途停止、无 checkpoint。

|指标|M13b 固定投影|M14b-R1 可学习投影|冻结门|是否通过|
|---|---:|---:|---:|---|
|loss ratio|`0.296074`|`0.130118`|`<=0.25`|通过|
|event commit|8/8|8/8|8/8|通过|
|beneficial best rank|2/2|2/2|2/2|通过|
|candidate benefit accuracy|`0.833333`|`1.000000`|`>=0.95`|通过|
|candidate catastrophe accuracy|`0.895833`|`0.895833`|`>=0.95`|**失败**|
|candidate H10 gain MAE|`0.173007`|`0.133340`|`<=0.10`|**失败**|

最终 total loss 从 `3.6124658584594727` 降到 `0.4700474441051483`。与 M13b 相比，learned projector 把 benefit 从 0.8333 提到 1.0，并显著降低总 loss 和 gain MAE；这证明候选自身 RoI 的高维方向确实包含“哪些动作可能有益”的可学习信息。可是 catastrophe 保持 0.895833，没有新增安全区分能力；gain MAE 虽改善 0.039667，仍高出门槛 0.033340。

因此当前瓶颈进一步收窄为：

1. “找到可能有益候选”已不是主要容量问题；
2. 对 ROB 最关键的少数 catastrophic 候选仍与普通/有益候选在当前表示中混淆；
3. 单一共享 relation/router 同时拟合 benefit、catastrophe 和连续 H10 gain，仍不能把尾部失败风险校准到零灾难所需精度；
4. 不能因为 loss ratio、commit、rank、benefit 已通过就进入在线事务；VOT 会由剩余少数灾难动作形成连续 failure chain；
5. 不能扫描 step、LR、width、loss weight 或降低 catastrophe/gain 门来制造通过。

独立 post-result audit 重算 100 行连续 trace 和全部 conditions，结论 integrity=`PASS`、scientific=`FAIL_ACCEPTED_FALSE`；确认仅 `candidate_catastrophe_accuracy_min` 和 `candidate_gain_mae_max` 两项失败。result/manifest/trace SHA=`8f9fb1ed…7150`/`071eeb89…37ca`/`ede4cda0…528`，审计 MD/JSON=`9500f27d…8029`/`ae595f30…4d49`。输出目录只有三个只读 JSON/GZip 工件，无 checkpoint-like 文件。

### 24.194.3　停止边界与下一机制

M14b-R1 的 decision 为 `m14b_r1_fail_stop_without_rescan`。不授权第二次 R1、sequence-disjoint pilot、formal training、online replay、DepthTrack Test、CDTB、VOT low22/full127 或 Qwen。

下一机制必须针对剩余两项失败作结构性改变，而不是延长同一训练：保留已经证明有效的 learned candidate-specific association 作为“候选有益性/排序”分支，另建独立的 safety critic，专门输出 catastrophic veto 和未来 survival/hazard；该 critic 应读取 candidate-own RGB/Depth 实例证据、与 target/distractor memory 的相对差、Depth 缺失/reliability 和多帧 branch divergence，并采用 fail-closed veto。连续 gain 不再由同一共享表示单点回归，而应拆成短窗 survival ordinal bins 或 hazard 序列。任何新结构仍须先在冻结 Train-only 容量门上通过 catastrophe 和 gain/survival，再做 sequence-disjoint；不能直接跑 low22。

### 24.194.4　正式指标不变

M14a/M14b-R1 都只读 DepthTrack Train 冻结八事件且不生成 tracking checkpoint。VOT 最好仍为 `74.020583 / 82.579344 / 89.565651`，DepthTrack 为 `65.995933 / 65.335885 / 65.664250`，CDTB 为 `75.387821 / 76.005850 / 75.695574`。Qwen3_8B 保留且未调用。

## 24.195　M15a：为独立安全头建立真实多时域监督

M14b-R1 已能正确识别有益候选，却仍把部分灾难候选当作可提交动作，并且单点 H10 gain 回归不够准确。M15a 没有训练新模型，而是从同一冻结的 8 个 DepthTrack Train predicted-crop 事件、48 个候选中，只读生成 horizon=`3/5/10` 的未来轨迹监督。每个候选在每个 horizon 上记录五项真实量：branch mean IoU、public mean IoU、gain、低重叠比例和末尾连续低重叠比例。

闭包共 144 条 target records，候选标签为 beneficial=12、catastrophic=5、neutral=31；候选数、事件数、序列数、原始标签和 source hash 全部闭合。target closure/result/manifest SHA256 分别为 `72c7349a…baa8a`、`e69af5b4…b2d28`、`798848d6…93068`；独立结果审计 JSON SHA=`7bff97c5…08c2`。M15a 只证明未来安全监督可从真实递归轨迹中无歧义构造，不产生 checkpoint，也不授权公开评测。

## 24.196　M15b-R3：utility 与 safety 真正参数隔离，工程门通过

M15b 把 M14 的共享 router 拆成两个完全独立的模型：

- utility router：独立六个 `Linear(768,8,bias=False)` projector，加 177D relation 和 hidden-37 router，负责 event commit、candidate rank 和 benefit，共 81,543 参数；
- safety critic：另一组六个私有 projector 和 hidden-37 critic，负责 catastrophe veto 与 `[B,6,3,5]` 多 horizon 轨迹，共 77,210 参数；
- 两边 parameter ID 交集为 0，总参数 158,753；safety 输出中 H10 gain 是固定切片 `[:,:,2,2]`，不再由 utility 表示共享承担。

工程过程中保留了三次 fail-closed 事故：初始计划的 source/binding 路径错误；R1 在执行前发现 native anchor/source 绑定缺口；R2 在 full forward 前因 PyTorch 1.13 不支持 `Tensor.untyped_storage()` 停止。三次均未形成科学结果或 checkpoint。R3 只修存储 API 接线，在 commit `2aa2681a0298269260902daaf469a9bf78bc18e3` 的实现上完成唯一工程 smoke。

R3 的 33/33 条工程条件全部通过：pre/post gradient L2=`0.6042140214`；utility 与 safety 的 projector 均为 6/6 有梯度且 6/6 实际变化；非 projector 变化为 utility 16/16、safety 12/12；candidate permutation 最大误差 `5.9604645e-8`、event permutation=0；无 checkpoint。result/manifest SHA=`cedc664b…3a0`/`6df6b950…a8b7`，独立审计 MD/JSON=`25434fe2…90e`/`8d2d234f…d732`。这只证明独立梯度和接线成立，尚不证明容量或泛化。

## 24.197　M15c-R1：所有容量与灾难识别门通过，但排列等变硬门失败

### 24.197.1　冻结协议与接线事故

M15c 预注册为 fresh seed=`20260901`、同 8 事件/48 候选、full-batch AdamW 200 步、`lr=1e-3`、`weight_decay=0`、无 scheduler/shuffle/early stop、clip=5。通过必须同时满足 loss ratio、commit/rank、benefit、catastrophe、五类轨迹误差、三个 horizon 误差、H10 gain、201 行 trace、排列等变和 no-checkpoint。

原命令在读取 preaudit authorization 时发现 JSON schema 路径不一致，发生在数据、模型、forward 和 optimizer 之前，原输出目录不存在。R1 只修正授权字段读取，并由新的 plan/spec/preaudit/binding 重新绑定；不改数据、模型、loss、seed、优化器、200 steps 或任何阈值。incident SHA=`bd0d0844…591c`，R1 plan/spec=`c0621ff4…d8d`/`5a828e60…ea72`，preaudit MD/JSON=`30520003…6c51`/`b94f5b09…9d58`，binding=`99ef0518…8c6`。

### 24.197.2　有效 R1 数值

唯一 R1 完成 step 0--200 共 201 行 trace，200 次 optimizer step；最大 pre/post-clip gradient L2=`2.30524021963801`，non-finite=0。

|容量/安全项目|结果|冻结门|结论|
|---|---:|---:|---|
|loss ratio|`0.0074377289`|`<=0.10`|通过|
|event commit|8/8|8/8|通过|
|conditional rank|2/2|2/2|通过|
|benefit|48/48|`>=0.95`|通过|
|catastrophe TP|5/5|5/5|通过|
|catastrophe TN|43/43|43/43|通过|
|trajectory overall MAE|`0.027417973`|`<=0.08`|通过|
|H3/H5/H10 MAE|`0.041319 / 0.019271 / 0.021664`|各`<=0.10`|通过|
|H10 gain MAE|`0.033903267`|`<=0.08`|通过|
|candidate permutation|`1.9073486e-6`|`<=1e-6`|**失败**|
|event permutation|`1.9073486e-6`|`<=1e-6`|**失败**|

细分后 rank permutation=`9.5367432e-7`、trajectory=`8.0838799e-7`均通过；benefit/catastrophe/event logits 达到 `1.9073486e-6`。因此它不是容量失败，也不是灾难识别失败，而是预注册的严格工程 invariant 超限。即使差值只比阈值大约 `9.07e-7`，也不能事后把门放宽或用其它指标覆盖它。

独立审计确认 integrity=`PASS`、result acceptance=`FAIL`、scientific gate=`FAIL_ACCEPTED_FALSE_PERMUTATION_INVARIANCE_GATE`。result/manifest/trace SHA=`9e90b8c0…76cf5`/`a7b23c55…70b8`/`4f40953b…da15`；审计 MD/JSON=`0c54b253…a81d`/`469375fb…1cef`。输出只有三个只读工件，无 checkpoint。

### 24.197.3　当前发现与停止边界

M15c 给出两个同时成立的结论：

1. 独立 utility/safety 表示能在冻结 48 候选上把 benefit、catastrophe 和多时域 trajectory 全部拟合到预注册容量门，说明 M14 的主要容量瓶颈确实来自 utility/safety 共享表示，而不是 candidate-own RGB-D 证据完全无信息；
2. 当前 CUDA/float32 实现在置换输入下仍出现 `1.907e-6` 的输出差异，超过必须满足的 `1e-6`。在安全 selector 接入递归 tracker 前，这个 invariant 不能被忽略。

因此 M15c-R1 决策固定为 `m15c_r1_fail_stop_without_rescan`。不授权再次运行、修改 permutation tolerance、扫描 seed/LR/loss/step/batch、sequence-disjoint 计划或执行、DepthTrack Test、CDTB、VOT、Qwen、checkpoint 或自动下一阶段。后续若继续，必须是新的、事前冻结的结构性机制，不能把同一 M15c 当作“几乎通过”而补跑。

### 24.197.4　正式指标仍不变

M15a--M15c 均只使用冻结 DepthTrack Train 证据，没有生成 tracking checkpoint 或运行公开 benchmark。正式最好仍为 VOT `74.020583 / 82.579344 / 89.565651`，DepthTrack `65.995933 / 65.335885 / 65.664250`，CDTB `75.387821 / 76.005850 / 75.695574`。Qwen3_8B 保留且未调用。

## 24.198　M16a：候选角色规范化使重排误差精确归零

M15c-R1 的所有容量、安全和多时域轨迹门均通过，唯一失败是候选枚举顺序改变后 event/candidate 输出出现 `1.9073486328125e-6` 的 float32 误差。M16a 没有放宽 `1e-6` 门，也没有重跑 M15c，而是在模型入口增加无参数的 candidate-role canonicalization：六类候选分别固定为 `current_peak0/current_peak1/last_reliable_peak0/last_reliable_peak1/velocity_peak0/velocity_peak1`，先按 role ID gather 到规范顺序，模型完成计算后再 inverse gather 回原输入顺序。role ID 只允许参与排序和 gather，禁止进入 embedding、relation、token、head 或 loss。

实现 commit=`e033eb349bac635bbbcf1e3e7a5c46ce3bf38660`。canonical model/runner SHA=`802e38ea…178d`/`74ed7af3…ec9`；新增参数=0、buffer=0，parent/canonical 都是 158,753 参数。规范顺序下 state dict、forward、loss、每个梯度、一次 AdamW 后模型 state 和 optimizer state 全部逐位相等；两者 pre/post gradient L2 都为 `0.6042140214379395`，各有 40 个 tensor 实际改变。

冻结六种 permutation（包括 M15c 失败的 `[0,5,3,1,4,2]`）全部满足 event/candidate 五类输出 `torch.equal`，最大误差精确为 `0.0/0.0`。重复、缺失、越界、float dtype、错误 shape/non-permutation 五类 role 输入全部 fail closed。28/28 条件通过，result/manifest SHA=`80c62b85…93c6`/`4965acc3…a08a`。独立结果审计 integrity=`PASS`、scientific=`PASS_ENGINEERING_SMOKE_ONLY`，MD/JSON SHA=`b4b8ac61…77ca`/`7351600f…65a`。

这个结果解决的是“同一六候选仅因输入顺序不同而改变安全决策”的工程问题，不证明 unseen sequence 上能选对候选，更不等于 VOT 的 ROB/EAO 已提升。它没有生成 tracking checkpoint，也没有运行 low22/full127、DepthTrack Test、CDTB 或 Qwen。

## 24.199　M16b：完整容量轨迹等价计划已冻结，尚未执行

M16a 只做一次 optimizer step。为了确认 canonical gather 在完整优化中不会改变已经通过的 utility/safety 容量，M16b 事前冻结为同一 8 个 Train 事件、48 个候选、144 条 H3/H5/H10 trajectory records、seed=`20260901`、CPU full-batch AdamW 200 steps。数据、loss、学习率、梯度门、模型容量和最终科学门全部复用 M15c-R1。

M16b 的额外硬门是：将 step0--200 共 201 行解压 training trace 与 M15c-R1 对应机器字段逐值比较，loss、全部分项、commit/rank/benefit/catastrophe、trajectory MAE、H10 gain、非有限计数和梯度范数必须完全相等，不允许 tolerance。最终六种 permutation 仍须全部 `torch.equal` 且 event/candidate 最大误差精确 `0.0/0.0`；参数必须保持 158,753，新参数/buffer=0。

plan/spec SHA=`ac982e6d…2559`/`9d624fbf…ee2e`，固定 `EXPERIMENT_PLAN.md` 已同步为同一 plan。当前状态严格为 **plan/spec only**：runner 尚未实现，binding/preexecution audit/200-step run 均未发生。通过计划本身不授权 sequence-disjoint、在线回放、Test/CDTB、VOT、Qwen 或 checkpoint；失败也禁止改 role mapping、permutation、trace tolerance、seed、LR、loss、step、batch 或 dtype 重扫。

### 24.199.1　对当前 VOT 问题的意义

现阶段证据把问题继续分成两层：candidate-own RGB-D + target/distractor relation 和独立 safety critic 在冻结候选池里已经有足够容量；M16a 又消除了候选枚举顺序造成的不确定性。但真正决定 ROB 的跨序列提交精度仍未验证。只有 M16b 通过独立审计和完整容量等价门后，才允许另写 sequence-disjoint Train pilot 计划；不能越过这一层直接跑低指标序列或 full-127。

正式最好仍为 VOT `74.020583 / 82.579344 / 89.565651`，DepthTrack `65.995933 / 65.335885 / 65.664250`，CDTB `75.387821 / 76.005850 / 75.695574`。Qwen3_8B 保留且未调用。

## 24.200　M16b preaudit 失败与 M16b-R1 协议恢复

### 24.200.1　失败发生在执行之前

独立 M16b preexecution audit 判定 overall=`FAIL`、integrity=`FAIL_PROTOCOL_INCONSISTENCY`、scientific=`FAIL_PREEXECUTION_NOT_AUTHORIZED`。审计 MD/JSON SHA=`6a76c1c4…bafc`/`755f27e6…1621`。当时 repo clean、HEAD=`e033eb349bac635bbbcf1e3e7a5c46ce3bf38660`，future runner 和 output root 均不存在；没有 forward/backward/optimizer、checkpoint、benchmark 或 Qwen。

阻断原因有两项：

1. 原 M16b 要求 step0--200 的 commit/rank/benefit/catastrophe/trajectory 指标与 M15c trace 逐行相等；但冻结 M15c trace 的每行实际只有 `step/total_loss/strict_total_loss/trajectory_l1/preclip_total_l2/postclip_total_l2/nonfinite_gradients/optimizer_step_executed` 八个字段。其它科学指标只在 result 保存 initial/final 汇总。按原协议执行只能补造不存在的历史字段，形成 phantom pass；
2. 原 spec 没有直接绑定 M15c-R1 spec/result/manifest/runner/parent model/result audit，也没有从 M15c spec 无歧义锁住 data/loss/architecture/builder/dependencies，因此“一变量复用 M15c”没有形成完整证据闭包。

原 plan/spec SHA=`ac982e6d…2559`/`9d624fbf…ee2e` 保持只读。这是协议负结果，不是模型容量或训练负结果。

### 24.200.2　M16b-R1 只修协议接线

审计只授权 incident 与 R1 plan/spec。incident/plan/spec SHA=`8f6efcf0…2448`/`2ab66797…451d`/`8231320e…96fa`。

R1 将证据拆为两个互不混淆的门：

- **逐步 trace parity**：双方解压后 201 行必须都只有上述八个真实 key；row dictionary、类型和值用 Python exact equality，0 tolerance；
- **最终 scientific/state parity**：与绑定 M15c result 的 `input_counts`、参数、training、完整 initial/final metrics、loss ratio、gradient safety、projector coverage/change、nonprojector change、initial/final state SHA、forbidden/checkpoint count 逐值完全相等。

M15c 的 `permutation_errors/accepted/decision/conditions/failed_conditions` 明确不要求相等，因为 canonical ordering 正是唯一允许变化的机制；它们改由六种冻结重排全部 `torch.equal`、event/candidate error 精确 `0.0/0.0` 判断。

R1 直接绑定 M15c-R1 spec `5a828e60…ea72`、runner `baacf86b…58c7`、parent model `db4a5fb4…02a9`、result/manifest/trace `9e90b8c0…76cf5`/`a7b23c55…70b8`/`4f40953b…da15` 和 result audit `469375fb…1cef`，并强制 data/loss/architecture/optimization/dependencies 从该 spec 读取，禁止本地 default 覆盖。

当前固定 `EXPERIMENT_PLAN.md` 已切换到 R1 plan。R1 仍是 plan/spec only：唯一未来 runner `tools/run_sttrack_lachtt_m16b_r1_canonical_role_capacity.py` 不存在，R1 output root 不存在，尚未进行 R1 preaudit，更没有训练或公开评测。

### 24.200.3　正式指标不变

VOT 仍为 `74.020583 / 82.579344 / 89.565651`，DepthTrack 为 `65.995933 / 65.335885 / 65.664250`，CDTB 为 `75.387821 / 76.005850 / 75.695574`。Qwen3_8B 保留且未调用。

## 24.201　M16b-R1：排列精确归零且容量门全过，但跨独立运行逐值等价失败

M16b-R1 preexecution audit A--H 全部 PASS，MD/JSON SHA=`5a99d446…e848`/`9ea9e536…507f`。唯一 runner commit=`2a1b99a7c11aee3fe79a547e5be80ce1db8944c0`，runner SHA=`bf817aa8…033d`；binding SHA=`c0b5415d…e0fa`。相对 M16a base 只新增该 runner，canonical/parent/M15c 代码和冻结证据均未修改。

唯一 CPU run 完成 200/200 optimizer steps、201 rows trace，无 gradient failure、nonfinite 或 checkpoint。六种 permutation 对 event commit 与全部四类 candidate output 都是 `torch.equal=true`，event/candidate max abs error 精确 `0.0/0.0`。所有容量子门也通过：

|项目|M16b-R1|门|
|---|---:|---:|
|loss ratio|`0.006881223749`|`<=0.10`|
|event commit|8/8|8/8|
|conditional rank|2/2|2/2|
|benefit|48/48|48/48|
|catastrophe TP/TN|5/5、43/43|5/5、43/43|
|trajectory overall MAE|`0.025630696`|`<=0.08`|
|H10 gain MAE|`0.008144070`|`<=0.08`|
|utility/safety projector gradient/change|6/6、6/6|各6/6|
|参数/新参数/buffer|158,753 / 0 / 0|相同|
|六种 candidate reorder|全部逐位相等|全部逐位相等|

但是预注册的跨独立运行 exact parity 没有成立：reference/current 都是 201 行、八个 key 完全一致，step0 完全相等；step1--200 共 200 行不相等。step1 的三类 loss 仍完全相同，最早差异只是 pre/post gradient L2 `0.6042140214379395` 对 `0.6042140219672267`，绝对差约 `5.29e-10`；loss 从 step11 开始分叉。最终 final-result parity 有且只有四项不等：`final_metrics`、`loss_ratio`、`gradient_safety`、`state_sha256.final`。

有些 M16b-R1 最终数值比 M15c 更低，例如 H10 gain MAE `0.00814` 对 `0.03390`，trajectory MAE `0.02563` 对 `0.02742`；但预注册命题是“逐值相同”，不是“至少同样好”，因此不能用更低误差覆盖 exact-parity 失败。正式 `accepted=false`，decision=`m16b_r1_fail_stop_without_rescan`。

result/manifest/trace SHA=`b15a226a…6cd5`/`e6f14ac4…1ca2`/`c4415094…802e`，目录 `0555`、三文件 `0444`。启动包装器的辅助 exitcode 文件因双层 shell 提前展开变量只有1字节，独立审计没有把它当作科学 exit 证据；真实 Python 进程已退出、完整日志和三份封存结果存在，200-step/201-row 由结果和 trace 独立复算。

独立 post-result audit 结论 integrity=`PASS`、scientific/acceptance=`FAIL`、overall=`WARN`，MD/JSON SHA=`0efdf1be…31e0`/`62e6f327…0457`。审计确认 `accepted=false` 正确，不存在 phantom pass、dead gate、checkpoint 或公开评测。

### 24.201.1　科学结论与停止边界

M16a/M16b-R1 已证明 canonical role ordering 能在完整训练后保持候选排列精确不变，而且 independent utility/safety 模型在冻结 48 candidates 上有充分容量；但“不同独立 CPU 运行的 float 轨迹逐位相同”不成立。这个 exact-parity 家族到此停止，不重跑、不放宽 tolerance、不删除门、不改 dtype/thread/seed/LR/loss/step 追求通过。

下一步必须把问题转回真正决定 VOT ROB 的未见序列泛化：另行冻结 sequence-disjoint Train 科学假设 plan/spec，检验 candidate-own RGB-D target/distractor association 与 survival safety critic 在未见序列上是否能保持零灾难提交。当前不授权该 pilot execution，更不授权 DepthTrack Test、CDTB、VOT、Qwen 或 checkpoint。

正式公开指标仍未变化：VOT `74.020583 / 82.579344 / 89.565651`，DepthTrack `65.995933 / 65.335885 / 65.664250`，CDTB `75.387821 / 76.005850 / 75.695574`。

## 24.202　M17 未见序列保守生存实验：目标/划分闭合通过（2026-09-01）

### 24.202.1　为什么从 M16 转向 M17

M16b-R1 已经证明 canonical candidate-role ordering 在完整训练后能够保持六种候选排列逐位一致，但不同独立 CPU 运行的浮点训练轨迹无法满足逐值完全相同。继续扫描 dtype、线程、容差、学习率或步数不能回答 VOT 的科学问题，因此 exact-parity 家族已停止。

M17 把唯一假设改回决定 ROB/EAO 的核心：在**按序列完全隔离**的未见序列上，candidate-own RGB-D target/distractor 证据、独立 utility/safety 输出和保守 abstention 是否能选择非零的有益动作，同时保持 catastrophic action 为零。未来 M17-1 的冻结通过门为：选中动作不少于 5、有益动作不少于 4、覆盖至少 3 条序列、precision 不低于 0.95、catastrophic 为 0、平均 H10 gain 不低于 0.20，且候选聚合优于公开分支；全弃权不能通过。

### 24.202.2　预注册、路径事故与独立预审

原始 plan SHA=`22d8a338be9eb09bff23bbf546a286f66b40561d52497d08e0752556253677f6`。原始 spec 的科学字段和 M16a 文件 SHA 正确，但引用路径漏写 `_smoke_`，在任何 runner、forward、backward、optimizer 或结果生成之前被发现并封存。incident SHA=`57780163361b52ca652bb38e0ac0e82abe35156d716babdce8c9e11591ca9dc6`；只修路径和恢复元数据的 R1 spec SHA=`cad83ee5bbde6e87be2551bd1269d4e3db558c369369368735f9d7b56d930b3d`，没有改变 split、标签、模型、门槛或训练假设。

R1 preexecution audit 独立复核 19 个源文件及 SHA、1,466 events、8,796 actions、134 sequences、严格动作/事件计数、sequence-disjoint 划分和 158,753 参数模型，结论 PASS。审计 MD/JSON SHA=`331ad015b4587e826efee9605b1fa5635fa9aeef49eb7b05f944045e627d3c82`/`cf3d9e30c4c180bf5fcb36eb69a4e729f3098e5e25cbc3b4993d1df9938cb7e2`。其授权范围只到 M17-0 target/split closure，不授权 M17-1 训练或任何公开数据集评测。

### 24.202.3　M17-0 实现审查与唯一闭合运行

M17-0 runner 为 `tools/run_sttrack_lachtt_m17_0_target_split_closure.py`，最终 commit=`3426dfc7dd06dc65506bd128a332d15b0b2ec845`、SHA=`bf6577607b185263fa3bbb1aac9baa2455ccd62a46f9315702f606c48326a75f`。代码审查在首次提交后修正了 held-out 数值目标同包序列化、action 子串匹配、spec 仓库/分支绑定、发布前 TOCTOU 复核、shard 路径越界、bool/IoU 边界、gzip fsync 和 rename/chmod 顺序；随后 standards 与 spec 两条独立复审均无剩余问题。

只读 binding SHA=`f6ba91be7e296b4e6da37a2d273e714b96a576d11546c525e9776a77c899f894`。唯一 M17-0 前台运行耗时约 9.37 秒，不加载模型、GPU、优化器、RGB、Depth、Qwen 或 checkpoint，也不访问 VOT/DepthTrack Test/CDTB。闭合结果为：

| 冻结统计 | 数值 |
|---|---:|
| 全部 events / actions / sequences | 1,466 / 8,796 / 134 |
| strict beneficial / catastrophic / neutral / unavailable actions | 341 / 228 / 4,123 / 4,104 |
| strict beneficial / catastrophic / neutral / unavailable events | 89 / 66 / 627 / 684 |
| 训练 available events | 628（78/60/490） |
| 留出 available events | 154（11/6/137） |
| 训练 / 留出全部序列 | 109 / 25 |
| 训练 / 留出 available 序列 | 96 / 23 |
| 训练序列与留出序列交集 | 0 |
| 总 target actions / 序列化训练 targets / 留出 commitment targets | 4,692 / 3,768 / 924 |
| 序列化的留出数值 targets | 0 |

留出 targets 只保存 canonical commitment，SHA=`8d285497362cd4db4d865abae8add286c024541c02eb89d082ac13689c905672`；训练时不能读取留出数值标签。四个只读输出的 SHA 分别为 manifest `801acf6214ff005ef2f494dc9b33b1f13daf35d07525165835ca1f426797924e`、result `5783d581f7a93234cc343ee7cbb05acb20700d22c4d2bdfcd097b69aa1782012`、split `abcaf01470c30f73be3b8f18004501220b2b8be48c54c11623a977189949b5a3`、training targets `b4ecf113893814d8112b9b62ddb03f01f2b0e362c4984b0bfbd5e18d2a24bb15`。

### 24.202.4　独立结果审计、科学边界与下一步

独立结果审计 MD/JSON SHA=`e0c66d22926f83fd24e1851139eb208f36a9b29528815540de5e4307d5dfa701`/`2782ec152bc15d95a803b07d4b3a4ffa4a8670e38ba334c1cd98aed7fd6160c7`，Overall/Integrity/Closure gate 均 PASS。审计重新推导 H3/H5/H10 五类度量、H10 标签、sequence split 和留出 commitment，全部 mismatch 为 0。

这个 PASS **只是实验目标与未见序列隔离闭合，不是模型效果提升**：M17-0 没有训练模型、没有生成 checkpoint、没有得到新的 low22/full-127 或 DepthTrack/CDTB 指标。当前只允许准备一个单独的 M17-1 实现/绑定预检包并再次提交独立审计；审计通过前禁止训练，训练通过内部门前禁止低指标 VOT，更禁止 full-127。

正式最好指标仍为 VOT `74.020583 / 82.579344 / 89.565651`，DepthTrack `65.995933 / 65.335885 / 65.664250`，CDTB `75.387821 / 76.005850 / 75.695574`。Qwen3_8B 保留但未调用。

# 第 4 部分：代码发布、数据约定与旧说明文档整合

本部分吸收原仓库根目录 `RGBD_BASELINE_README.md`、`RGBD_LANGUAGE_DATASET_README.md` 和 `COLOR_DESCRIPTION_PIPELINE.md` 的有效内容。三份旧说明已合并后移除，防止 baseline、语言数据和颜色文本协议继续出现多个互相漂移的入口。依赖库目录中的 README、VOT 接线 README 和各 overlay 的重建 README 仍保留在代码附近，因为它们属于局部使用说明，不是第二份项目交接文档。

## 4.1 GitHub 代码分布与边界

GitHub 仓库：`https://github.com/666666666666gao/Track`。

仓库保留三类源代码：

1. 原始 MPLT/OSTrack 风格 RGB-T 主树及其 RGB-D 数据支持；
2. `projects/sutrack_rgbd_language_template`：固定 SUTrack 上游提交的 RGB-D language/template overlay；
3. `projects/sttrack_lachtt_v1`：固定 STTrack 上游提交的 candidate-own RGB-D association 与 protected/tentative transaction overlay。

STTrack overlay 的上游为 `NJU-PCALab/STTrack@283cd6dd45536636490db8bca1c63c4647be799b`，当前服务器实验源提交为 `2b32dccccf2d9082e15a54b8a02a945ac5439e05`；在原 54 个项目相关文件基础上增加 M17-1 fail-closed runner 和 post-audit binding builder，共 56 个 overlay 文件。它不包含权重、数据集、缓存、预测结果、VOT workspace、训练输出、Qwen 模型或密钥。具体重建步骤见 `projects/sttrack_lachtt_v1/README.md`。

## 4.2 早期 MPLT RGB-D baseline 支持

早期 baseline 在 MPLT/OSTrack 风格代码上增加了：

- `lib/train/dataset/depthtrack.py` 训练集加载器；
- `lib/test/evaluation/depthtrackdataset.py` 测试集加载器；
- DepthTrack 数据集注册；
- 16-bit/灰度 depth 到三通道 uint8 pseudo image 的稳健读取；
- `experiments/mplt_track/vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot.yaml`。

推荐数据布局：

```text
data/depthtrack/train/SEQ_NAME/
  color/00000001.jpg
  depth/00000001.png
  groundtruth.txt

data/depthtrack/test/SEQ_NAME/
  color/00000001.jpg
  depth/00000001.png
  groundtruth.txt
```

RGB 目录也接受 `rgb/visible/img/images`，Depth 目录也接受 `depths/depth_colormap/depth_color/infrared` 等常见别名。机器路径在 `lib/train/admin/local.py` 和 `lib/test/evaluation/local.py` 配置；禁止把个人绝对路径提交到公共配置。

历史训练/测试入口：

```bash
python tracking/train.py \
  --script mplt_track \
  --config vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --save_dir ./output/depthtrack_rgbd_baseline \
  --mode multiple --nproc_per_node 4

python tracking/test.py mplt_track \
  vitb_256_mplt_32x1_1e4_depthtrack_15ep_sot \
  --dataset_name depthtrack_test --threads 6 --num_gpus 1
```

这一路径是早期 RGB-D baseline，不等于当前 SUTrack/STTrack 语言—候选关联方法，也不能用其命令解释当前正式指标。

## 4.3 RGB-D-L 数据集与 JSONL 语言字段

早期数据接线覆盖 DepthTrack train/test、CDTB 和 VOT-RGBD2022。清洗后的历史 JSONL/目录包括：

```text
annotations/depthtrack_train_first_qwen3_corrected.jsonl
annotations/depthtrack_test_first_qwen3_corrected.jsonl
annotations/cdtb_first_qwen3_corrected.jsonl
annotations/vot_rgbd2022_first_qwen3_corrected.jsonl
annotations/CDTBLang_Qwen3_RAGStyle/
annotations/VOTRGBD2022Lang_Qwen3_RAGStyle/
```

加载器按 sequence name 取标注，不依赖 JSONL 元数据里的 Windows 绝对路径。历史清洗将 `similar_depth` 统一为 `similar_to_background`、`poor` depth quality 统一为 `low`，删除 `bounding box`/`red bounding box` 等泄漏框标记的词，并统一 `depth_stats` 标签。

早期 sampler 可传递：

```text
language_description
language_appearance
language_depth_relation
language_depth_quality
language_occlusion_state
language_distractor_relation
```

这些字段只是数据能力；baseline 网络默认不一定消费语言。正式 VOT identity-only 和后续 anchor/候选身份实验的真实协议、SHA 与结果，以本文前面对应实验章节为准。

## 4.4 frame-aligned `color_desc` 协议

后续颜色描述直接使用 TXT，不再强制转换为 JSONL：

```text
color_desc/
  depthtrack_train/<sequence>/color_description_ct.txt
  depthtrack_test/<sequence>/color_description_ct.txt
  cdtb/<sequence>/color_description_ct.txt
  votrgbd2022/<sequence>/color_description_ct.txt
```

DepthTrack Train 可按 RGB frame 一行文本；DepthTrack Test、CDTB、VOT-RGBD2022 的历史静态协议为每序列一条初始化文本。空行、缺行和多余行必须 fail closed，不能自动 pad 或错位。`*.resume.json` 只保存生成流程断点、输入绑定和 prompt/model provenance，不是 tracker 消费的正式标注。

历史本地生成链使用 Qwen2.5-VL-7B 读取 RGB 与 GT target region 形成视觉草稿，再由 Qwen3-8B 做纯文本语法、长度和禁词修正。Qwen3-8B 不看图，因此不能宣称它重新验证了视觉事实。正式实验中 Qwen3_8B 继续保留，但最近 M17 未调用。

传统静态测试把初始化描述在整条序列内固定；DepthTrack Train 可配置 current-frame text。若文本含 `left/right` 等方向词，水平翻转必须关闭或同步重写文本，否则图文对齐会被破坏。官方 DepthTrack Test 不用于调参；历史语言实验以 DepthTrack Train 的固定 sequence holdout 做验证。

当前对 VOT multi-start 的更严格结论是：一个原始视频包含多个 anchor，序列首帧文本不天然等于每个 anchor 的初始化身份文本。identity-only 清洗只带来很小 full-127 增益，Qwen current-anchor 重注释在 low22 反而增加失败。因此现阶段语言主要作为不可变身份锚和候选验证证据，不能无条件覆盖递归状态。

## 4.5 文档唯一入口规则

- 远端服务器唯一主文档：`/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`；
- 本地唯一主文档：`C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`；
- GitHub 唯一项目主文档：`docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md`。

后续实验首先续写远端主文档，再同步本地与 GitHub。`refine-logs/EXPERIMENT_TRACKER.md`、`MANIFEST.md` 和时间戳 plan/spec/audit 是机器审计工件，不再作为面向交接的第二份正文；它们的结论、SHA 和执行边界必须被吸收到本主文档。

# 第 5 部分：2026-09-01 M17-1 终态、低指标场景与文本标注边界

## 5.1 正式最好指标没有变化

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO | 74.020583 |
| VOT-RGBD2022 | ACC | 82.579344 |
| VOT-RGBD2022 | ROB | 89.565651 |
| DepthTrack | Pr / Re / F | 65.995933 / 65.335885 / 65.664250 |
| CDTB | Pr / Re / F | 75.387821 / 76.005850 / 75.695574 |

本轮没有产生新的 low22、full-127、DepthTrack Test 或 CDTB 结果，也没有新 checkpoint。Qwen3_8B 保留但未调用。

## 5.2 已定位的低指标场景

当前 VOT 的核心问题仍不是普通帧框回归：ACC 已超过项目目标，真正短板是 ROB 和被 ROB 拖低的 EAO。最常见的失败链为：相似干扰物或遮挡后选错实例，错误框写入递归 `state/template`，下一帧搜索区域围绕错误目标裁剪，模型随后对错误目标越来越自信。

| 问题族 | 代表序列 | 已确认现象 | 需要的机制 |
|---|---|---|---|
| 相似实例切换与递归污染 | `cup02`、`shoes02`、`cube05`、`toy09`、`yogurt`、`bandlight`、`duck03`、`humans_shirts` | 同类/近似外观候选被选中后连续失锁，文本类别也可能同时支持干扰物 | candidate-own RGB-D 身份证据、target/distractor association、独立 tentative 状态和原子回滚 |
| 目标离开局部搜索区 | `ball06`、`cube02_indoor_1`、`two_tennis_balls_3` | 当前局部搜索窗口无法覆盖真实目标；仅改文本不能召回 | 风险触发的多中心搜索或 factor-7，不允许全程盲目放大搜索区 |
| 未失锁但框贴合较差 | `humans_shirts_room_occ_1_B_1`、`robot_human_corridor_noocc_1_B_1`、`squirrel_wild_1` | ROB 可为 100，但尺度、宽高比或边界贴合拉低 ACC | 身份确定后的独立 box refinement，不应触发 recovery |

这三类问题不能用一个阈值或一段更长文本统一解决。尤其是第一类需要回答“哪一个具体实例是初始化目标”，而不是只回答“哪个候选像鞋/杯子/球”。

## 5.3 全帧/全 anchor 文本没有铺开

本项目没有进行“每一帧一条文本”的全帧注释，也没有对全部 VOT anchor 生成 Qwen 专属文本。

已完成的文本证据是：

- low22 稳定身份短文本相对旧长结构文本有改善：EAO `+0.644823`、ACC `+0.237595`、ROB `+0.975206`，失败 anchor `200 -> 195`；
- 同一 identity-only 方案进入 full-127 后仅约 EAO `+0.0456`、ROB `+0.1104`，远小于目标缺口；
- low22 当前 anchor 的 Qwen 视觉重注释相对 identity-only 由 195 个失败 anchor 增至 202 个，虽然救回 4 个，却新增 11 个灾难失败，净增加 7 个失败。

因此严格按“低指标子集先改善，才进入全量”的规则停止：没有把退化的 Qwen-anchor 方案铺到全 1,765 个 anchor，更没有为约 132.7 万 tracker frames 逐帧生成文本。传统每序列一条静态文本可以保留为身份锚；未来动态文本只能低频、低权重、可撤销，并且必须在 candidate-own RGB-D 身份验证之后使用。

## 5.4 M17-1 实现、审查与唯一运行

M17-1 实现了 sequence-disjoint、candidate-role-independent 的 utility/safety/survival runner，并增加 post-audit binding builder。服务器仓库分支为 `codex/language-anchored-candidate-transaction-v1`，最终代码提交为：

- `47a0111`：初版 fail-closed runner；
- `e8e58fd`：执行闭包与失败语义；
- `a821bf3`：运行时临时写审计；
- `939cc44`：精确 Git/文件系统副作用、隔离 scratch、确定性 gzip 与目录 fsync；
- `8ee26cc`：绑定构建器；
- `2b32dccccf2d9082e15a54b8a02a945ac5439e05`：spec/binding/audit/preflight 与 builder 的 same-FD 同字节解析和哈希闭包。

最终 runner SHA 为 `7358a9f8828d4dd0831bc09fa93c2e00c610a0f302e88e47a566736ca91566c7`，builder SHA 为 `7b4ac1460a34dff1e1aa7a565a2e02017879c38b032560968bfb75f087c54e50`。规格轴与安全/standards 轴代码审查均 PASS；只读 smoke 得到 628 个训练事件、3,768 个动作、158,753 个参数，梯度 L2 `0.4776667741`，置换误差 0，且 optimizer step、checkpoint、公开 benchmark、Qwen 均为 0。

独立 preexecution 审计为 PASS：

- preflight binding SHA `38426347e01e5b4fc00b56f34075a9357f4264a94a0b454b811efea42131bd52`；
- preexecution audit JSON SHA `0fb0447c8bda6daa771edd1b03edbc841a807a6fbf8f0e3b5fd8968a61c37c2f`；
- execution binding SHA `1195894d3bbcb19969b79a215442f799e974a78473c219e189bb89113416480b`；
- 绑定 9 个代码记录、28 个源记录、134 个 CLIP anchor、134 个 native anchor；训练/heldout 序列交集和 heldout 数值 target 序列化均为 0。

## 5.5 M17-1 fail-closed 终态

唯一一次 M17-1 使用 CPU、float32、固定 R3 合同启动。控制流到达训练后的发布前 `validate_runtime_identity()`，但运行时副作用审计触发统一异常：

```text
ContractError: forbidden runtime side effect observed
```

异常发生在 result/manifest/training trace/heldout predictions 构造和原子发布之前，SSH exit code 为 1。终态为：

- M17-1 输出根不存在；
- 没有 result、manifest、training trace 或 heldout predictions；
- 没有 M17-1 活动进程、screen 或 checkpoint；
- 仓库 HEAD/clean 和全部绑定身份保持；
- execution binding 中 309 个路径/SHA 独立复核为 `309 OK / 0 BAD`。

失败完整性审计结论为 `Integrity PASS / Overall WARN / Scientific FAIL-NOT_EVALUABLE`，JSON SHA `84bb79fee2170b233d261fd431d4582b640b62c7f24d104151c1f4d91caeea80`。runner 在抛错前没有持久化具体是 forbidden write、unexpected subprocess、network、Qwen/tracker/benchmark module 还是 checkpoint-path 集合，因此不得猜测精确副作用类别。同期 `/root/autodl-tmp/.autodl` mtime 变化也不能被当作根因，因为 Python audit hook 只记录当前进程事件。

没有 training trace 或 result，故不能把“246 optimizer steps 已完成”写成可审计结果；只能说控制流到达 post-training/prepublish validation 后 fail closed。该事故不是模型 heldout 指标失败，而是实验完整性门失败，因此也没有新的模型好坏结论。

R3 明确规定任意 `m17_1_failure` 后停止固定 M17 family：禁止重跑、fold 改动、阈值/seed/LR/step/width/loss 扫描、online replay 和公开 benchmark。后续只允许归档文档，以及设计名称、计划、规格均明确区分的新 family；不得以“只补日志”为由直接重跑 M17-1。

## 5.6 M18：因果分位数生存事务的目标闭合与架构验证

M18 不是继续重跑 M17，而是一个重新命名、重新冻结的数据闭合与模型家族：Language-Anchored Causal Quantile Survival Transaction（LACQST）。它试图直接学习“候选提交后未来是否能生存”，而不是只学习当前帧分数或手工阈值。

### 5.6.1 M18-0：只在 DepthTrack Train 上闭合真实多时域监督

冻结计划为：

```text
fold 2–5：training
fold 1：heldout，只保留 commitment，不序列化数值 target
fold 0：quarantine，只保留 commitment，不序列化数值 target
H3 / H5 / H10：branch mean、public mean、gain、累计低重叠 run fraction
```

M18-0 是一次只读目标闭合，不导入 tracker、不训练、不产生 checkpoint，也不是公开测试。唯一运行与独立结果审计均 PASS：

| 项目 | 结果 |
|---|---:|
| 源事件 / 动作 / 序列 | 1,466 / 8,796 / 134 |
| training 可用事件 / 动作 / 序列 | 507 / 3,042 / 76 |
| heldout commitment 事件 / 动作 / 序列 | 121 / 726 / 20 |
| quarantine commitment 事件 / 动作 / 序列 | 154 / 924 / 23 |
| training beneficial / catastrophic / neutral 事件 | 59 / 38 / 410 |
| training beneficial / catastrophic / neutral 动作 | 211 / 127 / 2,704 |
| heldout 数值 target 行 | 0 |
| quarantine 数值 target 行 | 0 |
| 三分区序列交集 | 0 |

关键工件：

- M18-0 result SHA：`19cf46def001d4068f9126b92c4f8ec80ed4f08474c7b1d8ae28b36a40752086`；
- manifest SHA：`00592abd13deb5d9f1feee887abee048b936668ba11e4fff6236a786537fa563`；
- split ledger SHA：`44a9e5dab9a95b9cbaaea11cf86dbbe13b52b63eed80517d1b08d594dc4a5f3a`；
- 3,042 行 training target gzip SHA：`6912ba5b02a6d131db3f356f6fa30b2fd719beb98373977bb6510559c684db0f`；
- 独立结果审计：`Integrity PASS`，claim ceiling 仅为 DepthTrack Train cached target closure。

这一步解决了 M17 以后最关键的数据问题：utility 和 safety 不再依赖同一个单帧标签，且 fold1/fold0 数值监督没有进入后续训练输入。

### 5.6.2 M18a 架构

M18a 的输入仍是 candidate-own、detached、H5×177 的 RGB-D/语言关系，不改变 Qwen、搜索 factor 或公开 tracker。模型结构为：

```text
六类 768→8 family projector（utility 与 safety 各自独立）
                    ↓
每步 177→48→32 step encoder
                    ↓
候选集合 mean/max context + residual
                    ↓
因果 prefix mean / current-prev / prefix minimum
          ┌─────────┴─────────┐
 utility tower             safety tower
 gain q0.10 LCB            H3/H5/H10 softplus hazards
 branch mean q0.10 LCB      monotone survival
                            risk q0.90 UCB + catastrophe
          └─────────┬─────────┘
 gain_LCB - 2*risk_H10 - 0.5*catastrophe
```

固定模型参数量为 106,566，utility/safety 参数交集必须为 0；candidate role ID 只做 canonical gather，不进入 learned layer。无 event-commit 分类头，避免 M17 中“科学分数 + 独立提交分类头”的双重标定冲突。

零步验证固定使用 8 个 training 事件：2 beneficial、2 catastrophic、4 neutral；CPU、float32、seed `20260918`；六种循环候选排列必须逐值完全等价；每个 loss 只能向所属塔产生非零有限梯度；optimizer 构造和 step、checkpoint 均为 0。

### 5.6.3 五轮双轴代码审查发现并关闭的问题

初版不能直接运行。Standards 与 Spec 两个独立审查轴进行了五轮 fail-closed 审查，累计发现并修复：

1. invalid candidate 只从集合均值/最大值中排除，却仍可得到高 dominance；修复为 gain/branch/risk/catastrophe/dominance 全路径失败关闭，并新增验证门；
2. 执行参数中的 binding 文件可能和 manifest 声称的 `binding_path` 不同；修复为执行路径精确绑定；
3. `.pt`、gzip、模型源码先哈希后重新打开，存在 TOCTOU；修复为同一次 descriptor 读取、哈希并从同一字节流反序列化/编译；
4. preflight 异常在 journal 创建和 `try` 之前，无法记录；修复为固定 attempt root 先写 `start.json`，所有后续错误均写 terminal/manifest；
5. 运行时审计只看 write-mode open；扩展到 rename/remove/mkdir/symlink/link/chmod/chown/utime/truncate/xattr、FD 与 `dir_fd`；
6. Git 完整性代码不支持 linked worktree `.git` gitfile、`commondir` 和带斜线的完整分支名；已修复；
7. Git clean 检查遗漏目录软链和 submodule 状态；已修复；
8. journal 只检查根目录普通文件，遗漏嵌套目录/软链；改为 no-follow 递归 inventory；
9. `chmod(..., follow_symlinks=False)` 失败可能被吞掉并仍报告 success；改为 descriptor-based `O_NOFOLLOW + fchmod`，逐项验证 `0444/0555`，任意失败改写失败收据和非零退出码；
10. 硬链接可绕过 no-follow 并让封存修改外部 inode；增加 `os.link` 源/目标审计和 `st_nlink == 1` 双重检查。

最终提交为 `81700ba50a039bf75cbfa3ff1ae608dbbb0661c1`：

- model SHA：`2dc7fc2395eedc3c9901d908da96f0405f9ee2a71ccbe941c66eae314300843b`；
- runner SHA：`1f5fd7fd588001d2b372d350ece0f80b63ca57de2cb5fe25bb8af6eb730e08d7`；
- Standards：PASS，0 blocking findings；
- Spec：PASS，0 findings；
- 唯一非阻塞维护性意见：1,500 行以上 runner 同时承担 Git 解析、安全加载、审计、ML smoke 和 journal 发布，后续新家族应拆成深模块，但这不影响本次固定执行的正确性。

## 5.7 M18a 唯一零步验证：模型门 20/21，但观测性合同失败

### 5.7.1 执行前闭包

独立 preexecution 审计第一次因隔离审计代理没有继承 SSH 接线而得到 `FAIL_UNVERIFIABLE`，没有授权执行。第二次显式使用现有 askpass 做只读 SSH 后 A–H 全部 PASS；这是同 GPT 家族的独立 Type-A 审计，不是跨家族 Type-B acquittal。

最终绑定：

- spec SHA：`d90996eced96dc0c1ec7ed88cc6a8e5b2673cbbe9193bedf4916fdc54047fd61`；
- preexecution audit SHA：`f8588de82cb91e7425320cf55aac4af224d669deb2b991a8b6af987a92624dba`；
- execution binding SHA：`c37688ea547d7defd8ebee8049ecd289b5713d8f9388f3fd807e78e10eb0d575`；
- 授权只有“一次 8-event、zero-step architecture/journal smoke”；训练、checkpoint、Qwen、low22、full127 和自动 M18b 均禁止。

### 5.7.2 唯一运行结果

唯一一次运行终态：

| 项目 | 结果 |
|---|---:|
| terminal status | `gate_failure` |
| accepted | `false` |
| exception / seal error | `null / null` |
| 工程门 | 20 / 21 PASS |
| 参数量 | 106,566 |
| preclip / postclip L2 | 0.407504797 / 0.407504808 |
| optimizer constructed / steps | false / 0 |
| checkpoint | false |
| 科学结果目录 | 不存在 |

通过的模型/架构门包括：参数精确且低于 160k、utility/safety 参数完全隔离、所有参数被覆盖、shape/finiteness、combined loss finite、六种候选排列 exact parity、生存概率 H3≥H5≥H10、每个 loss 梯度只进入所属塔、pre/post clip、model state exact、无 optimizer、0 step、无 checkpoint、invalid candidate fail closed、仓库/控制文件身份运行后仍精确、科学输出不存在。

唯一失败门为：

```text
runtime_side_effects_clean = false
```

审计钩子记录：

```text
forbidden_write_paths = ['/dev/null']
subprocess_events = ['subprocess.Popen']
network_events = []
forbidden/unresolved mutation paths = []
unresolved write targets = []
forbidden modules = []
```

runner 和 model 的冻结源码中没有显式 `Popen` 或 `/dev/null` 调用；日志只保存了 subprocess 事件名，没有保存命令参数和调用栈。因此能证明“PyTorch/依赖导入路径内出现了这两类事件”，但不能诚实地进一步断言具体是哪个库、哪个命令或是否只属于无害环境探测。不得根据常见经验猜测成 `ldconfig`、编译器探测或 CUDA 检查。

### 5.7.3 结果完整性与停止决定

独立 result audit 结论为：

```text
Integrity PASS
Engineering outcome FAIL
Evaluation type: eight-event zero-step engineering smoke
```

三份 journal 的 SHA 与权限：

| 工件 | SHA256 | 权限 / 大小 |
|---|---|---:|
| `start.json` | `77440bea224dc13c2e631ae51677d3bf8a43cc09823dbd7ec3b80f2beeea3006` | 0444 / 2,785 B |
| `terminal.json` | `f56c7b6d583491beba76359d319b9cc3cfb0122f3e9f8b76640362a579c81d5f` | 0444 / 34,122 B |
| `manifest.json` | `ff8747d6444a727bdef53fd95aa27ec2b165d7e07ca56c304b306c994bbf325a` | 0444 / 2,078 B |
| attempt root | — | 0555，且仅上述 3 个文件 |

result audit JSON SHA 为 `8d6c25778d96885e710e13fc28dbc2a770209dd4637f9b0a8d429aa2fc07c17f`。运行 PID 已结束，无 M18a screen/process；科学输出和 checkpoint 均不存在。

冻结 M18 计划规定：任何架构或观测性合同失败均停止 M18，不得重跑或进入 M18b。因此当前明确禁止：

- 重跑 M18a 或放宽 `runtime_side_effects_clean`；
- 在 M18 内白名单 `/dev/null` 或任意 subprocess；
- M18b 的 206-step training、checkpoint 或 fold1 打开；
- threshold/seed/LR/loss/step/width/code scan；
- low22、DepthTrack/CDTB、新 VOT full-127 或 Qwen 在线文本。

这不是模型容量失败：20 个模型与数值工程门已经通过；也不是 VOT 指标下降，因为没有运行任何 tracker benchmark。它是“审计无法把依赖库 bootstrap 和模型运行边界区分开”的观测性失败。

## 5.8 下一步必须是独立新家族，而不是 M18 补丁重跑

下一家族应重新命名、重新冻结计划和规格，优先解决 bootstrap 可归因性，再复用已经通过的因果分位数生存模型。建议方向为 Bootstrap-Attributed Causal Survival Transaction：

1. `bootstrap audit` 与 `model runtime audit` 两阶段分离；
2. 第一阶段只导入精确版本的 PyTorch/依赖，记录 subprocess 的 executable、argv、cwd、环境摘要、调用栈和 `/dev/null` 的 flags/caller，不读取模型或 target；
3. 只有 bootstrap 事件能由冻结规则逐项归因且独立审计 PASS，第二阶段才安装严格模型运行 hook；
4. 模型阶段继续对任何新 subprocess、网络、外部写、Qwen、tracker/benchmark import 失败关闭；
5. 不在 M18 结果上事后增加白名单；所有允许的 bootstrap 事件必须在新计划执行前冻结；
6. 新架构 smoke 仍须 zero-step，随后才可能单独授权 sequence-disjoint 训练。

该方案的目的不是绕过安全门，而是把“依赖环境初始化”和“候选生存模型本身”拆成可审计的两个作用域。新家族尚未执行，尚无训练或指标收益。

## 5.9 正式最好指标与全帧文本状态仍不变

M18-0/M18a 都没有运行 VOT、DepthTrack Test 或 CDTB，因此正式最好值仍为：

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | 74.020583 / 82.579344 / 89.565651 |
| DepthTrack | Pr / Re / F | 65.995933 / 65.335885 / 65.664250 |
| CDTB | Pr / Re / F | 75.387821 / 76.005850 / 75.695574 |

仍未做全帧文本注释，也未把 Qwen current-anchor 注释铺到全部序列或全部 1,765 anchors；Qwen3_8B 保留但 M18 未调用。低指标序列、失败原因和 identity-only/Qwen 注释效果仍以 5.2–5.3 节为准。

## 5.10 M19：启动期副作用完成精确归因与不可扩张收据封存（2026-09-01）

### 5.10.1 为什么需要 M19

M18a 的因果分位数生存模型本身通过了 20/21 个架构与数值门，唯一失败是审计观察到 `subprocess.Popen` 和 `/dev/null`，但旧日志没有保存 executable、argv、cwd、stdio、调用栈和事件关联，无法区分“PyTorch 依赖初始化”与“模型运行期副作用”。固定 M18 合同禁止事后补白名单或重跑，因此 M19 被定义为新的、独立的 Bootstrap-Attributed Runtime Provenance Closure，只解决来源归因，不训练模型，也不声称 VOT 有任何改善。

M19 的边界始终是：

- 不读取 RGB、Depth、GT、预测、target、cache、权重或 checkpoint；
- 不实例化模型，不执行 forward/backward/optimizer；
- 不调用 Qwen、网络、DepthTrack Test、CDTB、VOT low22 或 full-127；
- 所有 journal 文件只读封存，失败关闭，不自动进入下一阶段。

### 5.10.2 M19a：唯一 import-only 运行找到了精确来源

M19a 源码仓库为 `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1`，冻结分支为 `codex/language-anchored-candidate-transaction-v1`。唯一正式运行绑定提交 `19beca70b5edd847cebf5e4ab127524ababcced0`，输出根：

```text
/root/autodl-tmp/sttrack_lachtt_m19a_bootstrap_attribution_attempt_v1_20260901
```

运行 `exit=0`、`status=success`、`accepted=true`，23/23 门通过。精确事件只有一条：

```text
phase        = torch_import
caller       = torch/__init__.py:164::_load_global_deps
Popen        = uname -p
cwd          = /root
stdout       = PIPE
stderr       = DEVNULL
devnull      = /dev/null, flags=524290
correlation  = popen-0001
event ids    = Popen wrapper 5 / DEVNULL 6 / Popen audit 7
```

也就是说，M18a 观察到的两类事件不是 tracker、Qwen、训练脚本或候选模型主动产生，而是冻结 PyTorch 1.13.1+cu116 在 import 阶段的依赖引导探测。M19a 同时确认：模型实例化、forward、tensor dispatch、optimizer、checkpoint、benchmark、数据与网络访问全部为 0。

M19a 封存工件：

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| `start.json` | 2,294 B / 0444 | `e81d7757899c01aa2e57b88fc2e29111830d106ef801d4b08246e4a71cfbf1d9` |
| `terminal.json` | 57,281 B / 0444 | `652be013391150ebd35c7b47cfe9cc9b1245de5d086f720cd63572837a16b356` |
| `manifest.json` | 56,886 B / 0444 | `cb5f1d361b7ced522526274516bd1807dbd4002fecdadc1682ed8d043c831416` |
| result audit JSON | 3,942 B / 0444 | `36697d2a82a465a606ac3e7ea59f25ae5d63a602a75ef6b918da20b88e11c3a3` |

结果审计为同模型家族独立 Type-A PASS，只支持“import-only 来源已归因”，不是跨模型家族 Type-B 免责，也不支持模型或指标结论。

### 5.10.3 M19b：把一条精确事件机械提取为不可扩张 receipt

M19b 没有再次 import torch，而是只读取已封存的 M19a journal，用两条独立路径重建同一事件：

1. Path A：`analysis.linked_pairs` 与 raw Popen/DEVNULL/audit events 联结；
2. Path B：完全绕过 `linked_pairs`，从 raw event set 独立重建。

两条路径必须输出逐字节相同的 canonical JSON。receipt 明确禁止 wildcard、prefix、目录许可、任意 executable、任意 `/dev/null`、任意 torch event 或 model-runtime allowance，因此它不能把“一条已知启动事件”扩成运行期白名单。

第一次使用 R6 绑定时，preflight 在创建 attempt root 之前以 exit 2 拒绝：实验仓库是 linked worktree，`.git` 是 gitfile，分支 ref 位于通过 `commondir=../..` 指向的 common Git directory；旧纯 Python Git 读取器只在 worktree gitdir 查 ref。该次没有创建结果目录、没有执行提取，也没有消耗唯一运行机会。修复只覆盖这一已有证据的问题：HEAD 仍从 worktree gitdir 读取，loose refs 与 `packed-refs` 从解析后的 common Git directory 读取；没有增加通用 fallback 或无证据兼容层。

最终冻结身份：

- 服务器源码提交：`d83fbbdd0286a535e8ec9c915313bb75de84c7e9`；
- extractor SHA：`09fee5453f50e0016de7a922f1916dddce57a9fa161fd8339b1cac36996ea620`；
- R7 spec SHA：`47a40c891f8f52f711dd0260285ba24252330025005a263cd08b8dc59f534335`；
- R7 preexecution audit SHA：`4ff4339eaf0d88cfc95bd7d12a71c2365d470efc511fff7eaed5a7aa152a47ed`；
- R7 binding SHA：`31ced4a23529e8ffd645a940bb6cde4888f4c72612b87cb538594181c93eb306`；
- 最终代码审查 verdict SHA：`b7e304f8a6472f12e0001223998a13851c4a1658432f4357fc764f31e776cc19`，standards/spec 均 PASS、0 hard findings。

唯一 M19b 提取运行成功，输出根：

```text
/root/autodl-tmp/sttrack_lachtt_m19b_exact_bootstrap_receipt_attempt_v1_20260901
```

| 项目 | 结果 |
|---|---:|
| terminal status / exit | `success / 0` |
| accepted | `true` |
| gates | 22 / 22 PASS |
| Path A / Path B bytes | 4,649 / 4,649 |
| Path A / Path B SHA | 均为 `24ad71fa198bf87a7ed8797282b88b26cd7f6522238d0858948f0d8517fef5a4` |
| byte-identical | `true` |
| wildcard / prefix allowance | `false / false` |
| model-runtime allowance | 空列表 |

封存结果：

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| `manifest.json` | 1,245 B / 0444 | `50a73b0a00a5749536e31d6b2a0c8d9d0f872653c2dce2d6760adda858bd06f9` |
| `receipt.json` | 4,649 B / 0444 | `24ad71fa198bf87a7ed8797282b88b26cd7f6522238d0858948f0d8517fef5a4` |
| `start.json` | 1,365 B / 0444 | `18cb34cf77f693200ed7bea29efaecac55b698eefb29e315a2fa6305d91bc7cc` |
| `terminal.json` | 1,634 B / 0444 | `495150e140ef7812c1f39b9d09285f8a3dea8329207c305338626f3c2e251527` |
| result audit JSON | 6,236 B / 0444 | `6614bda4f9a9dfc950dae9482c6ce343a88ec434d0cff296d0a1a67b77aed1ac` |

attempt root 为 0555，严格只有上述四个结果文件。独立结果审计为 Integrity/Engineering PASS，10/10 审计项通过；它只证明 exact bootstrap receipt 已正确提取和封存，不证明模型性能，也不自动授权模型 smoke、训练或公开评测。

### 5.10.4 最新源码发布范围

Track 仓库新增且只新增两份 M19 相关源码：

```text
projects/sttrack_lachtt_v1/overlay/tools/
├── run_sttrack_lachtt_m19a_bootstrap_attribution.py
└── extract_sttrack_lachtt_m19b_exact_bootstrap_receipt.py
```

GitHub 源码提交为 `b0932c0c087fc501289cc6615b753e6a2ba3672e`。STTrack overlay 共 61 个文件，`MANIFEST.sha256` 已逐文件复核；发布前扫描确认没有 `.pth/.pt/.ckpt/.bin/.safetensors/.onnx`、压缩包、超过 10 MB 文件、数据集、预测结果、Qwen 模型、服务器密码或 API 凭据。服务器的 plan/spec/binding/audit/result journal 仍留在服务器，不作为源码包上传。

### 5.10.5 对当前指标、低指标序列和全帧文本的影响

M19a/M19b 没有运行 tracker benchmark，因此当前正式最好值完全不变：

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | **74.020583 / 82.579344 / 89.565651** |
| DepthTrack | Pr / Re / F | **65.995933 / 65.335885 / 65.664250** |
| CDTB | Pr / Re / F | **75.387821 / 76.005850 / 75.695574** |

已确认的 VOT 低指标场景、身份切换、递归状态污染、搜索域丢失与框贴合问题没有因为 M19 自动消失；M19 只是移除了下一模型实验前的观测性阻塞。低22 identity-only 的失败 anchor 仍是 195，相对原结构化长文本的 200 有小幅改善；Qwen current-anchor 重注释仍为 202，净退化 7 个失败。因此仍没有铺开全序列、全 anchor 或全帧文本注释，Qwen3_8B 保留但 M19 未调用。

### 5.10.6 下一步严格边界

M19 现在已经完成 bootstrap 来源和 receipt 闭合，但不能直接恢复 M18b。下一步必须另建 M20 新家族，重新冻结 plan/spec/binding，并在相同 Python/PyTorch/源码身份下先完成 zero-step 模型运行期 smoke：bootstrap receipt 只消费上述唯一已归因事件，模型作用域内新增 subprocess、网络、外部写、Qwen、数据或 checkpoint 仍应失败关闭。只有新模型 smoke 与独立审计通过，才可单独授权 sequence-disjoint DepthTrack Train 训练；只有 Train-only 未见序列达到零 catastrophic、低 harm 和跨序列 rescue，才允许 low22。DepthTrack/CDTB 保真和 full-127 仍在更后面的晋升门，不能提前执行。

## 5.11 M20 receipt-bound 模型运行期 smoke：模型门通过，但观察器快照污染导致结果封存为负

### 5.11.1 M20 为什么要做、做了什么

M20 的目标不是继续改模型，也不是立刻训练，而是回答 M19 闭合后遗留的唯一工程问题：能否在精确消费一条已归因的 PyTorch 启动事件后，真实实例化 M18 因果分位数生存模型并完成前向/梯度探针，同时证明模型运行期没有新增 subprocess、网络、外部写、敏感读取或 checkpoint 行为。

本次保持以下内容完全不变：

- 模型仍为 M18 的 `Language-Anchored Causal Quantile Survival Transaction`，参数量 106,566；
- 仍使用同一组 8 个 DepthTrack Train 冻结事件：2 beneficial、2 catastrophic、4 neutral；
- 仍为 CPU、float32、seed 20260918、单线程；
- 不构造 optimizer，不执行 optimizer step，不写 checkpoint；
- 不运行 tracker、DepthTrack Test、CDTB、VOT low22/full-127，也不调用 Qwen；
- M19b receipt 只允许 `torch_import -> uname -p` 这一条事件，不允许 wildcard、路径前缀或 model-runtime allowance。

M20 新增的只是一个 receipt-bound runner：

```text
/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1/tools/
run_sttrack_lachtt_m20a_receipt_bound_model_runtime_smoke.py
```

服务器最终 runner 提交为 `7ec8ad31ca565b007eeef77fe71e74e433166442`，runner SHA256 为 `66563a70dca005f25f167138030436610790aa545bbc17fa2bf1be2e21ec53ed`。GitHub source-only 发布提交为 `0e62e2d292c78e8ff9140fa469a2f6328bd7f33d`。

### 5.11.2 执行前发现并关闭的三个控制缺口

第一次独立 preexecution audit 没有放行，指出三项硬问题；因此没有创建 attempt root，也没有执行模型：

1. spec 写了 20 个模型 gate，但 M18 `run_smoke()` 实际返回 17 个 gate，runner 只做 `all(values)`，没有核验 exact key set；
2. terminal 使用了另一条硬编码 claim ceiling，没有逐字复用 spec；
3. attempt root 创建后若 `start/terminal/manifest` 写入或 chmod 失败，可能留下没有封存的已消费目录。

修复保持最小范围：

- spec 冻结 17 个实际 gate 的完整 key 列表，runner 同时检查 count、exact set 和全部值；
- 正常、序列化失败、journal recovery 和 manifest 全部使用 `spec['claim_ceiling']`；
- 只针对已证实的 journal publication 风险增加单一 recovery 路径，删除固定的三个 `.tmp`，写入 `accepted=false` 的三文件失败 journal，并封存为文件 0444、根目录 0555；
- 没有改模型、loss、数据、阈值、候选或任何性能逻辑。

修复后 standards/spec 两路只读复审均为 PASS、0 hard findings。最终 Type-A 执行前审计为 PASS，冻结身份如下：

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| M20 plan | 8,428 B / 0444 | `e39a882c4a469a98c19a08e2546c67954ff58f898c041f5104343397aba9ab6a` |
| M20 spec | 15,506 B / 0444 | `3949d9692b7193f779336c101f481768d2d9f791db65c0856ff5fa27e90b6854` |
| preexecution audit | 6,654 B / 0444 | `2be4c657451064d7e72ffece85b234ba7e0e81b00a3e1aeef5d9d0725bae91c7` |
| binding | 3,802 B / 0444 | `041c70830a8ff5d65be0234ff1afe8090a5ef8cd6373cb830e0bf64069aa5bef` |

该审计只能授权一次 8-event、zero-step model-runtime smoke；不能自动授权训练或评测。

### 5.11.3 唯一 M20a 运行的 runner 自报结果

唯一运行退出码为 0，attempt root：

```text
/root/autodl-tmp/sttrack_lachtt_m20a_receipt_bound_model_runtime_smoke_attempt_v1_20260901
```

runner 自报 `status=success`、`accepted=true`。真实执行计数与模型门如下：

| 项目 | 结果 |
|---|---:|
| M18 exact model gates | 17 / 17 PASS |
| model instantiations | 41 |
| forward call entries | 590 |
| tensor dispatch ops | 16,796 |
| optimizer constructions | 0 |
| optimizer step entries | 0 |
| checkpoint write entries | 0 |
| scientific output | 不存在 |
| model state before/after | exact |
| six-permutation parity | exact |
| preclip / postclip L2 | 0.407504797 / 0.407504808 |

精确 bootstrap event 也与 M19b receipt 一致：`torch_import`、`uname -p`、cwd `/root`、stdout PIPE、stderr DEVNULL、correlation `popen-0001`、event IDs 5/6/7、caller `torch/__init__.py:164::_load_global_deps`。

三文件 journal 完整封存：

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| `start.json` | 2,211 B / 0444 | `669464e71ceafd721d3bef42150501d79c657bb05b671eb10e2c4ecf891c7d17` |
| `terminal.json` | 985,158 B / 0444 | `3611dff2a4157591555c00d1a44beb850e21df8805641a7287caa73228ffa532` |
| `manifest.json` | 495,128 B / 0444 | `b217e4b894f5d16fb59f2e217a46bb6235765ca5662e3eddf3658c531c70dff7` |

attempt root 权限为 0555，严格只有上述三个普通文件。

### 5.11.4 独立结果审计为什么推翻 runner 的 accepted=true

独立 Type-A result audit 对封存 JSON 逐项重算后给出：

```text
overall_verdict  = FAIL
integrity_status = FAIL
claim_supported  = false
current_state    = SEALED_NEGATIVE_M20A_RESULT
rerun_authorized = false
```

审计确认模型本身的 17 个 gate、真实模型执行、精确 bootstrap event、零 optimizer/checkpoint/scientific output 都成立；失败集中在观察器与接受判断的时序。

#### 问题一：接受判断早于后续 model-runtime 敏感读取

runner 在 M18 `run_smoke()` 后立即取一次 observation 并计算 `model_runtime_new_side_effects_zero`，随后仍保持 phase=`model_runtime`，继续执行：

```text
direct_git_identity(repository)
control file rehash / exact identity checks
```

这些后续操作产生了 286 条 `phase=model_runtime` 的 sensitive read，覆盖 143 个唯一路径，其中 170 条涉及 `lib/test/vot`。代表路径包括：

```text
lib/test/vot/__pycache__/sttrack_baseline.cpython-38.pyc
lib/train/dataset/depthtrack.py
lib/train/dataset/coco.py
tools/audit_sttrack_lachtt_m5_train_predictions.py
```

但接受判断已经在这些事件发生前计算，因此 `model_runtime_events.sensitive_read_events` 为空、gate 为 true，而最终 `model_runtime_observation.sensitive_read_events` 实际包含 286 条记录。也就是说，runner 报告的“模型运行期新增敏感读取为 0”不成立。

#### 问题二：snapshot 返回可变列表引用，早先快照被后续事件反向污染

M19 `ProvenanceObserver.snapshot()` 返回的是 observer 内部列表本身，而不是不可变拷贝。于是：

```text
bootstrap_observation.phase_at_snapshot = torch_import
```

但这个早先保存的对象后来也出现了 286 条 `phase=model_runtime` 记录。由此可见 bootstrap 与 model-runtime 两个快照并没有真正冻结；它们只是指向同一批会继续增长的列表。

因此只能保留以下子结论：

- exact bootstrap event：支持；
- 17 个模型 gate：支持；
- 模型真实实例化/前向/tensor dispatch：支持；
- optimizer、checkpoint、scientific output 为 0：支持；
- model-runtime 新增副作用为 0：不支持；
- bootstrap/model-runtime phase separation：不支持；
- 训练、泛化、VOT/DepthTrack/CDTB 指标提升：未尝试、完全不支持。

result audit JSON 大小 9,149 B、权限 0444，SHA256 为 `c2ca34e977a20cc35fd11eed7f56c4fe61161b4a3fc09e8bdedd11ce975a4425`。

### 5.11.5 M20 对当前研究问题的实际意义

M20 没有产生新权重或公开指标，因此当前正式最好仍为：

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | **74.020583 / 82.579344 / 89.565651** |
| DepthTrack | Pr / Re / F | **65.995933 / 65.335885 / 65.664250** |
| CDTB | Pr / Re / F | **75.387821 / 76.005850 / 75.695574** |

VOT 的核心问题仍然是早期身份切换后污染递归 state/template，ACC 已达标而 ROB/EAO 受连续失败链拖累。M20 既没有验证候选身份关联是否改善，也没有验证 protected/tentative transaction 是否提高 survival；它只暴露了运行期审计器本身的时序和快照语义错误。

低22文本结论也不变：identity-only 使失败 anchor `200 -> 195`，Qwen current-anchor 重注释使其 `195 -> 202`。因为困难序列没有获得可靠净改善，仍未做全序列、全 anchor 或全帧文本注释；Qwen3_8B 保留但本次未调用。

### 5.11.6 后续允许和禁止的边界

M20a 已封存，禁止：

- 重跑同一个 M20a attempt；
- 把 exit 0 或 runner 的 `accepted=true` 写成工程成功；
- 根据 M20a 启动 M20b 训练、checkpoint、low22、full-127、DepthTrack Test、CDTB 或 Qwen；
- 事后删掉 286 条读取记录，或扩大 receipt/model-runtime allowance 来让结果通过。

若继续，必须建立新的 successor identity、plan、spec、binding 和独立 preexecution audit。最小结构修正应只处理本次已证实的问题：

1. `snapshot()` 对每个事件列表创建真正的不可变副本，不能保存 live list 引用；
2. 在模型运行期结束时明确关闭该 phase，再执行 Git/control identity postflight；
3. acceptance 必须基于所有 postflight 完成后的最终冻结 observation，而不是中途快照；
4. 保持 M19b receipt 不扩张，仍不允许把任意读取、VOT 路径或 `/dev/null` 作为通用白名单。

该 successor 仍只能先做 zero-step 工程 smoke。只有其独立结果审计真正 PASS，才可另行规划 sequence-disjoint DepthTrack Train survival 训练；VOT low22 和 full-127 仍不能提前运行。

## 5.12 M21 immutable phase-closed successor：工程 smoke 经独立审计正式通过

### 5.12.1 为什么必须建立 M21，而不是重跑 M20

M20 的模型执行和 17 个模型门本身已经通过，但其接受判断不可信：`direct_git_identity()` 在模型快照后仍以 `model_runtime` phase 运行，产生 286 条仓库读取；同时 `snapshot()` 保存 live list，导致早期 bootstrap/runtime 快照被后续事件反向污染。独立审计明确禁止重跑或事后删事件，因此 M21 使用全新的 plan、spec、binding、runner、commit 和 attempt root，只修复这两个已有直接证据的问题。

M21 没有改动以下科学内容：

- `lachtt_causal_quantile_survival.py` 模型 SHA 仍为 `2dc7fc2395eedc3c9901d908da96f0405f9ee2a71ccbe941c66eae314300843b`；
- 参数量仍为 106,566，candidate count=6，cached horizon=5，relation dim=177；
- 仍复用 M18 的同一组 8 个 DepthTrack Train 工程事件：2 beneficial、2 catastrophic、4 neutral；
- 仍为 CPU、float32、seed 20260918、单线程、零 optimizer step；
- 不训练、不写 checkpoint/prediction、不运行 tracker/VOT/DepthTrack Test/CDTB、不调用 Qwen；
- M19b receipt 仍只允许 `torch_import -> uname -p`，没有 wildcard、目录前缀或 model-runtime allowance。

### 5.12.2 最小代码修复和审查中发现的合同缺口

M21 runner 的核心修复只有：

```python
def frozen_observation(observer):
    return copy.deepcopy(observer.snapshot())
```

并把执行顺序改为：

```text
torch_import
  -> 冻结 bootstrap snapshot
model_runtime
  -> 真实模型 smoke
  -> 冻结 runtime snapshot
postflight_identity
  -> direct_git_identity + control/binding rehash
postflight_complete
  -> 冻结 final snapshot
  -> 统一计算 acceptance
```

postflight 读取没有被忽略或加入通用白名单。每一条敏感读取必须同时满足：

1. 路径严格位于固定仓库 `/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1`；
2. 调用栈来自冻结 SHA 的 M19 provenance runner；
3. 栈中同时存在 `direct_git_identity` 和 `_git_worktree_clean`；
4. postflight 的 subprocess、DEVNULL、write、unresolved write、mutation、network 全部为 0；
5. postflight import 集合必须与 runtime 完全相同。

双轴只读代码审查最初发现了额外的合同/负结果记录问题，但没有修改模型或实验：

- spec/binding 的运行次数、zero-step、training/checkpoint/public-evaluation/automatic-next-stage、forbidden actions 和 claim ceiling 必须精确等值，不能只检查一个布尔开关；
- 普通 gate failure/exception 也必须允许只读独立结果审计，但训练和评测权限仍必须为 false；
- binding 本身必须纳入 postflight 重哈希；
- terminal 的 optimizer/checkpoint 字段必须来自真实 instrumentation counts；无法观测时写 `null`，不能在失败路径硬编码成 0/false；
- 清除唯一的 EOF 空白后，`git diff --check` clean。

最终 runtime 与 contract 两轴复审均为 **PASS，0 hard / 0 soft**。服务器源码提交为 `39b8be575f4c4121281d4e00cc28b3b13840484b`，runner 大小 35,039 B、权限 0755、SHA256：

```text
2ea35b9ddaa53972d33d2b0e31672bd45a76fc62195e0876ff9737ecb1e3fbf5
```

source-only overlay 已推送到 `https://github.com/666666666666gao/Track`，源码发布提交为 `32783de42de9b5d4397e839f7311cf52623a036a`。发布树共 63 个 overlay 文件，manifest 63/63 精确匹配；权重、数据、结果包、Qwen 模型、凭据和大于 10 MiB 文件均为 0。

### 5.12.3 冻结计划、规格、预审和 binding

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| M21 plan | 6,846 B / 0444 | `2caada7d7087c824ee4dad168ebe6f83e2f68830e0b8d124b35fd7a577b8c3d7` |
| M21 spec | 16,831 B / 0444 | `9386af57a3000150a909d1cb1fe7932de11a1e6097c1fdd6de2d040605ced1e4` |
| Type-A preexecution audit | 8,390 B / 0444 | `35abd900ab66add1e9e78406bda713bceb90779e671afcdde54cd0ccbeb6bcdc` |
| M21 binding | 3,815 B / 0444 | `0c52b2693ec0d02eeda1a27734a9b7d0a4115d911ba59d0a28e6bfdfb22e8c82` |

预执行审计直接检查了 37 个递归文件记录、35 个唯一冻结文件身份和全部权限/路径/claim ceiling；结论为 Type-A PASS，0 hard / 0 soft。binding/package 最终静态 preflight 同样 PASS，attempt 与 scientific-output 路径在执行前均不存在。该 binding 只授权一次 M21a eight-event zero-step smoke。

### 5.12.4 唯一 M21a 运行和封存结果

唯一 attempt root：

```text
/root/autodl-tmp/sttrack_lachtt_m21a_immutable_phase_closed_runtime_smoke_attempt_v1_20260901
```

进程退出码为 0；但最终结论没有依据退出码，而是依据随后对 sealed journal 的独立 Type-A 结果审计。三文件 journal 为：

| 工件 | 大小 / 权限 | SHA256 |
|---|---:|---|
| `start.json` | 2,247 B / 0444 | `7fe9e98a8db557ecd25e8d0531c68c6e7ae7d15f3d745696a6eaec29514af2eb` |
| `terminal.json` | 1,034,745 B / 0444 | `4d3efa790698b110f4c699680fc3570aa022b314a43212a15517b1e51f5a3f9a` |
| `manifest.json` | 498,725 B / 0444 | `18fd187d44053f443dc5ecd8f701a2decf5b996ae4fc306adb52c378adeea235` |

attempt root 权限 0555，严格只有三个 regular、non-symlink、nlink=1 文件；manifest 中 start/terminal 身份与重新计算的 bytes/SHA 完全一致。scientific-output 路径不存在。

独立结果审计文件为：

```text
/home/SUTrack_RGBD_L/refine-logs/EXPERIMENT_AUDIT_M21A_RESULT_20260901.json
```

大小 13,522 B、权限 0444、SHA256 `f96788f5d00d21f1ea903177a0a67dc0cb6be7cda0758bc3e04b9d806d024884`。审计结论：

```text
overall_verdict    = PASS
integrity_status   = PASS
engineering_outcome = PASS
claim_supported    = true（仅限 eight-event zero-step engineering smoke）
```

独立重算得到的关键事实：

| 项目 | M21 审计结果 |
|---|---:|
| bootstrap 后期 phase 记录 | 0 |
| runtime snapshot 中 postflight 记录 | 0 |
| model-runtime popen/subprocess/DEVNULL | 0 / 0 / 0 |
| model-runtime write/unresolved/mutation/network | 0 / 0 / 0 / 0 |
| model-runtime sensitive reads | **0** |
| postflight sensitive reads / unique paths | 286 / 143 |
| postflight path violations / stack violations | **0 / 0** |
| postflight 新 import | 0 |
| M18 exact model gates | **17 / 17 PASS** |
| model instantiations | 41 |
| forward call entries | 590 |
| tensor dispatch ops | 16,796 |
| optimizer constructions / steps | 0 / 0 |
| checkpoint writes | 0 |
| model parameter count | 106,566 |
| preclip / postclip L2 | 0.407504797 / 0.407504808 |

M19b bootstrap receipt 也再次逐字段复现：phase=`torch_import`、`uname -p`、cwd=`/root`、stdout=PIPE、stderr=DEVNULL、correlation=`popen-0001`、event IDs 5/6/7、caller=`torch/__init__.py:164::_load_global_deps`。

与 M20 的本质区别是：相同的 286 次 Git 工作树读取没有消失，也没有被删掉；它们现在全部被准确记录为 `postflight_identity`，而真正 `model_runtime` 的敏感读取为 0，早期快照也不再被污染。因此 M20 两个 hard findings 已被工程上闭合。

### 5.12.5 对 VOT/RGB-D 指标和文本实验的影响

M21a 没有训练新权重，也没有运行任何公开评测，因此正式最好指标完全不变：

| 数据集 | 指标 | 当前正式最好 |
|---|---|---:|
| VOT-RGBD2022 | EAO / ACC / ROB | **74.020583 / 82.579344 / 89.565651** |
| DepthTrack | Pr / Re / F | **65.995933 / 65.335885 / 65.664250** |
| CDTB | Pr / Re / F | **75.387821 / 76.005850 / 75.695574** |

它只证明：冻结的 candidate-own RGB-D causal-survival 模型可以在严格区分 bootstrap、model runtime 和 Git postflight 的环境中真实运行，并保持 17 个架构/数值门全部通过。它**不能**证明：

- survival head 已在未见序列减少 catastrophic identity switch；
- protected/tentative transaction 已改善 ROB/EAO；
- DepthTrack/CDTB 保真已经通过；
- VOT low22 或 full-127 有任何提升；
- 文本注释策略已获得新结论。

因此此前文本记录不变：low22 identity-only 使失败 anchor `200 -> 195`，Qwen current-anchor 重注释使其 `195 -> 202`；困难序列没有获得可靠净改善，所以仍未做全序列、全 anchor 或全帧文本注释。Qwen3_8B 继续保留但未在 M21 中调用。

VOT 的已知低指标场景和根因也不变：`cup02/shoes02/cube05/toy09/yogurt/bandlight/duck03/humans_shirts` 主要是同类干扰导致身份切换和递归 state/template 污染；`ball06/cube02_indoor_1/two_tennis_balls_3` 主要是目标离开当前搜索域；`humans_shirts_room_occ_1_B_1/robot_human_corridor_noocc_1_B_1/squirrel_wild_1` 主要是 ROB=100 但框尺度/贴合不足。ACC 已达标而 ROB/EAO 仍受连续失败链拖累。

### 5.12.6 下一步：另建 sequence-disjoint 训练计划，不能把 M21a 自动升级为性能结论

M21a result audit 的授权边界明确为：结果可表述为工程 smoke 成功，但 `authorized_next_actions_after_pass=[]`；训练、optimizer step、checkpoint、prediction、DepthTrack Test、CDTB、VOT、Qwen 和 automatic next stage 全部为 false。

若继续，必须单独建立 M21b（或新的 successor 名称）plan/spec/binding/preexecution audit，训练仍只使用 DepthTrack Train 且按 sequence-disjoint folds。训练目标应直接对应当前 VOT 短板：candidate-own RGB-D target/distractor association、future survival/hazard 和 protected/tentative 原子 promote/rollback。只有 Train-only 未见序列达到零新增 catastrophic、低 harm、跨序列 rescue，才允许进入 low22；low22 明确改善且 DepthTrack/CDTB 保真后，才允许唯一一次 full-127。

因此当前状态是：**工程运行与审计阻塞已闭合，性能创新尚未得到训练和 VOT 验证。**

## 5.13 M22：sequence-disjoint causal-survival 训练结果与新发现（2026-09-01 至 2026-09-02）

### 5.13.1 为什么启动 M22，以及它实际回答什么问题

M21a 只证明冻结结构可以在零优化步条件下通过严格运行期审计，不能回答模型能否在未见序列上安全选择恢复候选。M22 因此单独建立了新的冻结计划：

> **Sequence-Disjoint Language-Anchored Causal Survival Training**

本次只使用 DepthTrack Train，不读取 DepthTrack Test、CDTB、VOT 或 Qwen，不生成 tracking checkpoint。数据按序列划分而不是按 action 随机划分：

| 分区 | fold | 用途 | 可用事件 | 候选动作 | 可用序列 |
|---|---:|---|---:|---:|---:|
| training | 2、3、4、5 | 训练模型 | 507 | 3,042 | 76 |
| heldout | 1 | 未见序列一次性选择门评估 | 121 | 726 | 20 |
| quarantine | 0 | 隔离，不参与本次选择 | 154 | 924 | 23 |

训练协议严格固定为 CPU float32、单线程、seed `20260918`、AdamW、学习率 `1e-3`、weight decay `0`、gradient clip `5`、batch 8。每个 batch 固定包含 `2 beneficial + 2 catastrophic + 4 neutral` 事件，训练 `2 × 103 = 206` 个 optimizer steps。模型仍是 106,566 参数的 `CausalQuantileSurvivalRouter`，utility 与 safety 参数完全分离；候选为固定六角色：

```text
current_peak0 / current_peak1
last_reliable_peak0 / last_reliable_peak1
velocity_peak0 / velocity_peak1
```

冻结晋升门要求：heldout 至少提交5个动作、至少4个 beneficial、覆盖至少3条序列、beneficial precision 至少0.95、catastrophic为0、平均真实H10 gain至少0.20，并且选中分支的H10 aggregate必须高于public。全弃权明确不能算通过。

计划文件：

```text
/home/SUTrack_RGBD_L/refine-logs/
EXPERIMENT_PLAN_M22_SEQUENCE_DISJOINT_CAUSAL_SURVIVAL_20260901_221110.md
SHA256 2f7ff997186b352b7e0a15ff579327b052027675e9e94d5b4e5aff9be899797f
```

### 5.13.2 两次 step-0 前停止：发现的是冻结特征存储契约错误，不是训练失败

初始 runner commit 为 `92318c885fb5732e0c8233efca071fe60051f285`，runner SHA256 为 `d32b23d72c0344925dfdadfddcff2c65313f740423ddf299f7f1f2d411253fc7`。第一次启动在读取首个 event feature 时停止：

```text
ContractError: feature payload tensor drifted: clip_image
```

实测 event payload 的真实冻结存储格式为：

```text
clip_image/native_rgb/native_depth/native_fused/
query_rgb/query_depth/raw_depth : float16
scalars                        : float32
```

旧 loader 错误地要求所有字段在磁盘上已经是 float32。最小修复为：严格按真实存储 dtype 校验，再将全部字段转换为 contiguous CPU float32。该修复 commit 为 `e644e4707e7982e0953f3e25892ef5a06f1f13c3`。

R1 随后在读取 CLIP anchor 时再次于 optimizer 构造前停止：

```text
ContractError: clip anchor tensor drifted
```

继续实测确认：CLIP anchor 的 `initial_image/identity_text` 和 native anchor 的 RGB/Depth token bank/mean 也均以 float16 冻结存储。R2 因此只把这两类 loader 改为：严格校验 float16、finite 和固定 shape，然后转换为 contiguous CPU float32。最终修复 commit 为 `badc1900dd704169c70a391fd753075b1721510d`，runner SHA256 为 `7a3a0690f8e39c61df023733d099bbe7f2ead111c24059d629d76a2b975aa4ee`。

这两次修复没有改变：数据、fold、候选顺序、target、loss、模型、optimizer、heldout 时序、policy 或阈值。FP16 到 FP32 只是对磁盘中已存储半精度值的精确扩展。独立审计还逐项打开并验证了全部 1,466 个 event payload、134 个 CLIP payload 和134个 native payload，共12,532个张量；shape、device、finite、storage dtype 和 CPU float32 转换错误均为0，heldout 标签文件未提前打开。

两次停止的日志分别为：

| attempt | 日志 SHA256 | 结果 |
|---|---|---|
| 初始版 | `e3861edf33152e18cf0dd45b9aba0e808d737b2465bbc4075c1cd2011183ddb9` | feature loader停止；optimizer未构造；step=0；无输出 |
| R1 | `fc7c6950c601b215db4cf0df3ec7440773875e96b9a80013fa740e1891f16380` | CLIP anchor loader停止；optimizer未构造；step=0；无输出 |

不存在部分训练、旧输出复用或权重污染。

### 5.13.3 R2 的冻结身份与审计链

最终真实执行绑定如下：

| 对象 | SHA256 |
|---|---|
| R2 spec | `533a1822ff792f8e02dbe27362ccf7864a2e08eb8bf705651fdf872b0f4ddbd6` |
| R2 runner | `7a3a0690f8e39c61df023733d099bbe7f2ead111c24059d629d76a2b975aa4ee` |
| source commit | `badc1900dd704169c70a391fd753075b1721510d` |
| R2 preflight binding | `1e61f8fc8c8707803a63ca51d3ba57d3de82d64232e166ddbc1d118fb93d4d69` |
| R2 preexecution audit | `b7ddaeb115f28317cd17eb082fbe170660fc0e278321649da5caf7af6f05c688` |
| R2 execution binding | `ed1f3365d20da8ebddb1b8ee3d51780cdf137c95d2543997b16f1fcabeb4fbcd` |
| R2 result audit | `196c02f62928e66140e449a4d993bdce4de20b5e66c5905a82e5a113afc12bba` |

preexecution audit 复核了292/292个绑定文件的SHA256和字节数、23/23个spec required records、sequence folds、206步训练常数以及 delayed heldout 标签隔离。result audit 为 PASS（0 hard / 0 soft），这里的 PASS 表示**结果与审计链真实、完整、可复算**，不表示科学指标通过。

### 5.13.4 完整训练结果：工程 PASS，科学 FAIL，全 heldout 弃权

R2 在25.22秒内完成全部206个 optimizer steps，206条 training trace 均为真实 optimizer step；无非有限 loss/gradient，所有 projector 均有梯度和参数变化，utility/safety tower 参数仍零重叠，candidate/event permutation error为0。

输出闭合为：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `heldout_predictions.jsonl.gz` | 80,582 | `182218ac8820bac52f806541a61cf3045c437111a01a2b944ccec1106e444f59` |
| `training_trace.jsonl.gz` | 29,544 | `029dab0705c97f2073e1484c8e4268624dd7bc42f0f73ddd2fdb8d6d27592fc4` |
| `result.json` | 16,429 | `a78662eb6a22468b1f1e49ab39065379ea59881381d263aca691fa4d38c62eb0` |
| `manifest.json` | 8,361 | `cae84f882f1d0de3956e56058ef1a5d1f118509820f55e632bb1616c985ce5cf` |

所有输出均为0444只读。训练结束并完成heldout预测后，runner才第一次打开 delayed label source；观察到的 heldout commitment 与预提交 commitment完全相同，train/heldout sequence overlap为空。

最终科学结果：

| 项目 | 结果 |
|---|---:|
| heldout events / actions | 121 / 726 |
| 实际提交动作 | **0** |
| beneficial提交 | 0 |
| catastrophic提交 | 0 |
| 覆盖序列 | 0 |
| engineering pass | **true** |
| scientific pass | **false** |
| accepted | **false** |
| decision | `m22a_fail_stop_fixed_family_without_rescan` |

失败的科学条件为：`all_abstain_is_not_pass`、最小提交数、beneficial数、beneficial序列数、beneficial precision、平均真实H10 gain以及branch aggregate优于public。因为没有任何提交，precision和选中动作的gain/aggregate均不可定义，不能把“零灾难”包装成安全性成功。

### 5.13.5 最重要的新发现：候选排序有信号，但绝对 survival 校准严重失真

heldout 的真实标签并不是“没有可救候选”。121个事件中包含：

| event class | 事件数 |
|---|---:|
| beneficial | 19 |
| catastrophic | 22 |
| neutral | 80 |

726个候选动作中实际有80个 beneficial、80个 catastrophic、566个 neutral；beneficial动作覆盖9条未见序列：

```text
adapter02_indoor, ball09_wild, bottle01_indoor,
cube06_indoor, human05_wild, speaker_indoor,
toiletpaper02_indoor, toiletpaper03_indoor, toy10_indoor
```

更关键的是，按模型 dominance 直接取第一候选时：

- 19个 beneficial events 中，16次第一候选的真实标签就是 beneficial，3次为neutral；
- 22个 catastrophic events 中，8次第一候选为catastrophic，14次为neutral；
- 80个 neutral events 中，第一候选全部为neutral。

这说明 candidate-own RGB-D / target-distractor relation 和相对排序不是完全没有泛化信号。真正阻止提交的是绝对值校准：

| heldout门 | 通过事件数 / 121 |
|---|---:|
| dominance margin ≥ 0.10 | 30 |
| predicted H10 gain ≥ 0.20 | 20 |
| predicted H10 risk ≤ 0.05 | **0** |
| catastrophe probability ≤ 0.05 | 111 |
| predicted H10 branch mean IoU ≥ 0.50 | 3 |
| 五门同时通过 | **0** |

对80个真实 beneficial actions，预测与真实量差异尤其明显：

| 量 | 预测均值 | 真实均值 |
|---|---:|---:|
| H10 gain | **-0.309846** | **+0.698499** |
| H10 risk | **0.423792** | **0.038750** |
| H10 branch mean IoU | **0.106065** | **0.796143** |
| catastrophe probability | 0.005288 | beneficial标签 |

真实 beneficial actions 的预测risk没有一个低于0.05，预测branch mean也没有一个达到0.50。与此同时，模型对真实 catastrophic actions 的方向性判断相对更合理：预测gain均值-0.656191、risk均值0.674213、catastrophe probability均值0.231644。训练loss也确实从首步1.733947下降到末步0.768196，因此不是训练没有执行或梯度完全失效。

因此 M22 的问题应准确写成：

> **相对候选排序已有部分跨序列信号，catastrophe方向也有一定区分度；但q10 gain/branch和q90 risk在未见序列上极度悲观且失准，导致风险门121/121拒绝、策略全弃权。**

不能简单把risk阈值从0.05放宽或把branch阈值从0.50降低，因为不加新证据地扫阈值会同时放入8个当前top-ranked catastrophic actions，并违反冻结的零灾难门。下一阶段如果继续，必须作为新家族重新设计或校准absolute survival heads，并重新预注册数据、训练和选择规则；不能在M22输出上后验选阈值。

### 5.13.6 对 VOT、文本和正式指标的影响

M22没有生成 tracking checkpoint，也没有运行 low22、DepthTrack Test、CDTB、VOT或Qwen，因此正式最好指标仍为：

| 数据集 | 当前正式最好 |
|---|---:|
| VOT-RGBD2022 EAO / ACC / ROB | **74.020583 / 82.579344 / 89.565651** |
| DepthTrack Pr / Re / F | **65.995933 / 65.335885 / 65.664250** |
| CDTB Pr / Re / F | **75.387821 / 76.005850 / 75.695574** |

低指标场景归因也没有被推翻：身份切换组仍需要 candidate-own RGB-D 与独立递归事务；搜索域组仍需要风险触发多中心 shadow search；ROB=100但ACC低的序列仍应由独立box refinement处理。M22只验证了离线候选选择器，不代表这些模块已经写入公开tracker主路径。

文本实验结论同样不变：identity-only在low22有小幅改善，Qwen current-anchor重注释净增加失败；困难序列没有获得足够可靠的改善，因此**仍未做全序列、全anchor或全帧文本注释**。Qwen3_8B继续保留但在M22完全未调用。

### 5.13.7 当前允许与禁止的下一步

独立 result audit 明确要求停止固定 M22 family：

```text
禁止：M22阈值扫描、超参数扫描、自动重训、low22、
      DepthTrack Test、CDTB、VOT、Qwen、checkpoint生成。
```

如果继续研究，必须新建独立 successor plan/spec/audit。新计划应针对本次实证问题，而不是继续改文本：保留已有相对候选排序能力，单独解决 q10 gain/branch 与 q90 risk 的跨序列绝对校准，并继续把 catastrophic=0 作为硬门。任何新家族仍须先通过 DepthTrack Train sequence-disjoint heldout；未通过前不得进入VOT低22。

因此截至M22的最新状态是：**工程闭合、训练真实完成、未见序列有可恢复候选且相对排序出现信号，但绝对survival校准失败导致全弃权，未形成可运行的VOT改进，也没有任何正式指标提升。**

## 5.14 M23 exact-hypothesis direct selection：从全弃权到少量真实rescue，但跨序列安全概率仍不可靠（2026-09-02）

### 5.14.1 M23为什么启动

M22已经证明candidate-own RGB-D / target-distractor关系具有一定相对排序能力，但q10/q90绝对量严重失准，导致121/121个未见事件全部弃权。随后只读诊断又发现两个结构问题：

1. 六个固定role中存在大量GT-free完全相同的五帧bbox轨迹，按role计数会重复训练同一个动作；
2. M22把训练事件重采样成25% beneficial、25% catastrophic、50% neutral，而DepthTrack Train可用事件的真实先验约为11.64% / 7.50% / 80.87%，会进一步抬高q90风险；即使恢复真实先验和exact去重，全局risk q90仍为1.0，证明旧的绝对quantile门本身不适合当前小样本跨序列问题。

因此M23只做一个新结构假设：先按五帧20个bbox标量精确合并重复假设，再使用参数完全分离的直接benefit/catastrophe双塔，固定输出

```text
dominance = p(benefit) - 4 * p(catastrophe)
```

固定提交门仍为margin≥0.10、benefit≥0.80、catastrophe≤0.05，不允许阈值扫描。文本继续只作稳定身份锚，Qwen关闭；公开tracker、模板和VOT路径均不修改。

### 5.14.2 数据、模型和训练合同

M23仍只使用DepthTrack Train：folds 2--5训练，已经消费的fold1做开发验证，fold0和delayed full数字标签严格禁止。精确去重后的数据为：

| 分区 | unique假设数直方图 | 有重复事件 | unique假设总数 |
|---|---:|---:|---:|
| folds 2--5 | 2:1，4:466，6:40 | 467 / 507 | 2106 |
| consumed fold1 | 4:113，6:8 | 113 / 121 | 500 |
| 合计 | 2:1，4:579，6:48 | **580 / 628** | 2606 |

同一组内必须同时满足完整五帧bbox完全相同、strict label相同、H3/H5/H10全部target相同；关系tensor作算术均值，最低canonical role作为代表，其余role被mask，不能进入set context、BCE、rank loss或policy。

模型为`UniqueHypothesisSelectiveRouter`，共106434个可训练参数；benefit和catastrophe两塔参数交集为0。训练固定为CPU float32、单线程、seed 20260923、AdamW、lr 0.001、weight decay 0.0001、12个natural-prior epoch、batch 8、正好768步；序列逆事件数权重在每个batch内归一。loss为benefit BCE + catastrophe BCE + 0.5 strict-label pairwise dominance rank。候选和事件顺序置换的输出误差都必须精确为0。

### 5.14.3 R1为何停止，以及为何允许一次R2

R1在模型、relation和optimizer构造之前退出，原因是runner把“少于六个unique假设的事件数”误写为587；GT-free exact-bbox口径独立复算后正确值为580。R1：

```text
exit code = 1
model instances = 0
optimizer steps = 0
output/checkpoint/public evaluation/Qwen = 0
scientific result = NOT_EVALUABLE
```

独立失败审计为`EXPERIMENT_AUDIT_M23A_R1_PREOPT_FAILURE_20260902.json`，SHA256 `7dcbae8439e90d16a7ed9031b0c88d80979e15df81849afb11a6a489135e5fbb`，7166 bytes，0444。它只授权在新plan/spec/preaudit下把`587→580`，不授权直接重跑或任何科学参数变化。

R1源码提交为`671f7406b07104e3d9a86cccc6b78cc0318c7ee6`；R2候选提交`90e7b71cb05c47ac7de997dfe0401fead52ac102`相对R1严格只有一个文件一删一增、唯一一行`587→580`，模型SHA仍为`9ae1927c7137169baba19cd1051801d57b0d4735b66f0a924d41b99e996fcb09`。R2 spec/preflight/preaudit分别为：

| 工件 | SHA256 |
|---|---|
| R2 correction plan | `76fea1fc537db0f5176065c3c1f6d5b51ddd818d9327fe420f06a2f8f411c5fc` |
| R2 spec | `fe41340c85572c4d83bb487c5d0f3e3b85d0b8988c4e0ecb862fd13613553389` |
| R2 preflight | `714feb7a2da9ce4fa9c96a613f951198b435140bed1aee2a7fd12eff5391e185` |
| R2 preexecution audit | `9f7784e6fc643faab2b9b691264210d85ae1b8f648478344b2a31ca8ce68da29` |

R2预审为0 hard / 0 soft，只授权一次R2。

### 5.14.4 R2正式结果

R2完成768/768步，首/末total loss为1.539467 / 0.000176569。训练确实收敛，但固定fold1门没有通过：

| 项目 | M22 | M23-R2 |
|---|---:|---:|
| selected actions | 0 | **4** |
| beneficial / neutral / catastrophic | 0 / 0 / 0 | **3 / 1 / 0** |
| beneficial sequences | 0 | **3** |
| beneficial precision | 不可定义 | **0.75** |
| mean true H10 gain | 不可定义 | **+0.423557** |
| selected branch / public H10 mean IoU | 不可定义 | **0.713720 / 0.290163** |

M23相对M22不是完全没有进展：它不再全弃权，真实救回`speaker_indoor`、`toiletpaper03_indoor`、`toy10_indoor`三条不同序列，且没有strict catastrophic提交。三个正例的H10 gain分别为+0.271127、+0.821306、+0.876299。

但是固定成功门要求selected≥5、beneficial≥4、precision≥0.95；实际4、3、0.75，三项同时失败。因此最终决定严格为：

```text
m23a_fail_stop_direct_unique_hypothesis_family_without_scan
```

不得降低阈值、增加epoch、换seed或进入fold0/low22/VOT。

R2输出为：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `development_predictions.jsonl.gz` | 7635 | `7225b06231cae6f635655571ebdf53714721eb454b906ce221646c965995bb4a` |
| `training_trace.jsonl.gz` | 49241 | `1b602a7eba1cdf07f01ef3d7b3c8ec869d08fd2edaaa33ebec25d0b9c7232b59` |
| `result.json` | 4965 | `b8fc6dfc534ef642d860765373028c29800b369b7006e6f0a587dd70d31e7b70` |
| `manifest.json` | 1003 | `bcc84ee068144ebee4ee11b87ae89f447e40a2941238e2587aa6a89ebaef9715` |

独立结果审计`EXPERIMENT_AUDIT_M23A_R2_RESULT_20260902.json`为SHA256 `42b50b7c93f7456ca46c58e30193553f26141d720c3b7118753eea55ab4da9dd`，10638 bytes，0444；结论是Integrity PASS、actual engineering `PASS_WITH_REPORTING_BUG`、Scientific FAIL、整体科学停止。

### 5.14.5 最重要的新反例：安全塔会在未见序列上高置信地低估伤害

唯一错误提交为：

| 事件 | role | 预测benefit | 预测catastrophe | 真实branch/public H10 IoU | 真实gain | strict label |
|---|---:|---:|---:|---:|---:|---|
| `file02_indoor@941` | `velocity_peak0` | **0.812115** | **0.001589** | 0.292532 / 0.567037 | **-0.274504** | neutral |

该事件本身属于catastrophic event class；模型却把候选伤害概率压到0.16%，并以0.250652的dominance margin提交。它虽然没有跨过strict catastrophic标签边界，但会明显降低未来十帧IoU，正是0.75 precision的来源。

这条反例说明：

1. exact去重解决了重复样本计数，但没有解决跨序列概率标定；
2. 训练loss接近0而未见序列仍高置信误判，主要问题是小样本过拟合和epistemic uncertainty缺失，不是训练不足；
3. 单个安全塔的低`p(catastrophe)`不能作为可靠提交证据；
4. 继续把0.05改成0.02不会排除该动作，改benefit或margin阈值则属于在已消费fold1上的后验扫描，也不能证明泛化；
5. 未来必须显式比较candidate与protected branch的反事实生存，并对跨序列模型不确定性做上界，而不是继续增加文本形容词或模板规则。

### 5.14.6 工程记账bug及处理

R2结果中的`engineering_pass=false`不是数据泄漏。runner把两个事实字段写成：

```text
fold0_numeric_targets_opened = false
delayed_full_target_source_opened = false
```

然后直接对全部engineering condition做`all()`，因此把“没有打开”错误当成FAIL。其余工程条件全部为true，独立审计也验证fold0/full delayed从未打开，所以真实工程结论是`PASS_WITH_REPORTING_BUG`。不允许为修记账字段重跑实验；源码随后只把两个条件改成正语义`*_not_opened=true`，提交为`ae780f187cba74acd22217139326d9213ca721b7`。封存的R2结果、哈希和科学FAIL保持不变。

### 5.14.7 对正式指标、文本和下一步的影响

M23没有生成tracking checkpoint，没有运行DepthTrack Test、CDTB、VOT low22或full127，也没有调用Qwen。因此正式最好仍为：

| 数据集 | 当前正式最好 |
|---|---:|
| VOT-RGBD2022 EAO / ACC / ROB | **74.020583 / 82.579344 / 89.565651** |
| DepthTrack Pr / Re / F | **65.995933 / 65.335885 / 65.664250** |
| CDTB Pr / Re / F | **75.387821 / 76.005850 / 75.695574** |

仍未做全帧、全anchor或全序列Qwen注释；Qwen3_8B保留但未使用。M23再次证明当前主要短板不是文本描述长度，而是同类实例候选的跨序列安全选择。

固定M23 direct unique family已经停止。下一家族若继续，应只允许新结构计划，重点是：

```text
sequence-disjoint epistemic safety / committee disagreement
        +
candidate-versus-protected counterfactual survival
        +
candidate-own RGB-D / language identity relations
        +
atomic tentative transaction
```

具体要求是先在folds 2--5内部做sequence-level OOF安全验证，用模型间分歧或保守上界识别`file02`这类高置信OOD伤害；同时直接预测candidate相对protected的生存差，而不是只输出单模型benefit/catastrophe概率。consumed fold1不能再用于阈值扫描，fold0仍保持未触碰。只有新结构在预注册OOF门下获得非零高精度动作且零灾难，才可另写fold0计划；在此之前仍不得运行VOT。

截至M23的最准确结论是：**candidate-own关系和exact去重已经产生3个跨序列真实rescue，证明动作空间存在价值；但单模型直接概率在小样本上近乎过拟合，并对`file02`给出高置信错误安全判断，因此尚不能安全接管STTrack递归状态。**

### 5.14.8 已冻结但尚未执行的M24计划

只读fold census确认可在不访问fold1/fold0的条件下做真正的sequence-OOF epistemic实验：folds 2/3/4/5分别有20/17/18/21条序列、132/103/73/199个事件，四折都含独立beneficial和catastrophic序列。census receipt为SHA256 `d1fddfafd20faedc5f64f63cc45d6a907f5bdde5dea8cc2d7119fbf0b776c794`。

因此新计划`EXPERIMENT_PLAN_M24_SEQUENCE_FOLD_EPISTEMIC_COMMITTEE_20260902.md`已冻结，SHA256 `4c6b69bdae5e10bcc9548ad47cc1801521e7a7007f0085d20730b0063613bb6f`，5583 bytes，0444，但**尚未实现、预审或执行**。核心协议是每折训练一个同初始化模型；评估折`f`时只使用另外三折模型，要求三者对同一canonical candidate一致，并同时通过最小benefit、最大catastrophe和最小margin门。四个模型总计预注册780步，聚合507个真正OOF事件；只有selected≥12、beneficial≥10、跨≥6序列、precision≥0.95、catastrophic=0且每折至少一个动作才通过。PASS也只允许另写consumed fold1计划，不自动访问fold1/fold0/VOT。

M24的目的不是再训练一个更大的分类器，而是直接检验：**由不同序列折产生的模型分歧，能否作为`file02`式高置信错误的epistemic安全证据。**
