> 发布说明：以下为审核员私有主件的正文，私有原件不改。仅新增此说明并规范行尾空白及末尾换行；JSON为原件逐字节复制。表中执行表T对应审核时旧字节，其不可变副本见`refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER_20261008_235856.md`；当前执行表随后只更新状态，不据此改写原审核结论。`.aris/`依赖为私有源码证据，不是公开下载地址。本报告仅为helper源码PASS，整体WARN、运行未执行，不允许完整部署。

# M122 template fork helper：新鲜源码审核主件

日期：2026-10-09（Asia/Shanghai）
审核：experiment-bridge Phase 2.5 / experiment-audit
总体结论：**WARN（源码范围限定）**；辅助模块源码结论：**PASS**；**blocking_count = 0**。
审核属性：**read-only / fresh-context / same-family / provisional**。请求模型 `gpt-6-astra`、推理等级 `max`；未独立核验后端实际模型或推理配置，不能据此声称跨模型独立验收。

本次没有发现需要阻断 `template_write_forks.py` 的实际源码缺陷。当前可见代码实现了独立分支历史、当前帧提交后的模板动作重建和逐帧 K/K 比较函数。**这不是 K/K 运行结果，不是完整 R1 或实验 PASS，也不允许据此完整部署。** runner、数据/GT 标签路径、实际 GPU 等价性和完整实验结果不在本次已完成范围内。未静默修复任何实现。

## 实际审核范围

主对象仅为新离线辅助模块 F。直接读取计划 P、执行表 T、三件旧源码调查 S1/S2/S3，并独立复核 W/N/M、decoder、CLIP hook 和当前导出 backbone/Mamba 活跃路径。旧笔记作为线索，不直接继承其结论。文末给出本次实际本地字节的 SHA256；这些值仅记录审核对象，不提出任何新运行时哈希绑定。

F:1–5 明确本模块没有 runner，不自行启动 GPU 工作；F:128–143 仅提供 K/K 与 W/K 辅助函数。本次没有采集/预检 runner、C 模型或优化器、标签缓存、真实 replay 回执或新结果文件可供验收。没有查询正在运行的训练/评价，也没有重新核验旧远端导出回执、原 52 项 gate 或原 32 源码全集；它们未被本审核扩大或修改。

## 模块源码核查

| 核查项 | 状态 | 直接源码证据及结论 |
|---|---|---|
| 可变历史所有权 | PASS | `template_write_forks.py:24`–38 对 tracker/native 分别浅复制后，另建 bbox 列表、模板列表和张量、patch 数组、query 列表和张量、selected_feature；`initial` 字典另建，值与文字张量只读共享。M:117–132 确实改写传入 query 列表元素，所以 F:35 的隔离有实际必要性。D:51–96、R:12–27 没有向共享首帧/文字输入写回。 |
| 本帧动作前状态重建 | PASS | `template_write_forks.py:67`–80 从刚完成当前 step 的 actor 复制，只有模板列表与辅助 patch 使用 step 前值。W:101–104 已提交当前 bbox/query/feature，F 没有退回前一帧，也没有重置 frame_id。actor 本身保留原写入轨迹。 |
| 候选、框、feature、query、frame 对齐 | PASS | W:95–103 以同一个 selected 索引选择框并核验分数/feature/quality，W:104 读取同位置原生响应，W:110–114 记录该索引、框和 frame。F:73–78 校验当前 record 与 actor 的 frame/bbox/资格并保留它；不重新选候选。query 与 selected_feature 从同次 step 后 actor 复制。 |
| W/K 唯一动作与永久槽 | PASS | `template_write_forks.py:83`–89 两支先复制相同 submission 并核对，只有 W 替换 `z_dict[1]` 与原生写入附带的 `z_patch_arr`。N:77–79 初始两槽引用同一 tensor；F:15–21 在单份快照内部保留这种别名，同时拥有新存储；F:88 用槽替换，没有原位改写首帧槽。W:107–109 的原生 append/pop 语义与之相符。 |
| 无 GT 重置或控制输入 | PASS | F:104–143 的未来 rollout 只接受 images，W:89–101 从分支自己的预测框形成下一 crop 并提交新预测。没有 GT 参数、initialize 调用或 GT 模板。SP:14–73 的已读 crop 实现按传入预测框裁剪；F 不计算标签。尚不能据此验收未实现的数据/GT runner。 |
| 共享模型和 CLIP hooks | PASS，限已读顺序路径 | F:26–27 检查 eval，F:114–123 顺序完成两条历史。E:28–31 每次正常 encode 完成后移除六个临时 hooks，闭包 `dense` 为调用局部对象；没有并发重叠调用。V:163 的 `backbone.keep_rate` 是真实属性重绑，但每次先由 W:58 的同一固定规则覆盖再于 V:216–221 读取，不形成分支特有历史。 |
| RNG 与窗口 | PASS，函数未执行 | F:93–101 保存/恢复 Python、NumPy、Torch CPU 及当前 CUDA RNG；F:108–109 要求两支设备相同且等于当前 CUDA 设备。F:112–124 每支开始恢复同一状态，正常结束恢复入口状态。事件 frame 是 50 的倍数、最多推进 32 帧，因而不经过下一个周期写入时点；每步还断言没有 template_write。没有证据要求多 GPU RNG 框架或额外异常恢复。 |
| K/K 逐帧控制比较 | PASS，函数未执行 | `template_write_forks.py:128`–138 在开始、结束比较 submission；每帧比较完整 record、query 张量列表和 selected_feature。record 包含 frame、previous_bbox/bbox、candidate selected、分数/quality/原生响应、observation、write、resize/crop_origin（W:110–114）。这能在调用时发现实际框/控制/query 分叉，不能把函数定义当作已观测一致。 |
| 最小实现 | PASS | F:1–143 没有新增 fallback、try/except、兼容层、运行时哈希、旗标或任务外重构。固定 2 槽、50 帧、0.75、32 帧与 eval 等断言对应当前接口/计划；初始别名处理对应 N:79 的真实对象关系。旧依赖中原有条件/异常分支不属于这次新增改动。 |

W/K 的辅助 patch 差异是原生模板写入的伴随状态，并非第二个独立干预：W:53–73、88–115 的后续预测不读取旧 `z_patch_arr`；原生 N:152 只在本调用路径未使用的可视化中读取它。首帧参照不随写入刷新，文字条件没有重选。辅助模块共享的是模型/固定参照，分支历史由各自对象推进。

## 共享依赖与真实未验证边界

直接核验的 backbone 派发为 M:73–76 → BB:145–158 → V:161–238。PE:23–28 先调用卷积产生局部特征，再 flatten/transpose/norm；V:174–181 的位置编码原位加法作用于这些特征。UT:20–30 的模板 mask 和 CL:50–71 的 token_mask 为局部新分配；CL:99–108 的 Up_Down 前向没有对象历史写入。因此已读项目 Python 路径没有证据表明模板原始载荷或共享 query 张量被其写坏。

TSG 的活跃扫描入口是 A:1539 的 `selective_scan_fn_v1`，调用于 A:1558–1569，经 A:1661–1689、1906–1919 接回 M；融合走 A:1280–1314、369–430、1964–1977，最终到 A:61 的 `selective_scan_cuda.fwd`。这些已读前向没有对象级跨帧缓存；A:129–132 写新建 `xs_fuse`，ctx 是单次 autograd 上下文。没有把未接入的其他 Mamba 类误列为本路径状态。H:8–21 的 BatchNorm 所处网络必须保持 eval，这由 F:26 与 W:31 的当前使用方式支持。

**仍未审计或执行**安装的 CLIP/PIL/torchvision 预处理、PyTorch/timm/einops/OpenCV 实现、外部 selective_scan 和编译 CUDA 内核，也未检查真实已加载二进制。不能将局部源码无新增缓存提升为完整无副作用保证或 CUDA 数值确定性证明。既有 S1:61–69、S2:52–54、S3:43–45 对这类边界和真实 K/K 必要性的限定仍适用。本次没有观测到错误、异常或污染，不据理论风险追加复制框架、hooks 清理框架、fallback 或防御旗标。

## 尚待 runner 与运行回执完成的范围

以下为已有计划的未完成工作，不计作本模块实际缺陷：

1. 调用顺序：在原 step 前持有**独立模板列表快照**和旧 patch；原 step 后、actor 再前进前，用该 step 的 record 构造 event。仅保留同一 `z_dict` 列表引用不能保存旧槽，因 W:109 会原地 append/pop。F:3–5、15–21、67–80 已提供所需接口；当前没有调用方可验收这一顺序。
2. 采集与标签：按 manifest 取前 12 合法事件；使用固定 P1 final、Train152/seed2027；传入同一视频的 t+1 至 t+n 图像并处理尾部；GT 只在分支外计算有效帧平均 IoU 差，保留所有原资格事件和排除统计。F 的 `len(images)<=32` 无法单独证明图像来源与 GT 读取边界（P:24–26、32–44、79–87）。
3. 真实预检：同 GPU K/K 逐帧无分叉、本帧 W/K 动作前一致及唯一写入差异需要实际回执；本次一个事件也未运行。源代码的断言、值复制和 SHA 都不能代替这些回执（P:44–48）。
4. 推进许可：P:3、78–80 要求旧完整评价封存/结果审核后才启动新 GPU 工作；P:48 要求源码和实际回执一起满足才能采集全量教师。本报告不改变该顺序，不授予完整部署许可。

## experiment-audit A–F

| 项 | 状态 | 本次可接受的陈述 |
|---|---|---|
| A：GT 来源/控制边界 | PASS（模块源码） | F:104–143、W:88–114 没有 GT 控制输入；没有审核未实现 runner 的数据集 GT 来源和有效性逻辑。 |
| B：分数归一化 | PASS（模块源码） | F:118–143 不计算评价指标或教师收益归一化；D:111–123 的 response 是现有推理逻辑，不能误判为实验指标造分。 |
| C：结果文件存在性 | WARN | 没有新 K/K/实验结果被提交或读取；P:3、87、95 与 T:3、8–12 仍是计划/待执行记录。没有认证任何新数字。 |
| D：实际调用 | WARN | 函数存在不等于完成运行；没有采集/预检 runner 接线供检查（F:1–5、128–143）。 |
| E：范围 | WARN | 仅辅助模块源码 PASS；前 12 事件与完整 R1 仍未完成（P:44–48）。 |
| F：评价类型 | WARN / 不适用 | `not_executed_source_only`；没有已执行评价可标为 real_gt 或 synthetic_proxy。 |

C1 的因果教师标签与 C2 的收益监督优势均保持未证明。无需修改本次 helper 源码；后续按已有计划实现并单独审核 caller/runner，再取得真实预检回执。此处没有新的完整实验验收。

## 输入身份、读取程度和操作记录

路径相对 `C:/Users/gb/.codex_track_publish_m29_20260902/`。上述 F/W 等符号加行号均映射到本表实际文件；SHA256 为本次本地计算，字节数为实际文件长度。23 件输入在写报告前再次核对，均与记录的 SHA256/字节数相同。该确认仅限这些输入，不假称重新验收原 32 文件全集或旧 52 项 gate。

| ID | 本地路径 | 字节 | SHA256 |
|---|---|---:|---|
| F | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_forks.py` | 6310 | `3f6143c7d27a74051cc2faf27ad7a9cc1cd1e4e9d42cf5f9d1a015d42dde7940` |
| W | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py` | 7693 | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| N | `projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py` | 10216 | `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f` |
| M | `projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/models/sttrack/sttrack.py` | 10684 | `d62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13` |
| P | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_PLAN.md` | 13682 | `cd9402aff88baa717e6a04875159818f6e1ab108a488610cf0ebbe7a15ea6f94` |
| T | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md` | 1819 | `55f34d8b147e706b60124340913d349bb4d7ddd54c43523796407b810c3d354a` |
| S1 | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_TEMPLATE_STATE_SNAPSHOT_RECHECK_20261009.md` | 18513 | `f98a3b80492a6d52e39db9ae8026e6642af89e604de993910efe93130a6e757b` |
| S2 | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_RUNTIME_DEPENDENCY_STATE_RECHECK_20261009.md` | 8931 | `1f0f18a6e512dfb8f3eefaffb28b54ca3c3f71916a9ead261ae3af33b82f44bc` |
| S3 | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_BACKBONE_DEPENDENCY_STATE_RECHECK_20261009.md` | 9174 | `a7d5639bd3413b03cb7df48e19186f5f0ec62554cf5b545f453c91447f2fb3c5` |
| D | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py` | 7739 | `055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2` |
| E | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_encoding.py` | 3120 | `71715a497147eb917e6a93ba6916ebbe01bc4a859cb275401289df2706627f33` |
| R | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_decoder.py` | 7272 | `66b9317529f741aba818aa9e9feba7fcebdb54c285f0bd9184f98f4b1f202e7b` |
| U | `.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009/lib/test/tracker/data_utils.py` | 2494 | `873337d0419ecfdd08b64123cdc22a4936ccdd6af98a092bc2106c32cf401fe5` |
| BT | `.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009/lib/test/tracker/basetracker.py` | 3155 | `f9bc6bbfb1a8f677f86e4f1eda024d89baaf454498a5f1321f1421df67a7aa36` |
| V | `.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009/lib/models/sttrack/vit_care.py` | 21902 | `05096a6cec0448e49f82264ad34647e267a6a50a6d96d3ee07ba10d61ce6ef0e` |
| A | `.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009/lib/models/layers/mamba.py` | 87654 | `a11e49551cea9c7a188b9b7783b541b315a1f684b86e1de364bb3ad6c04f00be` |
| BB | `.aris/m122_full_causal_20261007/runtime_backbone_dependency_sources_20261009/lib/models/sttrack/base_backbone.py` | 6183 | `b8f7c576072471b78a0cb025264b7d35d926b9044df45945625b0a8f35751b5a` |
| UT | `.aris/m122_full_causal_20261007/runtime_backbone_dependency_sources_20261009/lib/models/sttrack/utils.py` | 4438 | `557ce8e3d352e3ec69b2d21fcb0f8c18703b3152eff6f3f2e1a7470585e634ea` |
| PE | `.aris/m122_full_causal_20261007/runtime_backbone_dependency_sources_20261009/lib/models/layers/patch_embed.py` | 961 | `2259b7703f008748167dd65d12ac28a496235d1f20a1d5c84a3aa0da9c8c0499` |
| CL | `.aris/m122_full_causal_20261007/runtime_backbone_dependency_sources_20261009/lib/models/layers/cross_layer.py` | 3969 | `5008a012c53aaa0f5b7bb567b95b6118cef138644be9955e566cdf8c298d114e` |
| SP | `projects/sttrack_lachtt_v1/diagnostics/m55/code/control/lib/train/data/processing_utils.py` | 6758 | `7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6` |
| H | `projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/layers/head.py` | 10778 | `62cb2f363ea8bce491c9e6fc528076dd1e14c7d371ad69d6efd0b4062d063326` |
| CG | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/region_patch_evidence.py` | 2237 | `85b7f245dc79dd5ed8c04b075854cbd8b0863bf5428a01c49b4e4951f44af7e1` |

除 A/H 外，上表文件正文已读；A 读取导入、活跃路线搜索及 1–169、365–438、630–645、1092–1316、1437–1693、1863–1978 行，未声称整份 Mamba 文件或外部内核完整语义审计；H 读取 1–25、108–205 行。额外直接依赖 R/SP/CG 仅用于核对当前被调用的 regions/crop 方法，没有转移到其他训练或评价功能。技能/本地 Codex 策略文件仅提供审核流程指令。

全程是本地文本、行号和文件哈希检查；没有 Torch/NN 导入、GPU 计算/查询、SSH、网络、训练查询、测试或 Git 操作。一次文本读取用 Python 启动器报 `No pyvenv.cfg file`，脚本未运行；随后用 PowerShell/.NET 读取完成。没有改源码原字节、旧报告或计划/执行表。唯一写入为本 Markdown 与同名 JSON 私有主件。
