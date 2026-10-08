> 发布说明：私有审核主件保持原字节；以下仅加本说明并规范行尾空白/EOF。JSON是私有主件逐字节复制。审核输入T对应当时的执行表，其旧字节保留于`refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER_20261009_031056_before_pilot_review_update.md`；当前执行表的后续状态不追溯改变审核。`.aris/`为私有输入证据，不是公开下载链接。源码PASS、整体WARN、运行未执行；R0封存/结果审核后才启动新GPU工作。

# M122 template pilot caller：新鲜只读源码审核主件

日期：2026-10-09T03:02:20.3885586+08:00

总体结论：**WARN（仅源码与范围限定）**。新事件缓存/调用方源码结论：**PASS**。**blocking_count = 0**；没有发现需要阻断的新源码错误，也没有静默修复任何实现。

运行状态：**NOT_EXECUTED**。**deployment_authorized = false；full_experiment_pass = false**。P:3、78–80 与 T:7–12 的 R0 完整评价封存/结果审核门槛仍未完成；本次不能授予新 GPU 工作、全量教师或 C 训练许可。源码 PASS 不等于 R1 运行 PASS。

审核属性：**read-only / fresh-context / same-family / provisional**。请求模型为 `gpt-6-astra`，请求推理等级为 `max`；实际后端模型/推理配置未独立核验。不能把请求路由当作后端证明，也不声称跨模型家族验收。按 experiment-bridge Phase 2.5 / experiment-audit 读取当前源码；旧 helper 审核仅作为背景，不直接继承结论。

本次主对象为 `template_write_event_cache.py`（C）和 `run_template_write_pilot.py`（R）。下文 ID:行号均映射到文末真实路径与字节身份表。源码、计划、执行表、旧报告未改；唯一写入为本 Markdown 与同名 JSON 私有主件。

## 缓存、状态恢复与模板动作

| 核查项 | 源码结论与证据 |
|---|---|
| 有证据的可变状态 | C:18–49 显式缓存/恢复 native bbox、frame_id、z_dict、z_patch_arr、track_query_before、selected_feature，另存首帧 initial、words、word_mask、empty 和 RNG。R:129 先以合法 t0/first_box 初始化，再于 R:131 恢复事件。N:63–89 的初始化提供 box_mask_z=None、原生必要属性；C:20、34–35 对 None、两槽、50/.75 的断言符合当前接口。不是只恢复一个 bbox。 |
| 不保存模型对象 | R:82 保存 C:18–28 返回的嵌套 dict、张量、NumPy 数组与 Python/RNG 状态；没有将 actor、WriteEvent 或模型实例交给 torch.save。解包后才重建 WriteEvent（C:49）。 |
| 初始两槽别名和永久槽 | N:77–79 的 `[template]*2` 是真实对象别名。C:10–15 与 F:15–21 都按对象 id 每份只克隆一次，并另建列表；缓存 CPU 化和恢复设备时保留此关系。F:88 用槽替换写入动态槽1，不原位覆盖槽0。W:107–109 的原生 append/pop 与之对应。真实 torch.save/load 往返未执行。 |
| 当前帧提交一致 | R:74 在 step 前保存独立模板列表和旧 patch 引用；W:108 重绑 patch、W:109 修改模板列表，不改旧 patch。R:75 step 后立即构造事件（R:79）；F:67–80 只还原旧模板/patch，frame、框、query、feature 均来自该次提交。W:95–104 的同一个 selected 索引决定框/feature/quality/同位置原生响应；F:73–78 核对 frame、框及真实写入资格。采集 actor 继续真实写入后的原轨迹。 |
| 分支所有权 | F:24–38 为 bbox、模板列表/张量、patch、query 列表/张量和 selected_feature 建立分支所有权。M:117–132 确实改写传入 query 列表元素，因此这一隔离是必要且已实现的。C:41–46 解包也分配新张量。initial 字典单独复制，值与文字在分支内只读共享；D:51–96、DR:12–27 未写回它们。 |
| W/K 唯一干预 | F:83–89 每支先复制并检查相同 submission，W 仅替换动态模板槽及伴随 patch。旧 patch 不参与后续 W:53–73 的预测，只是原生模板写入伴随记录；N:152 的可视化读取不在本调用路径。固定首帧参照与本帧候选均未重新选择。 |

未保存的 network、CLIP、decoder、cell_response、preprocessor、output_window、keep_rate 与参数是同输入重新构建/共享的冻结或固定量（W:23–40，N:20–38），不是本路径新增的时序历史。BaseTracker 的其他可视化属性不被本调用路径使用（BT:13–15，W:29–31）。selected_feature 被保存是正确的，它当前并非另一路 recurrent 输入；不能因此漏掉它或把 query 等同于该特征。

## 采集顺序、帧边界和 GT

R:71–91 严格依 spec 的原 sequence_order 和 frame 递增循环，只有 `record['template_write']` 才保存，达到12件即停止。真实资格来自 W:104–105 的选中位置原生响应 >.75 与 frame%50==0，F:73–77 再核对。collect 不调用 load_truth；没有用当前/未来 GT、收益符号或 C 留出分组挑选事件。前12件是待运行程序的目标，不是已采集数量。

R:53–56 将内部 frame=f 映射到 color/depth 的 `%08d` 文件 f+1。t0 使用 spec.first_box；GT 数组索引仍为 f。R:84、128 的未来序列是 `range(f+1,min(f+33,N))`，即同视频 t+1 至 min(t+32,N−1)，尾部自然缩短，不跨序列。F:107–119 检查最多32帧、原 frame 为50倍数、每个未来 step 无模板写入；32帧不触及下一个50帧写入周期。

实际本地 SPEC 有152个序列，RGB/depth帧数逐序列相等，sum(rgb_frames−1)=219802。SPEC:1186–1204 的 toy07_indoor_320 记录1406条GT和1367对图像。TOY 的本地文件确有1406行，首行 `298,164,36,45` 与 first_box 相同，SHA与spec一致；L:14–22 保留既定 `target[:1367]` 裁切。这里验证了缓存的标注文件与已有manifest/loader约定；未重新枚举远端图像文件，也没有把标注尾部当作可读取图像。

R:129 仅以合法首帧初始化，R:131 恢复已采集历史；F:114–122 之后每步只传该时刻图像，W:89–101 从本分支自身预测框生成下一 crop。虽然 R:133 预先读入未来图像列表，step 每次只接收一幅对应图像，后续图像不会进入更早帧的候选、query、crop 或模板。SP:14–73 先切片，再生成新的 padding/resize 数组；已读路径不修改共享的原图数组。

R:135–143 完成 K/K、W/K 并保存预测轨迹后，R:145 才调用数据集 load_truth。L:15 验证 groundtruth.txt 的既有SHA，L:16 从文件读取真实标注；没有从预测生成伪GT。R:104–111、146–155 计算当前 IoU 和有效未来帧的 mean(IoUW−IoUK)，没有按模型输出最大值/均值归一化。D:111–123 的候选 response 是已有推理逻辑，不是实验得分。

无效GT保持 null/未知；无有效未来GT时 delta 为 null。本帧无效不制造负标签。R:156–161 为每件事件保留回执，记录公共 C 训练资格，但不按资格或负/零收益删除事件。所有当前/未来GT有效性与每帧标签均在回执中，后续可以据此汇总排除数量；此 pilot 不创建公共训练集或优化 C。

## 冻结输入、共享模型与 RNG

R:22、30–44 固定 P1第三pass final、seed2027、precision_weight=1、456序列运行/659406次track；final/spec/bank/labels均与旧训练回执绑定，native checkpoint按spec、CLIP按bank绑定。训练bank和labels必须人审、dataset=depthtrack、类别槽有效，152序列集合一致。R:49–51 和 W:31–34 使 decoder/native/CLIP 保持eval且冻结；R:68、93、123、162 比较冻结模型和decoder state_dict摘要，没有优化器或反传。

本地 FINAL 的实际SHA为 `3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811`，与 R:22 和 TRAINED:11908 一致；SPEC实际SHA `3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425` 与 TRAINED:18 一致。TRAINED:16–17 的bank/labels身份与原 L:46–47 协议一致。SPEC:2920 的旧category-only描述不被caller用来造词或选择bank：R读取显式传入、经训练回执绑定的人审bank，W:81–83按序列索引使用tokens/mask。没有重新生成词或按前缀填词。本次未加载bank tensor、native/CLIP权重内容；这是一项源码绑定审核，不能替代未来实际输入加载回执。

共享模型正常顺序路径正确：F:114–123 顺序执行两支；E:13 的hook闭包是调用局部变量，E:28–31在正常encode后移除临时hooks。V:163 的 backbone.keep_rate 属性重绑确实存在，但每次 forward 都先取 W:58 传入的相同固定native.keep_rate，再在 V:216–221 使用，不积累分支历史。没有证据要求额外hooks框架或异常恢复逻辑。

进一步独立核对了 M:73–76 → BB:145–158 → V:161–238 的派发。PE:23–28先卷积产生局部输出，V:174–181的位置编码加法不写原始模板；UT:20–30和CL:50–71创建局部mask。M:95–104的原位改动作用于已cat出的局部token，不是输入query载荷。A:1538–1575、1661–1689、1906–1919的TSG和A:1280–1314、369–430、1964–1977的融合路径未显示其他对象级跨帧缓存。H:8–21有BatchNorm，但冻结网络保持eval；摘要对注册参数/缓冲区有实际检查。这个结论仅覆盖读到的项目Python路径，不证明外部框架和编译内核绝无副作用。

C:28保存Python、NumPy、Torch CPU及当前CUDA RNG。R:134在初始化、解包和读取未来图像后恢复事件RNG；F:112–124每支开始恢复同一入口状态，正常返回再恢复入口状态。因此 K/K 后的 W/K 也从相同事件RNG开始。F:108–109检查配对设备相同且等于当前CUDA设备，C:36–46把事件张量移到该设备。没有把W固定分配GPU0、K固定分配GPU1。

R:85的ordinal%2把12件分为两个各6件的worker；R:116–121只选择事件。**`--shard`本身不选择物理GPU**；两物理GPU分配由未来启动命令/可见设备设置落实。未提交或执行该启动层，不能声称已经核验两个worker分别使用GPU0/1。代码内同事件配对设备一致的约束成立；没有据此制造源码阻断项。

## K/K与原始证据能证明什么

R:135真正调用 F:128–138；该函数创建两个独立K分支、实际逐帧比较完整record、query各张量和selected_feature，并比较结束submission。W:110–114的record包括frame、previous_bbox/bbox、selected、分数、quality、同位置响应、observation、write、resize与crop_origin。R:156的K_K_per_frame_exact=True只有这些比较全部通过后才构造，不是拿同一对象同自己比较。

R:140的4×未来帧数对应K/K两次加W/K两次。R:142–143保存第一条K/K轨迹、W轨迹和K轨迹，每帧含record/query/selected_feature；加上C的事件输入、R:70–87的native_prefix与R:151–160的逐帧IoU/有效性，可以检查帧号、框轨迹、标签和窗口，并为重放保留输入。

F:138只返回第一条K/K轨迹，第二条比较后未落盘。因此原始文件本身不能重新计算两条已保存K/K轨迹的相等性；该断言的一致性依赖未来真实成功执行回执。当前实现也不单独证明缓存恢复后的W与无中断原轨迹逐位等同，不能用K/K重复性越界推断这一点。这是证据边界，不是已发现恢复字段缺漏；本轮不追加计划外测试或框架。

真实CUDA数值确定性、序列化往返、实际事件覆盖、K/K无分叉、W/K收益和真实速率均未执行。计划P:48要求源码审核与运行回执共同满足后才建立全量教师；本报告不能越过该门槛。

## experiment-audit A–F

| 项 | 状态 | 结论 |
|---|---|---|
| A GT来源/控制边界 | PASS（源码） | R:145–155与L:14–22使用数据集GT，GT在全部分支预测后读入。 |
| B 分数归一化 | PASS（源码） | R:108–111、155为原始IoU与有效帧均值差，无预测统计归一化。 |
| C 结果文件与数字 | WARN（未执行） | 未产生或认证任何新事件、K/K、教师/C结果；旧P1 result只用于输入身份。 |
| D 调用接线 | PASS（静态） | R:172–187接collect/replay；R:75–82、129–145实际引用pack/unpack/probe/WK/load_truth。函数存在不代表调用已执行。 |
| E 范围 | WARN（限定） | 仅前12件R1 pilot准备；不是完整教师、C训练、完整R1或新性能验收。 |
| F 评价类型 | planned real_gt；executed not_executed_source_only | 计划标签来自真实GT，本次没有已执行的新评价可授予real_gt结果身份。 |

C1完整因果教师、C2未来收益监督优势仍未证明。当前无需修改本次cache/caller。下一步仍只有源码准备/审核；R0封存和结果审核完成后，再按既有计划取得R1真实回执并审核，不能立即运行全量教师或C训练。

## 本地输入身份与读取范围

路径相对 `C:/Users/gb/.codex_track_publish_m29_20260902/`。下表32件输入在写报告前重新计算SHA/字节，均与本次记录一致。哈希仅标明所读/所核对输入，不新增运行时绑定方案；也没有重新认证旧32源码全集或52项gate。

| ID | 实际本地路径 | 字节 | SHA256 |
|---|---|---:|---|
| C | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_event_cache.py` | 2659 | `4582167683cc2b28f69821249c7e5309fd92b5247cba4e7aff9643047153bb91` |
| R | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/run_template_write_pilot.py` | 10850 | `d19aaedcfd18560992645098a778ffbeb232cf4d453be4df127069e235c0003d` |
| F | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/template_write_forks.py` | 6310 | `3f6143c7d27a74051cc2faf27ad7a9cc1cd1e4e9d42cf5f9d1a015d42dde7940` |
| W | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py` | 7693 | `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778` |
| L | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/train_m122_full_causal.py` | 11546 | `ffa40a08d167d9a7f185cf83ecf0851e9c5ed548c2b1280bb969d6b8c689f891` |
| S | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/analyze_train_states.py` | 4469 | `2439a8d0cdb7a7d6c35bbf68aaacd28850bfb8db2211f07eb9e9e7dac66b6def` |
| D | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_target_decoder.py` | 7739 | `055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2` |
| E | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_encoding.py` | 3120 | `71715a497147eb917e6a93ba6916ebbe01bc4a859cb275401289df2706627f33` |
| DR | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/dense_region_decoder.py` | 7272 | `66b9317529f741aba818aa9e9feba7fcebdb54c285f0bd9184f98f4b1f202e7b` |
| IP | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/instance_ab_prototype.py` | 7951 | `4485a7ca29a90fa13cc868c32e4e4083c351cfba7b318d341c0165c32a886e4f` |
| CG | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/region_patch_evidence.py` | 2237 | `85b7f245dc79dd5ed8c04b075854cbd8b0863bf5428a01c49b4e4951f44af7e1` |
| N | `projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/test/tracker/sttrack.py` | 10216 | `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f` |
| M | `projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/models/sttrack/sttrack.py` | 10684 | `d62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13` |
| P | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_PLAN.md` | 13682 | `cd9402aff88baa717e6a04875159818f6e1ab108a488610cf0ebbe7a15ea6f94` |
| T | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/refine-logs/m122_trusted_memory/EXPERIMENT_TRACKER.md` | 2217 | `1bb29ef55a49cf318ac9af32f9ae13b9acb617e91b24d75c13c194c82133aafd` |
| OLD | `projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/M122_TEMPLATE_FORK_HELPER_SOURCE_AUDIT_20261009.md` | 16047 | `a4a5a90e4f9ae1de66db6edab6b7fc123d9e9d6fadbd1f2effde85cf2108237a` |
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
| RGBD | `projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/train/dataset/depth_utils.py` | 9538 | `f97336af94a3ec196ad239603896fc0a2387dd1d61b48a4fbe2e055d9466e92e` |
| SPEC | `.aris/m122_full_causal_20261007/train_diagnostic_inputs_20261008/training_spec.json` | 99659 | `3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425` |
| COPYMAP | `.aris/m122_full_causal_20261007/train_diagnostic_inputs_20261008/files.json` | 23970 | `7457e4929e31eb6d103f7ed8899f0bdf0afee14f5c290b1788c31938fbb0a827` |
| TOY | `.aris/m122_full_causal_20261007/train_diagnostic_inputs_20261008/groundtruth/toy07_indoor_320.txt` | 20581 | `683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2` |
| TRAINED | `.aris/m122_full_causal_20261007/observations/hour_0224/train_precision1/result.json` | 326225 | `765cb793b932898de33fc6bbfc00866cc0b598bab1dfb6aa5537bcbf196b7ed6` |
| FINAL | `.aris/m122_full_causal_20261007/trained_finals/precision1_final.pt` | 1538308 | `3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811` |

读取范围：C/R/F/W/L/S/D/E/DR/CG/N/M/P/T/OLD/U/BT/V/BB/UT/PE/CL/SP正文已读，语义核查聚焦本次活跃路径。IP读1–45行；H读1–25、108–205行；A为活跃路线搜索与1–169、369–438、1092–1316、1437–1693、1863–1978行。RGBD正文已读，当前调用为7–68行。SPEC读开头、toy07和尾部字段，并解析152件顺序/计数；COPYMAP仅读前50行；TOY做原始行数/首尾行与SHA核对；TRAINED仅读1–80行及绑定字段/最终SHA搜索；FINAL仅计算原始字节SHA，未反序列化。JSON保留逐项read_scope。

未审计安装的CLIP/PIL/torchvision、PyTorch/timm/einops/OpenCV、外部selective_scan及编译CUDA内核，也未核验远端真实已加载二进制、bank张量或整套原始RGB/depth图像。不能把本地导出Python文件审核扩写成完整运行环境无副作用保证。

全程仅本地文件读取、行号检视、JSON元数据解析与字节哈希；没有导入Torch/NN，没有GPU计算或查询、SSH、网络、训练查询、AST重跑或测试。用户给出的既有AST3.8通过背景没有被冒充成本审核重跑结果。没有改源码、计划、执行表、旧报告或旧结果；只写这两个新的私有主件。

报告落盘后的第一次只读回读命令因PowerShell对象字面量写法错误产生null路径错误，未完成输入核对，也未修改输入。随后改用原始显式32路径列表，成功重算全部32件输入的SHA/字节，逐件与报告前记录相同。只承认后一次成功回读，不把先前命令打印的计数当作验证。
