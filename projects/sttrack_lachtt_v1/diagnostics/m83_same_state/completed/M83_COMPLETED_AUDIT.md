# M83 全量终态实验完整性审阅

日期：2026-09-20。审阅者：`gpt-6-astra / max`，新上下文代理 `/root/m83_completed_integrity`。本轮按 `experiment-audit` 及本地 Codex 执行政策执行，是同模型家族审阅，语义接受状态为 **provisional**。

**总体结论：PASS。M83 核心终态与独立数值复算：PASS。** 全部 22 条原始读出、原 Category 轨迹、GT 对应、分组统计、救错伤对、Raw/Hann、更新资格、封存清单与归档检查通过。本轮发现的后续方法状态文字问题已在最终文件中修订。 未发现要求修改 M83 运行源码的真实缺陷。

```yaml
audit_skill: experiment-audit
audit_scope: completed_M83_local_artifacts_and_saved_output_GT_recomputation
overall_verdict: PASS
integrity_status: pass
m83_core_integrity: pass
deterministic_verification: PASS
review_independence: same-family
acceptance_status: provisional
reviewer_model: gpt-6-astra
reviewer_reasoning: max
remote_observation_performed: false
model_binary_opened: false
training_or_model_inference_performed: false
new_seed_or_experiment: false
```

## 交付物与复算方法

- `m83_independent_recompute.py`：本审阅自行编写的 Python 标准库复算源码；没有导入或执行作者的 `analyze_m83.py`、`verify_m83_saved.py`、`summarize_m83.py`。
- `m83_independent_recompute.json`：完整机器结果，包含全部哈希引用、输入指纹、逐序列分组、12 个替代救错/伤对事件、652 个 Raw/Hann 极端阈值转换事件、66 个更新资格分歧事件。
- `M83_COMPLETED_AUDIT.json`：A–F 结论、问题状态、审阅范围和可机读证据索引。
- `.aris/traces/experiment-audit/2026-09-20_run01/`：本轮任务、结论和执行元数据。

复算运行于已安装的 Python 3.13，使用 `math.fsum` 累加全部原始浮点 IoU。全部离散计数要求精确相等；浮点核对使用绝对容差 `1e-8`、不使用相对容差。1,750 个浮点字段核对的最大实际差值为 **1.0913936421275139e-11**，出现在 Empty 的总体 IoU 和；没有通过改阈值、少算序列或舍弃负面行取得匹配。终次本地复算退出码为 0，`failures=[]`。

## A. 真实 GT 来源：PASS

GT 性能参照来自训练配置的 `/root/autodl-tmp/depthtrack/train/sequences/<sequence>/groundtruth.txt`，本地审阅读取其 `../dataset_gt/<sequence>/groundtruth.txt` 副本。`../training_spec.json:8-13` 绑定原生 checkpoint、integration 与原始 inventory 指纹；`analyze_m83.py:34-37` 从该数据集位置读取 GT，并要求每条文件的 SHA 与冻结 spec 一致。

本轮将全部 **22/22 个 GT SHA** 同时与 `spec.json` 和经训练配置 SHA 绑定的 `../data_inventory.json` 逐一核对。所有 GT 行数、原 Category frame 0–N−1、M83 frame 1–N−1、冻结初框及 inventory 初框完全一致。没有用预测框生成参照。33,108 个跟踪位置包含 **28,897 个有效 GT**、4,211 个无效 GT；无效位置由 4,189 个含非有限数的 GT 和 22 个非正宽高 GT 构成。22 个初始化位置不计入 IoU 均值。

`m83_same_state.py:63-87` 读取 RGB-D 图像、首帧初框和封存原轨迹；没有打开后续帧 GT。`run_m83.sh:5-11` 先完成全量 replay 才调用分析，分析在 `analyze_m83.py:11-23` 校验完整 receipt、所有文件 SHA 与帧索引后，才在 `34-37` 读取 GT。这里“没有加载 GT”严格指推理没有加载后续帧 GT；初始化使用数据集初框是协议输入，不能被隐去。

GT 指纹证据覆盖当前副本与先前冻结 inventory 的一致性。此次没有重新下载数据集，也没有以图像人工重新标注每个 GT 框。此任务是自定义单步诊断，未调用 DepthTrack/VOT 官方综合评测工具；报告没有将本结果标成官方 success、precision、EAO、accuracy 或 robustness 指标。

## B. 归一化含义：PASS

`analyze_m83.py:25-29` 直接计算矩形交并比；`59-78` 汇总 IoU 后除以对应分组的有效位置数。IoU 没有被模型自身最大值、均值或峰值归一化。

`m83_same_state.py:110-119` 将 Raw 空间响应按自身空间质量和转成概率分布，并计算 `KL(Native || branch)`。这是显式的响应分布比较；Raw max、Hann max 和 Raw mass 仍分别保存，见 `116-119`。它没有改变 GT IoU，也没有把低性能除以自身最大值后报告为高性能。`code/lib/models/layers/head.py:175-201` 的中心响应已由 sigmoid 和 clamp 产生正值，所读源码不存在本次需要新增 epsilon、fallback 或异常分支的输入证据。

本地文件没有完整的 256 点响应图及 size/offset 图。因此，本轮**独立复算的是已保存 KL 与响应质量标量的分组和/均值**，没有独立重建每帧 KL、Raw 峰值的正确性或替代 head 的解码。平均 KL≈0.016033 不等于绝对分数或模板行为相等；当前仍有 Native 相对 Category 的 50 个资格分歧。报告保留了这个限制。

## C. 结果存在、终态、数字与 SHA：PASS

M83 数据与数字部分全部通过：

| 证据检查 | 本轮结果 |
|---|---:|
| 冻结开发序列与原始预测文件集合 | 22/22 完整，无重复或缺失 |
| M83 跟踪位置 | 33,108，逐序列连续索引 |
| Category Hann bbox 与原 Category bbox | 33,108/33,108 逐元素相同 |
| Category Hann max 与原 Category score | 33,108/33,108 完全相同 |
| previous bbox 与原 Category 上一帧 bbox | 33,108/33,108 逐元素相同 |
| search side 与 factor-4 公式 | 33,108/33,108 完全相同 |
| 两种 box × 四读出有效性检查 | 264,864 个有限、正宽高框 |
| replay / analysis / controller / verification exit | 全部为 0 |
| 全量 receipt、分析 log、验证 log 与各自 JSON | 一致 |
| `export_manifest.json` 项目 | 40/40 个文件 SHA 与字节数一致 |
| 完成归档 | SHA 与大小一致，41 个成员逐字节等于本地文件 |
| 全部本轮跟踪的 SHA 引用 | 299/299 一致 |
| 本地 integration 源码 | 146/146 个 SHA 一致 |
| 前检原始文件与全量最初 101 位置 | 所有保存字段完全相同 |
| 逐序列分组读出记录 | 436 条完整复算 |
| 分组 CSV / 逐序列 CSV | 20 / 88 行完整复算 |

证据：`predictions/receipt.json:2-8,142-144`、`saved_diagnostic_verification.json:2-9`、四个 `.exit:1`、`export_receipt.json:2-6`、`export_manifest.json:4-205`。源码、spec、launch、receipt、结果和描述性派生文件之间的 SHA 链均在复算 JSON 的 `hash_checks` 中逐项记录。原始 22 条预测是单行 JSON，具体事件用 `sequence + frame` 定位，不能把“第一行”当作只检查了一个位置。

本地 integration 清单还引用 14 个 `lib/train/data_specs/*` 文本文件，当前快照没有这些文件；它们完整列在 `integration_source_files_unavailable` 中，本次不声称已验证 160/160。M83 直接使用冻结 development22 和数据集路径，不调用这些训练分割文本来选择诊断序列。模型 final、原生基座与文本 bank 二进制未由本审阅读取或重算 SHA。推理源代码中的 final/bank SHA 断言、封存标识及全部 Category bbox/score 一致性是该部分现有证据，不能扩写为本地重新加载了这些二进制。

**独立复算的主要数字如下。** “救错”指 Category IoU≤0.1 且替代读出≥0.5；“伤对”指 Category IoU≥0.5 且替代读出≤0.1。Raw/Hann 两列是在同一分支中使用同样两端阈值比较，其他程度的改进/恶化不包含在这两个事件计数里。

| 读出 | 单步平均 IoU | IoU≤0.1 | IoU≥0.5 | 救错 | 伤对 | Raw 救 Hann / Raw 伤 Hann |
|---|---:|---:|---:|---:|---:|---:|
| Category | 0.729520896 | 4952 | 23662 | 0 | 0 | 62 / 99 |
| Empty 同状态 | 0.729481836 | 4953 | 23663 | 0 | 1 | 62 / 97 |
| Swapped 同状态 | 0.729643949 | 4950 | 23664 | 0 | 0 | 62 / 99 |
| Native 同状态 | 0.727848473 | 4953 | 23657 | 4 | 7 | 65 / 106 |

所有五类分组、四种读出都已核对，均值如下；完整 low、correct、Raw/Hann、KL 和响应质量字段见独立 JSON 与原 CSV。

| 分组 | 有效位置 | Category | Empty | Swapped | Native 同状态 |
|---|---:|---:|---:|---:|---:|
| all_valid | 28897 | 0.729520896 | 0.729481836 | 0.729643949 | 0.727848473 |
| centre_inside | 24323 | 0.866625202 | 0.866578914 | 0.866771709 | 0.864642273 |
| centre_outside | 4574 | 0.000446117 | 0.000445492 | 0.000444448 | 0.000424864 |
| native_correct | 23657 | 0.886398897 | 0.886346916 | 0.886541849 | 0.884763091 |
| native_not_correct | 5240 | 0.021264622 | 0.021283895 | 0.021297836 | 0.019426887 |

单步 Category−Empty = **+0.003905975 个百分点**，Category−Swapped = **−0.012305301 个百分点**；原始报告保留位数正确。历史递归的 −0.717389879 / +2.194584394 个百分点与 `../recursive_result.json:15-46` 一致。历史 Native 独立轨迹 0.652226263 以及主门 10/10、Category 增量门 4/4、内容门 4/8 均与其 SHA 绑定的父结果一致，见 `:7-14,940-952,980-1009`。这些历史非 Category 结果在此轮只核对了绑定来源和数字，没有另行重跑它们的独立轨迹。

**本轮发现并复核了一处已修订叙述。** 初次读取的 `NARRATIVE_REPORT.md:35` 写“未完成真实跟踪前检”，与 `../centered_real_interface_result.json:2-15` 的 101 帧真实 CUDA 接口探针回执不一致。本轮最终读取的报告已移除该过时表述，并保留未完成训练或 GT 性能评测的边界。此项已关闭；没有修改 M83 实验。

当前审阅的 `NARRATIVE_REPORT.md` SHA256：`e3334adf899e1967322cba0f89dd2d18c8f5e0089362be4dc240c39a31322592`。原始发现时指纹为 `dd37366f1ad3fe306d35d318336c5ed2e411ac876278ea9166f34de13f378384`。该叙述文件与两个描述性 CSV 是本地事后派生材料，不在 40 项原始远端导出清单内；本审阅另行记录了它们的输入指纹，并从原始读出复算表格。

## D. 代码是否调用、是否混淆状态：PASS

`m83_same_state.py:53-54` 安装捕获 hook，`84-90` 在一次原 Category `tracker.track` 中取得 RGB/depth/fused/initial/text/mask 及 head 结果。`94-97` 对同一 captured 特征执行 Empty、Swapped 和 Native head；这些调用没有执行新的 backbone、TSG 或 tracker 状态推进。`code/lib/models/sttrack/sttrack.py:144-154` 证明 captured fused 是本次 adapter 之前的搜索特征。

Category 在 `code/lib/test/tracker/sttrack.py:116-139` 提交 query、bbox 和模板。额外分支只走 adapter/head；`semantic_spatial_adapter.py:25-52` 没有对 captured 输入原地改写，head 的 ReLU/sigmoid 原地操作作用于新卷积结果。网络与 adapter 都在 eval，见 tracker `:29-30` 和 `sttrack_semantic.py:18`。运行源代码另以 `m83_same_state.py:91-100` 检查状态、query 值和模板对象身份。

模板身份断言本身不能证明任意张量内存均被独立验证；本次结合实际源码调用关系判断，没有发现额外分支污染 Category 输入的路径。全量原 bbox/score/previous bbox 复核提供外部可保存的一致性证据，但未序列化的 query/template 张量没有在本地独立重放。

解码使用本帧 `previous` 和 factor-4，见 `m83_same_state.py:101-120`，与 tracker `:119-127,192-198` 及 `processing_utils.py:32-41,69` 一致。GT 中心分组采用真实 sample_target 的 round 原点，不使用已经提交的当前 bbox。

IoU 在 `analyze_m83.py:52,60` 实际调用；五类分组及全部计数在 `53-78` 汇总并写入 `84-90` 的终态文件。Raw/Hann 与资格统计也全部出现在实际序列输出、最终结果和 CSV 中。作者已完成的 scalar verifier 原本只复核总体；本次另行覆盖了所有组及逐序列字段，因此没有把作者 verifier 的范围说成比它实际更广。

## E. 范围与叙述上限：PASS

这是固定 Category 最终权重标识、seed 2027、重复使用的 **DepthTrack Train development22** 上的全量固定历史读出诊断。`spec.json:274-287`、`diagnostic_result.json:6516-6517` 和 `NARRATIVE_REPORT.md:3-33` 对这点一致。Empty 是同一 Category checkpoint 的文本替代读出，不是另行训练的 M82 Empty 模型；Native 是同状态绕过 adapter 的原 head，不是独立原生轨迹。

中心在搜索方形内有 24,323 个有效位置，其中 Category 385 个低重叠；中心在方形外有 4,574 个位置，其中 Category 4,567 个低重叠。`4567/4952=92.225363489%` 是按已访问状态统计的低重叠帧占比。长失败会反复贡献帧，不能据此推断 92.225% 的失败起因是出画，也不能推出移除历史状态因素后的因果分解。

本审阅额外用 GT 框与搜索方形的几何交并关系直接核对：中心在外的 4,574 个 GT 中，**384 个仍与搜索方形有正面积交集，4,190 个无面积交集**。因此报告“中心在 crop 外不等于整个目标框不相交”的限定有直接数值证据。这个几何检查不评估遮挡、真实可见像素或模型恢复能力。

Category 的 Raw 读出救错 62、伤对 99；这两个极端转换计数支持保留全局移除 Hann 的谨慎边界，不能充当所有 Raw 与 Hann 收益的总度量。当前没有保存全部 256 个解码候选框的 GT oracle，因此不能声称框内 385 帧全部可被重排救回。

更新资格以所有跟踪位置为基础，先于 GT 有效性过滤，源码见 `analyze_m83.py:43-48`。共有 **654 个 frame%50=0 的机会**；严格条件为 Hann max>0.75。复算结果：

| 同状态分数 | 资格数 | 相对 Category 资格分歧 |
|---|---:|---:|
| Category | 260 | 0 |
| Empty 同状态 | 255 | 7 |
| Swapped 同状态 | 261 | 9 |
| Native 同状态 | 308 | 50 |

只有 Category 的 260 个资格对应这次真实轨迹中的模板提交；其他值是相同 Category 状态上的资格读出，并未形成替代分支的实际写入或未来历史。历史完整递归的同权重 Empty/Swapped 写入数 270/275 来自 `../posthoc_descriptive.json:7-12`，已与其父结果/source 指纹核对；不能与本表合并成只改变模板的因果实验。

当前结果支持“在这些 Category 状态上，更换当前文本很少触发跨越严重错误/正确阈值的框变化”等条件描述，不支持语言无用、跨数据集鲁棒性、独立轨迹恢复、模板未来效用或新的模型晋升。M82 原有 10/10、4/4、4/8 和未晋升状态未被本诊断改写。

## F. 评估类型：PASS，主评价为 real_gt

| 统计对象 | 分类 | 可接受解释 |
|---|---|---|
| GT IoU、低/正确帧、救错伤对、GT 中心分组、Raw/Hann GT 转换 | `real_gt` | 冻结真实数据集 GT 上的单步条件诊断 |
| Native 相对空间 KL | `synthetic_proxy` | 参照是模型产生的 Native 分布，已显式标成响应分布比较；不是 GT 准确率 |
| 峰值变动、响应质量、更新资格与分歧 | `self_supervised_proxy`，即无 GT 的内部描述性计数 | 不表示监督性能或模板未来效用；也不是新的自监督训练结果 |
| 文件 SHA、bbox/score 封存相等检查 | 确定性完整性校验 | 不单独构成跟踪性能评测 |

没有将模型自产参照伪装为真实 GT 性能，也没有把 CPU 合成张量或接口探针当作本次 M83 的 GT 结果。

## 问题处理与结论影响

1. 本轮发现的后续方法状态叙述已经修订；没有剩余修复项。
2. 不要求新增 seed、训练、远端运行、回退分支、防御代码或哈希框架。
3. 本轮没有读取 checkpoint/bank 二进制，没有远端观测，没有执行训练或模型推理。确定性复算的接受范围仅限实际核对的文件、标量与几何指标；语义审阅仍是同家族 provisional。

M83 的终态完成性、主要数字及限制性解释得到本轮文件证据支持。后续训练或部署决策仍需自己的协议与证据，不能由这份固定状态诊断自动推出。
