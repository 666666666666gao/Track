# M83 接口与 101 帧前检审阅

审阅时间：2026-09-20（Asia/Shanghai）。审阅者：gpt-6-astra / max，独立上下文的同模型家族审阅代理 `/root/m83_interface_audit`。

```yaml
audit_skill: experiment-audit
audit_scope: m83_interface_and_preflight_only
overall_verdict: PASS
integrity_status: pass
review_independence: same-family
acceptance_status: provisional
source_review: PASS
local_preflight_artifact_verification: PASS
full_m83_diagnostic_results: NOT_AUDITED
remote_observation_performed: false
training_or_inference_performed: false
ground_truth_opened_by_this_audit: false
```

**本次限定范围通过：未发现需要修改当前源码的真实接口缺陷。** 同一状态的特征捕获、额外读出的状态隔离、原 tracker 坐标解码、原 Category 逐帧封存校验和后置 GT 分析，在所审源码中相互一致。新增取得的前检原始 JSON 也已独立核对：101 个位置的原 Category bbox、score、previous bbox 和 search side 全部一致。

这一结论只覆盖源码及 `bag05_indoor` 的 frame 1–101 前检。它不确认全量 M83 已结束，不是 M83 性能结果，也不能证明替代分支的长期递归效果。没有连接远端、修改运行源码、运行推理或使用 GT 计算前检性能。

## 1. 原生和适配特征来自同一次状态计算：PASS

- `m83_same_state.py:53–54` 注册的两个 hook 仅将引用保存到 `capture`；`dict.update` 返回 `None`，不替换模型输入或输出。
- 模型先在 `code/lib/models/sttrack/sttrack.py:144–145` 计算 `fused_tokens` 并切出未适配的 search `feat_last`，随后在 `150–154` 以该张量及同一次前向得到的 RGB、depth token 调用 adapter，再将适配后特征送入原检测头。因而 pre-hook 捕获的 `fused` 确实是该 Category 帧的未适配特征，不是另一次前向或已经适配的返回字段。
- `m83_same_state.py:84–90` 先保存本帧使用的 previous bbox，运行一次原 Category `tracker.track`，再取出上述捕获值及原检测头输出。`94–97` 的 Empty、Swapped 使用同一组 RGB、depth、fused、initial、mask，只改变文本 token；Native 直接对同一个 `fused` 调用 `forward_head`。
- `m83_same_state.py:69–76` 为三套 bank 选取同名序列，并在 `73` 检查 mask 完全一致。`code/lib/test/tracker/sttrack_semantic.py:20–30` 构建本序列固定的 initial/text/mask；initial 提取只用初始化帧和初框，见 `sttrack_initial_instance_observation.py:8–20`。

这里的“同一状态”是 **Category 轨迹本帧访问到的 crop、template、query 历史以及初始化实例参考**。虽然额外 head 调用发生在 `tracker.track` 返回之后，输入已从这次前向中捕获；模板更新后并没有重新抽取替代分支特征。Native 读出因此应始终解释为 Category 状态上的未适配 head。

## 2. 反事实分支未提交状态，未见输入原地污染：PASS

`m83_same_state.py:94–97` 只调用 adapter 的 `forward` 和模型的 `forward_head`，未再次调用 `tracker.track`、backbone、TSG、初始化或模板更新流程。原轨迹的 bbox、query 和模板提交在 `code/lib/test/tracker/sttrack.py:116–139` 中完成。

`m83_same_state.py:91–100` 在额外调用前保存状态，随后检查 bbox 相等、模板列表长度与各对象身份相同，以及 query 张量值完全一致。模板检查本身是对象身份检查；不能单凭这一项声称任意原地张量污染均已被动态排除。本次同时追查了实际被调用代码：

- `semantic_spatial_adapter.py:25–52` 用 Linear、LayerNorm、矩阵运算及非原地 `masked_fill` 生成中间量，最后为 `enhanced = fused + self.delta(local)`；没有对 RGB、depth、fused、initial、text 或 mask 的原地赋值。
- `code/lib/models/sttrack/sttrack.py:175–205` 的额外 head 不引用 tracker 的 template、query 或 public bbox。其首个解码层为 `conv`，见同文件 `27`。
- `code/lib/models/layers/head.py:8–21` 的 `ReLU(inplace=True)` 位于新卷积和 BatchNorm 输出之后；`175–201` 的 `sigmoid_()` 作用于新生成的 centre/size 卷积结果，不作用于捕获的 `fused` 或先前保存的 Category 输出。`cal_bbox` 的 `142–160` 仅选择峰值和 gather 大小/偏移。
- 基础网络在 `code/lib/test/tracker/sttrack.py:29–30` 设为 eval；后添加的 adapter 在 `sttrack_semantic.py:18` 单独设为 eval。因此所审额外 head 中的 BatchNorm 不在训练模式积累 running statistics。

未找到需要增加 clone、fallback、异常分支或其他防御代码的当前证据。原模型主前向里的 query/list 更新及 TSG 张量改写不是额外反事实分支，它们属于原 Category 轨迹计算。

## 3. 解码与原 tracker 一致：PASS

| 环节 | M83 | 原 tracker / 依赖 |
|---|---|---|
| 裁剪边长 | `m83_same_state.py:101`：`ceil(sqrt(previous_w * previous_h) * 4)` | `processing_utils.py:32` 的 `sample_target` |
| resize factor | `m83_same_state.py:102`：`256 / side` | `processing_utils.py:69` |
| 原检测头峰值、尺寸和偏移 | `m83_same_state.py:105` | `sttrack.py:121–125`；`layers/head.py:142–160` |
| 尺度与中心回映射 | `m83_same_state.py:106–108` | tracker `sttrack.py:124–127,192–198` |
| 裁边规则 | `m83_same_state.py:109`：`clip_box(..., margin=10)` | tracker `sttrack.py:127`；`box_ops.py:97–106` |
| Hann 位置选择 | `m83_same_state.py:114–116` | tracker `sttrack.py:119–121` |

M83 解码使用 `previous`，避免用已经更新后的 `tracker.state` 作坐标原点。这里沿用原 tracker 的中心回映射；分析用的 crop 覆盖边界则按实际 `sample_target` 的 `round` 原点计算，见 `analyze_m83.py:49–51` 与 `processing_utils.py:37–41`。

`m83_same_state.py:120` 另有每帧断言，要求额外解码得到的 Category Hann bbox 与原 tracker 返回框严格相等。前检原始文件中的全部 101 个 Category Hann bbox 已再与原 M82 文件独立逐元素比较，未见差异。

## 4. 封存结果与 101 帧前检证据：PASS，范围有限

`m83_same_state.py:77–87` 先验证原 M82 序列 JSON 的 SHA，再逐帧精确比较原 Category `target_bbox` 和 `best_score`。`122–132` 按序列写出全部替代读出及 SHA，所有序列完成后才写总 receipt。

本次在本地进行了以下确定性复核，均未读取 GT：

| 检查 | 本地结果 |
|---|---|
| `m83_same_state.py`、`analyze_m83.py`、`run_m83.sh`、`m83_spec.json` 与 launch 记录 SHA | 4/4 一致 |
| 前检 receipt SHA 与 launch 记录 | 一致 |
| 前检原始 `m83_preflight/bag05_indoor.json` SHA 与 receipt | 一致：`c49057454a76d159ceb49613c8e5a104c4722f2db3e7f722af660453f18bf7da` |
| 前检 frame 序列 | 连续 1–101，共 101 个位置 |
| 前检 Category Hann bbox 与原 M82 bbox | 404 个坐标数值逐元素严格相等 |
| 前检 Category Hann max 与原 M82 score | 101/101 严格相等 |
| 前检 previous bbox 与原 M82 上一帧 bbox | 404 个坐标数值逐元素严格相等 |
| 前检 search side 与冻结 factor-4 公式 | 101/101 一致 |
| 前检替代分支键 | 每帧均为 category、empty、swapped、native |
| 22 个原 M82 Category JSON 与 M83 spec 指定 SHA/行数 | 22/22 一致 |
| 全量计划帧数 | 22 序列，排除初始化后共 33,108 个跟踪位置 |
| integration 清单中本地存在的文件 SHA | 146/146 一致；清单另有 14 个 data_specs 文本文件未收进本地快照 |

前检 receipt 的 scope 明确为 `mode=preflight`、`positions=101`，见 `m83_preflight_receipt.json:6–18`。状态隔离断言属于此次执行回执和源码链的证据；前检 JSON 没有序列化完整 query/template 张量，不能把本地 bbox/score 复核描述成独立重放了 CUDA 状态。

本地审阅范围内没有 M83 全量 `predictions/receipt.json`、`diagnostic_result.json` 或全量终态证据。本次没有远端观测，因此不对远端当前进度作判断。训练 final checkpoint 及文本 bank 的二进制也未在本地重算 SHA；运行脚本的校验逻辑和已核对的前检回执是该部分证据边界。

## 5. GT 进入分析的时点和来源：PASS（源码审阅）

推理阶段 `m83_same_state.py` 读取冻结 spec、训练配置、bank、checkpoint、RGB-D 帧和原 M82 预测，不打开 `groundtruth.txt`。冻结初框用于 tracker 初始化是本协议的必要输入；`no_groundtruth_loaded` 应理解为 **未打开后续帧 GT 序列**，不能扩写成“不使用初始化标注”。训练配置还含 fit 序列首框和旧汇总元数据，但未含待诊断 development 逐帧 GT，亦没有将这些元数据送入本帧预测或替代选择的路径。

`run_m83.sh:5–11` 仅在全部 replay 正常退出后运行分析。分析程序先在 `analyze_m83.py:11–23` 验证总 receipt 已完成、位置总数 33,108、spec SHA、每个预测文件 SHA、各序列 frame 列表以及 22 序列完整性，之后才在 `34–37` 打开并校验真实 DepthTrack Train `groundtruth.txt`。GT 只用于后置分组和 IoU 统计，没有回传到轨迹、词选择、模板提交或模型参数。

真实 GT 来源是冻结 SHA 对应的数据集文件，分类为 `real_gt`；当前阶段只有 **该 real_gt 分析路径的源码证据**，尚无本次审阅确认的 M83 全量 GT 分析结果。前检也没有使用 GT 计算任何性能数值。

## 6. 统计名称与单步反事实含义：PASS，保留限定语

| 输出 / 名称 | 源码定义和可支持的解释 |
|---|---|
| `category` | 原 Category 模型在自己访问的状态上的原始读出；每帧对原封存结果做严格校验。 |
| `empty`、`swapped` | 同一个冻结 Category adapter 换用相应文本 bank 的单步读出。不能将 `empty` 解释成 M82 的独立训练 Empty 模型。见 `m83_same_state.py:36,45,94–97`。 |
| `native` | 同一 Category 状态上 bypass adapter 的原 head 读出。spec 的名称是 `native_same_state`；结果键虽缩写为 `native`，最终叙述应保留 same-state 限定。 |
| `mean_one_step_iou` | 在各分组有效 GT 位置上汇总的单步 IoU 均值，见 `analyze_m83.py:48,52–62,70–78`。 |
| `rescue_vs_category` / `harm_vs_category` | 原 Category IoU ≤0.1 且本分支 ≥0.5 / 原 Category ≥0.5 且本分支 ≤0.1 的位置计数，见 `63–64`。表示本帧替代框变化，不表示已执行恢复。 |
| `raw_rescue_hann` / `raw_harm_hann` | 同分支 Raw 与 Hann 选择框跨越上述低/正确阈值的计数，见 `65–66`。没有切换后续轨迹。 |
| `native_correct` / `native_not_correct` | 按当前 same-state Native 框 IoU≥0.5 条件分组，见 `52–54`。这是按 Native 正确性分组后的条件诊断，不能当作无条件模型排名。 |
| `update_qualification_count` / `update_qualification_disagreements` | 每逢 frame%50=0 检查 Hann max>0.75 的阈值资格及相对 Category 的分歧，见 `43–46,69`；命名没有声称反事实模板已实际写入。原更新条件见 tracker `sttrack.py:129–139`。 |
| `native_spatial_kl` | 对正的 Raw centre score 按空间总质量归一化后计算 KL(Native || 分支)，见 `m83_same_state.py:110–119`。用于分布比较，不作为检测 IoU 的归一化分母。Raw max、Hann max、Raw mass 同时保留。 |

Center head 已将 score clamp 到 `[1e-4, 1-1e-4]`，见 `layers/head.py:175–201`，所审 KL 公式没有当前零概率 `log(0)` 输入证据；没有理由另外加入 epsilon 或 fallback。这个 KL 不保留绝对置信度，不能单独解释模板更新安全。spec 在 `m83_spec.json:285–287` 已注明单步和训练 eligibility 的边界，结果构造在 `analyze_m83.py:84–87` 同样明确禁止独立 Native 轨迹、替代 H10、长期恢复或新 promotion 结论。

## 7. 技能清单结论与交付边界

| experiment-audit 检查 | 本次结论 |
|---|---|
| A. GT 来源 | PASS：数据集 GT、冻结 SHA、全量封存后读取的源码路径；未核发全量性能结论。 |
| B. 分数归一化 | PASS：空间 KL 的概率归一化具有明确用途，同时保留绝对 score/mass；IoU 直接按矩形交并比计算。 |
| C. 结果存在及绑定 | PASS：所声称 101 帧前检原始输出、receipt、launch 与源码 SHA 可互相核对；全量结果不在本次接受范围。 |
| D. 死代码 | PASS（源码调用关系）：四分支均进入输出循环，分析读取全部分支并调用 IoU；全量分析的实际执行尚未由本次核实。 |
| E. 范围 | PASS：只接受源码和 101 帧接口前检，不扩展成全量、独立轨迹或基准性能。 |
| F. 评估类型 | 计划后置分析为 `real_gt`；已完成的本次独立数值复核为非 GT 的接口一致性检查。 |

**没有提出源码修复，也没有提出新增 seed、训练或推理任务。** 全量任务以后自然完成时，需依据真实终态 receipt、封存输出和最终分析文件另行判断完整性与统计结果；该工作未在本审阅中执行。

## 8. 关键输入 SHA256

```text
m83_same_state.py 976ff0c134f9f839db4e361344446632a5377ae41b8ac9cfa257ac99c00a589c
analyze_m83.py 2e14ffd9603d917c84457b2691bcd884afd76c4ab2baf1afeca6a6ae96b98533
run_m83.sh 206f78a2eed3bb877da98b8ab1877262c19c7c348461d7b98bd7b0833d73bc47
m83_spec.json 347ebaf2c15133b9c92d394f022ccb551b5e246283a6101718a01799bc9c4637
m83_launch.json 4bd0952252342e4766ca42e74e8270c42c3a25f7a9d74a8a36752d3f1a2ecfdb
m83_preflight_receipt.json e0094ba19a91b938ea93598449deae669dcb48057dee08a15c48b6a2ee69f166
m83_preflight/bag05_indoor.json c49057454a76d159ceb49613c8e5a104c4722f2db3e7f722af660453f18bf7da
training_spec.json 6163ef5b21f897ca9819c6358b03682c1278bc1b37e5ac93146ff47d332dd700
recursive_result.json 823d0560db3acfdd594c53ecfae239ebb59ef77b3b2776bc9037dcdaa3a259ea
integration.json 5881ef1594929f0e71b0f7e8da6564e1c4669cbef86b1986b2948d7e0f231a65
code/lib/test/tracker/sttrack.py 5b114e384942ecf2740e1b2da93bea855eb38246786c62e49bf5ce70b78ff8c6
code/lib/test/tracker/sttrack_semantic.py c8fc723cc4e2f2d9b8cbc49ecdd787981f4159764dfd2acb9bd52bc825267f7d
code/lib/models/sttrack/sttrack.py aa75dfcb6b8990a10293b7415332f6fe293e8584d0ad7348423e3a4a48129057
code/lib/models/sttrack/semantic_spatial_adapter.py dde93520ed569ede7f4f0b9a0c10d0b6ff179135b9928b2b3e03f41dfd3f4d88
code/lib/train/data/processing_utils.py 7aca916f5e5f62e1865fbd322ffb63dc08197c544a241938b6c822d4bfb89bf6
code/lib/models/layers/head.py 62cb2f363ea8bce491c9e6fc528076dd1e14c7d371ad69d6efd0b4062d063326
code/lib/test/tracker/sttrack_initial_instance_observation.py 2f63f49f90b08f3e205c9bb48cb22aad1d800b6bc7d1d823746c638bf9ffddde
code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml b6bda3238c9dd001aab62d87234d9ccecf1ae1cee3bd7bea5f21d6368ff4b344
```
