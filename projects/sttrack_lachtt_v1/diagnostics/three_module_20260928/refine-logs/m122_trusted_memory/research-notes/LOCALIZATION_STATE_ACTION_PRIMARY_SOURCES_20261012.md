# 不可靠定位怎样连接到实际动作：官方源码核对（2026-10-12）

本笔记只核对源码机制，不是新实验结果。官方版本固定为 [visionml/pytracking `7eb9e74bd3d40e29dbcec444902237da13de247b`](https://github.com/visionml/pytracking/commit/7eb9e74bd3d40e29dbcec444902237da13de247b) 和 [Daikenan/LTMU `121ddd07914a8577e336fed24fa47763b6780fdd`](https://github.com/Daikenan/LTMU/commit/121ddd07914a8577e336fed24fa47763b6780fdd)。下面的 L 行号按这两个 SHA 的原始文件逐行计数，链接不指向可变 master。

## 四种动作不能合并成一个“置信度”

| 实现/判定 | 位置/尺度提交 | 下一帧搜索 | 记忆/在线学习 | 对外输出 |
|---|---|---|---|---|
| DiMP `not_found` | 跳过位置、尺度及 IoU 框细化，保留原状态 | 同一局部采样函数继续读原位置/尺度；此路径没有失踪触发的扩区或全图检测 | 不调用 classifier 更新 | 默认返回保留状态的框；只有显式启用 `output_not_found_box` 才输出四个 -1 |
| DiMP `uncertain` | 仍可提交峰值位置并细化框；框宽高仍可改变，搜索用 `target_scale` 更新由配置控制 | 随提交状态继续局部观察 | 不调用 classifier 更新 | 返回当前位置框 |
| KeepTrack `not_found` | 不提交新位置/框细化；可单独改变搜索尺度 | 调用历史尺度重估；仍在保留位置采样 | 不调用 classifier 更新 | 默认框仍存在；另返回 `object_presence_score` |
| KeepTrack `uncertain` | 与 DiMP 同样仍可提交和细化；默认允许搜索尺度更新 | 不触发失踪尺度重估；跟随提交状态 | 不调用 classifier 更新 | 框与 presence 分数继续返回 |
| DiMP+LTMU 验证失败 | 局部步骤已先执行；满足重检测接受条件才用全局候选重设位置和大小 | 独立 GlobalTrack 接收完整当前图像，产生候选，再用 MDNet 验证 | Meta-Updater 控制局部 classifier 更新与 verifier 新样本收集；全局验证不回滚此前局部更新 | 仍返回 `last_gt` 框及组合置信度；没有自动以失踪状态输出空框/零分 |

DiMP 四列依据：[dimp.py L105–175](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L105-L175)。KeepTrack 四列依据：[keep_track.py L128–235](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L128-L235)。LTMU 的提交、更新、检测、输出顺序依据：[Dimp_LTMU.py L373–417](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L373-L417)、[L462–529](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L462-L529)，完整图像检测入口见 [global_track.py L63–76](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Global_Track/Global_tracker/global_track.py#L63-L76)。

## 必须保留的源码细节

- **DiMP 并非把全部不可靠帧都冻结。** 低峰值可标记 `not_found`，强竞争峰及位移关系可标记 `uncertain` 或 `hard_negative`；后者允许提交并使用不同学习率。默认 dimp50 启用 advanced localization 和 `update_scale_when_uncertain=True`。[定位分支 L238–303](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L238-L303)、[配置 L41–49](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/dimp/dimp50.py#L41-L49)。即使禁搜索尺度更新，IoU细化仍可写 `target_sz`；宽高和搜索尺度不是同一状态量。[细化 L697–711](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/dimp/dimp.py#L697-L711)。
- **KeepTrack 的搜索调整不是全局恢复。** `search_area_rescaling` 取最近最多60个已记录尺度，筛选不小于最后记录尺度的值，再按失踪计数选择历史样本求均值。它没有遍历全图，也不保证逐帧扩张或覆盖重现目标。候选仍来自当前局部 score map 的极大值；身份丢失后还可按近期分数重选候选。[尺度 L227–235](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L227-L235)、[局部候选 L437–459](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L437-L459)、[重选 L97–114](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/candidates.py#L97-L114)。
- **记忆写入和记忆利用也是不同动作。** KeepTrack 的正常/困难帧可以先存样本，优化时再把低 certainty 样本权重置零；因此“没有参与这次学习”不等于“没有写入”。默认配置启用此权重处理。[更新 L850–915](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/tracker/keep_track/keep_track.py#L850-L915)、[配置 L65–68](https://github.com/visionml/pytracking/blob/7eb9e74bd3d40e29dbcec444902237da13de247b/pytracking/parameter/keep_track/default.py#L65-L68)。
- **LTMU 的 MU 不等于重检测，也不是始终与 flag 求交。** `track_updater` 提交局部状态但不在其内部更新 classifier；wrapper 初始用 flag 产生 update，达到 `start_frame` 后被时序 logits 比较结果直接覆盖，随后才执行 classifier 更新。默认启动历史条件为200。其嵌入 DiMP 的 `update_classifier` 一旦被调用就写 memory。[嵌入 DiMP L156–218](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/pytracking/tracker/dimp/dimp.py#L156-L218)、[memory L594–601](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/pytracking/tracker/dimp/dimp.py#L594-L601)、[默认历史 L16–24](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/run_tracker.py#L16-L24)。官方 README 明确将 +MU 与增加 Global Detection/Verifier/Bbox refine 的 +LTMU 分开。[README L9–11](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/README.md#L9-L11)。
- **LTMU 也不能简化成“拒绝坏框，输出最后可信框”。** `local_track` 先把局部 state 写入 `last_gt`；之后 `md_score<0` 或 `flag==not_found` 才触发全局检测，取检测分数前10候选再由 MDNet选最高者。MDNet为正且 failure counter>=5 才接受全局框；源码未在 found 分支重置该 counter，不能称为“连续5帧”。单独 `uncertain` 不保证触发重检测；失踪时零分的代码是注释。以上都在 [wrapper L373–529](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L373-L529)，候选排序见 [L157–164](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L157-L164)。verifier训练函数仍每帧被调用，内部按样本收集计数调度，禁收集不等于禁用已有记忆训练。[L292–298](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L292-L298)、[L507–510](https://github.com/Daikenan/LTMU/blob/121ddd07914a8577e336fed24fa47763b6780fdd/DiMP_LTMU/Dimp_LTMU.py#L507-L510)。

## 当前实现的对应位置

[dense_target_decoder.py](../../../dense_target_decoder.py) L109–122：quality 乘进 response 后始终 `response.argmax`；observation logits 作为另一个输出，未参与提交拒绝。[full_dense_tracker.py](../../../full_dense_tracker.py) L53–73、L88–120：按上一预测做 factor4/256局部观察，选中框立即写 `native.state`，observation概率只记日志，下一帧继续同一路径。[trusted_template_tracker.py](../../../trusted_template_tracker.py) L44–67：框、query、selected feature 先提交；C仅在原生 interval50/threshold0.75 已满足时控制动态模板写入，不能阻止该帧状态跳转或启动新区域搜索。

上述三份本地源码本次读取 SHA256 依次为：
`055d9ddffe4e332fa9a04fca93a6ec301d031a35b198f93a26809d7baa7104f2`；
`0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`；
`1dd4873192b05f6921cdf729a525ed684ed96a18bffdfc48673cda14d50352e0d`。

当前推车证据是：所示帧只有一辆推车，类别正确；175→176预测中心跳约404px，标注中心约24px，首次低重叠时GT仍在crop内，后期才离开crop。候选生成、排名和精修的首次失误归因仍未测量，不能用“有状态动作缺口”替代原因证据。见 [已保存的定位诊断](../M124_EPOCHWISE_START_20261012/定位失败说明.md)及其 onset_rows.json；中心位移是本次任务给定的已核对量。

## 可借鉴与尚未证明的边界

可借鉴的是把**可靠性判断、位置提交、搜索观察、记忆写入/利用、报告分数**分别连接到可检查的动作，并核对调用先后；全局恢复还必须有实际看见新区域并验证候选的路径。质量/存在输出本身、只冻结模板或只降低报告分数，都不产生新的图像证据。这是上面源码与当前调用链的比较结论，不是移植后的性能结论。

不能直接移植官方响应/关联/验证阈值、历史窗口、失败计数、crop尺度或学习率：这些依赖其各自模型、训练和数值标定；本笔记不提出本项目新阈值。KeepTrack的局部身份传播与LTMU的GlobalTrack/MDNet并非可直接替换当前RGB-D/text输出的一段开关。

[M44 candidate association](../../../../m44/EXPERIMENT_TRACKER.md) 已真实训练、完成递归评估且性能失败；[M47 multipositive correspondence](../../../../m47/EXPERIMENT_TRACKER.md) 同样已训练并失败，不能列为新机制或未试方向。本次源码研究不证明首次推车错误的唯一原因、observation/quality已校准、任何恢复方案有效、或full9目标已达成。

M124现有完整Train152联合A+B+C、own history、人工核验文字、冻结native+CLIP、seed2027、max20继续原运行。本次只保存本笔记；没有改源码/训练、神经前向、GPU/SSH/进度查询、包安装或实验启动。主目标仍待真实full9指标验证。

