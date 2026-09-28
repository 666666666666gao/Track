# M100 候选分数只读诊断预部署复核

- verdict: PASS
- review_independence: same-family
- acceptance_status: provisional
- reviewer: fresh Codex agent /root/m100_score_review
- reviewer_model: gpt-6-astra
- reasoning_effort: max
- context: fresh agent / fork_turns none；实际 spawn 配置由主代理确认
- reviewed_at: 2026-09-28 11:21:04 +08:00
- scope: 本地源码、既有真实回执与 AST；未使用 SSH/GPU，未改实验代码，未读取私有人工 mapping。

未发现 BLOCKING 问题，也没有需要额外实现的 NON-BLOCKING 问题。当前脚本符合 M100 固定 final、完整面板、只读诊断范围；源码 PASS 不代替尚未执行的 M100 sanity 与完整读出。

## 已核对

1. `inspect_ab_candidate_scores.py:40-49` 要求 M99 complete 状态，核对所加载 final 权重的既有 SHA，复用 producer 的 `load_inputs`、`batch` 与原模型，使用 eval/no_grad。没有 optimizer、训练更新、checkpoint 写出或 tracker 状态操作；输出目录必须新建。
2. `train_ab_visual_control.py:75,86-93` 在数据加载时计算 GT-IoU，但明确把 `iou` 排除于 forward 的 batch；key/strata 也不进入模型。五个 Empty slot 原样复用，模型评分仍须与 visual 分支逐元素严格相等。计划中“GT after forward”应理解为输出评价用途，而非 IoU 张量的实际创建时序。
3. `inspect_ab_candidate_scores.py:53-96` 的 sanity 仅 forward 前 64 个 fit 事件；完整读出沿原顺序覆盖 2544 fit + 495 development。每条 development 原始评价字段逐字典比对旧行，两个 split 的所有原 strata 汇总均严格比对旧 summary，最后核对总数 3039；每个事件保存全部 10 候选。strata 保持原有重叠定义，不能把各层计数相加当作去重总数。
4. 原候选观察器的分数确为 `log(max(window * score_map, 1e-6))`；NMS 首个峰为原生首选，collector 又逐事件断言第 0 框与原生 tracker 框一致。`native_log_hann` 不是 log odds。导出的 residual 定义为“最终 selection logit − cached native log Hann”，native gap、residual advantage 和 selection gap 的方向均正确；它是 float32 加减后得到的有效残差。
5. quality 排名直接采用原始 `quality_logits.argmax`，sigmoid 仅用于估计 IoU 和 MAE。MAE 先对每事件 10 候选平均，再对事件平均，因候选数固定而等价于候选总体平均。quality rescues/breaks 的比较对象是 native；M99 selection 的统计另由原 `summarize` 保留。quality 是共享视觉表征上的单独监督头，其 argmax 仅为反事实读出，不构成独立模型验证或新部署策略。
6. 既有结果支持完整训练为 12 epoch / 480 update；本次对 495 个唯一 development key 重算，原 summary 与 `analysis/analysis.json` 完全一致。native 268 → selected 275，10 rescues / 3 breaks（egg_indoor@10、@46、@2439），其中 2 healthy。原生正确事件有 17 次改选，其中 14 次仍正确、3 次变错。这些是已有 M99 行的算术，未冒充 M100 分数结果。

## 实际检查与未知项

`uv run --no-project --offline python` 完成新脚本、producer、模型三个文件的 AST parse，并仅提取原纯函数 `summarize` 重算已存行；未导入或执行模型。读取的 M98 input audit 为 152 sequences / 3502 events / 219194 frames、queue exit 0；已存 M99 final reload 回执记录 495 行与 summary exact。

本复核没有执行 M100 64-fit sanity 或 3039-event 读出。当前 final 在新读出中的全候选有限数值、实际 score/residual/quality 分布、quality argmax 成败、fit/development MAE 以及新进程中双 split summary 严格相等，均须由实际运行回执确认。完整读出仍以 64-fit sanity exit 0 为前提；M99 healthy protection gate 仍为 FAIL。
