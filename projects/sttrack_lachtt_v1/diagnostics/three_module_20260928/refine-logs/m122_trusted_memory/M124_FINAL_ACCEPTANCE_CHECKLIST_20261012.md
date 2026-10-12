# M124最终完整验收清单

2026-10-12。本清单来自本地当前源码与已经捕获的实验状态，不是最终结果审核，也没有新增服务器/GPU/NN查询。最近实际服务器观察仍18:32:20，第3轮50/152；没有suite终态或M124 VOT结果。因此当前目标未完成。

## 1. 训练与选定权重

| 要核验的要求 | 完成时应检查的原始证据 | 当前证据范围 |
|---|---|---|
| 完整Train152、seed2027、A+B+C联合、自身预测历史、冻结底座 | source/input gate哈希；每轮fit/result.json及152条sequence_log；latest/final的实际哈希；冻结前后和保存重读记录；人工bank、spec/labels绑定 | E1/E2完成态已捕获，E3未完成；E1继承37条前缀与handoff RNG限制必须保留，不能写成从零全新训练 |
| 最多20轮，每轮完成两个完整OPE | epoch_history、每轮train/track/analyze实际退出回执和指标；实际序列/帧覆盖与预测字节 | 只观察到前两轮完整结果，不能由计划推断后续完成 |
| 连续两轮两个F都严格下降才早停 | 按相邻轮previous/current原始F重新计算；一项不降即清零；首次以既定CurrentC两F为参照 | 已捕获streak0，尚未终止 |
| 同轮DT/CDTB F等权均值选择best，平局保留较早；只选训练后轮次 | 从完整history重算score/replace_best；best.json、best.pt、对应fit/final.pt及evaluation bundle的实际哈希一致 | 当前best E1；保护CurrentC另列，不能悄悄放入训练轮次候选或拼接指标 |

`m124_epochwise_selection.py`实际使用两项严格小于判断、连续计数2停止、score=sum(F)/2、严格大于才替换best。完整suite不使用VOT挑checkpoint。

## 2. 同一最终权重的三数据集

| 范围 | 完成时应检查的原始证据 |
|---|---|
| DepthTrack Test50 / 76373帧 | 所选epoch的plan/bundle/final；metrics.json complete；receipt的50条与76373帧；每序列bbox和confidence实际字节及对应SHA；冻结前后、无外部优化、后续GT未进入跟踪 |
| CDTB80 / 101956帧 | 同一final与bank；同样的metrics/receipt/预测字节链，80条与101956帧；不能换另一轮或数据集专属策略 |
| VOT127 / 1765合法anchor | best_vot实际阶段退出0；selection/plan/final/bundle/bank/binding绑定；所有分片合并清单和实际结果文件SHA；官方完整分析及封存result complete_full127；external_optimizer_steps=0 |
| 完整收集 | collector的all_results.json三行均为同一final，selection/result/merge/analysis哈希可对照原字节；suite result与controller.exit及各stage回执真实存在且退出0 |

代码中的complete/status/计数/布尔声明不是独立重跑证明。最终审查需读取真实产物和关联原字节，不能只看汇总绿灯。固定输入bank不等于语义真值的独立再次确认。

## 3. 达标判断

原始精度下核对：DepthTrack P/R/F分别≥65.2/64.9/65.1；CDTB≥72.9/75.6/74.2；VOT EAO/ACC/ROB分别严格>77.9/82.1/93.7。不能以四舍五入数字通过。

`formal_nine_metrics=True`表示九项指标齐全；`any_joint_pass=True`才表示收集器计算的联合达标。两者都仍需最终产物审查。`independent_completed_audit=False`在collector源码中明确保留；之前的候选回放source PASS不等于本次完整结果审核。

## 4. 必须随正式结果披露

- DepthTrack Test50和CDTB80参与checkpoint选择，不称 untouched holdout；它们没有训练梯度，VOT不参与选择。
- 人工文字使用多帧辅助核验；该额外输入协议与自动首帧文字结果分开列。
- 继承旧前缀/初始化/预训练来源如实披露；固定seed不等于逐位确定性。
- 不混权重、bank、checkpoint或不同模型的最好单项。部分回放、单帧oracle、全框R与正式指标分别报告。

## 5. 验收后的动作

九项经实际产物审核均达标：按用户要求停止，不追加候选NN诊断。若未达标：完整记录最终九项和逐序列损害；整个suite终止且GPU释放后，再核验已准备回放入口的实际终态/同权重绑定并进行必要诊断。当前保持原训练/模板/搜索/损失不变，不因本地等待或观察超时重新启动训练。

下一次实际服务器检查不早于19:32:20.203063，原native45653/PID28588/functions149仍负责300秒本地等待。此文没有执行新的验收器、实验或NN回放。

## 源码记录

- `m124_epochwise_selection.py`: `24d73496437a60dea724bca14b51c8e8f0742657fb242ef4c793ea37ee9c4a24`
- `run_m124_epochwise_suite.py`: `63924cc9f7f5e38ab605cb8f40dddf083321bee12248e71a7e8516f5a2d17dfc`
- `collect_m124_epochwise_results.py`: `7a48b9af70c96ffc462ce00c32a38ef9cc053aeb0eba2598e1f8a0a2b202374f`
