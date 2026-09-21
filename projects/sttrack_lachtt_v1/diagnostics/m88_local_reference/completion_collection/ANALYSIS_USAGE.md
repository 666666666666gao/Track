# M88完成后描述性复算

当前状态：分析源码已用封存M87/M84证据验证；尚未用于M88完成态，不能据此报告M88指标。

在收集器成功结束、下载归档、逐文件验证manifest并解包到`completion_collection/evidence`后，在本目录运行：

```powershell
uv run --no-project python analyze_completed.py --evidence evidence --output analysis --control-rows evidence/references/M84_category --control-receipt evidence/references/M84_category_receipt.json --control-result evidence/references/M84_recursive_result.json
```

程序先验证三组各22条预测和M84对照回执及文件SHA，再打开开发GT。独立计算连续矩形IoU、有效帧、均值、低重叠帧和H10，与封存结果逐项比较。输出全部22条序列、Category相对M84/Empty/Swapped的首次bbox差异、连续至少10帧严格损害/改善区间及默认规则下的模板写入。

- 严格损害：Category每帧IoU≤0.1，对照每帧IoU≥0.5；改善反向定义。无效GT打断区间，初始化帧不评价。
- 写入：非初始化帧，frame%50==0且保存score严格大于0.75。GT仅用于事后质量分类。
- 首次bbox差异不是首次身份错误；各轨迹状态不同，区间和写入次数不是因果归因。
- 不改变冻结14项条件。JSON中的门来自封存结果，明确标记未独立复算；完成态仍需另行审阅。
- 自动原类别不作语义真值；不使用M87的助手初筛标签替代M88输入核验。

历史验证：以M87三组Category/Empty/Swapped及M84 Category为输入，88条完整原始轨迹的指标与对应封存结果一致；其中可对应的84个严格区间、787次写入与既有独立审阅CSV一致（浮点容差1e-10）。见`historical_replay_validation.json`。这不包含M87 Old条件，不重跑模型，不是M88效果验证。
