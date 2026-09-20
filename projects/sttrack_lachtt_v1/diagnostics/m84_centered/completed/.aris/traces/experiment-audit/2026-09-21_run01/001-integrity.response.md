已完成并封存审计：

- [M84_COMPLETED_AUDIT.md](<D:/Program Files/UserCache/gb/codex/tmp/sttrack_m84_centered_20260920/completed/M84_COMPLETED_AUDIT.md>)
- [M84_COMPLETED_AUDIT.json](<D:/Program Files/UserCache/gb/codex/tmp/sttrack_m84_centered_20260920/completed/M84_COMPLETED_AUDIT.json>)
- 独立复算脚本与详细结果位于 [audit 目录](<D:/Program Files/UserCache/gb/codex/tmp/sttrack_m84_centered_20260920/audit>)。

**保存证据范围内的结果完整性为 PASS；确定性检查为 PASS；冻结的性能接受条件为 FAIL，8/14 通过。** 语义审查保留 `WARN / same-family / provisional`，原因是单 seed、反复使用的开发集和部分原始证据仅在远端，未发现数字造假、以模型输出充当 GT、指标自归一化或虚构结果。

确定性检查结果：

- **4,480 项检查，0 错误**。
- **671 项 SHA 绑定比较**，覆盖 **367 个本地实际路径**；原始 manifest 的 **283 个文件**和新增 native manifest 的 **25 个文件**全部匹配。
- 独立重算 **110 个原始序列文件**：M84 三组共 66 个、M82 Category 22 个、原生参考 22 个。复现全部逐序列与汇总指标、44 个 LOO、14 个冻结门及全部 CSV。
- 最大数值差为 **7.275957614183426e-12**，来自求和顺序；所有整数计数与门槛布尔值精确一致。
- 原生与 Empty 的 **33,130 个 bbox 数组**及 **33,108 个非初始帧 score**全部逐值精确相等。
- 核验训练 **130 条序列、186,694 次 track、5,798 次更新、3,917 条采样状态记录**。直接解析 final checkpoint，核对元数据、23 个优化器 step 和 **868,253 个保存的浮点值**，全部有限。

性能门槛复算维持原结论：

| 比较 | 通过 |
|---|---:|
| 相对 M82 增量 | 0/4 |
| 相对 native | 4/5 |
| 相对 Swapped | 3/4 |
| Empty/native 逐序列指标一致 | 1/1 |

Category pooled IoU 为 **0.708521427828**，低于 M82 的 **0.729520895605**。唯一违反 native 零 H10 保护的序列是 `mobilephone02_indoor`，其 Category H10 为 **[497,701)**。Category−Swapped 的 LOO 在删除 `colacan01_indoor`、`flower02_wild` 或 `ghostmask_indoor` 时分别变负；19/22 为正，Category−Empty 则 22/22 为正。

新增损害定位也全部复现：**18 段 / 1,074 帧持续损害，26 段 / 2,924 帧持续改善**。这些是保存递归轨迹的事后定位，不支持原因归属。

报告明确列出 **175 个未本地重散列的哈希声明项**，其中 130 项是未导出的拟合 GT；其余包括 bank、t0、base、历史源分片等。原始两个 native 完整分片仍在远端，因此更强的 bbox/score 一致结论限定于已核验的 SHA 绑定提取包。未声称重新复现全部训练、原始分片、base 不变性、latest/t0 一致性或全部内部 query/template 状态。

输出 SHA256：

```text
M84_COMPLETED_AUDIT.md
e0014ea59813bd024363daf264ea2331e32db98944d6efe063633d83d7956fa4

M84_COMPLETED_AUDIT.json
249c36d06426ea29aa706cb6504ea02996a880138af6a0314d3325928c5130dd
```

现有代码、检查点、标签、日志与回执均未修改。最终 narrative 的更正文案和新增 native 边界已读过并绑定到审计。

