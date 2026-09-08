

## 5.156 M78完整低22结果与保存输出核验

本节记录完成态VOT低22（22序列、303个multi-start anchor），不是完整127序列成绩。沿用§5.154通过开发10/10及同权重内容8/8的M78 Category最终权重；固定seed2027、既有五槽类别保留协议和原生Hann/模板/query规则。没有增加seed、checkpoint选择、在线caption或新推理模块。

| 指标（%） | M39原生STTrack低22 | M78 Category低22 | 差值（百分点） |
| --- | ---: | ---: | ---: |
| EAO | 57.135993 | 54.045964 | -3.090029 |
| ACC | 75.719622 | 75.697432 | -0.022190 |
| ROB | 73.022401 | 68.721059 | -4.301342 |

确认失败anchor：124→140；救回5个，新增21个。原生七条零失败序列中新增失败的序列：cube02_indoor_2、humans_shirts_room_occ_1_B_1、robot_human_corridor_noocc_1_B_1。这些数量按完整303个anchor逐一配对，不能当作全127指标贡献。

冻结条件：EAO_minimum=FAIL；ACC_minimum=FAIL；ROB_minimum=FAIL；failure_reduction=FAIL；native_zero_failure_protection=FAIL；all303_bound_and_complete=PASS。

低22未通过全部冻结条件，本次不进入完整三数据集评测。保留开发集和内容对照的正结果，同时明确记录外部开发集合上的收益与损害；不改变本轮门槛，不更换权重或文本来改写结论。

| 序列 | anchor数 | M78确认失败数 |
| --- | ---: | ---: |
| `ball06_indoor_2` | 8 | 8 |
| `bandlight_indoor_1` | 25 | 6 |
| `cube02_indoor_1` | 13 | 7 |
| `cube02_indoor_2` | 13 | 3 |
| `cube05_indoor_1` | 4 | 2 |
| `cube05_indoor_2` | 4 | 0 |
| `cube05_indoor_4` | 7 | 3 |
| `cube05_indoor_5` | 11 | 9 |
| `cube05_indoor_6` | 16 | 2 |
| `cup02_indoor_1` | 36 | 36 |
| `duck03_wild_1` | 6 | 0 |
| `duck03_wild_2` | 6 | 0 |
| `earphone01_indoor_1` | 20 | 1 |
| `humans_shirts_room_occ_1_A_2` | 13 | 6 |
| `humans_shirts_room_occ_1_B_1` | 12 | 1 |
| `robot_human_corridor_noocc_1_B_1` | 19 | 2 |
| `shoes02_indoor_1` | 13 | 12 |
| `shoes02_indoor_2` | 4 | 2 |
| `squirrel_wild_1` | 9 | 0 |
| `toy09_indoor_1` | 26 | 26 |
| `two_tennis_balls_3` | 4 | 4 |
| `yogurt_indoor_1` | 34 | 10 |

逐序列原生对照、救回与新增失败见per_sequence_failures.csv；完整303个anchor、方向、长度及失败配对见anchor_comparison.csv。上述表为失败统计，不冒充逐序列EAO/ACC/ROB；三项聚合指标直接核对已有official toolkit analysis产物。

完成核验重新绑定全部303个初始化key，验证909份保存输出的SHA256，重算全部303条轨迹的失败及总长度220483，核对指标、救回/新增数与全部冻结条件。核验没有新跟踪调用、caption调用或优化步骤，也没有重跑官方analysis。检查完成不等于独立模型审阅PASS。

绑定bundle SHA256：`d1507b789040ae936721f4be65385d40e902d7fcf7f9a9c325bb3d5e19d95e79`；最终head：`2c3b8acc538605abb240b17654a66ab0fbd324fa1b1ca6a8688dfdcce02b23a1`；完成结果：`64f23763ee6df91d80e81754e7db3e654196eb377b624e258768fddd970d5fc3`；保存输出核验：`bfa68530944973824237b0131331fb7c0279bb75b4e3f636f3655d6cec87f796`。

发布时条件队列快照见followup_snapshot.json，采样时间UTC2026-09-08T03:56:32.448770+00:00；该快照不代表未来完成态。完整评测目录当时存在：False。无论队列进入何阶段，只有同一模型、文本协议与运行策略完成DepthTrack Test、CDTB和全VOT并核对目标后，才能判定项目目标达成。本节不宣称目标完成。

本轮证据发布于projects/sttrack_lachtt_v1/diagnostics/m78_raw_competition/low22_completed/。已有Train开发22及VOT低22均为反复使用的开发集合，不能称为完全未见测试。当前磁盘可用1670959104字节；本次核验与导出不删除权重，两份Qwen继续保留。
