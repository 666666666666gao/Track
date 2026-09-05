# M40：STTrack VOT low22 失效起点搜索域普查

## 1. 范围与结论边界

M40 只读取 M39 中 STTrack default 的 22 条序列、303 个 anchors 和 124 个正式确认失败，不运行新模型、不训练、不修改 checkpoint，也不产生新的 VOT 指标。失败起点严格使用 VOT toolkit 记录的 `progress`：连续 10 帧 overlap `<=0.1` 的第一帧。

每个失败起点使用上一轨迹帧的公开预测框作为真实递归状态；若它是初始化后的第一帧，则使用 anchor GT。搜索框严格复现 STTrack `sample_target()`：

```text
crop_side = ceil(sqrt(state_w * state_h) * search_factor)
crop_center = previous public state center
```

固定比较 factor `4/6/7`。这里的“覆盖”只说明真目标是否进入候选搜索图，不代表网络一定能定位或选择它。

## 2. 聚合原始结果

| 失败起点类别 | anchor 数 | 占 124 个失败的比例 |
| --- | ---: | ---: |
| 真目标中心在官方 factor-4 内 | 115 | 92.741935% |
| 只在 factor-6 内 | 7 | 5.645161% |
| 只在 factor-7 内 | 2 | 1.612903% |
| factor-7 仍不覆盖 | 0 | 0% |

补充统计：

| 统计量 | 结果 |
| --- | ---: |
| factor-4 完整覆盖真值框 | 112/124，90.322581% |
| factor-4 至少覆盖真值框面积一半 | 115/124，92.741935% |
| 起点 confidence `>=0.75` | 9/124，7.258065% |
| 高置信且真目标仍在 factor-4 | 9/124，7.258065% |
| 失效前一帧 IoU `>0.5` | 88/124，70.967742% |
| 其中仍在 factor-4 的突发失效 | 81 |
| 其中已出 factor-4 的突发失效 | 7 |

起点 confidence 分位数：

| q10 | q25 | q50 | q75 | q90 |
| ---: | ---: | ---: | ---: | ---: |
| 0.153428 | 0.197638 | 0.327705 | 0.514203 | 0.664889 |

失效后连续 H10 的 1,240 个帧级搜索框中，真目标中心覆盖率为：

| factor | 覆盖帧 | 覆盖率 |
| ---: | ---: | ---: |
| 4 | 1,059/1,240 | 85.403226% |
| 6 | 1,103/1,240 | 88.951613% |
| 7 | 1,109/1,240 | 89.435484% |

factor-4 到 factor-7 只增加 50/1,240 个覆盖帧。说明如果第一次错误框已经写入递归状态，继续围绕同一个错误中心扩大搜索框，也很快会再次失去真目标；宽搜索必须在失效起点立即作为 shadow branch 使用，不能等错误状态传播后才启动。

## 3. 22 条序列逐项原始表

| 序列 | anchors | 失败 | factor-4 内 | factor-6 only | factor-7 only | factor-7 外 | 起点 confidence 中位数 | 最小中心 factor 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ball06_indoor_2 | 8 | 7 | 7 | 0 | 0 | 0 | 0.379135 | 1.305437 |
| bandlight_indoor_1 | 25 | 5 | 5 | 0 | 0 | 0 | 0.425272 | 2.000402 |
| cube02_indoor_1 | 13 | 7 | 1 | 6 | 0 | 0 | 0.209889 | 5.785096 |
| cube02_indoor_2 | 13 | 0 | 0 | 0 | 0 | 0 | — | — |
| cube05_indoor_1 | 4 | 2 | 2 | 0 | 0 | 0 | 0.268858 | 0.198157 |
| cube05_indoor_2 | 4 | 0 | 0 | 0 | 0 | 0 | — | — |
| cube05_indoor_4 | 7 | 4 | 4 | 0 | 0 | 0 | 0.182027 | 0.032111 |
| cube05_indoor_5 | 11 | 9 | 9 | 0 | 0 | 0 | 0.871208 | 1.466597 |
| cube05_indoor_6 | 16 | 1 | 1 | 0 | 0 | 0 | 0.418677 | 1.780727 |
| cup02_indoor_1 | 36 | 36 | 36 | 0 | 0 | 0 | 0.481510 | 0.401983 |
| duck03_wild_1 | 6 | 0 | 0 | 0 | 0 | 0 | — | — |
| duck03_wild_2 | 6 | 0 | 0 | 0 | 0 | 0 | — | — |
| earphone01_indoor_1 | 20 | 2 | 2 | 0 | 0 | 0 | 0.323736 | 0.782221 |
| humans_shirts_room_occ_1_A_2 | 13 | 6 | 6 | 0 | 0 | 0 | 0.521682 | 0.121991 |
| humans_shirts_room_occ_1_B_1 | 12 | 0 | 0 | 0 | 0 | 0 | — | — |
| robot_human_corridor_noocc_1_B_1 | 19 | 0 | 0 | 0 | 0 | 0 | — | — |
| shoes02_indoor_1 | 13 | 12 | 12 | 0 | 0 | 0 | 0.232339 | 0.157066 |
| shoes02_indoor_2 | 4 | 2 | 2 | 0 | 0 | 0 | 0.263263 | 0.114690 |
| squirrel_wild_1 | 9 | 0 | 0 | 0 | 0 | 0 | — | — |
| toy09_indoor_1 | 26 | 26 | 26 | 0 | 0 | 0 | 0.199339 | 0.188634 |
| two_tennis_balls_3 | 4 | 3 | 0 | 1 | 2 | 0 | 0.066402 | 6.440211 |
| yogurt_indoor_1 | 34 | 2 | 2 | 0 | 0 | 0 | 0.103543 | 0.155530 |

## 4. 搜索域不足的 9 个具体 anchor

### cube02_indoor_1：6/7 个失败起点需要 factor-6

这 6 个 backward anchors 的真目标中心所需最小 factor 为 `5.767638～5.856460`，起点 confidence 为 `0.185701～0.226275`。其中 6 个失效前一帧 IoU 都在 `0.853315～0.907924`，属于从正确跟踪突然大位移到 crop 外；factor-6 在起点能够覆盖中心，但在随后 H10 中通常只覆盖 1/10 帧，因此必须在第一帧及时建立独立分支。

### two_tennis_balls_3：3/3 个失败起点在 factor-4 外

| anchor | 起点 confidence | 前一帧 IoU | 中心最小 factor | 完整框最小 factor | 类别 |
| --- | ---: | ---: | ---: | ---: | --- |
| @0F | 0.066402 | 0.302843 | 6.541807 | 7.740181 | factor-7 only |
| @50F | 0.043824 | 0.473593 | 6.440211 | 7.738946 | factor-7 only |
| @129B | 0.072996 | 0.868949 | 4.060946 | 5.077692 | factor-6 only |

前两个 anchor 即使 factor-7 也只覆盖目标中心，完整目标框约需 factor 7.74。扩大搜索框只提供一次候选可见机会，并不保证网球身份选择正确。

## 5. 仍在 factor-4 内的主要问题

115/124 个失败起点的目标中心仍在官方搜索框内，其中 112 个连完整 GT 框都在 crop 内。最严重的序列均属于这一组：

```text
cup02_indoor_1       36/36 failures inside factor-4
toy09_indoor_1       26/26 failures inside factor-4
shoes02_indoor_1     12/12 failures inside factor-4
cube05_indoor_5       9/9 failures inside factor-4
ball06_indoor_2       7/7 failures inside factor-4
humans_shirts A       6/6 failures inside factor-4
```

这组几何证据排除了“主要因为搜索框太小”的解释，但仅凭几何不能把全部失败都命名为身份切换；它们还可能包括遮挡、外观变化、响应峰错误和候选回归失败。下一步必须读取该帧真实 response top-K，判断正确候选是否已经存在但没被选中。

### 高置信持续错误不是全部失败，但确实存在

只有 9/124 个起点 confidence `>=0.75`，全部在 factor-4 内，集中于 `cube05_indoor_5`（6 个）和 `toy09_indoor_1`（3 个）。例如：

| anchor | confidence | 前一帧 IoU | 起点 IoU | 真值框是否完整在 factor-4 |
| --- | ---: | ---: | ---: | --- |
| cube05_indoor_5@491B | 0.923919 | 0.138046 | 0.062607 | 是 |
| cube05_indoor_5@387B | 0.916983 | 0.136268 | 0.062607 | 是 |
| cube05_indoor_5@100F | 0.914077 | 0.126911 | 0.092308 | 是 |
| toy09_indoor_1@750B | 0.832597 | 0.125624 | 0.098270 | 是 |

这些高置信帧的前一帧已经只有约 `0.13～0.20` IoU，因此它们更多表现为错误状态已经形成后的高置信强化，而不是第一次突发切换。单独使用低 confidence 只能发现部分初始风险，无法阻止后续高置信漂移。

## 6. 对旧判断的修正

此前根据较晚失败帧或粗略距离得到的 `ball06` 需要约 factor 5.42 的说法，不能再作为“失效起点出搜索域”的证据。M40 对正式 VOT `progress` 的逐 anchor 精确重建显示：

> `ball06_indoor_2` 的 7 个确认失败起点，真目标中心全部仍在官方 factor-4 内。

因此 `ball06` 目前应归入“crop 内定位/外观/候选选择失败”，而不是主要搜索域失败。真正被 M40 证明的起点搜索域序列只有 `cube02_indoor_1` 和 `two_tennis_balls_3`。

## 7. 为什么不能直接用 confidence 启动宽搜索并覆盖状态

搜索域不足的 9 个起点 confidence 全部 `<=0.226275`，所以固定 `confidence<=0.30` 对这 9 个事件的诊断召回率为 100%。但是同一阈值总共触发 56 个失败起点，其中 47 个目标本来就在 factor-4 内：

```text
真正出 factor-4： 9
总触发：          56
出框精度：        16.071429%
```

因此低 confidence 可以作为“启动 factor-7 shadow”的宽松风险信号，但不能作为“把 factor-7 结果直接写回 state/template”的提交规则。宽分支仍需候选身份和未来短窗验证。

## 8. 下一步唯一建议：M41 候选可恢复容量诊断

M41 不应立即实现复杂 router，而应先做一次最小的、只读的失败帧 counterfactual：

1. 对 115 个 factor-4 内失败起点，重放到失效前一帧，导出官方 factor-4 response 的 NMS top-K 候选及解码框；用 GT 仅作事后 oracle，统计是否存在 IoU `>0.5` 的非 top-1 正确候选。
2. 对 9 个 factor-4 外失败起点，从同一受保护状态只运行一帧 factor-7 shadow，记录 top-1 与 top-K oracle IoU；不提交 bbox、模板、query 或 memory。
3. 如果 factor-4 top-K 中存在跨多序列正确候选，主线继续 candidate-own RGB-D + language identity association + protected/tentative 原子事务；如果不存在，则问题在候选生成/表征，不能继续训练 selector。
4. 只有 factor-7 shadow 在 9 个事件中真实产生可恢复框，才实现低 confidence 风险触发的宽搜索；否则封存扩大搜索方向。

## 9. 证据身份

```text
source commit:
72b6446f5ba0e96c8882001f3286585fd81cff30

M39 result SHA256:
cf953c0d3c69609bcd83c11cb24ba57f37e30b38d3b3bcad32860b3a9ba9c1b5

M40 spec SHA256:
eaa3c69631d5950ad360e67bd99fc95f9850c32821e93426109fb9754aa6eef9

M40 analyzer SHA256:
4295f414283c2a9ac7ec9405cca8051eccf37f61e4fa79d7464132e15c8f6f24

M40 result SHA256:
385d088a99b9b6eeadaeb55526194c887bc5ee575a58b7a5dfb5a38bfc8068ec

M40 raw CSV SHA256:
5da4c09f2fa7c7aa1242342b7bd0ae405b5f78e7e88d152967c9426aef5b8837
```

M40 不改变项目正式最好指标；VOT full-127 仍为 `74.020583 / 82.579344 / 89.565651`，DepthTrack 与 CDTB 保护结果不变。
