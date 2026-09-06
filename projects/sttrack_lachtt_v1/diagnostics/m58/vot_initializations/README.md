# M58 低22初始化观测导出与真实 multi-start 调度核对

已从冻结的低22工作区导出 **303个实际toolkit初始化观测**，并通过原MultiStartExperiment.execute的CPU记录器检查：初始化框、每条轨迹的完整帧路径顺序及各工作区的case顺序全部一致。**本轮没有运行跟踪模型、Qwen或新TraX会话，也没有新增VOT指标。**

## 本轮解决的接口问题

上一阶段的文字生成器接收toolkit实际xywh清单，尚未实现VOT侧的上游导出。本轮直接加载既有低22四个分片的Workspace、stack和dataset，执行与VOT相同的experiment.transform，包含自动SingleObject转换和ignore-special处理；使用find_anchors及_get_initialization读取各个合法初始化。没有另写一套GT字符串解析或polygon外接框公式。

当前303个初始化区域实测全部是toolkit Rectangle，154个正向、149个反向。toolkit自身的Rectangle以float32保存xywh，本轮把这些实际属性值导出为`vot_toolkit_xywh`，后续由已验证的共用文字准备器应用完整TraX协议转换。导出器只支持当前实测的矩形输入；若后续集合出现其他region类型会明确失败，没有加入未经验证的转换路径。

冻结来源仍是既有low22的shard_manifest.json，SHA=600b1ebb8b0c2f69b831f954e907e63709fd69afb7ea94c5b58e8c7408a29eed；源集合、anchor索引/方向/值、预期轨迹名称和长度全部核对。caption清单按官方轨迹ID排序，跟踪继续使用各工作区的调度顺序；并行工作区之间不承诺统一的墙钟顺序。

完整私有清单已保存在服务器low22_inputs/caption_inputs.json，SHA=5149bee97236f96587d90191087eef7434373fc714d6f5dbfe4d00583a6859e8，可以直接交给上一阶段的文字准备器。此文件包含真实初始化框，仅留服务器；公开的是省略框和图像路径的anchor摘要、来源指纹和检查结果。303条初始化记录不能等同于已经生成了303份新文字，实际像素/框的去重仍在文字prepare阶段完成。

## 实际调度检查

| 检查 | 结果 |
| --- | ---: |
| 冻结低22序列/anchor | 22 / 303 |
| 正向/反向初始化 | 154 / 149 |
| 实际toolkit初始化区域类型 | 全部Rectangle |
| 原execute发出的初始化框与导出值 | 303/303相同 |
| 全部帧路径及方向顺序 | 220483个位置全部相同 |
| CPU记录器收到的update调用 | 220180 |
| 模型调用/Qwen调用/新TraX会话/正式预测文件 | 均0 |

检查调用的是安装版本0.7.1的真实MultiStartExperiment.execute及其transform；替换的是运行时工厂、结果存储和Trajectory。CPU记录器只接收图像路径与初始化区域，逐帧累计路径摘要，比较完整顺序。它返回的占位框只满足调度API，不用于任何精度/生存分析，也不写入正式结果目录。调度检查耗时40.77秒，export和scheduler退出码均为0。

toolkit在构建sequence时会加载既有GT与anchor等元数据。本轮只使用其中的初始化框与调度信息，不把后续GT用于监督、诊断选择或指标计算，也不把公开集像素送入Qwen/跟踪网络。因此它是评测输入准备与接口检查，不是新的模型开发效果证据。

## 四位小数问题在当前低22上的实际范围

将完整TraX转换应用于这303个toolkit框后，**改变的框为0/303，最大坐标变化为0**。这批坐标当前不触发§5.91高精度合成用例揭示的索引差异。

完整转换辅助函数仍准确表达协议，但不能用它解释既有低22的性能下降，也不能将这项准备记作模型指标修复。§5.91的高精度合成新旧索引拒绝/接受证据依然成立，其适用范围和当前303个实际输入必须分开。

## 当前状态与下一步

20:59:22 CST实查，两条训练及两条调度Python进程仍在；160个父推理文件、六个评测接口和内容方案SHA保持一致，尚无开发完成结果。本轮未修改训练、文字内容、模型策略或推广门槛。两个Qwen继续保留，磁盘余2921897984字节。

下一步等待M58固定预算训练及完整开发递归。主门和固定同权重内容门通过后，先验收真实GPU文字生成与模型入口，再使用本轮清单为低22生成当次初始化文字并进行正式配对。低22有实际改善后，才验证同一个最终bundle的DepthTrack Test、CDTB和完整VOT。此导出器尚未对完整127集合运行，也没有新建自动GPU队列。

无新独立Astra/max审阅PASS，项目目标仍未完成。

- [导出来源与文件SHA](export_result.json)
- [真实调度检查](scheduler_result.json)
- [303条anchor摘要，不含真实框](anchors_summary.json)
- [逐序列数量及方向](per_sequence_summary.json)
- [实际303框的精度变化](actual_initialization_precision.json)
- [导出程序](source/export_initializations.py)
- [CPU调度记录器](source/check_scheduler.py)
