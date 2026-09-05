# M51训练与递归启动报告

2026-09-05 19:36:07 CST启动。训练和运行接口检查已完成；GPU0上的22条完整递归仍运行，不能填写最终递归指标。

唯一变量为4维候选几何改成相对前一选择框的中心偏移及log宽高比。视觉局部token、448,739参数关联头、M45标签和损失、两模板默认更新、搜索区域、数据与优化均保持配对设置。主干是原官方STTrack冻结权重，新增训练权重是关联头，不能称为全主干重新训练。

DepthTrack Train拟合63序列/1,511事件，开发22序列/590事件；fresh seed2026，20epochs、960steps，固定末轮，训练计时32.88秒（不包括数据读取）。M45与本次全部2,101事件的当前/上一目标标签、初始参数、样本顺序精确一致。开发集反复使用，静态缓存结果仅作诊断。

| 静态缓存指标 | 拟合 | 开发 |
|---|---:|---:|
| 当前候选选择变化 | 106 | 17 |
| 严重错误→正确 | 29 | 6 |
| 正确→严重错误 | 1 | 2 |
| IoU≥0.5候选数 | 860 | 275 |
| 相对原生IoU≥0.5候选净增 | 98 | 7 |

GT不可用事件在沿用的训练静态统计中被赋0；这些静态均值不能与M49仅有效GT均值或完整递归均值混比。缓存中的救回不等于连续轨迹救回。

CPU坐标审计覆盖85份特征文件，数值有限，平移缩放不变性误差小于1e-10；包括非零previous-choice索引。训练head重新加载logits精确一致，实际tracker封装与训练forward在previous-choice0至9的logits/affinity精确一致。初始化head在chair01拟合序列120帧的框、分数、模板及query与原生完全相同，包含1次默认模板更新。

- 协议SHA：13ef61e8ca7ab07c401f503f201e6f29767763f97e62b6e26ee63f2d33675851
- 新head SHA：7eb1cc4e9d1e4c1c63e18429febf6999630e644763b4496686751dfee29fbf85，1,811,499bytes。
- 服务器权重：/root/autodl-tmp/sttrack_m51_relative_geometry_v1_20260905/geometry_final.pth
- 原主干SHA：cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98。
- controller45877，screen sttrack_m51_relative；19:38观察递归shard0实际进程46661。

训练启动前改正了source SHA循环变量覆盖样本顺序digest的报告代码问题，原协议与新协议SHA在pre_execution_amendment.json保留；没有训练后改优化或选权重。运行脚本顺序执行training、runtime_contract、recursive_s0、recursive_s1、analysis，任一阶段失败保存退出码并停止。不自动启动公开测试。

待全部33,130帧封存后，按原生STTrack门判断；M50额外模板更新和文本均未启用。GPU1原生full127继续运行，它是独立基线。
