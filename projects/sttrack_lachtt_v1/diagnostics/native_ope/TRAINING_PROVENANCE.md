# 原生STTrack训练来源与同权重比较范围

2026-09-06 04:40:56 CST只读核实当前服务器、上游固定提交和本机基础权重。此项没有读取训练图像/GT、执行优化器步骤或改变M54。12个所查训练/输入/参数源文件均与上游`283cd6dd45536636490db8bca1c63c4647be799b`逐字节相同，完整hash见[原始观察](training_provenance_20260906.json)。

## 已证实的范围

| 项目 | 当前证据 | 解释范围 |
| --- | --- | --- |
| 官方RGB-D训练数据 | 配置DepthTrack_train；固定源清单146条拟合＋6条验证，无交集；本机152目录与并集一致 | 当前拥有这些Train目录，不是本轮已训练全部152条 |
| 官方输入/预算 | 两模板128，四搜索256，每epoch 15,000个采样条目，batch8，15 epochs；第10 epoch降学习率 | 15,000条目×4搜索为60,000个搜索帧呈现，不应直接把“条目”与论文“sample pairs”混为同一种计数 |
| 官方优化组 | decoder/BSI/TSG/MambaFusion的可训练参数使用1e-4，其余可训练参数1e-5；AdamW，weight decay1e-4 | 描述优化器分组，不从配置推断下载权重实际完成了哪些步骤 |
| 输入规范 | 训练与运行均使用6通道均值/方差；训练DepthTrack loader与RGB-D运行入口均指定rgbcolormap、depth_clip=True | 本次没有发现这些开关或常量不一致；没有进行新的逐像素输入契约 |
| 官方DepthTrack入口 | run_id=15，经参数调用链选择STTrack_ep0015.pth.tar | 默认文件路径与VOT入口不同 |
| 官方VOT22入口 | run_id=16，经参数调用链选择STTrack_ep0016.pth.tar | 不能只凭方法名称把作者两数据集结果视作同一文件 |
| 当前下载底座 | STTrack_Vot22.pth.tar，532,407,510字节，顶层只有net，无epoch、optimizer、训练参数或日志 | 文件自身不足以证明具体训练履历，也不能由名称证明它就是上述ep0016 |

当前下载底座SHA256仍为`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`。此次CPU读取仅检视容器结构，未修改权重。原始观察SHA256为`c99d37618e3cd94e0b7d646db0fb77e3c952cb530c7ceb958351951ac6ecbfdb`。

公开入口的文件差异可由[DepthTrack wrapper](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/lib/test/vot/sttrack_depthTrack.py)、[VOT wrapper](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/lib/test/vot/sttrack_vot22.py)、[run_id到epoch传递](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/lib/test/evaluation/tracker.py)及[checkpoint路径生成](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/lib/test/parameter/sttrack.py)复核。它没有证明本次DepthTrack/CDTB下降是文件选择造成的；该因果关系尚未测试。

作者[论文表2及训练说明](https://arxiv.org/html/2412.15691v1#S4)应与[固定RGB-D配置](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/experiments/sttrack/deep_rgbd_256.yaml)分别引用。论文中DepthTrack的F-score为63.3，VOT的EAO为77.6；[固定提交的README性能汇总表](https://github.com/NJU-PCALab/STTrack/blob/283cd6dd45536636490db8bca1c63c4647be799b/README.md)将这两个数据集列的数值对调，不应引用该表声称DepthTrack F-score为77.6。论文结果不是本机复现结果，也没有覆盖CDTB。

## 对正在执行的M54和后续训练的影响

M54保持既定底座、63条拟合/22条Train开发划分、20 epochs读取头训练与完整递归门，不中途切换官方文件、扩大数据或更改标签。此次核对不改当前已完成的两个原生完整指标，也不改变正在运行的VOT任务。

后续验收继续绑定实际训练生成的读取头/网络权重、同一基础权重及同一运行策略，三个完整数据集都实际测量。作者公开配方可作为后续RGB-D底座微调的参照；准备时应先明确采样条目、搜索帧、全局batch与优化器步数，并使用项目自己的Train划分和真实递归验证。不能把一份外部模型文件的名称当作本项目重新训练完成的凭证。

这次获得的是训练来源和可比范围证据，没有新增模型收益或完整指标；M54结果仍由其已启动的训练与递归流程回答。
