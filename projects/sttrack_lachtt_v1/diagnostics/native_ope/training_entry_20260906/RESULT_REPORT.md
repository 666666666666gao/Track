# 原生STTrack底座训练入口准备核对

2026-09-06。这是后续RGB-D底座微调的入口核对，未执行训练或改动运行源码。当前M54使用独立的采集、读取头训练和递归实现；本报告不把原入口问题归因为当前STTrack评测下降或M54失效。

## 当前服务器的实际发现

| 项目 | 已核实的行为 | 对底座训练的要求 |
| --- | --- | --- |
| 数据与工作路径 | 原生训练local.py的DepthTrack、workspace、tensorboard、pretrained路径仍指向作者环境，观察时均不存在；实际Train在/root/autodl-tmp/depthtrack/train | 在独立训练运行中绑定真实数据及新输出目录 |
| 默认预训练文件 | 配置SOT_Pretrained_256.pth.tar在当前pretrained目录不存在 | 不能直接启动默认原命令 |
| 改成当前STTrack文件名 | build_sttrack对含STTrack的文件名跳过backbone预加载；完整模型加载又只由SOT字符串分支触发。原train_script在建优化器前没有显式加载当前完整权重 | 微调必须显式加载绑定的完整net，严格核对参数，不能只改PRETRAIN_FILE字符串 |
| 验证周期 | 配置总计15 epochs、验证间隔50；LTRTrainer仅在epoch对间隔取模为0时运行相应loader | 原15 epochs不会运行该验证loader，后续须明确定义实际验证周期 |
| 失败退出 | 原入口fail_safe=True；BaseTrainer只有num_tries=1，捕获训练异常后可打印Restarting和Finished training并正常返回，没有实际重试 | 后续训练须让异常传播，并核对真实epoch/optimizer步数及最终权重，不能仅看退出码或末行文字 |
| 官方验证划分 | 146/6两表互斥，但官方6条中bag04、bottle03、toy03已用于当前M54拟合，flower03已用于开发 | 不能把该官方6条称为本项目全新未见验证集，也不能无说明替换现有划分 |

这些是源代码和路径观察可直接支持的结论，不是已观察到的某次底座训练崩溃。当前原生推理在build_sttrack之后显式严格加载params.checkpoint；M54也从冻结spec传入同一基础权重，并另外训练读取头，因此上述原训练入口的加载分支不否定当前已封存结果。

## 接续方式

先完成当前M54的固定训练和完整开发递归，由实际结果决定是否晋升。若进入底座微调，使用独立目录与冻结方案，显式绑定基础checkpoint和项目Train划分；先完成一个真实训练batch的前向、反向、参数变化及严格回读，再按预定预算运行。记录实际样本条目、搜索帧、全局batch、优化器步数及验证周期，并以同一最终权重/运行策略验证三个数据集。当前没有启动或命名一个新的底座训练实验，也没有把准备核对记为模型收益。

## 证据与复核范围

[observation.json](observation.json)绑定9份原始源码/配置/划分以及05:26的路径存在性；[trainer_source_addendum.json](trainer_source_addendum.json)绑定后补的两份trainer源码。原始文件保存在本机私有native_training_entry_20260906目录及服务器，公开记录提供逐文件SHA256。

同族GPT-5.5 xhigh的Type-A审查独立核对了捕获源码。首次审查保留了缺少trainer源码的限制，补充审查在取得两份trainer源码后确认验证周期及异常返回行为；两份原文均逐字节保留。审查没有自行访问服务器、启动训练或测试GPU，不属于跨模型家族复核。这里的PASS表示准备问题得到证据支持，不表示原生训练入口已经修复或可以直接开跑。
