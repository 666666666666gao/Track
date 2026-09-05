# 原生STTrack的DepthTrack Test/CDTB完整参照已启动

2026-09-06 01:46:26 CST启动。此项补齐同一个原生STTrack权重的完整参照，与M52/M53新模块的晋升分开。启动时尚无新的DepthTrack、CDTB或full127完成指标。

## 启动前契约

两条DepthTrack Train拟合序列chair01、cube04各输出121帧（包括初始化），共240次跟踪、242个输出框、3次原生模板更新。新OPE入口与独立原生STTrack逐帧bbox、score误差均为0；保存和回读六位小数输出的误差在固定界限内。后续GT不用于推理，优化器步数为0。契约退出码0，耗时15.80秒。原始证据见[contract_receipt.json](contract_receipt.json)。

## 同一模型及协议边界

新入口直接调用原生STTrack：相同`STTrack_Vot22.pth.tar`、相同YAML、固定query窗口、两个模板、factor-4/256搜索和factor-2/128模板，50步且confidence>0.75时执行原生更新，语言与关联头关闭。RGB/Depth的`rgbcolormap`及`depth_clip=True`处理与实际VOT wrapper/bridge一致。VOT采用multi-start，DepthTrack/CDTB采用各自单起点OPE，不能混淆协议。

| 任务 | 完整范围 | 启动资源 | 启动观察 |
| --- | --- | --- | --- |
| DepthTrack Test | 50序列、76,373帧 | GPU0，一个进程 | controller 159777 / Python 159780 |
| CDTB | 80序列、101,956帧 | GPU0，一个进程 | controller 159779 / Python 159781 |
| 原生full127 VOT参照 | 127序列、1,765 anchors | GPU1，原有四worker任务继续 | 不重复启动 |

两个OPE控制器在01:47:13 CST核实存活且子进程命令匹配。跟踪完成封存后，各自自动调用既有PR/F-score评测器；沿用六位小数轨迹、初始化confidence=1、原有阈值网格和逐序列PR宏平均。不在预测阶段读后续GT，不改指标定义。

初步按契约速度估算两任务合计约2–2.5小时，长序列、I/O及GPU共享可能改变时间。按接近预计结束的时刻检查，然后以240秒间隔查看终态，避免短间隔重复轮询。

## 证据绑定

- base权重SHA256：`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`。
- spec：`e623ef63de89da423fc12b877f58d56219f88ef9f1232f20de9e42fb6c8665e1`；inputs：`61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601`。
- 运行入口：`9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7`；原有评测器：`05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc`。
- 实际VOT wrapper：`3ef2704c9e3d987383f2184ba5e45789278eee7d4b6de6470e647d883c10a608`；bridge：`230acf10f378a6babfacf9979ea07a1ce89c34952cc0c6b9568376249e265316`。

完整绑定和进程观察见[execution_binding.json](execution_binding.json)、[launch_observation.json](launch_observation.json)。之前的[实现审计](EXPERIMENT_AUDIT.md)审的是准备态代码；本次契约提供新增执行证据，并不改写此前审计的时间范围。当前原生参照完成也不能替代后续新模块的DepthTrack训练和三数据集验证。
