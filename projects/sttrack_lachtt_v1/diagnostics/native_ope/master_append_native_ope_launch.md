

### 5.61 同一原生STTrack的DepthTrack Test/CDTB完整参照实际启动（2026-09-06 01:46 CST）

§5.59准备的OPE入口通过实际Train契约：chair01和cube04各121输出帧（含初始化）、共240跟踪步、242输出帧、3次模板更新；独立原生bbox/score误差均0，六位小数序列化检查通过，exit0、耗时15.80秒。contract_receipt SHA256 `ba0df7abeb058fc2d35919598fcee370b2d7ec027519da6b6025fc020b32e340`。训练步数0，后续GT不进入推理。

2026-09-06 01:46:26 CST，GPU0启动两个独立数据集控制器；01:47:13核实控制器及Python子进程均存活且命令正确：

| 任务 | 完整序列 / 输出帧 | controller / Python | 状态 |
| --- | ---: | --- | --- |
| 原生DepthTrack Test OPE | 50 / 76,373 | 159777 / 159780 | 已启动，等待完整跟踪及自动分析 |
| 原生CDTB OPE | 80 / 101,956 | 159779 / 159781 | 已启动，等待完整跟踪及自动分析 |
| 原生VOT full127 | 127 / 1,765 anchors | 原有GPU1四worker | 原任务继续，不重复启动 |

两项OPE使用与M39/full127相同STTrack_Vot22权重、YAML、固定query窗口、两个模板、默认50步confidence>0.75更新，RGB-D读取与实际VOT wrapper/bridge一致。原生base SHA仍为`cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98`；新入口SHA `9947aa8dec69267e3bbd0d51b20dcfad5a2bbb9e4e0cfe23a9b8b868d6bddbf7`。语言和学习关联头关闭。VOT multi-start与OPE单起点的协议区别保留。

各控制器跟踪exit0并封存后，自动用原评测器`depthtrack_pr.py`计算，六位小数轨迹、初始化confidence=1、原阈值网格和PR宏平均不改。执行根`/root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906`。execution_binding和launch_observation保存spec、契约、实现审计、环境和实际进程证据。初估GPU0合计约2–2.5小时，接近预计结束再查，后续240秒间隔，长序列及资源共享会影响时间。

本项补齐原生同权重参照，不代表M52/M53通过晋升，也不能替代新模块DepthTrack训练。当前尚无这两个新完整参照或full127的完成指标；此前SRTrack与SUTrack数字不重复混列成当前模型结果。
