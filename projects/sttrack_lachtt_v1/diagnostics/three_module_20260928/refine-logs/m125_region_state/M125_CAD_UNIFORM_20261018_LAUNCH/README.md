# M125统一CAD U0/U1/U2实际启动

原R0 fixed6完整训练和三集评测已结束并通过结果完整性审查后，已按已审R2/R3准入入口启动统一CAD推理对照。此次没有新训练、没有新checkpoint，固定R0 final SHA `cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f`，统一profile为U0/U1/U2，根目录为`/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016`，监督器PID 240031。

准入远端验证了原R0六轮和DT/CDTB/VOT全套15 stage退出0、同final绑定；资源查询只使用`index,memory.used,memory.total`，GPU0/1均1MiB used/24576MiB total。没有温度或功率查询。启动入口exit0、自动重试0；启动时尚未有real64或正式指标。

执行顺序仍由coordinator固定：U0引用R0原完整结果并做64步合法前缀检查；U1、U2分别完整跑DT50/CDTB80/VOT127，然后封存官方指标和统一三行结果。不会按数据集切换profile，也不会提前启动ROI。下一次进度读取节点为北京时间10月19日13:30；期间不查日志/进程/GPU/句柄，不重启监督器。
