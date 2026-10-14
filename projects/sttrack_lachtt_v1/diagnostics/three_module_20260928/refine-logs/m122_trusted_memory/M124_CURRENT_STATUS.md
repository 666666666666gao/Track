# M124 当前实验状态

北京时间2026-10-14 13:17:25.183813实际只读观察：第7轮完整Train152及DepthTrack Test50/CDTB80双OPE均已完成，训练及四项track/analyze退出回执为0。正式第7轮指标见主文档§5.532：两F均较第6轮下降，连续双下降计数1、未早停，训练后best仍第6轮。第7轮DepthTrack R、CDTB R/F未达标；同best全VOT尚未开始，整套suite未终态。

第8轮实际于12:45:01.438119启动，已完整保存9/152条序列，最后toydog_indoor。本轮16157次调用、A+B/C各507次更新；累计1554771次调用及各48156次更新。第8轮final/完整指标未产生。两个F若相对第7轮再度同时严格下降将触发第二次双下降早停；任一不降则清零。best按两F等权均值和较早tie选择，保持原协议。

第7轮两个metric/plan/prediction receipt的final、bundle、评估bank、binding、50/80条和76373/101956帧相互匹配，训练完成报告和日志SHA匹配10:46原快照。训练bank和评估bank分别按既有角色核验。该结果是实际元数据及算术核查，没有独立原始PR/IoU/权重张量重放或最终VOT认证。

13:00完整快照首次读取退出255，第二次传输在360448字节处截断；原失败及部分包保留。改用仅含当前指标/决策/轮次进度的小型快照后，native59160/cfc26f退出0，22985字节压缩解码81919字节，SHA-256 66f21bcf6d10c6e5a7cdec731ff3b808ea68b3d503ad62a0a5b05cf95182e122。没有变更SSH代理、重启训练或新增NN。GPU查询0。

第8轮完整结果预计21:30—22:30，属于估计。下一观察不早于2026-10-14 21:45北京时间；原timer29756于13:00到点已消费，唯一新Hidden本地timer26960、parent37832、birth13:31:18.3232490，启动时身份已核验。独立本地计时器无SSH/NN，无native wrapper/functions wait cell。到点后读取原timer终态并执行M124_E8_paired_compact_observer_20261014.py一次；期间不查询NN/GPU或重复确认等待句柄。

完整Train152/no-dev、人工类别/属性、seed2027、A+B+C联合与冻结底座、max20、逐轮双OPE、相邻双F严格下降连续两轮早停、等权F best/较早tie及最终同best VOT127/1765保持。外部Test/CDTB参与选择、首轮37条/RNG交接、detach/no-BPTT和C当前IoU边界保留。整套suite及同best全VOT终态/GPU释放后再执行必要候选诊断，完整目标ACTIVE_UNMET。
