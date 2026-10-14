# M124 当前实验状态

北京时间2026-10-14 21:49实际观察：第8轮完整Train152与DepthTrack Test50/CDTB80均完成，训练和四项评价回执均为0。两F相对第7轮再次同时严格下降，连续双下降计数2，21:22按原协议正常早停；完整8轮训练/8轮双OPE，best仍第6轮。正式第8轮指标见主文档§5.533，第6轮best见§5.527，不重复成绩表。best的CDTB R/F仍未达标，整体目标未完成。

同一第6轮best的完整RGBD/VOT127条、1765个anchor验证已于21:22:47启动。21:55确认best.pt与fit/final.pt实际SHA均为4d5c7e411a047fa996835647d1efa3bb94f4184fdc3a592380523be8c04156eb，使用与既有OPE相同bundle/bank/binding。四分片两波使用GPU0/1；第一波原PID650915/650916实际存活，约32分钟各26/441份输出三件套代理，第二波尚未启动，最终EAO/ACC/ROB未生成。

已核对实际4份tracker配置、wrapper、driver/runtime及原模型进程657613/657771：M124入口严格加载第6轮A_B/C状态。manifest的旧M39/no-text文字是继承描述，原封存文件与运行源码保持；实际架构依据冻结配置/代码，详见§5.533。没有侵入式live tensor审查或最终VOT认证，没有新增NN、GPU/温度/功率查询或训练重启。

完整VOT粗估10月15日14:00—18:00结束，属于估计。下一次观察不早于10月15日06:30北京时间。原timer26960于21:45到点已消费；唯一新Hidden本地timer54312、parent34464、birth22:16:03.7893240，启动时身份已核验。只有独立本地计时器，无native wrapper/functions wait cell，不执行SSH/NN。到点读取原timer终态，再执行M124_best_VOT_wave_compact_observer_20261015.py一次；期间不查询NN/GPU或反复确认等待句柄。

原完整Train152/no-dev、人工文字、seed2027、A+B+C联合冻结底座、相邻双F严格下降连续两轮早停、两F等权best/较早tie及最终同best VOT127/1765协议保持。外部Test/CDTB参与选择、首轮37条/RNG交接与监督边界保留。先完成全部VOT/收集和同权重九項核验，未达标则等suite终态/GPU释放后开展既有同状态诊断及必要模型优化。Goal ACTIVE_UNMET。
