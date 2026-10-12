# M124逐轮结果表

最新实际观察2026-10-12 14:26:11。第2轮已真实完成全Train152，训练阶段13:58:41.861472退出0；本轮219802次调用、6807次A_B及C各自优化，累计439604次调用、13614次优化，304个完整序列运行。第2轮final SHA1d44ec68e788d31ebeddd528edc8de2afdf960cb67aa2392a952e4cb9048ab41。

第2轮全量OPE进行中，截至14:26保存Test9/50、CDTB14/80条bbox文件。两项完整计分尚未产生，P/R/F留空，best仍第1轮、streak0/earlyfalse；不按partial预测估计PR。两卡分别推理，GPU0/1瞬采4122MiB/80%、4122MiB/64%。

首轮完整两项OPE约2小时15分钟仅作参考，第二轮约16:15左右；初期按序列数外推约16:35，序列长度不同，因此均非结束保证。第1轮指标及四份metric/stage原始字节仍一致，见§5.493，不重复为新结果。M124没有完整VOT指标。

本地等待中断与服务器训练分开：原61182/PID48256/functions1746在13:54实际句柄unknown/PID缺失，原exit未知、terminal/joins文件不存在，未推断远端失败、未重启训练。保留loss JSON与原owner。仅补剩余时间的本地计时36810/PID20356，原14:24:38.140404节点不变；替代计时在14:24:38.141166退出0、5次同句柄join、functions8终态已消费。原计时不能伪写exit0或消费成功。记录脚本初版Unicode转义语法错误93986d在执行前失败，源文件归档；只修本地路径字符串后63f5bd成功，没有重做NN观察。

唯一小时观察native65689/1bf3dc退出0，14:26:11.156796，距13:24观察3693.016392秒；1次GPU/NN进度查询、0额外NN前向。下一原计时71742/PID4612，parent5176、birth14:32:53.495895、actual start14:32:53.564522、deadline15:26:11.156796，functions21只做300秒同句柄本地join；下一完整日志观察器仅准备未执行。

完整Train152/no-dev、人工文字、冻结底座、seed2027、max20、每轮Test50+CDTB80、相邻双F连续两轮严格下降早停、等权F均值best且持平较早轮、最终同一best全VOT127/1765不变。第1轮旧prefix37条与legacy_handoff_RNG_exact=false、detach/no-BPTT、C当前IoU目标继续披露。第2轮frozen_before_after_exact与final roundtrip是训练回执字段，不是本次独立张量/NN复核。最终full-nine仍未完成，Goal ACTIVE_UNMET。
