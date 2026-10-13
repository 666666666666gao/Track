# M124 当前实验状态

北京时间2026-10-14 06:36:07.290865实际只读观察：第7轮已保存完整63/152条训练序列，最后toy07_indoor_320（零基62），较05:29增加29条。本轮87225次跟踪调用、A+B/C各2716次优化；累计1406037次调用/各43558次优化。序列计数只含完整保存记录，不含正在处理的片段；第7轮final、训练终态及P/R/F尚未捕获。

已完成6轮完整Train152和6轮完整Test50/CDTB80双OPE，训练后best仍第6轮、两F等权均值69.92320836681976、双下降streak0、不早停。正式第6轮指标已见主文档§5.527，CDTB诊断已见§5.528，不重复表格。DepthTrack三项及CDTB P达目标，CDTB R/F仍未达到；M124所选best全VOT尚未开始，整套控制器未终态。预计本轮训练10:30—11:30、两项完整OPE13:00—14:30，均为估计。

本次NN观察间隔4002.314119秒，native34144/ba7675/e935c1/37a42e退出0。324739字节压缩完整解码2199552字节，原JSON SHA-256 e843d39515f7a5d3ba6b9835ddde567e4e77764806fea0d26dd64a1bb6db59e4；压缩SHA-256 389aecbdb2b32bb62a7571d0140a135303219aec5b4042c187d321c8d51b4200。85份捕获只有第7轮sequence_log改变，其余84份字符串一致，controller argv不变。GPU0/1瞬采1890MiB/5%、4118MiB/49%，不是平均利用率。原始捕获、7行ledger/CSV、实际native回执与后续入口保存在本机.aris/m122_full_causal_20261007的M124_fiftieth系列文件中。

原timer50于06:29:24.977207到点，wrapper72938/functions459实际12次join、6dce8f退出0，完整join已保留，不冒充独立timer原生退出码。下一唯一隐藏timer14384、parent2040、birth06:46:35.6485250，源码SHA-256 8e78336a6a21d6a78149325e7ead3d3357c103a2a927fc571756981cdd02a9a8。使用独立计时器，未建立持续native等待wrapper或functions等待cell；到07:36:07.290865节点再读取原计时器终态并执行一次准备好的fiftyfirst观察入口。期间不查询NN/GPU进度。

本次仅更新当前进度与tracker，主交接文档§5.528内容不变。完整Train152不划开发集、人工核验类别与稳定属性、seed2027、A+B+C联合及STTrack/CLIP冻结、最多20轮/逐轮双OPE/连续两轮相邻双F严格下降早停/两F等权均值best及较早tie/最终同best全VOT127与1765anchor协议保持。Test/CDTB选择checkpoint、多帧人工辅助标签、首轮37条/RNG交接、detach/no-BPTT和C当前IoU监督边界仍保留。未改变运行源码、模型、优化器、搜索或模板，未重启或插入NN回放；全suite与selected-best VOT终态/GPU释放后再进行候选诊断。Goal ACTIVE_UNMET。
