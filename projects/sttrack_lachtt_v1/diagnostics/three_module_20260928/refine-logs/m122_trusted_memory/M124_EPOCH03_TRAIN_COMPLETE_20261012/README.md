# M124第3轮完整Train152训练结束，逐轮OPE进行中

实际观察：2026-10-12 23:09:36.736564北京时间；原始快照SHA-256 d40a11c083392b81c193d90e3185dddee7210cf762b0be38933b1b4f30afa614。原始SSH观察native69347/8886e1实际exit0，距前次3,646.408654秒；没有额外NN forward。训练于22:57:21.345767实际exit0，152条/219,802 calls/6,807次A+B和C优化；三轮累计659,406 calls/20,421次各组优化。final SHA-256 3b9e613b4af8c3a2f009200ae8ecb6facf94b56b2033647ef344f08be07175af。

训练回执报告冻结前后和保存回读exact；本地核对捕获JSON的E3初始可训练状态与E2最终状态一致、bank/labels/spec/source指纹一致。train_complete_receipt_capture.json保存完整原始捕获字段。这不是独立读取原始checkpoint张量的最终审核，E1继承前缀/RNG限制仍在。

本次只看到DepthTrack Test 3份、CDTB 5份bbox结果文件，没有E3 metrics，也没有M124完整VOT。不能用文件数估计P/R/F。best仍E1，早停streak0；完整前两轮指标见主交接§5.493/5.506，不重复填写。

已有E1/E2完成态tracking receipts的CPU耗时分别显示并行两项需要2.2374/2.2904小时；若本轮类似，粗估10月13日1:10—1:30产生两项指标，实际以正式回执为准。GPU0/1瞬时4122MiB/81%、4122MiB/76%，非整段平均。

原native55478/PID18468/functions215真实到时exit0，保存8次此functions owner的原生join，之前e2b017单独记录，不混算join总数。新的唯一native27359/PID39380/functions233只等待；下一NN/进度检查不早于10月13日00:09:36.736564。无训练代码、损失、搜索、模板策略改变，无手动重启或候选NN回放。Goal ACTIVE_UNMET。
