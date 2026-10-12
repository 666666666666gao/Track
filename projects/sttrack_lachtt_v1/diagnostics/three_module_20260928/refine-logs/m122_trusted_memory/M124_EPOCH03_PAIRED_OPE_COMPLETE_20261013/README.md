# M124 第3轮两项完整OPE，原控制器启动第4轮

最后实际服务器检查：2026-10-13 01:15:18.593590 北京时间。原快照 SHA-256 84b3c736107aa61f6bdeb06039fceaad11652d73d897e6405aa3a1d994e06a2f；进度/指标/阶段回执均从此快照提取，没有为本文追加GPU或NN查询。

DepthTrack Test50/76373帧 P/R/F=66.14919927694646/63.33454583382472/64.71128078497416%；CDTB80/101956帧=71.85629404846993/72.25609687654313/72.05564088735755%。两份metrics complete、两项tracking exit0；同final 3b9e613b4af8c3a2f009200ae8ecb6facf94b56b2033647ef344f08be07175af、bundle e53ada72e25baad36e745d877fb0b7bc5edb1709d7bbba3d9204fef3e4aa68fd。未把metrics声明的receipt_sha256当作本目录独立取回的原始预测回执。

本轮平均F68.38346083616585%，best仍E1平均68.5415903449273%。相对E2，DepthTrack F下降0.28162429484034135pp，CDTB F上升0.601653045993416pp，双下降false/streak0；当前控制器记录第4轮于01:14:12.700763从E3/latest启动。第4轮保存序列日志仅一个换行，没有观察到完成序列/优化数量；不能从GPU瞬采推断精确进度。没有M124 VOT或suite终态。

独立上下文 experiment-audit 原始R1：WARN、限定捕获/源码范围0阻塞，same-family/provisional，requested Astra/max与实际运行时UNATTESTED分列。验证6份历史指标捕获文本哈希/选择/ledger/18份LF_EXACT；尚未独立取回plan/bundle/预测回执/checkpoint/GT/逐帧预测/analyze回执，故不是全九项最终审计。原报告字节不改，私有调用trace不提交。原CRLF物化目录16份仅换行变化，保留在私有取证区；此目录capture用短文件名保存LF_EXACT及原manifest，path_hash_map.json对应原路径与SHA，逐字节一致。

唯一代码修改为本地CPU summarize_m124_hourly_observation.py 在JSONL读取中过滤空白行，解决真实第4轮空日志的JSONDecodeError；非空坏JSON仍报错。真实上一快照旧/新/已存ledger全等、当前快照ledger复现的回归检查通过。没有更改隔离运行根中的44项锁定训练源码、优化器、损失、学习率、模板或搜索规则。

原native69508/PID23476/functions259真实exit0并消费12次原生join；唯一下一等待native28385/PID40856/functions299，每次300秒，只等待，下一GPU/NN/进度检查不早于02:15:18.593590。额外NN前向0，无手动重启。Full152/seed2027/最多20轮/逐轮Test-CDTB选择及人工多帧文本/继承前缀与RNG限制保持披露。候选回放待整套M124及best VOT终态、GPU释放后执行。Goal ACTIVE_UNMET。
