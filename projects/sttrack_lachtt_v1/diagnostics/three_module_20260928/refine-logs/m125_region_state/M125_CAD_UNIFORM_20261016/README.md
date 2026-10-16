# M125统一CAD模板推理对照：源码准备

当前有效模型协议见[统一协议](../UNIFIED_MODEL_PROTOCOL_20261016.md)。只在DepthTrack Train152训练；每候选同checkpoint和同一主推理规则完整三集九项。原R0固定六pass/final6不改，完成整套后才使用该同一final统一比较U0/U1/U2。不能按数据集或序列选权重/profile拼行。

当前科研源码是[source_R2](source_R2/PROTOCOL.md)，单次启动操作是[admission_R2](admission_R2/invoke_M125_CAD_reviewed_suite.py)。[R3审查](review_R3/EXPERIMENT_AUDIT.md)为PASS_SOURCE_ONLY、0阻断；same-family/provisional，请求Astra/max，实际模型/effort路由UNATTESTED。它不等于实际部署、real64、完整评测或科研收益。

| 对照 | 更新检查 | 写入资格 | C |
|---|---:|---|---|
| U0 | 每50帧 | ABC所选位置原生Hann响应>0.75 | >0.5 |
| U1 | 每5帧 | 同上 | >0.5 |
| U2 | 每5帧 | 同位置响应>0.65 | >0.5 |

保留首帧模板，仅替换动态槽；框/query/特征/报告分数走原ABC路径。U0整行引用原R0原入口全套结果，新U0只另做64步合法前缀检查；U1/U2各自重新完整推理DT50/CDTB80/VOT127。U3/U4为可选未排队源码。

原稿[source_R1](source_R1/PROTOCOL.md)及[原审查FAIL](review_R1/EXPERIMENT_AUDIT.md)保留：准备脚本不存在的args.training、收集事件变量覆盖模型名。科研R2仅修这两行。[第二轮审查](review_R2/EXPERIMENT_AUDIT.md)确认科研路径通过，但首次启动入口会将报告CRLF转LF后比原始SHA，因此操作FAIL。当前操作仅改报告原始字节读取及绑定独立R3报告，原入口/失败证据不变。

纯策略5416检查、600模拟步/2700断言和完整66布局下12入口CPU导入都有实际回执；它们没有打开真实评测图像或产生CAD指标。早期12入口导入回执的runtime为较早版本，最终原稿66回执及R2独立66回执分别对应各自冻结版本，不混称同一版本。完整prepare/collect的本地审查夹具使用模拟回执，不作数据集结果。

首次正式调用仍须等原R0全部6pass及三集正式退出/收集，核验同final、66源码、资产SHA、磁盘与GPU memory准入，再执行三个合法64步接口检查和新U1/U2完整三集。单次启动入口在本地/远端均有Oct17 12:45BJT下界；该日期不是GPU启动承诺。当前没有执行该入口，没有新CAD真实track或性能。原R0唯一既有等待和53份运行源码未改，不提前读取其进度/句柄或重复启动observer。

模板策略借鉴[CADTrack官方固定提交](https://github.com/IdolLab/CADTrack/blob/93d0a67d3eb328617751b18510bdab072705c1a0/lib/test/tracker/cadtrack.py)，不引入其网络/训练数据/预训练，不作为原创模块。官方原始下载保持私有，只发布URL/SHA来源回执。用户提到的ZIP未取得，不声称测试过该ZIP。

撤回原SOURCE_SPEC中的端到端一次hash效率表述：继承的R0→M122计划校验仍逐调用hash基线final/native/CLIP。原冻结spec保留供溯源，R2 spec和审查明确实际范围。

原始路径和逐文件SHA见[ARCHIVE_MAP.json](ARCHIVE_MAP.json)；归档字节不改，完整私有审查trace另行保留。主文§5.549为当前状态，E6仍5/9阶段目标，Goal ACTIVE_UNMET。
