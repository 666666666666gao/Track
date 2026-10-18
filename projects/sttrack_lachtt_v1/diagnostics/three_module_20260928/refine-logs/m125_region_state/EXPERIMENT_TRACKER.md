# 区域实例语义与状态观察执行表

日期：2026-10-15T01:14:05.461700+08:00。详细协议见[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。本表的新编号统一M125前缀，不覆盖旧FutureC/R0—R5历史。

| Run ID | 目的 | 数据与预算 | 决定性指标 | 优先级 | 实际状态 |
|---|---|---|---|---|---|
| M124-E6-VOT | 封存现有同权重九项 | 原full127/1765 | EAO/ACC/ROB与正式收集 | MUST | RUNNING_AT_LAST_OBSERVATION：10月14日22:01；未提前查询新进度 |
| M125-I | 源码与文献接口核对 | 本地8源码/3官方来源 | state/query/C/ROI/监督边界 | MUST | COMPLETE_SOURCE_INSPECTION_ONLY：无NN检查或新模型 |
| M125-D | 候选与失锁短前置 | 原suite终态后独立回放 | 覆盖/排序/精修/可达性 | MUST | NOT_RUN |
| M125-M0 | 最小真实接口检查 | 新代码冻结后 | 梯度/冻结/重载/坐标/256候选/无GT动作 | MUST | NOT_IMPLEMENTED |
| M125-R0 | 匹配固定预算基线 | 全Train152×6/同final完整三集 | 九项主表/运行成本 | MUST | NOT_IMPLEMENTED_NOT_RUN |
| M125-ROI | 普通ROI必要性 | 同R0初始化/6pass/三集 | 相对R0完整净收益 | MUST | NOT_IMPLEMENTED_NOT_RUN |
| M125-R1 | 条件区域核验 | 同共同起点/6pass/三集 | 相对ROI/R0的净收益 | MUST | NOT_IMPLEMENTED_NOT_RUN |
| M125-V | 同容量视觉控制 | 同共同起点/6pass/三集 | 文本内容独立增量 | MUST | NOT_IMPLEMENTED_NOT_RUN |
| M125-R2 | 下一状态/观察决策 | 同共同起点/6pass/三集 | 九项/恢复/接受拒绝/成本 | CONDITIONAL_MUST | NOT_IMPLEMENTED：动作接口先锁定 |
| M125-L | 最终文本完整干预 | 同冻结final五条件/完整三集 | 正确/泛化/冲突/同义/Empty | MUST | NOT_RUN |
| M125-S | 同预算简单搜索控制 | 最终同权重/完整三集 | 机制相对增加观察的增量 | MUST_FOR_C2 | NOT_IMPLEMENTED |
| M125-M | 新事件记忆价值 | 尚无新锁定协议 | 必须独立完整收益 | OPTIONAL | NOT_PLANNED_FOR_LAUNCH：保留旧FutureC负证据 |

固定seed2027；所有“待运行”不是启动证明。历史外部选模和人审输入边界保留，下一原VOT观察仍为北京时间10月15日06:30。

2026-10-15 06:30:43实际原VOT观察：分片0/1非空三件套代理391/441与441/441，0原进程仍在、1日志成功但原退出回执尚空；第二波2/3未launch，正式九项尚缺VOT。唯一后继39132计时器到08:00确认转换；新M125仅计划，无新增NN。

2026-10-15 08:00:18原VOT实际转换：分片0/1正式exit0，各441anchors，07:47:29收回；第二波2/3原PID760856/760857自动启动，非空输出代理10/441、12/442。仍无最终VOT指标。唯一后继44484计时器到18:00临近完成复查，无新NN/模型/配置修改。

2026-10-15原suite18:15:22全部exit0，E6 fullVOT127/1765/5295最终EAO/ACC/ROB75.756244/82.151385/92.127839，confirmed failures231；同权重九项5/9，目标未达。新鲜审核另存master§5.537，原independent_audit字段未改。M125-D仅新拆分源码/25状态面板和审核完成，NN尚未启动；M125区域主模型与全训练仍NOT_IMPLEMENTED/NOT_RUN。原计时器全部消费。

2026-10-15 19:05:09.588342 M125-D实际controller877266已启动，固定E6四段计划2894track/25Empty/25状态×256，2卡2波，NN预测与CPU计分拆分。首次唯一完成节点19:20:09.588342，未提前查询。状态RUNNING_CONTROLLER_LAUNCHED/完成数待实际结果；R0/ROI/R1/R2/V完整主实验尚未启动。

2026-10-15 M125-D第一次部署FAIL：两个子项19:05:10 import缺train_template_write_C/exit1、main未进入；不是完成2894调用。19:24两入口CPU import实际pass0；只补原M124 code路径。R2新目录controller879928于19:26:13单次启动、待19:41:13唯一节点；源码和正式E6九项不改，原失败回执保留。

2026-10-15 M125-D R2 COMPLETE（仅诊断）：全部4NN+4CPU0，2894track/25Empty/25states，四条完整bbox/score前缀六位小数一致；原suite不变。14valid/11invalid、13valid无>=0.5精修框仅为后验面板；覆盖与候选数组新鲜审核3798核对通过/WARN，官方IoU未复算。M125-R0入口和真实64检查已实现，53源码已隔离部署、九CPU导入0，尚无NN训练；R0完整训练和新增区域模型不能标COMPLETE。

M125-R0：SOURCE_READY（非RUNNING/COMPLETE），九份新入口+44冻结依赖已实际隔离部署、九CPU导入0，fresh source PASS_SOURCE_ONLY/0阻断。固定6完整pass/final6、seed2027、共同CurrentC/fresh optimizer+RNG、无逐轮外部选模。下一步唯一真实64接口检查，通过后全Train152×6+完整三集。新增区域模型未实现，未有新增正式指标。

M125-R0 RUNNING：真实64接口检查原exit0后，2026-10-15T21:31:23.509631+08:00从共同warm/freshAdam+seed单次启动原supervisor895667，固定6完整pass/final6后全三集。存储准入修订仅20GiB->2GiB，旧失败与审查保留；无科学代码/协议改动。唯一首次观察2026-10-15T21:41:23.509631+08:00，尚无新正式性能。

R0 VERIFIED_RUNNING 2026-10-15 22:07BJT：原supervisor895667 exactcmdline live，pass1完整10/152序列、18,516track、580AB/580C步，checkpoint5,498,004B；0pass完整完成，无新final/九项。唯一下一NN节点2026-10-17 12:45+08，估计训练39.55h；不做分钟轮询/中途外部选模。

M125 ROI/R1/V SOURCE_IMPLEMENTED_CPU_CHECKED_REVIEW_PENDING 2026-10-15：11源码，10组CPU合成断言/15次decoder前向，11入口实际导入exit0；真实track=0，无新性能。候选配置与冻结计划已封存，部署准入/真实64尚待实现；必须等原R0整个三集suite终态再运行新GPU工作。原R0下一NN观察2026-10-17 12:45+08，不变。

M125 ROI/R1/V SOURCE_REVIEW_PASS_EXECUTION_PENDING：新鲜源码审查325检查、75输入未改、0源码阻断，overall WARN/same-family/provisional/route UNATTESTED；部署操作另行草稿，尚未准入/实际真实64/全量训练。科研11源码保持原SHA，不把审查当性能。

DEPLOYMENT_R2_SOURCE_REVIEW_PASS_ACTUAL_ADMISSION_PENDING：原稿两处源码问题和FAIL_SOURCE_ONLY保留，R2按原审查SHA格式精确比较并在SSH前独占尝试记录；SOURCE_ONLY通过，不等同部署成功。唯一原R0 Oct17 12:45BJT观察由隐藏单次sleep owner 49616 持有，未发生未来观察；新区域GPU仍等待原R0整套终态。

STATIC_PACKAGE_CPU_LAYOUT_VERIFIED：22部署材料+64code实际暂存于独立m125_region_reviewed_admission_inputs_20261015，三臂真实审查绑定与11同目录CPU导入exit0。无原R0进度/资源读取，无新NN/track。新增本地phase入口仅做传输，真实real64/full未调用；原R0整套终态/资源/对应arm真实64仍是启动条件。

2026-10-16 UNIFIED_PROTOCOL_OVERRIDE：只DT Train152训练，每候选统一结构/训练预算/checkpoint/主profile，完整DT50/CDTB80/VOT127九项；不按数据集挑权重、seed、层数或profile拼行。小差距允许有限统一训练/推理调参，每个新训练候选完整重训三集。当前R0不热改；后继先R0整套→同R0 final CAD U0/U1/U2→再有限统一调参/区域/状态，旧region直接启动顺序被覆盖。CAD12源码/CPU证据不是正式运行：审查已发现两处问题，独立R2修正中，真实CAD track=0，无新指标。原R0唯一下一观察仍Oct17 12:45BJT，不提前查询。详见UNIFIED_MODEL_PROTOCOL_20261016.md，Goal ACTIVE_UNMET。

2026-10-16 CAD_UNIFORM_SOURCE_AND_ADMISSION_REVIEW_PASS_ONLY：原科研两处main错误/R1 FAIL和首次启动CRLF字节错误/R2操作FAIL已保留；科研R2+操作R2在独立R3审查均PASS_SOURCE_ONLY/0阻断，same-family/provisional/route UNATTESTED。66布局12CPU导入实测通过，完整prepare/collect仅隔离模拟夹具。尚无实际准入、CAD真实64或完整指标。原R0整套→同R0 final统一U0/U1/U2，原Oct17 12:45BJT唯一等待不改。见M125_CAD_UNIFORM_20261016/README.md，Goal ACTIVE_UNMET。

2026-10-17 12:47BJT R0_DUE_NODE_VERIFIED_PROGRESS：原895667 exactcmdline live，5完整pass，第6轮141/152，已提交901/912序列、1302307/1318812track，AB/C各40328步，无新final/九项。原本地49616缺失且无attempt/terminal，退出UNKNOWN，不作为NN失败；同已审R2入口一次实际2052 exit0，不重启NN。下一唯一读取Oct18 12:15BJT/04:15UTC，root calendar wait，无新Windows helper，之前不查NN/log/GPU/handle。预计训练13:17/整套次日12:30仅为估计。原R0完整suite后再统一CAD，当前不启动CAD/ROI。见end_node_20261017，Goal ACTIVE_UNMET。

2026-10-18 M125_R0_FIXED6_COMPLETE：同一final cf88e9c345c393caf09ad42fbe21b1bcd3b6c6bacd13bfc5124b12e511bc2a2f完成Train152×6及DT50/CDTB80/VOT127，15 stages全exit0、无自动重试；DT70.209667/61.503696/65.568958，CD75.459618/66.175778/70.513432，VOT75.786243/82.318545/91.359821，249失败anchors，4/9目标通过。指标文件本体与all_results SHA逐项一致。独立结果审查见M125_R0_FULL_RESULT_AUDIT_20261018，CAD U0/U1/U2待实际准入，ROI不提前启动，Goal ACTIVE_UNMET。
