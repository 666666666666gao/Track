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
