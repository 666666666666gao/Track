# M122之后：可信模板写入执行表

创建时间：2026-10-08T23:58:56.449683+08:00。本表记录实际阶段状态与后续计划；具体训练和结果由链接的真实回执证明。详细协议见[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，C拟合/开发分组见`TRAIN_SEQUENCE_SPLIT_20261008_235856.json`。

| Run ID | 目的 | 系统/数据 | 首要验收 | 优先级 | 状态 |
|---|---|---|---|---|---|
| R0 | 封存当前承诺完整验收 | 现有P0/P1，三数据集 | 两个一致完整九项行；结果审核 | MUST | COMPLETE：两版各9项完整指标已封存；fresh WARN/0阻断/5非阻断，P0 2/9、P1 3/9，联合目标未达到 |
| R1 | 证明模板动作可隔离 | 新C接口；P1固定final，Train前12合法事件 | K/K一致、W/K本帧动作前一致、无GT进入控制 | MUST | COMPLETE：三阶段exit0；12事件/一条序列；fresh WARN/0阻断/5非阻断，原GT/张量CPU复核；全量教师与C收益未证明 |
| R2 | 建立真实动作教师 | P1固定final Train152自身历史；所有原规则资格事件 | W/K 32帧有效GT平均IoU差；输入SHA和全量事件回执 | MUST | COMPLETE：152/219802调用、1297事件/81908未来调用；真实终态CPU与fresh审核完成 |
| R3 | 分离当前质量与未来写入价值 | 同参数C-current/C-future，C-fit120/C-dev32 | 完整递归序列等权IoU胜两参照、H10不增、有益写入覆盖报告 | MUST | COMPLETE：两组320更新和32dev全部exit0；fresh WARN/0阻断限定归档；Current宏IoU+1.0297pp/H10−3，Future宏IoU−0.2000pp/H10−1，Future科学门失败 |
| R4 | 覆盖C自身历史并形成final | 合并原规则/C历史Train152事件，只更新C | Full152固定预算、实际事件数与冻结摘要封存 | MUST | NOT_RUN：原Future晋升条件失败；未采集C自身历史，不由Current替代为已完成 |
| R5 | 同一最终模型正式评价 | 唯一A+B+C，Test50/CDTB80/VOT127 | 全九项目标及新增损害、耗时、结果审核 | MUST | NOT_RUN：原Future R4未执行；新Current质量分支正式成绩尚未产生 |

人工文本沿用最终核验类别和首4属性，单seed2027。当前VOT安排与每小时观察不变；不启动额外进程占用正在评测的GPU，不声称C已实现、训练、读取新视角或恢复crop外目标。

状态更新：2026-10-09T02:35:10.941114+08:00；旧审核输入T保持于EXPERIMENT_TRACKER_20261008_235856.md。没有把静态helper审核当作R1运行完成。

R1接线状态更新：2026-10-09T03:10:56.401364+08:00；新source caller仅AST3.8/源码审核，未导入或运行。原输入T保留于EXPERIMENT_TRACKER_20261009_031056_before_pilot_review_update.md。全Train教师与C优化仍未开始。

2026-10-10实际状态：R0六项已完整，新鲜审核/正式失败234与230已封存；R1仅启动，不能把控制器启动当作K/K通过或全Train教师/C训练。后续原生规则仍50/.75，固定P1与同人审bank；一条训练文本历史分隔差异见master5.435，输入未重生成。唯一小时等待PID40216/native18314，12:01:33.301177首次NN进度；此前只允许静态文档同步，无新NN轮询。旧审查输入执行表保留于EXPERIMENT_TRACKER_20261010_110133_before_R0_R1_update.md。

2026-10-10最新状态：R1原控制器11:05:51完成，12:02唯一小时观察确认终态，CPU/fresh审核已封存。R2只准备及源码审核，不宣称启动。输入T审前实际字节保存在私有R1审查目录；旧状态记录是历史，不覆盖旧回执。固定人审bank的pine02分隔差异仍保留。长任务最多每小时一次NN检查。

2026-10-10 R2全量教师已实际启动；source/部署审查与容量准入见§5.439。仅控制器启动回执，无终态性能；最多每小时一查，本地300秒等待。

2026-10-10 R3源码准备见§5.440；此后“未实现”历史文字不覆盖当前五源准备状态，实际训练/性能仍未完成。唯一原R2小时节点14:50:43.158897不变。

2026-10-10最新R3源码审查与B1闭合见§5.441；旧source pending/FAIL历史记录保留。源码PASS不是运行认证，原R2小时节点14:50:43.158897不变。

2026-10-10 R2首次小时进展见§5.442；原timer54468正常结束，唯一新timer53744/native31246/birth14:56:11.4248。每小时最多一次NN查看，C终态/开发/正式成绩尚未产生。

2026-10-10 §5.443：固定256帧constant1/0接线预检源码source PASS/0阻断/3限制，实际预检/C构造/优化/32dev/Full152/九项未执行。原R2小时观察仍14:51，下次不早于15:51:15.915699，原timer53744/native31246保留。

2026-10-10 R2第二次小时实测见§5.444；原53744/native31246正常结束，唯一新timer17716/native8217/birth15:54:50.052997。下一NN不早于16:52:14.108234，两卡采集119/152，无C/新正式指标。

2026-10-10 §5.445：R2终态CPU回读源码v1 FAIL/2阻断，最小两断言修正后v2 source PASS/0阻断；原材料保留。实际CPU回读和终态审计未执行，无新NN进度查询、无C优化、无正式指标。原timer17716/native8217，下一次观察不早于16:52:14.108234。

2026-10-10 §5.446：16:52唯一小时实测，Train152采集完成219802 calls/1297 events，两shard collect exit0；双GPU均W/K replay未终态，原controller376499未exit。无C优化/新正式指标。原17716/native8217正常退出199f7b；新唯一timer30632/native21826/birth16:57:07.50656，下一NN不早于17:52:45.946171。

2026-10-10 §§5.447—5.448：18:11唯一NN小时观察，原R2控制器及4阶段exit0，152/219802/1297/81908。首次CPU bool_ JSON保存exit1保留；仅valid显式bool的v3于18:22真实CPU全读exit0，607正552负119零19未知，共同eligible1257。独立终态raw审计与C实测仍未完成，无新九项指标；下一NN不早于19:11。

2026-10-10 §5.449：v3 source PASS/0block，原源码审查漏检如实保留；70482814B ZIP/195实际材料与227输入绑定完成，新的fresh R2终态审查运行中；未签发flag、无C优化/新九项，NN观察仍18:11唯一一次。

2026-10-10 §5.450：fresh R2 terminal WARN/0block/完成flag真实签发，993fit264dev共同1257，所有GT差值0。R3固定双卡controller源码实现/隔离exit3测试通过；21项源审进行，未部署/未NN运行，预检和C/dev/Full152/九项未完成。

2026-10-10 §5.451：R3 controller源审PASS/WARN0block，134checks，49366calls/dev臂。具体90pin包及最小外层真实异常退出记录已准备，补充源审进行；R3未部署/未NN运行，不早于19:11资源入场，源审不作性能证书。

2026-10-10 §5.452：concrete source PASS/WARN0block，137checks；实际19:18:28 supervised R3队列PID385974启动，预检→GPU0current/GPU1future120fit320steps→32dev49366calls/臂。运行结果未观察；唯一原timer30195/PID37988，20:18:28以后一次NN观察，零重启/空查询。

2026-10-10 §5.453：实际20:19单次NN观察，preflight513calls/1eligible精确通过；双C各993fit/264fixed-dev/10epochs320steps真实exit0。Own32dev current≥25seq36812calls/future≥30seq45978calls，两卡79/78%。无完整IoU/H10/正式结果；第一timer30195正常0消费，第二唯一18966/PID49376，下一21:19:01，不提前查。

| QF152新分支 | Current质量C固定final与全验证 | 冻结P1；原teacher1257事件、400更新、单一composite final | 同一final完整九项及结果审核 | MUST | COMPLETE_ALL9：§5.481同final全50/80/127、1765anchors，232失败；fresh受限报告WARN/0，4/9通过，未达联合目标 |

2026-10-10 §5.454：实际R3六阶段与supervisor exit0，21:19:36唯一小时查询确认全部结束；32dev GT复算完成，Future门失败。原seed初始化摘要的本地重建差异未确定，不称逐张量初始化已独立认证。新质量分支待源码审核，不自动续跑Future；NN下一准入不早于22:19:36.400301。

2026-10-10 §5.455：Current质量Full152独立队列实际启动；原Future R4仍NOT_RUN。一次GPU准入距上次NN观察4181.799790秒，后续NN最多每小时一次；原被动timer79637/PID15008，下一23:29:18.200091。源码PASS不是训练或性能证明。

2026-10-10 §5.456：CurrentC Full152固定final cc71e0b4…已训练完成；32former-dev已训练内。DT37919帧/CD44690帧已打印保存回执，完整OPE和VOT尚未完成，不预填指标。原第一timer79637正常exit0消费，第二唯一71834/PID54140。下一NN观察>=2026-10-11 00:29:55.759328；目前仅一小时一次观察。

| M123-FullJoint152 | A+B+C联合完整训练 | 全Train152×3，659406调用，450888可训练参数；不划分开发集 | 同一final三数据集完整九项 | MUST | ACTUAL_STARTED：22:35唯一启动，22:39真实完整trainer命令；§5.482用户改为M124最多20轮/每轮两OPE/双下降早停/best再VOT，交接源码修复审查中，尚未切换 |

2026-10-11 §5.457：用户明确直接训练完整DepthTrack Train152，不再保留开发分组。旧CurrentC缓存400步不算A+B+C完整训练；旧评价继续。新M123只做完整序列训练，不做新pilot，固定final。下一NN观察>=00:29:55.759328，原timer71834保留。

2026-10-11 §5.458：M123 source-only PASS/0；00:18:45已实际部署34代码与98源码pins，全部hash匹配；完整ABC Train152×3仍NOT_LAUNCHED，没有新NN。原71834/PID54140在00:10实际CIM活着，下一小时观察>=00:29:55，不重新划分开发集。

2026-10-11 §5.459：00:30唯一小时实测CurrentC DT50预测完成、CD72/80、VOT0/1765，未有完整九项；M123 deployment SOURCE_ONLY_PASS/0但NOT_LAUNCHED，双卡未空闲。保留原75282/PID30588第三小时等待，下一NN>=01:30:28.548952；全Train152×3联合ABC，不划开发集。

2026-10-11 §5.460：CurrentC实际两项OPE完成，同cc71final：DT67.365069/64.865103/66.091454，CD74.593625/69.886239/72.163245；不是新M123。VOT69/1765，完整九项未有。NN唯一01:30:58，下次>=02:30:58，原51380/PID54312被动等待保留。M123完整全Train152 ABC联合仍NOT_LAUNCHED，无开发划分。实际OPE审查pending，目标未通过。

2026-10-11 §5.461：实际CurrentC两项OPE独立只读审查WARN/0有限报告阻断，81checks/28files/89sealedinputs；非逐帧独立复算/非完整ABC联合。修正QF152与M123两当前状态行，旧Future门/R4/R5历史不改。下一NN>=02:30:58，原51380/PID54312等待保留。

2026-10-11 §5.462：02:31唯一小时实测CurrentC VOT165/1765 (74+91)，两卡74%/70%，本小时+96；累计/最近速率粗估剩17.03/16.81小时，不承诺终态；§5.460两OPE不重复。M123仍未启动，完整Train152 ABC协议不变。原21267/PID45392下一>=03:31:28。

2026-10-11 §5.463：03:32唯一小时实测CurrentC VOT226/1765 (105+121)，两卡65%/72%，本小时+61；累计/最近速率粗估剩18.83/25.46小时，不承诺终态；两OPE原字节一致不重复。M123完整ABC全Train152/no-dev仍NOT_LAUNCHED，等待原VOT完结及两GPU实际空闲；原1386/PID25808下一>=04:32:01。

2026-10-11 §5.464：04:32唯一小时实测CurrentC VOT306/1765 (142+164)，本小时+80；累计/最近速率剩17.99/18.40小时粗估。GPU0瞬时4122MiB73%，GPU1瞬时1MiB0%，两shard均无exit，根supervisor命令活着，不能从单次瞬时空闲认定停止。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。下一原6557/PID7892节点>=05:32:33。

2026-10-11 §5.465：05:33唯一小时实测CurrentC VOT381/1765 (176+205)，本小时+75；剩17.38/18.65小时粗估。GPU0瞬时4122MiB76%、GPU1瞬时1MiB48%，原supervisor命令仍在、完整终态未有，不据瞬时值重启或插入新占卡训练。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。下一原24295/PID54240节点>=06:33:11。

2026-10-11 §5.466：06:33唯一小时实测CurrentC VOT474/1765 (221+253)，本小时+93；剩15.78/14.03小时粗估。GPU0/1均4122MiB、73%/75%，原supervisor仍活着，完整终态未有。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。下一原58545/PID34464节点>=07:33:50。

2026-10-11 §5.467：07:34唯一小时实测CurrentC VOT570/1765 (262+308)，本小时+96；剩14.27/12.58小时粗估。两GPU4122MiB、83%/59%，原supervisor命令仍在，完整终态未有。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。下一原78430/PID45428节点>=08:34:29。

2026-10-11 §5.468：08:35唯一小时实测CurrentC VOT695/1765 (319+376)，本小时+125；剩12.04/8.68小时粗估。两GPU4122MiB、80%/73%，原supervisor命令仍在，完整终态未有。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。下一原32880/PID24380节点>=09:35:17。

2026-10-11 §5.469：09:36唯一小时实测CurrentC VOT809/1765 (381+428)，本小时+114；剩10.44/8.53小时粗估。两GPU4122MiB、82%/75%，原supervisor仍活着，完整终态未有。两OPE原字节一致，M123全Train152 ABC/no-dev仍NOT_LAUNCHED，C 519维训练/推理上一帧接口只读核对一致、非新运行证明。下一原21613/PID45532节点>=10:36:18。

2026-10-11 §5.470：10:42唯一小时实测CurrentC VOT864/1765 (423+441)，+55；GPU1本波已完成，GPU0余18，第二波883尚未开始。GPU0 4122MiB/68%，GPU1 1MiB/0%，原supervisor存活、九项终态未有。两OPE原字节不变，M123全Train152 ABC/no-dev仍NOT_LAUNCHED。余10.358累计/17.965近期小时粗估，近期受单卡收尾影响；下一原83016/PID40996节点>=11:42:06。

2026-10-11 §5.471：11:46唯一小时实测VOT943/1765，+79；首波441/441各exit0，次波28/33。原root仍活着，无九项终态；GPU瞬采4122MiB80%/1MiB0%，不据此断言第二波GPU1停工。两OPE原字节不变；余9.599累计/11.230近期小时粗估。M123 fullTrain152 ABC/no-dev仍NOT_LAUNCHED，下一原37904/PID47588节点>=12:46:51。

2026-10-11 §5.472：12:57唯一小时实测VOT1044/1765，+101，首波441/441各exit0，次波79/83。两GPU本次各4122MiB/73%及65%，root实际cmdline活着，无九项终态，OPE原字节不变；剩8.415累计/8.372近期小时粗估。M123 fullTrain152 ABC/no-dev仍NOT_LAUNCHED，下一原8098/PID40940节点>=13:57:13。

2026-10-11 §5.473：13:58唯一小时实测VOT1110/1765，+66；首波441/441各0，次波108/120，余655。两GPU各4122MiB80%/72%，原root实际cmdline活着，九项终态未有，OPE原字节不变；余7.788累计/10.056近期小时粗估。M123 fullTrain152 ABC/no-dev仍NOT_LAUNCHED，下一原53053/PID36224节点>=14:58:01。

2026-10-11 §5.474：15:03唯一小时实测VOT1189/1765，+79；首波441/441各0，次波136/171，余576。GPU瞬采4122MiB81%/1MiB0%，GPU1实际增51，不作持续停工判断；root实际cmdline活着、无终态，OPE原字节不变。余6.920累计/7.917近期小时粗估；M123 fullTrain152 ABC/no-dev仍NOT_LAUNCHED，下一原78961/PID47348节点>=16:03:10。

2026-10-11 §5.475：16:03唯一小时实测VOT1275/1765，+86；首波441/441各0，次波182/211，余490。两GPU本次各4122MiB82%/74%，root实际cmdline存在，无九项终态，OPE原字节不变；余5.879累计/5.773近期小时粗估。M123 fullTrain152 ABC/no-dev仍NOT_LAUNCHED，下一原9558/PID55304节点>=17:03:58。

2026-10-11 §5.476：17:04唯一小时实测VOT1368/1765，+93；首波441/441各0，次波219/267，余397。两GPU本次各4122MiB79%/77%，root实际cmdline存在，无九项终态，OPE原字节不变；余4.734累计/4.335近期小时粗估。用户重申全Train152，无fit/dev；已逐行核对M123三完整pass/ABC联合入口，实际仍NOT_LAUNCHED，下一原13521/PID14504节点>=18:04:54。

2026-10-11 §5.477：18:17唯一小时实测VOT1496/1765，+128；首波441/441各0，次波278/336，余269。GPU瞬采3470MiB11%/4122MiB59%，root实际cmdline存在，无九项终态，OPE原字节不变；余3.151累计/2.539近期小时粗估。M123 ABC全Train152三完整pass/no-dev仍NOT_LAUNCHED，下一原22419/PID17064节点>=19:17:23。

2026-10-11 §5.478：19:29唯一小时实测VOT1622/1765，+126；首波441/441各0，次波337/403，余143。两GPU各4122MiB78%/62%，root实际cmdline存在，无九项终态，OPE原字节不变。按较慢分片余2.113小时粗估（GPU0余104，GPU1余39）；合计1.360小时不等同两分片皆完成。M123 ABC全Train152三完整pass/no-dev仍NOT_LAUNCHED，下一原29849/PID38560节点>=20:29:18。

2026-10-11 §5.479：20:30唯一小时实测VOT1719/1765，+97；首波441/441各0，次波395/442，余46均GPU0。GPU0瞬采4122MiB78%，GPU1已空出1MiB0%；分片3保存exit仍尚无，全轮/root/all_results尚无，OPE字节不变。GPU0余0.810小时加分析收尾粗估。M123 ABC全Train152三完整pass/no-dev仍NOT_LAUNCHED，下一原92508/PID13156节点>=21:30:35。

2026-10-11 §5.480：21:32唯一小时实测VOT1763/1765，+44；首波441/441各0，次波439/442，尚余2个均GPU0。GPU0瞬采4122MiB76%，GPU1free1MiB0%；无全轮终态或九项结果，OPE字节不变。末两个平均外推2.8分钟不作承诺。M123完整ABC全Train152/no-dev admission仍未执行，下一原64382/PID22600节点>=22:32:44。

2026-10-11 §§5.481—5.482：CurrentC全九项真实完成4/9，fresh WARN/0已封存；M123全Train152已启动。用户明确20轮逐轮Test50/CDTB80选best/双下降早停，M124准备审查未切换；新协议使用外部两OPE选模必须披露。CDTB全框R70.554278，分数损失仅0.66804pp；严重低重叠10509/crop外8151，主要需修轨迹/实例选择/重新观察。


M124 actual Oct12 00:15: max20 full Train152 joint, every epoch Test50/CDTB80, consecutive dual F decline patience2, best equal mean F then complete VOT. User-authorized external checkpoint selection disclosed. R3 source-only0block, old37seq/54906calls/1711steps recovered; supervisor500073/controller500074/trainer500075 CPUstartup witnessed. No new complete metric. CurrentC CDTB onset frames: cart no secondcart jumped towardperson beforeoutcrop; backpackreappearanceoutside; trashcanstwoinstances. Hourly only native35639/PID48336 due01:15:52. Goalactive/unmet.

2026-10-12 §5.485: M124 epoch1 actual64/152 complete, 89069calls/2773steps; +34163calls/+1062steps from carried prefix. No complete epoch metric. Original35639 timer ended; next unique61508/PID7584 due02:19:08.927911. CurrentC onset candidate replay source-onlyPASS0/WARN, unexecuted; after M124 terminal/GPU release only. Accurate cart category/no secondcart still local jump at176-177 while insidecrop; causal candidate reason pending. Goalactive/unmet.

Oct12 sec5.486 localCPU saved-summary: C/P1 bbox+score SHA exact78/80, changed mug_gs/two_mugs; allboxR changed onlymug_gs +8.42048pp, macro+0.10526pp. Seriouslow10203/10509 in>=10frame segments; 98 onsets inside45/outside53, inside-to-out24/allinside21. No server/NN/progress query, no currenttraining change. Original timer61508 alive, due02:19:08.927911.

Oct12 sec5.487 actual02:42 observation: epoch1 saved95/152,134597calls/4197steps (+45528/+1424), no completedepoch metric. Remaining85205calls rough2.60-2.61h excludesOPE. Original61508 exited0; next unique99410/PID32436 due03:42:30.406579. Existing cart176 rawIoU0.155953 versusofficial0.155981; GT cropcenter[.513,.535], sourceRGBconversioncorrect; no uniquecause. No runtimeNN source change.

Oct12 sec5.488 local-only M124 epoch ledger: actual incomplete snapshot64/95 parsed, full-completion/metric branches notyet runtime-exercised. CSV underdesktopproject experimentdiagnostics, snapshot02:42. No extraGPU/NN/server call or training change; sameoriginal99410 timer due03:42:30.406579.

Oct12 sec5.489 actual03:48:21 epoch1 saved120/152,173959calls/5401steps; no completedepoch or OPE. Original99410 consumed0; next67058/PID35864 due04:48:21.593390. Cart onlyonecart: errorstartsinsidecrop, accurateclass remains soft response evidence; unique candidate cause unmeasured/read-only replay notexecuted. NNsource unchanged.

Oct12 sec5.490 actual04:48:59 epoch1 saved143/152,207253calls/6418steps; no completedepoch or OPE. Original67058 consumed0; next8991/PID35560 due05:48:59.759113. Existing cart175->176 GTboxcenter motion24.2058px versus submitted404.2770px; GT remainsinsidecrop, unique candidate cause unmeasured. NNsource unchanged.

Oct12 sec5.491 actual firstepoch Train152 complete05:08:08 exit0,219802calls/6807AB+C steps; final75aee7ac...; observed05:49 Test15/50 CDTB23/80 bboxfiles, no fullOPE metrics/best. BothGPU4122MiB74/77pct instant. Original8991 consumed0; next92589/PID4184 due06:49:41.554377. Real completed-training parser branch/source hashes checked; completedmetrics branch unexercised. No NNsource change/restart.

Oct12 sec5.492 actual06:50 Test43/50 CDTB60/80 bboxfiles, +28/+37; no metrics/best/history/earlystop. Epoch1training remains complete152/219802/6807 final75aee7ac. BothGPU4122MiB81/77pct snapshot. Original92589 consumed0 with7samejoins; sole88703/PID34708 birth07:00:33.681119 due07:50:17.764372. Approxremaining15.15/32.76min sequence-based only. No NNsource change/restart/diagnostic.

Oct12 sec5.493 actual07:52 epoch1 fullTest P/R/F66.493495/62.527686/64.449641; fullCDTB74.046124/71.273843/72.633540, samefinal75aee7ac; bothtrack0/complete/counts50:76373+80:101956. RelativeCurrentC deltaF-1.641813/+0.470295, dualfalse/streak0; trainedbestE1 mean68.541590 notclaimbeatingCurrentC. E2 actual9/152 16157calls/507updates cumulative235959/7314, noE2metrics/noM124VOT. Original88703 consumed0; sole4524/PID15572 due08:52:51.026749. Localrealmetric/best branch checked, rawfull-nine audit pending. Savedcart P1/C bbox+score hashes equal, samepriorfailure, no NN replay. No NNchange/restart.

Oct12 sec5.494 local captured metric deduction: M124 E1 formalCD R71.27384268855087 > CurrentC allR70.55427838723436, unfiltered gain lowerbound0.7195643013165096pp; newallR null/uncomputed, no causal claim. Accurate cart category/no secondcart: first local jump175to176 404px vsGT24px withtargetincrop, candidate/selection/geometry cause unmeasured. Official fixed DiMP/KeepTrack/LTMU source note saved; reliability/state/search/memory responsibilities distinguished, no module transplant or threshold change. Original4524/PID15572 andfunctions1528 retained/due08:52:51.026749; noNN/GPU/progressquery/restart. GoalACTIVE_UNMET.

Oct12 sec5.495 actual08:58 NN/GPU observation gap3918.902383s: E2 saved36/152 53156calls/1656AB+C updates, cumulative272958/8463; +27seq/+36999calls/+1149updates, latestcandlecup_indoor. E1 fourcomplete metrics/stagebytes unchanged, bestE1/streak0/earlyfalse, E2metricsblank/noM124VOT/fullgoalunmet. Original4524 trueexit0/91c0bf consumed10samejoins; next85792/PID31136birth09:00:23.804072 parent5908 actualCIMaa4d3c due09:58:09.929132, functions1566 same300s joins. Remainingtraining4.914741h committedrate estimate notpromise. No NNsourcechange/extraGPUquery/newdiagnostic/restart.

Oct12 sec5.496 localCPU actual6same-sequence E1/E2 cachedrecords: identities/order/calls/validGT AB+C counts equal, ABmeans3down3up/Cmeans1down5up; notall36, no fixed-input/PR/IoU proof. E1 result sequence_records full152 list (initialscalarassessmentcorrected), E2 snapshots tail3 only. Next09:58 collector one read-line changed toallcomplete sequence logs, AST/source-only/unexecuted, originaltail3 preserved; noNN/GPU/serverquery/model/trainrulechange. Newobserver9fc762c4..., purecollector20e91ffc..., sole85792/PID31136/functions1566 not restarted. GoalACTIVE_UNMET.

Oct12 sec5.497 actual09:59 E2 saved63/152 87225/2716AB+C cumulative307027/9523; +27/+34069/+1060, lasttoy07_indoor_320. FullE2logs0..62 captured; E1continuation115+preservedprefix37=result152, legacyRNGnotexact. Same63 countsallmatch, AB33down30up/C37down26up/writes447:449; no performance claim. Localmixed63/old36ledger assertionfailureffd8f5/originalpreserved, pathfix817fbe0/samecapture/noqueryrepeat. BestE1/streak0/earlyfalse/E2metricsblank/noM124VOT. Original85792 true0/4b53ba12joins consumed; sole97666/PID28288birth10:08:17.02302 parent23052/functions1608 due10:59:55.909137, nextfullcollectoronlyprepared. No NNsourcechange/extraquery/restart; goalACTIVE_UNMET.

Oct12 sec5.498 actual11:22 E2 saved94/152 calls132961/4145AB+C cumulative352763/10952, +31/+45736/+1429; full index0..93 captured/E2metricsblank/noM124VOT/bestE1/streak0. Original97666 true0/624d2c consumed11joins; sole69716/PID47320 birth11:25:24.171077 parent54140/functions1634 next12:22:39.116650; full11observerpreparednotrun. Localrange63 assert c78f90 rejected before writes, range94 samecapture fb2cd50; failedsource/receiptretained/noqueryrepeat. Cartaccuratecategory/no secondcart still localjump404px vsGT24px whileinside; unique candidatecauseunmeasured/deferred, no sourcechange/restart. GoalACTIVE_UNMET.

Oct12 sec5.499 actual CPU E1CDTB80/101956 native18420/0dd4cc0 raw238b30ba source508802fa/325inputs fixedfinal75aee7ac/metric05879f2e; allR72.75005956 vsC70.55427839 +2.19578117, reportloss1.47621687 vs.66803973, formaldelta1.38760403, fixedRgap2.84994044 under75.6. Low10509→8466/out8151→4982/inside2358→3484/long98→91/onsets54inside37out, GTvalidtransitions159/out60. Robot+50.85/cart+12.19/trashcans_A+30.11/jug−52.65, all80CSV preserved; notmodule/textcause. R1source/R2capture reports samefamilyprovisional UNATTESTED; no NN/GPU/progressquery/trainingchange/restart/fullnineclaim. Original hourlytimer unchanged. GoalACTIVE_UNMET.

Oct12 sec5.500 actual12:23 E2 saved115/152 167230/5194AB+C cumulative387032/12001 +21/+34269/+1049, full0..114captured lastpigeon07_wild, elapsed18008.21944/remain1.572565h notpromise. E1fourmetricstagebytesunchanged/bestE1/streak0/earlyfalse/E2metricsblank/noM124VOT. Original69716/PID47320 true0 finished12:22:39.129541 consumed12joins/cell1634; observer1333/2ad2980 gap3680.463502. Sole34413/PID42640 birth12:26:31.915021 parent20612/functions1682 next13:23:59.580152/full12observerpreparednotrun. CPU E1CDTB resultsec5.499 closed17staticfiles, notnewNN or E2performance. No trainingchange/restart, GoalACTIVE_UNMET.

Oct12 sec5.501 local captured E1/C onset comparison only: cart firstlong369->[178,261)83 but three later reappearance ranges unchanged/outside; backpack196 Coutside versusE1inside yetIoU0/849severelow both; robot549 E1insideIoU.0075(notfirstcorrect), late1362 E1low54allinside versusC388outside; jug286/326 newlyoutside/E1low548vs18. No oldcart404px jump transplantedtoE1/no candidatecause or module/textclaim. Four copiedeventrows/rawSHA bound, noSSH/GPU/NN/progressquery/sourcechange/restart; original34413/PID42640/functions1682 due13:23:59.580152 preserved; goalACTIVE_UNMET.

Oct12 sec5.502 local source label/action mapping: eight SHA exactexecutedmigration pins. observation BCE isGT/crop/image positiveintersection, notcentreinside/chosenboxcorrect/candidateexists; quality BCE dense refinedIoU; C MSE detached currentselectedIoU andtemplate-only native50/.75 thenC>.5. Partialtarget withcentreoutside canremainpositive; CPUcentreoutside notalltrainingnull/no-regression. Source scope only/noNNcalibration/unique cartcause/newmethodclaim, original34413/PID42640/functions1682 due13:23:59 preserved. NoSSH/GPU/NN/progressquery/sourceedit/restart; goalACTIVE_UNMET.

Oct12 sec5.503 actual13:24 E2 saved139/152 201087/6226AB+C cumulative420889/13033 +24/+33857/+1032, full0..138captured lastcolacan04_indoor/elapsed21697.10137/remain.560925h notpromise. E1fourmetricstagebytesunchanged/bestE1/streak0/earlyfalse/E2metricsblank/noM124VOT. Original34413/PID42640 true0 finished13:23:59.588809 consumed11joins/cell1682; observer23361/6939bb0 gap3638.560252. Sole61182/PID48256 birth13:27:21.29042 parent8648/functions1746 next14:24:38.140404/full13observerpreparednotrun. Sec5.501/5.502 CPU/source contract closed5/4staticfiles, notNN orE2performance. No trainingchange/restart; goalACTIVE_UNMET.
