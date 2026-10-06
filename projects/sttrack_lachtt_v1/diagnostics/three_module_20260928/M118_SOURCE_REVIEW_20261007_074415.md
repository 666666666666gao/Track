# M118 fresh 源码审查

审查时间：2026-10-07 07:44:15 +08:00。结论 **PASS**；blocking 0，尚未解决的 non-blocking 0。可以进入现有 GPU sanity 流程，不能把本次源码通过写成 M118 训练或正式评价通过。

请求路由为 `gpt-6-astra / max`；本次为 fresh context、same-family、provisional。没有独立的实际后端/model/effort attestation，也不冒充跨模型验收。审查期间 GPU 0 次、SSH 0 次、网络 0 次、安装/下载 0 次；审查者没有修改受审实验源码。

审查读取四份新源码、计划、十份实际导入的稳定 D 目录依赖及 private `deploy_m118.py`。最终摘要见同名 JSON 的 `source_sha256`（15 文件）和 `private_transport_sha256`。不把需续写的 master、contract、MANIFEST 或审查输出加入源码绑定。

- **形状和预算正确。** 区域/外围为每候选 16/12 个 768 维样本；首帧与当前四组形成 `[B×10,56,64]` memory，查询为 `[B×10,5,64]`，当前视觉为 `[B×10,16,64]`。B 拼接 `4×64+5+1+13+1+1=277`，读出为 `277→64→1`，最后一层权重零初始化。按源码逐模块计算为 150,528 参数，与运行断言一致。此项是静态推导，未伪称已执行 Torch forward。
- **采样保持 M117 定义。** `dense_region_decoder.py:12` 与 `region_patch_evidence.py:14` 都先扩大外围框，再使用同一 RING 点序；坐标、bilinear、zero padding、align_corners=False 一致。原 half 缓存与 M117 诊断一样先转 float 采样。现有 sanity 对前三个真实 fit 状态比较区域及外围，默认 `assert_close` 未放宽；真实 GPU 比较仍待执行。
- **mask 与 Empty 边界准确。** 所有臂共用人工五槽 mask，类别槽必须有效；无效槽在反向视觉 attention、池化和直接余弦路径排除。共享 full/null 的相同 Empty 输入和 dropout=0 保证残差相消。逐值恒等范围是评分、质量、index、box。277 维 selected_feature 只与本次选中 candidate_features 自洽，会随训练改变，既不是冻结 Parent 的 64 维特征，也不证明未来历史状态不变。
- **GT 没有进入模型特征。** M118 和 Parent 输入构造均排除 `iou`；几何由 native 候选评分、位置、框和 prior 得到。首帧框属于协议输入。GT 只供训练损失和离线评价。训练仅使用 fit；development 仅被评价。有效状态实际为 2544/495，另 463 个 GT 无效事件被排除。
- **两个损失实现与计划一致。** 正 IoU 差加权 pairwise hinge 区分所有定位质量差异，包含合格框内部精度；相同/全零 IoU 不造正排序。Parent 所选框 IoU≥.5 时，保持它相对所有更差框的原间隔，允许更准确框胜出。两项权重固定为 1，无扫描。真实 fit 目标每完整遍历有 95,260 对正排序、14,135 对 Parent 保护；355 个 fit/108 个 dev 全零 IoU 状态与前述无效 GT 事件是不同概念。没有把 IoU 当物理身份或当前短语可见性真值。
- **Parent 缓存与匹配训练正确。** Parent 权重同时匹配原 result 和固定摘要 `1d187d78…`，eval、requires_grad(False)、no_grad，按 canonical64 在每个进程的 load_panel 中前向一次并缓存，随后删除模型。参数/buffer 用 torch.equal 比较，dev 选择对原 M101 行核验；随机 minibatch 不重算 Parent。三臂均重新 seed2027 初始化同结构；每阶段比较三个实际初值摘要。sanity 为同一首个随机 minibatch 的 3 更新；full 各自全新初始化，12×40=480 更新、lr3e-4、AdamW 默认 weight decay、固定 final。eval 保留且 autograd 开启，attention dropout=0。
- **对照名称与信息边界正确。** human_text 是永久主协议；visual_query 是首帧局部 dense RGB 表示在相同有效槽重复，generic 是 object。三臂结构、名义参数预算和数据顺序匹配；这不等于输入信息完全相同。额外 CLIP RGB 容量、Empty 常量、人工槽数量元信息、四属性截断及 development 反复使用已披露；没有称它为未新增训练的原生 baseline。
- **实际选中字段与报告可核验。** index/box/score/quality/feature 由同一选择聚合并逐项断言；每臂保存 fit 及 Empty/generic/visual_query/human_text 四种 dev 行、全部候选 IoU/分数和 final 回读。运行代码的 own-Empty 门槛完整覆盖正确数、均值、healthy 正确数/均值、transition 正确数。跨训练臂的正确数和均值比较需要完成后的分析，代码不自动晋升。
- **启动与失败记录合适。** 两臂 GPU0/1 后第三臂 GPU0；三份 sanity 全通过后才进入 full。子进程有日志、启动记录和 exit 文件；失败后当前波均收尾，再停止，不重试。成功 driver.exit 只在全套结束写入；失败 traceback 保留 controller.log。private deployer 先核验审查/源码，检查原 M117 完成态、两卡空闲、已有输出/私有目录不存在，再单次启动 controller。只沿用环境/缓存，secret 从 stdin 读入授权 43811 连接；没有安装、下载或清理。审查者没有执行部署脚本或其内嵌远程代码。

本次实际执行了 15 份 Python 源文件的 Python 3.8 AST 检查及 deployer 两段内嵌代码的 AST 检查。工具解释器为已有 uv CPython 3.13.12，未安装 Torch；没有进行模型前向、反向或损失张量运行。

原始证据也已重算。M117 四个生产进程 exit0，111.702875 秒，152 序列（130/22）和 3502 当前事件；两份真实响应 SHA 与完成记录一致。全部 152 初始化与 3502 当前行的 mask、候选/短语维度、有效范围检查通过；首帧区域最小已观测比例为 0.886710405。M114 原始 Empty GT 行的 3039 key 都匹配 M117，分组为 2544/495。开发 Parent 272/495，dense raw 人工类别 250/495；直接重算得到 20 救回、42 损害、230 双方合格中 164 IoU 下降、15 改善，与计划引用吻合。

审查实际 AST 提取并运行现有 `summarize` 与 `paired_vs_empty` 纯 Python 函数，分组/均值无异常。两处报告边界在审查中由主代理作了最小澄清后重新读取最终源码：一是 Empty 恒等不扩展到学习中的 277 维特征；二是 `summarize.rescues/breaks` 对原 native 候选0用 .5 跨线，而 `paired_vs_own_empty.rescue/harm` 只记 ≤.1↔≥.5 的严重变化。同一份 M117 dev 选择按后一口径为 11/13，不能与对 Parent 全部跨线 20/42 混称。最终 result 的 `metric_definitions` 和计划已写清；没有修改模型、损失、阈值或指标函数。

接下来必须取得三臂真实 GPU sanity（含非零有限 A/B 梯度、3039 Empty 恒等、原 reader 比较、Parent 冻结及初值一致）和三个 480-step full 的完成证据，再按预定标准分析 human 相对自身 Empty 和两个单独训练控制臂的效果。本审查不产生新的 M118 分数，也不证明自身递归、C 记忆、缺框/回归容量、Full152 最终训练或三集九项正式结果。

