# M122 backbone 依赖状态复核（2026-10-09）

本件延续前两次同一研究上下文，归属为 **same-family / provisional**，不是新鲜独立审核；请求的模型/推理等级不是独立后端证明。只研究源码与导出身份，未实现 C、未执行模型或 K/K、无新指标，旧两份 primary 不改写。

**结论：当前导出补齐了 backbone 的 Python 入口和项目内 helper 调用。四文件未显示新的逐帧对象历史或模板载荷写回。** 构建期参数注册、每次 forward 的局部 mask，以及前件已发现的 `backbone.keep_rate` 属性绑定必须区分；本结论不等于外部算子无副作用或运行等价。

## 实际来源

四份新源码链接为与本次实际导出逐字节一致的固定Git副本；V链接沿用前次已核对的同SHA副本。F、采集及对应清单为私有本地证据。编号加行号引用仍绑定完整SHA256，D指实际采集目录。

| 编号 | 对应源码与证据 | SHA256 |
|---|---|---|
| F | `runtime_backbone_dependency_sources_20261009/files.json`（私有本地证据） | `652654cad7d74423c5d0c163f39a23787d71fda55485514bdcb40a65a123f207` |
| BB | [D/lib/models/sttrack/base_backbone.py](https://github.com/666666666666gao/Track/blob/ee3e62ed3c2249b2325ad8be8300b52afb53957c/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/sttrack/base_backbone.py) | `b8f7c576072471b78a0cb025264b7d35d926b9044df45945625b0a8f35751b5a` |
| UT | [D/lib/models/sttrack/utils.py](https://github.com/666666666666gao/Track/blob/ee3e62ed3c2249b2325ad8be8300b52afb53957c/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/sttrack/utils.py) | `557ce8e3d352e3ec69b2d21fcb0f8c18703b3152eff6f3f2e1a7470585e634ea` |
| PE | [D/lib/models/layers/patch_embed.py](https://github.com/666666666666gao/Track/blob/ee3e62ed3c2249b2325ad8be8300b52afb53957c/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/layers/patch_embed.py) | `2259b7703f008748167dd65d12ac28a496235d1f20a1d5c84a3aa0da9c8c0499` |
| CL | [D/lib/models/layers/cross_layer.py](https://github.com/666666666666gao/Track/blob/ee3e62ed3c2249b2325ad8be8300b52afb53957c/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/layers/cross_layer.py) | `5008a012c53aaa0f5b7bb567b95b6118cef138644be9955e566cdf8c298d114e` |
| V | [前次实际导出的 vit_care.py](https://github.com/666666666666gao/Track/blob/e302d048a0165c827f7a322df938484cd6e8c784/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/sttrack/vit_care.py) | `05096a6cec0448e49f82264ad34647e267a6a50a6d96d3ee07ba10d61ce6ef0e` |

沿用[第一件来源索引](M122_TEMPLATE_STATE_SNAPSHOT_RECHECK_20261009.md)：M 为原生 model，SHA `d62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13`；N 为原生 tracker，SHA `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f`；W 为 `full_dense_tracker.py`，SHA `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`。

F:2–30 记录采集北京时间 01:40:16.284395、四文件字节和 SHA；F:33–53 绑定 N/M/V/W，声明原52项 gate 未变、未建新 gate、未导出外部框架/内核。子任务已对四份实际本地文件和 F 重新哈希并读取行号，均匹配。历史副本在实际身份确认前仅作预读。

父任务采集记录见 `runtime_backbone_dependency_source_extraction.json`（私有本地证据），SHA `328c864949e8ba5ad5e45c21f6337191e8734c82147e1687bf617804de5d199f`：SSH78640/SCP77185 均退出0，ZIP6102字节、SHA `e65e565fde4fa08b910d233b0048667e91715b4f8fca0dbaf6f4692309bef414`。另有`runtime_backbone_dependency_verified_public_source_links.json`（私有本地证据），SHA `1656e2af9f2ca6fd6659d3a1e3bc082eef9924fedc3c85d8ebfb031b55461faa`，记录当前导出与固定 Git `ee3e62ed3c2249b2325ad8be8300b52afb53957c` 的 m84 对应文件逐字节一致。本子任务读取这些回执，没有再次 SSH、下载或读取 Git blob；本件优先引用实际导出。

## 调用与状态结论

| 对象/函数 | 当前源码所支持的结论 |
|---|---|
| 模型构建与 `finetune_track` | N:22 先 build，N:24 再加载 checkpoint；M:208–219 构建 ViT 并调用 `finetune_track`。BB:43–46 写配置属性，BB:64–85 生成并注册模板/搜索位置参数；BB:49–61、88–108 的条件适配也属于构建阶段。W:53–73、88–115 的逐帧路径不调用该方法。现有参数/配置应冻结保留，不把这些构建操作计为分支历史更新。 |
| 实际 backbone 入口 | V:92 的 VisionTransformer 继承 BaseBackbone；V:161 定义自己的 `forward_features`。M:73–76 调用 backbone 后，BB:145–158 将 z、x、query、keep_rate 派发给这个覆盖方法，入口没有 self 赋值。BB:110–143 的基类 `forward_features` 不在这条派发路径；不能把其中 `combine_tokens` 调用作为实际 M122 操作。 |
| kwargs 与模板 mask | BB:145 接受 `**kwargs`，BB:156 没有把这些 kwargs 传下去；M:75 的 `ce_template_mask` 在此入口未参与 V 方法。V:196 另行调用 UT:6–30 的 `generate_template_mask`：按模板特征网格/数量新建中心选区，UT:21 只写刚分配的零张量，随后按模板数量拼接。该局部 mask 不读取 GT，也不是 N:84 的 `native.box_mask_z=None` 或持久化对象缓存。 |
| `PatchEmbed.forward` 与输入载荷 | PE:20–21 定义 Conv2d 和归一化；PE:23–28 先用卷积结果替换局部 x，再 flatten/transpose/norm，没有 self 赋值、原输入索引赋值或模板列表改写。V:172–181 的 `+=pos_embed` 作用于其返回特征。已读项目 Python 代码没有把位置编码写回原 `z_dict` 载荷；实际 Conv2d 等框架实现仍是外部边界。 |
| `candidate_elimination` | V:220–221 调用 CL:10–76。CL:31–48 读模板/搜索/query切片及注意力，CL:50–52 新建 mask/index；CL:67 的 `scatter_` 只修改新建 token_mask。CL:69–71 对搜索特征做乘法，再与模板和 query 拼接返回。没有写输入 tokens、注意力、模板 mask 或 query 列表，也没有将选择结果存入对象级历史。 |
| 实际 BSI `Up_Down` | V:153–156 构造各层 Up_Down，V:223–229 调用。CL:84–97 注册线性层、GELU、dropout 和 dim，零初始化也只发生在构造期。CL:99–108 前向是局部 Linear→Linear→GELU→Linear，无 self 赋值；dropout 调用在 CL:105 是注释。不能把构造初始化误作每帧权重重置或把未执行的 dropout 算入当前路径。 |
| `recover_tokens` | V:233–238 调用 UT:74–92，再拼接/归一化。UT 的 direct/partition 分支直接返回同一张量引用，template_central 分支切片再 cat；均无对象写入或跨帧保留。这是当前局部特征的引用关系，不能当作快照深拷贝，也不是新增持久缓存。 |
| `backbone.keep_rate` | V:163 的 `self.keep_rate=keep_rate` 仍是实际 forward-time 属性赋值，V:216、220–221 读取；W:56–58 固定传 native.keep_rate。入口派发现在已补齐，因此前件所述赋值有完整项目源码接线。它每次先覆盖再读取，当前顺序 W/K 没有额外分支历史值的证据。 |

## 对分支快照的限定

本次没有发现应新增到 W/K 快照的隐藏项目级缓存。原有可变容器隔离仍必要：M:117–132 修改传入 `track_query_before` 的列表元素；W:107–109 对 `z_dict` append/pop。四个新 helper 的局部 mask/index、切片与临时特征不应被误列成跨帧状态。永久初始模板和初始语义参照继续只读；不引入 GT 裁剪、控制或重置。该结论只覆盖所引源码，不能宣称全部运行时状态已穷尽。

外部 PyTorch/timm 的 Conv2d、Linear、LayerNorm、张量分配/切片/排序/scatter 与设备实现未审计（BB:3–7；PE:1–3；CL:4–6；UT:3–4；V:26–38）。[第二件笔记](M122_RUNTIME_DEPENDENCY_STATE_RECHECK_20261009.md)所列两条 Mamba 外部扫描函数/编译 CUDA 内核、CLIP/框架实现边界继续保留；不因四份项目源码补齐而推断外部内核等价、无副作用或数值确定性。原有 CLIP 临时 hook 的顺序生命周期结论也未改变。

真实同 GPU K/K 预检与动作前 W/K 一致性检查仍必需；静态源码和哈希相同不能替代它们。本件不增加防御分支、fallback、异常处理或新模型设计。

实际读取：四份导出源码全文、files.json、两份父任务采集/对应清单；准备时也读同哈希历史副本。承接既有 M/N/W/V 的固定源码行号。本子任务没有 SSH、网络/GPU查询、NN/Torch导入、测试、安装、额外导出、委派、Git或原32源码/52门修改；仅新增本私有 Markdown。

发布说明：原始8,194字节主件保持不动，SHA256 c46b513cc6d0ee1cab219871302934c2a2f359a17228644eca8a66a84994f173。父任务仅修正四个新源码及一个前次源码的公开链接、三份私有清单标记、两份公开前件链接、来源表头/说明和EOF。本次导出在01:40进行，没有扩大旧52项门或证明运行时/K/K等价，没有增加性能结论。
