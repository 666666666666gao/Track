# M122 当前依赖的状态复核（2026-10-09）

这是同一研究上下文的追加源码调查，不是新鲜独立审核。归属为 same-family / provisional；请求的模型与推理等级不是独立后端证明。旧笔记保持原样。

结论：新增源码明确了 PreprocessorMM 与 BaseTracker 的可见状态，也发现 `VisionTransformer.forward_features` 的 `self.keep_rate` 赋值；已读的两条 Mamba 活跃实现未显示模型级跨帧缓存。**这仍不能证明完整调用链无副作用或 K/K 等价。**

## 本次证据

导出目录记为 D：`.aris/m122_full_causal_20261007/runtime_state_dependency_sources_20261009`；以下行号均为文件实际行号，每个编号绑定表中完整 SHA256。

| 编号 | 源码（公开链接为本次同SHA旧副本） | SHA256 |
|---|---|---|
| F | `D/files.json`（私有CPU导出清单，仅本地可读） | `a788906cdca0128f524999666a2baf86192ffc502ea1f142deadf1f67ad35ca2` |
| U | [D/lib/test/tracker/data_utils.py](https://github.com/666666666666gao/Track/blob/e302d048a0165c827f7a322df938484cd6e8c784/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/test/tracker/data_utils.py) | `873337d0419ecfdd08b64123cdc22a4936ccdd6af98a092bc2106c32cf401fe5` |
| B | [D/lib/test/tracker/basetracker.py](https://github.com/666666666666gao/Track/blob/e302d048a0165c827f7a322df938484cd6e8c784/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/test/tracker/basetracker.py) | `f9bc6bbfb1a8f677f86e4f1eda024d89baaf454498a5f1321f1421df67a7aa36` |
| V | [D/lib/models/sttrack/vit_care.py](https://github.com/666666666666gao/Track/blob/e302d048a0165c827f7a322df938484cd6e8c784/projects/sttrack_lachtt_v1/diagnostics/m84_centered/started/code/lib/models/sttrack/vit_care.py) | `05096a6cec0448e49f82264ad34647e267a6a50a6d96d3ee07ba10d61ce6ef0e` |
| A | [D/lib/models/layers/mamba.py](https://github.com/666666666666gao/Track/blob/e302d048a0165c827f7a322df938484cd6e8c784/projects/sttrack_lachtt_v1/diagnostics/tsg_semantics_20260906/source_snapshot/lib/models/layers/mamba.py) | `a11e49551cea9c7a188b9b7783b541b315a1f684b86e1de364bb3ad6c04f00be` |

原调用方沿用[旧笔记的文件索引](M122_TEMPLATE_STATE_SNAPSHOT_RECHECK_20261009.md)：W=`full_dense_tracker.py`，SHA `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`；N=原生 tracker，SHA `d67d551a612b80cee5b19a00f6fecd5d0f7ed0c907e800f452873afd684cc58f`；M=原生 model，SHA `d62cd0b2e6b383fd2049212f22d62334d32ea972150871522b874515e57ecb13`。

F:2–30 记录导出时间为北京时间 00:43:41.239410、四文件的字节数与哈希；F:33–49 绑定原三个受保护源码，声明原52项 gate 未修改、未创建新 gate、未导出框架或内核。四份本地字节均重新哈希匹配。U、A 的哈希与旧历史副本相同；本次额外导出证据现在把它们绑定到该时间的服务器来源，并不追溯改写旧笔记或扩大原 gate。

父任务提供的传输回执：SSH81888 退出0 / 43b56c，SCP46136 退出0 / f0b5b2；ZIP 22068 bytes，SHA `1f5e1437f12977ecb7a7aa8b4201714a4f9d908e85fa1280ac3dd7595be6e733`。本子任务未重新 SSH、下载或独立复验传输回执。

## 当前调用与状态

| 实际接线或已读函数 | 源码事实及分支含义 |
|---|---|
| `BaseTracker.__init__` | N:21 调用父构造；B:13–15 仅写 `params`、`visdom=None`。暂停/单步字段属于 B:59–82 的可视化方法；W:29–31 固定 debug=0，N:46–56 不进入 visdom 初始化。没有据此新增逐帧快照字段。 |
| `PreprocessorMM.process` | N:31 创建 U:16–19 的 mean/std；U:23–27 从图像构造张量并做非原位归一化，无 self 赋值，也无写回原图的语句。可见 Python 实现没有逐帧缓存；mean/std 是固定预处理状态。该结论现在有当前导出支持。 |
| backbone 构建/入口 | M:208–219 选择 `vit_base_patch16_224` 并调用 `finetune_track`。V:453–460、436–450 构建 `VisionTransformer`；V:92 继承 `BaseBackbone`，本文件未实现 `forward`。继承的 forward/finetune_track 未导出，因此不能宣称入口派发已完整核验。以下限定于 V:161–238 的 `forward_features`。 |
| `backbone.keep_rate` | V:163 执行 `self.keep_rate=keep_rate`，保留参数对象引用；V:216、220–221 读取它。W:56–58 每次传入固定 `native.keep_rate`。这是新增确认的 forward-time 属性赋值，不应称模型对象完全无变更；但该方法先覆盖再读取，当前顺序分支传同一规则，无证据需要额外历史缓存或分支特有值。 |
| 模板/query 的载荷 | V:171–181 从 `z[i]`、`x` 切片调用 PatchEmbed，`+=pos_embed` 的目标是其返回特征；V:199–207 读取/expand query，再 cat 到特征。没有直接写 `z[i]`、原始 query 张量或模板列表的语句。PatchEmbed 尚未核验，不能把“无直接写语句”升级成跨全部被调实现的存储无别名证明。 |
| backbone 的其他调用 | 本地 Attention/Block 前向只算局部值（V:51–66、81–90）；V:196、220–238 还调用模板 mask、candidate_elimination、BSI 与 recover_tokens。BSI 来自 V:153–156 的 Up_Down。它们的导入实现未包含在四文件中。 |

## 两条活跃 Mamba 路径

**TSG 路径。** M:36–42 构建 TSGBlock，M:100–101 调用；A:1887–1898、1906–1919 进入 CrossMambaFusion_SS2D_SSM，再经 A:1661–1689 的投影、一维卷积进入 Cross_Mamba_Attention_SSM。后者在 A:1539 明确选用 `selective_scan_fn_v1`，A:1558–1569 调用两次；其导入是 A:24–28 的外部 `selective_scan`。**不是**仅凭文件开头存在 `mamba_ssm` 导入就认定活跃 TSG 使用它。

**融合路径。** M:44–48、142 使用 MambaFusionBlock；A:1946–1977 进入 ConMB_SS2D；A:1280–1314 调用 `cross_selective_scan_multimodal_k2`。A:399–423 经 CrossScan_multimodal、SelectiveScan.apply、CrossMerge_multimodal；真正内核边界为 A:32、61 的 `selective_scan_cuda_core.fwd`。这与上述 TSG 扫描入口不同。

这两条已读 Python 活跃前向没有 self 字段赋值或显式传入/返回跨帧 inference cache。M 构造未覆盖 use_checkpoint=False，且 mlp_ratio=0；不能把 A 中未接入的 SS2D/SSM/Backbone_VSSM 替代实现或其缓存假设并入当前状态。证据为 M:36–48，A:1878–1919、1937–1977、1280–1314、1538–1575、1661–1689。

A:129–132 的原位赋值写向新建 `xs_fuse`；A:149–151 先相加再返回切片。A:40–63 的 ctx 属性与 save_for_backward 是单次 autograd 调用上下文，源码没有把它挂回 tracker 或模型充当跨帧缓存。外部扫描函数仍可能读取传入参数/张量；其具体副作用与内核实现未核验，不能据此作完整无状态保证。

A:16 修改 `DropPath.__repr__` 是模块导入时的全局类修改，A:639 的 DEV=False 是模块常量；均不是逐帧历史更新。未执行这些导入，也未加入防御分支、fallback 或异常处理。

## 对快照结论的影响与剩余边界

本次增加 `backbone.keep_rate` 的属性绑定记录，并把 PreprocessorMM/BaseTracker 及两条 Mamba Python 实现从“当前源码身份未确认”缩小到上述具体结论。没有源码证据要求新增隐藏状态快照。原有 `z_dict` 和 `track_query_before` 的列表隔离仍必要：M:117–132 会重绑输入 query 列表元素；W:107–109 会 append/pop 模板列表。永久初始模板与初始语义参照仍只读，不加入 GT 控制或重置。

仍未核验：V:30–38 导入的 BaseBackbone、PatchEmbed、cross_layer（Up_Down/candidate_elimination）、utils（generate_template_mask/recover_tokens）以及 timm/PyTorch；A:24–32 的外部 selective_scan 和编译 CUDA 内核、einops 等实现。本次也未扩大到 CLIP/框架审计。旧笔记已证实的顺序 CLIP hook 生命周期不因本次新增证据消失。

因此，实际同 GPU K/K 预检、动作前 W/K 一致性与只改变动态模板槽的执行回执仍须按原计划完成。静态源码没有给出数值等价、CUDA 确定性或完整无副作用证明；本次也不启动新实验。

读取范围：四文件全文或上述活跃段落、导入/赋值搜索、manifest 和固定 M 的调用段落。仅本地文本与哈希操作；无 SSH/网络/GPU/NN 导入、测试、安装、额外导出、委派或源码/Git/主文档修改。只新增本 Markdown。

发布说明：原始8,010字节研究笔记原封保留，SHA256 f45d03d9fed1b5694fdc176ea48dd4e26689a3aacd7aba81eaaa36ff817f4f70。父任务仅将4个源码链接改为逐字节核对过的固定Git副本，标明私有清单、指向上一公开笔记，并规范末尾及链接表头。当前服务器副本的采集时间为00:43；同SHA旧副本不意味着追溯扩大本轮52项门或证明已加载内核。没有新增源码事实、指标或运行结论。
