# M87 第二阶段代码审查：类别向量 bank 准备

结论：**PASS**。未发现阻塞问题，未提出非阻塞代码改动。本报告只覆盖 `prepare_banks.py`，在第一阶段真实完成并通过其绑定检查后，可进入 bank 准备。它不覆盖训练源代码、不允许把静态审查称为实际 bank 构造成功，也不评价类别语义正确性。

本阶段沿用已启动的 fresh-context Codex reviewer；初始请求路由为 `gpt-6-astra` / `max` / `fork_turns: none`，本阶段由父代理在第一阶段之后追加。`review_independence: same-family`，`acceptance_status: provisional`。没有重新声称第二个独立 reviewer，也没有服务端模型身份遥测。

## 已核对的实现

- **消费的是第一阶段的原始结果。** 第 13–24 行检查已完成状态、152 条、spec/source/plan/records 绑定、记录顺序、初始化图像与 crop、split 和单图输入字段，并核对 `raw` 的 JSON 内容及独立 `.raw.txt`。类别直接取真实 `records`，没有按旧类别、M86 标签或性能更正。
- **旧 bank 来源与冻结记录吻合。** 本地 M84 `training_spec.json` 的 fit/development Category 路径和完整 SHA256，与 M86 `provenance_result.json` 中的 current bank 完全一致。M86 原 bank 哈希也与历史 `text_preparation.json` 一致。M86 `prepare_audit.py` 第 47–55 行明确验证了旧 Category：槽 0 保留历史类别，其他有效槽替换为同一 Empty，padding 保留。这里复核的是已保存的来源记录和审计实现；本 reviewer 未读取远端 `.pt`。
- **相同字符串复用是内容确定的。** 第 37–41 行从原始完整 CLIP bank 的 `initialization_phrases` 与对应 token 构造缓存。同一字符串跨序列/槽位出现时要求向量逐值相等，不依赖语义标签、序列名称或性能选向量。缓存中找不到的字符串才进入冻结编码器；来自旧属性槽的相同字符串也是同一无上下文文本编码，不构成从旧属性复制新标签。
- **新字符串编码沿用历史实现。** ViT-L/14 SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` 与历史记录相同。`clip.load(..., device='cpu', jit=False)`、float/eval/冻结梯度、`clip.tokenize(..., truncate=False)`、32 条分批、`encode_text(...).float()` 与旧 `prepare_text.py` 一致；不做额外归一化、不训练 CLIP、不换模型、不下载权重。输出检查为有限值和 768 维。
- **新 Category 仅替换槽 0。** 第 63–70 行克隆旧 token 后逐行只赋值 `[i,0]`，其余槽再用 `torch.equal` 检查，mask 与 Empty 直接克隆。原 Empty bank 的路径/哈希被原样复用，同时检查序列、mask、Empty、所有有效槽和 padding；没有重新序列化或改变旧 Empty。
- **Swapped 在跟踪前固定。** 第 83–88 行按旧 development bank 序列顺序循环寻找第一个不同字符串，仅捐赠槽 0；每个接收者的 mask、Empty 和其他槽保持自身内容。映射写入独立 JSON，完整绑定进入 `bank_result.json`；脚本没有跟踪或评测调用。不同字符串不必语义冲突，代码 `lexical_policy` 已明确这一点。如果不存在不同字符串，`next(...)` 会停止，没有偷换默认 donor 的分支。
- **科学表述与数据边界。** 代码读取 caption、旧训练规格中的 bank 绑定、固定历史 bank 和 CLIP 权重；没有读取后续 GT 文件、指标结果、M86 人工/助手判断或后续图像，也不把新词称为真实类别。训练规格包含原有 GT 元数据，但此脚本仅使用其 `banks` 字段，没有使用 GT 内容或指标决定词、向量或 donor。`changed_category_vectors` 是向量变化数量，不是语义修复或跟踪收益。
- **资源与旧文件保留。** 模型编码显式在 CPU 上运行，设置 4 线程；不会调用 GPU、optimizer 或重训旧基座。新 `banks` 目录不覆盖创建，旧 bank 仅被读取和绑定；没有删除、移动或覆盖历史文件的路径。

## 本地验证与限制

源码 AST 解析通过；历史编码源哈希与其保存记录一致；M84/M86 Category bank 绑定和原始 CLIP bank 哈希全部一致；CLIP 权重固定值一致。直接从被审查 AST 提取 donor 表达式，对 `['cup','cup','bottle','object','cup']` 得到 `[2,2,3,4,2]`，确认连续同词跳过和末尾回绕行为。

未导入 torch，未访问 SSH、凭据或 GPU；未加载真实 bank tensor、执行 CLIP 或运行脚本主体。因此 `original_mask_padding_empty_and_other_slots_exact` 的最终运行凭据仍须由真实 bank 准备完成后产生，本报告只认可代码和来源记录具备对应检查/保持路径。未审查任何训练源代码；不能以本报告启动未经审查的训练阶段。

## 被审查的确切内容

| 文件 | SHA256 |
|---|---|
| `EXPERIMENT_PLAN.md` | `44f1f38b93258424fe1b15021979ac157d3354760420e00a9ad01f9599478e13` |
| `prepare_banks.py` | `e5fe015402ffc2882d36cc819ebcb2ebdceacb3c058c34123a39978200ceb960` |
| `caption_protocol.py` | `82bd117f1334517d4471c72022982e44e3b8f91204bc390def0cb5299a58394d` |
| `M84_training_spec.json` | `0f9bb841edd3e2c5171cd78ce9d1030d29a243561006d111d2c98eebfc74abd5` |
| `historical_prepare_text.py` | `61ab2bd5e879da5767c22d17e237f5a8182cc53cdcfa4ec19c18da84d5f578a1` |
| `historical_text_preparation.json` | `1510eeb53c15ef8779dd77ab7730f41e6ffe92fe4047a388536d0067bef2fe27` |
| `M86_provenance_result.json` | `fe68178143974c8291b59a1780ee3328cbc61e9d5ac3775f7dba726ff8340d8c` |
| `M86_prepare_audit.py` | `d7ce75c3967963e16f25b2cf867ac0c2e049e33afa932396a15277a43991a17a` |
| `historical_text_manifest.json` | `14ff0d72efc48aa06baae04859fec12d87b68c5107d2d92d8304cf9e6fdaf01d` |

第一阶段报告保持原样，本报告没有扩大其批准范围。
