# M87 第一阶段代码审查：裁剪图像字幕生成

结论：**PASS**。未发现阻塞问题，未提出非阻塞代码改动。该结论只覆盖 `caption_protocol.py` 的 `prepare` / `generate` 实现；可进入既定的第一阶段运行前检查与生成。它不构成 bank 构造、训练、开发评测或类别语义正确性的通过结论。

审查由 fresh-context Codex reviewer 完成；请求路由为 `gpt-6-astra` / `max` / `fork_turns: none`。`review_independence: same-family`，`acceptance_status: provisional`。路由依据为父代理提供的实际调用参数；本上下文没有独立的服务端模型身份遥测。

## 已核对的实现

- **相同输入来源与裁剪。** 历史 `plan.json` 的完整 SHA256 与新代码第 27 行固定值一致，历史生成源 SHA256 与旧计划记录一致。旧计划恰好有 152 条唯一序列、152 个唯一图像路径，130 fit + 22 development；全部指向 `00000001.jpg`，全部整数坐标在记录的图像边界内。`prepare` 直接继承 `old['rows']`，并在运行时检查图像字节和尺寸；新旧生成器均执行 RGB 转换后同一 `target_xyxy` 裁剪，没有重新量化框。
- **只有一张 crop 进入 Qwen。** 新代码第 63–69 行将 PIL RGB crop 和固定常量提示词组成唯一用户消息，并检查 `len(images) == 1`、`videos is None`。没有把完整图像、红框图、序列名、split、旧类别、M86 审查标签、指标或后续帧送入提示词。序列字段仅参与记录和文件命名。
- **固定模型和生成协议。** 模型路径、九项历史模型/处理器文件哈希、float16、SDPA、`use_fast=False` 和本地加载流程与历史来源一致。新协议固定 seed 2027、`do_sample=False`、48 个输出 token 和缓存；crop 像素预算仍为 100352–200704。代码没有采样、自动换模型、联网下载或多次生成选择。这里核对的是配置和数据流，没有实际执行 Qwen，也没有读取服务端 generation_config 的内容。
- **先保留原文，再解析。** 第 73 行先写独立 `.raw.txt`，第 74 行才执行 JSON 解析。解析要求唯一字段 `category`、字符串、非空、首尾无空白且为小写；不清洗 Markdown、不改词、不注入默认类别。`object` 必须来自模型的真实回复。失败直接终止；不存在异常吞噬、重试或 fallback。
- **保存与资源范围。** 旧目录仅被读取。新 `captions` 目录采用不覆盖创建，`records.jsonl` 用 `x` 模式逐条追加并 flush；生成成功后检查 152 条并保存原文/records/source/spec 证据及 `actual_qwen_calls`。生成逐条执行，使用 4 CPU 线程和一个可见的 `cuda:0`，没有 optimizer；现有模型不复制、不删除。实际 GPU 可用性与 `CUDA_VISIBLE_DEVICES` 绑定由主执行流程在启动时验证，本审查未接触 GPU。

## 本地验证

使用可用的 Codex bundled Python，仅执行标准库 AST/JSON 检查，未导入或执行 torch / transformers：

1. 新、旧 Python 源码 AST 均解析通过。
2. 152 条来源记录、split 数量、唯一性、初始化文件名、整数裁剪边界和两项历史绑定全部通过。
3. 直接从被审查 AST 提取第 74–76 行解析语句，检查九个用例：`coffee cup` 和 `object` 接受；空值、大写、首尾空白、额外字段、非字符串、Markdown 包裹与损坏 JSON 均拒绝。
4. 写报告前再次核对四个被审查文件完整哈希，未变化。

这项解析检查证明既定结构约束生效；它不验证“English object category”的语义真实性。冻结计划已经要求把语义判断与性能结论分开，源代码也将 `semantic_correctness_verified` 设为 false。当前没有已生成的新回复，因此不据此制造格式或语义缺陷，也不增加推测性的过滤分支。

未执行：SSH、凭据读取、GPU 检查、模型推理、服务器图像解码/字节验证、bank 构造、训练和评测。本报告不是实际生成完成凭据。

## 被审查的确切内容

| 文件 | SHA256 |
|---|---|
| `EXPERIMENT_PLAN.md` | `44f1f38b93258424fe1b15021979ac157d3354760420e00a9ad01f9599478e13` |
| `caption_protocol.py` | `82bd117f1334517d4471c72022982e44e3b8f91204bc390def0cb5299a58394d` |
| `historical_caption_initial_v2.py` | `42789e10651e70b0f4bf6685a14d0833d797174f121d6c01642202bf19f73335` |
| `historical_plan.json` | `50d83fe2811acbec045fd35dee255f634ada284aa39fc43e62142d2b36a2addb` |

任何上述源文件变更都不自动继承本报告；后续阶段须保持自己的审查边界。
