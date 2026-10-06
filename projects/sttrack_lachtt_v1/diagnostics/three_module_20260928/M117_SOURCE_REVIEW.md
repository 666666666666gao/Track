# M117 源码审查

PASS。0 个阻塞项，0 个非阻塞问题。当前实现正确，可继续既定两卡 sanity；full 仍须等两张卡的真实 sanity 门禁通过。

审查时间：2026-10-07T07:05:02.741523+08:00。请求模型 gpt-6-astra，reasoning effort=max；review_independence=same-family，acceptance_status=provisional。没有独立 backend attestation，不能把请求路由当作实际后端证明。

- **author_reference_origin — PASS**：实际下载 clip/clip_surgery_model.py 的 19868 字节与 author_directory.json 和 author_sources_receipt.json 一致；重算 Git blob SHA1=3fecea25c2450e5583cf827f50272588847e325a。没有使用猜测的 model.py。
- **VV_head_layout_and_accumulation — PASS**：dense_region_encoding.py:15-32 对原路径 LND 输入先 ln_1，取 QKV 最后一个 width 的 V，重排为 B,H,N,D，按 D**-0.5 做 VV attention，再共享 out_proj。第一个新路径残差使用原输入，之后累加新路径，但每块 attention 始终读取该块原路径输入；不使用新路径 FFN。对应作者 Attention:71-103、ResidualAttentionBlock:253-280、VisionTransformer:317-355。
- **original_path_and_reference_sanity — PASS**：六个 pre-hook 返回 None，仅额外计算观察；原 MHA/FFN 与 encode_image 不被替换。作者参考为独立 VisionTransformer，strict 加载原 visual 权重；其 lazy surgery 只改自身。sanity 对相同输入的原 CLS 使用 torch.equal，对归一化 dense 空间输出使用 torch.testing.assert_close 默认容差，并记录原始及归一化最大差；没有额外容差、自动重试或原模型更新。
- **M116_matched_geometry_inputs_and_storage — PASS**：M117 与 M116 使用同一 native 六通道 factor4/256 crop 后取 RGB、同一 224 预处理、同一完整精度 prior、10 原候选、逐序列 split(16) 分组、同一 ROI/ring/有效观测 reader。每序列比较 M116 相同模式的 half CLS、padding 观测比例、crop origin、split 和 event_frames 的逐值相等。已完成 M116 的 collector/interface 摘要与当前依赖完全一致。
- **human_input_GT_and_protected_baselines — PASS**：人审 Train152 M113 bank 通过 human_confirmed、dataset、labels/encoder 摘要及序列/split 核查。收集器只读原 M90 已选事件和候选，后续 GT 仅由独立 CPU analyzer 从 M114 真值读出用于定位分组；没有 Test/CDTB/VOT、M112 当前候选弱标签、optimizer、checkpoint、tracker/query/template/C 状态提交。已有事件选择曾使用 Train GT 的边界保留于 M116 协议。新缓存独立目录且不存在时才创建，不写旧缓存/权重。
- **sanity_then_full_and_transport — PASS**：两卡各一个 fit 序列的 3 当前事件+t0 sanity 均 exit0，且作者对应、冻结状态、CLS 和无 GT 检查通过后，才启动两卡 full。full 强制合计 3502 事件/152 序列；异常保留原日志并停止。部署器先检查已完成 M116、新输出不存在、两卡空闲、4GiB 空间、审查文件摘要，再传输并逐字节回读、只启动一个 controller；没有下载、安装或清理操作。
- **all_readouts_and_claim_boundaries — PASS**：raw/Empty-centered 各保留四种内容×两种 reader，七列输入不重新编码。IoU>=.5 与 <=.1 组、合格框内部排序和固定状态 argmax 均只用于诊断；无开发最好规则选择。centered Empty 全零时首候选 tie 已明确披露为非语义增益。last-six 与论文深度7差异、复用作者方法、反复 development、未知/多帧人审、未解决 B/C、无正式三集指标均有边界说明。
- **CPU_analyzer_against_actual_M116 — PASS**：使用本地 Python 3.13.12、stdlib 导入 M117 analyzer，读取实际 M116 3654 行和 M114 2544 fit/495 development 有效 GT；M116 原始 summary 的 144 个字段逐值相等，raw 与 centered 各12152条 margin records。另核对 centered Empty 的全部零分、全部 quality tie 和 argmax 首候选结果。未生成或冒充 M117 结果。

本次实际做了 11 个 Python 源文件的 Python 3.8 AST 语法检查，以及部署器两段远端代码的 AST 检查；使用已存在的本地 Python 3.13.12 完成上述 M116 CPU 重算。作者文件 Git blob 和 SHA256 实算通过。所有受审 D 文件、私有部署器与作者文件在写报告前保持同一摘要，详见同名 JSON。

本次 GPU 检查 0、Torch 数值执行 0、SSH 0、实验代码修改 0。没有实际运行 M117，因此不声称其作者默认容差、远端显存/时长或追踪性能已通过。M116 四个 producer 的 exit0 与 108.4552903175354 秒来自已经完成的历史原产物；M117 的 2–6 分钟只是计划估计。CPU analyzer 在本地 Python 3.13.12 验证，不在远端 GPU 传输清单内。
