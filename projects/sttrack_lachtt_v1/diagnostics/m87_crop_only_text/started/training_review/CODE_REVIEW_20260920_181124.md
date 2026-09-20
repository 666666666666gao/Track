# M87 训练与评测代码独立审查

**结论：PASS，当前 BLOCKING=0，NON-BLOCKING=0。** 可进入既定的 96 帧、3 次优化前检。前检真实退出 0、freeze 的来源核验全部通过后，才可按固定队列执行正式训练和四组开发递归。本报告不证明前检已通过、训练已完成或性能已改善。

审查者：/root/m87_training_code_review。请求路由：gpt-6-astra，reasoning_effort=max；服务侧实际模型遥测不可见。review_independence=same-family，acceptance_status=provisional。这是独立代理实际阅读源文件、运行本地检查后形成的完整书面答复；不是模拟审查或跨模型系列验收。

## 实质核查

1. **训练主因素符合计划。** train_causal.py 与 M84 逐字节比较，唯一差异是 M87 根路径。实际 training_spec 与 M84 相比只改变已声明的 11 个输入协议、对照、来源或队列字段，新增 3 个 caption/bank 来源字段；初始化 checkpoint 摘要、289154 存储参数、损失、学习率、AdamW、累积、梯度裁剪、序列顺序及状态协议全部保留。130 条拟合序列与 22 条开发序列无交集，预算为 186694 次跟踪调用、预定 5798 次优化、seed 2027。5896 个最大累积窗口与 5798 次有效优化的区别沿用 M84；无有效 GT 的窗口不更新，文字变化不改变 GT 有效性掩码。
2. **来源绑定核对到实际产物。** 最终准备退出 0；369 个导出文件的长度和 SHA 全部独立核验。M87 的全部 161 个 integration 文件与 M84 逐字节一致，7 个复制的顶层运行/损失/指标/清单文件也一致。实际 spec、caption、bank、准备回执与脚本哈希互相匹配。训练入口验证初始权重、基座、bank、损失与 integration；递归入口验证固定 final checkpoint、训练完成计数及四组 bank；freeze 验证报告、来源、spec、前检与队列。旧 M84 结果、原生结果及 M82 历史参照的本地封存文件摘要和逐序列字段也与引用一致。
3. **因果 GT 边界正确。** 训练 GT 文件会先载入内存，但 tracker.step 不接收 GT，先提交预测框、query 和模板状态，再由 loss 使用当前 gt[i]。原生教师是同一状态上的 detached 输出，只参与训练 KL，不提交轨迹。开发 run 不读取后续 GT，仅用首帧协议框和冻结文字。analyze 对四组全部退出码、22 序列/33130 帧、预测摘要与固定 head 完整验收后，才加载开发 GT。IoU/H10 使用数据集真实框；原生模型输出没有被用作评测真值。
4. **四组内容对照真实对应同一个新训练权重。** new、old、Empty、Swapped 都加载 training/category/final.pth。old 精确沿用 M84 原 Category bank，Empty 精确沿用历史 Empty bank；new 仅换槽 0，mask、padding、其余槽与 Empty 的逐值约束保留。22 条实际 Swapped 映射独立重算后均符合冻结顺序中第一个不同字符串的规则。不同字符串不保证语义错误；类别来源仍是自动生成协议，不是语义真值。
5. **18 项判定与失败停止一致。** M84 增量 4 项、原生比较及零 H10 保护 5 项、同权重 old 内容 4 项、Swapped 内容 4 项、Empty/原生逐序列指标一致 1 项，共 18 项；代码没有增加逐序列支配 M82 的要求。Empty 判定具体是逐序列 valid/low/H10 相等且 IoU sum 差不超过 1e-8，不应表述为已验证完整轨迹逐框一致。队列在训练失败时停止；每对递归均收集两个实际退出码，任一失败均不启动下一阶段；四组完成后才分析。没有按 loss 选中间权重、多 seed、自动 fallback 或正式数据集晋升。controller.exit=0 表示执行完成，科学接受仍须看 all_gates_pass。

## 实际准备问题与处理

prepare_training.py:25-26 将运行目录用于 trainer 根路径精确替换。Python 3.8 下用相对脚本入口调用时 R='.'，因此首次准备断言停止。该真实失败的日志、exit 1 和部分产物保存在 preparation_attempt1；其时没有 training_spec 或正式训练。父任务改用绝对脚本入口，源代码未改，最终 preparation.exit=0；两次准备的各 bank 摘要完全一致。后续准备、前检、freeze、launch 保持绝对入口即可，无需增加兼容或重试代码。

格式修订已另见 FORMAT_AMENDMENT_CODE_REVIEW.md。本轮新增的实物复核确认：v1 退出 1、空 records 与首条 raw 均封存；v1/v2 caption spec 除 source/plan 摘要外逐字段相同；全部 152 条 raw 与 records 一致并经实际 parser 解析；第一条 raw 与 v1 原件摘要一致。累计调用回执为 152，其中本次 151、复用 1。bank 回执记录 94 个类别、44 个历史精确字符串向量、50 个新编码向量，fit 73/130、dev 16/22 个类别向量改变。这些是准备数据与输入变化计数，不是语义准确率或跟踪指标。

## 已执行检查与限制

- 本地 AST：6 个训练/评测 Python 文件通过；Git Bash 对 run_m87.sh 的 bash -n 通过。trainer、preflight、freeze 与 M84 的精确差分符合预期。
- 从真实 analyze AST 提取判定段做 3 个合成 CPU 检查：所有指标打平时恰好 4 个严格均值条件失败；仅均值严格提高且其余不变时全部 18 项通过；原生零 H10 序列新增失败可单独触发保护失败。合成检查不代表 M87 实验结果。
- 369 文件来源核对、161 文件运行代码一致、实际 spec 逐字段对比、152 条 raw/record 核验和 22 条捐赠映射核验全部通过。完整检查摘要在 prepared_evidence_verification.json。
- bank 张量文件、初始 checkpoint 实物及大模型权重保留在远端；审查者未本地加载这些张量，也未重跑 CLIP/Qwen。相关张量断言通过由已绑定源码的准备回执支持，不能称为本地逐张量独立复算。
- 未访问 SSH/凭据、未导入 torch、未使用 GPU、未启动前检或训练。当前没有 M87 前检结果、训练权重或开发性能结果。preflight 要求的原生前缀一致、有限梯度/损失、冻结基座及训练后 Empty 恒等，仍须在真实运行验证。

实际训练 spec SHA256：d14946d36262ee9c23153af8927bce7f3d07aa0383ae2f1c2c02933662448860。
实际递归 spec SHA256：7e1259529215b4efd93e7751a7a3e1c41fd607a4aa56f5e7ef0539daf9a0cd19。
完整来源归档 SHA256：296782cca54cc4f48c3c57e0fb9dbe07cd1854650c9e5661acaf46cff884c88b。

未提出实现补丁。父任务已确认会把本目录 CODE_REVIEW.md 与 code_review_receipt.json 原字节复制到 M87 根目录并逐 SHA 核验，满足 freeze_m87.py 的实际读取路径。报告和回执仅授权按既定阶段继续验证，不等于性能接受。
