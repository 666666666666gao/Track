# M87 FORMAT 修订独立代码审查

结论：PASS，0 个代码阻塞。仅允许按修订后的固定协议准备 v2 spec，并继续首条原文复用、其余 151 条生成及后续已审查的 bank 准备；本报告不批准正式训练，也不证明 caption 内容正确。

审查者为 /root/m87_training_code_review。请求路由为 gpt-6-astra、reasoning_effort=max；实际服务端模型遥测未提供。review_independence=same-family，acceptance_status=provisional。本报告是当前独立审查代理实际阅读代码并执行下述本地检查后的完整书面答复，不是模拟审查或外部模型结果。

## 事实与修订范围

父任务提供的真实失败事实是：第 1 条原始回复为带 json 围栏的合法类别 JSON，v1 严格解析退出 1，records 为零行，尚未编码、训练或开发跟踪。提供的首条字节 SHA256 为 336b8950a984d7b4551ab1dc8b8a77b972f7cf9c109491d592de0f59dd8cee99，v1 spec SHA256 为 a5cd7713ff2bfec44cdad2aa9b6bdeafb343a3a79e1dc32138ff7f30ab5680a6。审查者未通过 SSH 查看远端原件；以上运行事实仍是父任务提供的证据。对提供的完整 raw 字符串独立计算 SHA，确实等于所报首条 SHA。

本地 caption_v1 中的原 caption_protocol.py、prepare_banks.py 和 EXPERIMENT_PLAN.md，逐文件 SHA 均等于先前 caption_review 报告中记录的原版哈希。当前 source 与 v1 的实际差分未改变 PROMPT、Qwen 权重路径与验证、greedy 参数、48 token、像素预算或 crop 选择。

## 实现核查

- caption_protocol.py:44-51 仅在去掉外围空白后的原文同时具有确切前缀 ` ```json\n ` 与后缀 ` \n``` ` 时切掉这层包装。随后仍执行 JSON 解析、仅一个 category 键、字符串、小写、非空检查。没有替换类别、默认词或解析失败后的再次生成。
- caption_protocol.py:64-68 在模型加载和后续生成前核对首条 raw SHA、新 source SHA 与新 plan SHA。caption_spec 自身另绑定源码及 plan。生成器按原冻结 rows 顺序工作，第一行重新处理同一个 crop，但 85-91 行明确绕过 model.generate 并读取已有 first_raw；后 151 行每行一次 generate。
- 原回复在解析前留存；records 采用独占创建。旧空 records 必须已按父任务说明归档，当前代码不会覆盖它继续。失败的其他格式仍停止，不能转入默认文字。
- prepare_banks.py:5、24 改为导入同一个 parse_category，避免生成器接受的 json 围栏在 bank 阶段再次按旧规则失败。其余 bank 构造未改，原始 raw 文件与 records 的逐字比较仍在。
- caption_result 的 actual_qwen_calls=152 表示包含 v1 首条的累计调用；同时记录 new_qwen_calls_this_launch=151、reused_previous_raw_responses=1 与 format_amendment SHA。只有这一已观测失败的一次续行适用，代码没有引入通用重试路径。

## 实际执行的本地检查

使用 uv run --no-project python 的标准库，对当前两个 Python 文件进行 AST 解析；从实际 source 的 AST 提取 parse_category 后执行 4 个应通过案例和 9 个应拒绝案例，全部符合预期。应通过案例包含父任务提供的确切首条回复、普通 JSON、object 和多词类别；应拒绝案例包含无 json 围栏、大小写不同的围栏、前置说明、首字母大写、空值、前置空格、额外键、非字符串及错误键。PROMPT AST 字面值与 v1 精确相同。

未导入 torch、未加载模型、未访问 SSH/凭据、未使用 GPU、未生成新 caption、未读取实际 bank 张量。

## 尚未执行的必要运行检查

审查时本地没有 format_amendment.json、v2 caption_spec.json 或远端失败日志副本，因而本报告是代码级 PASS。父任务需在实际执行前完成已声明的 v1 spec/log/empty records 归档，保留首条 raw，生成含已核对首条 SHA、新 source SHA、新 plan SHA 的 format_amendment，再运行 prepare v2。生成器的现有断言会检查其中供复用的三个值；完成后需核验 152 records、151 次本次生成与全部 bank 结果。正式训练仍受独立训练代码审查、实际 bank/spec 与前检约束。

没有新增 BLOCKING 或 NON-BLOCKING 代码问题。语义正确性、训练可运行性和性能在本阶段均未获验证。
