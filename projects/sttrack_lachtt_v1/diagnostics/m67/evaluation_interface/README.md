# M67独立评测接口：修正实际架构契约，CPU初始化路由通过

2026-09-07 11:26:16 CST完成。本轮检查发现，M58/M64公共`semantic_runtime.py`仍硬性要求bundle架构为`semantic_spatial_v1`，而M67的实际训练、checkpoint和tracker要求`semantic_spatial_support_v1`。因此不能把新的最终权重直接交给旧入口，也不能只把bundle的架构标签改成旧值来绕过校验。

已在M67目录下建立独立evaluation_interface。OPE入口、TraX入口、初始化文本路由和VOT桥四个文件与原版本逐字节相同；仅semantic_runtime改为严格检查support架构和null_support=True，并把checkpoint内architecture、null_support、arm、support_loss_weight与绑定配置一一核对。原来的完成状态、130条训练、training spec、use_text、base/head和源文件哈希校验继续保留。

这项修改修正的是评测接口契约，不改变模型前向、预测框、置信度、query、搜索与模板策略，也不是新增论文贡献。旧接口及M67运行中的160份模型源码均核验未改。

## 文本协议与已完成检查

新接口明确记录“类别策略用于M67训练与推理两端”。观察及编码协议沿用已有首帧生成流程，5槽、mask、类别槽与有效属性槽置空规则不变；独立协议SHA更新是为了正确记录新训练来源，不能继续写成M58完整属性权重上的事后删词方案。

| 已完成检查 | 结果与边界 |
| --- | --- |
| Train初始化路由 | 130条拟合＋22条开发，共152条实际首帧；输入key唯一，框、tokens、mask和empty向量与M67训练bank逐元素相同 |
| 类别策略 | 所有有效非类别槽等于既有CLIP空字符串向量，未生成新caption或新embedding |
| TraX矩形构造 | 152个实际Train初始化框经真实TraX rectangle转换均不变；这不是完整server通信检查 |
| OPE序列化 | 用已封存的M65 Null bag05共889帧验证原6位小数writer，框和分数舍入误差满足原接口要求；没有新跟踪调用 |
| 源码与设备 | 新入口均可编译，四个执行/路由文件与旧版相同；没有初始化CUDA，未影响双卡训练 |

接口runtime SHA为`a5d018351b6c02d7a3a88f376edc7303fcf9e1da4b0927bf989dfdac3cebc8ef`；接口spec SHA为`323620eab82b163a3a9e6de54ea8e17da5dbc0cf7336b3717ef443082b59e07b`；新文本协议SHA为`61814694e51d9747d42a86d2c36888dba91406213e912368be3600272ff20bcf`。

实际最终M67 checkpoint尚未加载，新head的GPU OPE及完整TraX回放一致性尚未验证。没有创建正式候选bundle或运行公共数据集。后续仍须完成M67原开发门与文字内容归因，再绑定最终权重和新接口，并通过实际回放检查；历史M62/M64的通过记录不能冒充新head通过。两个Qwen和现有训练继续保留。
