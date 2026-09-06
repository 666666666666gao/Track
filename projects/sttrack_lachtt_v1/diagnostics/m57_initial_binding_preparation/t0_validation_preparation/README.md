# M57 真实t0提取验收准备：尚未执行GPU

本目录准备一个固定拟合前缀的工程验收，不选择底座、不训练权重、不计算IoU，也不宣称实例语义已经正确绑定。真实执行等待M55/M56完成态递归及底座决定。

## 输入与三条路径

只使用chair01_indoor的合法初始框及初始化帧，随后逐帧推进。原M44输入清单还包含历史expected_rows等数据，本准备已另存仅含sequence、split、frames、init_bbox四个字段的单条清单；运行脚本拒绝额外字段。源码依赖已确认存在于M55两份快照中，但未加载模型执行。

| 计划路径 | 与原生路径的差别 | 要核对的问题 |
| --- | --- | --- |
| native | 无 | 原生短轨迹参照 |
| observation_t1 | 返回候选特征，按现有规则维护参照bank | 单纯观测是否改变原生输出 |
| initial_t0 | 在观测路径初始化时额外提取t0参照，只预置bank.initial | 额外前向是否改变状态和后续输出 |

每条路径固定102帧，即初始化加101个跟踪步。第50/100步若触发原生模板更新，第51/101步会实际读取新模板；同时记录写入和读取帧，不能只在更新当帧检查旧参照。没有实际写入时，模板写入覆盖字段为false，不能把该部分算作验收通过。三条路径都不调用候选关联头，也不使用文本。

## 计划检查

t0提取前后比较bbox、frame_id、query为None、模板张量与对象身份、模板patch、Hann窗、预处理统计、网络注册参数/缓冲区、module训练模式，以及Python/NumPy/Torch/当前CUDA设备随机状态。提取的2×16×768参照必须有限且不带梯度，dynamic、encoded_dynamic、previous仍保持构造后的状态。

随后逐帧比较三条路径的框、置信度、query摘要、模板摘要及模板别名关系；观测路径只增加特征返回和bank读写。记录真实模板写入后下一帧的新模板读取，并确认t0参照保持。全部预测封存后做路径相等比较，不读取后续GT。这个前缀验收不替代完整递归，也不证明属性定位或语言性能。

## 执行绑定

执行时提供一份底座决定后生成的spec，明确checker_sha256、code_root、完整source_sha256、checkpoint及其SHA、extractor及其SHA、initialization_manifest及其SHA、dataset_root、cuda_visible_devices、output、sequence和frames。固定sequence=chair01_indoor、frames=102。启动环境的CUDA_VISIBLE_DEVICES必须与spec一致。当前没有生成该spec，也没有提交GPU任务。

源码与初始化清单可复用；实际模型路径、权重和源码摘要必须在底座决定后冻结。若采用新的底座，真实提取和随后正式缓存均在该底座上执行，不复用旧t1缓存。此次只有Python语法和依赖文件存在性检查，没有GPU状态保真结果或新的独立审阅结论。
