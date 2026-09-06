# M55 最终权重配对递归：已排队，尚未执行

2026-09-06 11:08 CST 已核实两组训练原始进程仍在运行，Control 第11轮、第2750步；Clone 第10轮、第2500步。两组没有最终 result 或最终权重验收。本目录不提供新的跟踪指标。

固定使用两组完整第15轮权重，各跑同一22条 DepthTrack Train development、33130帧；与独立原生轨迹比较。训练配置、原生模板/query策略和既定晋升门不变；推理只用首帧框，全部预测封存后才解析后续GT。没有自动启动低22或全量评测。

## 实际队列

- GPU0 等待既有M56 GPU0外层控制器终止，再执行M55 Control；GPU1同理执行Clone。13:00 CST起每240秒检查前置任务，显存低于500MiB才启动。这个时间是检查起点，不是完成承诺。
- M56结束只作为释放资源的依赖，M56是否通过性能门不影响M55资格；M55自身两组训练必须正常完成，由runner逐项验证。
- GPU0等两组推理正常退出后执行统一分析，分析完成后才写队列成功标记。分析正常结束可能得到研究负结果，不能把exit=0称为性能晋升。
- 实际队列进程为GPU0 PID322434、GPU1 PID322595，完整来源见[启动收据](recursive_queue_launch.json)。这些PID只表示该次启动时观察到的状态。

## 启动前绑定修订

审阅指出旧入口只核对结果JSON和权重SHA，没有核对权重内部现成的variant/spec/epochs/optimizer_steps；execution_binding也没有连回其初始化摘要。新版仅补齐这些字段、固定该组最终路径，并绑定实际trainer、模型源码和初始状态。两组参数结构相同，成功加载不单独证明组别一致；本次没有发现已发生混组。

原runner/spec保留于`recursive_binding_v1/`。`revise_recursive_binding.py`在没有递归产物或队列启动收据时核对旧SHA并修订；训练源码、训练spec、输入和指标门均未改动。新runner SHA为`56f4604b1e1ee94c201d3285ffe6baf1924e48887a1ee2b9a884c1c11988a8bc`；新递归spec SHA为`e05a9814371f5974387f8fa9bbe259a3b74b4c062391ae461f7f45625f6094e2`。

初次外层启动使用`screen -DmS`导致启动器等待GPU0队列，尚未启动GPU1。已终止这个启动器，保留已存活的GPU0队列，用`screen -dmS`启动GPU1；没有重启训练或重新创建GPU0任务。实际纠正记录保留于[启动纠正收据](recursive_queue_launch_correction.json)。

## 审阅与复现范围

两次审阅实际使用gpt-6-astra/max，先WARN、补齐后源码delta PASS；原始回复未改写。它们没有执行GPU或读取尚未产生的最终权重。最终完成态仍需核验权重、训练记录及完整轨迹。

`prepare_recursive.py`是最初v1准备器；当前v2还需要`revise_recursive_binding.py`对应的修订。两个脚本均为一次性冻结步骤，不应在已有结果目录上重复执行。`queue_recursive.py`及`run_recursive_queue.sh`负责本次调度，原生推理和指标计算位于`run_recursive.py`。

最终底座选择仍待M55完成态递归。后续文本实例绑定、语义关联和视角记忆按M56研究方案执行；换底座需重新采集特征和训练语言参数，同一最终模型仍需验证DepthTrack Test、CDTB和完整VOT。
