# M123 完整 Train152 部署源码后续审计

**SOURCE_ONLY_PASS；blocking_count=0；source_gate_ready=true（仅源码范围）。**

实际 gate、包与启动 helper 没有发现阻止完整三遍训练的源码问题。原封存审计与实现未修改。当前结论不授予实际 hourly admission 或 GPU 运行证书。

审计时间：2026-10-10T16:34:36.779861+00:00。请求模型/effort：`gpt-6-astra` / `max`；实际 backend：`unattested`；`same-family` / `provisional`。

## 核验范围与结果

| 核验项 | 结果 |
|---|---|
| 六份指定部署文件 | 全文读取，哈希留存 |
| 原封存报告 | 60 个原制品逐项匹配；原 JSON SHA 未变 |
| 原实现 | 原 27 个审阅文件与 plan 26 项源码均未变 |
| 部署 ZIP | 38 个唯一且安全的 member；34 代码 + gate/plan/原 MD/JSON |
| 新 code_root | 34 文件与本地字节逐项一致，Python 3.8 语法 AST parse |
| 继承路径 | 旧 R3 的 56 个源码 pin + 8 个 native 导出 = 64；均匹配本地留存字节 |
| gate | 34 新代码 + 64 继承 = 98 个源码 pin；8 个输入 pin 完全一致 |
| 实际待用 hourly snapshot | 本次未读取、未生成、未认证 |

部署 preparer 首先核验原审计状态、plan SHA、27 项原源码、重要继承依赖及原 MD SHA；随后写入实际新 code_root 的 34 项和继承路径的 64 项（`prepare_M123_full_joint_deployment_20261011.py:17–61`）。全部 zip 成员逐字节核验，包含原封存报告与 plan（`:64–72`）。

实际 native tracker、模型、preprocessor、RGBD 数据加载、head、backbone 导出以及 `m121_dense_inputs.py` 的路径均被绑定；完整 98 路径/SHA 与对应本地来源见 JSON 的 `inherited_source_bindings`、`package_member_bindings` 和 trace。继承 map 中旧 pilot/dev/preflight 名称是旧源码 pin；当前调用链未执行这些旧入口。

## 单次 hourly snapshot 与启动链

`prepare_M123_admission_from_hourly_observation_20261011.py:12–23` 读取明确传入的 observation 文件，要求 `NN_progress_queries=1`、`NN_forward_calls=0`、`gap_since_admission_seconds>=3600`；GPU 行必须恰为 0、1 且显存使用均小于 500；旧 controller exit 必须为 0，完整旧训练与三套评测状态必须匹配。

生成的远端脚本在 admission 时要求 snapshot 不在未来且年龄不超过 300 秒，并检查 gate 固定最早时刻、gate SHA、98+8 路径 SHA、旧 exit 和磁盘空间（`:26–41`）。原 observation SHA、时间和 GPU 文本写入 launch receipt，下次允许 observation 时间设为本次加一小时（`:49–55`）。源码没有新增 GPU 查询、自动刷新 snapshot 或轮询 NN；`process.poll()` 只读取刚启动子进程的退出状态。

调用链为 `hourly admission → m123_joint_process_supervisor.py → run_m123_joint_full_suite.py → train_m123_ABC_joint_full152.py`（helper `:42–47`，supervisor `:14–20`，suite `:35–49`）。supervisor 隐藏 GPU，suite 给训练子进程显式设置 `CUDA_VISIBLE_DEVICES=0,1`（suite `:14–27`）；这与原两卡实现一致。

固定训练入口与原审计字节一致：全部 official Train152、三遍、A+B+C 联合 own-history；没有新增 pilot、dev split、缓存特征拟合或 runtime preflight。训练成功退出后才创建同一固定 final 的 Test50/CDTB80/VOT127 评测（suite `:50–73`）。无自动重试。源码意图不等于已完成训练。

## 留存 staging 证据与边界

静态 receiver 使用标准库，验证 package SHA 和 gate 的所有源码/输入 pins 后写 staging receipt（`M123_full_joint_static_receiver_20261011.py:4–23`）。留存 stdout 记录 `2026-10-10T16:18:45.898304+00:00` staging、98 源码项及零 NN/GPU 查询、未启动训练；其 package/gate 哈希与本地文件一致。本次未连接远端重复验证该状态。prepared 文件的 `not_deployed` 是更早的准备 receipt，不覆盖随后 retained staging 事件。

- 原审计 JSON SHA：`be2ea5b11df60bf4b667ef0f73718e62128ac4085f56b61cad5f5400f6bff6ce`
- gate SHA：`fadc5a3b10bdfedb3a3f6953a7f70e76a2f05ed0f8eddb57bec448d1c6d49af0`
- ZIP SHA：`e7098261b665bce6bb972f3c5da0bd12f34b07cea51d0df3402665170b1c65c3`；107177 bytes。

8 个输入精确绑定 spec、human bank、human labels、冻结 STTrack、冻结 CLIP、warm final、warm result、原审计 JSON；输入二进制与当前远端数据未在本次重新读取。它们由实际 admission 和 suite 的固定路径哈希检查保护。

本次只执行本地标准库读取/哈希/zip/AST 检查，无 SSH、GPU、模型导入、forward/backward、训练或评测。AST 使用 Python 3.11.14 的 3.8 feature grammar，不是实际 Python 3.8/GPU 测试。实际 fresh observation 尚未作为证据读取，因此不认证两卡当前空闲、不认证 admission 已通过、不认证 M123 launch、梯度、最终权重或指标。

首次 reviewer 静态检查器缺少旧 R3 controller 的本地路径映射而退出；加入真实留存目录映射后通过全部检查。初稿检查器与失败记录保存在 trace；未修改待审代码。

完整请求、响应、元数据、源码快照、SHA 映射、检查器与结果位于 `review_trace/`。原审计保持不可变。
