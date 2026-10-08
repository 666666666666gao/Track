# R1：前12个模板事件的采集与两GPU预检入口

状态：本地源码已编写并通过Python3.8 AST检查，新鲜源码审核已完成：源码PASS、整体范围WARN、blocking_count=0（same-family/provisional）。未导入Torch或执行入口；事件数量、K/K一致性、标签、耗时、显存和性能均未测。**现有R0六项完整评测封存及结果审核之后才执行新GPU工作。** 下面是预备接线说明，不能据此声称R1已通过。

入口为[`run_template_write_pilot.py`](../../run_template_write_pilot.py)，张量状态缓存为[`template_write_event_cache.py`](../../template_write_event_cache.py)，分支动作仍使用已经单独审核过的[`template_write_forks.py`](../../template_write_forks.py)。原32份评测源码不修改，旧52项门不追溯扩大。

## 两阶段接线

| 阶段 | GPU安排 | 实际代码职责 | 完成文件的范围 |
|---|---|---|---|
| collect | GPU0顺序运行固定P1原规则前缀 | 按原Train manifest/frame顺序保存前12个真实写入事件，主轨迹继续原step结果，不用GT挑事件 | pilot_events.json仅证明前缀采集；不证明K/K或全Train教师完成 |
| replay shard0 | GPU0检查偶数ordinal的6事件 | 每事件内K/K、W/K均在此GPU顺序运行；自身预测推进最多32帧 | replay_shard0/result.json只覆盖这6个事件 |
| replay shard1 | GPU1检查奇数ordinal的6事件 | 同样的本机设备内配对；不将W和K分别放两张卡 | replay_shard1/result.json只覆盖另外6个事件 |

collect完成后才启动两个replay worker。采集起点、顺序和12事件数量由原计划固定，不读取测试指标、当前IoU或未来收益。先采集的短前缀阶段只用一张卡；两个worker阶段使用两卡。这里未新增自动任务控制器，后续部署沿用项目实际运行/终止回执流程。

## 缓存与恢复

事件PT只保存模板、query、选中特征、框、frame、首帧视觉/文字参照、原生辅助patch、真实写入的模板/patch、当帧record和RNG；没有网络、decoder、优化器或整个tracker对象。模板转换保留单份快照内的别名，但建立独立张量存储。

replay先用协议首帧和首框初始化原生必要字段，再恢复保存事件的当帧框/query/feature/frame和模板。后续仅读取`t+1`至`t+n`的真实RGB-D图像，n最多32、序列尾部按实际图像数量缩短，不跨视频、不重置frame、不用GT crop/query/模板。

固定P1第三pass final、Train spec、人审bank/labels、原生权重、CLIP沿用训练结果记录核对。状态存储不是稀疏标量日志的伪重建，也不是加载不同checkpoint后重做本帧选择。

## GT与原始回执

collect不读取groundtruth文件；初始化首框来自原manifest。replay在K/K与W/K的全部预测完成后，复用原load_truth的真实GT读取与已存在toy07尾部裁切逻辑。当前GT有效时给本帧IoU，未来有效GT帧给mean(IoU_W−IoU_K)；无效GT继续为未知，无未来有效帧不产生future标签。

所有12事件都保留，不删负或零收益。common_C_training_eligible只是当前/未来有效性记录，不影响事件采集、分支动作或pilot覆盖。每个worker保存K/K第一条轨迹及W/K两条轨迹的完整record/query/feature，以及GT有效性、逐帧IoU和标签。K/K第二条由辅助函数逐帧严格比较，不保留重复的第二套完全相等张量。原始输出只在真实运行后存在；源码中的布尔记录不能代替运行回执。

两个replay result、实际进程exit、collector终止回执及全部原始PT齐备以后，才能审核12事件预检。没有把两个独立worker的局部complete字符串当作完整R1或三数据集性能结果。

## 预备参数

两阶段均使用以下真实已有输入，不更换文字：

| 参数 | 路径 |
|---|---|
| spec | /root/autodl-tmp/sttrack_full152_paired_20260925/M82/training_spec.json |
| trained-result | /root/autodl-tmp/sttrack_m122_full_causal_20261007/train_precision1/result.json |
| final | /root/autodl-tmp/sttrack_m122_full_causal_20261007/train_precision1/final.pt |
| repository | /root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1 |
| checkpoint | /root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar |
| clip-weight | /root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt |
| bank | /root/autodl-tmp/sttrack_m113_human_initialization_20261006/human_text.pt |
| labels | /home/SUTrack_RGBD_L/.aris/m113_human_initialization_20261006/human_train_labels.json |
| output | 部署时指定唯一新的数据盘pilot目录，两阶段共用此目录 |

phase=collect时不传shard；phase=replay时分别传shard=0、1，两个worker各设置CUDA_VISIBLE_DEVICES=0、1，worker内CUDA逻辑设备均为0。同一事件的两支不跨GPU比较。这里未执行这些命令，也未上传新代码到服务器。

## 推进边界

这两份新源码只完成R1调用方，不是全量教师收集器、C输入定义、C训练或正式评价。需要先获得限定范围源码审核；当前R0结束后才运行。若实际K/K或冻结摘要检查失败，先读取真实失败记录，修复新接线，再产生可用标签。不能改变旧VOT任务或把分叉当作模板收益。

限定范围审核见[源码报告](../../M122_TEMPLATE_PILOT_CALLER_SOURCE_AUDIT_20261009.md)。--shard只划分事件，物理GPU绑定由未来启动命令实现；第二条K/K轨迹比较后不重复落盘，一致性依赖真实成功运行回执。当前未测缓存往返、未证明无中断原W轨迹与恢复W逐位一致，不把K/K重复性扩大为这些结论。
