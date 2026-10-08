# R1模板动作预检辅助模块

状态：辅助模块已编写，Python 3.8 AST语法检查完成；新鲜源码审核已完成：helper源码PASS、整体范围WARN、blocking_count=0（same-family/provisional）。尚无采集/预检runner，没有导入执行模块、GPU预检或K/K结果。R0两版完整评测及结果审核完成后，才执行新GPU工作。本文描述接线，不是实验结果。

实现见[`template_write_forks.py`](../../template_write_forks.py)，复用原[`FullDenseTracker.step`](../../full_dense_tracker.py)，没有修改当前32份源代码或原52项门。

## 动作前状态的重建

原step在提交本帧框、query和selected_feature后，只通过z_dict的append/pop和z_patch_arr赋值完成模板动作。采集方在step前保存模板列表及patch，在真实原规则写入完成后，模块从本帧已经提交的状态建立独立副本，将副本的模板/patch还原到动作前。主轨迹仍保留原step的真实写入结果。这个重建不恢复旧框、旧query或旧frame_id。

| 状态 | 副本的来源/处理 |
|---|---|
| bbox、frame_id、选中特征、query | 本帧step后；可变列表和张量独立复制 |
| 动态模板与patch | 本帧step前的值；模板张量存储独立，保留首帧两槽同对象的别名关系 |
| 固定首帧证据、词语、mask、冻结模型 | 保持只读；模型依次调用，不能并发共享临时CLIP hook |
| W动作 | 仅替换动态槽1及原生辅助patch；保留永久槽0及本帧提交状态 |
| K动作 | 保留动作前模板/patch；本帧提交状态相同 |

采集方接线形式如下，仅为尚未执行的调用示例：

```python
templates_before = list(actor.native.z_dict)
patch_before = actor.native.z_patch_arr
with torch.no_grad():
    out, data, record = actor.step(current_image)
if record['template_write']:
    event = event_from_native_write(actor, templates_before, patch_before, record)
    # future_images由同一视频的后续最多32帧提供，不传GT到跟踪器。
    keep_rows = probe_keep_keep(event, future_images)
    write_rows, keep_rows = write_keep_rollouts(event, future_images)
```

调用方尚未实现，必须负责合法初始化、固定P1 final、按Train manifest顺序取前12个资格事件、视频帧边界、事件缓存及两个GPU独立事件对的分配。不能由本辅助模块推断这些全流程要求已经满足。

## 本模块可检查的条件

动作前比较框、原frame_id、query、选中特征、模板/patch、固定首帧和文字输入。K/K两支同GPU依次推进自身预测状态，保存并还原Python/NumPy/Torch CPU和当前CUDA RNG，每帧严格比较预测/控制记录、query及选中特征。最多32帧，小于固定50帧模板检查间隔，未来窗口中不再写模板。

W/K返回预测轨迹，不计算GT标签。未来有效GT的IoU差、当前GT质量、未知项及排除数量仍由后续采集runner在控制路径之外计算。K/K一致性是回放预检，不是数据集跟踪性能。

## 尚待完成

1. 已取得[限定范围源码审核](../../M122_TEMPLATE_FORK_HELPER_SOURCE_AUDIT_20261009.md)：没有发现阻断helper的实际缺陷。此项不代替runner审核或GPU预检。
2. 实现和审核真实采集/预检runner及事件缓存；保存全部资格事件，不用GT筛掉坏写入。
3. 等R0封存/结果审核后，运行12事件实测；若K/K分叉，定位新接线/确定性问题再建立标签。
4. 按原计划推进全Train教师、同参数C-current/C-future和完整递归开发。

本次没有C训练、未来收益标签或正式指标。共享模型内部框架/编译内核的运行行为仍需实际K/K检验；静态源码和AST通过不能替代它。
