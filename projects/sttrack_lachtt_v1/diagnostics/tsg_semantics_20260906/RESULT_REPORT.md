# TSG方向分支共享输入：源码、CPU语义与真实权重单帧诊断

2026-09-06。当前STTrack在构造反向TSG分支时，让`temp_x_flip`与`temp_x`、`temp_r_flip`与`temp_r`分别指向同一张量，再原地翻转空间切片。训练序列的一次真实权重前向确认：原生路径的两个方向实际收到相同输入；仅复制两处分支输入后，两方向输入不同。当前运行源码没有修改，未重新训练或评测这个修正版本。

## 已完成的三层证据

1. **源码。** source_snapshot/lib/models/sttrack/sttrack.py第93–107行先别名赋值、再切片写入，随后对两个分支调用TSG。这不是两份独立输入。查询尾部长度为4个历史query＋1个新query，空间前缀另行翻转。
2. **CPU语义。** cpu/probe.py从SHA绑定的模型文件提取原循环，用显式顺序敏感的cumsum探针算子检查一层和两层。原代码与函数式双向参考不同；clone-only输出与参考最大误差0，梯度在1e-12容差内一致。该算子不是实际TSG/Mamba，也不衡量跟踪效果。
3. **真实权重。** real/probe.py在DepthTrack Train的chair01_indoor上，分别从同一初始化框和同一权重创建两份独立跟踪器，各读取第一张初始化图和第二张当前图；只在第二个内存实例替换forward的两处赋值。实际执行了2次前向，未打开当前或后续GT，未计算IoU，optimizer steps=0。

| 真实单帧内部量 | 原生实现 | 仅翻转分支clone |
| --- | --- | --- |
| 两层TSG两次调用，RGB/Depth输入存储是否相同 | 全部相同 | 全部不同 |
| 两层TSG两次调用的输入最大绝对差 | 全部0 | 均非0 |
| 两层TSG两次调用的输出最大绝对差 | 全部0 | 均非0 |
| 进入融合层的RGB空间token与其倒序是否逐元素相同 | 是，最大差0 | 否，最大差44.444160 |
| 进入融合层的Depth空间token与其倒序是否逐元素相同 | 是，最大差0 | 否，最大差23.387903 |
| 输出query窗口长度 | RGB=4、Depth=4 | RGB=4、Depth=4 |

本次两模态TSG输入形状为[1,389,768]，去掉5个query后的融合输入为[1,384,768]。上述倒序是拼接后空间token序列的倒序，不是对物体旋转角度的测量。探针也记录了不同的单帧框和置信度，但置信度更大不能称作定位更准，报告不把它作为收益。

## 为什么与候选关联有关

模型第149–160行返回的search_rgb_tokens、search_depth_tokens与template tokens来自这段TSG处理后的temp_x/temp_r。现有overlay/lib/test/tracker/sttrack_local_spatial_observation.py据此采样候选与模板RoI；候选集合关联和M54读取器都调用该观测路径。因此该问题触及下游头实际使用的特征。

这条来源关系不证明它就是某条失败轨迹的原因，也不证明M42至M54的结果全部由此解释。只跑了一条训练序列的一个当前帧；后续Mamba融合和预测头仍可能利用其他信息。完整训练、递归与公开基准收益均未测量。

## 相邻帧训练还需要区分的事实

当前训练采样器的causal分支约束搜索帧晚于基准模板帧；_sample_visible_ids实际使用random.choices，带放回且不排序。四个搜索帧因此不保证独立帧、严格时间有序或相邻。dataset.get_frames按所给ID取图，processing与actor保持列表顺序，模型再按此顺序递归更新query。

这是当前捕获源码的行为，不是对下载权重实际训练历史的证明，也尚未量化其对失败的影响。仅排序搜索帧也不会把GT抖动裁剪变成预测框递归裁剪。采样顺序与TSG方向共享应单独设置训练对照，不能合并变化后归因。

## 下一步的最小改动与验证边界

候选修正仅为`temp_x_flip = temp_x.clone()`和`temp_r_flip = temp_r.clone()`，新增参数0。先建立独立源代码目录，在DepthTrack Train中从相同基础权重、相同数据与相同优化预算训练原生控制组和clone组。保持当前query窗口、模板、搜索和语言设置不变；不将源码修正热替换进正在执行的原生full127或既有M54结果。

需要先完成真实网络训练/反向契约，再冻结训练和22条完整开发递归；只有通过既定递归与保护门后才进入低22及同一权重的三个完整数据集。当前只有源码语义及单帧内部数值证据，没有新的已训练clone权重或公开指标。

## 复现与绑定

CPU示例：`python cpu/probe.py --source source_snapshot/lib/models/sttrack/sttrack.py --output cpu_replay.json`。真实前向需要原实验的STTrack环境、基础权重、训练图像及M44/M54绑定文件；在设置PYTHONPATH为该运行库后执行`python real/probe.py --m54-root /root/autodl-tmp/sttrack_m54_template_reader_v1_20260906 --output real_replay.json`，它要求原M54三项完成退出码均0。

CPU probe SHA256：a816ccfd048b3da76882bd71da0c27ec90bcbfc7939d05f94bcdf79c9238ae57。
真实probe SHA256：cf15f6edf82680d11b0541eeb40282115f1f3ec30a7c5052f495b9e2c847929e。
真实result SHA256：3086d4f856aad970fc50345b4684c6f56561bc363d6a65ff304980bcf94a412a。
底座SHA256：cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98。

独立代理复核已检查下载哈希、日志与JSON一致性、真实TSG wrapper、特征来源及采样源码。结论支持完成态诊断的内部一致性；对超出单帧的性能解释保留WARN。复核属于同GPT模型族的Type-A advisory，并非跨模型族验证，也没有独立连接服务器。原始复核文本与元数据按字节附在本目录。
