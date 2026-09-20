

### 5.181 M87输入预处理与模型来源核查：当前重建通过，语义正确性仍未建立（2026-09-21）

在训练等待期间，对同一152条初始化输入执行CPU预处理重建，无Qwen权重加载、无generate、无优化、不占用训练GPU，不修改文字或冻结实验。逆变换依据当前安装库源码恢复patch排列、RGB通道及归一化，再与送入processor的缩放RGB逐像素比较。

| 核查项 | 结果 |
|---|---:|
| 原图SHA、裁剪尺寸绑定 | 152/152 |
| 与历史生成记录image_grid_thw一致 | 152/152 |
| 单图占位、image token数量、有限tensor | 152/152 |
| patch逆变换后RGB取整逐像素一致 | 152/152 |
| 静态图两个时间patch逐值相同 | 152/152 |
| 当前本地模型文件与官方固定版本元数据一致 | 14/14 |

实际image token为128–180；原裁剪短边最小/中位/最大为16/57/349像素。反归一化最大浮点误差4.57763671875e-05，取整后完全一致，比较的是缩放输入而不是未缩放原图。当前重建未发现空图、RGB颠倒或patch错位；历史未保存pixel_values，因此不能将此写成历史内部tensor重放、生成器语义正确或全部输入链绝对无误。8项预定重建图已保留，放大不增加原图细节。

补充以Qwen官方固定revision `66285546d2b821cf421d4f5eb2576359d3770cd3` 的公开API元数据核对当前14个文件：权重用LFS SHA256，其余用Git blob SHA1并同时验证下载manifest SHA256，全部一致，总计7520919614字节。包括chat_template、tokenizer、merges、vocab等；没有下载/替换权重。这只确认当前来源一致，不重放生成数值或解释caption错误。

§5.180语义筛查另经新上下文审查：主审查65项确定性检查通过，预处理/来源补充41项通过，整体WARN。WARN来自助手语义proxy的测量局限，以及封存哈希不能独立证明历史操作顺序，未发现本轮数字或文件不符的FAIL。原始报告与补充检查分别保留于证据包。审查为gpt-6-astra/max请求路由、same-family/provisional，非跨模型人工真值；不得因确定性计数或预处理通过，把助手初筛提升为准确率证据。完整结论见公开`review/EXPERIMENT_AUDIT.md/json`。

2026-09-20T18:52:00.656468+00:00实际确认原控制器PID 25357与训练PID 25358仍在运行，已完成19/130条拟合、27455次跟踪、861次优化；磁盘可用1120014336字节。按已完成调用吞吐估计训练尚需约3.40小时，该估计不是完成证据。后续仍按原冻结四组开发与18项条件执行，无新正式三数据集指标。

预处理证据包SHA256 `fa76a71ba7fc90ff8112e4eac94b14c2fb9ed758c72076ce1843f385a225510e`；152项重建记录 `4df0c8deffb06e19229df48dce04a5b1a2477c759fa5b18f0b082b6a58241b3c`；模型来源结果 `ce0879de512c2fc4a42ef21c5a51fa3ce8ba78b6145a8802e1db7b743554f36b`；审查包 `e4409fc916538fd8a9d2dd9e7fcbaf71381407bab4b8e746e3b2d2907fee19a3`。代码与报告位于`projects/sttrack_lachtt_v1/diagnostics/m87_crop_only_text/input_reconstruction/`；本地图册`RGBD_TEXT_PROTOCOL_COMPARISON_20260921/input_reconstruction/index.html`检查了16张图片链接，未做浏览器渲染测试。
