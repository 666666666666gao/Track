# M58 共用初始化文字生成器：输入及编码与训练协议一致

已实现面向明确初始化观测清单的 `prepare / generate / encode` 入口，并完成CPU回放验证。**本轮没有调用Qwen生成新文字、读取公开测试集图像或运行跟踪；真实GPU生成入口尚待验收。**训练中的模型结构、底座权重、文字内容、默认模板策略和两条后续队列均保持冻结状态。

## 实现与适用边界

原训练注释器固定为152条DepthTrack Train首帧。新入口接收图像路径和初始化框，按同一图像SHA与实际框建立文字键，供已经实现的共用OPE/VOT跟踪入口读取。

初始化清单格式如下，框为示意值：

```json
{
  "coordinate_convention": "ope_raw_xywh",
  "cases": [
    {"id": "example", "image": "/path/to/initial_rgb.jpg", "bbox": [1.0, 2.0, 30.0, 40.0]}
  ]
}
```

OPE使用`ope_raw_xywh`，保留实际初始化坐标。VOT使用`vot_toolkit_xywh`，输入必须已经经过toolkit实际采用的region→xywh转换；随后调用上一阶段真实TraX测试通过的float32→四位小数文本→float32辅助函数。**本模块不负责从VOT实验中导出anchor/region，也没有把通用polygon外接矩形当成toolkit转换。**未来冻结正式VOT清单时仍需核对这一上游步骤。

同一图像内容与同一个实际初始化框复用一份文字；同图不同框保持不同的文字键。case ID和文件名只出现在审计清单中，不进入Qwen的消息文本。生成器使用当次初始化RGB及其紧裁剪，仅读取清单列出的图像，不打开GT文件或后续帧。

`prepare`封存图像SHA、实际框、裁剪范围、共享文字协议和源代码SHA。`generate`沿用训练时同一个Qwen2.5-VL-3B模型、红框全图＋无标记紧裁剪、greedy/160 token、float16/SDPA、seed2026、processor和JSON解析规则。原始回复先保存；只接受原协议的JSON及已经实测过的精确json代码围栏，不做重试、人工改写或类别补位。运行前核对模型文件/版本和指定GPU内存<500MiB。

`encode`沿用同一CLIP ViT-L/14权重、Python源代码、CPU float32、未归一化768维、固定5槽、类别后接属性的顺序、排序后的唯一短语和batch32；输出与InitializationTextBank直接兼容。空文本向量单独复制，避免将整张短语向量表连带保存。输入同属首次观测协议，未增加在线文本、位置序数、运动或模板干预。

当前共享文字协议SHA仍为d08acfb068ac5f7c428d5decb5f17af4655383543eeb4b4277e65cfda14bcac1。原自动文字中的误认与属性噪声保留；流程一致不能证明文字语义正确。

## 已完成的实际检查

| CPU检查 | 实测结果 |
| --- | --- |
| 原152条Train初始化图像SHA、尺寸与裁剪范围 | 全部相同 |
| 原152条自动回复的解析类别/属性及顺序 | 全部相同 |
| 小、中、大三种裁剪的实际Qwen processor输出 | chat、input_ids、attention_mask、pixel_values、image_grid_thw逐项完全相同 |
| 重新CLIP编码并经真实InitializationTextBank路由 | 152/152的token、mask、empty与bbox逐项完全相同 |
| 同一Train图像上的三次合成VOT初始化请求 | 相同框两次复用一键，不同框另取一键，共2个观测 |
| 新Qwen生成／公开集图像／跟踪／优化／GPU初始化 | 均为0 |

processor对照并非仅比较另一份手写公式：从SHA固定的旧caption_initial_v2.py中提取其实际预处理语句，在CPU上执行到inputs为止，再与新入口比较；没有执行旧脚本的model.generate。三条选择规则是按裁剪面积排序取最小、中位位置和最大，不依据跟踪收益选择。

CLIP复算实际加载原CLIP权重到CPU，312条唯一短语（含空串）编码耗时56.88秒；随后分别与原fit130、development22银行核对。新观测银行2350765字节，SHA=a0c8f5dd1d56bdf4a8086fc005135238e8231a15b0bb40c3ded34a78246b0f73，留在服务器；未复制主模型权重。原银行SHA仍为d7ce0833…/81879fa1…。

这些检查证明输入和编码迁移的一致性，不证明新文字生成的语义质量、GPU生成完整流程或跟踪性能。`generate`的真实GPU调用、正式数据集观测清单、最终权重跟踪与指标尚未完成。

## 后续运行

同一冻结训练与开发/文字内容门通过后，由主流程分配空闲GPU并在独立screen启动；本次没有新增自动GPU队列。先在Train初始化片段验收真实生成入口，再冻结低22的当次初始化清单。依次运行：

```bash
/home/qwen25_env/bin/python initialization_captions.py prepare --inputs INPUTS.json --output OUTPUT_DIR
CUDA_VISIBLE_DEVICES=0 /home/qwen25_env/bin/python initialization_captions.py generate --plan OUTPUT_DIR/plan.json
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python initialization_captions.py encode --plan OUTPUT_DIR/plan.json
```

以上命令是接口说明，本轮未运行新的GPU生成。正式评测计划仍需引用同一个最终base＋adapter＋runtime bundle及本阶段产出的银行SHA。低22实际改善后，再验证DepthTrack Test、CDTB和完整VOT；初始化文字生成耗时应单独报告。

20:28:27 CST实查训练两组各97/130条、138664次track、4325步；两条训练与两条调度Python进程均在。父推理160文件、六接口、训练/开发/内容方案及队列来源未改变，没有开发完成结果或候选bundle。两个Qwen仍完整保留，磁盘余2927144960字节。没有新独立Astra/max审阅PASS，目标仍未完成。

- [输入及processor回放结果](preprocessing_result.json)
- [实际CLIP编码和152次路由结果](encoding_replay_result.json)
- [生成与编码入口](source/initialization_captions.py)
- [独立于模型训练的回放检查](source/check_train_replay.py)
- [来源、训练进程与磁盘实查](live_status.json)

本目录公开源代码、结果、日志和省略真实初始化框的计划摘要。未发布图像、GT、词向量银行或模型权重。
