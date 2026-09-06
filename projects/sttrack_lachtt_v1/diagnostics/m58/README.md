# M58 密集语义与真实预测裁剪训练

v2两组训练与开发22完整递归已完成：文本mean IoU 0.712642，高于原生0.652226和同预算纯视觉0.708132；但H10由75增至78，冻结晋升门未通过，原内容队列停止，未运行公开集。v1监督修正及各准备记录保留为历史快照。

- [默认模板写入质量及跳过时刻补查](template_quality/README.md)
- [v2配对训练、完整递归结果与持续损害诊断](recursive_completed/README.md)
- [低22实际初始化清单与multi-start调度核对](vot_initializations/README.md)
- [共用初始化文字生成器与Train输入/编码回放](initialization_generator/README.md)
- [真实TraX通信与初始化文字索引精度修正](trax_transport/README.md)
- [固定同权重文字内容对照与条件队列](content_controls/README.md)
- [三数据集语义入口与CPU初始化绑定验收](evaluation_preparation/README.md)
- [监督问题、同状态证据与v2重启](supervision_correction/README.md)
- [零残差三路径接线验收](preparation/README.md)
- [首帧自动文字及v1历史训练方案](training_preparation/README.md)

后续已另立[M59同权重内容诊断](../m59/README.md)，原M58冻结晋升失败结论保持。

各阶段快照按时间保存。当前目标仍未完成。
