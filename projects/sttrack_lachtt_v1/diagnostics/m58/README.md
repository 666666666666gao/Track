# M58 密集语义与真实预测裁剪训练

当前v1因新训练器的中心监督与预训练底座不对齐而停止；v2修正后已从同一初始化重启，尚无学习后的开发结果。

- [低22实际初始化清单与multi-start调度核对](vot_initializations/README.md)
- [共用初始化文字生成器与Train输入/编码回放](initialization_generator/README.md)
- [真实TraX通信与初始化文字索引精度修正](trax_transport/README.md)
- [固定同权重文字内容对照与条件队列](content_controls/README.md)
- [三数据集语义入口与CPU初始化绑定验收](evaluation_preparation/README.md)
- [监督问题、同状态证据与v2重启](supervision_correction/README.md)
- [零残差三路径接线验收](preparation/README.md)
- [首帧自动文字及v1历史训练方案](training_preparation/README.md)

各阶段快照按时间保存。当前目标仍未完成。
