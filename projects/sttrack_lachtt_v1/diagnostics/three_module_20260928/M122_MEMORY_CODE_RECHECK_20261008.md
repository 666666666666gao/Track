# M122 记忆代码复核（2026-10-08）

访问：2026-10-08 22:06–22:12 CST。官方 main 固定 SHA `d18a88b850e1c819e64adba37db6eb01c54a3cf3`，由官方页面 currentOid 核验；API 返回限流403，未获 API 校验。

官方[固定树](https://github.com/XiaokunFeng/MemVLT/tree/d18a88b850e1c819e64adba37db6eb01c54a3cf3)只有 [README.md](https://github.com/XiaokunFeng/MemVLT/blob/d18a88b850e1c819e64adba37db6eb01c54a3cf3/README.md)，页面计1次提交。README仅项目名和论文标题；本次无法取得对应实现，不能宣称存在可直接移植的质量头、更新API或公开记忆代码。

论文证据：§3.4 section-top按时间段筛选短期记忆代表，confidence作准则；§3.5额外CNN预测 `p_c`，用当前预测框与GT的IoU作L2监督。因此论文将记忆筛选评分与位置预测分工，但它学习当前定位质量；未据此证明原实例身份判别或未来写入效用。[官方论文§3.4–3.5，p6](https://papers.nips.cc/paper_files/paper/2024/file/1af3e0bf5905e33789979f666c31192d-Paper-Conference.pdf#page=6)。上述正文索引由根代理实读；子代理后续PDF读取超时/SSL失败，未核实附录A.3或实现细节。

本项目[冻结源码L88起](https://github.com/666666666666gao/Track/blob/6fbcc764530da9b223cb165b0d0ad1e6018ab410/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928/full_dense_tracker.py#L88) SHA256 `0aee7de2932d7294976e2ed512218cd58d281e24bf3fb34b6267167470aa1778`：选中框/feature提交后，写入看同选中位置 `native_response` 与原生interval/threshold；`selected_quality`、observation仅记录。原50帧/>0.75为本次冻结协议，数值并非此文件内定义。双模板append/pop(1)是明确的接入位置。

下一步（设计推断，未实现）：定义 `write_quality(selected_feature, selected_box, memory_context)` 与 `update_memory(write_or_keep)`，将C门接到模板裁剪前。复用既有质量输出先校准当前IoU；另将原实例匹配标签、同历史“写/不写”未来IoU差分别用于身份与写入效用监督，不能共用IoU标签冒充三者。MemVLT可借鉴评分/存储分工，具体接口仍须自建并做受控验证。

当前两版M122 final在VOT；参数/52文件门冻结，新C未实现。此笔记不代表接入、涨点或首次质量头；未修改运行源码，未执行NN、SSH、安装或提交。
