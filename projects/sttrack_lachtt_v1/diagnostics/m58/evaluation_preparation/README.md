# M58 三数据集语义评测接口准备

已实现共用STTrackSemantic的OPE和VOT入口，并在已有DepthTrack Train初始化输入上通过CPU文字路由及输出精度检查。**尚未用最终训练权重运行GPU入口、真实TraX收发或任何正式测试集；当前没有新的跟踪指标或晋升结论。**

本目录独立于正在训练的v2/code，160个冻结推理文件、训练方案、原文本银行和两组运行状态均未修改。2026-09-06 18:31:55 CST实查text/visual均完成26/130条、36181次track、1132次优化，尚无终态；等待队列继续按240秒间隔运行。实验盘可用2952675328字节，两个Qwen仍分别有5/2个权重分片。

## 为什么需要初始化观测索引

原生评测入口没有文本输入。VOT又会在不同anchor重新初始化，不能仅凭序列名取一条序列首帧描述，也不能因为多个图片都叫00000001.jpg而共享身份文字。

`initialization_key`使用RGB文件SHA与**实际收到的四个初始化坐标**构建索引。坐标以双精度字节精确封存；OPE保留原始坐标，VOT输入准备先使用toolkit实际交给tracker的xywh，再按TraX的float32协议转换。`vot_wire_bbox`仅供离线准备VOT输入，运行时不改变送入模型的框，不做模糊匹配或序列级文字回退。没有登记的画面/框组合直接报错。

真实TraX 4.0.2把[1.1,2.2,30.3,40.4]转换为[1.1000000238,2.2000000477,30.2999992371,40.4000015259]。仅按原始JSON小数建立VOT键会不匹配。若存在polygon或其他region转换，后续输入冻结必须复现toolkit的转换，不使用旧文本脚本的通用包围框作为未经核对的替代。相同画面和相同实际初始化框可以共用同一份自动文字；不把方向/anchor索引不同计作独立生成。

## 已完成的检查

| CPU检查 | 实测 |
| --- | ---: |
| 真实Train初始化token、mask、bbox与原银行逐项一致 | 152/152 |
| 真实TraX构造器与float32封送一致 | 153/153 |
| 同一图片中两个不同初始化框取得各自文字 | 2/2 |
| 未登记框、未转换的VOT小数框 | 均拒绝 |
| OPE合成输出精度检查 | 3帧，最大confidence误差3.45e-7 |
| 正式训练/推理源码保持一致 | 160/160 |
| GPU初始化、真实语义track调用、测试集图像读取、新caption生成 | 均为0 |

这不是语言正确性检查。M58原始自动文字中的类别误认和瞬时属性仍保留，没有修改描述或重新编码文本。测试用银行约3.33MB，仅留服务器，不公开图片、GT框、嵌入或权重。执行者检查不冒充独立Astra/max审阅。

## 接口文件与保持不变的协议

- `initialization_text.py`：按观测取初始化文字，保留原框。
- `semantic_runtime.py`：核验同一个base、固定final adapter、推理源码、配置、文字协议；两个入口共同构建STTrackSemantic。
- `run_semantic_ope.py`：适用于DepthTrack Test/CDTB的既有color/depth命名约定；首帧置信度1，后续保留原始Hann置信度，框及分数均导出六位小数。全部输出封存并核验后才读取后续GT。
- `run_semantic_vot.py`：只接RGB-D，使用当次TraX初始化画面及框取文字，然后报告模型原框和原始置信度。
- `m39_vot_bridge.py`：原M39已使用的桥接文件逐字复用；本次不声称已完成新入口的真实TraX交换。
- `check_initialization_binding.py`：上述CPU检查及原Train银行的观测索引包装。

原生模板输入仍为128×128、搜索256×256，因子2/4，固定4个历史query，每50步且置信度>0.75更新动态模板。未增加候选头、运动规则、在线caption或模板策略。

OPE分析复用原生`depthtrack_pr.py`，其SHA应固定为05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc；使用原有栅格重叠、置信度阈值与序列平均P/R/F，不能替换为开发集continuous IoU/H10。VOT继续使用现有toolkit和冻结low22集合。

## 后续运行清单

最终bundle尚未创建，不能填入不存在的final SHA。训练与晋升通过后，bundle需封存architecture、repository、configuration、160项source_sha256、六个接口文件的interface_sha256、base_checkpoint及SHA、adapter_checkpoint及SHA、training_spec_sha256、use_text、seed、text_protocol_path及SHA。不同数据集的运行计划引用**同一bundle SHA**，各自另绑定text_bank_path及SHA。

OPE计划另含cases_path及SHA、dataset_root、output、metric_source及SHA。每个case包含sequence、frames、init_bbox与gt_sha256；track阶段不打开groundtruth.txt，分析阶段先核验全体输出，再核验GT并计算指标。两个模式分别为`run_semantic_ope.py --plan PLAN --mode track`和`--mode analyze`。VOT入口为`run_semantic_vot.py --plan PLAN`，由现有toolkit工作区驱动。

下一步先完成v2两组训练及开发22，再做固定同权重的文字内容反事实；通过后才准备low22当次初始化图像的自动文字。新公共数据集的生成器尚未实现/运行，需严格保持本页text_protocol.json中的输入裁剪、prompt、Qwen/CLIP设置，并与训练协议核对；不能复用旧SUTrack的category_hint、扩展crop或序列稳定文字。

正式低22前，还需用固定训练权重在Train片段验证新入口与直接STTrackSemantic的GPU轨迹一致，并完成真实TraX初始化/逐帧通信。低22改善后，才运行同一最终bundle的DepthTrack Test、CDTB及VOT全量。自动注释耗时需单列，不将预计算文字的耗时隐藏在宣称的端到端速度中。

- [CPU结果与源码SHA](cpu_contract_result.json)
- [冻结的初始化文字协议](text_protocol.json)
- [实查训练与两个Qwen](live_training_status.json)
