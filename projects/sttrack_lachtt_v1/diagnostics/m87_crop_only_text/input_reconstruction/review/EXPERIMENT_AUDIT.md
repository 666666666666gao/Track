# M87 初始化语义筛查完整性审查

审查日期：2026-09-21（Asia/Shanghai）。审查任务：`/root/m87_screening_integrity`。

请求配置为 `gpt-6-astra` / `max`；此处只记录请求的路由，不声称获得了服务端模型或推理强度遥测。审查为新上下文、同家族复核，`review_independence: same-family`，`acceptance_status: provisional`。执行者只提供路径、范围及检查要求；本结论直接依据文件、代码、重算和初始化图像抽查形成。

**总体结论：WARN。确定性内容一致性：PASS。未发现数字不符、文件缺失、错误哈希、评分自归一化或伪装成人工真值的 FAIL。**

主审查的 65 项确定性检查全部通过；补充预处理/来源检查的 41 项检查全部通过。WARN 的原因是助手语义初筛的测量局限，以及当前封存哈希不能独立证明历史操作顺序。这些限制已在原报告中明确披露，不应将 WARN 解释为确认存在造假、实验性能失败或应当修改正在运行的 M87。

本报告路径以 `W = D:/Program Files/UserCache/gb/codex/tmp/sttrack_m87_crop_only_text_20260921` 为根；`S = W/semantic_screening`、`P = W/prepared_evidence`、`A = W/input_audit`、`E = W/input_audit_evidence`。旧记录位于同级 `sttrack_m86_initialization_audit_20260921`。所有行号均为此次直接读取的文件行号。

## A. 真值来源：WARN

新判断来自助手对初始化 RGB 和协议框的视觉筛查。它们既不是数据集提供的类别真值，也不是人类评审标签。首帧协议框来自旧记录里的 `first_gt_line` / `protocol_bbox`，但框的来源不能使助手的类别判断升级为真实语义真值。`S/sealed_new_labels.json:3-6` 明确写出封存状态、记录哈希和 `assistant_initialization_only_screening_not_independent_human_ground_truth`；`S/SCREENING_PLAN.md:5-9` 和 `S/M87_INITIALIZATION_SCREENING.md:3,17,29` 明确保留上下文污染、非盲评及跨次不一致限制。

`make_sheets.py:7-8` 实际读取含旧判断的旧 register，以复用图像和 ID；绘制代码 `:20-27` 只显示匿名编号、新类别、首帧红框及裁剪。19 张图册按当前源码在内存完整重建后逐字节一致，因此“展示字段隐藏”有确定性支持；“评审者没有旧知识”没有被宣称，也不能被这些图册证明。

三项不变字符串却变化的判断确实保留：006/apple，uncertain → conflicting；072/box，supported → conflicting；144/book，uncertain → conflicting。新标签的证据在 `sealed_new_labels.json:34-36,364-366,724-726`，配对记录在 `paired_register.json:73-84,997-1008,2005-2016`；汇总在 `screening_summary.json:81-123`。因此不能把跨轮差额解释成生成器准确率差额，不能把旧 supported 当作可靠真值。

## B. 归一化：PASS

`S/analyze_screening.py:25-42` 对字符串变化、三状态计数及转移做原始整数求和，没有用模型输出的最大值、均值或其他预测统计量作分母。报告展示原始计数，没有把 supported 比例包装成准确率。

补充核查中的 mean/std/rescale 是图像数值预处理的逆变换，不是评价分数归一化；见 `A/audit_processor.py:66-74`。两类“归一化”应保持区分。

## C. 数值、文件存在与哈希：PASS，封存历史顺序保留限制

独立检查按 sequence/audit_id 键连接旧 register、生成记录和封存标签，没有直接信任汇总表，也没有复用原脚本的 zip 连接作为唯一证据。六组集合（旧记录、新生成记录、封存标签、配对表、私有映射、caption spec）各有 152 个唯一条目；152 行 JSON/CSV 的所有字段均与重建值一致。

| 范围 | 条目 | 旧 supported/conflicting/uncertain | 新 supported/conflicting/uncertain |
|---|---:|---:|---:|
| 全部 | 152 | 110 / 18 / 24 | 85 / 39 / 28 |
| fit | 130 | 93 / 13 / 24 | 71 / 34 / 25 |
| development | 22 | 17 / 5 / 0 | 14 / 5 / 3 |

完整转移矩阵（行=旧，列=新，顺序均为 supported/conflicting/uncertain）为 `[[77,20,13],[6,8,4],[2,11,11]]`；89 项字符串改变子集为 `[[21,19,13],[6,5,4],[2,9,10]]`；63 项不变子集为 `[[56,1,0],[0,3,0],[0,2,1]]`。因此改变子集中的 conflicting → supported 为 6，supported → conflicting 为 19。`object` 为 0 项，但这不表示所有输出都具有具体类别辨识力，例如 `accessory`、`cylinder` 仍较宽泛。原报告对应 `:9-29`，机器结果对应 `screening_summary.json:8-166`。

全部 152 个本地首帧 JPEG 的文件 SHA 与旧记录、新记录、spec、私有映射一致；全部裁剪 PNG 的 SHA 与旧 register 一致，且像素逐点等于同首帧按记录中的整数坐标裁剪所得。spec 中 152 条图像路径均指向 `00000001.jpg`。全部 152 个 raw 文本与 records 的 raw 字段一致，实际 `parse_category` 输出均等于 category；caption.log 中 152 条完整输出也逐字段一致，退出文件为 0。该核查只读取了允许的初始化图像。

私有 manifest 的 32 项文件均存在，字节数和 SHA 全部一致；19 张匿名图册逐字节重建成功。桌面私有 gallery 的 152 张裁剪、19 张图册及六份复制文档与证据源一致，307 个本地链接均存在；独立 `node --check` 通过。`artifact_validation.json:3-5` 声称的是链接及语法检查，并明确 `browser_tested: false`，本审查也没有运行浏览器行为测试。

| 绑定 | 独立重算 SHA256 |
|---|---|
| 封存标签 | `14f6a8d146de3bc4af6200828f7698dc410b163d5dacdb08f2bb72a29bb13c91` |
| 筛查计划 | `92056cc96bc253dfc6b7b01585f0cf4c44f95d0774c9bd0c27c8d890e63adcd8` |
| caption records | `8141f3746b6ddec25beab547fe2bbb7d242b5bb617da830f971af530678208f2` |
| M86 reviewed register | `b4ca66f5ba539234690d8f97e7f3a8271ad76db639a732c7a34cd0da71d447e6` |
| 原筛查报告 | `5fc2b0e85c7513bd038d2d8b3194b81ae6e057cf171069624bcc0c998bd2c81a` |

原分析脚本 `:11-13` 的硬编码封存 SHA 与当前字节相符。封存自述 UTC 为 18:33:05.889，现存标签 mtime 为 18:33:05，分析脚本为 18:34:03，配对输出为 18:34:04；这些时间与“先封存再配对”相容。但它们不是独立可信时间戳或不可变执行日志。本审查确认内容及引用一致，不能把操作先后升级为经过密码学证明的盲评历史。

`M87_INITIALIZATION_SCREENING.md:47` 关于“公开材料”的全局隐私主张没有在本范围内全面审计：此次只确认给定本地私有产物，未检查任何远端发布内容。`screening_phase_health.json:2-3,36` 只能作为当时状态快照，不能证明当前仍在运行，也不是性能证据；本审查没有读取后续训练结果或重新检查远端进程。

## D. 代码调用与结果落盘：PASS

`S/analyze_screening.py:31-42` 定义的 counts/matrix 在 summary 构造中实际调用；`:49-52` 将该对象和配对行写入 JSON/CSV。本审查只执行其无写入的 AST 前缀（原第 1–48 行），生成的 rows 和完整 summary 与现有文件完全相等；再以独立 Counter 计算交叉验证。没有发现只定义未调用的评分函数。

`P/caption_protocol.py:78-97` 实际使用 RGB 紧裁剪、单图消息、分类解析和逐行保存；`:106-108` 提供入口调用。`caption_result.json:9-14` 明确保留 151 次本次调用加 1 次复用旧 raw 的记录，且未声称完成语义正确性验证；格式修订和复用 raw 的 SHA 一致。该检查证明代码与已存输出相容，不等同于重新运行模型。

`build_report.py:17-19,35,37,76` 中部分文字计数为手写常量；当前全部与独立重算一致，因此不是本轮错误。若以后更新数据，必须重新核对这些文字，不能仅更新 JSON。

## E. 范围及主张：PASS（仅限初始化筛查）

实际范围为同一组 152 个初始化观测，fit 130、development 22，新生成设置记录一个 seed=2027。不存在多次独立语义标注、统计显著性检验或跟踪性能重复试验。`SCREENING_PLAN.md:3,7,9` 和报告 `:3,17,29,41-43` 把范围限定为可见类别相容性，说明图像上下文和指令同时改变，拒绝将变化单独归因于去掉整帧，并保留不良结果。

“89 项字符串变化”准确；这不是 89 项语义类别改变，连字符等书写变化也计入。报告的 6 项改善和 19 项恶化仅可写成助手状态转移。`报告:41` 的“没有自动建立可靠目标语义输入”在上下文中可理解为本次仍有明显冲突、可靠性尚未建立；不能外推为 Qwen 一般能力定论或 M87 性能预测。更精确的可复用措辞是：“本次 152 项输入仍出现与初始化可见目标明显冲突的类别，当前证据不足以认定该协议的目标语义可靠。”这只是结论边界建议，不要求改动当前冻结产物。

抽查既有图册 01、09、18，共 24 个条目，覆盖三项相同字符串分歧以及明确冲突、支持和不确定案例。005 的 8 号球/tennis ball、071 的蛋/pillow、140 的玩偶/lighter 等可见证据支持“仍存在类别冲突”的窄结论；072 文件夹样结构及 144 装饰玩偶也可解释跨次判断变化。抽查没有用于重标 152 项、修改原标签或计算第二套准确率，不构成人类评审。

## F. 评价类型：PASS，分类为 synthetic_proxy

主语义初筛按本技能分类映射为 `synthetic_proxy`，细分为 `assistant_visual_initialization_screening`：参考判断由模型助手产生，并已明确标注局限。它不是 `human_eval`，也不是语义类别的 `real_gt`。数据集框只用于定位首帧目标，不改变这一类型。

像素、哈希、文件存在和计数检查属于确定性产物一致性验证；预处理 roundtrip 属于无类别真值的实现一致性诊断。它们不能给语义 proxy 加上真实准确率的含义。机器报告单独记录这些类型，避免把数据来源和评价类型混为一谈。

## 补充观察：输入预处理与模型来源（不覆盖主审查）

补充 41 项本地核验全部通过。`A/audit_processor.py:66-74` 的逆变换正确：前向排列为 `[T,Hgroup,Wgroup,Hmerge,Wmerge,C,t,Hpatch,Wpatch]`，按 `(0,6,5,1,3,7,2,4,8)` 恢复 `[T,t,C,Hgroup,Hmerge,Hpatch,Wgroup,Wmerge,Wpatch]` 后合并空间维；与封存 `E/output/sources/image_processor.py:274-299` 的前向 reshape/transpose 一致。反归一化采用 `(normalized*std+mean)/rescale_factor`，对应前向 `:263-269`。token 公式 `T*H*W/merge_size²` 与封存 `processor.py:160-188` 一致。

逐条核对 152 项 sequence/category/split/image SHA、裁剪 RGB 摘要、历史 grid、输入形状、图像 token 公式与有限且非恒定的记录统计量，并从本地原裁剪按封存 `vision_process.py:60-86,125-144` 的缩放规则独立复算全部 resized RGB 字节摘要，均匹配。八组保存的 crop/reconstructed PNG 的像素摘要也匹配。21 项输出 manifest、证据包归档内文件与解压文件逐字节一致。run.exit 为 0，run.log 的完整 result 对象与 result.json 一致。

独立汇总重现：image tokens 为 128–180；原裁剪短边分位数为 `[16,38.75,57,98.75,349]`；记录的最大反归一化浮点误差为 `4.57763671875e-05`。8 项展示 ID 与计划固定集合相符。`A/M87_INPUT_RECONSTRUCTION.md:15,21-27` 已正确区分缩放后 RGB、原始未缩放裁剪、当前重建与历史运行，也明确八项为事后机制示例而非无偏质量抽样（计划 `:5`）。

此次复核没有运行完整 AutoProcessor/tokenizer，也没有加载模型；现有证据没有保存可供再次逐字节检查的 `pixel_values` 或 `input_ids` 张量本体，只有摘要。152 项有限性、完整 token 化及时间副本断言的原执行证据是源码、每项记录、退出码和日志；本审查没有把这些写成第二次完整环境重跑。当前可重复的 RGB 和排列一致性不能证明历史张量完全相同、历史库环境未变、模型理解正确，或 caption 错误根因已找到。原附录 `:23-25` 已明确这些边界。

模型来源附录仅做本地元数据对照：`official_model_tree.json` 与 `model_source_result.json` 的 14 条路径、大小及 Git/LFS 摘要逐项一致，总大小 7,520,919,614 字节；8 个已有冻结模型文件 SHA 及 pinned manifest SHA 与 caption spec 一致；元数据、核验脚本、结果报告的引用 SHA 一致。`verify_model_source.py:13-25` 对 LFS 文件用 SHA256，对其余文件用包含 `blob <bytes>\0` 头的 Git blob SHA1，算法符合其比较对象。审查者没有重新联网获取官方元数据，也没有远端重哈希权重，因此这是对已存回执和元数据一致性的确认，不是假称亲自读取 7.52 GB 远端模型。`MODEL_SOURCE_VERIFICATION.md:7` 的“当前文件来源不证明历史生成和语义正确性”限定应继续保留。

## 主张影响与行动边界

- 支持：152 项完整初筛记录、当前计数和转移矩阵、当前内容与引用哈希一致、观察中仍存在明显错词。
- 支持但需限定：匿名展示字段被隐藏；模型助手实际知识隔离或独立盲评未成立。封存内容可核验，历史先后只有自述及文件时间相容证据。
- 不支持：真实类别准确率、显著性、跟踪收益、M87 最终性能、去掉整帧的独立因果效应，以及历史输入链“完全无误”。原报告没有将这些作为已完成结果。
- 现有范围内无需回写标签、改 bank、改训练或改评测。若以后提出准确率或因果主张，需另立协议取得独立语义真值和相应对照，保留本轮原始分歧。

复核只在 `W/screening_review` 写入审查脚本、报告、JSON 和 trace；未修改既有记录、标签、bank、训练或报告。全部主审查输入在检查前后 SHA 不变。没有 SSH、凭据访问、后续帧读取或新的训练性能判断。

复核产物：`EXPERIMENT_AUDIT.json`、`deterministic_checks.json`、`input_supplement_checks.json`、`independent_checks.py`、`input_supplement_checks.py` 以及完整审查回复/trace。确定性检查可重复运行；本语义审查结论仍为同家族 provisional。
