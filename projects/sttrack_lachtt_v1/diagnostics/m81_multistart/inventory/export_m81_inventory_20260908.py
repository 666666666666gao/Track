"""Publish M81 Train initialization inventory and verified local caption startup."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';O=R/'inventory_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not O.exists()
s=read(R/'inventory_spec.json');c=read(R/'caption_preparation.json');launch=read(R/'caption_launch.json')
assert sha(R/'inventory_spec.json')==c['inventory_spec_sha256']
assert sha(R/'captions/plan.json')==c['caption_plan_sha256']==launch['caption_plan_sha256']
for split in ['fit','development']:assert sha(R/(split+'_initializations.json'))==s[split+'_manifest_sha256']
p=Path('/proc/121529');assert p.is_dir()
args=(p/'cmdline').read_bytes().split(b'\0');assert str(R/'captions/plan.json').encode() in args and b'generate' in args
assert b'CUDA_VISIBLE_DEVICES=1' in (p/'environ').read_bytes().split(b'\0')
assert not (R/'caption_generate.exit').exists()
records=[json.loads(x) for x in (R/'captions/records.jsonl').read_text().splitlines()]
status=dict(status='verified_caption_generation_live',observed_utc=datetime.now(timezone.utc).isoformat(),pid=121529,gpu=1,completed_captions=len(records),total_captions=348,training_started=False,latest_elapsed_seconds=records[-1]['elapsed_seconds'],free_disk_bytes=shutil.disk_usage(R).free)
note=f'''

## 5.161 M81初始化覆盖准备：同一训练转移预算，426个拟合episode

根据用户关于初始化覆盖的最新优先级，M81已完成DepthTrack Train内固定多起点清单和本地描述生成准备，尚未启动跟踪网络训练。M80仍按§5.160原方案运行，M81不继承文本dropout，也不在M80中途修改训练条件。

| 清单 | 原序列数 | 固定episode数 | 新增初始化数 | 跟踪调用 | 含有效监督的32帧窗口 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 拟合Train | 130 | 426 | 296 | 186694 | 5798 |
| 开发Train | 22 | 74 | 52 | 33108 | 1009 |

分段规则在任何新跟踪结果之前确定：首帧保留，之后每约512个转移帧选一个新起点；若该位置GT非有限、宽高非正，或目标中心在图像外，则以32帧步长向后寻找。所有非首帧边界都对齐原32帧累积窗口。相邻episode共享边界图像作为前一段末帧和后一段初始化，但每个原始转移帧只计算一次跟踪与损失；逐序列验证覆盖无遗漏、无重复，原梯度累积区间逐项相同。拟合最长episode608次跟踪，最短2次；开发最长672、最短5次，尾段不为凑长度而重复采样。

这不是跟丢后的自适应GT重置：初始化时刻在训练前由清单固定，段内仍必须完整使用自身预测crop/query/模板。它比单起点使用更多初始化GT框和额外初始化前向，因此只能称相同跟踪调用、相同优化窗口预算，不能说计算量或监督信息完全相同。合法框及中心在画面内不保证目标清晰可见，也不保证自动caption正确。

保持M78网络与目标：冻结STTrack和Center Head，训练289154参数密集语义适配器，原始响应竞争损失、优化器、原生Hann和模板规则均不变；固定seed2027。计划训练多起点Category与多起点Empty，两组使用同一初始化清单、数据顺序、预算和初始tensor。首轮不加入反向遍历、跨初始化一致性损失、新记忆或空间背景模块。封存M78单起点Category/Empty作为参照，具体训练代码和晋升条件还未冻结，不把清单准备说成训练已经开始。

描述协议：原130个拟合/22个开发首帧继续复用封存bank；新增296＋52＝348个起点只读取各自初始化RGB图像、红框标记和目标crop，沿用已有Qwen2.5-VL-3B生成器与prompt/processor/解码协议，无序列名、类别提示或后续图像。输出原始类别/属性都封存，Category建bank时仅保留类别内容、属性置空，Empty替换所有有效槽；槽数、mask、padding保持配对。开发文字不进入拟合bank。新的起点产生新的初始化描述，因此这是完整初始化协议覆盖实验，不能将全部差异解释为纯视觉重置。

描述任务于UTC{launch['observed_utc']}在空闲GPU1启动；UTC{status['observed_utc']}核实进程121529仍活跃，已生成{len(records)}/348条，未出现退出文件。随后CLIP编码在CPU进行。M80训练进程仍使用GPU0，两项通过独立screen及持久日志运行。两份Qwen都保留，无新模型下载，本次M81未产生训练权重或评测指标。

评测安排：保留常规首帧完整Train开发22，同时在本清单的74个开发episode上比较所有相关模型，包括原生和封存单起点模型。不得把重初始化后的新模型成绩只与未重初始化的原生相比，以免把评测条件变容易的收益误记为模型提升。该开发集仍为反复使用的DepthTrack Train，不是新的独立测试集；VOT起点、图像与GT未用于清单或文字生成。

拟合清单SHA256 `{s['fit_manifest_sha256']}`；开发清单`{s['development_manifest_sha256']}`；描述计划`{c['caption_plan_sha256']}`。源码、完整清单、固定生成协议索引及启动记录发布于projects/sttrack_lachtt_v1/diagnostics/m81_multistart/inventory/。

下一步先完成348条描述与CPU编码并核查bank绑定，再实现两组多起点训练及两种开发协议的配对运行。M81训练使用GPU前要核实M80训练/内容评测的实际进程与占用，不抢占或重启现有任务。没有新的正式三数据集成绩。
'''
O.mkdir();files=O/'files';files.mkdir()
for name in ['inventory_spec.json','fit_initializations.json','development_initializations.json','caption_inputs.json','caption_preparation.json','caption_launch.json','run_captions.sh']:
    shutil.copyfile(R/name,files/name)
shutil.copyfile(R/'captions/plan.json',files/'caption_plan.json')
for name in ['prepare_m81_initialization_inventory_20260908.py','prepare_m81_captions_20260908.py']:
    shutil.copyfile(B/name,files/name)
shutil.copyfile(__file__,files/Path(__file__).name)
write(files/'startup_status.json',status);(files/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files.iterdir())];write(files/'manifest.json',manifest)
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files.iterdir()):tar.add(p,arcname=p.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='3bd4c3bd93d351c6c0c50c5fe4e89c7a9205be7ff880b48889efe5a45ccc6bdd'
assert b'\n## 5.161 ' not in old
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
write(O/'publication_record.json',record);print(json.dumps(dict(publication=record,startup=status)))
