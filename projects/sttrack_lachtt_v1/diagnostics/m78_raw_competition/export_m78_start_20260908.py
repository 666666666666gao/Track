from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';O=R/'launch_snapshot'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists()
f=read(R/'frozen.json');t=read(R/'training_spec.json');c=read(R/'causal_check.json')
assert sha(R/'training_spec.json')==f['training_spec_sha256']
assert sha(R/'recursive_spec.json')==f['recursive_spec_sha256']
assert (R/'causal_check.exit').read_text().strip()=='0'
assert (R/'launch.json').exists()
launch=read(R/'launch.json');assert launch['seed']==2027
O.mkdir()
files={n:R/n for n in ['spec.json','training_spec.json','recursive_spec.json','prepared_training_spec.json','prepared_recursive_spec.json','frozen.json','causal_check.json','causal_check.exit','launch.json','EXPERIMENT_PLAN.md','integration.json','data_inventory.json','train_causal.py','run_recursive.py','run_pair.sh','window_competition.py','loss_change.diff','check_causal.py','causal_training.py','support_loss.py','recursive_metric.py']}
files['prepare_m78_20260908.py']=B/'prepare_m78_20260908.py'
files['export_m78_start_20260908.py']=Path(__file__)
tracker='''# M78 launch snapshot

Seed2027 RawCategory / RawEmpty paired causal training started after native-parity and gradient checks. Detached queue trains both arms, completes development22, then computes metrics. This is a launch snapshot, not completed training or new benchmark performance. Same-head Empty/Swapped diagnostics remain required after development, regardless of primary gate status. No automatic public evaluation.

All 160 inference sources match the sealed integration manifest. Training-only competition uses raw scores; inference retains Hann and the native template rule. Synthetic loss is exactly invariant to replacing the supplied Hann with all ones; same-instance negative exclusion and detached geometry checks pass. Zero-residual public states match native; both short optimization checks keep the base frozen. Smoke weights are not saved.

Prospective ten gates compare native and this pair's Empty only; historical protection outcomes are descriptive. M73 and M77 remain sealed reference results and retain their original conclusions. The requested independent reviewer was unavailable; these executable checks do not constitute an independent model review PASS.
'''
(O/'EXPERIMENT_TRACKER.md').write_text(tracker);files['EXPERIMENT_TRACKER.md']=O/'EXPERIMENT_TRACKER.md'
note=f'''

## 5.151 M78启动：单seed原始响应竞争对照，保留原Hann推理

用户最新分析引用的fa92bef仍把M77视为运行中；§5.149—5.150已记录其完成态负结果：M77类别训练开发IoU0.658379/H10=80，另行训练Empty为0.699519/70；固定类别权重输入空文本则为0.761539/59，原类别的内容条件0/8。M77不晋升，其失败结论和原分析错误、独立恢复记录全部保留。不能继续将它描述为尚无结果，也不能把同权重空词收益写成语言贡献。

按用户提出的原始竞争／最终竞争对照，M78只训练RawCategory、RawEmpty两组，固定seed2027。复用M73无新增竞争、M77 Hann竞争的封存结果，不重训这些对照、不增加seed。新损失使用log(raw score)，困难负位置也按raw排序；round正位置、解码中心在GT外且IoU≤0.1的detach负例筛选、最多9个负例、权重1均不变。由于排序依据改变，入选困难负例集合可能改变，不能宣称负例完全相同。

原focal＋2GIoU＋5L1、289154参数Null适配器、冻结STTrack及Center Head、5槽类别／空词协议、130条拟合序列和顺序、初始化张量、优化器均沿用同预算设定。每组计划186694次因果跟踪、5798次优化；推理继续使用原Hann选峰、size/offset、factor4、query及每50步score>0.75的模板规则。没有在线Qwen、新记忆、重检测、自适应Hann或其他新增模块。

新实验在任何训练结果产生前固定10项条件：类别按帧均值至少比原生高0.002、比同预算Empty高0.001；宏平均不低于两者，低重叠帧及H10不多于两者，并保护两者各自零H10序列。不再累计M65/M73等历史版本的全部成功序列为硬门，历史差异仍完整报告。此规则只适用于M78，绝不回改M73/M77失败结论。

两组开发22完成后，固定类别最终权重运行空词／既有替换类别内容诊断，不论开发门是否通过；统计应使用已修复的rows接口，不重复M77的bbox数组接口错误。原类别相对两种内容的按帧／宏平均、低重叠帧、H10共8项条件用于内容判断，不能覆盖主门失败。仅使用最终权重，不从中途checkpoint择优；所有开发集都是已重复使用的DepthTrack Train开发数据，不是正式外部成绩。

预检已完成：零残差情况下bbox、score、query、模板更新与原生完全一致；两组各96帧、3步短优化，底座保持冻结，未保存smoke权重。合成检查保留正确位置增分、错误峰降分、同实例假负例排除、回归mask无梯度，并验证raw损失不受传入Hann替换为全1影响。预检收据SHA`{sha(R/'causal_check.json')}`。这些是可执行契约检查，不是独立模型审阅PASS；用户指定gpt-6-astra/max审阅此前因不可用未获得完成态。

启动时间UTC `{launch['observed_utc']}`，双GPU独立进程运行，日志和退出码保存在`{R}`；训练后队列自动完成两组开发递归与统计。预计训练约4—4.5小时、开发约35分钟，按进度估计在接近完成时查看，必要轮询间隔240秒。启动前剩余磁盘约{launch['free_bytes']/1024**3:.2f}GiB，不下载新底座或文本模型，两个Qwen均保留。当前只有启动证据，没有M78完成指标，没有新三数据集结果，目标尚未完成。
'''
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
old=master.read_bytes();assert sha(master)=='5e2a1fa86495daef88732e1910a57be52914e6a35f78d8a35d11f3ce3b06601b'
assert '## 5.151 '.encode() not in old
(O/'master_append.md').write_bytes(note.encode());files['master_append.md']=O/'master_append.md'
master.write_bytes(old+note.encode())
manifest=[]
for n,p in files.items():
    dest=O/n
    if p!=dest:shutil.copyfile(p,dest)
    manifest.append(dict(path=n,bytes=dest.stat().st_size,sha256=sha(dest)))
(O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=B/'m78_launch_evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(O.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,manifest_files=len(manifest))))
