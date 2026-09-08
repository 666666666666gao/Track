from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';O=R/'development_audit_prepared_publication'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=master.read_bytes()
assert sha(master)=='3309eb325773c1e74bc154c50c7506858f4b0ba853233ed78b5db49536ed009c'
assert b'\n## 5.166 ' not in old and not O.exists()
report=read(R/'development_auditor_preparation.json')
assert report['status']=='auditor_preparation_validated_without_M81_evaluation'
assert report['source_sha256']==sha(B/'audit_m81_development_complete_20260908.py')=='d6bf14dad0809ff16961cfa9ea848785575beae0e39f31f91d304d2c96698975'
assert not (R/'result.json').exists() and not (R/'saved_development_audit.json').exists()
workers=[]
for pid,arm in [(130723,'category'),(130724,'empty')]:
    p=Path('/proc')/str(pid);assert p.exists() and (p/'cwd').resolve()==R
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();assert 'train_causal.py --arm '+arm in cmd
    workers.append(dict(pid=pid,arm=arm,command=cmd))
status=dict(observed_utc=datetime.now(timezone.utc).isoformat(),training_workers=workers,training_spec_sha256=sha(R/'training_spec.json'),evaluation_spec_sha256=sha(R/'evaluation_spec.json'),free_disk_bytes=shutil.disk_usage(R).free)
assert status['training_spec_sha256']=='c62574e32c051f183f562a3417c5e2ed27ed28ef6e3101dbf746de8bdf40d06f'
assert status['evaluation_spec_sha256']=='dee5f2d118a3fca1f41724104f2c39c2aba2e48d9f5430751d0a3a44777627c8'
note='''

## 5.166 M81完成态评测独立核验准备

M81训练继续运行，未改动§5.162冻结的训练、推理、评测或判定规则。本节只追加独立核验工具准备，不记录新的模型性能或重复已有架构与实验计划。

核验器audit_m81_development_complete_20260908.py将等待两组训练、九组评测及analysis/controller全部正常结束，先核对最终head、文字bank、冻结spec、全部receipt和轨迹SHA，再读取开发GT进行独立标量复算。检查范围为九组输出、458个episode、298430个位置，每组33108次跟踪调用和28897个有效非初始化位置。

首帧协议四组各22条完整episode；多起点协议五组各74条episode。独立核验逐序列检查每个非初始化物理帧恰好覆盖一次，并在共同episode边界重新计算H10，再按序列汇总，避免重复计数、漏帧或将多起点H10误写成VOT ROB。核验还将复算20项主条件、8项内容条件、4项初始化覆盖条件及成功序列保护名单。

2026-09-08 14:48:39 UTC准备验证完成：复用M80已经封存的66条真实轨迹、99390个位置，验证独立标量IoU/H10函数与既有完成态逐序列结果一致；M81开发74个episode清单的非初始化帧覆盖检查通过。该验证没有调用GPU、没有重新跟踪、没有运行M81评测，也没有获得M81性能指标，不能算M81完成态核验PASS。脚本仅在全部九组轨迹封存后正式执行。

脚本SHA256为`d6bf14dad0809ff16961cfa9ea848785575beae0e39f31f91d304d2c96698975`。源码、准备验证报告及绑定状态发布于diagnostics/m81_multistart/development_audit_prepared/；实际训练保存产物核验仍由§5.163的CPU观察器负责。确定性检查不等于gpt-6-astra/max独立模型审阅通过。
'''
O.mkdir();F=O/'files';F.mkdir()
for src,name in [(B/'audit_m81_development_complete_20260908.py','audit_m81_development_complete_20260908.py'),(R/'development_auditor_preparation.json','development_auditor_preparation.json'),(Path(__file__),Path(__file__).name)]:shutil.copyfile(src,F/name)
(F/'verified_status.json').write_text(json.dumps(status,indent=2)+'\n');(F/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())]
(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(F.iterdir()):tar.add(p,arcname=p.name)
master.write_bytes(old+note.encode('utf-8'))
record=dict(archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
