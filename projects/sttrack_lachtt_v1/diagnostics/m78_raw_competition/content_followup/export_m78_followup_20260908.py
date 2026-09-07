from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';O=R/'content_followup';D=O/'prepared_snapshot'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not D.exists();s=read(O/'spec.json');a=read(O/'launch.json')
assert sha(B/'m78_content_followup_20260908.py')==s['source_sha256']
assert sha(B/'audit_m78_completed_20260908.py')==s['auditor_sha256']
assert read(O/'preparation.json')['rows_metric_interface_verified']
assert read(O/'stage.json')['stage']=='waiting_for_original_training_and_recursion'
assert not (R/'recursive_result.json').exists() and not (O/'activation.json').exists()
D.mkdir()
files={n:O/n for n in ['spec.json','preparation.json','launch.json','run_controller.sh','controller.started','stage.json']}
for n in ['prepare_m78_followup_20260908.py','audit_m78_completed_20260908.py','m78_content_followup_20260908.py']:
    files[n]=B/n
files['export_m78_followup_20260908.py']=Path(__file__)
note=f'''

## 5.152 M78后续内容队列已准备：统计接口先验验证，等待完整训练与开发结果

M78双GPU训练启动后，两组均已完成首条cube04的2575次跟踪、81次优化，数值有限，尚无最终指标。这些训练日志只证明运行推进，不用于挑选权重或预测开发收益。

独立后续控制器已于UTC `{a['observed_utc']}` 启动，PID `{a['controller_pid']}`，按240秒检查原训练／开发控制器的PID、启动标识、参数及退出码。它不修改正在运行的冻结训练配置，等待两组训练、两组开发递归和统计全部成功退出后才激活。

激活前审计最终／latest权重、优化器、130序列覆盖、底座冻结记录、160源码及文本银行SHA，并用独立标量实现复算开发指标与本轮10项门。随后固定Category最终权重，先在bag05、cup08、mobilephone02各102帧检查与封存类别轨迹精确一致，再分别运行同权重Empty、Swapped完整开发22。新增完整调用计划66216次，prefix303次，无新优化或seed，不因主门失败跳过内容诊断。

M77曾将bbox数组传给要求逐帧记录的独立统计函数；M78新源码已使用`scalar(rows,gt)`，准备阶段用封存M77 bag05真实轨迹与GT验证两种指标实现一致。准备脚本中的两处具体路径错误在生成任何后续spec、启动后续进程前修正，未改训练源码、权重或数据。后续spec SHA `{sha(O/'spec.json')}`；它在M78最终权重和开发结果出现前封存。

当前状态为等待训练与开发，不是内容测试完成；没有激活权重，没有新的外部评测。用户要求仅seed2027、保留两个Qwen继续有效。后续队列不会自动触发公开数据集评测，内容结果不能反向覆盖本轮主门失败，也不能替代独立模型审阅。
'''
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
old=master.read_bytes();assert sha(master)=='845566830d6d25e00c2c5b85a64b6c18aecd5b942d3f5fe3ae9975a7172c73b7'
assert '## 5.152 '.encode() not in old
(D/'master_append.md').write_bytes(note.encode());files['master_append.md']=D/'master_append.md'
master.write_bytes(old+note.encode())
manifest=[]
for n,p in files.items():
    q=D/n
    if p!=q:shutil.copyfile(p,q)
    manifest.append(dict(path=n,bytes=q.stat().st_size,sha256=sha(q)))
(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=B/'m78_followup_evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(D.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),master_sha256=sha(master),manifest_files=len(manifest),master_bytes=master.stat().st_size)))
