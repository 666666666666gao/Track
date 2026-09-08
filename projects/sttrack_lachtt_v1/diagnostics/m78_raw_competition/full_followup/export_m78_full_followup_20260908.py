from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile

B=Path('/root/autodl-tmp');E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation';Q=E/'full_followup';O=Q/'published_start'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not O.exists()
s=read(Q/'spec.json');started=read(Q/'controller.started.json');stage=read(Q/'stage.json')
assert stage['stage']=='waiting_for_complete_low22' and not (Q/'controller.exit').exists()
assert not (E/'full_evaluation').exists() and not (E/'low22_controller.exit').exists()
pid=started['pid'];p=Path('/proc')/str(pid);st=(p/'stat').read_text().split()
assert st[2]!='Z' and b'/root/autodl-tmp/m78_low22_full_followup_20260908.py' in (p/'cmdline').read_bytes().split(b'\x00')
parent=Path('/proc')/str(s['parent_pid']);pst=(parent/'stat').read_text().split()
assert pst[21]==s['parent_start_ticks'] and pst[2]!='Z'
assert sha(parent/'cmdline')==s['parent_command_sha256']
assert sha(B/'m78_low22_full_followup_20260908.py')==s['source_sha256']
assert sha(B/'m78_low22_full_followup_20260908.sh')==s['runner_sha256']
times=list((E/'low22_run').glob('shard-*/results/**/*_time.value'))
snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),queue_pid=pid,queue_start_ticks=st[21],queue_state=st[2],parent_pid=s['parent_pid'],parent_start_ticks=pst[21],
    stage=stage,completed_low22_anchors=len(times),completed_low22_frame_positions=sum(len(t.read_text().splitlines()) for t in times),
    total_anchors=303,total_planned_frame_positions=220483,full_evaluation_started=False,disk_free_bytes=shutil.disk_usage(E).free)
O.mkdir();files={}
for name in ['spec.json','controller.started.json']:files[name]=Q/name
for name in ['m78_low22_full_followup_20260908.py','m78_low22_full_followup_20260908.sh','prepare_m78_full_followup_20260908.py']:files[name]=B/name
files[Path(__file__).name]=Path(__file__)
(O/'process_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n');files['process_snapshot.json']=O/'process_snapshot.json'
append_text=f'''

## 5.155 M78低22到完整三数据集的条件队列已启动

在§5.154的开发10/10、同权重内容8/8与真实入口一致性通过后，为避免长评测结束后人工排队停顿，新增独立条件控制器；没有修改M78权重、训练、推理、文字协议或任何已有门槛。仅seed2027，后续评测不产生新seed或选择新checkpoint。

控制器于UTC{started['observed_utc']}启动，进程PID{pid}；跟随已绑定的低22主脚本PID{s['parent_pid']}并核对启动标识和命令，每240秒检查一次退出状态。本快照低22已封存{snapshot['completed_low22_anchors']}/303个anchor、{snapshot['completed_low22_frame_positions']}/220483个计划帧位置；尚无完成指标。当前控制器处于waiting_for_complete_low22，完整三数据集尚未开始。

若低22执行失败，保存失败并终止本次后续队列；若执行完成但任一冻结指标/保护条件未通过，保存完整低22结果并结束队列，不调用全量脚本、不改阈值、不换文字或权重。只有低22全部条件通过，才调用§5.153已经冻结的m78_full_evaluation_20260908.sh：依次生成DepthTrack Test、CDTB的因果初始化文本并评测，随后补充剩余1462个VOT初始化文本、保留既有303个记录，验证完整127序列1765个anchor，最后进行同bundle的保存输出核验与数值目标检查。文本生成只读取各次初始化图像与初始化框。

后续脚本SHA256仍为`{s['full_script_sha256']}`，bundle仍为`{s['bundle_sha256']}`；没有为队列更换模型接口。队列spec SHA256为`{sha(Q/'spec.json')}`，控制器源码为`{s['source_sha256']}`。准备期发现全量脚本哈希保存在prepared_entry_and_full.json的source_files，而非入口spec的source_sha256；已在启动队列前修正索引，已冻结评测源码字节保持不变。检查包括真实进程标识、冻结源码/配置/模型绑定、Python语法和shell语法；不是独立模型审阅PASS。

队列目录：`{Q}`；阶段见stage.json，日志见controller.log/full_execution.log，退出状态见controller.exit/full_execution.exit。即使全量脚本完成，也需读取same_bundle_verification.json，核对三个数据集及目标、补齐交接和发布后才可完成项目目标，控制器不会自动宣称目标达成。

本节及启动证据发布到`projects/sttrack_lachtt_v1/diagnostics/m78_raw_competition/full_followup/`。当前低22四个分片是工作划分，不是多seed。按首批实际耗时预计整轮低22约2.5—3小时，在预计收尾前检查；当前磁盘{snapshot['disk_free_bytes']}字节可用，两份Qwen均保留。
'''
(O/'handoff_append.md').write_bytes(append_text.encode('utf-8'));files['handoff_append.md']=O/'handoff_append.md'
published=O/'files';published.mkdir();manifest=[]
for name,src in files.items():
    dst=published/name;shutil.copyfile(src,dst);manifest.append(dict(path=name,bytes=dst.stat().st_size,sha256=sha(dst)))
(published/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for item in sorted(published.iterdir()):
        assert item.is_file() and item.stat().st_size<3000000
        tar.add(item,arcname=item.name)
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');previous_bytes=master.read_bytes()
assert sha(master)=='c1245c8aa6629dbf46613fbfa7ccdccf577bf50c8ab8ce8ffe8cabd26d526878' and b'\n## 5.155 ' not in previous_bytes
master.write_bytes(previous_bytes+append_text.encode('utf-8'))
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(files))
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
