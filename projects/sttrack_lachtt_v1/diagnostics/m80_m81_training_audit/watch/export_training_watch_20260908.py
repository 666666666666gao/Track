from pathlib import Path
import json,hashlib,shutil,tarfile
B=Path('/root/autodl-tmp');O=B/'sttrack_training_audit_watch_20260908';F=O/'publication';F.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=json.loads((O/'state.json').read_text());proc=Path('/proc')/str(state['pid'])
assert proc.exists() and str(B/'watch_training_audits_20260908.py').encode() in (proc/'cmdline').read_bytes().split(b'\0')
assert state['status']=='waiting_near_M80_estimated_completion' and not (O/'watch.exit').exists()
for source in [B/'watch_training_audits_20260908.py',O/'run.sh',O/'state.json',Path(__file__)]:shutil.copyfile(source,F/source.name)
note='''

§5.163补充：训练完成态核验已启动独立CPU观察进程（screen sttrack_audit_watch_20260908），实际/proc命令行已核实。M80最早于2026-09-08 10:55 UTC开始检查训练终态，距当前预计结束约15分钟；届时每240秒读取退出记录，正常结束才执行--complete。随后等待既定M81启动，约3小时后开始同间隔检查两臂训练终态。观察器不参与GPU调度，不修改任何训练/评测脚本；训练退出异常、进程缺失或核验断言失败时保留日志并停止，未实现自动重启。两项完整训练核验仍未执行，观察进程启动不等于核验通过。源码、初始进程记录与启动脚本封存在diagnostics/m80_m81_training_audit/watch/。
'''
(F/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())];(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(F.iterdir()):t.add(p,arcname=p.name)
m=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md');old=m.read_bytes()
assert sha(m)=='20f9ac3076adf6839369ac3f949c56e1e6ea870ac996d0072b507e2584c08024'
m.write_bytes(old+note.encode())
record=dict(archive_sha256=sha(archive),master_sha256=sha(m),master_bytes=m.stat().st_size,files=len(manifest),observer_pid=state['pid'])
(O/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
