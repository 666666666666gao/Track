from pathlib import Path
import hashlib,json,shutil,tarfile
R=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909');O=R/'auditor_reference_publication';F=O/'files'
M=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=M.read_bytes();assert sha(M)=='0956363e2b3dcb9b79b43f1760c73aa20d7ddab7bf52b2105b5488434332982f' and not O.exists()
record=json.loads((R/'auditor_reference_validation.json').read_text())
assert record['status']=='scalar_auditor_verified_against_sealed_M78' and record['positions']==66260
assert record['auditor_sha256']==sha(R.parent/'audit_m82_completed_20260909.py')
assert record['validation_source_sha256']==sha(R.parent/'validate_m82_scalar_auditor_20260909.py')
note='''

M82复算程序补充验证（北京时间2026-09-09 03:09）：在M82两组训练仍运行期间，独立标量统计函数对已封存M78 Category/Empty的44条轨迹、66260个位置进行了复算，逐序列有效帧、IoU、低重叠帧、H10及无效GT数与封存结果一致。本次未读取M82当前开发预测，未产生M82性能结论；正式完成后的训练产物与四组开发结果核验仍在排队。验证源码及receipt发布于diagnostics/m82_native_preservation/auditor_reference_check/。\n'''
F.mkdir(parents=True)
for src in [R/'auditor_reference_validation.json',R.parent/'validate_m82_scalar_auditor_20260909.py',Path(__file__)]:shutil.copyfile(src,F/src.name)
(F/'handoff_append.md').write_text(note)
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(F.iterdir())]
(F/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(F.iterdir()):t.add(p,arcname=p.name)
M.write_bytes(old+note.encode('utf-8'))
receipt=dict(archive_sha256=sha(archive),master_sha256=sha(M),master_bytes=M.stat().st_size,files=len(manifest))
(O/'publication_record.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
