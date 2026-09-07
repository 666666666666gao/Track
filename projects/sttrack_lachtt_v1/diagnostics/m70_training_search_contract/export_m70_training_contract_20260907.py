from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m70_recovery_window_inventory_20260907/training_search_contract'
M67=BASE/'sttrack_m67_supervised_semantic_support_20260907';QUEUE=BASE/'sttrack_post_m67_diagnostic_queue_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
s=read(ROOT/'spec.json');r=read(ROOT/'result.json');v=read(ROOT/'verification.json')
assert r['spec_sha256']==v['spec_sha256']==sha(ROOT/'spec.json')
assert r['source_sha256']==s['source_sha256']==sha(BASE/'m70_training_search_contract_audit_20260907.py')
assert v['audit_result_sha256']==sha(ROOT/'result.json')
assert v['checker_sha256']==sha(BASE/'check_m70_training_search_contract_20260907.py')
assert sha(BASE/'m70_candidate_capacity_20260907.py')=='e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977'
assert sha(BASE/'post_m67_diagnostic_queue_20260907.py')=='4a4bc5debaf8581ede1cec7bd1c2020b3361ee5963509b7c7b18679fab3688cb'
bound=read(ROOT/'decoder_capacity_bounds.json')
assert bound['source_sha256']==sha(BASE/'m70_decoded_box_capacity_bounds_20260907.py')
out=ROOT/'publication_evidence_with_bounds';out.mkdir()
for n in ['m70_training_search_contract_audit_20260907.py','check_m70_training_search_contract_20260907.py','m70_decoded_box_capacity_bounds_20260907.py']:shutil.copyfile(BASE/n,out/n)
shutil.copyfile(__file__,out/Path(__file__).name)
for n in ['spec.json','result.json','verification.json','decoder_capacity_bounds.json']:shutil.copyfile(ROOT/n,out/n)
progress={}
for name,pid in [('control',465923),('support',465924)]:
    p=Path('/proc')/str(pid);assert p.exists() and (p/'cwd').resolve()==M67
    assert [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]==['/root/autodl-tmp/envs/sttrack/bin/python','-u','train_causal.py','--arm',name]
    records=[json.loads(x) for x in (M67/'training'/name/'sequence_log.jsonl').read_text().splitlines()]
    last=records[-1];progress[name]=dict(pid=pid,completed_sequences=len(records),total_track_calls=last['total_track_calls'],optimizer_steps=last['total_optimizer_steps'])
i=read(QUEUE/'running_identity.json')['identity'];p=Path('/proc')/str(i['pid'])
assert p.exists() and (p/'cwd').resolve()==QUEUE and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==i['start_ticks']
snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),M67_training=progress,queue_pid=i['pid'],queue_latest=read(QUEUE/'latest.json'),
    disk_free_bytes=shutil.disk_usage(ROOT).free,new_GPU_jobs_started=0,parent_sources_unchanged=True,new_formal_metrics=False)
(out/'live_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,verification_sha256=sha(ROOT/'verification.json'),snapshot=snapshot),indent=2))
