from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m71_equal_budget_routing_20260907'
M67=BASE/'sttrack_m67_supervised_semantic_support_20260907';QUEUE=BASE/'sttrack_post_m67_diagnostic_queue_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
s=read(ROOT/'spec.json');c=read(ROOT/'cpu_check_result.json');geometry=read(ROOT/'geometry_accounting.json')
assert s['source_sha256']==sha(BASE/'m71_equal_budget_routing_20260907.py')==c['source_sha256']
assert c['spec_sha256']==geometry['spec_sha256']==sha(ROOT/'spec.json') and c['geometry_sha256']==sha(ROOT/'geometry_accounting.json')
assert not (ROOT/'result.json').exists()
assert sha(BASE/'m70_candidate_capacity_20260907.py')=='e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977'
assert sha(BASE/'post_m67_diagnostic_queue_20260907.py')=='4a4bc5debaf8581ede1cec7bd1c2020b3361ee5963509b7c7b18679fab3688cb'
out=ROOT/'publication_evidence';out.mkdir()
for n in ['m71_equal_budget_routing_20260907.py','check_m71_prepared_20260907.py']:shutil.copyfile(BASE/n,out/n)
shutil.copyfile(__file__,out/Path(__file__).name)
for n in ['spec.json','cpu_check_result.json']:shutil.copyfile(ROOT/n,out/n)
(out/'geometry_accounting_summary.json').write_text(json.dumps(dict(status=geometry['status'],spec_sha256=geometry['spec_sha256'],
    source_full_geometry_sha256=sha(ROOT/'geometry_accounting.json'),summaries=geometry['summaries'],model_outputs_read=False,new_tracking_calls=0),indent=2)+'\n')
revision=dict(**s['report_schema_revision'],previous_source_sha256=sha(ROOT/'preparation_v1/m71_equal_budget_routing.py'),
    original_and_current_geometry_events_identical=read(ROOT/'preparation_v1/geometry_accounting.json')['events']==geometry['events'])
assert revision['original_and_current_geometry_events_identical']
(out/'prepublication_revision.json').write_text(json.dumps(revision,indent=2)+'\n')
training={}
for name,pid in [('control',465923),('support',465924)]:
    p=Path('/proc')/str(pid);assert p.exists() and (p/'cwd').resolve()==M67
    argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    assert argv==['/root/autodl-tmp/envs/sttrack/bin/python','-u','train_causal.py','--arm',name]
    rows=[json.loads(x) for x in (M67/'training'/name/'sequence_log.jsonl').read_text().splitlines()]
    training[name]=dict(pid=pid,completed_sequences=len(rows),total_track_calls=rows[-1]['total_track_calls'],total_optimizer_steps=rows[-1]['total_optimizer_steps'])
i=read(QUEUE/'running_identity.json')['identity'];p=Path('/proc')/str(i['pid'])
assert p.exists() and (p/'cwd').resolve()==QUEUE and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==i['start_ticks']
snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),M67_training=training,queue_pid=i['pid'],queue_latest=read(QUEUE/'latest.json'),
    M70_sources_unchanged=True,queue_sources_unchanged=True,disk_free_bytes=shutil.disk_usage(ROOT).free,new_GPU_jobs_started=0,new_formal_metrics=False)
(out/'live_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,check_sha256=sha(ROOT/'cpu_check_result.json'),snapshot=snapshot),indent=2))
