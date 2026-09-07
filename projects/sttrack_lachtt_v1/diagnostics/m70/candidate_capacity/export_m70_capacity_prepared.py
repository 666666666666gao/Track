from pathlib import Path
import hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m70_recovery_window_inventory_20260907/candidate_capacity'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (ROOT/'preparation_check.exit').read_text().strip()=='0'
r=json.loads((ROOT/'preparation_check.json').read_text())
assert r['status']=='M70_cross_region_capacity_preparation_CPU_checked'
assert r['source_sha256']==sha(BASE/'m70_candidate_capacity_20260907.py')
assert r['spec_sha256']==sha(ROOT/'spec.json') and r['queue_sha256']==sha(ROOT/'run_capacity.sh')
assert r['new_tracking_calls']==0 and not r['GPU_shadow_state_parity_verified']
for name in ['smoke','shard0','shard1']:assert not (ROOT/name).exists()
out=ROOT/'publication_evidence';out.mkdir()
for name in ['spec.json','run_capacity.sh','preparation_check.json','preparation_check.log','preparation_check.exit']:
    shutil.copyfile(ROOT/name,out/name)
for source,target in [('m70_candidate_capacity_20260907.py','m70_candidate_capacity.py'),('check_m70_capacity_prepared_20260907.py','check_m70_capacity_prepared.py')]:
    shutil.copyfile(BASE/source,out/target)
shutil.copyfile(__file__,out/'export_m70_capacity_prepared.py')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,check_sha256=sha(ROOT/'preparation_check.json')),indent=2))
