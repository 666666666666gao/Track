from pathlib import Path
import hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_post_m67_diagnostic_queue_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((ROOT/'spec.json').read_text());r=json.loads((ROOT/'launch_verification.json').read_text())
assert s['source_sha256']==r['source_sha256']==sha(BASE/'post_m67_diagnostic_queue_20260907.py')
assert r['spec_sha256']==sha(ROOT/'spec.json')
i=r['queue_identity']['identity'];p=Path('/proc')/str(i['pid'])
assert p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==i['start_ticks']
assert (p/'cwd').resolve()==ROOT and [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]==i['argv']
assert not (ROOT/'controller.exit').exists()
out=ROOT/'publication_evidence';out.mkdir()
for name in ['spec.json','launch.sh','running_identity.json','launch_verification.json']:shutil.copyfile(ROOT/name,out/name)
shutil.copyfile(ROOT/'events.jsonl',out/'launch_events_snapshot.jsonl')
shutil.copyfile(ROOT/'controller.log',out/'launch_log_snapshot.txt')
shutil.copyfile(BASE/'post_m67_diagnostic_queue_20260907.py',out/'post_m67_diagnostic_queue.py')
shutil.copyfile(__file__,out/'export_post_m67_queue.py')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,launch_verification_sha256=sha(ROOT/'launch_verification.json'),live_pid=i['pid']),indent=2))
