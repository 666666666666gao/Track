from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = PARENT / 'candidate_evaluation'
QUEUE = BASE / 'sttrack_post_m67_diagnostic_queue_20260907'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
spec = read(ROOT / 'spec.json')
for name, digest in spec['source_sha256'].items(): assert sha(BASE / name) == digest
check = read(ROOT / 'source_check_result.json')
assert check['preparation_spec_sha256'] == sha(ROOT / 'spec.json')
assert check['checker_sha256'] == sha(BASE / 'check_m67_candidate_prepared_20260907.py')
assert not (ROOT / 'bundle.json').exists() and not (PARENT / 'recursive_result.json').exists()
assert sha(BASE / 'post_m67_diagnostic_queue_20260907.py') == '4a4bc5debaf8581ede1cec7bd1c2020b3361ee5963509b7c7b18679fab3688cb'
assert sha(QUEUE / 'spec.json') == '5f34ea8f38dfe6f7047d4cf65fba388da076c4c178d0b7e3b3afa3f4c078eddf'
training = {}
for name, pid in [('control', 465923), ('support', 465924)]:
    p = Path('/proc') / str(pid); assert p.exists() and (p / 'cwd').resolve() == PARENT
    rows = [json.loads(x) for x in (PARENT / 'training' / name / 'sequence_log.jsonl').read_text().splitlines()]
    last = rows[-1]
    training[name] = dict(pid=pid, completed_sequences=len(rows), total_track_calls=last['total_track_calls'],
        total_optimizer_steps=last['total_optimizer_steps'], latest_completed_sequence=last['sequence'])
identity = read(QUEUE / 'running_identity.json')['identity']
p = Path('/proc') / str(identity['pid'])
assert p.exists() and (p / 'cwd').resolve() == QUEUE
assert (p / 'stat').read_text().rsplit(')', 1)[1].split()[19] == identity['start_ticks']
assert [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x] == identity['argv']
qwen = {}
for name, count, size in [('Qwen3_8B', 5, 16381516776), ('Qwen2.5-VL-3B-Instruct', 2, 7509337976)]:
    files = list((BASE / 'qwen' / name).glob('*.safetensors'))
    assert len(files) == count and sum(p.stat().st_size for p in files) == size
    qwen[name] = dict(weight_files=count, weight_bytes=size)
snapshot = dict(observed_utc=datetime.now(timezone.utc).isoformat(), M67_training=training, post_M67_queue_pid=identity['pid'],
    queue_latest=read(QUEUE / 'latest.json'), queue_sources_unchanged=True, disk_free_bytes=shutil.disk_usage(BASE).free,
    retained_Qwen=qwen, new_gpu_jobs_started=0, new_formal_metrics=False, candidate_bundle_created=False)
(ROOT / 'live_preparation_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
out = ROOT / 'publication_evidence'; out.mkdir()
for name in spec['source_sha256']: shutil.copyfile(BASE / name, out / name)
for name in ['spec.json', 'cpu_preparation_result.json', 'source_check_result.json', 'live_preparation_snapshot.json']:
    shutil.copyfile(ROOT / name, out / name)
for path in [BASE / 'check_m67_candidate_prepared_20260907.py', Path(__file__)]: shutil.copyfile(path, out / path.name)
(out / 'evidence_manifest.json').write_text(json.dumps([dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for path in sorted(out.iterdir()): t.add(path, arcname=path.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, snapshot=snapshot,
    preparation_spec_sha256=sha(ROOT / 'spec.json'), source_check_sha256=sha(ROOT / 'source_check_result.json')), indent=2))
