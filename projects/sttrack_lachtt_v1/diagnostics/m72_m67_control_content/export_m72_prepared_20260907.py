from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m72_m67_control_content_20260907'
M69 = BASE / 'sttrack_m69_m65_content_diagnostic_20260907'
QUEUE = BASE / 'sttrack_post_m67_diagnostic_queue_20260907'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
spec = read(ROOT / 'spec.json')
assert sha(BASE / 'm72_m67_control_content_20260907.py') == spec['source_sha256']
assert sha(ROOT / 'run_controls.sh') == spec['queue_sha256'] and sha(ROOT / 'launch_after_M71.sh') == spec['launch_sha256']
check = read(ROOT / 'cpu_check_result.json')
assert check['checker_sha256'] == sha(BASE / 'check_m72_prepared_20260907.py')
assert check['spec_sha256'] == sha(ROOT / 'spec.json') and not check['full_recursive_parity_verified']
assert sha(BASE / 'post_m67_diagnostic_queue_20260907.py') == '4a4bc5debaf8581ede1cec7bd1c2020b3361ee5963509b7c7b18679fab3688cb'
assert sha(BASE / 'm71_after_diagnostics_20260907.py') == spec['parent_source_sha256']

def live(record):
    folder = Path('/proc') / str(record['pid'])
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z' and fields[19] == record['start_ticks']
    assert str((folder / 'cwd').resolve()) == record['cwd']
    assert [part.decode() for part in (folder / 'cmdline').read_bytes().split(b'\0') if part] == record['argv']

registered = read(ROOT / 'waiting_identity.json')['identity']
live(registered); live(spec['parent_identity'])
assert registered['argv'] == ['/root/autodl-tmp/envs/sttrack/bin/python', '-u', str(BASE / 'm72_m67_control_content_20260907.py'), 'wait_parent']
latest = read(ROOT / 'latest_wait.json')
assert latest['event'] == 'waiting_for_registered_M71_runner' and latest['poll_seconds'] == 240
assert latest['parent_identity'] == spec['parent_identity']
assert not (ROOT / 'wait.exit').exists()
for name in ['prefix', 'empty', 'swapped', 'result.json']: assert not (ROOT / name).exists()
queue_identity = read(QUEUE / 'running_identity.json')['identity']; live(queue_identity)
progress = {}
for arm, pid in [('empty', 476496), ('swapped', 476495)]:
    record = dict(pid=pid, start_ticks='4578841518', cwd=str(M69),
        argv=['/root/autodl-tmp/envs/sttrack/bin/python', '-u', str(BASE / 'm69_m65_content_diagnostic_20260907.py'), arm])
    live(record)
    entries = [json.loads(line) for line in (M69 / (arm + '.log')).read_text().splitlines() if line.startswith('{"sequence":')]
    progress[arm] = dict(identity=record, completed_sequences=len(entries), completed_track_calls=sum(row['frames'] - 1 for row in entries))
qwen = {}
for name, count, total in [('Qwen3_8B', 5, 16381516776), ('Qwen2.5-VL-3B-Instruct', 2, 7509337976)]:
    files = list((BASE / 'qwen' / name).rglob('*.safetensors'))
    size = sum(path.stat().st_size for path in files)
    assert len(files) == count and size == total
    qwen[name] = dict(files=count, bytes=size, preserved=True)
verification = dict(status='M72_live_waiter_verified_after_CPU_checks', observed_utc=datetime.now(timezone.utc).isoformat(),
    exporter_sha256=sha(__file__), source_sha256=spec['source_sha256'], spec_sha256=sha(ROOT / 'spec.json'),
    cpu_check_sha256=sha(ROOT / 'cpu_check_result.json'), waiting_identity=registered, parent_identity=spec['parent_identity'],
    original_queue_identity=queue_identity, original_queue_latest=read(QUEUE / 'latest.json'), M69_progress=progress,
    disk_free_bytes=shutil.disk_usage(BASE).free, Qwen=qwen, M72_GPU_execution_started=False,
    new_GPU_jobs_started=0, original_queues_modified=False, new_formal_metrics=False, independent_model_review_pass=False)
(ROOT / 'launch_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
out = ROOT / 'publication_evidence'; out.mkdir()
for name in ['spec.json', 'preparation_result.json', 'cpu_check_result.json', 'premature_eligibility_rejection.log',
    'run_controls.sh', 'launch_after_M71.sh', 'waiting_identity.json', 'latest_wait.json', 'launch_verification.json']:
    shutil.copyfile(ROOT / name, out / name)
for name in ['m72_m67_control_content_20260907.py', 'check_m72_prepared_20260907.py']:
    shutil.copyfile(BASE / name, out / name)
shutil.copyfile(__file__, out / Path(__file__).name)
(out / 'evidence_manifest.json').write_text(json.dumps([dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path))
    for path in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as tar:
    for path in sorted(out.iterdir()): tar.add(str(path), arcname=path.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    launch_verification_sha256=sha(ROOT / 'launch_verification.json'), verification=verification), indent=2))
