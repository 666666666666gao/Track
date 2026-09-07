from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m71_equal_budget_routing_20260907/post_queue_runner'
M67 = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
QUEUE = BASE / 'sttrack_post_m67_diagnostic_queue_20260907'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
spec = read(ROOT / 'spec.json')
assert sha(BASE / 'm71_after_diagnostics_20260907.py') == spec['source_sha256']
assert sha(ROOT / 'launch.sh') == spec['launch_sha256']
for path, expected in spec['dependencies'].items(): assert sha(path) == expected

def live(record):
    folder = Path('/proc') / str(record['pid'])
    fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z' and fields[19] == record['start_ticks']
    assert str((folder / 'cwd').resolve()) == record['cwd']
    assert [s.decode() for s in (folder / 'cmdline').read_bytes().split(b'\0') if s] == record['argv']

registered = read(ROOT / 'running_identity.json')['identity']
live(registered)
assert registered['argv'] == ['/root/autodl-tmp/envs/sttrack/bin/python', '-u', str(BASE / 'm71_after_diagnostics_20260907.py'), 'run']
live(spec['parent_identity'])
latest = read(ROOT / 'latest.json')
assert latest['event'] == 'waiting_for_registered_diagnostic_queue'
assert latest['parent_identity'] == spec['parent_identity'] and latest['poll_seconds'] == 240
assert not (ROOT / 'analysis.exit').exists() and not (ROOT.parent / 'result.json').exists()

training = {}
for arm, pid in [('control', 465923), ('support', 465924)]:
    rows = [json.loads(s) for s in (M67 / 'training' / arm / 'sequence_log.jsonl').read_text().splitlines()]
    active = (Path('/proc') / str(pid)).exists()
    if active:
        live(dict(pid=pid, start_ticks='4577349167', cwd=str(M67),
            argv=['/root/autodl-tmp/envs/sttrack/bin/python', '-u', 'train_causal.py', '--arm', arm]))
    else:
        assert (M67 / ('training_' + arm + '.exit')).read_text().strip() == '0'
    training[arm] = dict(training_process_live=active, completed_sequences=len(rows),
        total_track_calls=rows[-1]['total_track_calls'], optimizer_steps=rows[-1]['total_optimizer_steps'])
qwen = {}
for name, count, total in [('Qwen3_8B', 5, 16381516776), ('Qwen2.5-VL-3B-Instruct', 2, 7509337976)]:
    files = list((BASE / 'qwen' / name).rglob('*.safetensors'))
    size = sum(p.stat().st_size for p in files)
    assert len(files) == count and size == total
    qwen[name] = dict(safetensors_files=count, safetensors_bytes=size, preserved=True)
verification = dict(status='live_M71_waiter_and_frozen_parent_verified', observed_utc=datetime.now(timezone.utc).isoformat(),
    verifier_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'), source_sha256=spec['source_sha256'],
    live_identity=registered, parent_identity=spec['parent_identity'], poll_seconds=240,
    M67_training=training, M67_exit_codes={n: (M67 / n).read_text().strip() for n in
        ['controller.exit', 'training_control.exit', 'training_support.exit', 'control_recursive.exit', 'support_recursive.exit', 'recursive_analysis.exit'] if (M67 / n).exists()},
    parent_latest=read(QUEUE / 'latest.json'), disk_free_bytes=shutil.disk_usage(BASE).free, Qwen=qwen,
    M71_analysis_started=False, new_GPU_jobs_started=0, source_dependencies_unchanged=True,
    new_formal_metrics=False, independent_model_review_pass=False)
(ROOT / 'launch_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
out = ROOT / 'publication_evidence'
out.mkdir()
for name in ['spec.json', 'launch.sh', 'running_identity.json', 'latest.json', 'launch_verification.json']:
    shutil.copyfile(ROOT / name, out / name)
shutil.copyfile(BASE / 'm71_after_diagnostics_20260907.py', out / 'm71_after_diagnostics_20260907.py')
shutil.copyfile(__file__, out / Path(__file__).name)
(out / 'evidence_manifest.json').write_text(json.dumps([dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p))
    for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as tar:
    for path in sorted(out.iterdir()): tar.add(str(path), arcname=path.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    verification_sha256=sha(ROOT / 'launch_verification.json'), verification=verification), indent=2))
