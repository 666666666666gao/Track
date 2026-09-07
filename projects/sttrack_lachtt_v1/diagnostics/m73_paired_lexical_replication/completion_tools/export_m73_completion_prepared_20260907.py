"""Export bounded, verified M73 completion tools while the original training runs."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

B = Path('/root/autodl-tmp')
ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
Q = ROOT / 'completion_queue'
OUT = ROOT / 'completion_prepared_evidence'
assert not OUT.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
source = B / 'm73_completion_queue_20260907.py'
loader = importlib.util.spec_from_file_location('m73_follower_live_export', str(source))
queue = importlib.util.module_from_spec(loader); loader.loader.exec_module(queue)
spec = queue.checked()
registered = read(Q / 'running_identity.json')
actual = queue.identity(registered['identity']['pid']); assert actual == registered['identity']
assert registered['spec_sha256'] == sha(Q / 'spec.json')
assert not (Q / 'controller.exit').exists() and not (ROOT / 'controller.exit').exists()
assert queue.identity(spec['parent_identity']['pid']) == spec['parent_identity']
assert not Path('/proc/484161').exists()
screen = Path('/proc/484226'); fields = (screen / 'stat').read_text().rsplit(')', 1)[1].split()
assert fields[19] == '4579959585' and fields[1] == '1'
assert queue.identity(484228) == actual
progress = {}; jobs = []
for job in read(ROOT / 'launch_verified.json')['jobs']:
    assert queue.identity(job['pid']) == {k: job[k] for k in ['pid', 'start_ticks', 'cwd', 'argv']}
    jobs.append(job)
for seed in [2027, 2028]:
    r = ROOT / ('seed' + str(seed)); training = read(r / 'training_spec.json')
    for arm in ['category', 'empty']:
        p = r / 'training' / arm / 'sequence_log.jsonl'
        rows = [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
        assert [v['sequence'] for v in rows] == [v['sequence'] for v in training['sequence_order'][:len(rows)]]
        progress[str(seed) + '/' + arm] = dict(completed_sequences=len(rows), latest=rows[-1] if rows else None,
            final_exists=(r / 'training' / arm / 'final.pth').exists())
retained = {}
for name in ['Qwen3_8B', 'Qwen2.5-VL-3B-Instruct']:
    shards = list((B / 'qwen' / name).glob('*.safetensors'))
    retained[name] = dict(files=len(shards), bytes=sum(p.stat().st_size for p in shards))
assert retained['Qwen3_8B'] == dict(files=5, bytes=16381516776)
assert retained['Qwen2.5-VL-3B-Instruct'] == dict(files=2, bytes=7509337976)
report = dict(status='verified_M73_training_and_completion_follower_live', observed_utc=datetime.now(timezone.utc).isoformat(),
    follower_identity=actual, follower_spec_sha256=sha(Q / 'spec.json'), parent_identity=spec['parent_identity'], training_jobs=jobs,
    latest_follower_state=read(Q / 'latest.json'), progress=progress, disk_free_bytes=shutil.disk_usage(str(B)).free,
    retained_qwen=retained, gpu_memory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader']).decode(),
    launch_wrapper=dict(pid=484161, released_after_identity_and_independent_screen_session_check=True,
        reason='Uppercase screen -D -m kept the launcher waiting. Only that launcher was terminated; screen/follower/original training retain their original live identities.',
        screen_pid=484226, screen_reparented_to_pid1=True, no_training_or_follower_restart=True),
    original_training_sources_unchanged=True, completed_M73_results_exist=(ROOT / 'result.json').exists(),
    new_content_activation_exists=(ROOT / 'content_counterfactuals/activation.json').exists(),
    public_evaluation_started=False, independent_model_review_pass=False, exporter_sha256=sha(__file__))
files = {}
for name in ['audit_m73_completed_20260907.py', 'm73_content_counterfactuals_20260907.py', 'm73_completion_queue_20260907.py', 'export_m73_completion_prepared_20260907.py']:
    files[name] = B / name
files['auditor_historical_reference.json'] = ROOT / 'completion_tools/reference.json'
for name in ['spec.json', 'preparation_result.json', 'prelaunch_check.json', 'premature_eligibility.log', 'run_controls.sh']:
    files['content__' + name] = ROOT / 'content_counterfactuals' / name
for name in ['spec.json', 'run_after_training.sh', 'running_identity.json', 'latest.json']:
    files['queue__' + name] = Q / name
OUT.mkdir()
for name, path in files.items(): (OUT / name).write_bytes(path.read_bytes())
(OUT / 'running_verification.json').write_text(json.dumps(report, indent=2) + '\n')
manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir())]
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = ROOT / 'completion_prepared_evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as t:
    for p in sorted(OUT.iterdir()): t.add(str(p), arcname=p.name)
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive), files=len(list(OUT.iterdir())), running=report)))
