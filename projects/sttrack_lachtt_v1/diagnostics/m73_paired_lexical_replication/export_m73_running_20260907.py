"""Verify the registered M73 jobs and export source/spec/checks plus bounded progress."""
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

B = Path('/root/autodl-tmp'); ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
PARENT = B / 'sttrack_m67_supervised_semantic_support_20260907'
OUT = ROOT / 'publication_evidence'
assert not OUT.exists()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


spec = read(ROOT / 'spec.json'); frozen = read(ROOT / 'frozen.json'); launch = read(ROOT / 'launch_verified.json')
assert spec['source_sha256'] == sha(B / 'm73_paired_lexical_replication_20260907.py')
assert frozen['spec_sha256'] == launch['spec_sha256'] == sha(ROOT / 'spec.json')
assert launch['frozen_sha256'] == sha(ROOT / 'frozen.json')
jobs = []
for job in launch['jobs']:
    p = Path('/proc') / str(job['pid'])
    assert p.exists() and (p / 'stat').read_text().split()[21] == job['start_ticks']
    args = [x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x]
    assert args == job['argv'] and str((p / 'cwd').resolve()) == job['cwd']
    jobs.append(job)
progress = {}; files = {}
for n in ['spec.json', 'frozen.json', 'run_m73.sh', 'launch.json', 'launch_verified.json']:
    files[n] = ROOT / n
for n in ['m73_paired_lexical_replication_20260907.py', 'check_m73_causal_20260907.py', 'export_m73_running_20260907.py']:
    files[n] = B / n
for seed in [2027, 2028]:
    r = ROOT / ('seed' + str(seed)); s = read(r / 'training_spec.json')
    assert sha(r / 'training_spec.json') == frozen['seeds'][str(seed)]['training_spec_sha256']
    assert sha(r / 'recursive_spec.json') == frozen['seeds'][str(seed)]['recursive_spec_sha256']
    assert sha(r / 'train_causal.py') == s['training_script_sha256']
    assert sha(r / 'causal_training.py') == s['causal_script_sha256']
    assert sha(r / 'run_recursive.py') == read(r / 'recursive_spec.json')['runner_sha256']
    assert sha(r / 'run_pair.sh') == s['run_queue_sha256']
    for name, digest in read(r / 'integration.json')['source_sha256'].items(): assert sha(r / 'code' / name) == digest
    assert len(read(r / 'integration.json')['source_sha256']) == 160
    for arm in ['category', 'empty']:
        assert sha(r / 'native_parity' / (arm + '_zero.pth')) == s['initial_checkpoint_sha256'][arm]
        log = r / 'training' / arm / 'sequence_log.jsonl'
        rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert [x['sequence'] for x in rows] == [x['sequence'] for x in s['sequence_order'][:len(rows)]]
        progress[str(seed) + '/' + arm] = dict(completed_sequences=len(rows), latest=rows[-1] if rows else None,
            trained_final_exists=(r / 'training' / arm / 'final.pth').exists())
    for n in ['prepared_training_spec.json', 'prepared_recursive_spec.json', 'training_spec.json', 'recursive_spec.json',
              'frozen.json', 'causal_check.json', 'causal_check.exit', 'recursive_preflight.log', 'recursive_preflight.exit',
              'train_causal.py', 'run_recursive.py', 'run_pair.sh', 'causal_training.py', 'integration.json']:
        files['seed' + str(seed) + '__' + n] = r / n
qwen = {}
for n in ['Qwen3_8B', 'Qwen2.5-VL-3B-Instruct']:
    paths = list((B / 'qwen' / n).glob('*.safetensors')); qwen[n] = dict(files=len(paths), bytes=sum(p.stat().st_size for p in paths))
assert qwen['Qwen3_8B'] == dict(files=5, bytes=16381516776)
assert qwen['Qwen2.5-VL-3B-Instruct'] == dict(files=2, bytes=7509337976)
report = dict(status='verified_live_M73_paired_training_and_frozen_sources', observed_utc=datetime.now(timezone.utc).isoformat(),
    jobs=jobs, progress=progress, spec_sha256=sha(ROOT / 'spec.json'), frozen_sha256=sha(ROOT / 'frozen.json'),
    all_160_model_sources_identical_to_M67=True, retained_qwen=qwen, disk_free_bytes=shutil.disk_usage(str(B)).free,
    gpu_memory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader']).decode(),
    exporter_sha256=sha(__file__), public_evaluation_started=False, independent_model_review_pass=False)
OUT.mkdir()
for n, p in files.items(): (OUT / n).write_bytes(p.read_bytes())
(OUT / 'running_verification.json').write_text(json.dumps(report, indent=2) + '\n')
for seed in [2027, 2028]:
    for name in ['train_causal.py', 'run_recursive.py']:
        previous = (PARENT / name).read_text().splitlines(keepends=True)
        current = (ROOT / ('seed' + str(seed)) / name).read_text().splitlines(keepends=True)
        diff = ''.join(difflib.unified_diff(previous, current, fromfile='M67/' + name, tofile='M73/seed' + str(seed) + '/' + name))
        (OUT / ('seed' + str(seed) + '__' + name + '.diff')).write_text(diff)
manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir())]
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
a = ROOT / 'running_evidence.tar.gz'
with tarfile.open(str(a), 'w:gz') as t:
    for p in sorted(OUT.iterdir()): t.add(str(p), arcname=p.name)
print(json.dumps(dict(archive=str(a), bytes=a.stat().st_size, sha256=sha(a), files=len(list(OUT.iterdir())), running=report)))
