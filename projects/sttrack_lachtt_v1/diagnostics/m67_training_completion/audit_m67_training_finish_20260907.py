"""Audit the two completed training artifacts with the frozen M67 auditor, before dev results."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
OUT = ROOT / 'training_completion_audit'
AUDITOR = BASE / 'audit_m67_completed_20260907.py'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
assert sha(AUDITOR) == '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
assert sha(ROOT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
assert sha(ROOT / 'recursive_spec.json') == 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
training = read(ROOT / 'training_spec.json')
assert sha(ROOT / 'integration.json') == training['integration_sha256']
for name, expected in read(ROOT / 'integration.json')['source_sha256'].items():
    assert sha(ROOT / 'code' / name) == expected
for name, key in [('train_causal.py', 'training_script_sha256'), ('causal_training.py', 'causal_script_sha256'),
    ('support_loss.py', 'support_loss_sha256'), ('run_m67.sh', 'run_queue_sha256'),
    ('text_fit.pt', 'text_fit_sha256'), ('text_development.pt', 'text_development_sha256')]:
    assert sha(ROOT / name) == training[key]
assert sha(Path(training['native_checkpoint'])) == training['native_checkpoint_sha256']
loader = importlib.util.spec_from_file_location('frozen_M67_training_auditor', str(AUDITOR))
audit = importlib.util.module_from_spec(loader)
loader.loader.exec_module(audit)
checked_training = audit.training_artifacts(training)

recursive = []
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    command = entry / 'cmdline'
    if not command.exists(): continue
    argv = [s.decode() for s in command.read_bytes().split(b'\0') if s]
    if len(argv) != 5 or argv[:4] != ['/root/autodl-tmp/envs/sttrack/bin/python', '-u', 'run_recursive.py', '--arm']: continue
    if (entry / 'cwd').resolve() != ROOT: continue
    fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z' and argv[4] in ['control', 'support']
    recursive.append(dict(pid=int(entry.name), start_ticks=fields[19], state=fields[0], argv=argv, cwd=str(ROOT)))
assert sorted(r['argv'][4] for r in recursive) == ['control', 'support']
queues = {}
for name, folder in [('post_M67', BASE / 'sttrack_post_m67_diagnostic_queue_20260907'),
    ('M71_after_queue', BASE / 'sttrack_m71_equal_budget_routing_20260907/post_queue_runner')]:
    record = read(folder / 'running_identity.json')['identity']
    proc = Path('/proc') / str(record['pid'])
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z' and fields[19] == record['start_ticks']
    assert str((proc / 'cwd').resolve()) == record['cwd']
    assert [s.decode() for s in (proc / 'cmdline').read_bytes().split(b'\0') if s] == record['argv']
    queues[name] = dict(identity=record, latest=read(folder / 'latest.json'))

result = dict(status='completed_paired_training_artifact_audit_recursive_live', observed_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(__file__), frozen_auditor_sha256=sha(AUDITOR), training_spec_sha256=sha(ROOT / 'training_spec.json'),
    recursive_spec_sha256=sha(ROOT / 'recursive_spec.json'), integration_sha256=sha(ROOT / 'integration.json'),
    training=checked_training,
    training_summaries={arm: {key: read(ROOT / 'training' / arm / 'result.json')[key] for key in
        ['training_label_counts', 'elapsed_seconds', 'observed_utc', 'base_parameters_and_buffers_unchanged',
         'gt_after_prediction_for_loss_only', 'gt_reinitialization_after_first_frame', 'backward_through_crops_or_time']}
        for arm in ['control', 'support']},
    live_recursive_processes=recursive, live_queues=queues, disk_free_bytes=shutil.disk_usage(BASE).free,
    checked_scope='The existing frozen auditor training_artifacts function: physical final/latest/initial models, optimizer states, sequence order, training GT hashes and validity counts, sampled states and scheduled write records. Base freezing is supported by the training runtime assertion, not a newly replayed training run.',
    dev_result_complete=(ROOT / 'recursive_result.json').exists(), paired_development_gate_pass=None,
    new_tracking_calls=0, new_optimizer_steps=0, public_evaluation_allowed=False,
    new_formal_metrics=False, independent_model_review_pass=False)
assert not result['dev_result_complete']
OUT.mkdir()
(OUT / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
