"""Export completed diagnostic summaries and receipts, without weights, images or traces."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

B = Path('/root/autodl-tmp')
ROOT = B / 'sttrack_m68_m72_completed_evidence_20260907'
OUT = ROOT / 'publication_evidence'
AUDIT = B / 'audit_m68_m72_completed_20260907.py'
s = importlib.util.spec_from_file_location('completion_audit', str(AUDIT))
app = importlib.util.module_from_spec(s); s.loader.exec_module(app)
sha = app.sha; read = app.read
assert not OUT.exists()
payload = {'audit_m68_m72_completed_20260907.py': AUDIT,
           'export_m68_m72_completed_20260907.py': Path(__file__)}
results = {}
for name, (folder, source) in app.JOBS.items():
    r = B / folder; result = read(r / 'result.json'); audit = read(ROOT / (name + '_audit.json'))
    assert audit['result_sha256'] == sha(r / 'result.json')
    assert audit['audit_source_sha256'] == sha(AUDIT)
    assert result['source_sha256'] == sha(B / source) and result['spec_sha256'] == sha(r / 'spec.json')
    results[name] = result
    payload[name + '_result.json'] = r / 'result.json'
    payload[name + '_audit.json'] = ROOT / (name + '_audit.json')
    if name in ['M69', 'M72']:
        for part in ['empty', 'swapped', 'prefix']: payload[name + '_' + part + '_receipt.json'] = r / part / 'receipt.json'
    if name == 'M68':
        for part in ['control', 'support']: payload[name + '_' + part + '_receipt.json'] = r / part / 'receipt.json'
    if name == 'M70':
        for part in ['shard0', 'shard1', 'smoke']: payload[name + '_' + part + '_receipt.json'] = r / part / 'receipt.json'
for label, folder in [('queue', 'sttrack_post_m67_diagnostic_queue_20260907'),
                       ('M71_runner', 'sttrack_m71_equal_budget_routing_20260907/post_queue_runner')]:
    r = B / folder; assert (r / 'controller.exit').read_text().strip() == '0'
    payload[label + '_result.json'] = r / 'result.json'
    payload[label + '_events.jsonl'] = r / 'events.jsonl'

# Absolute matched-budget readouts supplement the already frozen paired contrasts.
matched_path = B / app.JOBS['M71'][0] / 'matched_event_metrics.json'
assert sha(matched_path) == results['M71']['matched_metrics_sha256']
rows = read(matched_path)['events']; matched = {}
for group in ['all', 'H10', 'H10_local_inside', 'H10_local_outside', 'healthy']:
    selected = [r for r in rows if r['cohort'] == 'common' and
                (group == 'all' or (group == 'H10' and 'H10' in r['tags']) or
                 (group == 'H10_local_inside' and 'H10' in r['tags'] and r['null_local_centre_inside']) or
                 (group == 'H10_local_outside' and 'H10' in r['tags'] and not r['null_local_centre_inside']) or
                 (group == 'healthy' and 'healthy' in r['tags']))]
    values = {}
    for route in ['category', 'empty', 'prior']:
        for fine in ['category', 'empty']:
            k = route + '/' + fine
            values[k] = {m: sum(r['readouts'][k]['raw'][m] for r in selected) for m in ['capacity', 'correct', 'severe']}
    matched[group] = dict(events=len(selected), raw_readouts=values,
        fine_calls_per_condition=sum(r['fine_windows'] for r in selected),
        coarse_calls_per_text_route=len(selected), prior_coarse_calls=0)

qwen = {}
for name in ['Qwen3_8B', 'Qwen2.5-VL-3B-Instruct']:
    files = sorted((B / 'qwen' / name).glob('*.safetensors'))
    qwen[name] = dict(files=len(files), bytes=sum(p.stat().st_size for p in files))
assert qwen['Qwen3_8B'] == dict(files=5, bytes=16381516776)
assert qwen['Qwen2.5-VL-3B-Instruct'] == dict(files=2, bytes=7509337976)
for pid in [480041, 480042]: assert not (Path('/proc') / str(pid)).exists()
summary = dict(status='completed_and_recomputed_M68_M72_diagnostic_evidence',
    observed_utc=datetime.now(timezone.utc).isoformat(),
    M69_aggregates=results['M69']['aggregates'], M69_criteria=results['M69']['descriptive_lexical_criteria'],
    M68_metrics=results['M68']['metrics'], M68_support_groups=results['M68']['null_mass_and_confidence_by_GT_group']['support'],
    M70_summaries=results['M70']['summaries'], M71_matched_raw=matched,
    M72_aggregates=results['M72']['aggregates'], M72_criteria=results['M72']['descriptive_lexical_criteria'],
    M72_template_writes=results['M72']['reconstructed_template_writes'],
    disk_free_bytes=shutil.disk_usage(str(B)).free, retained_qwen=qwen,
    new_official_benchmark_results=False, original_gates_unchanged=True, independent_model_review_pass=False)
OUT.mkdir()
for name, path in payload.items(): (OUT / name).write_bytes(path.read_bytes())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
manifest = [dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(OUT.iterdir())]
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(str(archive), 'w:gz') as t:
    for p in sorted(OUT.iterdir()): t.add(str(p), arcname=p.name)
print(json.dumps(dict(archive=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
    files=len(list(OUT.iterdir())), M71_matched_raw=matched, disk_free_bytes=summary['disk_free_bytes'])))
