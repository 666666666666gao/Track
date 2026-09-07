from pathlib import Path
import csv, hashlib, json, shutil, tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m68_reported_confidence_20260907/historical_score_reference'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (ROOT / 'run.exit').read_text().strip() == '0'
result = json.loads((ROOT / 'result.json').read_text())
assert result['status'] == 'completed_historical_Train_confidence_capacity_reference'
assert result['source_sha256'] == sha(BASE / 'm68_historical_score_capacity_20260907.py')
assert result['spec_sha256'] == sha(ROOT / 'spec.json')
assert result['new_tracking_calls'] == result['new_optimizer_steps'] == 0
assert result['boxes_identical_across_readouts'] and result['GT_oracle_is_diagnostic_only']
for arm, readouts in result['prediction_and_score_files'].items():
    for readout, sequences in readouts.items():
        for seq, entry in sequences.items():
            assert sha(ROOT / (arm + '_' + readout) / (seq + '.txt')) == entry['bbox_sha256']
            assert sha(ROOT / (arm + '_' + readout) / (seq + '_all_scores.txt')) == entry['score_sha256']
out = ROOT / 'publication_evidence'; out.mkdir()
for name in ['spec.json', 'result.json', 'run.log', 'run.exit']:
    shutil.copyfile(ROOT / name, out / name)
shutil.copyfile(BASE / 'm68_historical_score_capacity_20260907.py', out / 'm68_historical_score_capacity.py')
shutil.copyfile(__file__, out / 'export_m68_historical_score.py')
with (out / 'aggregate_metrics.csv').open('w', newline='') as f:
    writer = csv.writer(f); writer.writerow(['arm', 'readout', 'precision_percent', 'recall_percent', 'f_score_percent', 'sequences', 'frames', 'scope'])
    for arm, readouts in result['metrics'].items():
        for readout, values in readouts.items():
            writer.writerow([arm, readout] + [values[k] for k in ['precision_percent', 'recall_percent', 'f_score_percent', 'sequences', 'frames']] + ['DepthTrack Train reused development22'])
(out / 'evidence_manifest.json').write_text(json.dumps([
    dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for p in sorted(out.iterdir()):
        t.add(p, arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    result_sha256=sha(ROOT / 'result.json'), spec_sha256=sha(ROOT / 'spec.json')), indent=2))
