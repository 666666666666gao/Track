"""Freeze full recursive evaluation inputs and metric source before results."""
import argparse
import datetime
import json
from pathlib import Path

from train import binding, sha, write


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
root = parser.parse_args().root
assert not (root / 'recursive_spec.json').exists()
spec = binding(root)
parent = Path(spec['source_root'])
p = json.loads((parent / 'spec.json').read_text())
assert sha(parent / 'inference_inputs.json') == p['inference_inputs_sha256']
cases = sorted([dict(sequence=row['sequence'], frames=row['frames'], init_bbox=row['init_bbox'])
    for row in json.loads((parent / 'inference_inputs.json').read_text()) if row['split'] == 'development'], key=lambda x: x['sequence'])
assert len(cases) == 22 and sum(row['frames'] for row in cases) == 33130
for row in cases:
    folder = Path(spec['dataset_root']) / row['sequence']
    assert sorted(path.name for path in (folder / 'color').glob('*.jpg')) == ['%08d.jpg' % (i + 1) for i in range(row['frames'])]
    assert sorted(path.name for path in (folder / 'depth').glob('*.png')) == ['%08d.png' % (i + 1) for i in range(row['frames'])]
for path, digest in p['baseline_trace_sha256'].items():
    assert sha(path) == digest
metric = Path(spec['repository']) / 'tools/analyze_sttrack_m42_recursive.py'
write(root / 'recursive_inputs.json', cases)
write(root / 'recursive_spec.json', dict(schema='sttrack_m56_equal_budget_recursive_development_v1',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), training_spec_sha256=sha(root / 'spec.json'),
    runner_sha256=sha(root / 'run_recursive.py'), preparer_sha256=sha(__file__),
    inputs_sha256=sha(root / 'recursive_inputs.json'), metric_sha256=sha(metric),
    baseline_trace_sha256=p['baseline_trace_sha256'],
    development_gt_sha256={row['sequence']: sha(Path(spec['dataset_root']) / row['sequence'] / 'groundtruth.txt') for row in cases},
    arms=spec['variants'], sequences=22, frames_per_arm=33130, native_default_templates=True,
    inference_uses_subsequent_gt=False, default_trajectory_is_independent=True,
    static_results_do_not_replace_full_recursive_gate=True, public_automatic_launch=False,
    weight_selection='All three fixed final epoch20 heads, frozen before development numerical analysis',
    scope='Previously reused Train development; static true/empty/shuffled/conflict controls are reported separately'))
print(json.dumps(dict(status='prepared', recursive_spec_sha256=sha(root / 'recursive_spec.json'), frames_per_arm=33130)))
