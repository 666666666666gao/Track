"""Freeze M55 development evaluation without running a tracker or reading GT labels."""
import datetime
import hashlib
import json
from pathlib import Path

root = Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906')
parent = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
repository = Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    assert not path.exists(), str(path)
    path.write_text(json.dumps(data, indent=2) + '\n')


train = json.loads((root / 'training_spec.json').read_text())
assert sha(root / 'training_spec.json') == 'b53991ca364cbe767641fcc3216faa41e8bd6699e9ed1e59c0224b140da56998'
assert sha(parent / 'inference_inputs.json') == train['fitting_manifest_sha256']
parent_spec = json.loads((parent / 'spec.json').read_text())
assert sha(parent / 'spec.json') == '8572eca25d04291186980c947400106ad6db91705222f2d7f1153ca6c8fdbd18'
cases = sorted([dict(sequence=x['sequence'], frames=x['frames'], init_bbox=x['init_bbox'])
                for x in json.loads((parent / 'inference_inputs.json').read_text())
                if x['split'] == 'development'], key=lambda x: x['sequence'])
assert [x['sequence'] for x in cases] == sorted(train['development_sequences'])
assert len(cases) == 22
for case in cases:
    folder = Path(train['dataset_root']) / case['sequence']
    rgb = sorted(p.name for p in (folder / 'color').glob('*.jpg'))
    depth = sorted(p.name for p in (folder / 'depth').glob('*.png'))
    assert rgb == ['%08d.jpg' % (f + 1) for f in range(case['frames'])], case['sequence']
    assert depth == ['%08d.png' % (f + 1) for f in range(case['frames'])], case['sequence']
for path, digest in parent_spec['baseline_trace_sha256'].items():
    assert sha(Path(path)) == digest
metric = repository / 'tools/analyze_sttrack_m42_recursive.py'
assert sha(metric) == '03f7dff279a950214ba6040e192f67dea061b541ba068fd0714a6e9da59ad939'
write(root / 'recursive_inputs.json', cases)
write(root / 'recursive_spec.json', dict(
    schema='sttrack_m55_final_epoch_paired_development_recursion_v1',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    training_spec_sha256=sha(root / 'training_spec.json'),
    inputs_sha256=sha(root / 'recursive_inputs.json'),
    runner_sha256=sha(root / 'run_recursive.py'),
    preparation_script_sha256=sha(Path(__file__)),
    metric_repository=str(repository), metric_sha256=sha(metric),
    baseline_trace_sha256=parent_spec['baseline_trace_sha256'],
    development_gt_sha256={x['sequence']: sha(Path(train['dataset_root']) / x['sequence'] / 'groundtruth.txt') for x in cases},
    scope='Fixed previously used Train development; native default trajectories reused independently of new arms',
    arms=['control', 'clone'], sequences=len(cases), frames_per_arm=sum(x['frames'] for x in cases),
    gate_source='training_spec.json recursive_validation, unchanged',
    weight_selection='Final epoch 15 of both completed training arms; no development epoch selection',
    text_enabled=False, public_evaluation=False, inference_uses_subsequent_gt=False,
    native_tracker=True, template_update_interval=50, template_update_threshold_strictly_above=.75,
    long_poll_seconds=240,
))
print(json.dumps({'status': 'prepared_not_executed', 'sequences': len(cases),
                  'frames_per_arm': sum(x['frames'] for x in cases),
                  'recursive_spec_sha256': sha(root / 'recursive_spec.json')}))
