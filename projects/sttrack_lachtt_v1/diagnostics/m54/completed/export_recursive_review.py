"""Copy sealed development predictions, native boxes and dataset GT for review."""
import datetime
import json
from pathlib import Path

from tools.sttrack_m54_common import check_sources, sha


root = Path('/root/autodl-tmp/sttrack_m54_template_reader_v1_20260906')
plan, parent, spec = check_sources(root)
for name in ['recursive.exit', 'analysis.exit', 'controller.exit']:
    assert (root / name).read_text().strip() == '0'
receipt = json.loads((root / 'recursive_receipt.json').read_text())
result = json.loads((root / 'recursive_result.json').read_text())
assert receipt['status'] == result['status'] == 'complete'
assert receipt['spec_sha256'] == result['spec_sha256'] == sha(root / 'spec.json')
assert result['recursive_receipt_sha256'] == sha(root / 'recursive_receipt.json')
cases = [c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['split'] == 'development']
names = {c['sequence'] for c in cases}
assert len(cases) == len(names) == 22
assert {s['sequence'] for s in receipt['sequences']} == names
for item in receipt['sequences']:
    assert sha(root / 'recursive' / (item['sequence'] + '.json')) == item['sha256']

baseline = {name: [] for name in names}
for path, expected in spec['baseline_trace_sha256'].items():
    assert sha(path) == expected
    for row in json.loads(Path(path).read_text())['rows']:
        if row['sequence'] in names:
            baseline[row['sequence']].append({'frame': row['frame_index'], 'bbox': row['public_bbox']})
for case in cases:
    rows = sorted(baseline[case['sequence']], key=lambda r: r['frame'])
    assert [r['frame'] for r in rows] == list(range(case['frames']))
    assert rows[0]['bbox'] == case['init_bbox']
    baseline[case['sequence']] = rows

# Only after the entire prediction family has been verified, copy dataset GT.
out = root / 'review_inputs'
out.mkdir()
(out / 'default').mkdir()
(out / 'groundtruth').mkdir()
files = {}
gt_sources = {}
for case in cases:
    name = case['sequence']
    default = out / 'default' / (name + '.json')
    default.write_text(json.dumps({'sequence': name, 'rows': baseline[name]}, allow_nan=False) + '\n')
    source_gt = Path(spec['dataset_root']) / name / 'groundtruth.txt'
    copied_gt = out / 'groundtruth' / (name + '.txt')
    copied_gt.write_bytes(source_gt.read_bytes())
    assert sha(copied_gt) == sha(source_gt)
    gt_sources[str(source_gt)] = sha(source_gt)
    for path in [default, copied_gt]:
        files[str(path.relative_to(out))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
check_sources(root)
binding = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    script_sha256=sha(__file__), spec_sha256=sha(root / 'spec.json'),
    recursive_result_sha256=sha(root / 'recursive_result.json'),
    original_native_trace_sha256=spec['baseline_trace_sha256'], dataset_gt_sha256=gt_sources,
    files=files, sequences=len(cases), frames=sum(c['frames'] for c in cases),
    all_recursive_outputs_sealed_before_gt_copy=True, metrics_computed=False,
    scope='Native frame_index/public_bbox values copied unchanged from bound original traces; GT copied byte-for-byte from DepthTrack Train dataset. No tracking or fitting.')
(out / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
print(json.dumps({'sequences': binding['sequences'], 'frames': binding['frames'], 'files': len(files), 'binding_sha256': sha(out / 'binding.json')}))
