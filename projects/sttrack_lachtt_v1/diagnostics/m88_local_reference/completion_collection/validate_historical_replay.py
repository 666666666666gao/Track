"""Compare the new descriptive reader with already sealed independent M87 tables."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root.parents[1]/'sttrack_m87_crop_only_text_20260921/completed_review'
replay = root/'historical_replay_m87'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = lambda p: list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))
checks = {}
inputs = {}
for new_name, old_name, selector, allowed in [
    ('strict_intervals.csv', 'strict_intervals_recomputed.csv', 'reference', ['category_empty', 'category_swapped']),
    ('template_write_events.csv', 'template_writes_recomputed.csv', 'arm', ['category', 'category_empty', 'category_swapped']),
]:
    new_path, old_path = replay/new_name, prior/old_name
    a = [r for r in rows(new_path) if r[selector] in allowed]
    b = [r for r in rows(old_path) if r[selector] in allowed]
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            if k in ['score', 'iou'] and x[k] and y[k]:
                assert abs(float(x[k])-float(y[k])) < 1e-10, (k, x, y)
            else:
                assert x[k] == y[k], (k, x, y)
    checks[new_name] = dict(rows=len(a), matched=True)
    inputs[old_name] = sha(old_path)
    inputs[new_name] = sha(new_path)
result = dict(status='historical_replay_pass', experiment_replayed='M87',
    current_M88_results_available=False, new_tracking_calls=0, new_training_steps=0,
    checks=checks, input_sha256=inputs, analyzer_sha256=sha(root/'analyze_completed.py'),
    validation_source_sha256=sha(Path(__file__)),
    scope='Historical reader validation only; not a completed M88 audit. M84 control metrics checked from its raw trajectories against its sealed result by the reader. Frozen gates are copied, not independently recomputed.')
(root/'historical_replay_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
