"""Independent scalar recomputation after all three M80 trajectories are sealed."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math

R = Path('/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
result = read(R/'result.json')
assert result['status'] == 'completed_M80_development_and_content'
for name in ['training_category', 'eval_category', 'eval_empty', 'eval_swapped', 'analysis', 'controller']:
    assert (R/(name+'.exit')).read_text().strip() == '0'
spec = read(R/'evaluation_spec.json')
train = read(R/'training_spec.json')
assert sha(R/'training/category/final.pth') == result['head_sha256']
assert sha(R/'evaluation_spec.json') == result['evaluation_spec_sha256']
assert sha(R/'training_spec.json') == result['training_spec_sha256']
cases = {c['sequence']: c for c in spec['cases']}
assert len(cases) == 22
sealed = {}
for arm in ['category', 'empty', 'swapped']:
    receipt = read(R/('receipt_'+arm+'.json'))
    assert sha(R/('receipt_'+arm+'.json')) == result['receipts'][arm]
    assert receipt['status'] == 'complete' and receipt['condition'] == arm
    assert receipt['head_sha256'] == result['head_sha256']
    assert receipt['evaluation_spec_sha256'] == result['evaluation_spec_sha256']
    assert receipt['subsequent_gt_opened'] is False and receipt['text_updated_online'] is False
    bank = spec['swapped_bank'] if arm == 'swapped' else train['banks']['development'][arm]
    assert receipt['text_bank_sha256'] == bank['sha256'] == sha(bank['path'])
    assert receipt['total_frames'] == 33130
    assert len(receipt['sequences']) == 22
    sealed[arm] = {}
    for entry in receipt['sequences']:
        seq = entry['sequence']; case = cases[seq]
        p = R/'recursive'/arm/(seq+'.json')
        assert sha(p) == entry['sha256']
        data = read(p); rows = data['rows']
        assert data['sequence'] == seq and data['condition'] == arm
        assert len(rows) == entry['frames'] == case['frames']
        assert rows[0]['bbox'] == case['init_bbox'] and rows[0]['score'] is None
        for index, row in enumerate(rows):
            assert row['frame'] == index and len(row['bbox']) == 4
            assert all(math.isfinite(x) for x in row['bbox'])
            assert row['bbox'][2] > 0 and row['bbox'][3] > 0
            if index: assert math.isfinite(row['score'])
        assert seq not in sealed[arm]
        sealed[arm][seq] = rows
    assert set(sealed[arm]) == set(cases)

# GT is read only after all 66 outputs and receipts have passed integrity checks.
per = {arm: {} for arm in sealed}
for seq, case in cases.items():
    path = Path(train['dataset_root'])/seq/'groundtruth.txt'
    assert sha(path) == case['gt_sha256']
    gt = [[float(x) for x in line.split(',')] for line in path.read_text().splitlines() if line.strip()]
    assert len(gt) == case['frames']
    for arm in sealed:
        n = low = count = invalid = run = 0; values = []
        for i, (row, box) in enumerate(zip(sealed[arm][seq], gt)):
            if i == 0: continue
            valid = len(box) == 4 and all(math.isfinite(x) for x in box) and box[2] > 0 and box[3] > 0
            if not valid:
                invalid += 1
                count += run >= 10; run = 0
                continue
            a = row['bbox']
            width = max(0., min(a[0]+a[2], box[0]+box[2])-max(a[0], box[0]))
            height = max(0., min(a[1]+a[3], box[1]+box[3])-max(a[1], box[1]))
            inter = width*height
            overlap = inter/(a[2]*a[3]+box[2]*box[3]-inter)
            values.append(overlap); n += 1
            if overlap <= .1: low += 1; run += 1
            else: count += run >= 10; run = 0
        count += run >= 10
        actual = dict(valid_frames=n, iou_sum=math.fsum(values), mean_iou=math.fsum(values)/n,
                      low_iou_frames=low, failure_episodes=count, invalid_gt_frames=invalid)
        expected = result['per_sequence'][arm][seq]
        for key, value in actual.items():
            assert math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-9), (arm, seq, key)
        per[arm][seq] = actual

aggregates = {}
for arm, seqs in per.items():
    a = {k: sum(s[k] for s in seqs.values()) for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
    a['mean_iou'] = a['iou_sum']/a['valid_frames']
    a['macro_sequence_mean_iou'] = math.fsum(s['mean_iou'] for s in seqs.values())/22
    assert a['valid_frames'] == 28897
    for key, value in a.items():
        assert math.isclose(value, result['aggregates'][arm][key], rel_tol=0, abs_tol=1e-9), (arm, key)
    aggregates[arm] = a
parent = read(spec['parent_result_path']); native = read(spec['native_result_path'])
assert sha(spec['parent_result_path']) == spec['parent_result_sha256'] == result['parent_result_sha256']
assert sha(spec['native_result_path']) == spec['native_result_sha256']
for label, source, key in [('native', native, 'native'), ('M78_category', parent, 'category'), ('M78_empty_trained', parent, 'empty')]:
    assert result['per_sequence'][label] == source['per_sequence'][key]

gates = {}; broken = {}; c = result['aggregates']['category']
for ref, margin in [('native', .002), ('M78_empty_trained', .001)]:
    b = result['aggregates'][ref]
    broken[ref] = [n for n in result['per_sequence'][ref] if result['per_sequence'][ref][n]['failure_episodes'] == 0 and per['category'][n]['failure_episodes'] > 0]
    gates.update({ref+'_pooled': c['mean_iou'] >= b['mean_iou']+margin,
                  ref+'_macro': c['macro_sequence_mean_iou'] >= b['macro_sequence_mean_iou'],
                  ref+'_low': c['low_iou_frames'] <= b['low_iou_frames'],
                  ref+'_H10': c['failure_episodes'] <= b['failure_episodes'], ref+'_protect': not broken[ref]})
assert gates == result['primary_gates'] and broken == result['broken_success_sequences']
for ref in ['empty', 'swapped', 'M78_category']:
    b = result['aggregates'][ref]
    expected = result['dropout_vs_no_dropout_gates'] if ref == 'M78_category' else result['content_gates']
    checks = {ref+'_pooled': c['mean_iou'] >= b['mean_iou'] if ref == 'M78_category' else c['mean_iou'] > b['mean_iou'],
              ref+'_macro': c['macro_sequence_mean_iou'] >= b['macro_sequence_mean_iou'] if ref == 'M78_category' else c['macro_sequence_mean_iou'] > b['macro_sequence_mean_iou'],
              ref+'_low': c['low_iou_frames'] <= b['low_iou_frames'], ref+'_H10': c['failure_episodes'] <= b['failure_episodes']}
    assert all(expected[k] == v for k,v in checks.items())
for flag, key in [('primary_pass','primary_gates'),('content_pass','content_gates'),('dropout_mechanism_pass','dropout_vs_no_dropout_gates')]:
    assert result[flag] == all(result[key].values())
assert result['all_development_checks_pass'] == all(result[k] for k in ['primary_pass','content_pass','dropout_mechanism_pass'])
audit = dict(status='complete_independent_scalar_metric_and_receipt_verification', observed_utc=datetime.now(timezone.utc).isoformat(),
             source_sha256=sha(__file__), result_sha256=sha(R/'result.json'), trajectories=66, positions=99390,
             valid_positions_per_condition=28897, aggregates=aggregates,
             primary_pass_count=sum(gates.values()),content_pass_count=sum(result['content_gates'].values()),
             dropout_pass_count=sum(result['dropout_vs_no_dropout_gates'].values()),
             independent_model_review_pass=False, gpu_inference_calls=0, scope='DepthTrack Train reused development22 only')
output = R/'saved_development_audit.json'
assert not output.exists()
output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
print(json.dumps(audit))
