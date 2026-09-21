"""Independent, standard-library-only audit of the sealed M82/M84 PR supplement.

Does not import either evaluator, VOT, NumPy, torch, or any tracking code.
Recomputes PR from sealed overlaps, not the nontrivial VOT raster overlaps.
Writes only reviewer_recompute.json and reviewer_pr_curves.csv beside this file.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

R = Path(__file__).resolve().parent
E = R / 'evidence'
SOURCE = {
    'M82': R.parent / 'm82_complete_20260920',
    'M84': R.parent / 'sttrack_m84_centered_20260920' / 'completed',
}
HIST = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\projects\sttrack_lachtt_v1\diagnostics\m68\historical_score_reference\result.json')
HIST_METRIC = HIST.parents[2] / 'native_ope' / 'source_snapshots' / 'depthtrack_pr.py'
HASHES = {}


def sha(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = digest
    return digest


def read(path):
    sha(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(path):
    sha(path)
    return [[float(x) for x in line.replace(',', ' ').split()]
            for line in Path(path).read_text().splitlines() if line.strip()]


spec, result = read(E / 'spec.json'), read(E / 'result.json')
arrays, geometry = read(E / 'overlap_score_arrays.json'), read(E / 'geometry.json')
historical, history_copy = read(HIST), read(E / 'historical_native_PR_reference.json')
manifest, collection = read(E / 'manifest.json'), read(R / 'collection_result.json')
assert historical == history_copy
assert spec['source_sha256'] == result['source_sha256'] == sha(R / 'evaluate_sealed_pr.py') == sha(E / 'evaluate_sealed_pr.py')
assert spec['metric_sha256'] == result['metric_sha256'] == sha(E / 'depthtrack_pr.py') == sha(HIST_METRIC) == historical['metric_sha256']
assert result['spec_sha256'] == sha(E / 'spec.json')
assert result['arrays_sha256'] == sha(E / 'overlap_score_arrays.json')
assert result['input_families'] == spec['families']
assert result['status'] == 'completed_sealed_Train_PR_supplement'
assert spec['resolution'] == 100 and spec['output_decimals'] == 6 and spec['initial_confidence'] == 1.0
assert result['new_tracking_calls'] == result['new_optimizer_steps'] == 0
assert result['M88_modified'] is False and result['independent_audit_completed'] is False

entries = {x['path']: x for x in manifest}
assert len(entries) == len(manifest) == collection['files'] == 345
assert set(entries) == {p.relative_to(E).as_posix() for p in E.rglob('*') if p.is_file() and p.name != 'manifest.json'}
for rel, entry in entries.items():
    p = E / rel
    assert sha(p) == entry['sha256'] and p.stat().st_size == entry['bytes'], rel
archive = R / 'completed_evidence.tar.gz'
assert sha(archive) == collection['archive_sha256'] and archive.stat().st_size == collection['bytes']
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert len(members) == 346 and {m.name for m in members} == set(entries) | {'manifest.json'}
    for m in members:
        assert m.isfile()
        assert hashlib.sha256(tar.extractfile(m).read()).hexdigest() == sha(E / m.name), m.name

for rel in ['PLAN.md', 'NARRATIVE_REPORT.md', 'run.exit', 'run.log', 'launch.json', 'reviewer_invocation.json']:
    sha(R / rel)
assert (R / 'run.exit').read_text().strip() == '0'
launch = read(R / 'launch.json')
logs = [json.loads(line) for line in (R / 'run.log').read_text().splitlines() if line.strip()]
assert {x['family']: x['metrics'] for x in logs} == result['metrics']
with (E / 'metrics.csv').open(newline='') as f:
    csv_rows = list(csv.DictReader(f))
assert len(csv_rows) == 7
for row in csv_rows:
    assert all(float(value) == result['metrics'][row['family']][key] for key, value in row.items() if key != 'family')

names = [c['sequence'] for c in spec['cases']]
assert len(names) == len(set(names)) == 22
assert sum(c['frames'] for c in spec['cases']) == 33130
assert set(arrays) == set(result['metrics']) == set(spec['families'])
gt, gt_evidence, unverified = {}, [], []
for case in spec['cases']:
    name = case['sequence']
    path = E / 'gt' / (name + '.txt')
    assert sha(path) == case['gt_sha256']
    for old in [SOURCE['M82'] / 'dataset_gt' / name / 'groundtruth.txt', SOURCE['M84'] / 'dataset_gt' / (name + '.txt')]:
        assert sha(old) == case['gt_sha256']
    rows = load_rows(path)
    assert len(rows) == case['frames'] and all(len(row) == 4 for row in rows)
    visible = [all(math.isfinite(x) for x in row) and row[2] > 0 and row[3] > 0 for row in rows]
    assert rows[0] == case['init_bbox'] and visible[0]
    gt[name] = visible
    gt_evidence.append(dict(sequence=name, frames=len(rows), visible_including_init=sum(visible), invalid=len(rows)-sum(visible), sha256=case['gt_sha256']))
    unverified.append(dict(kind='dataset_first_image', remote_path=spec['dataset_root']+'/'+name+'/color/00000001.jpg', sha256=geometry[name]['first_image_sha256'], reason='No original image opened or decoded in this audit; width/height remain recorded inputs.'))

parent_evidence = {}
for family, folder in SOURCE.items():
    parent_result = read(folder / 'recursive_result.json')
    parent_spec = read(folder / 'recursive_spec.json')
    training = read(folder / 'training_spec.json')
    frozen = read(folder / 'frozen.json')
    assert frozen['recursive_spec_sha256'] == sha(folder / 'recursive_spec.json')
    assert frozen['training_spec_sha256'] == sha(folder / 'training_spec.json')
    assert parent_spec['cases'] == spec['cases']
    assert sha(folder / 'run_recursive.py') == parent_spec['runner_sha256']
    assert parent_result['status'] == 'complete_recursive_development'
    integration = read(folder / 'integration.json')
    assert sha(folder / 'integration.json') == training['integration_sha256']
    for rel in ['lib/test/tracker/sttrack.py', 'lib/test/tracker/sttrack_semantic.py']:
        assert sha(folder / 'code' / rel) == integration['source_sha256'][rel]
    parent_evidence[family] = dict(local_root=str(folder), result_sha256=sha(folder/'recursive_result.json'), training_spec_sha256=sha(folder/'training_spec.json'), recursive_spec_sha256=sha(folder/'recursive_spec.json'), runner_sha256=sha(folder/'run_recursive.py'), seed=training['seed'])

families, curves_out = {}, []
for label, source in spec['families'].items():
    family = label.split('_')[0]
    folder, arm = SOURCE[family], source['arm']
    parent = parent_evidence[family]
    assert parent['result_sha256'] == source['result_sha256']
    assert parent['training_spec_sha256'] == source['training_spec_sha256']
    assert parent['recursive_spec_sha256'] == source['recursive_spec_sha256']
    receipt_path = folder / (arm + '_recursive_receipt.json')
    receipt = read(receipt_path)
    assert sha(receipt_path) == source['receipt_sha256'] == sha(E / (label+'_receipt.json'))
    assert read(folder/'recursive_result.json')['receipts'][arm] == source['receipt_sha256']
    assert receipt['status'] == 'complete' and receipt['total_frames'] == 33130 and len(receipt['sequences']) == 22
    assert receipt['head_sha256'] == source['head_sha256']
    assert (folder / (arm+'_recursive.exit')).read_text().strip() == '0'
    sha(folder / (arm+'_recursive.exit'))
    train_arm = 'empty' if family == 'M82' and arm == 'empty' else 'category'
    train_result = read(folder / 'training' / train_arm / 'result.json')
    assert sha(folder / 'training' / train_arm / 'result.json') == receipt['training_result_sha256']
    assert train_result['final_checkpoint_sha256'] == source['head_sha256']
    checkpoint = folder / 'training' / train_arm / 'final.pth'
    if checkpoint.is_file():
        assert sha(checkpoint) == source['head_sha256']
    else:
        unverified.append(dict(kind='trained_head', remote_path=source['folder']+'/training/'+train_arm+'/final.pth', sha256=source['head_sha256'], reason='Original checkpoint bytes absent at supplied local completed-source path; receipt and training result agree.'))
    assert set(arrays[label]) == set(names)
    n_frames = 0
    bbox_error, score_error = 0.0, 0.0
    provenance, score_values = [], []
    for case, item in zip(spec['cases'], receipt['sequences']):
        name = case['sequence']
        assert item['sequence'] == name and item['frames'] == case['frames']
        original = folder / 'recursive' / arm / (name+'.json')
        assert sha(original) == item['sha256']
        saved = read(original)
        rows = saved['rows']
        assert saved['sequence'] == name and saved['arm'] == arm
        assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames']))
        assert rows[0]['score'] is None and rows[0]['bbox'] == case['init_bbox']
        assert all(len(r['bbox']) == 4 and all(math.isfinite(x) for x in r['bbox']) and r['bbox'][2] > 0 and r['bbox'][3] > 0 for r in rows)
        assert all(math.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in rows[1:])
        boxes_path, score_path = E / label / (name+'.txt'), E / label / (name+'_all_scores.txt')
        scores = [1.0] + [r['score'] for r in rows[1:]]
        expected_bbox = ''.join(','.join('%.6f' % x for x in r['bbox'])+'\n' for r in rows).encode()
        expected_scores = ''.join('%.6f\n' % x for x in scores).encode()
        assert boxes_path.read_bytes() == expected_bbox and score_path.read_bytes() == expected_scores, label+'/'+name
        assert sha(boxes_path) == result['export_sha256'][label][name]['bbox_sha256']
        assert sha(score_path) == result['export_sha256'][label][name]['score_sha256']
        rounded_boxes, rounded_scores = load_rows(boxes_path), [r[0] for r in load_rows(score_path)]
        bbox_error = max(bbox_error, max(abs(a-b) for r, rr in zip(rows, rounded_boxes) for a,b in zip(r['bbox'],rr)))
        score_error = max(score_error, max(abs(a-b) for a,b in zip(scores,rounded_scores)))
        a = arrays[label][name]
        assert len(a['overlap']) == len(a['visible']) == len(a['confidence']) == len(rows)
        assert all(type(v) is bool for v in a['visible']) and a['visible'] == gt[name]
        assert a['confidence'] == rounded_scores
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in a['overlap'])
        assert a['overlap'][0] == 1.0
        assert all(v or x == 0 for x,v in zip(a['overlap'],a['visible']))
        if label == 'M84_category_empty':
            assert result['export_sha256'][label][name] == historical['prediction_and_score_files']['native']['raw'][name]
        n_frames += len(rows)
        score_values.extend(rounded_scores)
        provenance.append(dict(sequence=name, original_path=str(original), original_sha256=item['sha256'], frames=len(rows), six_decimal_bbox_and_score_bytes_match=True))
    assert n_frames == 33130 and bbox_error <= 5.01e-7 and score_error <= 5.01e-7

    # Separate standard-library implementation: a global confidence rank grid,
    # math.fsum-based per-sequence sums, then equal-weight sequence means.
    ranked = sorted(score_values, reverse=True)
    inner = 98
    delta = len(ranked) // inner
    step = (len(ranked) - 2*delta) / (inner-1)
    indices = [round(delta+i*step) for i in range(inner)]
    thresholds = [math.inf] + [ranked[i] for i in indices] + [-math.inf]
    pc, rc = [], []
    for ti, threshold in enumerate(thresholds):
        per_p, per_r = [], []
        for name in names:
            a = arrays[label][name]
            selected = [ov for ov,conf in zip(a['overlap'],a['confidence']) if conf >= threshold]
            total = math.fsum(selected)
            per_p.append(total/len(selected) if selected else 1.0)
            per_r.append(total/sum(a['visible']))
        pc.append(math.fsum(per_p)/22)
        rc.append(math.fsum(per_r)/22)
    fc = [2*p*r/(p+r) if p+r > 0 else 0.0 for p,r in zip(pc,rc)]
    best = max(range(100), key=lambda i: fc[i])
    metric = dict(precision=pc[best], recall=rc[best], f_score=fc[best], precision_percent=100*pc[best], recall_percent=100*rc[best], f_score_percent=100*fc[best], threshold=thresholds[best], sequences=22, frames=n_frames)
    differences = {k: abs(v-result['metrics'][label][k]) for k,v in metric.items()}
    assert max(differences.values()) < 1e-10, (label, differences)
    families[label] = dict(metrics=metric, reported_abs_differences=differences, best_index=best, threshold_count=100, max_bbox_rounding_error=bbox_error, max_score_rounding_error=score_error, initialization_frames=22, visible_frames=sum(sum(gt[n]) for n in names), invalid_gt_frames=sum(len(gt[n])-sum(gt[n]) for n in names), provenance=provenance)
    for i,t in enumerate(thresholds):
        curves_out.append(dict(family=label, index=i, threshold=str(t) if not math.isfinite(t) else t, precision=pc[i], recall=rc[i], f_score=fc[i], best=i==best))

with (R/'reviewer_pr_curves.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(curves_out[0]))
    writer.writeheader()
    writer.writerows(curves_out)
sha(R/'reviewer_pr_curves.csv')
sha(__file__)
output = dict(status='deterministic_checks_pass', generated_at=datetime.now(timezone.utc).isoformat(), python_executable=sys.executable, python_version=sys.version, review_independence='deterministic', acceptance_status='accepted_for_specified_checks_only', imported_evaluators=False, recomputed_vot_raster_overlaps=False, full_chain_independently_verified=False, metric_absolute_tolerance=1e-10, manifest_records_verified=345, archive_files_verified=346, families=families, gt=gt_evidence, parents=parent_evidence, historical_native_raw_metrics_exact_match=result['metrics']['M84_category_empty']==historical['metrics']['native']['raw'], historical_native_44_export_hashes_match=True, launch_record=launch, unverified_remote_inputs=unverified, audited_input_sha256=HASHES)
(R/'reviewer_recompute.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:output[k] for k in ['status','manifest_records_verified','archive_files_verified','historical_native_raw_metrics_exact_match','full_chain_independently_verified']}))
for label,x in families.items():
    print(json.dumps(dict(family=label,metrics=x['metrics'],max_abs_metric_difference=max(x['reported_abs_differences'].values()),max_bbox_rounding_error=x['max_bbox_rounding_error'],max_score_rounding_error=x['max_score_rounding_error'],visible=x['visible_frames'],invalid=x['invalid_gt_frames'])))
