"""Analyze all predeclared M61 events only after both reference replays finish."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/sttrack_m61_fixed_state_language_20260907')
SOURCE = Path('/root/autodl-tmp/m61_fixed_state_language_20260907.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(box, gt):
    a = np.asarray(box)
    intersection = np.maximum(0, np.minimum(a[:2] + a[2:], gt[:2] + gt[2:]) - np.maximum(a[:2], gt[:2])).prod()
    return float(intersection / (a[2:].prod() + gt[2:].prod() - intersection))


def summarize(events, condition):
    selected = [r for r in events if r['valid_GT']]
    rows = [r['conditions'][condition] for r in events]
    valid = [r['conditions'][condition] for r in selected]
    return dict(events=len(events), valid_GT_events=len(selected),
        mean_event_iou=float(np.mean([r['iou'] for r in valid])),
        correct_events=sum(r['iou'] >= .5 for r in valid), low_events=sum(r['iou'] <= .1 for r in valid),
        peak_changes=sum(r['peak_changed'] for r in rows),
        write_eligible=sum(r['write_eligible'] for r in rows),
        write_eligibility_flips=sum(r['write_flip'] for r in rows),
        same_peak_write_flips=sum(r['write_flip'] and not r['peak_changed'] for r in rows),
        fixed_reference_peak_write_flips=sum(r['fixed_peak_write_flip'] for r in rows),
        current_correct_to_severe_low=sum(r['reference_iou'] >= .5 and r['iou'] <= .1 for r in valid),
        current_severe_low_to_correct=sum(r['reference_iou'] <= .1 and r['iou'] >= .5 for r in valid),
        mean_fixed_peak_iou=float(np.mean([r['fixed_peak_iou'] for r in valid])))


def main():
    plan = json.loads((ROOT / 'analysis_plan.json').read_text())
    assert sha(Path(__file__)) == plan['analyzer_sha256']
    assert sha(ROOT / 'spec.json') == plan['collection_spec_sha256']
    spec = importlib.util.spec_from_file_location('fixed_state_language', str(SOURCE))
    source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
    _, frozen, training, _ = source.checked()
    receipts = {}
    for ref in source.REFERENCES:
        assert (ROOT / ('collect_' + ref + '.exit')).read_text().strip() == '0'
        p = ROOT / ('collect_' + ref) / 'receipt.json'
        receipt = json.loads(p.read_text())
        assert receipt['status'] == 'verified' and not receipt['prefix_only']
        assert receipt['spec_sha256'] == sha(ROOT / 'spec.json')
        assert receipt['frames'] == 33130 and receipt['events'] == 654
        assert [r['sequence'] for r in receipt['rows']] == [c['sequence'] for c in frozen['cases']]
        for item in receipt['rows']:
            assert item['max_bbox_error'] <= 1e-4 and item['max_score_error'] <= 1e-6
            folder = ROOT / ('collect_' + ref)
            assert sha(folder / (item['sequence'] + '.json')) == item['event_sha256']
            assert sha(folder / (item['sequence'] + '.npz')) == item['maps_sha256']
            with np.load(folder / (item['sequence'] + '.npz')) as arrays:
                for key, channels in [('score_map', 1), ('size_map', 2), ('offset_map', 2)]:
                    assert arrays[key].shape == (item['events'], 5, channels, 16, 16)
                    assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
        receipts[ref] = receipt
    groundtruth = {}
    for case in frozen['cases']:
        path = Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt'
        assert sha(path) == case['gt_sha256']
        gt = np.loadtxt(path, delimiter=',').reshape(-1, 4)
        assert len(gt) == case['frames']
        groundtruth[case['sequence']] = gt
    summaries = {}; by_sequence = {}; paired_events = {}; transforms = {}
    for ref in source.REFERENCES:
        events = []; by_sequence[ref] = {}
        for case in frozen['cases']:
            path = ROOT / ('collect_' + ref) / (case['sequence'] + '.json')
            data = json.loads(path.read_text())
            assert data['reference'] == ref and data['sequence'] == case['sequence']
            assert [r['frame'] for r in data['events']] == list(range(50, case['frames'], 50))
            current = []
            for event in data['events']:
                gt = groundtruth[case['sequence']][event['frame']]
                valid = bool(np.isfinite(gt).all() and (gt[2:] > 0).all())
                reference = event['conditions'][ref]
                assert reference['peak'] == event['reference_peak']
                assert reference['write_eligible'] == event['actual_template_write']
                reference_iou = overlap(reference['bbox'], gt) if valid else None
                observation = dict(sequence=case['sequence'], frame=event['frame'], valid_GT=valid,
                                   reference_score=reference['score'], actual_write=event['actual_template_write'], conditions={})
                for name in source.CONDITIONS:
                    row = event['conditions'][name]
                    assert row['write_eligible'] == (row['score'] > .75)
                    assert np.isfinite(row['bbox']).all() and (np.asarray(row['bbox'])[2:] > 0).all()
                    observation['conditions'][name] = dict(
                        iou=overlap(row['bbox'], gt) if valid else None,
                        fixed_peak_iou=overlap(row['fixed_reference_peak_bbox'], gt) if valid else None,
                        reference_iou=reference_iou, peak_changed=row['peak'] != reference['peak'],
                        write_eligible=row['write_eligible'], write_flip=row['write_eligible'] != reference['write_eligible'],
                        fixed_peak_write_flip=(row['score_at_reference_peak'] > .75) != reference['write_eligible'])
                current.append(observation)
            by_sequence[ref][case['sequence']] = {name: summarize(current, name) for name in source.CONDITIONS}
            events.extend(current)
        assert len(events) == 654
        summaries[ref] = {name: summarize(events, name) for name in source.CONDITIONS}
        transforms[ref] = {str(power): dict(
            original_writes=sum(e['actual_write'] for e in events),
            transformed_write_eligible=sum(e['reference_score'] ** power > .75 for e in events),
            eligibility_flips=sum((e['reference_score'] ** power > .75) != e['actual_write'] for e in events),
            boxes_and_RoI_features_changed=False, actual_updates_changed=False)
            for power in frozen['score_only_transforms']}
        assert summaries[ref][ref]['peak_changes'] == summaries[ref][ref]['write_eligibility_flips'] == 0
        assert summaries[ref][ref]['write_eligible'] == sum(r['actual_template_writes'] for r in receipts[ref]['rows'])
        paired_events[ref] = events
    result = dict(status='completed_matched_state_language_diagnostic', observed_utc=datetime.now(timezone.utc).isoformat(),
                  collection_spec_sha256=sha(ROOT / 'spec.json'), analysis_plan_sha256=sha(ROOT / 'analysis_plan.json'),
                  analyzer_sha256=sha(Path(__file__)), head_sha256=frozen['head_sha256'],
                  receipt_sha256={ref: sha(ROOT / ('collect_' + ref) / 'receipt.json') for ref in source.REFERENCES},
                  summaries=summaries, per_sequence=by_sequence, score_only_monotonic_transforms=transforms,
                  events_per_reference=654, references=source.REFERENCES,
                  public_evaluation_allowed=False, independent_review_pass=False,
                  scope='Current-event continuous IoU under a fixed replayed reference state. All diagnostic outputs remain uncommitted. Update eligibility is not future template utility or a new deployed controller. No full public metrics.')
    (ROOT / 'paired_event_analysis.json').write_text(json.dumps(paired_events, indent=2, allow_nan=False) + '\n')
    (ROOT / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_sequence'}, indent=2))


if __name__ == '__main__':
    main()
