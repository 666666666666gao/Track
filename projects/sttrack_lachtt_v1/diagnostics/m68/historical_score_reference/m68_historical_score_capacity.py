"""Historical Train-only confidence capacity reference; GT scores are diagnostic only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m65_category_null_support_20260907'
ROOT = BASE / 'sttrack_m68_reported_confidence_20260907/historical_score_reference'
AUDITOR = BASE / 'audit_m65_completed_20260907.py'
AUDITOR_SHA = 'ef6a3f5f9f7b8635497fc993fded0924e7f37f5b67d14c3fab46fc27b7e596db'
METRIC = Path('/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py')
METRIC_SHA = '05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
RESULT_SHA = '0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


def plans():
    assert sha(AUDITOR) == AUDITOR_SHA and sha(METRIC) == METRIC_SHA
    assert sha(PARENT / 'recursive_result.json') == RESULT_SHA
    assert sha(PARENT / 'recursive_spec.json') == '09f3f193de81f9cf91518b2c05497bfea30e8ab88db812e3585f6d3618767723'
    assert sha(PARENT / 'training_spec.json') == 'fc04a897f3d3e246203981a2cb3b83ea50075392e30c0c960c8908eaaeabb93b'
    return read(PARENT / 'recursive_spec.json'), read(PARENT / 'training_spec.json')


def prepare():
    recursion, training = plans()
    ROOT.mkdir()
    spec = dict(status='frozen_before_historical_PR_computation', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), auditor_sha256=AUDITOR_SHA, metric_sha256=METRIC_SHA, M65_result_sha256=RESULT_SHA,
        cases=recursion['cases'], dataset_root=training['dataset_root'], arms=['native', 'control', 'null'],
        readouts=['raw', 'constant', 'GT_overlap_oracle'], output_decimals=6, pr_resolution=100,
        definitions=dict(raw='Historical sealed raw Hann score, initial confidence 1.', constant='Every frame confidence 1.',
            GT_overlap_oracle='Bounded VOT overlap of the same six-decimal box with GT; initialization confidence 1. Uses GT only for post-hoc diagnostic.'),
        unchanged='All boxes, historical tracking states and original experimental conclusions.',
        interpretation='GT-overlap ranking is an oracle reference, not a proved optimum, deployable score or promotion gate.',
        new_tracking_calls=0, new_optimizer_steps=0, public_test_GT_allowed=False,
        M67_training_changed=False, M68_fixed_formula_changed=False, public_evaluation_allowed=False)
    write(ROOT / 'spec.json', spec)
    print(json.dumps(dict(status=spec['status'], spec_sha256=sha(ROOT / 'spec.json')), indent=2))


def run():
    import numpy as np
    spec = read(ROOT / 'spec.json'); assert spec['source_sha256'] == sha(__file__)
    recursion, training = plans(); assert spec['cases'] == recursion['cases']
    audit = module('historical_m65_sealed_loader', AUDITOR)
    data = {}; receipts = {}
    for arm in ['control', 'null']:
        data[arm], receipts[arm] = audit.sealed_family(PARENT, arm, recursion, training)
    native_boxes, native_receipt = audit.sealed_native(recursion, training)
    native = {c['sequence']: [] for c in spec['cases']}
    for path, digest in native_receipt['trace_sha256'].items():
        assert sha(path) == digest
        for row in read(path)['rows']:
            if row['sequence'] in native:
                native[row['sequence']].append(row)
    data['native'] = {}
    for case in spec['cases']:
        seq = case['sequence']; ordered = sorted(native[seq], key=lambda r: r['frame_index'])
        rows = [dict(frame=r['frame_index'], bbox=r['public_bbox'], score=r['public_score']) for r in ordered]
        assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(case['frames']))
        assert [r['bbox'] for r in rows] == [r['bbox'] for r in native_boxes[seq]]
        assert all(np.isfinite(r['score']) and 0 <= r['score'] <= 1 for r in rows[1:])
        data['native'][seq] = rows
    receipts['native'] = native_receipt
    # All complete historical families are now verified; GT is first used below.
    evaluator = module('historical_fixed_long_term_pr', METRIC)
    results = {}; per_sequence = {}; constant_checks = {}; files = {}
    for arm in spec['arms']:
        results[arm] = {}; per_sequence[arm] = {}; files[arm] = {}
        for readout in spec['readouts']:
            (ROOT / (arm + '_' + readout)).mkdir()
            files[arm][readout] = {}
        unconditional = []
        for case in spec['cases']:
            seq = case['sequence']; rows = data[arm][seq]; sequence_root = Path(training['dataset_root']) / seq
            boxes = np.asarray([r['bbox'] for r in rows], dtype=np.float64)
            raw_dir = ROOT / (arm + '_raw')
            np.savetxt(raw_dir / (seq + '.txt'), boxes, fmt='%.6f', delimiter=',')
            rounded = evaluator._load_rows(raw_dir / (seq + '.txt'), 4)
            assert np.max(np.abs(rounded - boxes)) <= 5.01e-7
            gt_path = sequence_root / 'groundtruth.txt'; assert sha(gt_path) == case['gt_sha256']
            gt = evaluator._load_rows(gt_path, 4)
            image = evaluator.cv2.imread(str(sequence_root / 'color/00000001.jpg')); h, w = image.shape[:2]
            overlap, visible = evaluator._vot_overlaps(rounded, gt, w, h)
            assert np.isfinite(overlap).all() and len(overlap) == len(rows)
            unconditional.append([float(overlap.mean()), float(overlap.sum() / visible.sum())])
            score_arrays = dict(raw=np.asarray([1.] + [r['score'] for r in rows[1:]]),
                constant=np.ones(len(rows)), GT_overlap_oracle=np.concatenate(([1.], overlap[1:])))
            for readout, scores in score_arrays.items():
                output = ROOT / (arm + '_' + readout)
                np.savetxt(output / (seq + '.txt'), boxes, fmt='%.6f', delimiter=',')
                np.savetxt(output / (seq + '_all_scores.txt'), scores, fmt='%.6f')
                files[arm][readout][seq] = dict(bbox_sha256=sha(output / (seq + '.txt')), score_sha256=sha(output / (seq + '_all_scores.txt')))
            assert len({files[arm][x][seq]['bbox_sha256'] for x in spec['readouts']}) == 1
        for readout in spec['readouts']:
            result = evaluator.evaluate_depthtrack_results(training['dataset_root'], ROOT / (arm + '_' + readout),
                resolution=100, sequence_names=[c['sequence'] for c in spec['cases']])
            assert result['sequences'] == 22 and result['frames'] == 33130
            results[arm][readout] = result
            per_sequence[arm][readout] = {c['sequence']: evaluator.evaluate_depthtrack_results(training['dataset_root'], ROOT / (arm + '_' + readout),
                resolution=100, sequence_names=[c['sequence']]) for c in spec['cases']}
        mean_p, mean_r = np.mean(unconditional, axis=0); f = 2 * mean_p * mean_r / (mean_p + mean_r)
        for key, value in [('precision', mean_p), ('recall', mean_r), ('f_score', f)]:
            assert abs(results[arm]['constant'][key] - value) < 1e-12
        constant_checks[arm] = dict(macro_precision=float(mean_p), macro_recall=float(mean_r), f_score=float(f))
    result = dict(status='completed_historical_Train_confidence_capacity_reference', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'), metric_sha256=METRIC_SHA,
        source_family_receipts=receipts, metrics=results, per_sequence_individual_optimum_metrics=per_sequence,
        constant_score_analytical_checks=constant_checks, prediction_and_score_files=files,
        deltas={a: {x + '_minus_raw_F_pp': results[a][x]['f_score_percent'] - results[a]['raw']['f_score_percent']
            for x in ['constant', 'GT_overlap_oracle']} for a in spec['arms']},
        boxes_identical_across_readouts=True, new_tracking_calls=0, new_optimizer_steps=0,
        GT_oracle_is_diagnostic_only=True, oracle_is_not_proved_global_optimum=True,
        M65_failed_promotion_unchanged=True, M67_or_M68_source_modified=False, public_evaluation_allowed=False,
        independent_model_review_pass=False, scope='Repeatedly used DepthTrack Train development22; no new Test/CDTB/VOT metrics or deployable oracle.')
    write(ROOT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['per_sequence_individual_optimum_metrics', 'prediction_and_score_files', 'source_family_receipts']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run']); args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
