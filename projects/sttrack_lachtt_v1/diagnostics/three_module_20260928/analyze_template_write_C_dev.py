"""CPU only: compare full C development trajectories with bound native prefixes.

GT is loaded after predictions exist. This is Train/C selection evidence,
not official Test/CDTB/VOT, and teacher utility labels stay on teacher states.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SPEC_SHA = '3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425'
SPLIT_SHA = 'ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def overlaps(boxes, truth):
    valid = np.isfinite(truth).all(axis=1) & (truth[:, 2:] > 0).all(axis=1)
    result = np.zeros(len(truth))
    left = np.maximum(boxes[valid, :2], truth[valid, :2])
    right = np.minimum(boxes[valid, :2] + boxes[valid, 2:], truth[valid, :2] + truth[valid, 2:])
    intersection = np.maximum(0, right - left).prod(axis=1)
    union = boxes[valid, 2:].prod(axis=1) + truth[valid, 2:].prod(axis=1) - intersection
    result[valid] = intersection / union
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result, valid


def summarize(sequence, rows, case, truth):
    assert [row['frame'] for row in rows] == list(range(1, case['rgb_frames']))
    boxes = np.asarray([row['bbox'] for row in rows], dtype=np.float64)
    assert boxes.shape == (case['rgb_frames'] - 1, 4) and np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
    values, valid = overlaps(boxes, truth[1:])
    assert valid.any()
    severe = valid & (values <= .1)
    edges = np.diff(np.r_[False, severe, False].astype(np.int8))
    segments = [[int(a) + 1, int(b) + 1] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b - a >= 10]
    writes = np.asarray([row['template_write'] for row in rows], dtype=bool)
    qualified = np.asarray([row['frame'] % 50 == 0 and row['native_same_position_response'] > .75 for row in rows], dtype=bool)
    assert not (writes & ~qualified).any()
    return dict(sequence=sequence, frames=len(rows), valid_GT_frames=int(valid.sum()),
        IoU_sum=float(values[valid].sum()), mean_IoU=float(values[valid].mean()),
        low_overlap_frames=int(severe.sum()), H10_segments=len(segments), H10_zero_based_half_open=segments,
        native_rule_qualified_events=int(qualified.sum()), accepted_writes=int(writes.sum()),
        accepted_writes_unknown_GT=int((writes & ~valid).sum()),
        accepted_writes_low_IoU=int((writes & severe).sum()))


def load_predictions(path, sequences):
    result = {sequence: [] for sequence in sequences}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sequence'] in result:
                result[row['sequence']].append(row)
    return result


def fixed_panel(root, arm):
    fitted = read(root / 'result.json')
    path = root / 'fixed_teacher_development_predictions.json'
    assert sha(path) == fitted['fixed_teacher_development_predictions_sha256']
    panel = read(path)
    assert panel['arm'] == arm and fitted['arm'] == arm
    rows = panel['events']
    positive = [row for row in rows if row['future'] > 0]
    negative = [row for row in rows if row['future'] < 0]
    zeros = [row for row in rows if row['future'] == 0]
    return dict(scope=panel['scope'], common_eligible_events=len(rows),
        accepted=sum(row['accepted_at_fixed_teacher_state'] for row in rows),
        positive_future_events=len(positive), positive_future_events_accepted=sum(row['accepted_at_fixed_teacher_state'] for row in positive),
        negative_future_events=len(negative), negative_future_events_accepted=sum(row['accepted_at_fixed_teacher_state'] for row in negative),
        zero_future_events=len(zeros), zero_future_events_accepted=sum(row['accepted_at_fixed_teacher_state'] for row in zeros),
        excluded_count_scope=panel['excluded_count_scope'],
        unknown_or_tail_events_excluded=panel['unknown_or_tail_events_excluded'])


def main():
    parser = argparse.ArgumentParser()
    for name in ['teacher-root', 'teacher-audit', 'current-fit', 'future-fit', 'current-predictions', 'future-predictions', 'spec', 'split', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.spec) == SPEC_SHA and sha(args.split) == SPLIT_SHA
    spec, split = read(args.spec), read(args.split)
    audit = read(args.teacher_audit)
    assert audit['review_call_status'] == 'completed' and audit['blocking_count'] == 0
    assert audit['R2_terminal_raw_audit_complete']
    audit_sha = sha(args.teacher_audit)
    teacher_sha = sha(args.teacher_root / 'result.json')
    assert teacher_sha == audit['audited_teacher_result_sha256']
    complete = read(args.teacher_root / 'result.json')
    assert complete['status'] == 'complete_M122_full_train_template_teacher' and (args.teacher_root / 'controller.exit').read_text().strip() == '0'
    assert complete['inputs']['spec_sha256'] == SPEC_SHA
    cases = [case for index, case in enumerate(spec['sequence_order']) if split['sequence_order'][index]['C_split'] == 'development']
    assert len(cases) == 32 and [case['sequence'] for case in spec['sequence_order']] == [row['sequence'] for row in split['sequence_order']]
    sequences = [case['sequence'] for case in cases]
    native = {sequence: [] for sequence in sequences}; source_hashes = {}
    for shard in [0, 1]:
        folder = args.teacher_root / ('teacher_shard%d' % shard)
        collected = read(folder / 'teacher_events.json')
        assert collected['inputs'] == complete['inputs'] and collected['shard'] == shard
        path = folder / 'native_prefix.jsonl'
        prefix_sha = sha(path)
        assert prefix_sha == audit['audited_teacher_prefixes_sha256'][shard]
        source_hashes[str(path)] = prefix_sha
        for sequence, rows in load_predictions(path, sequences).items():
            native[sequence].extend(rows)
    trajectories = {'native': native}
    fitted_results = []
    for arm, fitted_root, prediction_root in [('current', args.current_fit, args.current_predictions), ('future', args.future_fit, args.future_predictions)]:
        fit = read(fitted_root / 'result.json'); receipt = read(prediction_root / 'result.json')
        assert fit['status'] == 'complete_M122_C_fixed_teacher_fit120' and fit['arm'] == arm
        assert fit['teacher_result_sha256'] == teacher_sha and fit['teacher_audit_sha256'] == audit_sha
        assert receipt['status'] == 'complete_M122_C_development32_own_history_predictions' and receipt['arm'] == arm
        assert receipt['inputs'] == complete['inputs'] and receipt['split_sha256'] == SPLIT_SHA
        assert receipt['C_final_sha256'] == fit['final_sha256'] == sha(fitted_root / 'final.pt')
        assert receipt['C_result_sha256'] == sha(fitted_root / 'result.json')
        path = prediction_root / 'predictions.jsonl'
        assert sha(path) == receipt['predictions_sha256']
        source_hashes[str(path)] = sha(path)
        trajectories[arm] = load_predictions(path, sequences); fitted_results.append(fit)
    assert fitted_results[0]['common_dataset_sha256'] == fitted_results[1]['common_dataset_sha256']
    assert fitted_results[0]['initial_state_sha256'] == fitted_results[1]['initial_state_sha256']
    assert fitted_results[0]['optimizer_steps'] == fitted_results[1]['optimizer_steps']
    assert fitted_results[0]['order_sha256'] == fitted_results[1]['order_sha256']
    summaries = {name: [] for name in trajectories}
    for case in cases:
        path = Path(spec['dataset_root']) / case['sequence'] / 'groundtruth.txt'
        assert sha(path) == case['groundtruth_sha256']
        truth = np.loadtxt(path, delimiter=',').reshape(-1, 4)
        if case['sequence'] == 'toy07_indoor_320':
            assert len(truth) == 1406 and case['rgb_frames'] == 1367
            truth = truth[:1367]
        assert len(truth) == case['rgb_frames'] and np.array_equal(truth[0], np.asarray(case['first_box']))
        source_hashes[str(path)] = sha(path)
        for name, rows in trajectories.items():
            summaries[name].append(summarize(case['sequence'], rows[case['sequence']], case, truth))
    aggregates = {}
    for name, rows in summaries.items():
        aggregates[name] = dict(sequences=len(rows), valid_frames=sum(r['valid_GT_frames'] for r in rows),
            frame_mean_IoU=sum(r['IoU_sum'] for r in rows) / sum(r['valid_GT_frames'] for r in rows),
            sequence_equal_mean_IoU=float(np.mean([r['mean_IoU'] for r in rows])),
            low_overlap_frames=sum(r['low_overlap_frames'] for r in rows), H10_segments=sum(r['H10_segments'] for r in rows),
            native_rule_qualified_events=sum(r['native_rule_qualified_events'] for r in rows),
            accepted_writes=sum(r['accepted_writes'] for r in rows), accepted_writes_unknown_GT=sum(r['accepted_writes_unknown_GT'] for r in rows),
            accepted_writes_low_IoU=sum(r['accepted_writes_low_IoU'] for r in rows), per_sequence=rows)
    panels = {arm: fixed_panel(root, arm) for arm, root in [('current', args.current_fit), ('future', args.future_fit)]}
    native, current, future = [aggregates[name] for name in ['native', 'current', 'future']]
    passed = future['sequence_equal_mean_IoU'] > native['sequence_equal_mean_IoU'] and future['sequence_equal_mean_IoU'] > current['sequence_equal_mean_IoU'] and future['H10_segments'] <= native['H10_segments']
    args.output.mkdir()
    result = dict(status='complete_M122_C_development32_comparison', aggregates=aggregates, fixed_teacher_panels=panels,
        prescribed_macro_IoU_and_H10_gate_passed=bool(passed),
        gate_requires_fresh_result_audit_before_R4=True, coverage_review_required=True,
        split_sha256=SPLIT_SHA, spec_sha256=SPEC_SHA, teacher_result_sha256=teacher_sha,
        teacher_audit_sha256=audit_sha,
        source_hashes=source_hashes, source_sha256=sha(Path(__file__)),
        metrics_scope='noninitialization frames with valid dataset GT; Train/C optimization holdout, A+B previously saw all152',
        H10_is_not_VOT_ROB=True, identity_or_crop_out_recovery_claim=False, formal_nine_metrics=False,
        fixed_teacher_labels_not_transferred_to_changed_C_history=True)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'],gate_passed=passed,
        macro_IoU={k:v['sequence_equal_mean_IoU'] for k,v in aggregates.items()},H10={k:v['H10_segments'] for k,v in aggregates.items()})),flush=True)


if __name__ == '__main__':
    main()
