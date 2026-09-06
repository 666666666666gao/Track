"""Read sealed M58 results and audit completed training and development trajectories."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
OUT = Path('/root/autodl-tmp/sttrack_m58_completion_audit_20260906')
CF = Path('/root/autodl-tmp/sttrack_m58_content_controls_v1_20260906')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def values(boxes, gt):
    valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
    valid[0] = False
    a, b = boxes[valid], gt[valid]
    inter = np.maximum(0, np.minimum(a[:, :2] + a[:, 2:], b[:, :2] + b[:, 2:]) - np.maximum(a[:, :2], b[:, :2])).prod(1)
    ious = np.full(len(gt), np.nan)
    ious[valid] = inter / (a[:, 2:].prod(1) + b[:, 2:].prod(1) - inter)
    return valid, ious


def main():
    started = time.time()
    torch.set_num_threads(4)
    OUT.mkdir()
    assert sha(PARENT / 'recursive_result.json') == '54fd8445297e1eef761fa36e8a17def01afa027f71371984998f08229f6f5ee9'
    sys.path.insert(0, str(PARENT))
    from run_recursive import trained
    from recursive_metric import statistics
    recursive, training, trained_results = trained()
    result = json.loads((PARENT / 'recursive_result.json').read_text())
    assert not result['primary_pass'] and not result['gates']['H10']
    assert (PARENT / 'recursive_analysis.exit').read_text().strip() == '0'
    assert (PARENT / 'post_training_queue.exit').read_text().strip() == '0'
    cf = json.loads((CF / 'queue_result.json').read_text())
    assert cf['status'] == 'not_run_parent_gate_failed'
    assert cf['parent_result_sha256'] == sha(PARENT / 'recursive_result.json')
    assert cf['actual_control_tracking_calls'] == 0 and not cf['candidate_bundle_created']

    initial_path = PARENT / 'native_parity/text_zero.pth'
    assert sha(initial_path) == training['initial_checkpoint_sha256']
    initial = torch.load(initial_path, map_location='cpu')['model']
    checked_weights, models = {}, {}
    for arm in ['text', 'visual']:
        folder = PARENT / 'training' / arm
        final = torch.load(folder / 'final.pth', map_location='cpu')
        assert final['status'] == 'complete' and final['completed_sequences'] == 130
        assert final['optimizer_steps'] == 5798 and final['frame_count'] == 186694
        assert final['use_text'] == (arm == 'text') and final['seed'] == training['seed']
        assert final['training_spec_sha256'] == sha(PARENT / 'training_spec.json')
        model = final['model']
        assert set(model) == set(initial)
        assert sum(v.numel() for v in model.values()) == 289154
        for name, tensor in model.items():
            assert tensor.shape == initial[name].shape and torch.isfinite(tensor).all()
        optimizer_tensors = [v for state in final['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v)]
        assert optimizer_tensors and all(torch.isfinite(v).all() for v in optimizer_tensors)
        logs = [json.loads(line) for line in (folder / 'sequence_log.jsonl').read_text().splitlines()]
        assert sha(folder / 'sequence_log.jsonl') == trained_results[arm]['sequence_log_sha256']
        assert sha(folder / 'sampled_state_trace.jsonl') == trained_results[arm]['sampled_trace_sha256']
        assert [r['sequence'] for r in logs] == training['sequence_order']
        assert [r['sequence_index'] for r in logs] == list(range(130))
        assert sum(r['track_calls'] for r in logs) == 186694
        assert sum(r['frames'] for r in logs) == 186824
        counts = {k: sum(r['label_counts'][k] for r in logs) for k in trained_results[arm]['training_label_counts']}
        assert counts == trained_results[arm]['training_label_counts']
        assert logs[-1]['total_optimizer_steps'] == final['optimizer_steps']
        changed = [n for n in model if not torch.equal(model[n], initial[n])]
        assert changed
        checked_weights[arm] = dict(final_checkpoint_sha256=sha(folder / 'final.pth'),
            final_checkpoint_bytes=(folder / 'final.pth').stat().st_size,
            tensor_count=len(model), learned_parameters=sum(v.numel() for v in model.values()),
            all_parameters_and_optimizer_tensors_finite=True, changed_tensors_from_initial=changed,
            l2_change_from_initial=float(torch.sqrt(sum((model[n].double() - initial[n].double()).square().sum() for n in model))),
            complete_sequence_log_verified=True, optimizer_steps=final['optimizer_steps'],
            track_calls=final['frame_count'], label_counts=counts)
        models[arm] = model
    between_l2 = float(torch.sqrt(sum((models['text'][n].double() - models['visual'][n].double()).square().sum() for n in initial)))
    assert between_l2 > 0

    names = [c['sequence'] for c in recursive['cases']]
    sealed = {'native': {}, 'text': {}, 'visual': {}}
    scores = {'text': {}, 'visual': {}}
    for arm in ['text', 'visual']:
        receipt_path = PARENT / (arm + '_recursive_receipt.json')
        assert sha(receipt_path) == result['receipts'][arm]
        receipt = json.loads(receipt_path.read_text())
        assert receipt['total_frames'] == 33130 and not receipt['subsequent_gt_opened']
        assert receipt['head_sha256'] == checked_weights[arm]['final_checkpoint_sha256']
        assert [r['sequence'] for r in receipt['sequences']] == names
        assert (PARENT / (arm + '_recursive.exit')).read_text().strip() == '0'
        for entry in receipt['sequences']:
            path = PARENT / 'recursive' / arm / (entry['sequence'] + '.json')
            assert sha(path) == entry['sha256']
            rows = json.loads(path.read_text())['rows']
            assert [r['frame'] for r in rows] == list(range(entry['frames']))
            boxes = np.asarray([r['bbox'] for r in rows])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            case = next(c for c in recursive['cases'] if c['sequence'] == entry['sequence'])
            assert boxes[0].tolist() == case['init_bbox']
            sealed[arm][entry['sequence']] = boxes
            scores[arm][entry['sequence']] = [r['score'] for r in rows]
    native_spec_path = Path(recursive['native_result_path']).parent / 'recursive_spec.json'
    native_spec = json.loads(native_spec_path.read_text())
    baseline_rows = defaultdict(list)
    for path, digest in native_spec['baseline_trace_sha256'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in names:
                baseline_rows[row['sequence']].append(row)
    for case in recursive['cases']:
        rows = sorted(baseline_rows[case['sequence']], key=lambda r: r['frame_index'])
        assert [r['frame_index'] for r in rows] == list(range(case['frames']))
        sealed['native'][case['sequence']] = np.asarray([r['public_bbox'] for r in rows])

    episodes, sequence_table = [], []
    totals = {a: dict(h10_low_frames=0, maximum_episode_length=0, h10_episodes=0) for a in sealed}
    for case in recursive['cases']:
        name = case['sequence']
        path = Path(training['dataset_root']) / name / 'groundtruth.txt'
        assert sha(path) == case['gt_sha256']
        gt = np.loadtxt(path, delimiter=',').reshape(-1, 4)
        assert len(gt) == case['frames']
        val = {}
        for arm in sealed:
            boxes = sealed[arm][name]
            current = statistics(boxes, gt)
            prior = result['per_sequence'][arm][name]
            for key in ['valid_frames', 'low_iou_frames', 'failure_episodes', 'invalid_gt_frames']:
                assert current[key] == prior[key]
            assert abs(current['iou_sum'] - prior['iou_sum']) < 1e-8
            val[arm] = values(boxes, gt)
        for arm in sealed:
            valid, iou = val[arm]
            selected = [(a, b) for a, b in spans(valid & (iou <= .1)) if b - a >= 10]
            assert len(selected) == result['per_sequence'][arm][name]['failure_episodes']
            for a, b in selected:
                row = dict(sequence=name, arm=arm, start_frame_zero_based=a,
                    end_frame_exclusive=b, length=b-a, start_score=scores[arm][name][a] if arm != 'native' else None,
                    previous_gt_valid=bool(valid[a-1]), next_gt_valid=bool(valid[b]) if b < len(gt) else None,
                    counterpart_window={})
                for other in sealed:
                    o = val[other][1][a:b]
                    assert np.isfinite(o).all()
                    row['counterpart_window'][other] = dict(mean_iou=float(o.mean()),
                        correct_frames_at_iou_05=int((o >= .5).sum()), low_frames=int((o <= .1).sum()),
                        start_iou=float(o[0]))
                episodes.append(row)
                totals[arm]['h10_low_frames'] += b-a
                totals[arm]['maximum_episode_length'] = max(totals[arm]['maximum_episode_length'], b-a)
                totals[arm]['h10_episodes'] += 1
        sequence_table.append(dict(sequence=name, valid_frames=result['per_sequence']['native'][name]['valid_frames'],
            arms={a: result['per_sequence'][a][name] for a in sealed},
            text_minus_native_iou=result['per_sequence']['text'][name]['mean_iou']-result['per_sequence']['native'][name]['mean_iou'],
            text_minus_visual_iou=result['per_sequence']['text'][name]['mean_iou']-result['per_sequence']['visual'][name]['mean_iou']))
    comparisons = {}
    for baseline in ['native', 'visual']:
        a, b = result['aggregates']['text'], result['aggregates'][baseline]
        leave_one_out = {}
        for name in names:
            x, y = result['per_sequence']['text'][name], result['per_sequence'][baseline][name]
            leave_one_out[name] = ((a['iou_sum']-x['iou_sum']) - (b['iou_sum']-y['iou_sum']))/(a['valid_frames']-x['valid_frames'])
        comparisons[baseline] = dict(pooled_mean_iou_delta=a['mean_iou']-b['mean_iou'],
            relative_mean_iou_change=(a['mean_iou']/b['mean_iou']-1),
            macro_sequence_mean_iou_delta=a['macro_sequence_mean_iou']-b['macro_sequence_mean_iou'],
            positive_sequences=sum(r['text_minus_'+baseline+'_iou'] > 0 for r in sequence_table),
            negative_sequences=sum(r['text_minus_'+baseline+'_iou'] < 0 for r in sequence_table),
            leave_one_sequence_out_pooled_delta=leave_one_out,
            negative_leave_one_out_sequences=[n for n, d in leave_one_out.items() if d < 0])
    distinct = {}
    for other in ['native', 'visual']:
        selected = [e for e in episodes if e['arm']=='text' and e['counterpart_window'][other]['correct_frames_at_iou_05']==e['length']]
        distinct[other] = dict(text_h10_episodes_with_counterpart_correct_entire_window=len(selected),
            total_frames=sum(e['length'] for e in selected), episodes=selected)
    write(OUT / 'episode_intervals.json', episodes)
    write(OUT / 'sequence_table.json', sequence_table)
    audit = dict(status='sealed_training_and_recursive_results_verified', observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), parent_result_sha256=sha(PARENT / 'recursive_result.json'),
        frozen_inference_sources_verified=160, training=checked_weights, between_arm_parameter_l2=between_l2,
        baseline_trace_sha256=native_spec['baseline_trace_sha256'],
        complete_trajectory_families=3, sequences_per_family=22, frames_per_family=33130,
        native_and_both_candidate_statistics_recomputed_and_equal=True,
        aggregates=result['aggregates'], gates=result['gates'], primary_pass=False,
        episode_summary=totals, distinct_harms=distinct, comparisons=comparisons,
        episode_intervals_sha256=sha(OUT/'episode_intervals.json'), sequence_table_sha256=sha(OUT/'sequence_table.json'),
        content_queue_status=cf['status'], frozen_content_controls_run=False,
        formal_public_dataset_evaluation_run=False, independent_review_pass=False,
        scope='Posthoc analysis of one seed on reused DepthTrack Train development22. Episode overlap is descriptive, not intervention causality.',
        low_definition='valid continuous IoU <= 0.1; H10 >=10 consecutive original timeline frames; invalid GT breaks runs; frame0 excluded',
        free_disk_bytes=shutil.disk_usage(OUT).free, elapsed_seconds=time.time()-started)
    write(OUT / 'audit_result.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['training','distinct_harms','comparisons']},indent=2))


if __name__ == '__main__':
    main()
