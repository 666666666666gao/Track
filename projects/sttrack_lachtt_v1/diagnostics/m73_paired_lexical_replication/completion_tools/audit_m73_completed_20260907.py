"""Complete-checkpoint and scalar-metric audit for the frozen two-seed M73 study."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

B = Path('/root/autodl-tmp')
ROOT = B / 'sttrack_m73_paired_lexical_replication_20260907'
OUT = ROOT / 'completion_tools'
OLD = B / 'audit_m67_completed_20260907.py'
OLD_SHA = '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
SPEC_SHA = '9b5203ad2c6f890ab55105a7738a46b212e57ec411e5490a78a0435545bb44e7'
FROZEN_SHA = '27de59df77bf12383747f39f84583c0386bcf57f90dd1a345451a498b00f054f'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def checked():
    assert sha(ROOT / 'spec.json') == SPEC_SHA and sha(ROOT / 'frozen.json') == FROZEN_SHA
    spec = read(ROOT / 'spec.json'); frozen = read(ROOT / 'frozen.json')
    assert sha(B / 'm73_paired_lexical_replication_20260907.py') == spec['source_sha256']
    assert sha(ROOT / 'run_m73.sh') == spec['run_queue_sha256']
    for seed in [2027, 2028]:
        root = ROOT / ('seed' + str(seed)); bound = frozen['seeds'][str(seed)]
        assert sha(root / 'training_spec.json') == bound['training_spec_sha256']
        assert sha(root / 'recursive_spec.json') == bound['recursive_spec_sha256']
        t = read(root / 'training_spec.json'); rs = read(root / 'recursive_spec.json')
        assert sha(root / 'train_causal.py') == t['training_script_sha256']
        assert sha(root / 'causal_training.py') == t['causal_script_sha256']
        assert sha(root / 'run_recursive.py') == rs['runner_sha256'] and sha(root / 'run_pair.sh') == rs['queue_sha256']
        assert sha(root / 'recursive_metric.py') == rs['metric_sha256']
        assert sha(root / 'integration.json') == t['integration_sha256']
        for n, h in read(root / 'integration.json')['source_sha256'].items(): assert sha(root / 'code' / n) == h
        for split in ['fit', 'development']:
            for arm in ['category', 'empty']:
                bank = t['banks'][split][arm]; assert sha(bank['path']) == bank['sha256']
    assert sha(OLD) == OLD_SHA
    s = importlib.util.spec_from_file_location('m73_sealed_scalar_reference', str(OLD))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return spec, frozen, m


def state_sha(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(('semantic_adapter.' + name).encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def training_arm(root, arm, training):
    import torch
    assert (root / ('training_' + arm + '.exit')).read_text().strip() == '0'
    folder = root / 'training' / arm; r = read(folder / 'result.json')
    assert r['status'] == 'one_full_causal_fit_pass_complete' and r['arm'] == arm
    assert r['sequences'] == 130 and r['total_track_calls'] == 186694 and r['learned_parameters'] == 289154
    assert r['training_spec_sha256'] == sha(root / 'training_spec.json')
    assert r['base_parameters_and_buffers_unchanged'] and r['gt_after_prediction_for_loss_only']
    assert not r['gt_reinitialization_after_first_frame'] and not r['backward_through_crops_or_time']
    assert r['support_loss_weight'] == training['support_loss_weights'][arm] == 0.
    for n, k in [('final.pth', 'final_checkpoint_sha256'), ('sequence_log.jsonl', 'sequence_log_sha256'), ('sampled_state_trace.jsonl', 'sampled_trace_sha256')]: assert sha(folder / n) == r[k]
    initial_path = root / 'native_parity' / (arm + '_zero.pth')
    assert sha(initial_path) == training['initial_checkpoint_sha256'][arm]
    initial = torch.load(initial_path, map_location='cpu'); final = torch.load(folder / 'final.pth', map_location='cpu'); latest = torch.load(folder / 'latest.pth', map_location='cpu')
    assert all(v['architecture'] == 'semantic_spatial_support_v1' and v['use_text'] and v['null_support'] for v in [initial, final, latest])
    assert final['status'] == 'complete' and final['completed_sequences'] == 130 and final['frame_count'] == 186694
    assert final['seed'] == training['seed'] and final['base_checkpoint_sha256'] == training['native_checkpoint_sha256']
    assert final['training_spec_sha256'] == sha(root / 'training_spec.json')
    assert final['optimizer_steps'] == final['actual_dataset_optimizer_steps'] == r['optimizer_steps']
    assert final['support_loss_weight'] == latest['support_loss_weight'] == 0.
    assert initial['model'].keys() == final['model'].keys() == latest['model'].keys()
    assert sum(v.numel() for v in final['model'].values()) == 289154
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    assert all(torch.equal(v, latest['model'][k]) for k, v in final['model'].items())
    assert any(not torch.equal(v, initial['model'][k]) for k, v in final['model'].items())
    assert state_sha(initial['model']) == r['initial_adapter_state_sha256']
    assert len(final['optimizer']['state']) == len(final['model'])
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {r['optimizer_steps']}
    records = [json.loads(x) for x in (folder / 'sequence_log.jsonl').read_text().splitlines()]
    traces = [json.loads(x) for x in (folder / 'sampled_state_trace.jsonl').read_text().splitlines()]
    assert [x['sequence'] for x in records] == [x['sequence'] for x in training['sequence_order']]
    grouped = {x['sequence']: [] for x in records}
    for row in traces: grouped[row['sequence']].append(row)
    calls = steps = writes = supervised = 0; labels = Counter()
    for index, (case, record) in enumerate(zip(training['sequence_order'], records)):
        seq = case['sequence']; n = case['rgb_frames']; p = Path(training['dataset_root']) / seq / 'groundtruth.txt'
        assert sha(p) == case['groundtruth_sha256']; gt = np.loadtxt(p, delimiter=',').reshape(-1, 4)
        if seq == 'toy07_indoor_320':
            assert len(gt) == 1406 and n == 1367 and case['groundtruth_sha256'] == '683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
            gt = gt[:n]
        assert len(gt) == n
        valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1); valid[0] = False
        calls += n - 1; steps += sum(bool(valid[f:min(f+32, n)].any()) for f in range(1, n, 32))
        supervised += int(valid.sum()); labels.update(record['label_counts'])
        assert record['sequence_index'] == index and record['frames'] == n and record['total_sequences'] == 130
        assert record['total_track_calls'] == calls and record['total_optimizer_steps'] == steps
        assert record['supervised_frames'] == int(valid.sum()) and record['track_calls'] == n - 1
        assert sum(record['label_counts'].values()) == n - 1 and record['label_counts'].get('invalid', 0) == int((~valid[1:]).sum())
        assert math.isfinite(record['mean_training_loss']) and math.isfinite(record['maximum_preclip_gradient_norm'])
        sampled = grouped[seq]; assert [x['frame_index'] for x in sampled] == sorted({1, n-1} | set(range(50, n, 50)))
        count = 0
        for x in sampled:
            f = x['frame_index']; assert math.isfinite(x['best_score']) and 0 <= x['best_score'] <= 1
            assert x['template_write'] == (f % 50 == 0 and x['best_score'] > .75)
            count += x['template_write']; assert (x['label'] == 'invalid') == (not valid[f])
            assert 'support_loss' not in x and 'support_weight' not in x
            for k in ['previous_bbox', 'bbox']: assert np.isfinite(x[k]).all() and np.asarray(x[k])[2:].min() > 0
        assert record['template_writes'] == count; writes += count
    assert calls == 186694 and steps == r['optimizer_steps'] and dict(labels) == r['training_label_counts']
    return dict(result_sha256=sha(folder / 'result.json'), head_sha256=sha(folder / 'final.pth'),
        initial_tensor_sha256=state_sha(initial['model']), final_tensor_sha256=state_sha(final['model']),
        base_state_sha256=r['base_state_before_sha256'], sequences=130, track_calls=calls, optimizer_steps=steps,
        supervised_frames=supervised, sampled_rows=len(traces), template_writes=writes,
        base_freeze_scope='Runtime full-base tensor/hash assertion at training end; not a new independent full training replay.')


def reference():
    assert not (OUT / 'reference.json').exists()
    _, _, old = checked(); root = B / 'sttrack_m67_supervised_semantic_support_20260907'
    assert sha(root / 'recursive_result.json') == '9f2cfe2457a967fdfa2ba9beae737b6a1575e7224c4029ee438472e547db4d07'
    values = old.recompute(root, ['control', 'support'], read(root / 'recursive_result.json'))
    train = training_arm(root, 'control', read(root / 'training_spec.json'))
    r = dict(status='completed_M73_auditor_historical_reference_check', observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), frozen_M73_spec_sha256=SPEC_SHA, historical_result_sha256=sha(root / 'recursive_result.json'),
        reference_training=train, scalar_aggregates=values[3], actual_M73_final_outputs_opened=False, new_model_calls=0,
        independent_model_review_pass=False)
    OUT.mkdir(exist_ok=True); write(OUT / 'reference.json', r); print(json.dumps(r))


def completed():
    _, frozen, old = checked()
    assert not (OUT / 'completed.json').exists()
    ref = read(OUT / 'reference.json'); assert ref['auditor_sha256'] == sha(__file__)
    assert (ROOT / 'controller.exit').read_text().strip() == '0'
    parent = read(ROOT / 'result.json'); assert parent['status'] == 'completed_two_seed_paired_lexical_replication'
    assert parent['spec_sha256'] == SPEC_SHA and parent['candidate_seed'] == 2027
    legacy_path = B / 'sttrack_m65_category_null_support_20260907/recursive_result.json'
    assert sha(legacy_path) == '0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
    legacy = read(legacy_path); outputs = {}
    for seed in [2027, 2028]:
        root = ROOT / ('seed' + str(seed)); result = read(root / 'recursive_result.json')
        assert (root / 'controller.exit').read_text().strip() == '0'
        assert parent['results'][str(seed)]['result_sha256'] == sha(root / 'recursive_result.json')
        spec, training, per, aggregate, overlap, intervals, families = old.recompute(root, ['category', 'empty'], result)
        train = {arm: training_arm(root, arm, training) for arm in ['category', 'empty']}
        for k in ['initial_tensor_sha256', 'base_state_sha256', 'track_calls', 'optimizer_steps', 'supervised_frames']:
            assert train['category'][k] == train['empty'][k]
        primary = aggregate['category']; control = aggregate['empty']; native = aggregate['native']; rule = training['promotion_gates']
        protected = [n for n, v in legacy['per_sequence']['control'].items() if v['failure_episodes'] == 0]
        assert protected == training['protected_prior_control_sequences']
        broken = {a: [n for n, v in per[a].items() if v['failure_episodes'] == 0 and per['category'][n]['failure_episodes'] > 0] for a in ['native', 'empty']}
        broken['prior_control'] = [n for n in protected if per['category'][n]['failure_episodes'] > 0]
        gates = dict(prior_control_success_protection=not broken['prior_control'],
            mean_vs_native=primary['mean_iou'] >= native['mean_iou'] + rule['category_pooled_mean_vs_native_minimum'],
            mean_vs_control=primary['mean_iou'] >= control['mean_iou'] + rule['category_pooled_mean_vs_control_minimum'],
            macro_vs_native=primary['macro_sequence_mean_iou'] >= native['macro_sequence_mean_iou'],
            macro_vs_control=primary['macro_sequence_mean_iou'] >= control['macro_sequence_mean_iou'],
            low_frames_vs_native=primary['low_iou_frames'] <= native['low_iou_frames'], low_frames_vs_control=primary['low_iou_frames'] <= control['low_iou_frames'],
            H10_vs_native=primary['failure_episodes'] <= native['failure_episodes'], H10_vs_control=primary['failure_episodes'] <= control['failure_episodes'],
            native_success_protection=not broken['native'], control_success_protection=not broken['empty'])
        assert gates == result['gates'] and all(gates.values()) == result['primary_pass']
        assert broken['native'] == result['new_failure_sequences'] and broken['empty'] == result['broken_control_success_sequences']
        assert broken['prior_control'] == result['broken_prior_control_success_sequences']
        harms = []
        for baseline in ['native', 'empty']:
            for seq, runs in intervals['category'].items():
                for start, end in runs:
                    if np.all(overlap[baseline][seq][start:end] >= .5): harms.append(dict(sequence=seq, reference=baseline, start=start, end_exclusive=end, frames=end-start))
        outputs[str(seed)] = dict(result_sha256=sha(root / 'recursive_result.json'), training=train, families=families,
            aggregates=aggregate, gates=gates, gate_pass=all(gates.values()), broken_success_sequences=broken,
            sustained_category_H10_with_reference_correct_every_frame=harms)
    passed = all(x['gate_pass'] for x in outputs.values()); assert passed == parent['both_seed_development_gates_pass']
    assert outputs['2027']['training']['category']['initial_tensor_sha256'] != outputs['2028']['training']['category']['initial_tensor_sha256']
    r = dict(status='completed_M73_two_seed_checkpoint_and_scalar_audit', observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__), spec_sha256=SPEC_SHA, frozen_sha256=FROZEN_SHA, parent_result_sha256=sha(ROOT / 'result.json'),
        historical_reference_sha256=sha(OUT / 'reference.json'), seeds=outputs, both_seed_development_gates_pass=passed,
        content_counterfactuals_allowed=passed, candidate_seed=2027,
        candidate_head_sha256=outputs['2027']['training']['category']['head_sha256'],
        public_evaluation_allowed=False, new_tracking_calls=0, new_optimizer_steps=0, independent_model_review_pass=False)
    write(OUT / 'completed.json', r); print(json.dumps({k: v for k, v in r.items() if k != 'seeds'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['check', 'reference', 'completed']); a = p.parse_args()
    if a.action == 'check': checked(); print('M73_FROZEN_INPUTS_VERIFIED')
    elif a.action == 'reference': reference()
    else: completed()
