"""Read-only sampled training mechanism audit; no checkpoint or hyperparameter selection."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/root/autodl-tmp/sttrack_m67_supervised_semantic_support_20260907')
OUT = ROOT / 'quarter_prefix_34'
TRAIN_SHA = '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
COUNT = 34


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    import numpy as np
    assert sha(ROOT / 'training_spec.json') == TRAIN_SHA
    training = json.loads((ROOT / 'training_spec.json').read_text())
    for name, key in [('train_causal.py', 'training_script_sha256'), ('causal_training.py', 'causal_script_sha256'), ('support_loss.py', 'support_loss_sha256')]:
        assert sha(ROOT / name) == training[key]
    cases = training['sequence_order'][:COUNT]; names = [x['sequence'] for x in cases]
    assert len(names) == COUNT and all(c['split'] == 'fit' for c in cases)
    OUT.mkdir()
    spec = dict(status='fixed_first34_training_prefix_before_statistical_analysis', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), training_spec_sha256=TRAIN_SHA, sequences=names,
        selection='The first 34 sequences already confirmed complete in both arms at 11:36 CST. No selection by null probability, loss or outcome.',
        expected_completed_calls=48421, expected_completed_optimizer_steps=1512,
        bins=[[0, 250], [250, 500], [500, 1000], [1000, 1512]],
        observation='Only existing sampled training diagnostics at frame1, every50 frames and final frame.',
        scope='Changing weights and different sequences; not a fixed-head evaluation, causal learning-curve attribution, word-truth measure or promotion gate.',
        training_may_continue=True, alter_training=False, new_GT_opened=False, new_tracking_calls=0, optimizer_steps_added=0)
    write(OUT / 'spec.json', spec)
    traces = {}; records = {}; snapshots = {}
    for arm in ['control', 'support']:
        log_rows = [json.loads(x) for x in (ROOT / 'training' / arm / 'sequence_log.jsonl').read_text().splitlines()]
        records[arm] = log_rows[:COUNT]
        assert [x['sequence'] for x in records[arm]] == names
        trace_rows = [json.loads(x) for x in (ROOT / 'training' / arm / 'sampled_state_trace.jsonl').read_text().splitlines()]
        traces[arm] = [x for x in trace_rows if x['sequence'] in names]
        for case, record in zip(cases, records[arm]):
            rows = [x for x in traces[arm] if x['sequence'] == case['sequence']]
            n = record['frames']; assert n == case['rgb_frames'] == case['depth_frames']
            assert [x['frame_index'] for x in rows] == sorted({1, n - 1} | set(range(50, n, 50)))
            assert sum(x['template_write'] for x in rows) == record['template_writes']
            assert all(x['template_write'] == (x['frame_index'] % 50 == 0 and x['best_score'] > .75) for x in rows)
            assert math.isfinite(record['mean_training_loss']) and math.isfinite(record['maximum_preclip_gradient_norm'])
        assert records[arm][-1]['total_track_calls'] == 48421 and records[arm][-1]['total_optimizer_steps'] == 1512
        write(OUT / (arm + '_records.json'), records[arm]); write(OUT / (arm + '_sampled_trace.json'), traces[arm])
        snapshots[arm] = dict(records_sha256=sha(OUT / (arm + '_records.json')), trace_sha256=sha(OUT / (arm + '_sampled_trace.json')))
    for row in traces['control']:
        assert 'support_loss' not in row and 'positive_null_mass' not in row
    for row in traces['support']:
        if row['label'] == 'invalid':
            assert 'support_loss' not in row and 'positive_null_mass' not in row
        else:
            assert row['support_weight'] == .1 and math.isfinite(row['support_loss']) and row['support_loss'] >= 0
            assert row['support_positive_cells'] == int(row['label'] == 'centre_inside')
            assert 0 <= row['support_negative_cells'] <= 256 - row['support_positive_cells']
            assert 0 <= row['positive_null_mass'] <= 1 and 0 <= row['negative_null_mass'] <= 1

    def stats(rows):
        valid = [x for x in rows if x['label'] != 'invalid']
        positive = [x for x in valid if x['support_positive_cells'] > 0]
        negative = [x for x in valid if x['support_negative_cells'] > 0]
        paired = [x for x in positive if x['support_negative_cells'] > 0]

        def summarize(values):
            return dict(count=len(values), mean=float(np.mean(values)), median=float(np.median(values)), minimum=float(np.min(values)), maximum=float(np.max(values))) if values else dict(count=0)

        return dict(samples=len(rows), labels=dict(Counter(x['label'] for x in rows)),
            support_loss=summarize([x['support_loss'] for x in valid]),
            GT_centre_null_mass=summarize([x['positive_null_mass'] for x in positive]),
            outside_box_mean_null_mass=summarize([x['negative_null_mass'] for x in negative]),
            paired_background_minus_centre=summarize([x['negative_null_mass'] - x['positive_null_mass'] for x in paired]),
            paired_background_greater_fraction=float(np.mean([x['negative_null_mass'] > x['positive_null_mass'] for x in paired])) if paired else None)

    support = traces['support']
    bins = []
    for start, end in spec['bins']:
        selected = [x for x in support if start <= x['optimizer_steps_before_update'] < end]
        bins.append(dict(steps_start_inclusive=start, steps_end_exclusive=end, statistics=stats(selected)))
    assert sum(b['statistics']['samples'] for b in bins) == len(support)
    result = dict(status='completed_first34_sampled_training_mechanism_audit', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), spec_sha256=sha(OUT / 'spec.json'), snapshots=snapshots,
        sequences_per_arm=COUNT, track_calls_per_arm=48421, optimizer_steps_per_arm=1512,
        sampled_rows={arm: len(rows) for arm, rows in traces.items()}, support_statistics=stats(support), bins=bins,
        per_sequence_support={seq: stats([x for x in support if x['sequence'] == seq]) for seq in names},
        reconstructed_template_writes={arm: sum(r['template_writes'] for r in rows) for arm, rows in records.items()},
        Control_and_invalid_GT_have_no_support_supervision=True, GT_centre_mass_is_not_selected_peak_mass=True,
        current_weights_not_frozen_across_samples=True, no_generalization_or_semantic_truth_claim=True,
        new_tracking_calls=0, new_GT_opened=False, optimizer_steps_added=0, training_source_modified=False,
        checkpoint_selected=False, thresholds_or_original_gates_changed=False, public_evaluation_allowed=False,
        independent_model_review_pass=False)
    write(OUT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sequence_support'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
