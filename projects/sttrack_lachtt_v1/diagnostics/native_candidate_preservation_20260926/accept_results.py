"""Audit six sealed results and report per-model nine-metric target attainment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

TARGETS = {'depthtrack': {'precision_percent': 65.2, 'recall_percent': 64.9, 'f_score_percent': 65.1},
           'cdtb': {'precision_percent': 72.9, 'recall_percent': 75.6, 'f_score_percent': 74.2},
           'vot': {'EAO': 77.9, 'ACC': 82.1, 'ROB': 93.7}}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(path):
    all_results = read(path)
    assert all_results['status'] == 'six_Full152_evaluations_complete'
    assert all_results['checkpoint_selection_from_external_metrics'] is False
    models = all_results['models']
    assert len(models) == 2 and len(all_results['results']) == 6
    seen = set()
    rows = []
    for row in all_results['results']:
        model, dataset = row['model'], row['dataset']
        assert model in models and dataset in TARGETS
        assert (model, dataset) not in seen
        seen.add((model, dataset))
        assert row['checkpoint_sha256'] == models[model]
        result_path = Path(row['result_path'])
        assert sha(result_path) == row['result_sha256']
        result = read(result_path)
        if dataset == 'vot':
            assert result['status'] == 'complete_full127'
            assert result['checkpoint_sha256'] == models[model]
            assert len(result['failure_outcomes']) == 1765
            assert len(result['per_sequence_failures']) == 127
            assert row['metrics'] == result['metrics_percent']
        else:
            count, frames = (50, 76373) if dataset == 'depthtrack' else (80, 101956)
            assert result['status'] == 'complete'
            receipt_path = result_path.parent / 'receipt.json'
            assert sha(receipt_path) == result['receipt_sha256']
            receipt = read(receipt_path)
            assert len(receipt['sequences']) == count and receipt['frames'] == frames
            assert row['metrics'] == result['metrics']
            assert result['metrics']['sequences'] == count and result['metrics']['frames'] == frames
            bundle_path = result_path.parent.parent.parent / 'bundle.json'
            assert sha(bundle_path) == result['bundle_sha256']
            assert read(bundle_path)['adapter_checkpoint_sha256'] == models[model]
        for metric, target in TARGETS[dataset].items():
            value = float(row['metrics'][metric])
            assert math.isfinite(value) and 0 <= value <= 100
            passed = value > target if dataset == 'vot' else value >= target
            rows.append(dict(model=model, dataset=dataset, metric=metric, value=value,
                             target=target, delta_pp=value-target, comparison='>' if dataset == 'vot' else '>=', passed=passed))
    assert seen == {(model, dataset) for model in models for dataset in TARGETS}
    outcomes = {model: all(row['passed'] for row in rows if row['model'] == model) for model in models}
    return dict(status='complete_metric_and_binding_audit', source_sha256=sha(path),
                model_checkpoints=models, metrics=rows, all_nine_pass_by_model=outcomes,
                any_single_model_passes=any(outcomes.values()),
                scope='Metric files, coverage receipts and checkpoint bindings; not a raw-prediction recomputation or semantic attribution audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.results)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['all_nine_pass_by_model']))
