"""Independent consistency audit of the six sealed Full152 result files."""
import csv
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / 'selected_full_evaluation_20260921'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    collected = read(HERE / 'all_results.json')
    assert collected['status'] == 'six_Full152_evaluations_complete'
    assert not collected['checkpoint_selection_from_external_metrics']
    assert len(collected['results']) == 6
    assert {(row['model'], row['dataset']) for row in collected['results']} == {
        (model, dataset) for model in ('M67', 'M82') for dataset in ('depthtrack', 'cdtb', 'vot')}
    metrics = {}
    for row in collected['results']:
        model, dataset = row['model'], row['dataset']
        assert row['checkpoint_sha256'] == collected['models'][model]
        filename = f'{model}_{dataset}_result.json' if dataset == 'vot' else f'{model}_{dataset}_metrics.json'
        path = HERE / filename
        assert row['result_sha256'] == sha(path)
        payload = read(path)
        if dataset == 'vot':
            assert payload['status'] == 'complete_full127'
            assert payload['checkpoint_sha256'] == row['checkpoint_sha256']
            assert payload['metrics_percent'] == row['metrics']
        else:
            assert payload['status'] == 'complete'
            assert payload['metrics'] == row['metrics']
            assert payload['metrics']['sequences'] == (50 if dataset == 'depthtrack' else 80)
            assert payload['metrics']['frames'] == (76373 if dataset == 'depthtrack' else 101956)
        metrics[(model, dataset)] = row['metrics']
    for model in ('M67', 'M82'):
        assert read(HERE / (model + '_bundle.json'))['adapter_checkpoint_sha256'] == collected['models'][model]
    new = read(HERE / 'M82_vot_result.json')
    old = read(OLD / 'M82_vot_result.json')
    m67 = read(HERE / 'M67_vot_result.json')
    native_path = HERE.parents[1] / 'native_vot_full127/result.json'
    native_result = read(native_path)
    merge = read(HERE / 'M82_vot_merge_result.json')
    execution = read(HERE / 'M82_vot_execution.json')
    assert new['merge_sha256'] == sha(HERE / 'M82_vot_merge_result.json')
    assert new['bundle_sha256'] == execution['bundle_sha256'] == sha(HERE / 'M82_bundle.json')
    assert execution['anchors'] == merge['anchor_count'] == 1765
    assert execution['sequences'] == 127 and execution['workers'] == 4
    assert merge['status'] == 'complete' and merge['result_file_count'] == len(merge['result_sha256']) == 5295
    assert len(new['per_sequence_failures']) == 127 and len(new['failure_outcomes']) == 1765
    assert sum(x['anchors'] for x in new['per_sequence_failures'].values()) == 1765
    assert sum(x['confirmed_failures'] for x in new['per_sequence_failures'].values()) == new['confirmed_failures']
    assert sum(bool(x['failed']) for x in new['failure_outcomes'].values()) == new['confirmed_failures'] == 283
    counts = {}
    for key, outcome in new['failure_outcomes'].items():
        assert key == outcome['anchor_key']
        name = outcome['sequence']
        counts.setdefault(name, [0, 0])
        counts[name][0] += 1
        counts[name][1] += bool(outcome['failed'])
    assert {name: {'anchors': v[0], 'confirmed_failures': v[1]} for name, v in counts.items()} == new['per_sequence_failures']
    assert set(new['failure_outcomes']) == set(old['failure_outcomes']) == set(m67['failure_outcomes']) == set(native_result['failure_outcomes'])

    with (OLD / 'M82_vot_per_sequence_comparison.csv').open(newline='') as source:
        native_rows = {row['sequence']: row for row in csv.DictReader(source)}
    assert set(native_rows) == set(new['per_sequence_failures'])
    rows = []
    for name in sorted(native_rows):
        source = native_rows[name]
        native = int(source['native'])
        old_m82 = old['per_sequence_failures'][name]['confirmed_failures']
        full_m67 = m67['per_sequence_failures'][name]['confirmed_failures']
        full_m82 = new['per_sequence_failures'][name]['confirmed_failures']
        assert int(source['anchors']) == new['per_sequence_failures'][name]['anchors']
        assert int(source['M82']) == old_m82
        rows.append(dict(sequence=name, anchors=int(source['anchors']), native=native,
                         old_M82=old_m82, full_M67=full_m67, full_M82=full_m82,
                         full_M82_minus_native=full_m82 - native,
                         full_M82_minus_old_M82=full_m82 - old_m82,
                         full_M82_minus_full_M67=full_m82 - full_m67))
    assert sum(row['native'] for row in rows) == 183
    assert sum(row['old_M82'] for row in rows) == old['confirmed_failures'] == 212
    assert sum(row['full_M67'] for row in rows) == m67['confirmed_failures'] == 232
    assert sum(row['full_M82'] for row in rows) == 283
    with (HERE / 'full152_vot_per_sequence_comparison.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    relation = {}
    for label, comparator in [('native', native_result), ('old_M82', old), ('full_M67', m67)]:
        new_failure_keys = [k for k in new['failure_outcomes']
                            if new['failure_outcomes'][k]['failed'] and not comparator['failure_outcomes'][k]['failed']]
        relation[label] = dict(
            new_failures=len(new_failure_keys),
            rescued=sum(not new['failure_outcomes'][k]['failed'] and comparator['failure_outcomes'][k]['failed']
                        for k in new['failure_outcomes']),
            new_failure_median_progress=statistics.median(new['failure_outcomes'][k]['progress'] for k in new_failure_keys),
            new_failure_progress_bins={
                label: sum(lower <= new['failure_outcomes'][k]['progress'] < upper for k in new_failure_keys)
                for label, lower, upper in [('0_49', 0, 50), ('50_199', 50, 200),
                                           ('200_499', 200, 500), ('500_plus', 500, float('inf'))]})
        assert relation[label]['new_failures'] - relation[label]['rescued'] == \
            new['confirmed_failures'] - comparator['confirmed_failures']
    result = dict(status='complete_consistency_audit', scope='Six result JSONs, checkpoint bindings, VOT merge metadata, 1765 anchor outcomes and per-sequence counts; raw VOT predictions and metric are not independently recomputed.',
                  all_results_sha256=sha(HERE / 'all_results.json'),
                  m82_vot_result_sha256=sha(HERE / 'M82_vot_result.json'),
                  m82_vot_merge_sha256=sha(HERE / 'M82_vot_merge_result.json'),
                  native_vot_result_sha256=sha(native_path),
                  old_m82_vot_result_sha256=sha(OLD / 'M82_vot_result.json'),
                  full_m67_vot_result_sha256=sha(HERE / 'M67_vot_result.json'),
                  metrics={model: {dataset: metrics[(model, dataset)] for dataset in ('depthtrack', 'cdtb', 'vot')}
                           for model in ('M67', 'M82')},
                  confirmed_failures=dict(native=183, old_M82=212, full_M67=232, full_M82=283),
                  full_M82_anchor_relation=relation,
                  failure_timing_scope='Posthoc relative run indices, including backward runs. Progress is the start of the first ten-frame confirmed low-overlap segment, not the confirmation index or a global video frame. Timing does not identify the cause of failure.',
                  full_M82_worst_vs_native=sorted(rows, key=lambda row: row['full_M82_minus_native'], reverse=True)[:12],
                  full_M82_best_vs_native=sorted(rows, key=lambda row: row['full_M82_minus_native'])[:12],
                  per_sequence_csv_sha256=sha(HERE / 'full152_vot_per_sequence_comparison.csv'))
    write(HERE / 'full152_completed_consistency_audit.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'confirmed_failures', 'full_M82_anchor_relation',
                                                    'full_M82_worst_vs_native')}, indent=2))


if __name__ == '__main__':
    main()
