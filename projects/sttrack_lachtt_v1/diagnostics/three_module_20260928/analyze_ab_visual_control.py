"""Recompute completed M99 fixed-state results; no training or promotion."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def statistics(rows):
    count = len(rows)
    native = sum(row['native_iou'] >= .5 for row in rows)
    selected = sum(row['selected_iou'] >= .5 for row in rows)
    rescues = sum(row['native_iou'] < .5 <= row['selected_iou'] for row in rows)
    breaks = sum(row['selected_iou'] < .5 <= row['native_iou'] for row in rows)
    assert selected - native == rescues - breaks
    return dict(valid_gt=count, native_iou50=native, selected_iou50=selected,
                oracle_iou50=sum(row['oracle_iou'] >= .5 for row in rows),
                rescues=rescues, breaks=breaks,
                native_mean_iou=math.fsum(row['native_iou'] for row in rows) / count,
                selected_mean_iou=math.fsum(row['selected_iou'] for row in rows) / count)


def verify_statistics(recomputed, reported):
    assert recomputed.keys() == reported.keys()
    for key, value in recomputed.items():
        if key.endswith('mean_iou'):
            assert math.isclose(value, reported[key], rel_tol=0, abs_tol=1e-12), key
        else:
            assert value == reported[key], key


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding='utf-8'))
    assert result['status'] == 'complete_visual_control' and result['mode'] == 'train'
    assert (result['seed'], result['batch_size'], result['epochs'], result['optimizer_steps']) == (2027, 64, 12, 480)
    assert result['learning_rate'] == 3e-4
    assert (result['total_parameters'], result['optimized_parameters']) == (241994, 142086)
    assert result['frozen_semantic_observation_parameters_unchanged']
    assert result['fixed_five_slot_empty_input'] and result['no_public_evaluation'] and result['no_recursive_action']
    assert not result['category_attribute_identity_labels_used']
    assert not result['active_capacity_matched_to_semantic_variant']
    assert len(result['history']) == 12
    assert [row['epoch'] for row in result['history']] == list(range(1, 13))
    assert all(row['optimizer_steps'] == 40 for row in result['history'])
    assert sha(args.weights) == result['final_weights_sha256']

    rows = result['development_rows']
    assert len(rows) == 495 and len({row['key'] for row in rows}) == 495
    groups, sequences = defaultdict(list), defaultdict(list)
    for row in rows:
        assert type(row['selected']) is int and 0 <= row['selected'] < 10
        assert all(math.isfinite(row[key]) and 0 <= row[key] <= 1
                   for key in ('native_iou', 'selected_iou', 'oracle_iou'))
        assert row['oracle_iou'] >= max(row['native_iou'], row['selected_iou'])
        groups['all'].append(row)
        assert len(row['strata']) == len(set(row['strata'])) and 'all' not in row['strata']
        for tag in row['strata']:
            groups[tag].append(row)
        sequences[row['key'].rsplit('@', 1)[0]].append(row)
    assert len(sequences) == 22
    recomputed = {tag: statistics(values) for tag, values in sorted(groups.items())}
    assert recomputed.keys() == result['development'].keys()
    for tag, values in recomputed.items():
        verify_statistics(values, result['development'][tag])
    assert (recomputed['all']['native_iou50'], recomputed['all']['oracle_iou50']) == (268, 305)
    checks = dict(correct_exceeds_native=recomputed['all']['selected_iou50'] > 268,
                  mean_iou_exceeds_native=recomputed['all']['selected_mean_iou'] > recomputed['all']['native_mean_iou'],
                  healthy_breaks_zero=recomputed['healthy']['breaks'] == 0,
                  transition_correct_at_least_native=recomputed['transition']['selected_iou50'] >= recomputed['transition']['native_iou50'])
    assert checks == result['fixed_state_capacity_checks']

    comparison = []
    directory = Path(__file__).resolve().parent
    current = {row['key']: row for row in rows}
    for name, relative in [('M91 first-use mean pool', 'm91_completed/visual/result.json'),
                           ('M96 t0-search mean pool', 'm96_completed/t0_search/result.json')]:
        path = directory / relative
        previous = json.loads(path.read_text(encoding='utf-8'))
        previous_rows = previous['development_rows']
        assert {row['key'] for row in previous_rows} == current.keys()
        for row in previous_rows:
            assert row['native_iou'] == current[row['key']]['native_iou']
            assert row['oracle_iou'] == current[row['key']]['oracle_iou']
        baseline = statistics(previous_rows)
        verify_statistics(baseline, previous['development']['all'])
        comparison.append(dict(model=name, result_sha256=sha(path), **baseline))
    comparison.append(dict(model='M99 token/context Empty control', result_sha256=sha(args.result), **recomputed['all']))
    sequence_rows = [dict(sequence=name, **statistics(values)) for name, values in sorted(sequences.items())]
    for row in comparison + sequence_rows:
        row['correct_delta_vs_native'] = row['selected_iou50'] - row['native_iou50']
        row['mean_iou_delta_pp_vs_native'] = 100 * (row['selected_mean_iou'] - row['native_mean_iou'])
    analysis = dict(status='complete_M99_fixed_state_analysis_only', result_sha256=sha(args.result),
                    final_weights_sha256=sha(args.weights), analysis_source_sha256=sha(Path(__file__)),
                    recomputed_development=recomputed, capacity_checks=checks,
                    all_capacity_checks_pass=all(checks.values()), comparison=comparison,
                    sequence_rows=sequence_rows, no_semantic_recursive_or_official_acceptance=True,
                    comparison_is_not_capacity_matched_causal_ablation=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'analysis.json').write_bytes((json.dumps(analysis, indent=2) + '\n').encode('utf-8'))
    write_csv(args.output / 'strata.csv', [dict(stratum=tag, **values) for tag, values in recomputed.items()])
    write_csv(args.output / 'per_sequence.csv', sequence_rows)
    write_csv(args.output / 'comparison.csv', comparison)
    write_csv(args.output / 'events.csv', [dict(row, strata=';'.join(row['strata'])) for row in rows])
    report = ['# M99 fixed-state visual control', '',
              'Train-only fixed-state diagnosis: 130 fitting sequences and 22 reused development sequences.',
              'Empty input; IoU localization supervision; no semantic/physical-identity labels, recursive action or official evaluation.',
              'M91/M96 comparisons are descriptive and differ in representation/origin/capacity; they are not a causal matched ablation.', '',
              f"Native reference: {recomputed['all']['native_iou50']}/495 correct, mean IoU {recomputed['all']['native_mean_iou']:.9f}.", '',
              '| Model | Correct / 495 | Mean IoU | Delta correct | Delta IoU (pp) | Rescues | Breaks |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparison:
        report.append(f"| {row['model']} | {row['selected_iou50']} | {row['selected_mean_iou']:.9f} | {row['correct_delta_vs_native']:+d} | {row['mean_iou_delta_pp_vs_native']:+.6f} | {row['rescues']} | {row['breaks']} |")
    report.extend(['', '| Predeclared capacity check | Result |', '|---|---|'])
    report.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    report.extend(['', 'All events and all 22 sequences are retained in the accompanying CSVs.',
                   'The result SHA-256 was recorded and the final weight SHA-256 was verified against the result. This report does not independently validate input tensors or rerun the model.',
                   'These checks alone do not establish semantic benefits, recursive improvements, C effectiveness or official acceptance.'])
    (args.output / 'REPORT.md').write_bytes(('\n'.join(report) + '\n').encode('utf-8'))
    print(json.dumps(dict(status=analysis['status'], checks=checks, all_pass=all(checks.values()))), flush=True)


if __name__ == '__main__':
    main()
