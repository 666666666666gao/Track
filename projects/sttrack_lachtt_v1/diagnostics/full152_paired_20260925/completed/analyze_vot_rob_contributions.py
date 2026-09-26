"""Exact sequence accounting of published multistart ROB, using sealed outcomes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import inspect
import json
from pathlib import Path

import vot.analysis.multistart as official


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = dict(
        native=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905/result.json'),
        M67=Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/M67/vot/result.json'),
        M82=Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/M82/vot/result.json'),
    )
    source = Path(inspect.getsourcefile(official))
    assert sha(source) == '5a09065e2315387405f4cb8f96c0b8fd32d7428996a34eedd92ac4e2a4deeb02'
    assert sha(paths['native']) == '51213ef55085d270a56ae2952b8871c06557891d418ae572ec3fb932102f1d8b'
    assert sha(paths['M82']) == 'a54bef4663d5a190ad613dc6f632e3a5cade27fb20fb871dc2c862f5b74bdb29'
    results = {name: json.loads(path.read_text()) for name, path in paths.items()}
    anchors = results['native']['failure_outcomes']
    assert len(anchors) == 1765
    grouped, lengths = {}, {}
    for model, result in results.items():
        outcomes = result['failure_outcomes']
        assert set(outcomes) == set(anchors)
        assert result['failure_settings'] == results['native']['failure_settings']
        groups = defaultdict(list)
        for key, item in outcomes.items():
            for field in ['sequence', 'anchor', 'direction', 'run_length']:
                assert item[field] == anchors[key][field]
            assert 0 <= item['progress'] <= item['run_length']
            assert item['failed'] == (item['progress'] < item['run_length'])
            groups[item['sequence']].append(item)
        assert len(groups) == 127
        grouped[model] = {}
        for sequence, items in groups.items():
            zero = outcomes[sequence + '@0F']
            length = zero['run_length']
            assert zero['anchor'] == 0 and zero['direction'] == 'forward'
            lengths[sequence] = length
            survived = sum(item['progress'] for item in items)
            available = sum(item['run_length'] for item in items)
            failures = sum(item['failed'] for item in items)
            assert failures == result['per_sequence_failures'][sequence]['confirmed_failures']
            assert len(items) == result['per_sequence_failures'][sequence]['anchors']
            grouped[model][sequence] = dict(anchors=len(items), survived=survived,
                                            available=available, failures=failures,
                                            robustness=survived / available)
    denominator = sum(lengths.values())
    assert denominator == 80741
    reconstructed = {}
    for model, groups in grouped.items():
        value = 100 * sum(lengths[name] * item['robustness'] for name, item in groups.items()) / denominator
        recorded = results[model]['metrics_percent']['rob' if model == 'native' else 'ROB']
        assert abs(value - recorded) < 1e-10, (model, value, recorded)
        reconstructed[model] = value
    rows = []
    for sequence in sorted(lengths):
        row = dict(sequence=sequence, sequence_frames=lengths[sequence],
                   weight=lengths[sequence] / denominator)
        for model in paths:
            item = grouped[model][sequence]
            row[model + '_ROB_percent'] = 100 * item['robustness']
            row[model + '_failures'] = item['failures']
            row[model + '_survived_anchor_frames'] = item['survived']
            row[model + '_available_anchor_frames'] = item['available']
        for model in ('M67', 'M82'):
            row[model + '_ROB_delta_pp'] = (row[model + '_ROB_percent'] - row['native_ROB_percent']) * row['weight']
        rows.append(row)
    summaries = {}
    for model in ('M67', 'M82'):
        field = model + '_ROB_delta_pp'
        total = sum(row[field] for row in rows)
        assert abs(total - (reconstructed[model] - reconstructed['native'])) < 1e-10
        ranked = sorted(rows, key=lambda r: (r[field], r['sequence']))
        summaries[model] = dict(net_delta_pp=total,
                               negative_sum_pp=sum(r[field] for r in rows if r[field] < 0),
                               positive_sum_pp=sum(r[field] for r in rows if r[field] > 0),
                               worst_five_sum_pp=sum(r[field] for r in ranked[:5]),
                               worst_ten_sum_pp=sum(r[field] for r in ranked[:10]),
                               worst_ten=ranked[:10], best_five=list(reversed(ranked[-5:])))
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'Full152_VOT_ROB_contributions.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', source_sha256={str(p): sha(p) for p in [*paths.values(), source, Path(__file__)]},
                  metric='Official VOT multistart ROB only; not failure rate, EAO, or ACC.',
                  formula='100 * sum_s [sequence_length_s / sum_length * sum_anchor(progress) / sum_anchor(run_length)]',
                  sequences=127, anchors=1765, total_sequence_frames=denominator,
                  reconstructed_ROB_percent=reconstructed, comparisons_vs_native=summaries,
                  csv_sha256=sha(csv_path),
                  interpretation='Additive metric accounting, not a deployment result or causal module attribution. A failed anchor retains its pre-failure progress; later failures lose less ROB than earlier ones of the same run length.')
    path = args.output / 'Full152_VOT_ROB_contributions.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report_sha256=sha(path), reconstructed=reconstructed,
                         comparisons={model: {k: v for k, v in summary.items() if k not in ('worst_ten', 'best_five')}
                                      for model, summary in summaries.items()},
                         M82_worst=[{k: row[k] for k in ['sequence', 'native_failures', 'M82_failures', 'M82_ROB_delta_pp']}
                                    for row in summaries['M82']['worst_ten']]), indent=2))


if __name__ == '__main__':
    main()
