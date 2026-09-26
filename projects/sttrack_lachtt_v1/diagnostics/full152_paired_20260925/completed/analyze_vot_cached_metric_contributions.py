"""Read official per-sequence caches and reconstruct EAO/ACC/ROB accounting."""
import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import pickle

import numpy as np
import vot.analysis.multistart as official
from vot.analysis.processor import hashkey
from vot.workspace import Workspace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
    rob_path = root / 'posthoc/Full152_VOT_ROB_contributions.json'
    assert sha(rob_path) == 'f8c027f62f4b7821641fec701f2445e83d663c14b0db4b6cfc8f60db04272d68'
    rob_audit = read(rob_path)
    source = Path(inspect.getsourcefile(official))
    assert sha(source) == rob_audit['source_sha256'][str(source)]
    sources = {str(p): sha(p) for p in [rob_path, source, Path(__file__)]}
    models = dict(native=(Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905'), 'sttrack_default_full127'),
                  M67=(root / 'M67/vot', 'sttrack_full152_m67_full127'),
                  M82=(root / 'M82/vot', 'sttrack_full152_m82_full127'))
    contributions, metrics, curve_checks = {}, {}, {}
    for model, (folder, tracker_name) in models.items():
        result_path = folder / 'result.json'
        assert sha(result_path) == rob_audit['source_sha256'][str(result_path)]
        result = read(result_path)
        master = folder / 'run/master'
        analysis_path = Path(result['analysis']) if model == 'native' else master / 'analysis' / ('full152_' + model.lower() + '_full127.json')
        assert sha(analysis_path) == result['analysis_sha256']
        sources.update({str(result_path): sha(result_path), str(analysis_path): sha(analysis_path)})
        published = read(analysis_path)['results']['baseline']['results']
        workspace = Workspace.load(str(master))
        experiment = workspace.stack.experiments['baseline']
        tracker, = workspace.registry.resolve(tracker_name, storage=workspace.storage.substorage('results'), skip_unknown=False)
        sequences = experiment.transform(workspace.dataset)
        assert len(sequences) == 127
        score_analysis, curve_analysis, ar_analysis = experiment.analyses
        assert isinstance(score_analysis, official.EAOScore)
        assert isinstance(curve_analysis, official.EAOCurve)
        assert isinstance(ar_analysis, official.AverageAccuracyRobustness)
        partial_curve, partial_ar = curve_analysis.curves, ar_analysis.analysis
        assert (score_analysis.low, score_analysis.high, partial_curve.high) == (115, 755, 755)

        def cached(analysis, sequence):
            path = master / 'cache/analysis' / Path(*hashkey(analysis, experiment, tracker, sequence))
            data = path.read_bytes()
            sources[str(path)] = hashlib.sha256(data).hexdigest()
            item, = list(pickle.loads(data))
            return item

        partials = {}
        for sequence in sequences:
            (phi, active), weight = cached(partial_curve, sequence)
            accuracy, robustness, _, survived, length = cached(partial_ar, sequence)
            assert weight == 1 and length == len(sequence)
            phi, active = np.asarray(phi), np.asarray(active)
            assert phi.shape == active.shape == (755,)
            assert np.isfinite(phi).all() and np.isfinite(active).all()
            outcomes = [v for v in result['failure_outcomes'].values() if v['sequence'] == sequence.name]
            assert survived == sum(v['progress'] for v in outcomes)
            assert abs(robustness - survived / sum(v['run_length'] for v in outcomes)) < 1e-12
            expected_active = np.asarray([0] + [sum(j < v['run_length'] or v['failed'] for v in outcomes) for j in range(1, 755)])
            assert np.array_equal(active, expected_active)
            partials[sequence.name] = dict(phi=phi, active=active, accuracy=accuracy,
                                          robustness=robustness, survived=survived, length=length)
        total_active = sum(p['active'] for p in partials.values())
        numerator = sum(p['phi'] * p['active'] for p in partials.values())
        curve = np.divide(numerator, total_active, out=np.zeros_like(numerator), where=total_active > 0)
        curve_error = float(np.abs(curve - np.asarray(published[1][0][0])).max())
        assert curve_error < 1e-12
        total_survived = sum(p['survived'] for p in partials.values())
        total_lengths = sum(p['length'] for p in partials.values())
        assert total_survived == published[2][0][3] and total_lengths == published[2][0][4] == 80741
        shares = {}
        for name, p in partials.items():
            contribution_curve = np.divide(p['phi'] * p['active'], total_active,
                                           out=np.zeros_like(numerator), where=total_active > 0)
            shares[name] = dict(
                EAO=100 * float(np.mean(contribution_curve[score_analysis.low:score_analysis.high + 1])),
                ACC=100 * p['accuracy'] * p['survived'] / total_survived,
                ROB=100 * p['robustness'] * p['length'] / total_lengths)
        reconstructed = {metric: sum(p[metric] for p in shares.values()) for metric in ('EAO', 'ACC', 'ROB')}
        expected = dict(EAO=published[0][0][0] * 100, ACC=published[2][0][0] * 100, ROB=published[2][0][1] * 100)
        assert all(abs(reconstructed[k] - expected[k]) < 1e-10 for k in expected)
        metrics[model] = dict(reconstructed_percent=reconstructed, published_percent=expected,
                             accuracy_survived_frame_denominator=total_survived)
        contributions[model], curve_checks[model] = shares, curve_error
    assert set(contributions['native']) == set(contributions['M67']) == set(contributions['M82'])
    rows = []
    for sequence in sorted(contributions['native']):
        row = dict(sequence=sequence)
        for metric in ('EAO', 'ACC', 'ROB'):
            for model in models:
                row[model + '_' + metric + '_contribution_pp'] = contributions[model][sequence][metric]
            for model in ('M67', 'M82'):
                row[model + '_' + metric + '_delta_pp'] = contributions[model][sequence][metric] - contributions['native'][sequence][metric]
        rows.append(row)
    rankings = {}
    for model in ('M67', 'M82'):
        rankings[model] = {}
        for metric in ('EAO', 'ACC', 'ROB'):
            key = model + '_' + metric + '_delta_pp'
            ranked = sorted(rows, key=lambda r: (r[key], r['sequence']))
            rankings[model][metric] = dict(net_delta_pp=sum(r[key] for r in rows),
                                          worst_five_sum_pp=sum(r[key] for r in ranked[:5]),
                                          worst_ten_sum_pp=sum(r[key] for r in ranked[:10]),
                                          worst_ten=[dict(sequence=r['sequence'], delta_pp=r[key]) for r in ranked[:10]])
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'Full152_VOT_all_metric_contributions.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', source_sha256=sources, reconstructed=metrics,
                  cached_global_EAO_curve_max_error=curve_checks,
                  official_EAO_slice='[115:756] on the actual 755-entry curve, exactly as the installed official code',
                  rankings_vs_native=rankings, csv_sha256=sha(csv_path),
                  scope='Official cached per-sequence metric accounting; no new tracker inference or metric tuning.',
                  limitations=[
                      'EAO contributions use each model own active-run denominator at each curve position.',
                      'ACC contributions use each model own surviving-frame denominator; failed tails are excluded.',
                      'These additive differences reconstruct totals but are not fixed-denominator sequence replacement effects.',
                      'A positive ACC contribution is not evidence that the sequence boxes became more accurate.',
                      'No per-sequence oracle selection or causal attribution to language, memory, or a module is made.',
                  ])
    path = args.output / 'Full152_VOT_all_metric_contributions.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(report_sha256=sha(path), reconstructed=metrics,
                          EAO_rankings={name: ranking['EAO'] for name, ranking in rankings.items()}), indent=2))


if __name__ == '__main__':
    main()
