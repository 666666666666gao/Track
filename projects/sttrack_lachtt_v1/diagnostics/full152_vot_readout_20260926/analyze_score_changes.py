"""Posthoc score ranks and normalized KL from sealed same-state maps."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_full152_vot_readout_20260926')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_kl(student, teacher):
    s = student.astype(np.float64)
    t = teacher.astype(np.float64)
    assert (s > 0).all() and (t > 0).all()
    s, t = s / s.sum(), t / t.sum()
    return float((t * (np.log(t) - np.log(s))).sum())


def summarize(rows):
    keys = ['native_to_category_kl', 'empty_to_category_kl', 'native_peak_rank_under_category_hann',
            'native_peak_rank_under_category_raw', 'category_hann_relative_margin',
            'category_empty_max_abs_score_difference']
    return dict(positions=len(rows), same_category_empty_hann_peak=sum(r['category_peak'] == r['empty_peak'] for r in rows),
                same_category_native_hann_peak=sum(r['category_peak'] == r['native_peak'] for r in rows),
                distributions={k: dict(min=float(np.min([r[k] for r in rows])),
                                       median=float(np.median([r[k] for r in rows])),
                                       p90=float(np.quantile([r[k] for r in rows], .9)),
                                       max=float(np.max([r[k] for r in rows]))) for k in keys})


def main():
    report_path = ROOT / 'analysis.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete'
    records = {(r['anchor_key'], r['run_index']): r for r in report['rows']}
    rows = []
    for shard in (0, 1):
        folder = ROOT / 'predictions' / str(shard)
        receipt_path = folder / 'receipt.json'
        assert sha(receipt_path) == report['receipts'][str(shard)]
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'complete' and not receipt['counterfactual_state_committed']
        for case in receipt['cases']:
            key = case['anchor_key']
            jp, npz = folder / (key + '.json'), folder / (key + '.npz')
            assert sha(jp) == case['json_sha256'] and sha(npz) == case['dense_sha256']
            observations = json.loads(jp.read_text())['rows']
            with np.load(npz, allow_pickle=False) as archive:
                scores = {a: archive[a][:, 0].reshape(20, 256) for a in ('category', 'empty', 'native')}
                window = archive['window'].reshape(256)
            for index, observation in enumerate(observations):
                existing = records[key, observation['run_index']]
                raw = {a: v[index] for a, v in scores.items()}
                hann = {a: v * window for a, v in raw.items()}
                peaks = {a: int(v.argmax()) for a, v in hann.items()}
                assert all(peaks[a] == observation['variants'][a]['hann_peak'] for a in peaks)
                native_peak = peaks['native']
                ordered = np.sort(hann['category'])
                row = dict(anchor_key=key, run_index=observation['run_index'],
                           relative_to_failure=existing['relative_to_failure'],
                           category_iou=existing['readouts']['category_hann'], native_iou=existing['readouts']['native_hann'],
                           category_peak=peaks['category'], empty_peak=peaks['empty'], native_peak=native_peak,
                           native_to_category_kl=normalized_kl(raw['category'], raw['native']),
                           empty_to_category_kl=normalized_kl(raw['category'], raw['empty']),
                           native_peak_rank_under_category_hann=1 + int((hann['category'] > hann['category'][native_peak]).sum()),
                           native_peak_rank_under_category_raw=1 + int((raw['category'] > raw['category'][native_peak]).sum()),
                           category_hann_relative_margin=float((ordered[-1] - ordered[-2]) / ordered[-1]),
                           category_empty_max_abs_score_difference=float(np.abs(raw['category'] - raw['empty']).max()))
                rows.append(row)
    assert len(rows) == len(records) == 2480
    onset = [r for r in rows if r['relative_to_failure'] == 0]
    native_rescue = [r for r in onset if r['native_iou'] >= .5]
    assert len(onset) == 124 and len(native_rescue) == 43
    result = dict(status='complete', scope='Posthoc maps on Category histories, selected failed anchors only. KL uses float64 arithmetic for the same normalization formula as training; not an exact replay of the float32 training loss. Ranks and KL are descriptive, not recovery actions or loss efficacy proof.',
                  source_sha256=sha(Path(__file__)), input_report_sha256=sha(report_path),
                  summaries=dict(all=summarize(rows), onset=summarize(onset), native_correct_onset=summarize(native_rescue)), rows=rows)
    out = ROOT / 'score_changes.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='complete', summaries=result['summaries'], output_sha256=sha(out))))


if __name__ == '__main__':
    main()
