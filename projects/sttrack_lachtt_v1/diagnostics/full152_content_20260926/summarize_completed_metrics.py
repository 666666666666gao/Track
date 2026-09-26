"""Consolidate sealed Category/Empty/Swapped OPE metrics without rerunning tracking."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent / 'full152_paired_20260925' / 'completed'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    for model in ('M67', 'M82'):
        for dataset, count, frames in [('depthtrack', 50, 76373), ('cdtb', 80, 101956)]:
            reference = json.loads((FORMAL / f'{model}_{dataset}_metrics.json').read_text())
            for content in ('category', 'empty', 'swapped'):
                folder = FORMAL if content == 'category' else ROOT / 'completed'
                prefix = f'{model}_{dataset}' + ('' if content == 'category' else f'_{content}')
                metric_path = folder / f'{prefix}_metrics.json'
                receipt_path = folder / f'{prefix}_receipt.json'
                metric = json.loads(metric_path.read_text())
                receipt = json.loads(receipt_path.read_text())
                assert metric['status'] == receipt['status'] == 'complete'
                assert sha(receipt_path) == metric['receipt_sha256']
                assert metric['bundle_sha256'] == reference['bundle_sha256'] == receipt['bundle_sha256']
                assert metric['metric_source_sha256'] == reference['metric_source_sha256']
                assert metric['plan_sha256'] == receipt['plan_sha256']
                assert len(receipt['sequences']) == count
                assert len({s['sequence'] for s in receipt['sequences']}) == count
                assert sum(s['frames'] for s in receipt['sequences']) == frames
                if content != 'category':
                    plan_path = folder / f'{prefix}_plan.json'
                    plan = json.loads(plan_path.read_text())
                    assert sha(plan_path) == metric['plan_sha256']
                    assert plan['text_bank_sha256'] == receipt['text_bank_sha256']
                m = metric['metrics']
                assert (m['sequences'], m['frames']) == (count, frames)
                row = dict(model=model, dataset=dataset, content=content, sequences=count, frames=frames)
                for key in ('precision', 'recall', 'f_score'):
                    assert abs(m[key] * 100 - m[key + '_percent']) < 1e-10
                    row[key + '_percent'] = m[key + '_percent']
                    row['delta_' + key + '_pp'] = m[key + '_percent'] - reference['metrics'][key + '_percent']
                row.update(metric_sha256=sha(metric_path), receipt_sha256=sha(receipt_path))
                rows.append(row)
    with (ROOT / 'completed/content_metrics_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(status='complete', scope='12 sealed OPE metric records: 4 Category references and 8 content controls. Same-weight independent recursive trajectories; Empty is not native fallback; Swapped is not verified semantic conflict. No VOT content results or causal contribution decomposition.',
                  source_sha256=sha(Path(__file__)), rows=rows)
    (ROOT / 'completed/content_metrics_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='complete', records=len(rows), summary_sha256=sha(ROOT / 'completed/content_metrics_summary.json'))))


if __name__ == '__main__':
    main()
