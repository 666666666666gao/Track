"""Describe the 520 sealed extra writes; this audit does not change the policy."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_m50_scale_template_v1_20260905')
plan = json.loads((ROOT/'spec.json').read_text())
parent = Path(plan['source_root'])
spec = json.loads((parent/'spec.json').read_text())
sys.path.insert(0, spec['repository'])
from tools.audit_sttrack_m43 import independent_overlap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((ROOT/'recursive_result.json').read_text())
    assert result['status'] == 'complete' and result['integrity_pass']
    receipt = json.loads((ROOT/'recursive_receipt.json').read_text())
    rows = []
    for item in receipt['sequences']:
        path = ROOT/'recursive'/(item['sequence']+'.json')
        assert sha(path) == item['sha256']
        predictions = json.loads(path.read_text())['rows']
        gt = np.loadtxt(Path(spec['dataset_root'])/item['sequence']/'groundtruth.txt', delimiter=',')[:len(predictions)]
        valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1)
        values, metrics = independent_overlap(np.asarray([r['bbox'] for r in predictions]), gt)
        for key, value in result['per_sequence'][item['sequence']].items():
            assert metrics[key] == value
        reference, preceding_low = 0, 0
        for row in predictions[1:]:
            frame = row['frame']
            if row['template_update'] == 'scale':
                factor = None
                if valid[frame] and valid[reference]:
                    ratio = float(np.sqrt(np.prod(gt[frame, 2:]) / np.prod(gt[reference, 2:])))
                    factor = max(ratio, 1 / ratio)
                rows.append(dict(sequence=item['sequence'], frame=frame, reference_frame=reference,
                                 score=row['score'], predicted_scale_factor=row['template_scale_ratio'],
                                 gt_scale_factor=factor, current_iou=float(values[frame]) if np.isfinite(values[frame]) else None,
                                 preceding_low_iou_frames=preceding_low))
            if row['template_update'] is not None:
                reference = frame
            preceding_low = preceding_low + 1 if np.isfinite(values[frame]) and values[frame] <= .1 else 0
    assert len(rows) == result['scale_updates'] == 520
    labeled = [r for r in rows if r['current_iou'] is not None]
    geometric = [r for r in rows if r['gt_scale_factor'] is not None]
    low = [r for r in labeled if r['current_iou'] <= .1]
    out = dict(status='complete', total_extra_writes=len(rows), current_gt_valid=len(labeled),
               current_iou_at_least_half=sum(r['current_iou'] >= .5 for r in labeled), current_iou_at_most_tenth=len(low),
               low_writes_after_at_least_10_low_frames=sum(r['preceding_low_iou_frames'] >= 10 for r in low),
               current_or_reference_gt_invalid=len(rows)-len(geometric), both_gt_valid=len(geometric),
               gt_scale_factor_median=float(np.median([r['gt_scale_factor'] for r in geometric])),
               gt_scale_factor_below_1p1=sum(r['gt_scale_factor'] < 1.1 for r in geometric),
               gt_scale_factor_at_least_1p25=sum(r['gt_scale_factor'] >= 1.25 for r in geometric),
               result_sha256=sha(ROOT/'recursive_result.json'), receipt_sha256=sha(ROOT/'recursive_receipt.json'),
               source_sha256=sha(Path(__file__)), claim='Post-hoc descriptive audit of all extra writes. No threshold tuning, new tracking or causal attribution.')
    (ROOT/'update_audit.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    with (ROOT/'extra_writes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(out, indent=2))


if __name__ == '__main__': main()
