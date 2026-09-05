"""Describe paired predicted states after training; this is not GT motion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    plan = json.loads((root/'spec.json').read_text())
    audit = json.loads((root/'data_audit.json').read_text())
    assert audit['status'] == 'PASS'
    assert audit['spec_sha256'] == sha(root/'spec.json')
    assert audit['collection_receipt_sha256'] == sha(root/'collection_receipt.json')
    assert all((root/(name+'.exit')).read_text().strip() == '0'
               for name in ['training_control', 'training_mixed'])
    parent = Path(plan['source_root'])
    old_receipts = {}
    for shard in [0, 1]:
        old_receipts.update({r['sequence']: r for r in
                             json.loads((parent/f'shard{shard}_receipt.json').read_text())['sequences']})
    new_receipts = {r['sequence']: r for r in json.loads((root/'collection_receipt.json').read_text())['sequences']}
    rows = []
    for name in plan['sequences']:
        paths = [parent/'features'/(name+'.pt'), root/'features'/(name+'.pt')]
        assert sha(paths[0]) == old_receipts[name]['feature_sha256']
        assert sha(paths[1]) == new_receipts[name]['feature_sha256']
        old, new = [torch.load(p, map_location='cpu') for p in paths]
        keys = [r['key'] for r in old['records']]
        assert keys == [r['key'] for r in new['records']]
        a, b = old['public_bbox'].double().numpy(), new['selected_bbox'].double().numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert (a[:, 2:] > 0).all() and (b[:, 2:] > 0).all()
        distance = np.linalg.norm((b[:, :2]+b[:, 2:]/2)-(a[:, :2]+a[:, 2:]/2), axis=1)
        old_scale = np.sqrt(np.prod(a[:, 2:], axis=1))
        ratio = np.sqrt(np.prod(b[:, 2:], axis=1))/old_scale
        for i, key in enumerate(keys):
            rows.append(dict(key=key, max_coordinate_difference_px=float(np.max(np.abs(a[i]-b[i]))),
                             center_distance_px=float(distance[i]),
                             center_distance_over_old_predicted_scale=float(distance[i]/old_scale[i]),
                             symmetric_linear_scale_ratio=float(max(ratio[i], 1/ratio[i]))))
    assert len(rows) == audit['physical_fit_events'] == 1511
    def values(key):
        return np.array([r[key] for r in rows])
    difference = values('max_coordinate_difference_px')
    assert int((difference > 0).sum()) == audit['changed_event_predictions']
    summaries = {key: dict(median=float(np.median(values(key))), p95=float(np.quantile(values(key), .95)),
                          maximum=float(values(key).max())) for key in rows[0] if key != 'key'}
    result = dict(status='complete', events=len(rows), exact_different=int((difference > 0).sum()),
                  max_coordinate_difference_over_0_001px=int((difference > .001).sum()),
                  center_distance_over_1px=int((values('center_distance_px') > 1).sum()),
                  center_distance_over_quarter_old_predicted_scale=int((values('center_distance_over_old_predicted_scale') > .25).sum()),
                  summaries=summaries, rows=rows, source_sha256=sha(Path(__file__)),
                  spec_sha256=sha(root/'spec.json'), data_audit_sha256=sha(root/'data_audit.json'),
                  labels_opened=False,
                  scope='Post-training descriptive statistics only. Two predicted trajectories at the same physical events, not GT motion, not adjacent-frame displacement, and not model-advancement gates.')
    (root/'state_difference_description.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
