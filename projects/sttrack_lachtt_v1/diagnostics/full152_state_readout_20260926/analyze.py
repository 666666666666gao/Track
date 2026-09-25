"""Evaluate sealed same-state head readouts against CDTB labels."""
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


EVAL = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925')
ROOT = Path('/root/autodl-tmp/sttrack_full152_state_readout_20260926')
VARIANTS = (
    'adapted_score_adapted_geometry',
    'adapted_score_native_geometry',
    'native_score_adapted_geometry',
    'native_score_native_geometry',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    receipt_path = ROOT / 'receipt.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'complete' and receipt['source_sha256'] == sha(ROOT / 'readout.py')
    plan_path = EVAL / 'M82/cdtb/plan.json'
    plan = read(plan_path)
    assert receipt['plan_sha256'] == sha(plan_path)
    cases = {row['sequence']: row for row in read(plan['cases_path'])}
    spec = importlib.util.spec_from_file_location('sealed_metric', plan['metric_source'])
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    assert sha(plan['metric_source']) == plan['metric_source_sha256']
    results = []
    for sealed in receipt['cases']:
        sequence = sealed['sequence']
        path = ROOT / (sequence + '.json')
        assert sha(path) == sealed['output_sha256']
        rows = read(path)['rows']
        assert len(rows) == sealed['positions']
        gt_path = Path(plan['dataset_root']) / sequence / 'groundtruth.txt'
        assert sha(gt_path) == cases[sequence]['gt_sha256']
        gt = metric._load_rows(gt_path, 4)
        image = cv2.imread(str(Path(plan['dataset_root']) / sequence / 'color/00000001.jpg'))
        height, width = image.shape[:2]
        indexed = []
        for row in rows:
            frame = row['frame']
            scores = {}
            valid = None
            for name in VARIANTS:
                box = np.asarray(row['variants'][name]['bbox'], dtype=np.float64).reshape(1, 4)
                overlap, visible = metric._vot_overlaps(box, gt[frame:frame + 1], width, height)
                valid = bool(visible[0]) if valid is None else valid
                assert valid == bool(visible[0])
                scores[name] = float(overlap[0]) if valid else None
            indexed.append(dict(frame=frame, visible=valid, overlap=scores,
                                raw_peak={name: row['variants'][name]['raw_peak'] for name in VARIANTS},
                                hann_peak={name: row['variants'][name]['hann_peak'] for name in VARIANTS}))
        visible_rows = [row for row in indexed if row['visible']]
        summary = {}
        for name in VARIANTS:
            values = [row['overlap'][name] for row in visible_rows]
            summary[name] = dict(visible_frames=len(values), mean_iou=float(np.mean(values)) if values else None,
                                 severe_frames=sum(value <= .1 for value in values))
        results.append(dict(sequence=sequence, summary=summary, rows=indexed))
    output = dict(status='complete', readout_receipt_sha256=sha(receipt_path),
                  metric_source_sha256=sha(plan['metric_source']),
                  scope='Posthoc selected frames on sealed Category history; alternate heads do not commit state.',
                  cases=results)
    (ROOT / 'analysis.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({case['sequence']: case['summary'] for case in results}, indent=2))


if __name__ == '__main__':
    main()
