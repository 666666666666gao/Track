"""Export sealed OPE sequence curves at the full-dataset selected threshold."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads((args.bundle / 'download_binding.json').read_bytes())
    dataset = binding['dataset']
    report_path = args.bundle / ('metrics_' + dataset + '.json')
    assert binding['status'] == 'complete' and binding['all_result_gt_hashes_verified']
    assert binding['tracking_analysis_controller_exits_zero']
    assert sha(report_path) == binding['metrics_sha256']
    report = json.loads(report_path.read_bytes())
    source = Path(__file__).parent / 'source_snapshots/depthtrack_pr.py'
    assert sha(source) == report['metric_source_sha256']
    module_spec = importlib.util.spec_from_file_location('sealed_ope_metric', source)
    metric = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(metric)
    rows, arrays, all_scores = [], [], []
    for item in binding['sequences']:
        name = item['sequence']
        result_dir = args.bundle / 'results'
        sequence_dir = args.bundle / 'dataset' / name
        paths = [(result_dir / (name + '.txt'), 'bbox_sha256'),
                 (result_dir / (name + '_all_scores.txt'), 'confidence_sha256'),
                 (sequence_dir / 'groundtruth.txt', 'groundtruth_sha256'),
                 (sequence_dir / 'color' / item['first_image'], 'first_image_sha256')]
        for path, key in paths:
            assert sha(path) == item[key], (name, key)
        prediction = metric._load_rows(paths[0][0], 4)
        confidence = metric._load_rows(paths[1][0], 1).reshape(-1)
        groundtruth = metric._load_rows(paths[2][0], 4)
        assert len(prediction) == len(confidence) == len(groundtruth) == item['frames']
        assert np.isfinite(confidence).all() and confidence[0] == 1.
        height, width = cv2.imread(str(paths[3][0])).shape[:2]
        overlaps, visible = metric._vot_overlaps(prediction, groundtruth, width, height)
        assert visible.any()
        arrays.append((overlaps, visible, confidence))
        all_scores.extend(confidence.tolist())
        rows.append(dict(sequence=name, frames=len(confidence), visible_frames=int(visible.sum())))
    thresholds = metric._determine_thresholds(all_scores, 100)
    precision = np.zeros((len(rows), len(thresholds)))
    recall = np.zeros_like(precision)
    selected_count = np.zeros_like(precision, dtype=np.int64)
    for i, (overlaps, visible, confidence) in enumerate(arrays):
        for j, threshold in enumerate(thresholds):
            selected = confidence >= threshold
            selected_count[i, j] = selected.sum()
            precision[i, j] = overlaps[selected].mean() if selected.any() else 1.
            recall[i, j] = overlaps[selected].sum() / visible.sum()
    mean_p, mean_r = precision.mean(0), recall.mean(0)
    total = mean_p + mean_r
    f_curve = np.divide(2 * mean_p * mean_r, total, out=np.zeros_like(total), where=total > 0)
    best = int(np.argmax(f_curve))
    threshold = float(thresholds[best]) if np.isfinite(thresholds[best]) else None
    values = report['metrics']
    assert threshold == values['threshold']
    for key, actual in [('precision', mean_p[best]), ('recall', mean_r[best]), ('f_score', f_curve[best])]:
        assert abs(float(actual) - values[key]) <= 1e-12, key
    assert len(rows) == values['sequences'] and sum(r['frames'] for r in rows) == values['frames']
    for i, row in enumerate(rows):
        p, r = precision[i, best], recall[i, best]
        row.update(selected_frames=int(selected_count[i, best]), precision_percent=float(100 * p),
                   recall_percent=float(100 * r), f_score_percent=float(200 * p * r / (p + r)) if p + r > 0 else 0.)
    output = dict(dataset=dataset, metrics_sha256=sha(report_path), download_binding_sha256=sha(args.bundle / 'download_binding.json'),
                  source_sha256=sha(Path(__file__)), metric_source_sha256=sha(source), threshold=threshold,
                  threshold_index=best, global_metrics_match=True, sequences=rows,
                  scope='All sequences at the dataset-wide maximum-F threshold; sequence F is not individually maximized. Aggregate F is formed from macro P/R, not the mean of sequence F.')
    json_path, csv_path = args.output / 'per_sequence.json', args.output / 'per_sequence.csv'
    assert not json_path.exists() and not csv_path.exists()
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(dataset=dataset, sequences=len(rows), frames=values['frames'],
                         threshold=threshold, global_metrics_match=True, source_sha256=output['source_sha256'])))


if __name__ == '__main__':
    main()
