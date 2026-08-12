import json
import os

import numpy as np
import torch

from lib.test.analysis.extract_results import calc_iou_overlap
from lib.test.utils.load_text import load_text


def _load_boxes(path):
    boxes = np.asarray(load_text(str(path), delimiter=('\t', ',', ' '), dtype=np.float64))
    return boxes.reshape(-1, 4)


def _load_scores(path, length):
    if os.path.isfile(path):
        scores = np.asarray(load_text(str(path), delimiter=('\t', ',', ' '), dtype=np.float64)).reshape(-1)
    else:
        scores = np.ones(length, dtype=np.float64)
    if scores.shape[0] > length:
        scores = scores[:length]
    elif scores.shape[0] < length:
        pad_value = scores[-1] if scores.shape[0] > 0 else 1.0
        scores = np.concatenate([scores, np.full(length - scores.shape[0], pad_value, dtype=np.float64)])
    return scores


def _tracker_result_paths(tracker, seq_name):
    base_path = os.path.join(tracker.results_dir, seq_name)
    return '{}.txt'.format(base_path), '{}_best_score.txt'.format(base_path)


def _thresholds_from_scores(scores):
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.array([0.0], dtype=np.float64)
    thresholds = np.unique(finite)
    thresholds.sort()
    return np.concatenate([[thresholds[0] - 1e-6], thresholds])


def _compute_curve(overlaps, valid_gt, scores, valid_pred):
    thresholds = _thresholds_from_scores(scores)
    total_gt = max(float(valid_gt.sum()), 1.0)
    best = {'Pr': 0.0, 'Re': 0.0, 'F-score': 0.0, 'threshold': float(thresholds[0])}

    for threshold in thresholds:
        predicted = (scores >= threshold) & valid_pred
        pred_count = float(predicted.sum())
        if pred_count <= 0:
            precision = 0.0
            recall = 0.0
        else:
            matched_overlap = float(overlaps[predicted & valid_gt].sum())
            precision = matched_overlap / pred_count
            recall = matched_overlap / total_gt
        f_score = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
        if f_score > best['F-score']:
            best = {
                'Pr': precision,
                'Re': recall,
                'F-score': f_score,
                'threshold': float(threshold),
            }
    return best


def _calc_overlaps_ignore_invalid_gt(pred_bb, anno_bb, target_visible=None):
    """Calculate overlaps while masking invalid DepthTrack annotations.

    Some DepthTrack test sequences contain NaN annotations on invalid/absent
    frames. The generic evaluator raises on those frames, but Pr/Re/F-score
    should simply exclude invalid ground-truth frames from the GT count.
    """
    length = anno_bb.shape[0]
    if pred_bb.shape[0] > length:
        pred_bb = pred_bb[:length, :]
    elif pred_bb.shape[0] < length:
        pad = torch.zeros((length - pred_bb.shape[0], 4), dtype=pred_bb.dtype, device=pred_bb.device)
        pred_bb = torch.cat((pred_bb, pad), dim=0)

    valid_gt = torch.isfinite(anno_bb).all(dim=1) & (anno_bb[:, 2] > 0.0) & (anno_bb[:, 3] > 0.0)
    if target_visible is not None:
        valid_gt = valid_gt & target_visible.bool()

    valid_pred = torch.isfinite(pred_bb).all(dim=1) & (pred_bb[:, 2] > 0.0) & (pred_bb[:, 3] > 0.0)

    anno_safe = anno_bb.clone()
    pred_safe = pred_bb.clone()
    anno_safe[~valid_gt] = 0.0
    pred_safe[~valid_pred] = 0.0

    if valid_gt[0] and torch.isfinite(anno_safe[0]).all():
        pred_safe[0, :] = anno_safe[0, :]
        valid_pred[0] = True

    overlaps = torch.zeros(length, dtype=torch.float64, device=anno_bb.device)
    valid_for_iou = valid_gt & valid_pred
    if valid_for_iou.any():
        overlaps[valid_for_iou] = calc_iou_overlap(pred_safe[valid_for_iou], anno_safe[valid_for_iou])
        overlaps[~torch.isfinite(overlaps)] = 0.0

    return overlaps, valid_gt, valid_pred


def evaluate_depthtrack_prre(trackers, dataset, output_json=None, skip_missing_seq=False):
    """Compute DepthTrack-style Pr/Re/F-score from saved tracking results.

    The metric follows the long-term RGB-D convention: for each confidence
    threshold, precision is the mean overlap over predicted target states,
    recall is the mean overlap over ground-truth valid target states, and the
    reported F-score is the best point on that curve.
    """
    results = []
    for tracker in trackers:
        overlaps_all, valid_gt_all, scores_all, valid_pred_all = [], [], [], []
        sequence_metrics = []
        missing = []

        for seq in dataset:
            boxes_path, scores_path = _tracker_result_paths(tracker, seq.name)
            if not os.path.isfile(boxes_path):
                if skip_missing_seq:
                    missing.append(seq.name)
                    continue
                raise FileNotFoundError('Result not found: {}'.format(boxes_path))

            pred_bb = torch.tensor(_load_boxes(boxes_path), dtype=torch.float64)
            anno_bb = torch.tensor(seq.ground_truth_rect.reshape(-1, 4), dtype=torch.float64)
            target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
            err_overlap, valid_frame, valid_pred_tensor = _calc_overlaps_ignore_invalid_gt(
                pred_bb, anno_bb, target_visible)

            length = anno_bb.shape[0]
            scores = _load_scores(scores_path, length)
            pred_np = pred_bb.detach().cpu().numpy()
            if pred_np.shape[0] > length:
                pred_np = pred_np[:length]
            elif pred_np.shape[0] < length:
                pad = np.zeros((length - pred_np.shape[0], 4), dtype=pred_np.dtype)
                pred_np = np.concatenate([pred_np, pad], axis=0)

            overlaps = err_overlap.detach().cpu().numpy().astype(np.float64)
            valid_gt = valid_frame.detach().cpu().numpy().astype(bool)
            valid_pred = valid_pred_tensor.detach().cpu().numpy().astype(bool)
            valid_pred = np.isfinite(scores) & valid_pred
            overlaps[~valid_gt] = 0.0

            overlaps_all.append(overlaps)
            valid_gt_all.append(valid_gt)
            scores_all.append(scores)
            valid_pred_all.append(valid_pred)
            seq_metric = _compute_curve(overlaps, valid_gt, scores, valid_pred)
            sequence_metrics.append({
                'sequence': seq.name,
                'Pr': seq_metric['Pr'] * 100.0,
                'Re': seq_metric['Re'] * 100.0,
                'F-score': seq_metric['F-score'] * 100.0,
                'threshold': seq_metric['threshold'],
            })

        if not overlaps_all:
            metric = {'Pr': 0.0, 'Re': 0.0, 'F-score': 0.0, 'threshold': 0.0}
            sequence_average = {'Pr': 0.0, 'Re': 0.0, 'F-score': 0.0, 'threshold': 0.0}
        else:
            metric = _compute_curve(
                np.concatenate(overlaps_all),
                np.concatenate(valid_gt_all),
                np.concatenate(scores_all),
                np.concatenate(valid_pred_all),
            )
            sequence_average = {
                'Pr': float(np.mean([m['Pr'] for m in sequence_metrics])),
                'Re': float(np.mean([m['Re'] for m in sequence_metrics])),
                'F-score': float(np.mean([m['F-score'] for m in sequence_metrics])),
                'threshold': float(np.mean([m['threshold'] for m in sequence_metrics])),
            }

        result = {
            'tracker': tracker.name,
            'parameter': tracker.parameter_name,
            'run_id': tracker.run_id,
            'Pr': metric['Pr'] * 100.0,
            'Re': metric['Re'] * 100.0,
            'F-score': metric['F-score'] * 100.0,
            'threshold': metric['threshold'],
            'sequence_average': sequence_average,
            'per_sequence': sequence_metrics,
            'missing_sequences': missing,
        }
        results.append(result)

    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def format_depthtrack_prre(results):
    lines = ['DepthTrack Pr/Re/F-score']
    lines.append('{:<18} {:<54} {:<8} {:>8} {:>8} {:>8} {:>10}'.format(
        'tracker', 'parameter', 'epoch', 'Pr', 'Re', 'F-score', 'thr'))
    for item in results:
        seq_avg = item.get('sequence_average', {})
        seq_text = ''
        if seq_avg:
            seq_text = '  seq-avg F={:.2f}'.format(seq_avg.get('F-score', 0.0))
        lines.append('{:<18} {:<54} {:<8} {:>8.2f} {:>8.2f} {:>8.2f} {:>10.4f}{}'.format(
            item['tracker'], item['parameter'], str(item['run_id']),
            item['Pr'], item['Re'], item['F-score'], item['threshold'], seq_text))
    return '\n'.join(lines)
