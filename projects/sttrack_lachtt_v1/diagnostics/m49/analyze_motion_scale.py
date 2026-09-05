#!/usr/bin/env python3
"""Read-only motion/scale/template diagnosis of the sealed M39 native paths."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from vot.region import RegionType
from vot.region.io import read_trajectory
from vot.region.raster import calculate_overlap
from vot.region.shapes import Rectangle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid(box):
    return bool(np.isfinite(box).all() and min(box[2:]) > 0)


def center(box):
    return np.asarray(box[:2]) + 0.5 * np.asarray(box[2:])


def crop(state, target):
    side = math.ceil(math.sqrt(state[2] * state[3]) * 4)
    lo = np.asarray([round(v - 0.5 * side) for v in center(state)])
    hi = lo + side
    tc = center(target)
    intersection = np.maximum(0, np.minimum(hi, target[:2] + target[2:]) - np.maximum(lo, target[:2]))
    fraction = float(np.prod(intersection) / np.prod(target[2:]))
    return bool(np.all(tc >= lo) and np.all(tc < hi)), fraction >= 1 - 1e-12


def motion(previous, current, prefix):
    scale = math.sqrt(previous[2] * previous[3])
    distance = float(np.linalg.norm(center(current) - center(previous)))
    ratio = math.sqrt(float(np.prod(current[2:]) / np.prod(previous[2:])))
    return {
        prefix + '_displacement_px': distance,
        prefix + '_displacement_norm': distance / scale,
        prefix + '_linear_scale_ratio': ratio,
        prefix + '_scale_change_factor': max(ratio, 1 / ratio),
        prefix + '_width_ratio': float(current[2] / previous[2]),
        prefix + '_height_ratio': float(current[3] / previous[3]),
    }


def stats(rows, keys):
    result = {'count': len(rows)}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        result[key] = {'count': len(values), 'mean': float(np.mean(values)),
                       **{label: float(np.quantile(values, q)) for label, q in
                          [('q50', .5), ('q90', .9), ('q95', .95), ('max', 1)]}} if values else {'count': 0}
    return result


METRICS = ['gt_displacement_px', 'gt_displacement_norm', 'gt_scale_change_factor',
           'pred_jump_gt_norm', 'pred_scale_change_factor', 'prior_error_gt_norm',
           'current_error_gt_norm', 'template_age', 'gt_template_scale_change_factor']


def summarize(rows):
    out = stats(rows, METRICS)
    out['motion_counts'] = {str(t): sum(r['gt_displacement_norm'] <= t for r in rows)
                            for t in [.25, .5, 1, 2]}
    out['scale_change_counts'] = {str(t): sum(r['gt_scale_change_factor'] <= t for r in rows)
                                 for t in [1.1, 1.25, 1.5, 2]}
    coverage = [r for r in rows if 'actual_center_inside' in r]
    out['coverage'] = {key: sum(r[key] for r in coverage) for key in
                       ['actual_center_inside', 'actual_full_inside', 'oracle_previous_gt_center_inside',
                        'oracle_previous_gt_full_inside']}
    comparable = [r for r in coverage if r['velocity_available']]
    out['velocity'] = {'count': len(comparable), **{key: sum(r[key] for r in comparable) for key in
                       ['velocity_center_inside', 'velocity_full_inside', 'velocity_center_gain',
                        'velocity_center_loss', 'velocity_full_gain', 'velocity_full_loss']}}
    return out


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    paths = {k: Path(v) for k, v in spec['paths'].items()}
    for name, expected in spec['input_sha256'].items():
        assert sha(paths[name]) == expected, name
    default = json.loads(paths['m39_result'].read_text())['arms']['default']
    dimensions = json.loads(paths['default_analysis'].read_text())['sequences']
    gt_boxes, gt_regions, hashes = {}, {}, {}
    physical, physical_invalid = [], 0
    for sequence in sorted(default['per_sequence_failures']):
        path = paths['sequence_root'] / sequence / 'groundtruth.txt'
        boxes = np.loadtxt(path, delimiter=',', ndmin=2)
        assert boxes.shape == (dimensions[sequence]['length'], 4), sequence
        gt_boxes[sequence] = boxes
        gt_regions[sequence] = [Rectangle(*b) for b in boxes]
        hashes[str(path)] = sha(path)
        for i in range(1, len(boxes)):
            if not (valid(boxes[i-1]) and valid(boxes[i])):
                physical_invalid += 1
                continue
            physical.append({'sequence': sequence, 'source_frame': i,
                             **motion(boxes[i-1], boxes[i], 'gt')})

    onsets, healthy, writes = [], [], []
    invalid_pairs, frame_count, reconstruction_matches = 0, 0, 0
    m40_rows = {r['anchor_key']: r for r in csv.DictReader(paths['m40_rows'].open())}
    for anchor_key, outcome in sorted(default['failure_outcomes'].items()):
        sequence, anchor = outcome['sequence'], outcome['anchor']
        root = paths['trajectory_root'] / sequence
        stem = f'{sequence}_{anchor:08d}'
        path, conf_path = root / (stem + '.bin'), root / (stem + '_confidence.value')
        with path.open('rb') as stream:
            regions = read_trajectory(stream)
        confs = [None if not v.strip() else float(v) for v in conf_path.read_text().splitlines()]
        assert len(regions) == len(confs) == outcome['run_length'], anchor_key
        hashes[str(path)], hashes[str(conf_path)] = sha(path), sha(conf_path)
        direction = 1 if outcome['direction'] == 'forward' else -1
        indices = anchor + direction * np.arange(len(regions))
        boxes = gt_boxes[sequence]
        bounds = (dimensions[sequence]['width'], dimensions[sequence]['height'])
        predictions = [boxes[anchor]]
        for region in regions[1:]:
            box = region.convert(RegionType.RECTANGLE)
            predictions.append(np.asarray([box.x, box.y, box.width, box.height]))
        assert all(valid(b) for b in predictions), anchor_key
        ious = [float(calculate_overlap(Rectangle(*p), gt_regions[sequence][s], bounds))
                if valid(boxes[s]) else None for p, s in zip(predictions, indices)]
        last_write, low_streak = 0, 0
        onset_step = outcome['progress'] if outcome['failed'] else None
        for i in range(1, len(regions)):
            frame_count += 1
            s, prev_s = int(indices[i]), int(indices[i-1])
            current_gt, previous_gt = boxes[s], boxes[prev_s]
            current_pred, previous_pred = predictions[i], predictions[i-1]
            is_onset = i == onset_step
            is_healthy = ious[i-1] is not None and ious[i] is not None and min(ious[i-1], ious[i]) >= .5
            write_now = i % 50 == 0 and confs[i] > .75
            if valid(current_gt) and valid(previous_gt) and (is_onset or is_healthy):
                gs = math.sqrt(float(np.prod(previous_gt[2:])))
                actual, oracle = crop(previous_pred, current_gt), crop(previous_gt, current_gt)
                velocity_available = i >= 2
                vel = None
                if velocity_available:
                    extrapolated = previous_pred.copy()
                    extrapolated[:2] += center(previous_pred) - center(predictions[i-2])
                    vel = crop(extrapolated, current_gt)
                prior_error = center(previous_gt) - center(previous_pred)
                true_motion = center(current_gt) - center(previous_gt)
                residual = center(current_gt) - center(previous_pred)
                assert np.max(np.abs(residual - prior_error - true_motion)) < 1e-9
                template_source = int(indices[last_write])
                template_gt = boxes[template_source]
                template_ratio = (math.sqrt(float(np.prod(current_gt[2:]) / np.prod(template_gt[2:])))
                                  if valid(template_gt) else None)
                row = {'anchor_key': anchor_key, 'sequence': sequence, 'anchor': anchor,
                       'direction': outcome['direction'], 'step': i, 'source_frame': s,
                       'previous_source_frame': prev_s, 'previous_iou': ious[i-1], 'current_iou': ious[i],
                       'confidence': confs[i], **motion(previous_gt, current_gt, 'gt'),
                       'pred_jump_gt_norm': float(np.linalg.norm(center(current_pred)-center(previous_pred))) / gs,
                       'pred_scale_change_factor': motion(previous_pred, current_pred, 'pred')['pred_scale_change_factor'],
                       'prior_error_gt_norm': float(np.linalg.norm(prior_error)) / gs,
                       'current_error_gt_norm': float(np.linalg.norm(center(current_pred)-center(current_gt))) / gs,
                       'search_offset_gt_norm': float(np.linalg.norm(residual)) / gs,
                       'previous_pred_to_gt_linear_scale': math.sqrt(float(np.prod(previous_pred[2:]) / np.prod(previous_gt[2:]))),
                       'actual_center_inside': actual[0], 'actual_full_inside': actual[1],
                       'oracle_previous_gt_center_inside': oracle[0], 'oracle_previous_gt_full_inside': oracle[1],
                       'velocity_available': velocity_available,
                       'velocity_center_inside': vel[0] if vel else None,
                       'velocity_full_inside': vel[1] if vel else None,
                       'velocity_center_gain': bool(not actual[0] and vel[0]) if vel else None,
                       'velocity_center_loss': bool(actual[0] and not vel[0]) if vel else None,
                       'velocity_full_gain': bool(not actual[1] and vel[1]) if vel else None,
                       'velocity_full_loss': bool(actual[1] and not vel[1]) if vel else None,
                       'last_template_write_step': last_write, 'last_template_source_frame': template_source,
                       'template_age': i - last_write, 'template_write_iou': ious[last_write],
                       'template_is_initial': last_write == 0, 'write_on_current_frame': write_now,
                       'gt_template_scale_change_factor': max(template_ratio, 1/template_ratio) if template_ratio else None}
                if is_onset:
                    old = m40_rows[anchor_key]
                    assert s == int(old['source_frame_zero_based'])
                    assert actual[0] == (old['factor4_center_inside'] == 'True')
                    assert actual[1] == (old['factor4_box_fully_inside'] == 'True')
                    assert abs(ious[i] - float(old['failure_iou'])) < 1e-9
                    assert all(v is not None and v <= .1 for v in ious[i:i+10])
                    reconstruction_matches += 1
                    onsets.append(row)
                if is_healthy:
                    healthy.append(row)
            elif not(valid(current_gt) and valid(previous_gt)):
                invalid_pairs += 1
                assert not is_onset, anchor_key
            if write_now:
                writes.append({'anchor_key': anchor_key, 'sequence': sequence, 'step': i, 'source_frame': s,
                               'confidence': confs[i], 'iou': ious[i], 'previous_low_streak': low_streak,
                               'before_first_failure': onset_step is None or i < onset_step,
                               'on_first_failure': is_onset})
                last_write = i
            low_streak = low_streak + 1 if ious[i] is not None and ious[i] <= .1 else 0
    assert len(onsets) == reconstruction_matches == default['confirmed_failures'] == 124
    unique_directed = { (r['sequence'], r['previous_source_frame'], r['source_frame']): r for r in onsets}
    unique_physical = { (r['sequence'], min(r['source_frame'],r['previous_source_frame'])) for r in onsets}
    current_templates = [r for r in onsets if r['template_write_iou'] is not None]
    bad_writes = [r for r in writes if r['iou'] is not None and r['iou'] <= .1]
    per_sequence = []
    for sequence in sorted(default['per_sequence_failures']):
        fr = [r for r in onsets if r['sequence'] == sequence]
        pr = [r for r in physical if r['sequence'] == sequence]
        hr = [r for r in healthy if r['sequence'] == sequence]
        fs, ps = summarize(fr), summarize(pr)
        per_sequence.append({'sequence': sequence, 'failure_anchors': len(fr), 'unique_gt_pairs': len(pr),
                             'gt_pair_motion_px_median': ps['gt_displacement_px'].get('q50'),
                             'gt_pair_motion_norm_q95': ps['gt_displacement_norm'].get('q95'),
                             'onset_gt_motion_px_median': fs['gt_displacement_px'].get('q50'),
                             'onset_gt_motion_norm_median': fs['gt_displacement_norm'].get('q50'),
                             'onset_gt_scale_change_factor_median': fs['gt_scale_change_factor'].get('q50'),
                             'onset_prediction_jump_norm_median': fs['pred_jump_gt_norm'].get('q50'),
                             'onset_prior_error_norm_median': fs['prior_error_gt_norm'].get('q50'),
                             'onset_center_outside': sum(not r['actual_center_inside'] for r in fr),
                             'onset_oracle_previous_gt_center_outside': sum(not r['oracle_previous_gt_center_inside'] for r in fr),
                             'onset_velocity_gain': fs['velocity']['velocity_center_gain'],
                             'onset_velocity_loss': fs['velocity']['velocity_center_loss'],
                             'healthy_frames': len(hr),
                             'healthy_velocity_gain': sum(r['velocity_center_gain'] is True for r in hr),
                             'healthy_velocity_loss': sum(r['velocity_center_loss'] is True for r in hr),
                             'onset_latest_template_bad': sum(r['template_write_iou'] is not None and r['template_write_iou'] <= .1 for r in fr)})
    out = paths['output_dir']
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'failure_onsets.csv', onsets)
    write_csv(out / 'per_sequence.csv', per_sequence)
    write_csv(out / 'template_writes.csv', writes)
    write_csv(out / 'unique_gt_pairs.csv', physical)
    write_csv(out / 'healthy_paired_frames.csv', healthy)
    result = {'schema': 'sttrack_m49_motion_scale_v1', 'status': 'complete',
              'scope': {'sequences': 22, 'anchors': 303, 'tracked_steps': frame_count,
                        'failure_onsets': len(onsets), 'unique_directed_failure_pairs': len(unique_directed),
                        'unique_undirected_failure_pairs': len(unique_physical),
                        'unique_gt_invalid_pairs': physical_invalid, 'anchor_invalid_gt_pairs': invalid_pairs,
                        'optimizer_steps': 0, 'gpu_inference_frames': 0, 'new_public_metrics': False},
              'integrity': {'m40_exact_onset_and_crop_matches': reconstruction_matches, 'inputs': hashes,
                            'spec_sha256': sha(args.spec), 'source_sha256': sha(__file__)},
              'physical_unique_chronological': summarize(physical), 'onsets': summarize(onsets),
              'onsets_unique_directed': summarize(list(unique_directed.values())),
              'onsets_inside_actual_factor4': summarize([r for r in onsets if r['actual_center_inside']]),
              'onsets_outside_actual_factor4': summarize([r for r in onsets if not r['actual_center_inside']]),
              'healthy_paired_frames': summarize(healthy),
              'templates': {'writes': len(writes), 'valid_gt_writes': sum(r['iou'] is not None for r in writes),
                            'bad_writes_iou_le_0_1': len(bad_writes),
                            'bad_writes_after_at_least_10_low_frames': sum(r['previous_low_streak'] >= 10 for r in bad_writes),
                            'bad_writes_before_first_failure': sum(r['before_first_failure'] for r in bad_writes),
                            'onset_latest_template_gt_valid': len(current_templates),
                            'onset_latest_template_iou_ge_0_5': sum(r['template_write_iou'] >= .5 for r in current_templates),
                            'onset_latest_template_iou_le_0_1': sum(r['template_write_iou'] <= .1 for r in current_templates),
                            'onset_using_initial_template': sum(r['template_is_initial'] for r in onsets)},
              'output_sha256': {p.name: sha(p) for p in out.glob('*.csv')}}
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'integrity'}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
