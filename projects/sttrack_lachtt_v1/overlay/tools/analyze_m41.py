"""Posthoc-only candidate capacity analysis. Never called by the tracker."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def iou(a, b):
    x = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def annotate(candidates, gt):
    result = {}
    for name in ['hann', 'raw', 'dense']:
        overlaps = [iou(c['bbox'], gt) for c in candidates[name]]
        result[name + '_top1_iou'] = overlaps[0]
        for k in [5, 10]:
            result[name + '_top' + str(k) + '_oracle_iou'] = max(overlaps[:k])
        result[name + '_all_oracle_iou'] = max(overlaps)
        result[name + '_first_rank_ge_0_5'] = next((i + 1 for i, v in enumerate(overlaps) if v >= .5), None)
    top = candidates['hann'][0]['bbox']
    cx, cy = top[0] + top[2] / 2, top[1] + top[3] / 2
    gx, gy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
    result['top1_center_inside_gt'] = gt[0] <= cx <= gt[0] + gt[2] and gt[1] <= cy <= gt[1] + gt[3]
    result['top1_iou_if_gt_size'] = iou([cx - gt[2] / 2, cy - gt[3] / 2, gt[2], gt[3]], gt)
    result['top1_iou_if_gt_center'] = iou([gx - top[2] / 2, gy - top[3] / 2, top[2], top[3]], gt)
    result['hann_changes_top1_cell'] = (candidates['hann'][0]['grid_row'], candidates['hann'][0]['grid_column']) != (candidates['raw'][0]['grid_row'], candidates['raw'][0]['grid_column'])
    return result


def summarize(rows):
    return dict(events=len(rows),
        top1_good=sum(r['hann_top1_iou'] >= .5 for r in rows),
        hann_top5_good=sum(r['hann_top5_oracle_iou'] >= .5 for r in rows),
        hann_top10_good=sum(r['hann_top10_oracle_iou'] >= .5 for r in rows),
        raw_top10_good=sum(r['raw_top10_oracle_iou'] >= .5 for r in rows),
        dense_good=sum(r['dense_all_oracle_iou'] >= .5 for r in rows),
        hann_top10_any_overlap=sum(r['hann_top10_oracle_iou'] > .1 for r in rows),
        hann_changes_peak=sum(r['hann_changes_top1_cell'] for r in rows),
        raw_top1_good_hann_bad=sum(r['raw_top1_iou'] >= .5 and r['hann_top1_iou'] < .5 for r in rows),
        top1_gt_size_good=sum(r['top1_iou_if_gt_size'] >= .5 for r in rows),
        mean_top1_iou=mean(r['hann_top1_iou'] for r in rows),
        mean_hann_top10_iou=mean(r['hann_top10_oracle_iou'] for r in rows))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    inputs = json.loads((root / 'inputs.json').read_text())
    labels = json.loads((root / 'labels_for_posthoc_only.json').read_text())
    files = list((root / 'candidates').glob('*.json'))
    assert {p.stem for p in files} == {c['key'] for c in inputs}, 'Incomplete candidate census'
    rows, wide, hashes = [], [], {}
    for file in sorted(files):
        event = json.loads(file.read_text()); key = event['key']
        assert event['public_state_unchanged_by_export_and_shadow']
        assert hashlib.sha256(file.with_suffix('.npz').read_bytes()).hexdigest() == event['maps_sha256']
        hashes[key] = hashlib.sha256(file.read_bytes()).hexdigest()
        gt = labels[key]['gt_bbox']
        row = dict(key=key, sequence=event['sequence'], geometry=labels[key]['geometry']['onset_search_class'],
                   **annotate(event['factor4'], gt))
        rows.append(row)
        if event['factor7'] is not None:
            wide.append(dict(key=key, sequence=event['sequence'], **annotate(event['factor7'], gt)))
    assert len(rows) == 124 and len(wide) == 9
    summary = dict(status='complete', all_factor4=summarize(rows),
        inside_factor4=summarize([r for r in rows if r['geometry'] == 'inside_factor4']),
        outside_factor4=summarize([r for r in rows if r['geometry'] != 'inside_factor4']),
        factor7=summarize(wide),
        per_sequence={seq: summarize([r for r in rows if r['sequence'] == seq]) for seq in sorted({r['sequence'] for r in rows})},
        replay_frames=sum(c['progress'] for c in inputs), candidate_json_sha256=hashes,
        claim_limit='GT-selected failure-onset diagnosis; oracle capacity is not a deployed rescue or a VOT metric.',
        overlap_definition='Continuous xywh IoU for candidate geometry; M39 formal benchmark metrics remain toolkit raster overlaps.')
    (root / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    for name, values in [('factor4', rows), ('factor7', wide)]:
        with (root / (name + '_diagnosis.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    print(json.dumps({k: v for k, v in summary.items() if k != 'candidate_json_sha256'}, indent=2))


if __name__ == '__main__':
    main()
