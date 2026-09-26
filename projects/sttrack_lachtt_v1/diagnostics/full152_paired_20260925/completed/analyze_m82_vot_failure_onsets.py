"""CPU-only posthoc crop census of M82 Full152 failures newly added vs native."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from vot.dataset.proxy import FrameMapSequence
from vot.region import RegionType, calculate_overlaps
from vot.tracker import Trajectory
from vot.workspace import Workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def box(region):
    assert not region.is_empty()
    rectangle = region.convert(RegionType.RECTANGLE)
    return [float(rectangle.x), float(rectangle.y),
            float(rectangle.width), float(rectangle.height)]


def crop_geometry(state, target):
    # Same integer extent/origin as frozen processing_utils.sample_target.
    x, y, w, h = state
    tx, ty, tw, th = target
    assert w > 0 and h > 0 and tw > 0 and th > 0
    side = math.ceil(4. * math.sqrt(w * h))
    left, top = round(x + .5 * w - .5 * side), round(y + .5 * h - .5 * side)
    right, bottom = left + side, top + side
    cx, cy = tx + .5 * tw, ty + .5 * th
    fraction = (max(0., min(right, tx + tw) - max(left, tx))
                * max(0., min(bottom, ty + th) - max(top, ty)) / (tw * th))
    return dict(crop_xywh=[left, top, side, side],
                center_inside=left <= cx < right and top <= cy < bottom,
                target_box_fraction_inside=fraction)


def failure_start(overlaps, proxy, grace, threshold):
    remaining = grace
    for index, overlap in enumerate(overlaps):
        if overlap <= threshold and not proxy.groundtruth(index).is_empty():
            remaining -= 1
            if remaining == 0:
                return index + 1 - grace
        else:
            remaining = grace
    return len(proxy)


def summarize(rows):
    return dict(
        anchors=len(rows),
        onset_center_inside=sum(r['onset_center_inside'] for r in rows),
        onset_center_outside=sum(not r['onset_center_inside'] for r in rows),
        onset_full_target_box_inside=sum(r['onset_box_fraction_inside'] >= 1. - 1e-12 for r in rows),
        onset_at_least_half_target_box_inside=sum(r['onset_box_fraction_inside'] >= .5 for r in rows),
        onset_score_gt_0_75=sum(r['onset_confidence'] > .75 for r in rows),
        previous_gt_empty=sum(r['previous_gt_empty'] for r in rows),
        any_empty_gt_in_previous10=sum(r['previous10_empty_gt_count'] > 0 for r in rows),
        inside_onset_then_outside_within10=sum(r['onset_center_inside'] and r['first10_center_inside_count'] < 10 for r in rows),
        outside_onset_then_inside_within10=sum(not r['onset_center_inside'] and r['first10_center_inside_count'] > 0 for r in rows),
        center_inside_all10=sum(r['first10_center_inside_count'] == 10 for r in rows),
        center_outside_all10=sum(r['first10_center_inside_count'] == 0 for r in rows),
        first10_center_inside_frames=sum(r['first10_center_inside_count'] for r in rows),
        first10_total_frames=10 * len(rows),
        onset_inside_by_relative_offset=[sum(r['first10_center_inside'][offset] for r in rows) for offset in range(10)],
        directions=dict(Counter(r['direction'] for r in rows)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.spec)
    for record in spec['inputs'].values():
        assert sha(record['path']) == record['sha256'], record['path']
    new = read(spec['inputs']['m82_result']['path'])
    native = read(spec['inputs']['native_result']['path'])
    merge = read(spec['inputs']['merge']['path'])
    assert new['merge_sha256'] == spec['inputs']['merge']['sha256']
    assert new['failure_settings'] == native['failure_settings'] == dict(
        burnin=10, grace=10, threshold=.1, ignore_masks='_ignore')
    assert set(new['failure_outcomes']) == set(native['failure_outcomes'])
    selected = {key: row for key, row in new['failure_outcomes'].items()
                if row['failed'] and not native['failure_outcomes'][key]['failed']}
    assert len(selected) == 124
    assert min(row['progress'] for row in selected.values()) > 1
    master = Path(merge['master_workspace'])
    workspace = Workspace.load(str(master))
    tracker, = workspace.registry.resolve(
        merge['tracker'], storage=workspace.storage.substorage('results'), skip_unknown=False)
    experiment = workspace.stack.experiments['baseline']
    sequences = {s.name: s for s in experiment.transform(workspace.dataset)}
    rows, raw_hashes, annotation_hashes = [], {}, {}
    for key, outcome in sorted(selected.items()):
        sequence = sequences[outcome['sequence']]
        name = f"{sequence.name}_{outcome['anchor']:08d}"
        for suffix in ('.bin', '_confidence.value'):
            relative = f"results/{merge['tracker']}/baseline/{sequence.name}/{name}{suffix}"
            assert sha(master / relative) == merge['result_sha256'][relative]
            raw_hashes[relative] = merge['result_sha256'][relative]
        folder = Path(sequence.frame(0).filename('color')).parent.parent
        if sequence.name not in annotation_hashes:
            annotation_hashes[sequence.name] = {str(p): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
        mapping = (list(range(outcome['anchor'], len(sequence))) if outcome['direction'] == 'forward'
                   else list(reversed(range(outcome['anchor'] + 1))))
        proxy = FrameMapSequence(sequence, mapping)
        trajectory = Trajectory.read(experiment.results(tracker, sequence), name)
        assert len(trajectory) == len(proxy) == outcome['run_length']
        regions = trajectory.regions()
        overlaps = list(calculate_overlaps(regions, proxy.groundtruth(), proxy.size, ignore=proxy.object('_ignore')))
        start = failure_start(overlaps, proxy, 10, .1)
        assert start == outcome['progress'], key
        assert start + 10 <= len(proxy)
        confidence_path = master / f"results/{merge['tracker']}/baseline/{sequence.name}/{name}_confidence.value"
        confidence = [None if not line.strip() else float(line) for line in confidence_path.read_text().splitlines()]
        assert len(confidence) == len(proxy)
        frames = []
        for index in range(start, start + 10):
            assert not proxy.groundtruth(index).is_empty() and overlaps[index] <= .1
            assert confidence[index] is not None and math.isfinite(confidence[index])
            previous, target = box(regions[index - 1]), box(proxy.groundtruth(index))
            frames.append(dict(run_index=index, source_frame=mapping[index],
                               previous_prediction_xywh=previous, target_xywh=target,
                               prediction_xywh=box(regions[index]), iou=float(overlaps[index]),
                               confidence=confidence[index], **crop_geometry(previous, target)))
        first = frames[0]
        row = dict(anchor_key=key, sequence=sequence.name, anchor=outcome['anchor'],
                   direction=outcome['direction'], progress=start, source_frame=mapping[start],
                   run_length=len(proxy), previous_iou=float(overlaps[start - 1]),
                   previous_gt_empty=bool(proxy.groundtruth(start - 1).is_empty()),
                   previous10_empty_gt_count=sum(proxy.groundtruth(j).is_empty() for j in range(start - 10, start)),
                   onset_iou=first['iou'], onset_confidence=first['confidence'],
                   onset_center_inside=first['center_inside'],
                   onset_box_fraction_inside=first['target_box_fraction_inside'],
                   first10_center_inside_count=sum(f['center_inside'] for f in frames),
                   first10_center_inside=[f['center_inside'] for f in frames], frames=frames)
        rows.append(row)
    assert {row['anchor_key'] for row in rows} == set(selected)
    output = Path(spec['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in row.items() if k not in ('frames', 'first10_center_inside')} for row in rows]
    csv_path = output / 'M82_VOT_new_failure_onsets.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    report = dict(status='complete', scope='Posthoc 124 new failed anchors versus native; no training, GPU inference, prediction edits or official metric replacement.',
                  geometry='Nominal factor4 crop reconstructed from saved previous predictions with sample_target integer rounding. Saved trajectory precision and padding prevent a claim of exact input-pixel replay. Center inside does not prove a correct dense candidate exists.',
                  timing='First confirmed 10-frame low-overlap segment start, not necessarily first tracking error. Backward runs use decreasing source-frame indices. Empty GT is an annotation state, not an absence label.',
                  spec_sha256=sha(args.spec), source_sha256=sha(__file__), inputs=spec['inputs'],
                  checked_trajectory_sha256=raw_hashes, sequence_annotation_sha256=annotation_hashes,
                  summary=summarize(rows),
                  per_sequence={name: summarize([r for r in rows if r['sequence'] == name]) for name in sorted({r['sequence'] for r in rows})},
                  rows=rows, csv_sha256=sha(csv_path))
    path = output / 'M82_VOT_new_failure_onsets.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], summary=report['summary'], output=str(path), sha256=sha(path)), indent=2))


if __name__ == '__main__':
    main()
