"""Export completed M39 evidence and isolate M41 inference from onset GT."""
import csv
import hashlib
import json
from pathlib import Path
import argparse
import numpy as np
from vot.region import RegionType
from vot.region.io import read_trajectory
from vot.workspace import Workspace
from vot.analysis.multistart import AccuracyRobustness, EAOCurves


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box(region):
    r = region.convert(RegionType.RECTANGLE)
    return [r.x, r.y, r.width, r.height]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    m39 = Path('/root/autodl-tmp/sttrack_lachtt_m39_vot_low22_template_ablation_v1_20260902')
    m40 = Path('/root/autodl-tmp/sttrack_lachtt_m40_failure_start_census_v1_20260902')
    repo = Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
    sequence_root = Path('/root/autodl-tmp/VOT-RGBD2022/sequences')
    m39_result = json.loads((m39 / 'm39_result.json').read_text())
    rows = list(csv.DictReader((m40 / 'result/m40_failure_start_rows.csv').open()))
    assert len(rows) == 124
    cases, labels, per_sequence = [], {}, []
    for row in rows:
        seq, key = row['sequence'], row['anchor_key']
        anchor, progress = int(row['anchor']), int(row['progress'])
        with (m39 / 'default/master/results/sttrack_m39_default_low22/baseline' / seq / f'{seq}_{anchor:08d}.bin').open('rb') as f:
            trajectory = read_trajectory(f)
        gt = [[float(v) for v in line.split(',')] for line in (sequence_root / seq / 'groundtruth.txt').read_text().splitlines()]
        direction = 1 if row['direction'] == 'forward' else -1
        cases.append(dict(key=key, sequence=seq, anchor=anchor, direction=direction,
                          progress=progress, onset_frame=anchor + direction * progress,
                          init_bbox=gt[anchor], expected_boxes=[box(r) for r in trajectory[1:progress + 1]],
                          expected_confidence=float(row['failure_confidence']),
                          wide=row['onset_search_class'] != 'inside_factor4'))
        labels[key] = dict(gt_bbox=gt[anchor + direction * progress], geometry=row)
    priority = ['cup02_indoor_1', 'toy09_indoor_1', 'shoes02_indoor_1', 'cube05_indoor_5']
    cases.sort(key=lambda c: (priority.index(c['sequence']) if c['sequence'] in priority else 4, c['sequence'], c['anchor']))
    totals = [0, 0]
    for c in cases:
        shard = totals.index(min(totals))
        c['shard'] = shard
        totals[shard] += c['progress']
    (root / 'inputs.json').write_text(json.dumps(cases, indent=2) + '\n')
    (root / 'labels_for_posthoc_only.json').write_text(json.dumps(labels, indent=2) + '\n')
    # Compute per-sequence metrics with the installed official toolkit on saved trajectories.
    for arm, tracker_id in [('default', 'sttrack_m39_default_low22'), ('no_update', 'sttrack_m39_no_update_low22')]:
        workspace = Workspace.load(str(m39 / arm / 'master'))
        tracker = workspace.registry.resolve(tracker_id, storage=workspace.storage.substorage('results'), skip_unknown=False)[0]
        experiment = workspace.stack.experiments['baseline']
        ar = AccuracyRobustness(burnin=10, grace=10, bounded=True, threshold=.1)
        eao = EAOCurves(burnin=10, grace=10, bounded=True, threshold=.1, high=755)
        for sequence in experiment.transform(workspace.dataset):
            acc, rob, _, _, _ = ar.subcompute(experiment, tracker, sequence, [])
            (curve, active), _ = eao.subcompute(experiment, tracker, sequence, [])
            masked_curve = [v if a else 0.0 for v, a in zip(curve, active)]
            entry = dict(arm=arm, sequence=sequence.name, eao=100 * float(np.mean(masked_curve[115:756])),
                         acc=100 * acc, rob=100 * rob)
            entry.update(m39_result['arms'][arm]['per_sequence_failures'][sequence.name])
            per_sequence.append(entry)
    (root / 'm39_per_sequence_metrics.json').write_text(json.dumps(per_sequence, indent=2) + '\n')
    with (root / 'm39_per_sequence_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_sequence[0]))
        writer.writeheader(); writer.writerows(per_sequence)
    tracked = ['lib/test/tracker/sttrack.py', 'lib/test/tracker/sttrack_lachtt_observation.py',
               'lib/models/sttrack/sttrack.py', 'lib/train/data/processing_utils.py',
               'lib/train/dataset/depth_utils.py', 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml']
    spec = dict(schema='sttrack_m41_candidate_capacity_v1', repository=str(repo),
                sequence_root=str(sequence_root), checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
                checkpoint_sha256='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98',
                source_sha256={p: sha(repo / p) for p in tracked}, inputs_sha256=sha(root / 'inputs.json'),
                nms_top_k=10, nms_kernel=3, search_factors=[4, 7], expected_events=124,
                expected_wide_events=9, replay_frames=sum(totals), shard_frames=totals,
                bbox_serialization_tolerance_px=.001, confidence_tolerance=.00001,
                gt_policy='Only anchor initialization GT enters tracker; onset GT is posthoc-only. GT-selected event times are diagnosis, not a deployment trigger.',
                no_training=True, no_state_commit_from_shadow=True, full127=False)
    (root / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(dict(events=len(cases), replay_frames=sum(totals), shards=totals, per_sequence_rows=len(per_sequence))))


if __name__ == '__main__':
    main()
