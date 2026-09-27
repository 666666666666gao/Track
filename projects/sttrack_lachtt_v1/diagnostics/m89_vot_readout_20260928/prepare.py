"""Freeze replay inputs for all 57 new M89 Candidate VOT failures; CPU only."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from vot.region import RegionType
from vot.tracker import Trajectory
from vot.workspace import Workspace


ROOT = Path('/root/autodl-tmp/sttrack_m89_vot_readout_20260928')
EVAL = Path('/root/autodl-tmp/sttrack_m89_evaluation_20260926')
CENSUS = Path('/root/autodl-tmp/sttrack_m89_vot_onsets_20260928/M89_Candidate_VOT_new_failure_onsets.json')
sys.path.insert(0, '/root/autodl-tmp/sttrack_full152_evaluation_20260925/interface')
from initialization_text import vot_wire_bbox


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert sha(CENSUS) == 'f37db10e2684eca18271eba3341b6fb4f3dbfab0625ead704ff1cc23b57eeefc'
    census = read(CENSUS)
    merge_path = EVAL / 'candidate/vot/run/merge_result.json'
    assert sha(merge_path) == census['inputs']['merge']['sha256']
    merge = read(merge_path)
    workspace = Workspace.load(merge['master_workspace'])
    tracker, = workspace.registry.resolve(merge['tracker'], storage=workspace.storage.substorage('results'), skip_unknown=False)
    experiment = workspace.stack.experiments['baseline']
    sequences = {s.name: s for s in experiment.transform(workspace.dataset)}
    reference_dir = ROOT / 'references'
    reference_dir.mkdir()
    cases = []
    for row in census['rows']:
        sequence = sequences[row['sequence']]
        anchor, start = row['anchor'], row['progress']
        direction = 1 if row['direction'] == 'forward' else -1
        mapping = list(range(anchor, len(sequence))) if direction == 1 else list(range(anchor, -1, -1))
        name = f'{sequence.name}_{anchor:08d}'
        relative = f"results/{merge['tracker']}/baseline/{sequence.name}/{name}"
        for suffix in ('.bin', '_confidence.value'):
            assert sha(Path(merge['master_workspace']) / (relative + suffix)) == merge['result_sha256'][relative + suffix]
        trajectory = Trajectory.read(experiment.results(tracker, sequence), name)
        assert len(trajectory) == len(mapping) == row['run_length']
        # Only the protocol initialization box goes into the GPU replay input.
        init = sequence.groundtruth(anchor).convert(RegionType.RECTANGLE)
        init_box = vot_wire_bbox([init.x, init.y, init.width, init.height])
        prefix_length = start + 10
        boxes = [init_box]
        for region in trajectory.regions()[1:prefix_length]:
            assert not region.is_empty()
            region = region.convert(RegionType.RECTANGLE)
            boxes.append([region.x, region.y, region.width, region.height])
        confidence_path = Path(merge['master_workspace']) / (relative + '_confidence.value')
        scores = [0.] + [float(line) for line in confidence_path.read_text().splitlines()[1:prefix_length]]
        assert len(boxes) == len(scores) == prefix_length
        folder = Path(sequence.frame(0).filename('color')).parent.parent
        for source_index in (0, anchor, mapping[prefix_length - 1]):
            for channel, extension in [('color', 'jpg'), ('depth', 'png')]:
                assert Path(sequence.frame(source_index).filename(channel)) == folder / channel / f'{source_index + 1:08d}.{extension}'
        reference = reference_dir / (row['anchor_key'] + '.npz')
        np.savez_compressed(reference, boxes=np.asarray(boxes, dtype=np.float64), scores=np.asarray(scores, dtype=np.float64))
        cases.append(dict(anchor_key=row['anchor_key'], sequence=sequence.name, anchor=anchor,
                          direction=row['direction'], direction_step=direction,
                          folder=str(folder), init_bbox=init_box, prefix_length=prefix_length,
                          observe_indices=list(range(start - 10, start + 10)), failure_start=start,
                          reference=str(reference), reference_sha256=sha(reference)))
    assert len(cases) == 57
    # Balance known replay lengths, without selecting by candidate quality.
    costs = [0, 0]
    for case in sorted(cases, key=lambda c: (-c['prefix_length'], c['anchor_key'])):
        shard = min(range(2), key=lambda index: (costs[index], index))
        case['shard'] = shard
        costs[shard] += case['prefix_length'] - 1
    plan_path = EVAL / 'candidate/vot/plan.json'
    spec = dict(status='prepared', selection='All 57 added failures versus native, posthoc external diagnostic, not training data.',
                census_sha256=sha(CENSUS), merge_sha256=sha(merge_path),
                plan_path=str(plan_path), plan_sha256=sha(plan_path),
                source_sha256={name: sha(ROOT / name) for name in ['prepare.py', 'readout.py']},
                observe_count=sum(len(c['observe_indices']) for c in cases),
                replay_calls=sum(costs), shard_replay_calls=costs, cases=cases,
                later_groundtruth_in_replay=False, candidate_rescue_is_not_executed_recovery=True)
    output = ROOT / 'spec.json'
    output.write_text(json.dumps(spec, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=spec['status'], anchors=len(cases), replay_calls=sum(costs),
                         observe_count=spec['observe_count'], shard_calls=costs, spec_sha256=sha(output)), indent=2))


if __name__ == '__main__':
    main()
