"""Count training annotation gaps; these are not physical-absence labels."""
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def main():
    spec_path = ROOT / 'training_spec.json'
    spec = json.loads(spec_path.read_text())
    result = json.loads((ROOT / 'training/category/result.json').read_text())
    assert sha(spec_path) == result['training_spec_sha256']
    old_path = Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909/training_spec.json')
    old = json.loads(old_path.read_text())
    fit130 = {r['sequence'] for r in old['sequence_order']}
    assert len(fit130) == 130 and spec['sequence_order'][:130] == old['sequence_order']
    assert {r['sequence'] for r in spec['sequence_order'][130:]} == set(old['development_sequences'])
    records, sequence_counts = [], []
    invalid_count = calls = 0
    for row in spec['sequence_order']:
        path = Path(spec['dataset_root']) / row['sequence'] / 'groundtruth.txt'
        assert sha(path) == row['groundtruth_sha256']
        gt = np.loadtxt(path, delimiter=',').reshape(-1, 4)
        n = row['rgb_frames']
        if row['sequence'] == 'toy07_indoor_320':
            assert len(gt) == 1406 and n == 1367
            assert sha(path) == '683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
            gt = gt[:n]
        assert len(gt) == n
        gt32 = gt.astype(np.float32)
        valid = np.isfinite(gt32).all(axis=1) & (gt32[:, 2:] > 0).all(axis=1)
        assert valid[0]
        invalid_count += int((~valid[1:]).sum())
        calls += n - 1
        gaps = intervals(~valid)
        sequence_counts.append(dict(sequence=row['sequence'], in_previous_fit130=row['sequence'] in fit130,
                                    invalid_frames=int((~valid).sum()), gaps=len(gaps),
                                    returning_gaps=sum(end < n for start, end in gaps)))
        for start, end in gaps:
            before = start - 1
            while before > 0 and valid[before - 1]:
                before -= 1
            after = end
            while after < n and valid[after]:
                after += 1
            records.append(dict(sequence=row['sequence'], in_previous_fit130=row['sequence'] in fit130,
                                invalid_interval=[start, end], invalid_frames=end - start,
                                has_return=end < n, preceding_valid_frames=start - before,
                                following_valid_frames=after - end, groundtruth_sha256=sha(path)))
    assert len(sequence_counts) == 152 and calls == result['total_track_calls'] == 219802
    assert invalid_count == result['training_label_counts']['invalid'] == 16426
    returning = [r for r in records if r['has_return']]
    panel = [r for r in returning if r['invalid_frames'] >= 10 and r['preceding_valid_frames'] >= 10 and r['following_valid_frames'] >= 10]
    summary = dict(sequences=152, tracking_calls=calls, invalid_frames=invalid_count,
                   sequences_with_invalid=sum(r['invalid_frames'] > 0 for r in sequence_counts),
                   all_gaps=len(records), returning_gaps=len(returning),
                   returning_sequences=len({r['sequence'] for r in returning}),
                   panel_10_valid_10_invalid_10_valid=len(panel),
                   panel_sequences=len({r['sequence'] for r in panel}),
                   panel_previous_fit130=sum(r['in_previous_fit130'] for r in panel),
                   panel_former_development=sum(not r['in_previous_fit130'] for r in panel))
    report = dict(status='complete', scope='DepthTrack Train152 annotation-only census. Invalid annotation does not establish physical absence, occlusion, identity or recoverability. No test/VOT input; no model forward or training changes. 10/10/10 panel is a candidate diagnostic subset, not a learned decision threshold.',
                  source_sha256=sha(Path(__file__)), training_spec_sha256=sha(spec_path),
                  training_result_sha256=sha(ROOT / 'training/category/result.json'),
                  previous_training_spec_sha256=sha(old_path),
                  summary=summary, sequence_counts=sequence_counts, events=records, panel_events=panel)
    output = ROOT.parent / 'training_annotation_gaps.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, output=str(output), sha256=sha(output))))


if __name__ == '__main__':
    main()
