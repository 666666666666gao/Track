"""Posthoc VOT overlap and candidate capacity, after both replay shards seal."""
import hashlib
import json
from pathlib import Path

import numpy as np
from vot.region import RegionType, calculate_overlaps
from vot.region.shapes import Rectangle
from vot.workspace import Workspace


ROOT = Path('/root/autodl-tmp/sttrack_full152_vot_readout_20260926')
MASTER = Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/M82/vot/run/master')
ARMS = ('category', 'empty', 'native')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dense_boxes(maps, row):
    # Reused M85 decoder: all 256 locations, same search-to-image map and clip.
    grid = np.arange(256, dtype=np.float32)
    m = maps.reshape(5, 256)
    normalized = np.stack([(grid % 16 + m[3]) / 16, (grid // 16 + m[4]) / 16, m[1], m[2]], axis=1)
    values = (normalized * np.float32(256) / np.float32(256. / row['search_side'])).astype(np.float64)
    previous, (height, width) = row['previous_bbox'], row['image_hw']
    values[:, 0] += previous[0] + .5 * previous[2] - .5 * row['search_side']
    values[:, 1] += previous[1] + .5 * previous[3] - .5 * row['search_side']
    lower = values[:, :2] - .5 * values[:, 2:]
    upper = lower + values[:, 2:]
    lower = np.minimum(np.maximum(0, lower), [width - 10, height - 10])
    upper = np.minimum(np.maximum(10, upper), [width, height])
    return np.concatenate([lower, np.maximum(10, upper - lower)], axis=1)


def summarize(rows):
    bad = [r for r in rows if r['readouts']['category_hann'] <= .1]
    good = [r for r in rows if r['readouts']['category_hann'] >= .5]
    out = dict(positions=len(rows), category_low=len(bad), category_good=len(good),
               low_center_outside=sum(not r['center_inside'] for r in bad),
               low_center_inside_no_category_candidate=sum(r['center_inside'] and r['capacity']['category']['correct_count'] == 0 for r in bad),
               low_center_inside_with_category_candidate=sum(r['center_inside'] and r['capacity']['category']['correct_count'] > 0 for r in bad),
               low_all_heads_no_correct_candidate=sum(all(v['correct_count'] == 0 for v in r['capacity'].values()) for r in bad),
               rescued_single_readout={name: sum(r['readouts'][name] >= .5 for r in bad) for name in rows[0]['readouts']},
               harmed_single_readout={name: sum(r['readouts'][name] <= .1 for r in good) for name in rows[0]['readouts']})
    assert out['low_center_outside'] + out['low_center_inside_no_category_candidate'] + out['low_center_inside_with_category_candidate'] == len(bad)
    return out


def main():
    spec = read(ROOT / 'spec.json')
    cases = {case['anchor_key']: case for case in spec['cases']}
    receipts, files = {}, {}
    for shard in (0, 1):
        folder = ROOT / 'predictions' / str(shard)
        receipt = read(folder / 'receipt.json')
        assert receipt['status'] == 'complete' and receipt['mode'] == 'predictions'
        assert receipt['spec_sha256'] == sha(ROOT / 'spec.json')
        assert receipt['source_sha256'] == spec['source_sha256']['readout.py']
        assert receipt['counterfactual_state_committed'] is False
        for item in receipt['cases']:
            key = item['anchor_key']
            assert cases[key]['shard'] == shard and key not in files
            assert sha(folder / (key + '.json')) == item['json_sha256']
            assert sha(folder / (key + '.npz')) == item['dense_sha256']
            files[key] = folder
        receipts[str(shard)] = sha(folder / 'receipt.json')
    assert set(files) == set(cases) and len(cases) == 124
    # No subsequent GT is read until every raw diagnostic output has sealed.
    workspace = Workspace.load(str(MASTER))
    sequences = {s.name: s for s in workspace.stack.experiments['baseline'].transform(workspace.dataset)}
    census = read('/root/autodl-tmp/sttrack_full152_vot_onsets_20260926/M82_VOT_new_failure_onsets.json')
    assert sha('/root/autodl-tmp/sttrack_full152_vot_onsets_20260926/M82_VOT_new_failure_onsets.json') == spec['census_sha256']
    for source_files in census['sequence_annotation_sha256'].values():
        for path, digest in source_files.items():
            assert sha(path) == digest
    records, max_decode_error = [], 0.
    for key in sorted(cases):
        case, folder = cases[key], files[key]
        sequence = sequences[case['sequence']]
        rows = read(folder / (key + '.json'))['rows']
        assert [r['run_index'] for r in rows] == case['observe_indices']
        with np.load(folder / (key + '.npz'), allow_pickle=False) as archive:
            maps = {arm: archive[arm] for arm in ARMS}
            window = archive['window'].reshape(256)
        assert all(v.shape == (20, 5, 16, 16) and np.isfinite(v).all() for v in maps.values())
        for j, row in enumerate(rows):
            source = row['source_frame']
            assert source == case['anchor'] + case['direction_step'] * row['run_index']
            gt = sequence.groundtruth(source)
            assert not gt.is_empty()
            ignore = sequence.object('_ignore', source)
            rectangle = gt.convert(RegionType.RECTANGLE)
            center = np.array([rectangle.x + .5 * rectangle.width, rectangle.y + .5 * rectangle.height])
            origin = np.asarray(row['crop_origin'])
            record = dict(anchor_key=key, run_index=row['run_index'], source_frame=source,
                          relative_to_failure=row['run_index'] - case['failure_start'],
                          center_inside=bool(((center >= origin) & (center < origin + row['search_side'])).all()),
                          capacity={}, readouts={})
            candidate_overlaps = {}
            for arm, values in maps.items():
                dense = dense_boxes(values[j], row)
                raw = values[j][0].reshape(256)
                hann = raw * window
                reported = row['variants'][arm]
                assert int(raw.argmax()) == reported['raw_peak'] and int(hann.argmax()) == reported['hann_peak']
                for kind in ('raw', 'hann'):
                    error = float(np.abs(dense[reported[kind + '_peak']] - reported[kind + '_bbox']).max())
                    max_decode_error = max(max_decode_error, error)
                    assert error <= 1e-4, (key, row['run_index'], arm, kind, error)
                overlaps = np.asarray(calculate_overlaps([Rectangle(*b) for b in dense], [gt] * 256, sequence.size, ignore=[ignore] * 256))
                candidate_overlaps[arm] = overlaps
                record['capacity'][arm] = dict(best_iou=float(overlaps.max()), correct_count=int((overlaps >= .5).sum()),
                                                best_index=int(overlaps.argmax()))
                # Selected readout uses its direct Torch-decoded box, avoiding
                # numerical ambiguity at raster boundaries in vector decoding.
                direct = calculate_overlaps([Rectangle(*reported[k + '_bbox']) for k in ('raw', 'hann')],
                                            [gt] * 2, sequence.size, ignore=[ignore] * 2)
                record['readouts'][arm + '_raw'], record['readouts'][arm + '_hann'] = map(float, direct)
            for score_arm, geometry_arm in [('category', 'native'), ('native', 'category')]:
                peak = row['variants'][score_arm]['hann_peak']
                record['readouts'][score_arm + '_score_' + geometry_arm + '_geometry'] = float(candidate_overlaps[geometry_arm][peak])
            records.append(record)
    assert len(records) == spec['observe_count'] == 2480
    onset = [r for r in records if r['relative_to_failure'] == 0]
    first10 = [r for r in records if r['relative_to_failure'] >= 0]
    prior10 = [r for r in records if r['relative_to_failure'] < 0]
    # Serialized official trajectory and direct replay overlap can differ at
    # raster boundaries. Report the count; do not replace official outcomes.
    report = dict(status='complete', scope='All 124 posthoc new failures; Category histories only. Candidate capacity and alternate boxes are not executed recoveries. Pre-onset frames are from failed anchors, not an unbiased healthy-trajectory control.',
                  spec_sha256=sha(ROOT / 'spec.json'), analysis_source_sha256=sha(__file__), receipts=receipts,
                  nominal_crop_precision='Readout uses actual in-memory previous state; saved TraX input/output parity is checked separately.',
                  summaries=dict(onset=summarize(onset), first10=summarize(first10), prior10=summarize(prior10)),
                  first10_direct_category_iou_above_0_1=sum(r['readouts']['category_hann'] > .1 for r in first10),
                  max_selected_dense_decoder_error=max_decode_error, rows=records)
    output = ROOT / 'analysis.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], summaries=report['summaries'], output_sha256=sha(output)), indent=2))


if __name__ == '__main__':
    main()
