"""Collect two-view visual evidence on native predicted crops, without GT."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tools.sttrack_m54_common import check_sources, event_frames, parameters, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--contract', action='store_true')
    args = parser.parse_args()
    root = args.root
    plan, parent, spec = check_sources(root)
    from lib.models.sttrack.lachtt_template_reader import TemplateReader
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_template_reader import READER_FIELDS, STTrackTemplateReader
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    torch.manual_seed(plan['optimization']['seed'])
    params = parameters(spec)
    tracker = STTrackTemplateReader(params, TemplateReader())
    tracker.choose_view = lambda observation: 0
    plain = STTrack(params) if args.contract else None
    cases = [c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['split'] == 'fit']
    assert len(cases) == 63 and sum(len(event_frames(c)) for c in cases) == 10615
    if args.contract:
        cases = [c for c in cases if c['sequence'] in plan['contract_sequences']]
    else:
        contract = json.loads((root / 'contract.json').read_text())
        assert contract['status'] == 'PASS' and contract['spec_sha256'] == sha(root / 'spec.json')
        assert (root / 'contract.exit').read_text().strip() == '0'
        runtime = json.loads((root / 'runtime_contract.json').read_text())
        assert runtime['status'] == 'PASS' and runtime['spec_sha256'] == sha(root / 'spec.json')
        assert (root / 'runtime_contract.exit').read_text().strip() == '0'
    outdir = root / ('contract_features' if args.contract else 'features')
    outdir.mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        name = case['sequence']
        folder = Path(spec['dataset_root']) / name

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
                str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

        tracker.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        if plain is not None:
            plain.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        events = set(range(1, 121)) if args.contract else set(event_frames(case))
        tracker.read_alternate = lambda: tracker.frame_id in events and tracker.z_dict[0] is not tracker.z_dict[1]
        arrays = {k: [] for k in READER_FIELDS + ['boxes']}
        records = []
        updates = different_reads = 0
        maxbox = maxscore = 0.
        for frame in range(1, max(events) + 1):
            dynamic = tracker.z_dict[1]
            image = image_at(frame)
            result = tracker.track(image)
            expected = case['expected_rows'][frame]
            box_error = float(np.abs(np.asarray(result['target_bbox']) - expected['bbox']).max())
            score_error = abs(result['best_score'] - expected['score'])
            assert box_error <= 1e-4 and score_error <= 1e-6, (name, frame, box_error, score_error)
            maxbox, maxscore = max(maxbox, box_error), max(maxscore, score_error)
            if plain is not None:
                reference = plain.track(image)
                assert reference['target_bbox'] == result['target_bbox']
                assert float(reference['best_score']) == result['best_score']
                assert all(torch.equal(a, b) for a, b in zip(plain.z_dict, tracker.z_dict))
                assert all(torch.equal(a, b) for a, b in zip(plain.track_query_before, tracker.track_query_before))
                assert np.array_equal(plain.z_patch_arr, tracker.z_patch_arr)
            if frame in events:
                for key in READER_FIELDS:
                    arrays[key].append(tracker.last_observation[key].detach().cpu())
                arrays['boxes'].append(torch.tensor(tracker.last_boxes, dtype=torch.float64))
                records.append(dict(key=f'{name}@{frame}', frame=frame, previous_frame=frame-1))
                different_reads += int(tracker.last_boxes[0] != tracker.last_boxes[1])
            updates += int(tracker.z_dict[1] is not dynamic)
        data = {key: torch.stack(value) for key, value in arrays.items()}
        assert all(torch.isfinite(value).all() for value in data.values())
        data.update(sequence=name, split='fit', fold=case['fold'], records=records, spec_sha256=sha(root / 'spec.json'))
        path = outdir / (name + '.pt')
        torch.save(data, path)
        item = dict(sequence=name, events=len(records), frames=frame, native_updates=updates,
            different_read_boxes=different_reads, max_bbox_error_px=maxbox, max_score_error=maxscore,
            feature_sha256=sha(path), bytes=path.stat().st_size, elapsed_seconds=time.time()-started)
        receipts.append(item)
        print(json.dumps(item), flush=True)
    check_sources(root)
    result = dict(status='PASS' if args.contract else 'complete', sequences=receipts,
        events=sum(x['events'] for x in receipts), frames=sum(x['frames'] for x in receipts),
        native_updates=sum(x['native_updates'] for x in receipts), different_read_boxes=sum(x['different_read_boxes'] for x in receipts),
        spec_sha256=sha(root / 'spec.json'), source_unchanged=True, labels_opened=False, optimizer_steps=0,
        plain_native_contract_exact=args.contract, elapsed_seconds=time.time()-started)
    if args.contract:
        assert result['frames'] == 240 and result['native_updates'] > 0 and result['different_read_boxes'] > 0
    else:
        assert result['events'] == 10615 and result['frames'] == 93362 and len(receipts) == 63
    (root / ('contract.json' if args.contract else 'collection_receipt.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
