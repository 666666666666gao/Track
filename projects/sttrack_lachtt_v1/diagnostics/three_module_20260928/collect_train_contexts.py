"""Fill only the missing A+B context/support fields on the fixed M90 states."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from collect_ab_interface_panel import expanded, valid_samples, depth_fraction
from collect_train_states import sha
from instance_ab_prototype import RING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--origins', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full152-spec', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prep = json.loads((args.cache / 'preparation.json').read_text())
    assert prep['source_spec_sha256'] == sha(args.full152_spec)
    assert prep['inference_inputs_sha256'] == sha(args.cache / 'inference_inputs.json')
    spec = json.loads(args.full152_spec.read_text())
    plans = [row for row in json.loads((args.cache / 'inference_inputs.json').read_text())
             if row['shard'] == args.shard]
    if args.smoke:
        plans = [next(row for row in plans if row['split'] == 'fit')]
    original_receipt = json.loads((args.cache / f'collect_shard{args.shard}.json').read_text())
    assert original_receipt['status'] == 'complete' and not original_receipt['smoke']
    original_rows = {row['sequence']: row for row in original_receipt['sequences']}
    initial_rois = {}
    for shard in (0, 1):
        receipt = json.loads((args.origins / f'shard{shard}.json').read_text())
        path = args.origins / f'shard{shard}.pt'
        assert receipt['feature_sha256'] == sha(path)
        origins = torch.load(path, map_location='cpu')
        for i, row in enumerate(origins['records']):
            initial_rois[row['sequence']] = origins['search_rois'][i]
    sys.path.insert(0, str(args.repository))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set_observation import observe_candidate_set
    from lib.test.tracker.sttrack_local_spatial_observation import search_rois
    from lib.train.data.processing_utils import sample_target
    from lib.train.dataset.depth_utils import get_rgbd_frame

    torch.set_num_threads(1)
    torch.manual_seed(2027)
    update_config_from_file(str(args.repository / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=str(args.checkpoint), template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    forward = tracker.network.forward
    capture = {}

    def observed(*pos, **kwargs):
        kwargs['return_candidate_features'] = True
        outputs = forward(*pos, **kwargs)
        capture['output'] = outputs[0]
        return outputs

    tracker.network.forward = observed
    outdir = args.output / ('smoke_features' if args.smoke else 'features')
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = []
    for case in plans:
        sequence = case['sequence']
        folder = Path(spec['dataset_root']) / sequence

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / f'{frame + 1:08d}.jpg'),
                                 str(folder / 'depth' / f'{frame + 1:08d}.png'),
                                 dtype='rgbcolormap', depth_clip=True)

        source_path = args.cache / 'features' / f'{sequence}.pt'
        source_sha = sha(source_path)
        assert source_sha == original_rows[sequence]['feature_sha256']
        cached = torch.load(source_path, map_location='cpu')
        assert cached['event_frames'] == case['event_frames'] and cached['split'] == case['split']
        events = case['event_frames'][:3] if args.smoke else case['event_frames']
        event_indices = {frame: index for index, frame in enumerate(cached['event_frames'])}
        selected_events = set(events)
        initial_image = image_at(0)
        initial_box = list(case['init_bbox'])
        tracker.initialize(initial_image, {'init_bbox': initial_box})
        patch, resize, mask = sample_target(initial_image, initial_box, 4., output_sz=256)
        with torch.no_grad():
            output = tracker.network.forward(template=tracker.z_dict, search=[tracker.preprocessor.process(patch)],
                ce_template_mask=tracker.box_mask_z, track_query_before=None,
                keep_rate=tracker.keep_rate, return_candidate_features=True)[0]
            initial = search_rois(output['candidate_features'], [{'bbox': initial_box}], initial_box, resize)[0].half().cpu()
            initial_context = search_rois(output['candidate_features'], [{'bbox': expanded(initial_box)}],
                                           initial_box, resize)[0][:, RING].half().cpu()
        assert tracker.frame_id == 0 and tracker.track_query_before is None and list(tracker.state) == initial_box
        assert torch.equal(initial, initial_rois[sequence])
        data = dict(initial_context=initial_context,
                    initial_mask=valid_samples([initial_box], initial_box, resize, mask)[0],
                    initial_context_mask=valid_samples([initial_box], initial_box, resize, mask, True)[0],
                    initial_depth_valid=depth_fraction(cv2.imread(str(folder / 'depth' / '00000001.png'), -1),
                                                       [initial_box])[0])
        values = {key: [] for key in ('contexts', 'candidate_mask', 'context_mask', 'depth_valid')}
        maximum_box_error = maximum_score_error = 0.
        for frame in range(1, max(events) + 1):
            prior = list(tracker.state)
            image = image_at(frame)
            public = tracker.track(image)
            expected = case['expected_rows'][frame]
            box_error = float(np.abs(np.asarray(public['target_bbox']) - expected['bbox']).max())
            score_error = abs(float(public['best_score']) - expected['score'])
            assert box_error <= 1e-4 and score_error <= 1e-6, (sequence, frame, box_error, score_error)
            maximum_box_error, maximum_score_error = max(maximum_box_error, box_error), max(maximum_score_error, score_error)
            output = capture.pop('output')
            if frame in selected_events:
                index = event_indices[frame]
                _, resize, mask = sample_target(image, prior, 4., output_sz=256)
                with torch.no_grad():
                    current = observe_candidate_set(output, tracker.output_window, prior, resize, image.shape)
                    assert torch.equal(current['rois'].half().cpu(), cached['candidate_rois'][index]), (sequence, frame, 'region')
                    assert torch.equal(current['boxes'], cached['boxes'][index]), (sequence, frame, 'boxes')
                    boxes = current['boxes'].tolist()
                    context = search_rois(output['candidate_features'], [{'bbox': expanded(b)} for b in boxes],
                                           prior, resize)[:, :, RING].half().cpu()
                values['contexts'].append(context)
                values['candidate_mask'].append(valid_samples(boxes, prior, resize, mask))
                values['context_mask'].append(valid_samples(boxes, prior, resize, mask, True))
                raw_depth = cv2.imread(str(folder / 'depth' / f'{frame + 1:08d}.png'), -1)
                values['depth_valid'].append(depth_fraction(raw_depth, boxes))
        data.update({key: torch.stack(items) for key, items in values.items()})
        assert all(bool(torch.isfinite(value).all()) for value in data.values())
        data.update(sequence=sequence, split=case['split'], event_frames=events,
                    original_feature_sha256=source_sha, GT_loaded=False, text_loaded=False,
                    auxiliary_query_committed=False, prototype_state_committed=False)
        target = outdir / f'{sequence}.pt'
        assert not target.exists()
        torch.save(data, target)
        row = dict(sequence=sequence, split=case['split'], events=len(events), frames=frame,
                   maximum_box_error_px=maximum_box_error, maximum_score_error=maximum_score_error,
                   candidate_regions_exact=True, candidate_boxes_exact=True, initial_region_exact=True,
                   original_feature_sha256=source_sha, feature_sha256=sha(target), bytes=target.stat().st_size)
        records.append(row)
        print(json.dumps(dict(completed=len(records), total=len(plans), elapsed_seconds=time.time()-started, **row)), flush=True)
    result = dict(status='complete_context_collection_only', shard=args.shard, smoke=args.smoke,
                  sequences=records, events=sum(r['events'] for r in records),
                  frames=sum(r['frames'] for r in records), bytes=sum(r['bytes'] for r in records),
                  elapsed_seconds=time.time()-started, source_preparation_sha256=sha(args.cache/'preparation.json'),
                  checkpoint_sha256=sha(args.checkpoint), GT_loaded=False, text_loaded=False,
                  auxiliary_query_committed=False, prototype_state_committed=False, optimizer_steps=0)
    suffix = f"{'smoke_' if args.smoke else ''}shard{args.shard}"
    (args.output / f'{suffix}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
