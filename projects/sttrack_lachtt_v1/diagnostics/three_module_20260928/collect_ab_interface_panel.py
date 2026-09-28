"""Small native-equivalent Train panel with candidate rings and depth support."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from collect_train_states import sha
from instance_ab_prototype import RING


def expanded(box):
    x, y, w, h = box
    return [x - .5 * w, y - .5 * h, 2 * w, 2 * h]


def valid_samples(boxes, prior, resize, padding_mask, ring=False):
    from lib.test.tracker.sttrack_lachtt_observation import _search_origin
    left, top, _ = _search_origin(prior, 256, resize)
    unit = (np.arange(4) + .5) / 4
    masks = []
    for box in boxes:
        x, y, w, h = expanded(box) if ring else box
        yy, xx = np.meshgrid(y + unit * h, x + unit * w, indexing='ij')
        grid = torch.from_numpy(np.stack(((xx - left) * resize / 128 - 1,
                                          (yy - top) * resize / 128 - 1), axis=-1)).float()[None]
        support = torch.from_numpy(~padding_mask).float()[None, None]
        flat = F.grid_sample(support, grid, mode='nearest', padding_mode='zeros',
                             align_corners=False)[0, 0].bool().reshape(-1)
        masks.append(flat[RING] if ring else flat)
    return torch.stack(masks)


def depth_fraction(depth, boxes):
    values = []
    for x, y, w, h in boxes:
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(depth.shape[1], math.ceil(x + w)), min(depth.shape[0], math.ceil(y + h))
        region = depth[top:bottom, left:right]
        assert region.size > 0
        values.append(float((region > 0).mean()))
    return torch.tensor(values, dtype=torch.float32)


def geometry_features(data, index):
    score = data['scores'][index].float()
    cell = data['candidate_cells'][index]
    box = data['boxes'][index].float()
    prior = data['prior_bbox'][index].float()[None]
    rank = torch.arange(10, dtype=torch.float32)[:, None] / 9
    coordinates = torch.stack((cell % 16, cell // 16), dim=-1).float() / 15
    displacement = (box[:, :2] + .5 * box[:, 2:] - prior[:, :2] - .5 * prior[:, 2:]) / prior[:, 2:].clamp_min(1)
    log_size = torch.log(box[:, 2:].clamp_min(1) / prior[:, 2:].clamp_min(1))
    return torch.cat((score[:, None], (score - score[:1])[:, None], rank,
                      data['geometry'][index].float(), coordinates, displacement, log_size), dim=-1)


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
    plans = [row for row in json.loads((args.cache / 'inference_inputs.json').read_text())
             if row['split'] == 'fit'][:8]
    plans = plans[args.shard::2]
    if args.smoke:
        plans = plans[:1]
    spec = json.loads(args.full152_spec.read_text())
    saved_origins = {}
    for shard in (0, 1):
        receipt = json.loads((args.origins / f'shard{shard}.json').read_text())
        path = args.origins / f'shard{shard}.pt'
        assert receipt['feature_sha256'] == sha(path)
        data = torch.load(path, map_location='cpu')
        for i, row in enumerate(data['records']):
            saved_origins[row['sequence']] = data['search_rois'][i]
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
    original = tracker.network.forward
    capture = {}

    def observed(*pos, **kwargs):
        kwargs['return_candidate_features'] = True
        outputs = original(*pos, **kwargs)
        capture['output'] = outputs[0]
        return outputs

    tracker.network.forward = observed
    fields = {key: [] for key in ('initial_rois', 'initial_context', 'initial_mask', 'initial_context_mask',
              'initial_depth_valid', 'candidate_rois', 'contexts', 'candidate_mask', 'context_mask',
              'depth_valid', 'geometry', 'base_scores', 'boxes')}
    records = []
    for case in plans:
        folder = Path(spec['dataset_root']) / case['sequence']

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / f'{frame + 1:08d}.jpg'),
                                 str(folder / 'depth' / f'{frame + 1:08d}.png'),
                                 dtype='rgbcolormap', depth_clip=True)

        initial_image = image_at(0)
        initial_box = case['init_bbox']
        tracker.initialize(initial_image, {'init_bbox': initial_box})
        patch, resize, padding_mask = sample_target(initial_image, initial_box, 4., output_sz=256)
        with torch.no_grad():
            output = tracker.network.forward(template=tracker.z_dict, search=[tracker.preprocessor.process(patch)],
                ce_template_mask=tracker.box_mask_z, track_query_before=None,
                keep_rate=tracker.keep_rate, return_candidate_features=True)[0]
            initial = search_rois(output['candidate_features'], [{'bbox': initial_box}], initial_box, resize)[0].half().cpu()
            initial_context = search_rois(output['candidate_features'], [{'bbox': expanded(initial_box)}],
                                           initial_box, resize)[0][:, RING].half().cpu()
        assert tracker.frame_id == 0 and tracker.track_query_before is None and list(tracker.state) == initial_box
        assert torch.equal(initial, saved_origins[case['sequence']])
        fields['initial_rois'].append(initial)
        fields['initial_context'].append(initial_context)
        fields['initial_mask'].append(valid_samples([initial_box], initial_box, resize, padding_mask)[0])
        fields['initial_context_mask'].append(valid_samples([initial_box], initial_box, resize, padding_mask, True)[0])
        raw_initial_depth = cv2.imread(str(folder / 'depth' / '00000001.png'), -1)
        fields['initial_depth_valid'].append(depth_fraction(raw_initial_depth, [initial_box])[0])
        event = min(case['event_frames'])
        cached = torch.load(args.cache / 'features' / f"{case['sequence']}.pt", map_location='cpu')
        at = cached['event_frames'].index(event)
        max_box_error = max_score_error = 0.
        for frame in range(1, event + 1):
            prior = list(tracker.state)
            image = image_at(frame)
            public = tracker.track(image)
            expected = case['expected_rows'][frame]
            box_error = float(np.abs(np.asarray(public['target_bbox']) - expected['bbox']).max())
            score_error = abs(float(public['best_score']) - expected['score'])
            assert box_error <= 1e-4 and score_error <= 1e-6
            max_box_error, max_score_error = max(max_box_error, box_error), max(max_score_error, score_error)
            output = capture.pop('output')
        _, resize, padding_mask = sample_target(image, prior, 4., output_sz=256)
        with torch.no_grad():
            current = observe_candidate_set(output, tracker.output_window, prior, resize, image.shape)
            rois = current['rois'].half().cpu()
            assert torch.equal(rois, cached['candidate_rois'][at])
            assert torch.equal(current['boxes'], cached['boxes'][at])
            boxes = current['boxes'].tolist()
            contexts = search_rois(output['candidate_features'], [{'bbox': expanded(b)} for b in boxes],
                                    prior, resize)[:, :, RING].half().cpu()
        fields['candidate_rois'].append(rois)
        fields['contexts'].append(contexts)
        fields['candidate_mask'].append(valid_samples(boxes, prior, resize, padding_mask))
        fields['context_mask'].append(valid_samples(boxes, prior, resize, padding_mask, True))
        depth = cv2.imread(str(folder / 'depth' / f'{event + 1:08d}.png'), -1)
        fields['depth_valid'].append(depth_fraction(depth, boxes))
        fields['geometry'].append(geometry_features(cached, at))
        fields['base_scores'].append(cached['scores'][at])
        fields['boxes'].append(cached['boxes'][at])
        records.append(dict(sequence=case['sequence'], frame=event, split='fit',
                            prefix_calls=event, maximum_box_error_px=max_box_error,
                            maximum_score_error=max_score_error, candidate_rois_exact=True,
                            native_proposal_boxes_exact=True, initial_roi_exact=True))
    panel = {key: torch.stack(values) for key, values in fields.items()}
    assert all(bool(torch.isfinite(value).all()) for value in panel.values())
    panel.update(records=records, GT_loaded=False, text_loaded=False, tracker_state_committed_by_prototype=False)
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"{'smoke_' if args.smoke else ''}shard{args.shard}"
    target = args.output / f'{suffix}.pt'
    assert not target.exists()
    torch.save(panel, target)
    report = dict(status='complete_train_only', shard=args.shard, smoke=args.smoke,
                  records=records, panel_sha256=sha(target),
                  shapes={key: list(value.shape) for key, value in panel.items() if torch.is_tensor(value)},
                  GT_loaded=False, text_loaded=False, prototype_state_committed=False,
                  depth_descriptor='Raw depth nonzero fraction; not calibrated reliability')
    (args.output / f'{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
