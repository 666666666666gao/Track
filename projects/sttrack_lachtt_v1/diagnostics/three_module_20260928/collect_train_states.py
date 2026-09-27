"""Collect native candidate RGB-D RoIs and immutable instance tokens on Train only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dense_boxes(output, prior, resize, image_shape):
    size = output['size_map'][0].detach().float().cpu()
    offset = output['offset_map'][0].detach().float().cpu()
    height, width = size.shape[-2:]
    rows, columns = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    scale = 256. / resize
    center_x = (columns + offset[0]) / width * scale + prior[0] + .5 * prior[2] - .5 * scale
    center_y = (rows + offset[1]) / height * scale + prior[1] + .5 * prior[3] - .5 * scale
    box_width = size[0] * scale
    box_height = size[1] * scale
    x1 = center_x - .5 * box_width
    y1 = center_y - .5 * box_height
    x2 = x1 + box_width
    y2 = y1 + box_height
    image_height, image_width = image_shape[:2]
    x1 = x1.clamp(0, image_width - 10)
    y1 = y1.clamp(0, image_height - 10)
    x2 = x2.clamp(10, image_width)
    y2 = y2.clamp(10, image_height)
    return torch.stack((x1, y1, (x2 - x1).clamp_min(10),
                        (y2 - y1).clamp_min(10)), dim=-1).reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--full152-spec', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    preparation = json.loads((args.root / 'preparation.json').read_text())
    assert preparation['source_spec_sha256'] == sha(args.full152_spec)
    assert preparation['inference_inputs_sha256'] == sha(args.root / 'inference_inputs.json')
    spec = json.loads(args.full152_spec.read_text())
    sys.path.insert(0, str(args.repository))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_candidate_set_observation import observe_candidate_set
    from lib.test.tracker.sttrack_local_spatial_observation import NativeReferenceBank
    from lib.train.dataset.depth_utils import get_rgbd_frame

    torch.set_num_threads(1)
    update_config_from_file(str(args.repository / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=str(args.checkpoint), template_factor=2.,
                             template_size=128, search_factor=4., search_size=256,
                             save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    forward = tracker.network.forward
    capture = {}

    def observed(*pos, **kwargs):
        kwargs['return_candidate_features'] = True
        outputs = forward(*pos, **kwargs)
        capture['output'] = outputs[0]
        return outputs

    tracker.network.forward = observed
    plans = json.loads((args.root / 'inference_inputs.json').read_text())
    plans = [row for row in plans if row['shard'] == args.shard]
    if args.smoke:
        plans = [next(row for row in plans if row['split'] == 'fit')]
        plans[0]['event_frames'] = [10, 50, 60]
    outdir = args.root / ('smoke_features' if args.smoke else 'features')
    outdir.mkdir(exist_ok=True)
    started = time.time()
    receipts = []

    def image_at(sequence, frame):
        folder = Path(spec['dataset_root']) / sequence
        return get_rgbd_frame(str(folder / 'color' / f'{frame + 1:08d}.jpg'),
                              str(folder / 'depth' / f'{frame + 1:08d}.png'),
                              dtype='rgbcolormap', depth_clip=True)

    for index, case in enumerate(plans):
        sequence = case['sequence']
        tracker.initialize(image_at(sequence, 0), dict(init_bbox=case['init_bbox']))
        bank = NativeReferenceBank(case['init_bbox'])
        events = set(case['event_frames'])
        values = {key: [] for key in ('candidate_rois', 'geometry', 'scores', 'boxes',
                  'candidate_cells', 'dense_boxes', 'score_map', 'response_map',
                  'size_map', 'offset_map', 'prior_bbox', 'resize_factor',
                  'image_shape', 'public_bbox')}
        initial = None
        maximum_box_error = maximum_score_error = 0.
        for frame in range(1, max(events) + 1):
            prior = list(tracker.state)
            dynamic = tracker.z_dict[1]
            image = image_at(sequence, frame)
            public = tracker.track(image)
            expected = case['expected_rows'][frame]
            box_error = float(np.abs(np.asarray(public['target_bbox']) - expected['bbox']).max())
            score_error = abs(float(public['best_score']) - expected['score'])
            assert box_error <= 1e-4 and score_error <= 1e-6, (sequence, frame, box_error, score_error)
            maximum_box_error = max(maximum_box_error, box_error)
            maximum_score_error = max(maximum_score_error, score_error)
            output = capture.pop('output')
            resize = 256 / math.ceil(math.sqrt(prior[2] * prior[3]) * 4.)
            with torch.no_grad():
                references = bank.before_decision(output['candidate_features'], dynamic)
                if initial is None:
                    initial = references[0].half().cpu()
                if frame in events:
                    current = observe_candidate_set(output, tracker.output_window, prior, resize, image.shape)
                    assert np.abs(current['boxes'][0].numpy() - np.asarray(public['target_bbox'])).max() < .001
                    full_boxes = dense_boxes(output, prior, resize, image.shape)
                    cells = torch.tensor([c['grid_row'] * output['score_map'].shape[-1] + c['grid_column']
                                          for c in current['candidates']], dtype=torch.long)
                    assert torch.max(torch.abs(full_boxes[cells] - current['boxes'])) < .001
                    values['candidate_rois'].append(current['rois'].half().cpu())
                    for key in ('geometry', 'scores', 'boxes'):
                        values[key].append(current[key])
                    values['candidate_cells'].append(cells)
                    values['dense_boxes'].append(full_boxes)
                    values['score_map'].append(output['score_map'][0].detach().float().cpu())
                    values['response_map'].append((tracker.output_window * output['score_map'])[0].detach().float().cpu())
                    values['size_map'].append(output['size_map'][0].detach().float().cpu())
                    values['offset_map'].append(output['offset_map'][0].detach().float().cpu())
                    values['prior_bbox'].append(torch.tensor(prior, dtype=torch.float32))
                    values['resize_factor'].append(torch.tensor(resize, dtype=torch.float32))
                    values['image_shape'].append(torch.tensor(image.shape[:2], dtype=torch.int32))
                    values['public_bbox'].append(torch.tensor(public['target_bbox']))
                bank.after_decision(output['candidate_features'], prior, resize, public['target_bbox'], tracker.z_dict[1])
        data = {key: torch.stack(items) for key, items in values.items()}
        assert all(bool(torch.isfinite(tensor).all()) for tensor in data.values())
        data.update(initial_rois=initial, sequence=sequence, split=case['split'],
                    event_frames=case['event_frames'], labels_loaded=False,
                    feature_candidate_count=10, dense_candidate_count=256,
                    preparation_sha256=sha(args.root / 'preparation.json'))
        path = outdir / f'{sequence}.pt'
        torch.save(data, path)
        receipt = dict(sequence=sequence, split=case['split'], frames=frame,
                       events=len(events), maximum_box_error_px=maximum_box_error,
                       maximum_score_error=maximum_score_error,
                       feature_sha256=sha(path), bytes=path.stat().st_size)
        receipts.append(receipt)
        print(json.dumps(dict(done=index + 1, total=len(plans), **receipt)), flush=True)
    result = dict(status='complete', shard=args.shard, smoke=args.smoke,
                  sequences=receipts, events=sum(row['events'] for row in receipts),
                  frames=sum(row['frames'] for row in receipts),
                  labels_loaded=False, model_updates=0,
                  checkpoint_sha256=sha(args.checkpoint),
                  preparation_sha256=sha(args.root / 'preparation.json'),
                  elapsed_seconds=time.time() - started)
    (args.root / (f'smoke_shard{args.shard}.json' if args.smoke else f'collect_shard{args.shard}.json')).write_text(
        json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
