"""Collect t0 template/search references without committing the auxiliary read."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from collect_train_states import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full152-spec', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    preparation = json.loads((args.cache / 'preparation.json').read_text())
    assert preparation['source_spec_sha256'] == sha(args.full152_spec)
    assert preparation['inference_inputs_sha256'] == sha(args.cache / 'inference_inputs.json')
    spec = json.loads(args.full152_spec.read_text())
    plans = [row for row in json.loads((args.cache / 'inference_inputs.json').read_text())
             if row['shard'] == args.shard]
    if args.smoke:
        plans = [next(row for row in plans if row['split'] == 'fit')]
    sys.path.insert(0, str(args.repository))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_local_spatial_observation import template_roi, search_rois
    from lib.train.data.processing_utils import sample_target
    from lib.train.dataset.depth_utils import get_rgbd_frame

    torch.set_num_threads(1)
    torch.manual_seed(2027)
    update_config_from_file(str(args.repository / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=str(args.checkpoint), template_factor=2.,
                             template_size=128, search_factor=4., search_size=256,
                             save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    refs, records = [], []
    started = time.time()
    for case in plans:
        folder = Path(spec['dataset_root']) / case['sequence']
        rgb = folder / 'color' / '00000001.jpg'
        depth = folder / 'depth' / '00000001.png'
        initial = get_rgbd_frame(str(rgb), str(depth), dtype='rgbcolormap', depth_clip=True)
        bbox = list(case['init_bbox'])
        tracker.initialize(initial, {'init_bbox': bbox})
        assert tracker.frame_id == 0 and tracker.track_query_before is None and list(tracker.state) == bbox
        patch, resize, _ = sample_target(initial, bbox, params.search_factor, output_sz=params.search_size)
        search = tracker.preprocessor.process(patch)
        with torch.no_grad():
            output = tracker.network.forward(template=tracker.z_dict, search=[search],
                ce_template_mask=tracker.box_mask_z, track_query_before=None,
                keep_rate=tracker.keep_rate, return_candidate_features=True)[0]
            features = output['candidate_features']
            template = template_roi(features, 0, bbox).detach().half().cpu()
            search_reference = search_rois(features, [{'bbox': bbox}], bbox, resize)[0].detach().half().cpu()
        assert template.shape == search_reference.shape == (2, 16, 768)
        assert torch.isfinite(template).all() and torch.isfinite(search_reference).all()
        assert tracker.frame_id == 0 and tracker.track_query_before is None and list(tracker.state) == bbox
        next_image = get_rgbd_frame(str(folder / 'color' / '00000002.jpg'),
                                   str(folder / 'depth' / '00000002.png'),
                                   dtype='rgbcolormap', depth_clip=True)
        public = tracker.track(next_image)
        expected = case['expected_rows'][1]
        box_error = float(np.abs(np.asarray(public['target_bbox']) - expected['bbox']).max())
        score_error = abs(float(public['best_score']) - expected['score'])
        assert box_error <= 1e-4 and score_error <= 1e-6
        refs.append((template, search_reference))
        records.append({'sequence': case['sequence'], 'split': case['split'],
                        'initial_rgb_sha256': sha(rgb), 'initial_depth_sha256': sha(depth),
                        'first_native_box_error_px': box_error, 'first_native_score_error': score_error})
        print(json.dumps({'completed': len(records), 'total': len(plans),
                          'elapsed_seconds': round(time.time() - started, 1)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"{'smoke_' if args.smoke else ''}shard{args.shard}"
    feature_path = args.output / f'{suffix}.pt'
    assert not feature_path.exists()
    torch.save({'records': records, 'template_rois': torch.stack([x[0] for x in refs]),
                'search_rois': torch.stack([x[1] for x in refs]),
                'GT_loaded': False, 'auxiliary_query_committed': False}, feature_path)
    report = {'status': 'complete', 'shard': args.shard, 'smoke': args.smoke,
              'sequences': len(records), 'records': records,
              'source_preparation_sha256': sha(args.cache / 'preparation.json'),
              'checkpoint_sha256': sha(args.checkpoint), 'feature_sha256': sha(feature_path),
              'auxiliary_calls': len(records), 'native_verification_calls': len(records),
              'elapsed_seconds': time.time() - started, 'GT_loaded': False,
              'auxiliary_query_committed': False}
    (args.output / f'{suffix}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
