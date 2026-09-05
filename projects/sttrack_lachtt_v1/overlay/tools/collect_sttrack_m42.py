"""Replay frozen DepthTrack Train paths; export causal, candidate-own observations."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    root = args.root
    spec = json.loads((root / 'spec.json').read_text())
    assert sha(root / 'inference_inputs.json') == spec['inference_inputs_sha256']
    repo = Path(spec['repository'])
    for name, digest in spec['source_sha256'].items():
        assert sha(repo / name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    sys.path.insert(0, str(repo))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates
    from lib.test.tracker.sttrack_local_spatial_observation import search_rois, candidate_scalars, NativeReferenceBank
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    assert cfg.TEST.UPDATE_INTERVALS == 50 and cfg.TEST.UPDATE_THRESHOLD == .75
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    original_forward = tracker.network.forward
    capture = {}

    def observed_forward(*pos, **kw):
        kw['return_candidate_features'] = True
        result = original_forward(*pos, **kw)
        capture['output'] = result[0]
        return result

    tracker.network.forward = observed_forward
    plans = json.loads((root / 'inference_inputs.json').read_text())
    if args.smoke:
        plans = [c for c in plans if c['split'] == 'fit'][:2]
        for case in plans:
            case['event_frames'] = [10, 50, 60]
    else:
        plans = [c for c in plans if c['shard'] == args.shard]
    outdir = root / ('smoke_features' if args.smoke else 'features')
    outdir.mkdir(exist_ok=True)

    def image_at(sequence, frame):
        folder = Path(spec['dataset_root']) / sequence
        return get_rgbd_frame(str(folder / 'color' / f'{frame + 1:08d}.jpg'),
                              str(folder / 'depth' / f'{frame + 1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

    receipts = []
    start = time.time()
    for index, case in enumerate(plans):
        tracker.initialize(image_at(case['sequence'], 0), {'init_bbox': list(case['init_bbox'])})
        bank = NativeReferenceBank(case['init_bbox'])
        records = []
        tensors = {name: [] for name in ['candidates', 'references', 'scalars', 'bboxes', 'public_bbox']}
        max_error = max_score_error = 0.
        updates = 0
        events = set(case['event_frames'])
        for frame in range(1, case['event_frames'][-1] + 1):
            prior = list(tracker.state)
            dynamic_before = tracker.z_dict[1]
            current = image_at(case['sequence'], frame)
            result = tracker.track(current)
            expected = case['expected_rows'][frame]
            error = float(np.max(np.abs(np.array(result['target_bbox']) - expected['bbox'])))
            score_error = abs(float(result['best_score']) - expected['score'])
            assert error <= 1e-4 and score_error <= 1e-6, (case['sequence'], frame, error, score_error)
            max_error, max_score_error = max(max_error, error), max(max_score_error, score_error)
            output = capture.pop('output')
            features = output['candidate_features']
            assert features['template_rgb_tokens'].shape == (1, 128, 768)
            assert features['search_rgb_tokens'].shape == (1, 256, 768)
            resize = 256 / math.ceil(math.sqrt(prior[2] * prior[3]) * 4.)
            with torch.no_grad():
                references = bank.before_decision(features, dynamic_before)
                if frame in events:
                    candidates = decode_nms_candidates(tracker.output_window * output['score_map'],
                        output['size_map'], output['offset_map'], [prior], [resize], current.shape, 256, 10, 3)
                    assert np.max(np.abs(np.array(candidates[0]['bbox']) - result['target_bbox'])) < .001
                    tensors['candidates'].append(search_rois(features, candidates, prior, resize).half().cpu())
                    tensors['references'].append(references.half().cpu())
                    tensors['scalars'].append(candidate_scalars(candidates, prior))
                    tensors['bboxes'].append(torch.tensor([c['bbox'] for c in candidates]))
                    tensors['public_bbox'].append(torch.tensor(result['target_bbox']))
                    records.append(dict(key=f"{case['sequence']}@{frame}", frame=frame, prior=prior,
                        dynamic_reference_bbox=list(bank.dynamic_bbox), template_updates_before_frame=updates))
                bank.after_decision(features, prior, resize, result['target_bbox'], tracker.z_dict[1])
            updates += int(tracker.z_dict[1] is not dynamic_before)
        data = {name: torch.stack(values) for name, values in tensors.items()}
        assert all(torch.isfinite(value).all() for value in data.values())
        data.update(records=records, sequence=case['sequence'], fold=case['fold'], split=case['split'],
                    spec_sha256=sha(root / 'spec.json'))
        path = outdir / (case['sequence'] + '.pt')
        torch.save(data, path)
        receipt = dict(sequence=case['sequence'], events=len(records), frames=frame,
                       max_bbox_error_px=max_error, max_score_error=max_score_error,
                       template_updates=updates, feature_sha256=sha(path), bytes=path.stat().st_size)
        receipts.append(receipt)
        print(json.dumps(dict(done=index + 1, total=len(plans), elapsed=time.time()-start, **receipt)), flush=True)
    for name, digest in spec['source_sha256'].items():
        assert sha(repo / name) == digest, name
    receipt = dict(status='complete', elapsed_seconds=time.time()-start, sequences=receipts,
                   frames=sum(r['frames'] for r in receipts), events=sum(r['events'] for r in receipts),
                   checkpoint_sha256=spec['checkpoint_sha256'], source_unchanged=True, training_steps=0,
                   labels_opened=False, spec_sha256=sha(root / 'spec.json'))
    name = 'smoke_receipt.json' if args.smoke else f'shard{args.shard}_receipt.json'
    (root / name).write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
