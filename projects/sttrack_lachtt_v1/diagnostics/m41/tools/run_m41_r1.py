"""Replay M39 default exactly; seal candidate maps before posthoc GT analysis."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(box, factor):
    side = math.ceil(math.sqrt(box[2] * box[3]) * factor)
    return [round(box[0] + box[2] / 2 - side / 2), round(box[1] + box[3] / 2 - side / 2), side]


def state_hash(tracker):
    digest = hashlib.sha256(json.dumps([tracker.state, tracker.frame_id]).encode())
    for tensor in tracker.z_dict + (tracker.track_query_before or []):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root / 'spec.json').read_text())
    repo = Path(spec['repository'])
    assert sha(root / 'inputs.json') == spec['inputs_sha256']
    for name, digest in spec['source_sha256'].items():
        assert sha(repo / name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    sys.path.insert(0, str(repo))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates, clone_query_state
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.train.data.processing_utils import sample_target
    torch.set_num_threads(1)
    update_config_from_file(str(repo / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    original_forward = tracker.network.forward
    capture = {}

    def observed_forward(*pos, **kw):
        if capture.get('enabled'):
            kw['return_candidate_features'] = True
        result = original_forward(*pos, **kw)
        if capture.get('enabled'):
            capture['output'] = result[0]
        return result

    tracker.network.forward = observed_forward
    cases = json.loads((root / 'inputs.json').read_text())
    if args.smoke:
        cases = [min([c for c in cases if c['sequence'] == 'cup02_indoor_1'], key=lambda c: c['progress']),
                 min([c for c in cases if c['wide']], key=lambda c: c['progress'])]
    else:
        cases = [c for c in cases if c['shard'] == args.shard]
    outdir = root / ('smoke' if args.smoke else 'candidates')
    outdir.mkdir(exist_ok=True)

    def image_at(sequence, frame):
        folder = Path(spec['sequence_root']) / sequence
        return get_rgbd_frame(str(folder / 'color' / f'{frame + 1:08d}.jpg'),
                              str(folder / 'depth' / f'{frame + 1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

    def export_maps(output, prior, resize, image_shape):
        raw = output['score_map']
        response = tracker.output_window * raw
        records = {}
        for name, values in [('hann', response), ('raw', raw)]:
            records[name] = decode_nms_candidates(values, output['size_map'], output['offset_map'],
                [prior], [resize], image_shape, 256, spec['nms_top_k'], spec['nms_kernel'])
        records['dense'] = decode_nms_candidates(raw, output['size_map'], output['offset_map'],
                [prior], [resize], image_shape, 256, 256, 1)
        maps = {k: output[k].detach().cpu().numpy() for k in ['score_map', 'size_map', 'offset_map']}
        # Pre-fusion modality tokens are evidence; neither branch has a separately trained box head.
        features = output['candidate_features']
        for k in ['search_rgb_tokens', 'search_depth_tokens', 'search_fused_tokens']:
            maps[k] = features[k].detach().cpu().numpy()
        return records, maps

    start = time.time()
    for index, case in enumerate(cases):
        capture.clear()
        tracker.initialize(image_at(case['sequence'], case['anchor']), {'init_bbox': list(case['init_bbox'])})
        maximum_error = 0.
        for step in range(1, case['progress'] + 1):
            expected_prior = case['init_bbox'] if step == 1 else case['expected_boxes'][step - 2]
            assert geometry(tracker.state, 4) == geometry(expected_prior, 4), (case['key'], step, 'crop mismatch')
            current = image_at(case['sequence'], case['anchor'] + case['direction'] * step)
            if step == case['progress']:
                prior = list(tracker.state)
                templates = [z.detach().clone() for z in tracker.z_dict]
                queries = clone_query_state(tracker.track_query_before)
                mask = tracker.box_mask_z
                capture['enabled'] = True
            result = tracker.track(current)
            error = float(np.max(np.abs(np.array(result['target_bbox']) - case['expected_boxes'][step - 1])))
            maximum_error = max(maximum_error, error)
            assert error <= spec['bbox_serialization_tolerance_px'], (case['key'], step, error)
        assert abs(float(result['best_score']) - case['expected_confidence']) <= spec['confidence_tolerance']
        output = capture.pop('output')
        capture['enabled'] = False
        resize = 256 / geometry(prior, 4)[2]
        records, arrays = export_maps(output, prior, resize, current.shape)
        assert np.max(np.abs(np.array(records['hann'][0]['bbox']) - result['target_bbox'])) < .001
        event = dict(key=case['key'], sequence=case['sequence'], onset_frame=case['onset_frame'],
                     prior=prior, public_bbox=result['target_bbox'], public_score=float(result['best_score']),
                     replay_frames=case['progress'], max_replay_bbox_error_px=maximum_error,
                     factor4=records, factor7=None)
        before = state_hash(tracker)
        if case['wide']:
            patch, wide_resize, _ = sample_target(current, prior, 7., output_sz=256)
            search = tracker.preprocessor.process(patch)
            with torch.no_grad():
                wide = original_forward(template=templates, search=[search], ce_template_mask=mask,
                      track_query_before=queries, keep_rate=tracker.keep_rate, return_candidate_features=True)[0]
            event['factor7'], wide_arrays = export_maps(wide, prior, wide_resize, current.shape)
            arrays.update({'wide_' + k: v for k, v in wide_arrays.items()})
        event['public_state_unchanged_by_export_and_shadow'] = state_hash(tracker) == before
        assert event['public_state_unchanged_by_export_and_shadow']
        path = outdir / (case['key'] + '.npz')
        np.savez_compressed(path, **arrays)
        event['maps_sha256'] = sha(path)
        (outdir / (case['key'] + '.json')).write_text(json.dumps(event, indent=2) + '\n')
        print(json.dumps(dict(done=index + 1, total=len(cases), key=case['key'], elapsed=time.time() - start,
                              max_replay_bbox_error_px=maximum_error)), flush=True)
    for name, digest in spec['source_sha256'].items():
        assert sha(repo / name) == digest, name
    receipt = dict(status='complete', events=len(cases), elapsed_seconds=time.time() - start,
                   checkpoint_sha256=spec['checkpoint_sha256'], source_unchanged=True, training_steps=0)
    (root / ('smoke_receipt.json' if args.smoke else f'shard{args.shard}_receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
