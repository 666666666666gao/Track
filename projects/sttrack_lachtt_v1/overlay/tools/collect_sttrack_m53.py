"""Seal historical native-template counterfactuals without changing recursion."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--contract', action='store_true')
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'spec.json').read_text())
    parent = Path(plan['source_root'])
    spec = json.loads((parent/'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.train_sttrack_m44 import sha, check_binding
    from tools.run_m41 import state_hash
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_lachtt_observation import clone_query_state, decode_nms_candidates
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.utils.box_ops import clip_box

    def check():
        check_binding(parent, spec)
        assert sha(parent/'spec.json') == plan['source_spec_sha256']
        assert sha(parent/'inference_inputs.json') == spec['inference_inputs_sha256']
        for name, digest in plan['source_sha256'].items():
            assert sha(repo/name) == digest, name
        assert sha(Path(plan['m52_root'])/'recursive_result.json') == plan['m52_result_sha256']

    check()
    cases = [c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split'] == 'fit']
    assert sorted(c['sequence'] for c in cases) == plan['sequences']
    assert len(cases) == 63 and sum(len(c['event_frames']) for c in cases) == 1511
    if args.contract:
        cases = [c for c in cases if c['sequence'] in plan['contract_sequences']]
    else:
        contract = json.loads((root/'contract.json').read_text())
        assert contract['status'] == 'PASS' and contract['spec_sha256'] == sha(root/'spec.json')
        assert (root/'contract.exit').read_text().strip() == '0'
    torch.set_num_threads(1)
    update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrack(params)
    plain = STTrack(params) if args.contract else None
    forward = tracker.network.forward
    capture = {}

    def observed(*pos, **kw):
        if capture.get('enabled'):
            capture['search'] = kw['search']
            capture['templates'] = [z.detach().clone() for z in kw['template']]
            capture['queries'] = clone_query_state(kw.get('track_query_before'))
            capture['mask'] = kw['ce_template_mask']
        output = forward(*pos, **kw)
        if capture.get('enabled'):
            capture['output'] = output[0]
        return output

    tracker.network.forward = observed

    def decode(output, prior, resize, shape):
        response = tracker.output_window * output['score_map']
        boxes = tracker.network.box_head.cal_bbox(response, output['size_map'], output['offset_map'])
        cx, cy, width, height = (boxes.view(-1, 4).mean(0)*256/resize).tolist()
        half_side = .5*256/resize
        cx_real = cx + (prior[0] + .5*prior[2] - half_side)
        cy_real = cy + (prior[1] + .5*prior[3] - half_side)
        bbox = clip_box([cx_real-.5*width, cy_real-.5*height, width, height], shape[0], shape[1], margin=10)
        top = decode_nms_candidates(response, output['size_map'], output['offset_map'],
                                   [prior], [resize], shape, 256, 10, 3)
        # Preserve the native arithmetic for candidate zero, as in M44/M52.
        top[0]['bbox'] = list(bbox)
        return dict(bbox=list(bbox), score=float(response.max()),
                    candidates=[dict(bbox=x['bbox'], score=x['score']) for x in top])

    outdir = root/('contract_events' if args.contract else 'events')
    outdir.mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        name = case['sequence']
        folder = Path(spec['dataset_root'])/name

        def image_at(frame):
            return get_rgbd_frame(str(folder/'color'/f'{frame+1:08d}.jpg'),
                                  str(folder/'depth'/f'{frame+1:08d}.png'),
                                  dtype='rgbcolormap', depth_clip=True)

        image = image_at(0)
        tracker.initialize(image, dict(init_bbox=list(case['init_bbox'])))
        if plain is not None:
            plain.initialize(image, dict(init_bbox=list(case['init_bbox'])))
        archive = [dict(frame=0, bbox=list(case['init_bbox']), score=1., tensor=tracker.z_dict[1].detach().clone())]
        event_frames = set(range(2, 121)) if args.contract else set(case['event_frames'])
        events = []
        shadows = updates = 0
        maxbox = maxscore = 0.
        for frame in range(1, max(event_frames)+1):
            capture.clear()
            capture['enabled'] = frame in event_frames
            image = image_at(frame)
            prior = list(tracker.state)
            dynamic = tracker.z_dict[1]
            result = tracker.track(image)
            expected = case['expected_rows'][frame]
            box_error = float(np.abs(np.asarray(result['target_bbox'])-expected['bbox']).max())
            score_error = abs(float(result['best_score'])-expected['score'])
            assert box_error <= 1e-4 and score_error <= 1e-6, (name, frame, box_error, score_error)
            maxbox, maxscore = max(maxbox, box_error), max(maxscore, score_error)
            if plain is not None:
                reference = plain.track(image)
                assert reference == result, (name, frame)
                assert all(torch.equal(a, b) for a, b in zip(plain.z_dict, tracker.z_dict))
                assert all(torch.equal(a, b) for a, b in zip(plain.track_query_before, tracker.track_query_before))
                assert np.array_equal(plain.z_patch_arr, tracker.z_patch_arr)
            if frame in event_frames:
                resize = 256/math.ceil(math.sqrt(prior[2]*prior[3])*4.)
                output = capture['output']
                baseline = decode(output, prior, resize, image.shape)
                assert baseline['bbox'] == result['target_bbox']
                assert baseline['score'] == float(result['best_score'])
                before = state_hash(tracker)

                def shadow(template):
                    with torch.no_grad():
                        return forward(template=[capture['templates'][0].detach().clone(), template.detach().clone()],
                                       search=capture['search'], ce_template_mask=capture['mask'],
                                       track_query_before=clone_query_state(capture['queries']),
                                       keep_rate=tracker.keep_rate)[0]

                control = shadow(capture['templates'][1])
                assert all(torch.equal(control[k], output[k]) for k in ['score_map', 'size_map', 'offset_map'])
                assert state_hash(tracker) == before
                alternatives = []
                assert archive[-1]['frame'] < frame
                for memory in archive[:-1]:
                    alternate = shadow(memory['tensor'])
                    alternatives.append(dict(template_frame=memory['frame'],
                                             **decode(alternate, prior, resize, image.shape)))
                    assert state_hash(tracker) == before
                    shadows += 1
                events.append(dict(key=f'{name}@{frame}', frame=frame, prior=prior,
                                   active_template_frame=archive[-1]['frame'], baseline=baseline,
                                   alternatives=alternatives, current_template_replay_exact=True,
                                   public_state_unchanged=True))
            # A frame-t write is appended only after all frame-t counterfactuals.
            if tracker.z_dict[1] is not dynamic:
                archive.append(dict(frame=frame, bbox=list(result['target_bbox']), score=float(result['best_score']),
                                    tensor=tracker.z_dict[1].detach().clone()))
                updates += 1
        path = outdir/(name+'.json')
        data = dict(sequence=name, fold=case['fold'], split=case['split'], events=events,
                    template_writes=[{k: v for k, v in memory.items() if k != 'tensor'} for memory in archive])
        path.write_text(json.dumps(data, allow_nan=False)+'\n')
        item = dict(sequence=name, events=len(events), frames=frame, template_updates=updates,
                    past_template_shadows=shadows, max_bbox_error_px=maxbox, max_score_error=maxscore,
                    sha256=sha(path), bytes=path.stat().st_size, elapsed_seconds=time.time()-started)
        receipts.append(item)
        print(json.dumps(item), flush=True)
    check()
    result = dict(status='PASS' if args.contract else 'complete', sequences=receipts,
                  events=sum(x['events'] for x in receipts), frames=sum(x['frames'] for x in receipts),
                  past_template_shadows=sum(x['past_template_shadows'] for x in receipts),
                  native_template_updates=sum(x['template_updates'] for x in receipts),
                  current_template_replay_exact=True, public_state_unchanged=True,
                  plain_native_contract_exact=args.contract, labels_opened=False, optimizer_steps=0,
                  checkpoint_sha256=spec['checkpoint_sha256'], spec_sha256=sha(root/'spec.json'),
                  source_unchanged=True, elapsed_seconds=time.time()-started)
    if args.contract:
        assert result['native_template_updates'] > 0 and result['past_template_shadows'] > 0
    else:
        assert result['events'] == 1511 and result['frames'] == 93362 and len(receipts) == 63
    (root/('contract.json' if args.contract else 'collection_receipt.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
