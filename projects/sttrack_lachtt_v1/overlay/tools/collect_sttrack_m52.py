"""Capture fitting events on the frozen M45 model's own predicted-state path."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

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
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    from lib.utils.box_ops import clip_box

    def binding():
        check_binding(parent, spec)
        assert sha(parent/'spec.json') == plan['source_spec_sha256']
        assert sha(parent/'inference_inputs.json') == spec['inference_inputs_sha256']
        for name, digest in plan['source_sha256'].items():
            assert sha(repo/name) == digest, name
        assert sha(plan['policy_checkpoint']) == plan['policy_checkpoint_sha256']
        assert sha(Path(plan['policy_root'])/'geometry_result.json') == plan['policy_training_result_sha256']
        assert sha(Path(plan['m51_root'])/'recursive_result.json') == plan['m51_result_sha256']

    binding()
    cases = [c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split'] == 'fit']
    assert sorted(c['sequence'] for c in cases) == plan['sequences'] and len(cases) == 63
    assert sum(len(c['event_frames']) for c in cases) == 1511
    if args.contract:
        cases = [c for c in cases if c['sequence'] in plan['contract_sequences']]
    else:
        contract = json.loads((root/'contract.json').read_text())
        assert contract['status'] == 'PASS' and contract['spec_sha256'] == sha(root/'spec.json')
    torch.set_num_threads(1)
    update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    tracker = STTrackCandidateSet(params, plan['policy_checkpoint'])
    plain = STTrackCandidateSet(params, plan['policy_checkpoint']) if args.contract else None
    capture = {}
    frame = 0
    needed, events = set(), set()
    forward = tracker.network.forward

    def observed_forward(*pos, **kw):
        output = forward(*pos, **kw)
        if frame in needed:
            capture['output'] = output[0]
        return output

    def observed_association(module, inputs):
        if frame in events:
            capture['inputs'] = [x.detach().cpu() for x in inputs]

    tracker.network.forward = observed_forward
    hook = tracker.association.register_forward_pre_hook(observed_association)
    if not args.contract:
        (root/'features').mkdir()
        (root/'traces').mkdir()
    receipts = []
    started = time.time()
    for case in cases:
        name = case['sequence']
        folder = Path(spec['dataset_root'])/name
        def image_at(i):
            return get_rgbd_frame(str(folder/'color'/f'{i+1:08d}.jpg'), str(folder/'depth'/f'{i+1:08d}.png'),
                                  dtype='rgbcolormap', depth_clip=True)
        image = image_at(0)
        tracker.initialize(image, dict(init_bbox=list(case['init_bbox'])))
        if plain is not None:
            plain.initialize(image, dict(init_bbox=list(case['init_bbox'])))
        events = set(range(2, 121)) if args.contract else set(case['event_frames'])
        needed = events | {f-1 for f in events}
        assert min(events) >= 2
        tensors = {k: [] for k in ['current', 'previous', 'references', 'geometry', 'scores',
                                   'current_boxes', 'previous_boxes', 'public_bbox', 'previous_public_bbox',
                                   'selected_bbox', 'previous_selected_bbox']}
        records, trace = [], [dict(frame=0, bbox=case['init_bbox'], score=1., choice=0)]
        updates = 0
        previous_default, previous_default_frame = None, None
        for frame in range(1, max(events)+1):
            image = image_at(frame)
            prior = list(tracker.state)
            previous_set = tracker.previous_set
            previous_choice = tracker.previous_choice
            dynamic = tracker.z_dict[1]
            output = tracker.track(image)
            if plain is not None:
                reference = plain.track(image)
                assert reference == output, (name, frame)
                assert all(torch.equal(a, b) for a, b in zip(plain.z_dict, tracker.z_dict))
                assert all(torch.equal(a, b) for a, b in zip(plain.track_query_before, tracker.track_query_before))
            trace.append(dict(frame=frame, bbox=output['target_bbox'], score=float(output['best_score']),
                              choice=output['association_candidate']))
            if frame in needed:
                head = capture.pop('output')
                resize = 256 / math.ceil(math.sqrt(prior[2]*prior[3])*4.)
                with torch.no_grad():
                    response = tracker.output_window*head['score_map']
                    box = tracker.network.box_head.cal_bbox(response, head['size_map'], head['offset_map'])
                    cx, cy, width, height = (box.view(-1, 4).mean(0)*256/resize).tolist()
                # Same native map/clip operations, using the pre-frame state.
                half_side = .5*256/resize
                cx_real = cx + (prior[0] + .5*prior[2] - half_side)
                cy_real = cy + (prior[1] + .5*prior[3] - half_side)
                default = clip_box([cx_real-.5*width, cy_real-.5*height, width, height],
                                   image.shape[0], image.shape[1], margin=10)
                if output['association_candidate'] == 0:
                    assert default == output['target_bbox'], (name, frame)
                if frame in events:
                    current = tracker.previous_set
                    assert previous_default_frame == frame-1 and previous_set is not None
                    inputs = capture.pop('inputs')
                    assert int(inputs[5][0]) == previous_choice
                    assert torch.equal(inputs[0][0].half(), current['rois'].half().cpu())
                    assert torch.equal(inputs[1][0].half(), previous_set['rois'].half().cpu())
                    assert torch.equal(inputs[3][0], torch.cat([current['geometry'], previous_set['geometry']]))
                    for key, value in zip(['current', 'previous', 'references', 'geometry', 'scores'], inputs[:5]):
                        tensors[key].append(value[0].half() if key in ['current', 'previous', 'references'] else value[0])
                    tensors['current_boxes'].append(current['boxes'])
                    tensors['previous_boxes'].append(previous_set['boxes'])
                    tensors['public_bbox'].append(torch.tensor(default))
                    tensors['previous_public_bbox'].append(torch.tensor(previous_default))
                    tensors['selected_bbox'].append(torch.tensor(output['target_bbox']))
                    tensors['previous_selected_bbox'].append(torch.tensor(prior))
                    records.append(dict(key=f'{name}@{frame}', frame=frame, previous_frame=frame-1,
                                        previous_choice=previous_choice, current_choice=output['association_candidate'],
                                        template_updates_before_frame=updates))
                previous_default, previous_default_frame = default, frame
            updates += tracker.z_dict[1] is not dynamic
        assert not capture
        data = {key: torch.stack(value) for key, value in tensors.items()}
        assert all(torch.isfinite(x).all() for x in data.values()) and (data['geometry'][..., 2:] > 0).all()
        data.update(records=records, sequence=name, fold=case['fold'], split=case['split'],
                    spec_sha256=sha(root/'spec.json'), policy_checkpoint_sha256=plan['policy_checkpoint_sha256'])
        item = dict(sequence=name, events=len(records), frames=frame, template_updates=updates,
                    nonzero_previous_choice_events=sum(r['previous_choice'] != 0 for r in records),
                    changes=sum(r['choice'] != 0 for r in trace), elapsed_seconds=time.time()-started)
        if not args.contract:
            feature = root/'features'/(name+'.pt')
            torch.save(data, feature)
            path = root/'traces'/(name+'.json')
            path.write_text(json.dumps(dict(sequence=name, rows=trace), allow_nan=False)+'\n')
            item.update(feature_sha256=sha(feature), bytes=feature.stat().st_size, trace_sha256=sha(path))
        receipts.append(item)
        print(json.dumps(item), flush=True)
    hook.remove()
    binding()
    result = dict(status='PASS' if args.contract else 'complete', spec_sha256=sha(root/'spec.json'),
                  policy_checkpoint_sha256=plan['policy_checkpoint_sha256'], sequences=receipts,
                  events=sum(r['events'] for r in receipts), frames=sum(r['frames'] for r in receipts),
                  nonzero_previous_choice_events=sum(r['nonzero_previous_choice_events'] for r in receipts),
                  labels_opened=False, optimizer_steps=0, source_unchanged=True, elapsed_seconds=time.time()-started)
    if args.contract:
        assert result['nonzero_previous_choice_events'] > 0
        assert sum(r['template_updates'] for r in receipts) > 0
        result['observed_and_plain_recursive_state_exact'] = True
    else:
        assert result['events'] == 1511 and result['frames'] == 93362 and len(receipts) == 63
    (root/('contract.json' if args.contract else 'collection_receipt.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sequences'}, indent=2), flush=True)


if __name__ == '__main__': main()
