"""Verify actual M57 inputs on a fitting trajectory with explicitly forced native choices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    result = hashlib.sha256()
    for name, value in state.items():
        result.update(name.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert sha(__file__) == spec['checker_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    for path, digest in spec['input_sha256'].items():
        assert sha(path) == digest, path
    code = Path(spec['code_root'])
    for name, digest in spec['source_sha256'].items():
        assert sha(code / name) == digest, name
    sys.path.insert(0, str(code))
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.models.sttrack.lachtt_initial_instance_alignment import InitialInstanceCandidateSetAssociation, content_tokens
    from lib.test.tracker.sttrack_initial_instance_candidate_set import STTrackInitialInstanceCandidateSet
    import lib.test.tracker.sttrack_candidate_set as candidate_runtime
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(code / 'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
        search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    initial = torch.load(spec['initial_fit'], map_location='cpu')
    text = torch.load(spec['text_fit'], map_location='cpu')
    cache = torch.load(spec['fit_cache'], map_location='cpu')
    assert cache['sequence'] == spec['sequence'] == 'chair01_indoor' and cache['split'] == 'fit'
    cases = json.loads(Path(spec['initialization_inputs']).read_text())
    case = next(row for row in cases if row['sequence'] == spec['sequence'])
    assert case['split'] == 'fit'
    initial_index = initial['sequences'].index(case['sequence'])
    text_index = text['sequences'].index(case['sequence'])
    reference = initial['references'][initial_index].float()
    mask = text['mask'][text_index][None]
    events = {row['frame']: index for index, row in enumerate(cache['records'])}
    assert len(events) == 22
    output = Path(spec['output'])
    output.mkdir()
    # Engineering-only action intervention: real four-arm forwards, native selection.
    candidate_runtime.select_candidate = lambda logits: torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    folder = Path(spec['dataset_root']) / case['sequence']

    def image_at(frame):
        return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
                              str(folder / 'depth' / ('%08d.png' % (frame + 1))), dtype='rgbcolormap', depth_clip=True)

    receipts = []
    started = time.time()
    for mode, use_text in [('candidate', True), ('candidate', False), ('initial', True), ('initial', False)]:
        variant = mode + ('_text' if use_text else '_visual')
        torch.manual_seed(2026)
        head = InitialInstanceCandidateSetAssociation(mode)
        digest = state_digest(head.state_dict())
        assert digest == '234b6c97a3f53bd0832679794c9e9f37b749555daa36b818e550dafb3557395a'
        checkpoint = output / (variant + '_temporary.pth')
        torch.save(dict(model=head.state_dict(), variant=variant, reference_mode=mode, use_text=use_text,
                        contract_only=True, trained=False), checkpoint)
        del head
        tracker = STTrackInitialInstanceCandidateSet(params, str(checkpoint))
        tracker.initialize(image_at(0), dict(init_bbox=case['init_bbox'], text_tokens=text['tokens'][text_index],
            text_mask=text['mask'][text_index], empty_text=text['empty']))
        assert torch.equal(tracker.reference_bank.initial.half().float().cpu(), reference)
        captured = {}

        def capture_inputs(module, values):
            if tracker.frame_id in events:
                captured[tracker.frame_id] = [value.detach().cpu().clone() for value in values]

        hook = tracker.association.head.register_forward_pre_hook(capture_inputs)
        tokens = content_tokens(text['tokens'][text_index][None], text['empty'], use_text)
        checked = []
        for frame in range(1, case['frames']):
            prediction = tracker.track(image_at(frame))
            assert prediction['association_candidate'] == 0
            if frame in events:
                index = events[frame]
                values = captured.pop(frame)
                expected = [cache[key][index:index + 1].float().clone()
                            for key in ['current', 'previous', 'references', 'geometry', 'scores']]
                expected[2][:, 0] = reference
                expected += [torch.zeros(1, dtype=torch.long), tokens, mask]
                assert len(values) == len(expected) == 8
                for key, actual, wanted in zip(['current', 'previous', 'references', 'geometry', 'scores', 'previous_choice', 'text', 'mask'], values, expected):
                    assert torch.equal(actual, wanted), (variant, frame, key, float((actual.float() - wanted.float()).abs().max()))
                assert torch.equal(torch.tensor(prediction['target_bbox']), cache['public_bbox'][index]), (variant, frame, 'bbox')
                checked.append(frame)
        hook.remove()
        assert checked == sorted(events) and not captured
        receipt = dict(variant=variant, frames=case['frames'], checked_events=checked, initial_state_sha256=digest,
            runtime_input_tensors_exactly_equal=True, event_public_boxes_exactly_equal=True,
            temporary_checkpoint_sha256=sha(checkpoint))
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
        del tracker
    result = dict(status='complete_runtime_input_contract', spec_sha256=sha(args.spec), arms=receipts,
        elapsed_seconds=time.time() - started, native_choices_forced=True, real_four_arm_heads_executed=True,
        subsequent_gt_opened=False, numeric_development_targets_opened=False, formal_training_steps=0,
        scope='One fitting sequence, all 22 cached event pairs, four real runtime paths; not an unconstrained tracking-performance comparison')
    (output / 'contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}), flush=True)


if __name__ == '__main__':
    main()
