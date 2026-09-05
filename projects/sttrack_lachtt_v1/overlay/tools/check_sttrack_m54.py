"""Check reader gradients and full-state commits against native template substitution."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tools.sttrack_m54_common import check_sources, parameters, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan, parent, spec = check_sources(root)
    from lib.models.sttrack.lachtt_template_reader import TemplateReader
    from lib.test.tracker.sttrack import STTrack
    from lib.test.tracker.sttrack_template_reader import READER_FIELDS, STTrackTemplateReader
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    torch.manual_seed(plan['optimization']['seed'])
    net = TemplateReader()
    inputs = [torch.randn(3, 2, 2, 16, 768), torch.randn(3, 2, 2, 16, 768),
              torch.rand(3, 2, 2, 16, 16), torch.rand(3, 2, 4), torch.rand(3, 2)]
    logits = net(*inputs)
    assert torch.equal(logits, inputs[-1].clamp_min(1e-6).log())
    torch.nn.init.normal_(net.quality[-1].weight, std=.1)
    net.eval()
    original = net(*inputs)
    swapped = [inputs[0].flip(1), inputs[1], inputs[2].flip(1), inputs[3].flip(1), inputs[4].flip(1)]
    torch.testing.assert_close(net(*swapped), original.flip(1), rtol=0, atol=1e-6)
    torch.nn.functional.cross_entropy(original, torch.tensor([0, 1, 0])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert all(any(p.grad.abs().sum() > 0 for p in block.parameters())
               for block in [net.cell, net.match, net.local, net.response, net.quality])
    parameter_count = sum(p.numel() for p in net.parameters())

    params = parameters(spec)
    tracker = STTrackTemplateReader(params, TemplateReader())
    plain = STTrack(params)
    cases = [c for c in json.loads((parent / 'inference_inputs.json').read_text()) if c['sequence'] in plan['contract_sequences']]
    assert len(cases) == 2
    started = time.time()
    rows = []
    for case in cases:
        folder = Path(spec['dataset_root']) / case['sequence']

        def image_at(frame):
            return get_rgbd_frame(str(folder / 'color' / f'{frame+1:08d}.jpg'),
                str(folder / 'depth' / f'{frame+1:08d}.png'), dtype='rgbcolormap', depth_clip=True)

        tracker.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        plain.initialize(image_at(0), dict(init_bbox=list(case['init_bbox'])))
        tracker.choose_view = lambda observation: int(tracker.frame_id in [50, 60, 61, 62, 63, 64, 65, 100])
        updates = nonidentical = 0
        for frame in range(1, 121):
            image = image_at(frame)
            old = plain.z_dict[1]
            use_initial = frame in [50, 60, 61, 62, 63, 64, 65, 100]
            if use_initial:
                plain.z_dict[1] = plain.z_dict[0]
            supplied = plain.z_dict[1]
            expected = plain.track(image)
            wrote = plain.z_dict[1] is not supplied
            if use_initial and not wrote:
                plain.z_dict[1] = old
            actual = tracker.track(image)
            assert actual['target_bbox'] == expected['target_bbox'], (case['sequence'], frame)
            assert actual['best_score'] == float(expected['best_score'])
            assert all(torch.equal(a, b) for a, b in zip(tracker.z_dict, plain.z_dict))
            assert all(torch.equal(a, b) for a, b in zip(tracker.track_query_before, plain.track_query_before))
            assert np.array_equal(tracker.z_patch_arr, plain.z_patch_arr)
            updates += int(wrote)
            nonidentical += int(use_initial and tracker.last_boxes[0] != tracker.last_boxes[1])
            # Exercise the actual reader input path independently of the forced choice.
            observation = tracker.last_observation
            with torch.no_grad():
                expected_choice = int(tracker.reader(*[observation[k][None].float() for k in READER_FIELDS]).argmax(1)[0])
                actual_choice = STTrackTemplateReader.choose_view(tracker, observation)
            assert actual_choice == expected_choice
        rows.append(dict(sequence=case['sequence'], frames=120, native_updates=updates,
            forced_initial_reads=8, forced_reads_with_different_boxes=nonidentical, full_native_substitution_state_exact=True))
    assert sum(x['forced_reads_with_different_boxes'] for x in rows) > 0
    assert sum(x['native_updates'] for x in rows) > 0
    check_sources(root)
    result = dict(status='PASS', spec_sha256=sha(root / 'spec.json'), parameters=parameter_count,
        gradient_blocks_finite_and_active=True, view_exchange_check=True, initial_logits_equal_native_log_scores=True,
        actual_choose_view_input_path_exact=True, sequences=rows, frames=240, gt_opened=False, optimizer_steps=0,
        scope='Synthetic head wiring and causal native-substitution interface contract; no performance result',
        elapsed_seconds=time.time() - started)
    (root / 'runtime_contract.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
