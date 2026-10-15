"""Real Train152-prefix interface witness; never a model-selection experiment."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.nn import functional as F

from train_m125_region_fixed_pass import initialize_training
from train_m122_full_causal import load_truth, precision_objective
from train_m121_dense_target import box_overlap
from full_dense_tracker import state_digest
from bind_m122_official_initializations import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parsed = parser.parse_args()
    gate = json.loads(parsed.gate.read_text())
    for path, digest in gate['source_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in gate['input_sha256'].items():
        assert sha(path) == digest, path
    args = SimpleNamespace(**{name: Path(gate[name]) for name in ['spec', 'repository', 'checkpoint',
        'clip_weight', 'bank', 'labels', 'warm_final', 'warm_result']}, output=parsed.output, epoch=1, resume=None, arm=gate['region_arm'])
    spec, bank, warm, decoder, writer, actor, tracker, frozen, initial, optimizer, resume, before = initialize_training(args)
    assert resume['track_calls'] == resume['optimizer_steps'] == resume['C_optimizer_steps'] == 0
    assert not optimizer.state and not resume['sequence_records']
    import sys
    sys.path.insert(0, str(args.repository))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    case = spec['sequence_order'][0]
    folder = Path(spec['dataset_root']) / case['sequence']
    truth = load_truth(folder, case)

    def image(frame):
        return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
            str(folder / 'depth' / ('%08d.png' % (frame + 1))), dtype='rgbcolormap', depth_clip=True)

    parsed.output.mkdir()
    started = time.time()
    tracker.initialize(image(0), case['first_box'], bank['sequences'].index(case['sequence']))
    optimizer.zero_grad(set_to_none=True)
    AB_count = C_count = 0
    batches, actions = [], []
    for frame in range(1, 65):
        out, data, state, C_prediction = tracker.step(image(frame))
        assert out['boxes'].shape == (1, 256, 4)
        assert np.isfinite(truth[frame]).all() and (truth[frame, 2:] > 0).all()
        target = torch.tensor(truth[frame:frame + 1], device='cuda:0', dtype=torch.float32)
        loss, _, _ = precision_objective(out, data, target, 1)
        assert bool(torch.isfinite(loss))
        (loss / 32).backward()
        AB_count += 1
        if C_prediction is not None:
            C_target = box_overlap(out['selected_box'].detach().float()[:, None], target)[0][:, 0].detach()
            C_loss = F.mse_loss(C_prediction, C_target)
            assert bool(torch.isfinite(C_loss))
            (C_loss / 32).backward()
            C_count += 1
        actions.append(state)
        if frame % 32 == 0:
            for parameter in decoder.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(32 / AB_count)
            for parameter in writer.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(32 / C_count)
            assert all(bool(p.grad.isfinite().all()) for model in [decoder, writer]
                       for p in model.parameters() if p.grad is not None)
            norms = [float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)) for model in [decoder, writer]]
            assert all(np.isfinite(norm) and norm > 0 for norm in norms)
            region_norms = {prefix: sum(float(p.grad.norm()) for name, p in decoder.named_parameters()
                if name.startswith(prefix) and p.grad is not None) for prefix in decoder.NEW_PREFIXES}
            batches.append(dict(frame=frame, region_gradient_norms=region_norms, AB_supervised=AB_count, C_supervised=C_count,
                                AB_gradient_norm=norms[0], C_gradient_norm=norms[1]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            AB_count = C_count = 0
        del out, data, C_prediction
    assert all(value > 0 for value in batches[-1]['region_gradient_norms'].values())
    assert actor.frozen_digest() == frozen
    assert all(p.grad is None for model in [actor.native.network, actor.clip] for p in model.parameters())
    final_state = dict(A_B=state_digest(decoder), C=state_digest(writer))
    assert all(final_state[key] != initial[key] for key in initial)
    saved = dict(A_B=decoder.state_dict(), C=writer.state_dict(), optimizer=optimizer.state_dict())
    path = parsed.output / 'sanity_only.pt'
    torch.save(saved, path)
    restored = torch.load(path, map_location='cpu')
    for kind in ['A_B', 'C']:
        assert all(torch.equal(value.cpu(), restored[kind][name]) for name, value in saved[kind].items())
    assert restored['optimizer']['param_groups'] == saved['optimizer']['param_groups']
    for key, state in saved['optimizer']['state'].items():
        for name, value in state.items():
            assert torch.equal(value.cpu(), restored['optimizer']['state'][key][name])
    optimizer.load_state_dict(restored['optimizer'])
    result = dict(status='complete_M125_region_real64_interface_witness', seed=2027,
        training_sequence=case['sequence'], track_calls=64, region_arm=args.arm,
        AB_parameters=413517, additional_region_parameters=33350, combined_parameters=484238, AB_optimizer_steps=2, C_optimizer_steps=2,
        batches=batches, initial_state_sha256=initial, final_state_sha256=final_state,
        warm_sha256=sha(args.warm_final), initial_parameters_equal_common_warm=True,
        initial_optimizer_empty=True, formal_training_resume_uses_this_sanity=False,
        frozen_before_after_exact=True, saved_model_and_optimizer_roundtrip_exact=True,
        candidates_per_state=256, GT_used_after_action_for_loss_only=True, GT_reinitializations_after_first_frame=0,
        gate_sha256=sha(parsed.gate), sanity_only_checkpoint_sha256=sha(path),
        source_sha256=sha(__file__), elapsed_seconds=time.time() - started, actions=actions)
    (parsed.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'track_calls', 'AB_optimizer_steps', 'C_optimizer_steps',
        'frozen_before_after_exact', 'saved_model_and_optimizer_roundtrip_exact']}), flush=True)


if __name__ == '__main__':
    main()
