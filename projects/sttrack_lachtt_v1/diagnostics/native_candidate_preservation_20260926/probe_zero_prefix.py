"""Replay the first 64 training calls with the old or zero-weight objective."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


M82 = Path('/root/autodl-tmp/sttrack_full152_paired_20260925/M82')
M89 = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')


def state_sha(network, gradients=False):
    digest = hashlib.sha256()
    if gradients:
        items = ((name, parameter.grad) for name, parameter in network.named_parameters()
                 if name.startswith('semantic_adapter.') and parameter.grad is not None)
    else:
        items = ((name, tensor) for name, tensor in network.state_dict().items()
                 if name.startswith('semantic_adapter.'))
    for name, tensor in items:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('old', 'zero'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    spec = json.loads((M82 / 'training_spec.json').read_text())
    row = spec['sequence_order'][0]
    assert row['sequence'] == 'cube04_indoor' and row['rgb_frames'] > 65
    sys.path.insert(0, str(M82 / 'code'))
    sys.path.insert(0, str(M82))
    sys.path.insert(0, str(M89))
    from causal_training import CausalTrainingTracker
    from native_preservation import supervision
    from native_candidate_supervision import augment_loss
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.train.dataset.depth_utils import get_rgbd_frame

    torch.set_num_threads(1)
    torch.manual_seed(spec['seed'])
    torch.cuda.manual_seed_all(spec['seed'])
    update_config_from_file(str(M82 / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['native_checkpoint'],
        base_checkpoint_sha256=spec['native_checkpoint_sha256'], template_factor=2.,
        template_size=128, search_factor=4., search_size=256,
        save_all_boxes=False, debug=0)
    tracker = CausalTrainingTracker(params, str(M82 / 'native_parity/category_zero.pth'))
    tracker.support_loss_weight = spec['support_loss_weights']['category']
    initial_sha = state_sha(tracker.network)
    bank = torch.load(spec['banks']['fit']['category']['path'], map_location='cpu')
    index = bank['sequences'].index(row['sequence'])
    info = dict(init_bbox=row['first_box'], text_tokens=bank['tokens'][index],
                text_mask=bank['mask'][index], empty_text=bank['empty'])
    folder = Path(spec['dataset_root']) / row['sequence']
    gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)

    def frame(i):
        return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (i + 1))),
            str(folder / 'depth' / ('%08d.png' % (i + 1))),
            dtype='rgbcolormap', depth_clip=True)

    input_digest = hashlib.sha256()
    first = frame(0)
    input_digest.update(first.tobytes())
    tracker.initialize(first, info)
    optimizer = torch.optim.AdamW(tracker.network.semantic_adapter.parameters(),
                                  lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    optimizer.zero_grad(set_to_none=True)
    accumulated = spec['gradient_accumulation_frames']
    assert accumulated == 32
    frame_rows = []
    updates = []
    gradient_frames = 0
    for frame_index in range(1, 65):
        image = frame(frame_index)
        input_digest.update(image.tobytes())
        out, state = tracker.step(image)
        loss, diagnostic = supervision(tracker.network, out, gt[frame_index],
            state['previous_bbox'], state['resize_factor'], 256,
            output_window=tracker.output_window,
            preservation_weight=spec['preservation_weight'])
        if args.mode == 'zero':
            original = loss
            loss, diagnostic = augment_loss(loss, diagnostic, out, gt[frame_index],
                state['previous_bbox'], state['resize_factor'], tracker.output_window, 0)
            assert loss is original
        frame_rows.append(dict(frame_index=frame_index, **state,
                               label=diagnostic['label'],
                               loss=None if loss is None else float(loss.detach())))
        if loss is not None:
            (loss / accumulated).backward()
            gradient_frames += 1
        if frame_index % accumulated == 0:
            assert gradient_frames
            for parameter in tracker.network.semantic_adapter.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulated / gradient_frames)
            before_clip = state_sha(tracker.network, gradients=True)
            norm = torch.nn.utils.clip_grad_norm_(tracker.network.semantic_adapter.parameters(),
                                                  spec['gradient_clip'])
            after_clip = state_sha(tracker.network, gradients=True)
            optimizer.step()
            updates.append(dict(frame_index=frame_index, gradient_sha256=before_clip,
                                clipped_gradient_sha256=after_clip,
                                gradient_norm=float(norm),
                                adapter_sha256=state_sha(tracker.network)))
            optimizer.zero_grad(set_to_none=True)
            gradient_frames = 0
        del out, loss
    assert len(updates) == 2
    report = dict(mode=args.mode, seed=spec['seed'], sequence=row['sequence'],
                  calls=64, training_spec_sha256=hashlib.sha256((M82 / 'training_spec.json').read_bytes()).hexdigest(),
                  input_frames_sha256=input_digest.hexdigest(), initial_adapter_sha256=initial_sha,
                  frames=frame_rows, updates=updates)
    args.output.write_text(json.dumps(report) + '\n')
    print(json.dumps(dict(output=str(args.output), input_frames_sha256=report['input_frames_sha256'],
                          updates=updates)))


if __name__ == '__main__':
    main()
