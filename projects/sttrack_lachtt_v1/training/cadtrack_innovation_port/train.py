"""Separate CADTrack + M125 architecture port; no external-test selection."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


BASELINE_SHA = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'


def sha(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['legacy-code', 'repository', 'checkpoint', 'config', 'clip-weight',
                 'bank', 'labels', 'spec', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=['smoke', 'train'], required=True)
    return parser.parse_args()


def prepare(args):
    sys.path.insert(0, str(args.legacy_code))
    from region_conditioned_target_decoder import RegionConditionedTargetDecoder
    from template_write_C import TemplateWriteC
    from runtime import CADTrackActor, CADTrackABC

    assert not args.output.exists(), 'Use a new output directory.'
    assert sha(args.checkpoint) == BASELINE_SHA
    spec = json.loads(args.spec.read_text())
    bank = torch.load(args.bank, map_location='cpu')
    labels = json.loads(args.labels.read_text())
    assert spec['seed'] == 2027 and len(spec['sequence_order']) == 152
    assert sum(case['rgb_frames'] - 1 for case in spec['sequence_order']) == 219802
    assert {case['sequence'] for case in spec['sequence_order']} == set(bank['sequences'])
    assert len(bank['sequences']) == len(labels['initial']) == 152
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['dataset'] == labels['dataset'] == 'depthtrack'
    assert bank['labels_sha256'] == sha(args.labels)
    assert sha(args.bank) == 'a599e063b79b9458aab9e63ec21b9f4ac9735e420bceeec95275bfd1289603b2'
    assert bank['encoder_sha256'] == sha(args.clip_weight)
    assert torch.cuda.device_count() == 2
    torch.set_num_threads(1)
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    decoder = RegionConditionedTargetDecoder('roi').to('cuda:0')
    writer = TemplateWriteC().to('cuda:0')
    with torch.cuda.device(1):
        actor = CADTrackActor(args.repository, args.checkpoint, args.config,
            args.clip_weight, bank, decoder)
    return spec, bank, decoder, writer, actor, CADTrackABC(actor, writer)


def main():
    args = arguments()
    spec, bank, decoder, writer, actor, tracker = prepare(args)
    from full_dense_tracker import state_digest
    from train_m122_full_causal import load_truth, precision_objective
    from train_m121_dense_target import box_overlap
    from lib.train.dataset.depth_utils import get_rgbd_frame

    frozen = actor.frozen_digest()
    initial = dict(A_B=state_digest(decoder), C=state_digest(writer))
    optimizer = torch.optim.AdamW(list(decoder.parameters()) + list(writer.parameters()),
        lr=3e-5, weight_decay=.01)
    args.output.mkdir(parents=True)
    protocol = dict(format='CADTrack_M125_port_v1', seed=2027,
        baseline_sha256=BASELINE_SHA, initializer='official_epoch20_not_old_epoch31',
        backbone_frozen=True, clip_frozen=True, new_modules='fresh_seed2027',
        region_arm='roi', regional_semantics=False, inherited_full_null_language=True,
        native_template_interval=5, native_template_threshold=.65, C_threshold=.5,
        C_target='selected-frame current GT IoU; not future utility',
        learning_rate=3e-5, weight_decay=.01, gradient_accumulation=32, passes=6,
        final_selection='fixed pass06 after all152 sequences; no external test selection',
        mode=args.mode, independent_code_review='REVIEW_UNAVAILABLE: no agent tool in client catalog',
        inputs={name: sha(getattr(args, name)) for name in
            ['checkpoint', 'config', 'clip_weight', 'bank', 'labels', 'spec']},
        source={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        legacy_source={p.name: sha(p) for p in args.legacy_code.glob('*.py')})
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    started = time.time()
    calls = steps = c_steps = 0
    gradients = []
    order = spec['sequence_order'] if args.mode == 'train' else spec['sequence_order'][:1]
    with (args.output / 'sequences.jsonl').open('w') as log:
        for epoch in range(6 if args.mode == 'train' else 1):
            for index, case in enumerate(order):
                folder = Path(spec['dataset_root']) / case['sequence']
                truth = load_truth(folder, case)

                def image(frame):
                    return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
                        str(folder / 'depth' / ('%08d.png' % (frame + 1))),
                        dtype='rgbcolormap', depth_clip=True)

                tracker.initialize(image(0), case['first_box'], bank['sequences'].index(case['sequence']))
                decoder.train()
                writer.train()
                optimizer.zero_grad(set_to_none=True)
                pending = ab_count = c_count = 0
                loss_sum = 0.
                limit = case['rgb_frames'] if args.mode == 'train' else 65
                for frame in range(1, limit):
                    out, data, record, prediction = tracker.step(image(frame))
                    calls += 1
                    pending += 1
                    if np.isfinite(truth[frame]).all() and (truth[frame, 2:] > 0).all():
                        target = torch.tensor(truth[frame:frame + 1], device='cuda:0', dtype=torch.float32)
                        measured = dict(out, boxes=out['boxes'].float())
                        loss, _, _ = precision_objective(measured, data, target, 1)
                        assert bool(torch.isfinite(loss))
                        (loss / 32).backward()
                        ab_count += 1
                        loss_sum += float(loss.detach())
                        if prediction is not None:
                            target_c = box_overlap(out['selected_box'].detach().float()[:, None], target)[0][:, 0]
                            loss_c = F.mse_loss(prediction, target_c)
                            assert bool(torch.isfinite(loss_c))
                            (loss_c / 32).backward()
                            c_count += 1
                    if pending == 32 or frame == limit - 1:
                        norms = {}
                        for name, model, count in [('A_B', decoder, ab_count), ('C', writer, c_count)]:
                            if count:
                                for parameter in model.parameters():
                                    if parameter.grad is not None:
                                        parameter.grad.mul_(32 / count)
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                                assert bool(torch.isfinite(norm))
                                norms[name] = float(norm)
                        if ab_count:
                            optimizer.step()
                            steps += 1
                            c_steps += int(c_count > 0)
                        if args.mode == 'smoke':
                            gradients.append(norms)
                        optimizer.zero_grad(set_to_none=True)
                        pending = ab_count = c_count = 0
                    del out, data, prediction
                row = dict(epoch=epoch + 1, sequence=case['sequence'], frames=limit - 1,
                    calls=calls, steps=steps, C_steps=c_steps, loss_sum=loss_sum, elapsed=time.time() - started)
                log.write(json.dumps(row) + '\n')
                log.flush()
                print(json.dumps(row), flush=True)
                snapshot = dict(protocol=protocol, A_B=decoder.state_dict(), C=writer.state_dict(),
                    optimizer=optimizer.state_dict(), completed_epoch=epoch + 1, completed_sequences=index + 1,
                    phase='sequence_complete_optimizer_flushed', calls=calls, steps=steps, C_steps=c_steps,
                    RNG=dict(python=random.getstate(), numpy=np.random.get_state(),
                        torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all()))
                torch.save(snapshot, args.output / 'latest.tmp')
                (args.output / 'latest.tmp').replace(args.output / 'latest.pt')
    assert actor.frozen_digest() == frozen
    assert all(p.grad is None for p in actor.native.network.parameters())
    assert all(p.grad is None for p in actor.clip.parameters())
    final = dict(A_B=decoder.state_dict(), C=writer.state_dict(), protocol=protocol)
    torch.save(final, args.output / 'final.pt')
    reloaded = torch.load(args.output / 'final.pt', map_location='cpu')
    for name, model in [('A_B', decoder), ('C', writer)]:
        assert all(torch.equal(value.cpu(), reloaded[name][key]) for key, value in model.state_dict().items())
        assert state_digest(model) != initial[name]
    if args.mode == 'train':
        assert calls == 6 * 219802
    else:
        assert calls == 64 and steps == c_steps == 2
        assert all(g['A_B'] > 0 and g['C'] > 0 for g in gradients)
    result = dict(status='complete_' + args.mode, calls=calls, steps=steps, C_steps=c_steps,
        frozen_before_after_exact=True, module_roundtrip_exact=True, gradient_norms=gradients,
        final_sha256=sha(args.output / 'final.pt'), elapsed=time.time() - started, formal_metrics=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
