"""Region arms: common warm plus identical new parameters, six complete Train152 passes, final six."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from analyze_train_states import sha
from region_conditioned_target_decoder import RegionConditionedTargetDecoder
from full_dense_tracker import FullDenseTracker, state_digest
from m123_joint_full_tracker import JointFullABCTracker
from template_write_C import TemplateWriteC
from train_m121_dense_target import PARAMETER_GROUPS, box_overlap
from train_m122_full_causal import load_truth, precision_objective

WARM_SHA = 'cc71e0b4f42c2e489325627d0a16dcdfc4fa6bcb941908c30bc20c8c3c5828c0'


def initialize_training(args):
    spec = json.loads(args.spec.read_text())
    bank = torch.load(args.bank, map_location='cpu')
    labels = json.loads(args.labels.read_text())
    warm_result = json.loads(args.warm_result.read_text())
    assert spec['seed'] == 2027 and spec['total_training_track_calls'] == 219802
    assert len(spec['sequence_order']) == len(bank['sequences']) == len(labels['initial']) == 152
    assert sum(row['rgb_frames'] - 1 for row in spec['sequence_order']) == 219802
    assert {row['sequence'] for row in spec['sequence_order']} == set(bank['sequences'])
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['dataset'] == labels['dataset'] == 'depthtrack'
    assert sha(args.spec) == '3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425'
    assert bank['labels_sha256'] == sha(args.labels) == '6ffb6e9907fee6a520e31fa78b4d3ff044a46a7daf8ad1aaa42c4d638b8c50c2'
    assert sha(args.bank) == 'a599e063b79b9458aab9e63ec21b9f4ac9735e420bceeec95275bfd1289603b2'
    assert sha(args.checkpoint) == spec['native_checkpoint_sha256'] == 'cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98'
    assert sha(args.clip_weight) == bank['encoder_sha256'] == 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    assert warm_result['status'] == 'complete_M122_current_quality_C_full152'
    assert warm_result['optimizer_steps'] == 400 and warm_result['final_sha256'] == sha(args.warm_final) == WARM_SHA
    assert sha(args.warm_result) == 'b9cbaf752c839afbf5854138c36c1e5cd35f58d024c45a694e56b3164e74435c'
    assert not args.output.exists() and torch.cuda.device_count() == 2
    torch.set_num_threads(1)
    random.seed(2027)
    np.random.seed(2027)
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    warm = torch.load(args.warm_final, map_location='cpu')
    if args.epoch == 1:
        assert args.resume is None
        weights = warm
    else:
        assert args.resume is not None
        weights = torch.load(args.resume, map_location='cpu')
        assert weights['format'] == 'M125_region_fixed_sequence_boundary_v1'
        assert weights['epoch'] == args.epoch - 1 and weights['completed_sequences'] == 152
        assert weights['spec_sha256'] == sha(args.spec)
        assert weights['region_arm'] == args.arm
        assert weights['fixed_budget_passes'] == 6 and weights['fresh_common_warm_start']
    with torch.cuda.device(0):
        decoder = RegionConditionedTargetDecoder(args.arm).cuda()
        if args.epoch == 1:
            decoder.load_common_warm(weights['A_B'])
        else:
            decoder.load_state_dict(weights['A_B'], strict=True)
        writer = TemplateWriteC().cuda()
        writer.load_state_dict(weights['C'], strict=True)
    with torch.cuda.device(1):
        actor = FullDenseTracker(args.repository, args.checkpoint, args.clip_weight, bank, decoder)
    tracker = JointFullABCTracker(actor, writer)
    frozen = actor.frozen_digest()
    initial = dict(A_B=state_digest(decoder), C=state_digest(writer))
    before = {kind: {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
              for kind, model in [('A_B', decoder), ('C', writer)]}
    assert sum(p.numel() for p in decoder.parameters()) == 413517
    assert sum(p.numel() for p in writer.parameters()) == 70721
    optimizer = torch.optim.AdamW([dict(params=list(decoder.parameters())), dict(params=list(writer.parameters()))],
                                 lr=3e-5, weight_decay=.01)
    if args.epoch == 1:
        assert not optimizer.state
        for kind, model in [('A_B', decoder), ('C', writer)]:
            assert all(torch.equal(model.state_dict()[name].cpu(), value) for name, value in warm[kind].items())
        # Start the training RNG after module construction, with the same fresh seed for all new arms.
        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        torch.cuda.manual_seed_all(2027)
        resume = dict(track_calls=0, optimizer_steps=0, C_optimizer_steps=0,
                      supervised_frames=0, C_supervised_frames=0, sequence_records=[])
    else:
        resume = weights
        assert initial == resume['model_state_sha256']
        assert frozen == resume['frozen_state_sha256'] and resume['resume_RNG_saved']
        optimizer.load_state_dict(resume['optimizer'])
        random.setstate(resume['RNG']['python'])
        np.random.set_state(resume['RNG']['numpy'])
        torch.set_rng_state(resume['RNG']['torch_cpu'])
        torch.cuda.set_rng_state_all(resume['RNG']['torch_cuda'])
    return spec, bank, warm, decoder, writer, actor, tracker, frozen, initial, optimizer, resume, before


def main():
    parser = argparse.ArgumentParser()
    for name in ['spec', 'repository', 'checkpoint', 'clip-weight', 'bank', 'labels',
                 'warm-final', 'warm-result', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--arm', choices=['roi', 'language', 'visual'], required=True)
    args = parser.parse_args()
    assert 1 <= args.epoch <= 6
    spec, bank, warm, decoder, writer, actor, tracker, frozen, initial, optimizer, resume, before = initialize_training(args)
    import sys
    sys.path.insert(0, str(args.repository))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    args.output.mkdir()
    started = time.time()
    calls, steps, C_steps = (resume[name] for name in ['track_calls', 'optimizer_steps', 'C_optimizer_steps'])
    supervised, C_supervised = (resume[name] for name in ['supervised_frames', 'C_supervised_frames'])
    records = list(resume['sequence_records'])
    assert calls == sum(row['track_calls'] for row in records)
    assert len(records) == (args.epoch - 1) * 152
    visited = hashlib.sha256()
    with (args.output / 'sequence_log.jsonl').open('w') as log, (args.output / 'sampled_state_trace.jsonl').open('w') as trace:
        for epoch in [args.epoch - 1]:
            for sequence_index, case in enumerate(spec['sequence_order']):
                folder = Path(spec['dataset_root']) / case['sequence']
                truth = load_truth(folder, case)

                def image(frame):
                    return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (frame + 1))),
                                          str(folder / 'depth' / ('%08d.png' % (frame + 1))),
                                          dtype='rgbcolormap', depth_clip=True)

                tracker.initialize(image(0), case['first_box'], bank['sequences'].index(case['sequence']))
                pending = AB_count = C_count = local_supervised = local_C_supervised = writes = 0
                AB_loss_sum = C_loss_sum = 0.
                optimizer.zero_grad(set_to_none=True)
                for frame in range(1, case['rgb_frames']):
                    out, data, state, C_prediction = tracker.step(image(frame))
                    calls += 1
                    pending += 1
                    writes += state['template_write']
                    valid = bool(np.isfinite(truth[frame]).all() and (truth[frame, 2:] > 0).all())
                    if valid:
                        target = torch.tensor(truth[frame:frame + 1], device='cuda:0', dtype=torch.float32)
                        AB_loss, _, _ = precision_objective(out, data, target, 1)
                        assert bool(torch.isfinite(AB_loss))
                        (AB_loss / 32).backward()
                        AB_count += 1
                        supervised += 1
                        local_supervised += 1
                        AB_loss_sum += float(AB_loss.detach())
                        if C_prediction is not None:
                            C_target = box_overlap(out['selected_box'].detach().float()[:, None], target)[0][:, 0].detach()
                            C_loss = F.mse_loss(C_prediction, C_target)
                            assert bool(torch.isfinite(C_loss))
                            (C_loss / 32).backward()
                            C_count += 1
                            C_supervised += 1
                            local_C_supervised += 1
                            C_loss_sum += float(C_loss.detach())
                    visited.update(json.dumps([epoch, case['sequence'], frame, state], separators=(',', ':')).encode())
                    if frame == 1 or frame % 500 == 0 or frame == case['rgb_frames'] - 1:
                        trace.write(json.dumps(dict(epoch=epoch + 1, sequence=case['sequence'], GT_valid_for_loss=valid, **state)) + '\n')
                        trace.flush()
                    if pending == 32 or frame == case['rgb_frames'] - 1:
                        if AB_count:
                            for parameter in decoder.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.mul_(32 / AB_count)
                            if C_count:
                                for parameter in writer.parameters():
                                    if parameter.grad is not None:
                                        parameter.grad.mul_(32 / C_count)
                            for model in [decoder, writer]:
                                assert all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
                            assert bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)))
                            if C_count:
                                assert bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.)))
                            optimizer.step()
                            steps += 1
                            C_steps += int(C_count > 0)
                        optimizer.zero_grad(set_to_none=True)
                        pending = AB_count = C_count = 0
                    del out, data, C_prediction
                record = dict(epoch=epoch + 1, sequence=case['sequence'], sequence_index=sequence_index,
                              sequences_per_epoch=152, track_calls=case['rgb_frames'] - 1,
                              supervised_frames=local_supervised, C_supervised_frames=local_C_supervised,
                              template_writes=writes, total_track_calls=calls, total_optimizer_steps=steps,
                              total_C_optimizer_steps=C_steps, elapsed_seconds=time.time() - started)
                if local_supervised:
                    record['AB_loss_mean'] = AB_loss_sum / local_supervised
                if local_C_supervised:
                    record['C_MSE_mean'] = C_loss_sum / local_C_supervised
                records.append(record)
                log.write(json.dumps(record) + '\n')
                log.flush()
                print(json.dumps(record), flush=True)
                checkpoint = dict(format='M125_region_fixed_sequence_boundary_v1',
                    A_B=decoder.state_dict(), C=writer.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch + 1, completed_sequences=sequence_index + 1,
                    track_calls=calls, optimizer_steps=steps, C_optimizer_steps=C_steps,
                    supervised_frames=supervised, C_supervised_frames=C_supervised, sequence_records=records,
                    spec_sha256=sha(args.spec), frozen_state_sha256=frozen,
                    model_state_sha256=dict(A_B=state_digest(decoder), C=state_digest(writer)),
                    resume_RNG_saved=True, fixed_budget_passes=6, fresh_common_warm_start=True, region_arm=args.arm, RNG=dict(python=random.getstate(),
                    numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all()))
                # Atomic replacement is required for the requested live sequence-boundary handoff.
                torch.save(checkpoint, args.output / 'latest.tmp')
                (args.output / 'latest.tmp').replace(args.output / 'latest.pt')
    assert calls == args.epoch * 219802 and len(records) == args.epoch * 152
    assert actor.frozen_digest() == frozen
    assert all(p.grad is None for p in actor.native.network.parameters())
    assert all(p.grad is None for p in actor.clip.parameters())
    changes = {group: sum(int(value.detach().cpu().ne(before['A_B'][name]).sum())
                         for name, value in decoder.named_parameters() if name.startswith(group + '.'))
               for group in PARAMETER_GROUPS + ['region_encoder', 'region_support', 'region_box_delta', 'region_quality']}
    assert all(changes.values())
    C_changes = sum(int(value.detach().cpu().ne(before['C'][name]).sum()) for name, value in writer.named_parameters())
    assert C_changes > 0 and C_steps > 0
    final = args.output / 'final.pt'
    torch.save(dict(A_B=decoder.state_dict(), C=writer.state_dict(),
                    region_arm=args.arm, format='M125_region_fixed_final_v1'), final)
    reloaded = torch.load(final, map_location='cpu')
    assert reloaded['region_arm'] == args.arm and reloaded['format'] == 'M125_region_fixed_final_v1'
    for kind, model in [('A_B', decoder), ('C', writer)]:
        assert all(torch.equal(value.cpu(), reloaded[kind][name]) for name, value in model.state_dict().items())
    result = dict(status='complete_M125_region_fixed_Train152_pass', seed=2027, epochs=args.epoch,
                  training_sequences=152, sequence_runs=len(records), track_calls=calls,
                  supervised_frames=supervised, C_supervised_frames=C_supervised,
                  optimizer_steps=steps, C_optimizer_steps=C_steps, gradient_accumulation=32,
                  learning_rate=3e-5, weight_decay=.01, parameters=484238,
                  AB_parameters=413517, C_parameters=70721, initial_state_sha256=initial,
                  final_state_sha256=dict(A_B=state_digest(decoder), C=state_digest(writer)),
                  AB_parameter_changes=changes, C_parameter_changes=C_changes,
                  parameter_changes_relative_to_current_pass_start=True, region_arm=args.arm,
                  specific_text_content_used_in_new_stage=args.arm != 'visual',
                  historical_common_warm_text_training=True, additional_region_parameters=33350,
                  warm_final_sha256=WARM_SHA, warm_result_sha256=sha(args.warm_result),
                  final_sha256=sha(final), final_state_roundtrip_exact=True,
                  bank_sha256=sha(args.bank), labels_sha256=sha(args.labels), spec_sha256=sha(args.spec),
                  source_sha256=sha(__file__),
                  region_decoder_source_sha256=sha(Path(__file__).with_name('region_conditioned_target_decoder.py')), tracker_source_sha256=sha(Path(__file__).with_name('m123_joint_full_tracker.py')),
                  resumed_checkpoint_sha256=None if args.epoch == 1 else sha(args.resume), initial_resume_state_sha256=initial,
                  fresh_common_warm_start=True, fixed_budget_passes=6, fixed_final_pass=6,
                  first_pass_optimizer_fresh=True, exact_seeded_training_start=True,
                  no_development_split=True, no_cached_feature_fit=True,
                  backbone_device=1, AB_C_learning_device=0, frozen_before_after_exact=True,
                  GT_reinitializations_after_first_frame=0, GT_used_after_action_for_loss_only=True,
                  state_detached_no_temporal_backprop=True, C_feature_inputs_detached=True,
                  C_target='selected-frame real GT IoU; not future memory utility',
                  fixed_final_after_complete_budget=args.epoch == 6, complete_epoch_checkpoint=True, external_metric_checkpoint_selection=False,
                  no_external_test_cdtb_vot_gradient_training=True, Test50_CDTB80_used_for_checkpoint_selection=False,
                  visited_sha256_current_continuation_segment=visited.hexdigest(),
                  elapsed_seconds=time.time() - started, sequence_records=records)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], track_calls=calls, optimizer_steps=steps,
                          C_optimizer_steps=C_steps, final_sha256=result['final_sha256'])), flush=True)


if __name__ == '__main__':
    main()
