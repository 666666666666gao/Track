"""R2 only: full Train152 own-history events and same-GPU W/K labels.

Run collection and replay shard N on the same physical GPU. This entry
reuses the frozen, hash-bound R1 context and forks; it never trains C.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_train_states import sha
from full_dense_tracker import state_digest
from run_template_write_pilot import context, valid_gt, iou
from template_write_event_cache import pack_event, unpack_event, cpu_trajectory, cpu_tensor
from template_write_forks import event_from_native_write, write_keep_rollouts, restore_rng
from train_m122_full_causal import load_truth
from template_write_features import CONTRACT, INPUT_DIM, initial_reference, write_features


@torch.no_grad()
def collect(args, spec, bank, actor, image, inputs):
    assert not args.output.exists()
    args.output.mkdir()
    (args.output / 'events').mkdir()
    (args.output / 'references').mkdir()
    frozen = actor.frozen_digest()
    decoder = state_digest(actor.decoder)
    started = time.time()
    rows = []
    calls = 0
    cases = [(index, case) for index, case in enumerate(spec['sequence_order']) if index % 2 == args.shard]
    assert len(cases) == 76
    with (args.output / 'native_prefix.jsonl').open('w') as log:
        for sequence_index, case in cases:
            actor.initialize(image(case, 0), case['first_box'], bank['sequences'].index(case['sequence']))
            reference = initial_reference(actor)
            shared = dict(initial={key: cpu_tensor(value) for key, value in actor.initial.items()},
                words=cpu_tensor(actor.words), word_mask=cpu_tensor(actor.word_mask), empty=cpu_tensor(actor.empty),
                C_initial_reference=cpu_tensor(reference))
            shared_path = args.output / 'references' / ('s%03d.pt' % sequence_index)
            torch.save(shared, shared_path)
            shared_sha = sha(shared_path)
            for frame in range(1, case['rgb_frames']):
                previous_feature = actor.selected_feature.detach().clone() if frame > 1 else None
                templates = list(actor.native.z_dict)
                patch = actor.native.z_patch_arr
                _, _, record = actor.step(image(case, frame))
                calls += 1
                assert record['frame'] == frame
                log.write(json.dumps(dict(sequence=case['sequence'], **record)) + '\n')
                if record['template_write']:
                    assert previous_feature is not None
                    event = event_from_native_write(actor, templates, patch, record)
                    payload = pack_event(event)
                    payload['previous_selected_feature'] = cpu_tensor(previous_feature)
                    payload['C_input'] = cpu_tensor(write_features(actor.selected_feature, previous_feature, reference, record))
                    payload['C_input_contract'] = CONTRACT
                    for key in ['initial', 'words', 'word_mask', 'empty']:
                        del payload['state'][key]
                    path = args.output / 'events' / ('s%03d_f%06d.pt' % (sequence_index, frame))
                    torch.save(payload, path)
                    row = dict(sequence_index=sequence_index, sequence=case['sequence'], frame=frame,
                        future_frames=list(range(frame + 1, min(frame + 33, case['rgb_frames']))),
                        shard=args.shard, payload=path.relative_to(args.output).as_posix(),
                        shared_reference=shared_path.relative_to(args.output).as_posix(),
                        shared_reference_bytes=shared_path.stat().st_size, shared_reference_sha256=shared_sha,
                        payload_bytes=path.stat().st_size, payload_sha256=sha(path))
                    rows.append(row)
                    log.flush()
                    print(json.dumps(row), flush=True)
            log.flush()
            print(json.dumps(dict(sequence_complete=case['sequence'], calls=calls, events=len(rows))), flush=True)
    expected_calls = sum(case['rgb_frames'] - 1 for _, case in cases)
    assert calls == expected_calls
    after = actor.frozen_digest()
    decoder_after = state_digest(actor.decoder)
    assert after == frozen and decoder_after == decoder
    result = dict(status='complete_M122_full_train_template_event_collection_shard', shard=args.shard,
        seed=2027, inputs=inputs, sequence_indices=[index for index, _ in cases], events=rows,
        native_track_calls=calls, expected_track_calls=expected_calls, elapsed_seconds=time.time() - started,
        event_order='all actual native-rule writes in original manifest/frame order within shard',
        current_or_future_GT_used_for_event_selection=False, GT_loaded_after_initialization=False,
        optimizations=0, model_objects_serialized=False, frozen_before_after_exact=True,
        frozen_states_before=frozen, frozen_states_after=after, decoder_before=decoder, decoder_after=decoder_after,
        C_input_contract=CONTRACT, C_input_dim=INPUT_DIM, C_input_dtype='float32',
        physical_gpu_environment=args.physical_gpu, cuda_device=torch.cuda.current_device())
    (args.output / 'teacher_events.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], calls=calls, events=len(rows))), flush=True)


@torch.no_grad()
def replay(args, spec, bank, actor, image, inputs):
    manifest = json.loads((args.output / 'teacher_events.json').read_text())
    assert manifest['status'] == 'complete_M122_full_train_template_event_collection_shard'
    assert manifest['inputs'] == inputs and manifest['shard'] == args.shard
    assert manifest['physical_gpu_environment'] == args.physical_gpu
    prefix = {}
    with (args.output / 'native_prefix.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            sequence = row.pop('sequence')
            prefix[(sequence, row['frame'])] = row
    folder = args.output / 'replay'
    folder.mkdir()
    frozen = actor.frozen_digest()
    decoder = state_digest(actor.decoder)
    started = time.time()
    receipts = []
    future_calls = 0
    for row in manifest['events']:
        case = spec['sequence_order'][row['sequence_index']]
        assert case['sequence'] == row['sequence'] and row['sequence_index'] % 2 == args.shard
        assert row['future_frames'] == list(range(row['frame'] + 1, min(row['frame'] + 33, case['rgb_frames'])))
        payload_path = args.output / row['payload']
        assert payload_path.stat().st_size == row['payload_bytes'] and sha(payload_path) == row['payload_sha256']
        actor.initialize(image(case, 0), case['first_box'], bank['sequences'].index(case['sequence']))
        payload = torch.load(payload_path, map_location='cpu')
        shared_path = args.output / row['shared_reference']
        assert shared_path.stat().st_size == row['shared_reference_bytes'] and sha(shared_path) == row['shared_reference_sha256']
        shared = torch.load(shared_path, map_location='cpu')
        for key in ['initial', 'words', 'word_mask', 'empty']:
            payload['state'][key] = shared[key]
        assert payload['C_input_contract'] == CONTRACT and payload['C_input'].shape == (1, INPUT_DIM)
        event = unpack_event(actor, payload)
        assert event.record == prefix[(case['sequence'], row['frame'])]
        future_images = [image(case, frame) for frame in row['future_frames']]
        restore_rng(payload['rng'])
        written, kept = write_keep_rollouts(event, future_images)
        assert len(written) == len(kept) == len(row['future_frames'])
        for frame, w, k in zip(row['future_frames'], written, kept):
            assert w['record']['frame'] == k['record']['frame'] == frame
            assert w['record'] == prefix[(case['sequence'], frame)]
        future_calls += 2 * len(row['future_frames'])
        raw_path = folder / payload_path.name
        torch.save(dict(write=cpu_trajectory(written), keep=cpu_trajectory(kept)), raw_path)
        # Dataset GT enters only label calculation after both branches have finished.
        truth = load_truth(Path(spec['dataset_root']) / case['sequence'], case)
        current_valid = valid_gt(truth[row['frame']])
        current_iou = iou(event.record['bbox'], truth[row['frame']]) if current_valid else None
        labels = []
        for frame, w, k in zip(row['future_frames'], written, kept):
            valid = valid_gt(truth[frame])
            labels.append(dict(frame=frame, GT_valid=valid,
                write_iou=iou(w['record']['bbox'], truth[frame]) if valid else None,
                keep_iou=iou(k['record']['bbox'], truth[frame]) if valid else None))
        usable = [label for label in labels if label['GT_valid']]
        delta = float(np.mean([label['write_iou'] - label['keep_iou'] for label in usable])) if usable else None
        receipt = dict(**row, current_GT_valid=current_valid, current_iou=current_iou,
            valid_future_frames=len(usable), delta_future=delta, future_GT_labels=labels,
            common_C_training_eligible=current_valid and bool(usable),
            rollout=raw_path.relative_to(args.output).as_posix(), rollout_bytes=raw_path.stat().st_size,
            rollout_sha256=sha(raw_path), native_write_window_record_exact=True,
            negative_zero_and_unknown_events_not_filtered=True)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    after = actor.frozen_digest()
    decoder_after = state_digest(actor.decoder)
    assert after == frozen and decoder_after == decoder
    result = dict(status='complete_M122_full_train_template_teacher_shard', shard=args.shard, seed=2027,
        inputs=inputs, events=receipts, sequence_indices=manifest['sequence_indices'],
        physical_gpu_environment=args.physical_gpu, cuda_device=torch.cuda.current_device(),
        same_GPU_for_every_pair=True, original_collection_GPU_reused=True,
        future_tracking_calls=future_calls, elapsed_seconds=time.time() - started,
        frozen_before_after_exact=True, optimizations=0, GT_reinitializations_after_first_frame=0,
        frozen_states_before=frozen, frozen_states_after=after, decoder_before=decoder, decoder_after=decoder_after,
        C_input_contract=CONTRACT, C_input_dim=INPUT_DIM, C_input_dtype='float32',
        GT_used_for_labels_only_after_rollouts=True, current_or_future_GT_used_for_event_selection=False,
        C_training_or_formal_evaluation=False)
    (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], events=len(receipts), calls=future_calls)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['collect', 'replay'], required=True)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--physical-gpu', choices=['0', '1'], required=True)
    for name in ['spec', 'trained-result', 'repository', 'checkpoint', 'clip-weight', 'bank', 'labels', 'final', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.physical_gpu == str(args.shard)
    spec, bank, actor, image, inputs = context(args)
    if args.phase == 'collect':
        collect(args, spec, bank, actor, image, inputs)
    else:
        replay(args, spec, bank, actor, image, inputs)


if __name__ == '__main__':
    main()
