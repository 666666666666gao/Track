"""R1 only: first twelve native-write events, then same-GPU K/K and W/K.

Not an optimizer or a full teacher collector. Run only after R0 completion
and its result audit. Collection precedes two independent replay workers.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from analyze_train_states import sha
from dense_target_decoder import DenseTargetDecoder
from full_dense_tracker import FullDenseTracker, state_digest
from template_write_event_cache import pack_event, unpack_event, cpu_trajectory
from template_write_forks import event_from_native_write, probe_keep_keep, write_keep_rollouts, restore_rng
from train_m122_full_causal import load_truth

P1_FINAL = '3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811'


def context(args):
    spec = json.loads(args.spec.read_text())
    trained = json.loads(args.trained_result.read_text())
    labels = json.loads(args.labels.read_text())
    bank = torch.load(args.bank, map_location='cpu')
    assert trained['status'] == 'complete_M122_full152_training'
    assert trained['precision_weight'] == 1 and trained['epochs'] == 3 and trained['seed'] == 2027
    assert trained['track_calls'] == 659406 and trained['sequence_runs'] == 456
    assert sha(args.final) == trained['final_sha256'] == P1_FINAL
    assert sha(args.spec) == trained['spec_sha256']
    assert sha(args.bank) == trained['bank_sha256']
    assert sha(args.labels) == trained['labels_sha256'] == bank['labels_sha256']
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['dataset'] == labels['dataset'] == 'depthtrack'
    assert sha(args.checkpoint) == spec['native_checkpoint_sha256']
    assert sha(args.clip_weight) == bank['encoder_sha256']
    assert spec['seed'] == 2027 and spec['total_training_track_calls'] == 219802
    assert len(spec['sequence_order']) == len(bank['sequences']) == 152
    assert {x['sequence'] for x in spec['sequence_order']} == set(bank['sequences'])
    assert bool(bank['mask'][:,0].all())
    torch.set_num_threads(1)
    random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    sys.path.insert(0, str(args.repository))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    model = DenseTargetDecoder().cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(args.final, map_location='cpu'), strict=True)
    actor = FullDenseTracker(args.repository, args.checkpoint, args.clip_weight, bank, model)

    def image(case, frame):
        folder = Path(spec['dataset_root']) / case['sequence']
        return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(frame+1))),
            str(folder/'depth'/('%08d.png'%(frame+1))), dtype='rgbcolormap', depth_clip=True)

    inputs = dict(spec_sha256=sha(args.spec), final_sha256=P1_FINAL, bank_sha256=sha(args.bank),
        labels_sha256=sha(args.labels), native_checkpoint_sha256=sha(args.checkpoint), clip_weight_sha256=sha(args.clip_weight))
    return spec, bank, actor, image, inputs


@torch.no_grad()
def collect(args, spec, bank, actor, image, inputs):
    assert not args.output.exists() and args.shard is None
    args.output.mkdir()
    folder = args.output / 'events';folder.mkdir()
    frozen = actor.frozen_digest();decoder = state_digest(actor.decoder)
    started = time.time();rows = [];calls = 0
    with (args.output/'native_prefix.jsonl').open('w') as log:
        for sequence_index, case in enumerate(spec['sequence_order']):
            actor.initialize(image(case, 0), case['first_box'], bank['sequences'].index(case['sequence']))
            for frame in range(1, case['rgb_frames']):
                templates = list(actor.native.z_dict);patch = actor.native.z_patch_arr
                _, _, record = actor.step(image(case, frame));calls += 1
                assert record['frame'] == frame
                log.write(json.dumps(dict(sequence=case['sequence'], **record))+'\n')
                if record['template_write']:
                    event = event_from_native_write(actor, templates, patch, record)
                    ordinal = len(rows)
                    path = folder / ('event_%02d.pt'%ordinal)
                    torch.save(pack_event(event), path)
                    row = dict(ordinal=ordinal, sequence_index=sequence_index, sequence=case['sequence'],
                        frame=frame, future_frames=list(range(frame+1, min(frame+33, case['rgb_frames']))),
                        replay_shard=ordinal%2, payload=path.relative_to(args.output).as_posix(),
                        payload_bytes=path.stat().st_size, payload_sha256=sha(path))
                    rows.append(row);log.flush();print(json.dumps(row), flush=True)
                    if len(rows) == 12:
                        break
            if len(rows) == 12:
                break
    assert len(rows) == 12
    assert actor.frozen_digest() == frozen and state_digest(actor.decoder) == decoder
    result = dict(status='complete_M122_template_pilot_event_collection_only', seed=2027, events=rows,
        inputs=inputs, native_prefix_track_calls=calls, elapsed_seconds=time.time()-started,
        event_order='first 12 actual native-rule writes in original Train152 manifest/frame order',
        current_or_future_GT_used_for_event_selection=False, GT_loaded_after_initialization=False,
        optimizations=0, K_K_or_W_K_executed=False, model_objects_serialized=False,
        frozen_before_after_exact=True, collection_gpu=torch.cuda.current_device())
    (args.output/'pilot_events.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], events=len(rows), calls=calls)), flush=True)


def valid_gt(box):
    return bool(np.isfinite(box).all() and (box[2:]>0).all())


def iou(box, truth):
    box = np.asarray(box, dtype=np.float64)
    intersection = np.maximum(0., np.minimum(box[:2]+box[2:], truth[:2]+truth[2:])-np.maximum(box[:2],truth[:2])).prod()
    return float(intersection/(box[2:].prod()+truth[2:].prod()-intersection))


@torch.no_grad()
def replay(args, spec, bank, actor, image, inputs):
    assert args.shard in [0,1]
    manifest = json.loads((args.output/'pilot_events.json').read_text())
    assert manifest['status'] == 'complete_M122_template_pilot_event_collection_only'
    assert manifest['inputs'] == inputs and len(manifest['events']) == 12
    rows = [row for row in manifest['events'] if row['replay_shard'] == args.shard]
    assert len(rows) == 6
    folder = args.output / ('replay_shard%d'%args.shard);folder.mkdir()
    frozen = actor.frozen_digest();decoder = state_digest(actor.decoder)
    started = time.time();receipts = [];future_calls = 0
    for row in rows:
        case = spec['sequence_order'][row['sequence_index']]
        assert case['sequence'] == row['sequence']
        assert row['future_frames'] == list(range(row['frame']+1, min(row['frame']+33, case['rgb_frames'])))
        actor.initialize(image(case,0), case['first_box'], bank['sequences'].index(case['sequence']))
        payload = torch.load(args.output/row['payload'], map_location='cpu')
        event = unpack_event(actor, payload)
        assert event.record['frame'] == row['frame']
        future_images = [image(case,frame) for frame in row['future_frames']]
        restore_rng(payload['rng'])
        keep_repeat = probe_keep_keep(event, future_images)
        write_rows, keep_rows = write_keep_rollouts(event, future_images)
        assert len(keep_repeat) == len(write_rows) == len(keep_rows) == len(future_images)
        for expected, written, kept in zip(row['future_frames'],write_rows,keep_rows):
            assert written['record']['frame'] == kept['record']['frame'] == expected
        future_calls += 4*len(future_images)
        raw_path = folder/('event_%02d.pt'%row['ordinal'])
        torch.save(dict(keep_repeat=cpu_trajectory(keep_repeat), write=cpu_trajectory(write_rows),
            keep=cpu_trajectory(keep_rows)), raw_path)
        # Dataset truth is read only after all three prediction records exist.
        truth = load_truth(Path(spec['dataset_root'])/case['sequence'], case)
        current_valid = valid_gt(truth[row['frame']])
        current_iou = iou(event.record['bbox'],truth[row['frame']]) if current_valid else None
        labels = []
        for frame, written, kept in zip(row['future_frames'],write_rows,keep_rows):
            valid = valid_gt(truth[frame])
            labels.append(dict(frame=frame, GT_valid=valid,
                write_iou=iou(written['record']['bbox'],truth[frame]) if valid else None,
                keep_iou=iou(kept['record']['bbox'],truth[frame]) if valid else None))
        usable = [x for x in labels if x['GT_valid']]
        delta = float(np.mean([x['write_iou']-x['keep_iou'] for x in usable])) if usable else None
        record = dict(**row, K_K_per_frame_exact=True, current_GT_valid=current_valid, current_iou=current_iou,
            valid_future_frames=len(usable), delta_future=delta, future_GT_labels=labels,
            common_C_training_eligible=current_valid and bool(usable),
            rollouts=raw_path.relative_to(args.output).as_posix(), rollout_bytes=raw_path.stat().st_size,
            rollout_sha256=sha(raw_path), negative_or_zero_labels_not_filtered=True)
        receipts.append(record);print(json.dumps(record), flush=True)
    assert actor.frozen_digest() == frozen and state_digest(actor.decoder) == decoder
    result = dict(status='complete_M122_template_fork_pilot_shard', shard=args.shard, seed=2027,
        inputs=inputs, events=receipts, same_GPU_for_every_pair=True, cuda_device=torch.cuda.current_device(),
        future_tracking_calls=future_calls, elapsed_seconds=time.time()-started, frozen_before_after_exact=True,
        optimizations=0, GT_reinitializations_after_first_frame=0, GT_used_for_labels_only_after_rollouts=True,
        current_or_future_GT_used_for_event_selection=False, full_teacher_or_C_training=False)
    (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], shard=args.shard, events=len(receipts))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['collect','replay'], required=True)
    parser.add_argument('--shard', type=int, choices=[0,1])
    for name in ['spec','trained-result','repository','checkpoint','clip-weight','bank','labels','final','output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    spec, bank, actor, image, inputs = context(args)
    if args.phase == 'collect':
        collect(args, spec, bank, actor, image, inputs)
    else:
        replay(args, spec, bank, actor, image, inputs)


if __name__ == '__main__':
    main()
