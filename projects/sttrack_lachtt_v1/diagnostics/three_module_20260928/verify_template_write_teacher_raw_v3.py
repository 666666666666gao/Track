"""CPU readback of a terminal R2 teacher; never creates an audit certificate.

This checks stored outputs against dataset GT and the bound human bank. It
does not import tracker/decoder code, construct models, run NN or change data.
The independent reviewer still decides the terminal raw-audit verdict.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


SPEC_SHA = '3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425'
GATE_SHA = '26f555c0d1eb13ddac5469d333926780c2fc760a1e45866e51a497533874fbd7'
SPLIT_SHA = 'ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6'
INPUTS = dict(spec_sha256=SPEC_SHA,
    final_sha256='3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811',
    bank_sha256='a599e063b79b9458aab9e63ec21b9f4ac9735e420bceeec95275bfd1289603b2',
    labels_sha256='6ffb6e9907fee6a520e31fa78b4d3ff044a46a7daf8ad1aaa42c4d638b8c50c2',
    native_checkpoint_sha256='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98',
    clip_weight_sha256='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836')
CONTRACT = 'M122-C-current128-past128-initial4x64-quality3-motion4-v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def valid(box):
    return bool(all(math.isfinite(float(x)) for x in box) and box[2] > 0 and box[3] > 0)


def overlap(box, target):
    x, y, w, h = map(float, box)
    a, b, c, d = map(float, target)
    intersection = max(0., min(x + w, a + c) - max(x, a)) * max(0., min(y + h, b + d) - max(y, b))
    return intersection / (w * h + c * d - intersection)


def equal_number(actual, reported):
    if actual is None:
        assert reported is None
    else:
        assert math.isfinite(reported) and math.isclose(actual, reported, rel_tol=0., abs_tol=1e-12)


def main():
    parser = argparse.ArgumentParser()
    for name in ['teacher-root', 'spec', 'bank', 'labels', 'split', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.teacher_root
    assert sys.flags.optimize == 0 and sys.version_info[:2] == (3, 8)
    assert not args.output.exists()
    assert (root / 'controller.exit').read_text().strip() == '0'
    aggregate = read(root / 'result.json')
    assert aggregate['status'] == 'complete_M122_full_train_template_teacher'
    assert aggregate['sequences'] == 152 and aggregate['native_track_calls'] == 219802
    assert aggregate['inputs'] == INPUTS and aggregate['optimizations'] == 0
    assert aggregate['C_training_or_formal_evaluation'] is False
    assert sha(root / 'gate.json') == GATE_SHA
    gate = read(root / 'gate.json')
    assert len(gate['bound_files']) == 77
    bound = []
    for row in gate['bound_files']:
        path = Path(row['path'])
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256']
        bound.append(dict(row))
    assert sha(args.spec) == SPEC_SHA and sha(args.bank) == INPUTS['bank_sha256']
    assert sha(args.labels) == INPUTS['labels_sha256'] and sha(args.split) == SPLIT_SHA
    spec, labels, split = read(args.spec), read(args.labels), read(args.split)
    assert len(spec['sequence_order']) == len(split['sequence_order']) == 152
    assert [x['sequence'] for x in split['sequence_order']] == [x['sequence'] for x in spec['sequence_order']]
    assert split['fit_sequences'] == 120 and split['dev_sequences'] == 32
    assert labels['human_confirmed'] and labels['dataset'] == 'depthtrack'

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import torch
    torch.set_num_threads(1)
    assert torch.__version__.split('+')[0] == '1.13.1' and not torch.cuda.is_initialized()
    bank = torch.load(args.bank, map_location='cpu')
    assert bank['human_confirmed'] and bank['dataset'] == 'depthtrack'
    assert bank['labels_sha256'] == INPUTS['labels_sha256'] and bank['encoder_sha256'] == INPUTS['clip_weight_sha256']
    assert bank['sequences'] == [x['sequence'] for x in labels['initial']]
    assert len(bank['sequences']) == 152 and set(bank['sequences']) == {x['sequence'] for x in spec['sequence_order']}
    assert tuple(bank['tokens'].shape) == (152, 5, 768) and tuple(bank['mask'].shape) == (152, 5)
    assert bank['mask'].dtype == torch.bool and bool(bank['mask'][:, 0].all())
    for index, label in enumerate(labels['initial']):
        assert bank['mask'][index].tolist() == [slot < len(label['phrases']) for slot in range(5)]
        assert not bool(bank['tokens'][index, len(label['phrases']):].any())

    def tensor_check(value):
        if torch.is_tensor(value):
            assert value.device.type == 'cpu' and not value.requires_grad
            assert bool(torch.isfinite(value).all())
        elif isinstance(value, dict):
            for item in value.values():
                tensor_check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensor_check(item)

    tensor_check(bank)
    started = time.time()
    hashes = []
    def checked_file(path, expected_bytes=None, expected_sha=None):
        digest = sha(path)
        if expected_bytes is not None:
            assert path.stat().st_size == expected_bytes
        if expected_sha is not None:
            assert digest == expected_sha
        hashes.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))
        return digest

    aggregate_sha = checked_file(root / 'result.json')
    teacher_shas, prefix_shas, details, gt_details = [], [], [], []
    counts = Counter()
    split_counts = {'fit': Counter(), 'development': Counter()}
    all_indices = []
    for shard in [0, 1]:
        folder = root / ('teacher_shard%d' % shard)
        assert (root / ('collect%d.exit' % shard)).read_text().strip() == '0'
        assert (root / ('replay%d.exit' % shard)).read_text().strip() == '0'
        collection = read(folder / 'teacher_events.json')
        teacher = read(folder / 'replay/result.json')
        checked_file(folder / 'teacher_events.json')
        teacher_shas.append(checked_file(folder / 'replay/result.json'))
        prefix_shas.append(checked_file(folder / 'native_prefix.jsonl'))
        assert collection['status'] == 'complete_M122_full_train_template_event_collection_shard'
        assert teacher['status'] == 'complete_M122_full_train_template_teacher_shard'
        assert collection['inputs'] == teacher['inputs'] == INPUTS
        assert collection['shard'] == teacher['shard'] == shard
        assert collection['physical_gpu_environment'] == teacher['physical_gpu_environment'] == str(shard)
        assert teacher['same_GPU_for_every_pair'] and teacher['original_collection_GPU_reused']
        assert collection['seed'] == teacher['seed'] == 2027
        assert not collection['current_or_future_GT_used_for_event_selection']
        assert not teacher['current_or_future_GT_used_for_event_selection']
        assert collection['GT_loaded_after_initialization'] is False and teacher['GT_used_for_labels_only_after_rollouts']
        assert collection['frozen_before_after_exact'] and teacher['frozen_before_after_exact']
        assert collection['frozen_states_before'] == collection['frozen_states_after'] == teacher['frozen_states_before'] == teacher['frozen_states_after']
        assert collection['decoder_before'] == collection['decoder_after'] == teacher['decoder_before'] == teacher['decoder_after']
        assert collection['optimizations'] == teacher['optimizations'] == 0
        assert collection['C_input_contract'] == teacher['C_input_contract'] == CONTRACT
        assert collection['C_input_dim'] == teacher['C_input_dim'] == 519
        indices = list(range(shard, 152, 2))
        assert collection['sequence_indices'] == teacher['sequence_indices'] == indices
        all_indices.extend(indices)
        assert len(collection['events']) == len(teacher['events'])
        event_keys = [(r['sequence_index'], r['frame']) for r in collection['events']]
        assert len(set(event_keys)) == len(event_keys) and event_keys == sorted(event_keys)
        by_sequence = {index: [] for index in indices}
        for original, row in zip(collection['events'], teacher['events']):
            assert all(row[key] == value for key, value in original.items())
            by_sequence[row['sequence_index']].append(row)
        shard_calls = shard_future_calls = 0
        observed_events = []
        with (folder / 'native_prefix.jsonl').open() as stream:
            for index in indices:
                case = spec['sequence_order'][index]
                prefix = {}
                previous = case['first_box']
                for frame in range(1, case['rgb_frames']):
                    record = json.loads(next(stream))
                    assert record.pop('sequence') == case['sequence'] and record['frame'] == frame
                    assert record['previous_bbox'] == previous and valid(record['bbox'])
                    expected_write = frame % 50 == 0 and record['native_same_position_response'] > .75
                    assert record['template_write'] == expected_write
                    if expected_write:
                        observed_events.append((index, frame))
                    prefix[frame] = record
                    previous = record['bbox']
                    shard_calls += 1
                gt_path = Path(spec['dataset_root']) / case['sequence'] / 'groundtruth.txt'
                checked_file(gt_path, expected_sha=case['groundtruth_sha256'])
                truth = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)
                original_rows = len(truth)
                assert original_rows == case['groundtruth_rows']
                if case['sequence'] == 'toy07_indoor_320':
                    assert len(truth) == 1406 and case['rgb_frames'] == 1367
                    truth = truth[:1367]
                assert len(truth) == case['rgb_frames'] == case['depth_frames']
                assert np.array_equal(truth[0], np.asarray(case['first_box']))
                gt_details.append(dict(sequence_index=index, sequence=case['sequence'],
                    raw_GT_rows=original_rows, used_GT_rows=len(truth), sha256=case['groundtruth_sha256'],
                    legal_first_box=truth[0].tolist(), events=len(by_sequence[index])))
                rows = by_sequence[index]
                shared_path = folder / 'references' / ('s%03d.pt' % index)
                checked_file(shared_path)
                shared = torch.load(shared_path, map_location='cpu')
                tensor_check(shared)
                bi = bank['sequences'].index(case['sequence'])
                assert torch.equal(shared['words'], bank['tokens'][bi:bi + 1].float())
                assert torch.equal(shared['word_mask'], bank['mask'][bi:bi + 1])
                assert torch.equal(shared['empty'], bank['empty'].float())
                assert torch.equal(shared['initial']['initial_box'], torch.tensor([case['first_box']], dtype=torch.float32))
                assert tuple(shared['C_initial_reference'].shape) == (1, 256)
                if not rows:
                    counts['sequences_without_write_events'] += 1
                    continue
                assert folder / rows[0]['shared_reference'] == shared_path
                assert shared_path.stat().st_size == rows[0]['shared_reference_bytes']
                assert hashes[-1]['sha256'] == rows[0]['shared_reference_sha256']
                for row in rows:
                    frame = row['frame']
                    assert row['sequence'] == case['sequence'] and row['shard'] == shard
                    assert row['shared_reference'] == rows[0]['shared_reference']
                    assert row['shared_reference_sha256'] == rows[0]['shared_reference_sha256']
                    assert row['shared_reference_bytes'] == rows[0]['shared_reference_bytes']
                    future_frames = list(range(frame + 1, min(frame + 33, case['rgb_frames'])))
                    assert row['future_frames'] == future_frames
                    payload_path, rollout_path = folder / row['payload'], folder / row['rollout']
                    checked_file(payload_path, row['payload_bytes'], row['payload_sha256'])
                    checked_file(rollout_path, row['rollout_bytes'], row['rollout_sha256'])
                    payload = torch.load(payload_path, map_location='cpu')
                    rollout = torch.load(rollout_path, map_location='cpu')
                    tensor_check(payload); tensor_check(rollout)
                    assert payload['record'] == prefix[frame] and payload['record']['template_write']
                    assert payload['state']['bbox'] == payload['record']['bbox'] and payload['state']['frame'] == frame
                    assert len(payload['state']['templates']) == 2
                    assert all(key not in payload['state'] for key in ['initial', 'words', 'word_mask', 'empty'])
                    features = payload['C_input']
                    assert payload['C_input_contract'] == CONTRACT and tuple(features.shape) == (1, 519)
                    assert features.dtype == torch.float32
                    assert torch.equal(features[:, :128], payload['state']['selected_feature'].float())
                    assert torch.equal(features[:, 128:256], payload['previous_selected_feature'].float())
                    assert torch.equal(features[:, 256:512], shared['C_initial_reference'].float())
                    record = payload['record']
                    quality = torch.tensor([[record['selected_quality'], record['observation_intersection_probability'], record['native_same_position_response']]], dtype=torch.float32)
                    assert torch.equal(features[:, 512:515], quality)
                    before = torch.tensor(record['previous_bbox'], dtype=torch.float32)
                    now = torch.tensor(record['bbox'], dtype=torch.float32)
                    motion = torch.cat((((now[:2] + .5 * now[2:]) - (before[:2] + .5 * before[2:])) / before[2:], (now[2:] / before[2:]).log()))[None]
                    assert torch.allclose(features[:, 515:], motion, rtol=1e-6, atol=1e-6)
                    current_valid = valid(truth[frame])
                    current_iou = overlap(record['bbox'], truth[frame]) if current_valid else None
                    assert row['current_GT_valid'] == current_valid
                    equal_number(current_iou, row['current_iou'])
                    assert len(rollout['write']) == len(rollout['keep']) == len(row['future_GT_labels']) == len(future_frames)
                    deltas = []
                    previous_write = previous_keep = record['bbox']
                    for f, w, k, label in zip(future_frames, rollout['write'], rollout['keep'], row['future_GT_labels']):
                        assert w['record']['frame'] == k['record']['frame'] == label['frame'] == f
                        assert w['record'] == prefix[f]
                        assert valid(k['record']['bbox'])
                        assert not w['record']['template_write'] and not k['record']['template_write']
                        assert w['record']['previous_bbox'] == previous_write and k['record']['previous_bbox'] == previous_keep
                        previous_write, previous_keep = w['record']['bbox'], k['record']['bbox']
                        assert isinstance(w['query'], list) and isinstance(k['query'], list)
                        assert tuple(w['selected_feature'].shape) == tuple(k['selected_feature'].shape) == (1, 128)
                        future_valid = valid(truth[f])
                        wi = overlap(w['record']['bbox'], truth[f]) if future_valid else None
                        ki = overlap(k['record']['bbox'], truth[f]) if future_valid else None
                        assert label['GT_valid'] == future_valid
                        equal_number(wi, label['write_iou']); equal_number(ki, label['keep_iou'])
                        counts['future_rows'] += 1
                        if future_valid:
                            deltas.append(wi - ki)
                            counts['future_' + ('positive' if wi > ki else 'negative' if wi < ki else 'zero')] += 1
                        else:
                            counts['future_GT_unknown'] += 1
                    delta = float(np.mean(deltas)) if deltas else None
                    equal_number(delta, row['delta_future'])
                    eligible = current_valid and bool(deltas)
                    assert row['valid_future_frames'] == len(deltas) and row['common_C_training_eligible'] == eligible
                    assert row['negative_zero_and_unknown_events_not_filtered'] and row['native_write_window_record_exact']
                    kind = 'unknown' if delta is None else 'positive' if delta > 0 else 'negative' if delta < 0 else 'zero'
                    target_split = split['sequence_order'][index]['C_split']
                    assert target_split in split_counts
                    counts['events'] += 1; counts['event_' + kind] += 1
                    counts['eligible' if eligible else 'excluded'] += 1
                    counts['current_GT_unknown'] += int(not current_valid)
                    counts['future_GT_unavailable'] += int(not deltas)
                    counts['short_tail_events'] += int(len(future_frames) < 32)
                    split_counts[target_split]['event_' + kind] += 1
                    split_counts[target_split]['eligible' if eligible else 'excluded'] += 1
                    details.append(dict(sequence_index=index, sequence=case['sequence'], frame=frame,
                        C_split=target_split, current_GT_valid=current_valid, current_iou=current_iou,
                        valid_future_frames=len(deltas), delta_future=delta, common_C_training_eligible=eligible,
                        label_kind=kind, input_sha256=hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest(),
                        payload_sha256=row['payload_sha256'], rollout_sha256=row['rollout_sha256']))
                    shard_future_calls += 2 * len(future_frames)
                    del payload, rollout
            assert next(stream, None) is None
        assert observed_events == event_keys
        assert shard_calls == collection['native_track_calls'] == collection['expected_track_calls']
        assert shard_future_calls == teacher['future_tracking_calls']
        counts['native_track_calls'] += shard_calls
        counts['future_tracking_calls'] += shard_future_calls
    assert sorted(all_indices) == list(range(152))
    assert counts['events'] == aggregate['events'] and counts['native_track_calls'] == 219802
    assert counts['future_tracking_calls'] == aggregate['future_tracking_calls']
    assert sha(root / 'result.json') == aggregate_sha
    assert not torch.cuda.is_initialized()
    result = dict(status='complete_M122_R2_terminal_teacher_CPU_readback',
        recorded_at=datetime.now(timezone.utc).isoformat(), source_sha256=sha(Path(__file__)),
        teacher_result_sha256=aggregate_sha, teacher_shards_sha256=teacher_shas,
        teacher_prefixes_sha256=prefix_shas, gate_sha256=GATE_SHA, inputs=INPUTS,
        spec_sha256=SPEC_SHA, split_sha256=SPLIT_SHA, counts=dict(counts),
        split_counts={key: dict(value) for key, value in split_counts.items()},
        events=details, GT=gt_details, verified_bound_files=bound, raw_files=hashes,
        python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
        NN_calls=0, NN_constructors=0, optimizer_steps=0, experiment_module_imports=0,
        CUDA_initialized=False, NN_progress_queries=0, elapsed_seconds=time.time() - started,
        overlap_comparison_absolute_tolerance=1e-12, motion_CPU_GPU_rtol=1e-6, motion_CPU_GPU_atol=1e-6,
        human_review_recreated=False, learned_C_training_complete=False, final_nine_metrics_complete=False,
        independent_terminal_audit_certified=False,
        limits=['Readback of saved records/tensors; no RGB-D/CLIP/model forward reconstruction.',
            'Native prefix saves records but no full per-frame query/features, so cached previous feature history is not independently reconstructed.',
            'Labels are recomputed only for original teacher states; they do not label changed C-policy histories.',
            'A fresh reviewer must separately issue the terminal raw-audit verdict and required audited digests.'])
    args.output.mkdir()
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], counts=result['counts'], split_counts=result['split_counts'],
        teacher_result_sha256=aggregate_sha, teacher_shards_sha256=teacher_shas, teacher_prefixes_sha256=prefix_shas,
        CUDA_initialized=False, NN_calls=0, independent_terminal_audit_certified=False)), flush=True)


if __name__ == '__main__':
    main()
