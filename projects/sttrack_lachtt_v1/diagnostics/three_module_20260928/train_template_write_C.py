"""R3 fit120 only, after the complete R2 teacher and its raw-result audit.

No backbone, tracker, image, template, public test or official metric is loaded.
The two arms use every common eligible event and the same initialization/order.
"""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from analyze_train_states import sha
from template_write_C import TemplateWriteC
from template_write_features import INPUT_DIM, CONTRACT

SPLIT_SHA = 'ccffeff5f42df16a544fe402d53c0a8410d200574d24c3b87383040164a92fc6'
SPEC_SHA = '3569d9c42b8db332281ad70bec719650b876e8b9003c48ba73b17d1451e4b425'
P1_SHA = '3fd4087ca0e9f892ce2b2a0647fa1b706bfc7e869099f6c24c129225ede96811'


def read(path):
    return json.loads(path.read_text())


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def common_samples(root, spec, split, audit):
    complete = read(root / 'result.json')
    assert sha(root / 'result.json') == audit['audited_teacher_result_sha256']
    assert (root / 'controller.exit').read_text().strip() == '0'
    assert complete['status'] == 'complete_M122_full_train_template_teacher'
    assert complete['sequences'] == 152 and complete['native_track_calls'] == 219802
    assert complete['inputs']['spec_sha256'] == SPEC_SHA and complete['inputs']['final_sha256'] == P1_SHA
    assert len(spec['sequence_order']) == len(split['sequence_order']) == 152
    assert split['fit_sequences'] == 120 and split['dev_sequences'] == 32
    assert [r['sequence'] for r in spec['sequence_order']] == [r['sequence'] for r in split['sequence_order']]
    groups = {'fit': [], 'development': []}
    excluded = Counter()
    all_rows = []
    source_receipts = []
    for shard in [0, 1]:
        folder = root / ('teacher_shard%d' % shard)
        source = folder / 'replay/result.json'
        assert sha(source) == audit['audited_teacher_shards_sha256'][shard]
        teacher = read(source)
        assert teacher['status'] == 'complete_M122_full_train_template_teacher_shard'
        assert teacher['shard'] == shard and teacher['inputs'] == complete['inputs']
        assert teacher['C_input_contract'] == CONTRACT and teacher['C_input_dim'] == INPUT_DIM
        assert teacher['GT_used_for_labels_only_after_rollouts']
        assert not teacher['current_or_future_GT_used_for_event_selection']
        assert teacher['same_GPU_for_every_pair'] and teacher['original_collection_GPU_reused']
        source_receipts.append(dict(path=str(source), sha256=sha(source)))
        for row in teacher['events']:
            index = row['sequence_index']
            assert index % 2 == shard and spec['sequence_order'][index]['sequence'] == row['sequence']
            assert row['common_C_training_eligible'] == (row['current_GT_valid'] and row['valid_future_frames'] > 0)
            if not row['common_C_training_eligible']:
                excluded['current_GT_unknown' if not row['current_GT_valid'] else 'future_GT_unavailable'] += 1
                all_rows.append(dict(sequence_index=index, sequence=row['sequence'], frame=row['frame'], eligible=False))
                continue
            assert row['current_iou'] is not None and row['delta_future'] is not None
            assert math.isfinite(row['current_iou']) and 0 <= row['current_iou'] <= 1
            assert math.isfinite(row['delta_future']) and -1 <= row['delta_future'] <= 1
            payload_path = folder / row['payload']
            assert payload_path.stat().st_size == row['payload_bytes'] and sha(payload_path) == row['payload_sha256']
            payload = torch.load(payload_path, map_location='cpu')
            features = payload['C_input']
            assert payload['C_input_contract'] == CONTRACT and features.shape == (1, INPUT_DIM)
            assert features.dtype == torch.float32 and bool(torch.isfinite(features).all())
            target_split = split['sequence_order'][index]['C_split']
            assert target_split in groups
            item = dict(sequence_index=index, sequence=row['sequence'], frame=row['frame'],
                C_split=target_split, input_sha256=hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest(),
                current=row['current_iou'], future=row['delta_future'], payload_sha256=row['payload_sha256'])
            groups[target_split].append((item, features[0].clone()))
            all_rows.append(dict(**item, eligible=True))
            del payload
    assert len(all_rows) == complete['events']
    for rows in groups.values():
        rows.sort(key=lambda r: (r[0]['sequence_index'], r[0]['frame']))
        assert rows
        assert len({(r[0]['sequence'], r[0]['frame']) for r in rows}) == len(rows)
    manifest = dict(teacher_result_sha256=sha(root / 'result.json'), teacher_shards=source_receipts,
        inputs=complete['inputs'], events=len(all_rows), excluded=dict(excluded),
        rows=sorted(all_rows, key=lambda r: (r['sequence_index'], r['frame'])))
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return groups, manifest, digest


def main():
    parser = argparse.ArgumentParser()
    for name in ['teacher-root', 'teacher-audit', 'spec', 'split', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=['current', 'future'], required=True)
    parser.add_argument('--physical-gpu', choices=['0', '1'], required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == args.physical_gpu
    assert not args.output.exists()
    audit = read(args.teacher_audit)
    assert audit['review_call_status'] == 'completed' and audit['blocking_count'] == 0
    assert audit['R2_terminal_raw_audit_complete']
    assert sha(args.spec) == SPEC_SHA and sha(args.split) == SPLIT_SHA
    spec, split = read(args.spec), read(args.split)
    groups, manifest, dataset_sha = common_samples(args.teacher_root, spec, split, audit)
    fit_rows, dev_rows = groups['fit'], groups['development']
    x = torch.stack([r[1] for r in fit_rows]).cuda()
    y = x.new_tensor([r[0][args.arm] for r in fit_rows])
    dev_x = torch.stack([r[1] for r in dev_rows]).cuda()
    dev_y = dev_x.new_tensor([r[0][args.arm] for r in dev_rows])
    torch.set_num_threads(1)
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    model = TemplateWriteC()
    initial = state_digest(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    generator = torch.Generator().manual_seed(2027)
    args.output.mkdir()
    (args.output / 'common_dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    steps = 0; visits = hashlib.sha256(); started = time.time(); epochs = []
    with (args.output / 'epoch_log.jsonl').open('w') as log:
        for epoch in range(10):
            model.train(); total = 0.
            order = torch.randperm(len(fit_rows), generator=generator)
            for start in range(0, len(fit_rows), 32):
                indices = order[start:start + 32].cuda()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x[indices])
                loss = F.mse_loss(prediction, y[indices])
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                optimizer.step(); steps += 1
                total += float(loss.detach()) * len(indices)
                visits.update(order[start:start + 32].numpy().tobytes())
            model.eval()
            with torch.no_grad():
                dev_loss = float(F.mse_loss(model(dev_x), dev_y))
            row = dict(epoch=epoch + 1, fit_MSE=total / len(fit_rows), heldout_teacher_MSE=dev_loss,
                optimizer_steps=steps, elapsed_seconds=time.time() - started)
            epochs.append(row); log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
    assert steps == 10 * math.ceil(len(fit_rows) / 32)
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_path = args.output / 'final.pt'; torch.save(final_state, final_path)
    reloaded = torch.load(final_path, map_location='cpu')
    assert reloaded.keys() == final_state.keys() and all(torch.equal(reloaded[k], v) for k, v in final_state.items())
    with torch.no_grad():
        fixed_predictions = model(dev_x).cpu().tolist()
    panel = [dict(**row[0], prediction=prediction,
                  accepted_at_fixed_teacher_state=prediction > (.5 if args.arm == 'current' else 0.))
             for row, prediction in zip(dev_rows, fixed_predictions)]
    panel_path = args.output / 'fixed_teacher_development_predictions.json'
    panel_path.write_text(json.dumps(dict(scope='same original teacher states; not labels for changed C-policy history',
        arm=args.arm, events=panel, excluded_count_scope='all152 teacher events',
        unknown_or_tail_events_excluded=manifest['excluded']), indent=2) + '\n')
    result = dict(status='complete_M122_C_fixed_teacher_fit120', arm=args.arm, seed=2027,
        epochs=10, batch=32, learning_rate=3e-5, weight_decay=.01, architecture='519->128 ReLU->32 ReLU->1 linear',
        parameters=sum(p.numel() for p in model.parameters()), initial_state_sha256=initial,
        final_sha256=sha(final_path), final_roundtrip_exact=True, common_dataset_sha256=dataset_sha,
        common_fit_events=len(fit_rows), common_development_events=len(dev_rows), excluded=manifest['excluded'],
        optimizer_steps=steps, order_sha256=visits.hexdigest(),
        teacher_result_sha256=manifest['teacher_result_sha256'], teacher_audit_sha256=sha(args.teacher_audit),
        split_sha256=SPLIT_SHA, spec_sha256=SPEC_SHA, input_contract=CONTRACT, input_dim=INPUT_DIM,
        source_sha256=sha(Path(__file__)), model_source_sha256=sha(Path(__file__).with_name('template_write_C.py')),
        physical_gpu_environment=args.physical_gpu, cuda_device=torch.cuda.current_device(),
        input_or_label_positive_filter=False, backbone_models_instantiated=False,
        C_development_used_for_optimizer=False, best_checkpoint_selection=False,
        fixed_teacher_development_predictions_sha256=sha(panel_path),
        public_test_CDTB_VOT_used=False, recursive_development_complete=False,
        epochs_log=epochs, elapsed_seconds=time.time() - started)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
