"""New quality-only final fit; this is not the failed Future-C R4 experiment.

Use all common eligible original-teacher events. Freeze the already trained P1
A+B and save one composite final for all three official datasets.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_template_write_C import common_samples, read, state_digest, P1_SHA, SPEC_SHA, SPLIT_SHA
from template_write_C import TemplateWriteC
from analyze_train_states import sha


def main():
    parser = argparse.ArgumentParser()
    for name in ['teacher-root', 'teacher-audit', 'result-audit', 'spec', 'split', 'base-final', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.teacher_audit)
    assert audit['review_call_status'] == 'completed' and audit['blocking_count'] == 0
    assert audit['R2_terminal_raw_audit_complete']
    result_audit = read(args.result_audit)
    assert result_audit['review_call_status'] == 'completed' and result_audit['blocking_count'] == 0
    assert result_audit['R3_terminal_raw_audit_complete']
    assert sha(args.spec) == SPEC_SHA and sha(args.split) == SPLIT_SHA
    assert sha(args.base_final) == P1_SHA
    groups, manifest, dataset_sha = common_samples(args.teacher_root, read(args.spec), read(args.split), audit)
    rows = sorted(groups['fit'] + groups['development'], key=lambda item: (item[0]['sequence_index'], item[0]['frame']))
    assert len(rows) == 1257 and len(manifest['rows']) == 1297
    x = torch.stack([row[1] for row in rows]).cuda()
    y = x.new_tensor([row[0]['current'] for row in rows])
    torch.set_num_threads(1)
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    model = TemplateWriteC()
    initial_sha = state_digest(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=.01)
    generator = torch.Generator().manual_seed(2027)
    args.output.mkdir()
    (args.output / 'common_dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = time.time(); steps = 0; visits = hashlib.sha256(); logs = []
    with (args.output / 'epochs.jsonl').open('w') as log:
        for epoch in range(10):
            model.train(); total = 0.
            order = torch.randperm(len(rows), generator=generator)
            for start in range(0, len(rows), 32):
                indices = order[start:start + 32].cuda()
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(model(x[indices]), y[indices])
                assert bool(torch.isfinite(loss))
                loss.backward()
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                optimizer.step(); steps += 1
                total += float(loss.detach()) * len(indices)
                visits.update(order[start:start + 32].numpy().tobytes())
            row = dict(epoch=epoch + 1, fit_MSE=total / len(rows), optimizer_steps=steps)
            logs.append(row); log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
    assert steps == 10 * math.ceil(len(rows) / 32) == 400
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    final = args.output / 'final.pt'
    base = torch.load(args.base_final, map_location='cpu')
    torch.save(dict(A_B=base, C=state), final)
    reloaded = torch.load(final, map_location='cpu')
    assert all(torch.equal(reloaded['A_B'][name], value) for name, value in base.items())
    assert all(torch.equal(reloaded['C'][name], value) for name, value in state.items())
    result = dict(status='complete_M122_current_quality_C_full152', arm='current', seed=2027,
        epochs=10, batch=32, learning_rate=3e-5, weight_decay=.01, optimizer_steps=steps,
        final_sha256=sha(final), base_final_sha256=P1_SHA, frozen_base_roundtrip_exact=True,
        C_final_state_sha256=state_digest(model), initial_state_sha256=initial_sha,
        parameters=sum(p.numel() for p in model.parameters()), common_events=len(rows),
        fitted_event_sequences=len({row[0]['sequence'] for row in rows}), teacher_sequences=152,
        teacher_native_track_calls=219802, common_dataset_sha256=dataset_sha, order_sha256=visits.hexdigest(),
        exclusions=manifest['excluded'], teacher_result_sha256=manifest['teacher_result_sha256'],
        teacher_audit_sha256=sha(args.teacher_audit), R3_result_audit_sha256=sha(args.result_audit),
        source_sha256=sha(Path(__file__)), reusable_loader_source_sha256=sha(Path(__file__).with_name('train_template_write_C.py')),
        model_source_sha256=sha(Path(__file__).with_name('template_write_C.py')),
        C_development_used_for_optimizer=True, former32_now_training=True,
        target='current GT IoU at original frozen-P1 native-rule event; not future utility',
        own_C_history_teacher_recollected=False, backbone_models_instantiated=False,
        public_test_CDTB_VOT_used_for_optimization=False, checkpoint_selection='fixed epoch10 final',
        deployment_threshold=.5, elapsed_seconds=time.time() - started, epochs_log=logs)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
