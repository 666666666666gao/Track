"""Check M73 zero-residual native parity and real causal optimization on fit data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, choices=[2027, 2028], required=True); args = p.parse_args()
ROOT = Path('/root/autodl-tmp/sttrack_m73_paired_lexical_replication_20260907') / ('seed' + str(args.seed))
assert not (ROOT / 'causal_check.json').exists() and not (ROOT / 'training').exists()
sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT))
from causal_training import CausalTrainingTracker, supervision, base_supervision
from train_causal import sha, tensor_state_sha
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack import STTrack
from lib.train.dataset.depth_utils import get_rgbd_frame

spec = json.loads((ROOT / 'prepared_training_spec.json').read_text())
assert spec['seed'] == args.seed and spec['support_loss_weights'] == {'category': 0., 'empty': 0.}
assert sha(ROOT / 'train_causal.py') == spec['training_script_sha256']
assert sha(ROOT / 'causal_training.py') == spec['causal_script_sha256']
for n, h in json.loads((ROOT / 'integration.json').read_text())['source_sha256'].items(): assert sha(ROOT / 'code' / n) == h
torch.set_num_threads(1); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
update_config_from_file(str(ROOT / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params = SimpleNamespace(cfg=cfg, checkpoint=spec['native_checkpoint'], base_checkpoint_sha256=spec['native_checkpoint_sha256'],
    template_factor=2., template_size=128, search_factor=4., search_size=256, save_all_boxes=False, debug=0)
native = STTrack(params)
trackers = {arm: CausalTrainingTracker(params, str(ROOT / 'native_parity' / (arm + '_zero.pth'))) for arm in ['category', 'empty']}
initial = {arm: tensor_state_sha(t.network, adapter_only=True) for arm, t in trackers.items()}
before = {arm: tensor_state_sha(t.network) for arm, t in trackers.items()}
assert len(set(initial.values())) == 1 and len(set(before.values())) == 1
row = next(x for x in spec['sequence_order'] if x['sequence'] == 'chair01_indoor')
folder = Path(spec['dataset_root']) / row['sequence']; infos = {}
for arm in trackers:
    b = spec['banks']['fit'][arm]; assert sha(b['path']) == b['sha256']
    bank = torch.load(b['path'], map_location='cpu'); j = bank['sequences'].index(row['sequence'])
    infos[arm] = dict(init_bbox=row['first_box'], text_tokens=bank['tokens'][j], text_mask=bank['mask'][j], empty_text=bank['empty'])
    t = trackers[arm]; assert t.use_text and t.network.semantic_adapter.null_support
    assert sum(p.numel() for p in t.network.parameters() if p.requires_grad) == 289154
assert torch.equal(infos['category']['text_mask'], infos['empty']['text_mask'])
assert not torch.equal(infos['category']['text_tokens'][0], infos['empty']['text_tokens'][0])


def frame(i):
    return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (i + 1))),
        str(folder / 'depth' / ('%08d.png' % (i + 1))), dtype='rgbcolormap', depth_clip=True)


started = time.time(); native.initialize(frame(0), dict(infos['category']))
for arm, tracker in trackers.items(): tracker.initialize(frame(0), dict(infos[arm]))
writes = {arm: [] for arm in trackers}
with torch.no_grad():
    for i in range(1, 102):
        image = frame(i); baseline = native.track(image)
        for arm, tracker in trackers.items():
            _, state = tracker.step(image)
            assert baseline['target_bbox'] == state['bbox'] and float(baseline['best_score']) == state['best_score'], (arm, i)
            assert all(torch.equal(x, y) for x, y in zip(tracker.track_query_before, native.track_query_before))
            assert len(tracker.track_query_before) == len(native.track_query_before)
            assert len(tracker.z_dict) == len(native.z_dict) and all(torch.equal(x, y) for x, y in zip(tracker.z_dict, native.z_dict))
            if state['template_write']: writes[arm].append(i)
assert writes['category'] == writes['empty'] == [100]
del native
assert sha(folder / 'groundtruth.txt') == row['groundtruth_sha256']
gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)
results = {}; same_loss_checks = 0
for arm, tracker in trackers.items():
    tracker.initialize(frame(0), dict(infos[arm]))
    opt = torch.optim.AdamW(tracker.network.semantic_adapter.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    opt.zero_grad(set_to_none=True); valid = steps = 0; losses = []; last_grads = {}
    for i in range(1, 97):
        out, state = tracker.step(frame(i))
        loss, diagnostic = supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256, support_weight=0.)
        original, _ = base_supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256)
        assert (loss is None and original is None) or (loss is not None and original is not None and torch.equal(loss, original))
        same_loss_checks += 1; del original
        assert 'semantic_features' not in out
        if loss is not None:
            (loss / 32).backward(); valid += 1; losses.append(float(loss.detach()))
        if i % 32 == 0:
            assert valid > 0
            for name, parameter in tracker.network.semantic_adapter.named_parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(32 / valid); assert torch.isfinite(parameter.grad).all()
                    last_grads[name] = float(parameter.grad.abs().sum())
            norm = torch.nn.utils.clip_grad_norm_(tracker.network.semantic_adapter.parameters(), spec['gradient_clip']); assert torch.isfinite(norm)
            opt.step(); steps += 1; opt.zero_grad(set_to_none=True); valid = 0
        del out, loss
    assert steps == 3 and all(last_grads[k + '.0.weight'] > 0 for k in ['rgb', 'depth', 'text'])
    assert tensor_state_sha(tracker.network) == before[arm]
    assert all(p.grad is None for n, p in tracker.network.named_parameters() if not n.startswith('semantic_adapter.'))
    assert sha(ROOT / 'native_parity' / (arm + '_zero.pth')) == spec['initial_checkpoint_sha256'][arm]
    results[arm] = dict(frames=96, optimizer_steps=steps, loss_mean=float(np.mean(losses)), final_smoke_adapter_sha256=tensor_state_sha(tracker.network, True), last_gradient_l1=last_grads)
assert same_loss_checks == 192
result = dict(status='completed_M73_native_parity_and_causal_smoke', observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__), prepared_spec_sha256=sha(ROOT / 'prepared_training_spec.json'), seed=args.seed,
    initial_adapter_state_sha256=initial, base_state_sha256=before, zero_residual_public_state_exact=True,
    zero_residual_frames_each=101, default_template_writes=writes, same_output_original_loss_exact=True,
    same_output_loss_checks=same_loss_checks, base_frozen_all_arms=True, arms=results,
    formal_optimizer_steps=0, smoke_weights_saved=False, independent_model_review_pass=False, elapsed_seconds=time.time()-started)
(ROOT / 'causal_check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
