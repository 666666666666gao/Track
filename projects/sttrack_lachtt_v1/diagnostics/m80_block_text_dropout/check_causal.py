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

ROOT = Path('/root/autodl-tmp/sttrack_m80_block_text_dropout_20260908')
assert not (ROOT / 'causal_check.json').exists() and not (ROOT / 'training').exists()
sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT))
from text_dropout import make_contexts
from causal_training import CausalTrainingTracker, base_supervision
from window_competition import supervision, competition_loss
from train_causal import sha, tensor_state_sha
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack import STTrack
from lib.train.dataset.depth_utils import get_rgbd_frame

spec = json.loads((ROOT / 'prepared_training_spec.json').read_text())
assert spec['seed'] == 2027 and spec['support_loss_weights'] == {'category': 0., 'empty': 0.}
assert sha(ROOT / 'train_causal.py') == spec['training_script_sha256']
assert sha(ROOT / 'causal_training.py') == spec['causal_script_sha256']
for n, h in json.loads((ROOT / 'integration.json').read_text())['source_sha256'].items(): assert sha(ROOT / 'code' / n) == h
torch.set_num_threads(1); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
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


# Synthetic contract: same-instance decoded hypotheses must not become negatives.
sample=torch.full((1,1,16,16),.01,device='cuda',requires_grad=True)
with torch.no_grad():
    sample[0,0,12,12]=.8;sample[0,0,8,8]=.6;sample[0,0,7,7]=.9
sizes=torch.full((1,2,16,16),.05,device='cuda',requires_grad=True)
offsets=torch.zeros((1,2,16,16),device='cuda',requires_grad=True)
with torch.no_grad():offsets[0,:,7,7]=5.
synthetic={'score_map':sample,'size_map':sizes,'offset_map':offsets}
target=sample.new_tensor([.7,.7,.1,.1])
rank,diagnostic=competition_loss(synthetic,target,native.output_window)
alternate,_=competition_loss(synthetic,target,torch.ones_like(native.output_window))
assert torch.equal(rank,alternate)
rank.backward()
assert diagnostic['competition_positive_index']==204 and diagnostic['competition_negatives']==9
assert sample.grad[0,0,12,12]<0 and sample.grad[0,0,8,8]>0 and sample.grad[0,0,7,7]==0
assert sizes.grad is None and offsets.grad is None
synthetic_check=dict(loss=float(rank.detach()),positive_gradient=float(sample.grad[0,0,12,12]),false_peak_gradient=float(sample.grad[0,0,8,8]),same_instance_gradient=float(sample.grad[0,0,7,7]),no_regression_mask_gradients=True)
del sample,sizes,offsets,synthetic,rank

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
results = {}; loss_contract_checks = 0
for arm, tracker in trackers.items():
    tracker.initialize(frame(0), dict(infos[arm]))
    contexts=make_contexts(tracker,infos[arm]['empty_text'])
    assert torch.equal(contexts[0][:,~infos[arm]['text_mask']],contexts[1][:,~infos[arm]['text_mask']])
    mask_before=tracker.semantic_context['mask'].clone()
    initial_before=tracker.semantic_context['initial'].clone()
    opt = torch.optim.AdamW(tracker.network.semantic_adapter.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    opt.zero_grad(set_to_none=True); valid = steps = 0; losses = []; last_grads = {}
    for i in range(1, 97):
        tracker.semantic_context['text']=contexts[int(arm=='category' and 33<=i<=64)]
        assert torch.equal(tracker.semantic_context['mask'],mask_before)
        assert torch.equal(tracker.semantic_context['initial'],initial_before)
        out, state = tracker.step(frame(i))
        loss, diagnostic = supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256, output_window=tracker.output_window)
        original, _ = base_supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256)
        if original is None:
            assert loss is None
        elif diagnostic['label']=='centre_inside':
            assert abs(float((loss-original).detach())-diagnostic['competition_loss'])<1e-4
            assert diagnostic['competition_negatives']>0
        else:
            assert torch.equal(loss,original)
        loss_contract_checks += 1; del original
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
assert loss_contract_checks == 192
result = dict(status='completed_M80_native_parity_and_causal_smoke', observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__), prepared_spec_sha256=sha(ROOT / 'prepared_training_spec.json'), seed=2027,
    initial_adapter_state_sha256=initial, base_state_sha256=before, zero_residual_public_state_exact=True,
    zero_residual_frames_each=101, default_template_writes=writes, combined_loss_contract_verified=True,
    combined_loss_checks=loss_contract_checks, base_frozen_all_arms=True, arms=results,
    synthetic_competition_contract=synthetic_check, formal_optimizer_steps=0, smoke_weights_saved=False, training_text_switch_checked=True, independent_model_review_pass=False, elapsed_seconds=time.time()-started)
(ROOT / 'causal_check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
