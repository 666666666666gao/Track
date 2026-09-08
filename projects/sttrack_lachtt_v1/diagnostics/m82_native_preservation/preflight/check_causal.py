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

ROOT = Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909')
assert not (ROOT / 'causal_check.json').exists() and not (ROOT / 'training').exists()
sys.path.insert(0, str(ROOT / 'code')); sys.path.insert(0, str(ROOT))
from causal_training import CausalTrainingTracker, base_supervision
from window_competition import supervision as raw_supervision, competition_loss
from native_preservation import supervision, spatial_kl
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
            parity_out, state = tracker.step(image)
            assert parity_out['native_bbox'] == state['bbox']
            assert torch.equal(parity_out['score_map'], parity_out['native_score_map'])
            assert float(spatial_kl(parity_out['score_map'], parity_out['native_score_map'])) == 0.
            assert not parity_out['native_score_map'].requires_grad
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
    opt = torch.optim.AdamW(tracker.network.semantic_adapter.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    opt.zero_grad(set_to_none=True); valid = steps = 0; losses = []; last_grads = {}
    for i in range(1, 97):
        out, state = tracker.step(frame(i))
        loss, diagnostic = supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256, output_window=tracker.output_window)
        original, _ = base_supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256)
        if original is None:
            assert loss is None
        elif diagnostic['label']=='centre_inside':
            assert abs(float((loss-original).detach())-diagnostic['competition_loss']-diagnostic['preservation_weighted'])<1e-4
            assert diagnostic['competition_negatives']>0
        else:
            assert torch.equal(loss,original)
        disabled, _ = supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256, output_window=tracker.output_window, preservation_weight=0.)
        raw, _ = raw_supervision(tracker.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256, output_window=tracker.output_window)
        if raw is not None:
            assert torch.equal(raw, disabled)
            selected = list(tracker.network.semantic_adapter.parameters())[-1]
            a = torch.autograd.grad(raw, selected, retain_graph=True)[0]
            b = torch.autograd.grad(disabled, selected, retain_graph=True)[0]
            repeat = torch.autograd.grad(raw, selected, retain_graph=True)[0]
            output_keys=['score_map','size_map','offset_map'] if diagnostic['label']=='centre_inside' else ['score_map']
            ga=torch.autograd.grad(raw,[out[k] for k in output_keys],retain_graph=True)
            gb=torch.autograd.grad(disabled,[out[k] for k in output_keys],retain_graph=True)
            print(json.dumps(dict(gradient_probe_arm=arm,frame=i,parameter_max_abs_difference=float((a-b).abs().max()),same_raw_repeat_max_abs_difference=float((a-repeat).abs().max()),output_gradient_equal=[torch.equal(x,y) for x,y in zip(ga,gb)])),flush=True)
            assert all(torch.equal(x,y) for x,y in zip(ga,gb))
            assert torch.allclose(a,b,rtol=1e-5,atol=1e-7)
        assert not out['native_score_map'].requires_grad
        loss_contract_checks += 1; del original, raw, disabled
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
    results[arm] = dict(frames=96, supervised_loss_frames=len(losses), optimizer_steps=steps, loss_mean=float(np.mean(losses)), final_smoke_adapter_sha256=tensor_state_sha(tracker.network, True), last_gradient_l1=last_grads)
assert loss_contract_checks == 192

# Independent numerical reference and detached-teacher gradient contract.
student = torch.tensor([[[[.1,.2],[.3,.4]]]], device='cuda', requires_grad=True)
teacher = torch.tensor([[[[.4,.3],[.2,.1]]]], device='cuda', requires_grad=True)
kl = spatial_kl(student, teacher)
expected = sum(t*np.log(t/s) for s,t in zip([.1,.2,.3,.4],[.4,.3,.2,.1]))
assert abs(float(kl)-expected)<1e-6
kl.backward()
assert teacher.grad is None and torch.isfinite(student.grad).all() and student.grad.abs().sum()>0
# Test masks against actual network outputs, with GT introduced only after step.
probe_out, probe_state = trackers['category'].step(frame(97))
assert not probe_out['native_score_map'].requires_grad
for probe_gt, expected_label in [(np.array([np.nan]*4),'invalid'), (np.array([1e6,1e6,10.,10.]),'centre_outside')]:
    value, d = supervision(trackers['category'].network, probe_out, probe_gt, probe_state['previous_bbox'], probe_state['resize_factor'],256,output_window=trackers['category'].output_window)
    assert d['label']==expected_label and not d['native_eligible'] and d['preservation_weighted']==0.
probe_out['native_bbox']=[1e6,1e6,10.,10.]
value,d=supervision(trackers['category'].network,probe_out,probe_state['previous_bbox'],probe_state['previous_bbox'],probe_state['resize_factor'],256,output_window=trackers['category'].output_window)
assert d['label']=='centre_inside' and d['native_iou']==0. and not d['native_eligible'] and d['preservation_weighted']==0.

result = dict(status='completed_M82_native_parity_preservation_and_causal_smoke', observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__), prepared_spec_sha256=sha(ROOT / 'prepared_training_spec.json'), seed=2027,
    initial_adapter_state_sha256=initial, base_state_sha256=before, zero_residual_public_state_exact=True,
    zero_residual_frames_each=101, default_template_writes=writes, combined_loss_contract_verified=True,
    combined_loss_checks=loss_contract_checks, base_frozen_all_arms=True, arms=results,
    synthetic_competition_contract=synthetic_check, spatial_kl_reference_verified=True, detached_teacher_verified=True, excluded_gt_and_wrong_teacher_masks_verified=True, disabled_loss_gradient_parity_frames=sum(v["supervised_loss_frames"] for v in results.values()), gradient_parity_interface='exact at supervised score/size/offset outputs; selected adapter parameter allclose rtol1e-5 atol1e-7; same-raw repeated backward logged for comparison', formal_optimizer_steps=0, smoke_weights_saved=False, independent_model_review_pass=False, elapsed_seconds=time.time()-started)
(ROOT / 'causal_check.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(result, indent=2))
