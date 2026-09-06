from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

root = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v1_20260906')
sys.path.insert(0, str(root / 'code'))
sys.path.insert(0, str(root))
from causal_training import CausalTrainingTracker, supervision
from lib.config.sttrack.config import cfg, update_config_from_file
from lib.test.tracker.sttrack_semantic import STTrackSemantic
from lib.train.dataset.depth_utils import get_rgbd_frame

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
integration = json.loads((root / 'integration.json').read_text())
for path, digest in integration['source_sha256'].items():
    assert sha(root / 'code' / path) == digest
torch.set_num_threads(1)
torch.manual_seed(2026)
torch.cuda.manual_seed_all(2026)
update_config_from_file(str(root / 'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
params = SimpleNamespace(cfg=cfg, checkpoint='/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar',
    base_checkpoint_sha256=integration['native_checkpoint_sha256'], template_factor=2., template_size=128,
    search_factor=4., search_size=256, save_all_boxes=False, debug=0)
checkpoint = str(root / 'native_parity/text_zero.pth')
native = STTrackSemantic(params, checkpoint)
causal = CausalTrainingTracker(params, checkpoint)
inventory = json.loads((root / 'data_inventory.json').read_text())
row = next(r for r in inventory['sequences_detail'] if r['sequence'] == 'chair01_indoor')
assert row['split'] == 'fit'
bank = torch.load('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/text_fit.pt', map_location='cpu')
idx = bank['sequences'].index(row['sequence'])
info = dict(init_bbox=row['first_box'], text_tokens=bank['tokens'][idx],
            text_mask=bank['mask'][idx], empty_text=bank['empty'])
folder = Path(inventory['dataset_root']) / row['sequence']


def frame(i):
    return get_rgbd_frame(str(folder / 'color' / ('%08d.jpg' % (i + 1))),
                          str(folder / 'depth' / ('%08d.png' % (i + 1))), dtype='rgbcolormap', depth_clip=True)


def base_digest():
    digest = hashlib.sha256()
    for name, tensor in causal.network.state_dict().items():
        if not name.startswith('semantic_adapter.'):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


started = time.time()
before = base_digest()
for tracker in [native, causal]:
    tracker.initialize(frame(0), dict(info))
writes = []
for i in range(1, 102):
    image = frame(i)
    expected = native.track(image)
    with torch.no_grad():
        _, actual = causal.step(image)
    assert expected['target_bbox'] == actual['bbox'], (i, 'bbox')
    assert float(expected['best_score']) == actual['best_score'], (i, 'score')
    assert all(torch.equal(a, b) for a, b in zip(native.track_query_before, causal.track_query_before))
    assert all(torch.equal(a, b) for a, b in zip(native.z_dict, causal.z_dict))
    if actual['template_write']:
        writes.append(i)
assert 100 in writes
parity_elapsed = time.time() - started

gt = np.loadtxt(folder / 'groundtruth.txt', delimiter=',').reshape(-1, 4)
assert sha(folder / 'groundtruth.txt') == row['groundtruth_sha256']
causal.initialize(frame(0), dict(info))
optimizer = torch.optim.AdamW(causal.network.semantic_adapter.parameters(), lr=1e-4, weight_decay=1e-4)
losses = []
counts = dict(invalid=0, centre_inside=0, centre_outside=0)
gradient_l1 = {}
training_started = time.time()
optimizer.zero_grad(set_to_none=True)
for i in range(1, 97):
    out, state = causal.step(frame(i))
    loss, diagnostic = supervision(causal.network, out, gt[i], state['previous_bbox'], state['resize_factor'], 256)
    counts[diagnostic['label']] += 1
    if loss is not None:
        (loss / 32).backward()
        losses.append(float(loss.detach()))
    if i % 32 == 0:
        for name, parameter in causal.network.semantic_adapter.named_parameters():
            if parameter.grad is not None:
                assert bool(torch.isfinite(parameter.grad).all()), name
                gradient_l1[name] = float(parameter.grad.abs().sum())
        torch.nn.utils.clip_grad_norm_(causal.network.semantic_adapter.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
assert all(gradient_l1[n + '.0.weight'] > 0 for n in ['rgb', 'depth', 'text'])
assert base_digest() == before
assert all(p.grad is None for n, p in causal.network.named_parameters() if not n.startswith('semantic_adapter.'))
assert not causal.network.training
result = dict(status='causal_wiring_and_fit_smoke_complete', observed_utc=datetime.now(timezone.utc).isoformat(),
    sequence=row['sequence'], split=row['split'], parity_frames=102, parity_template_writes=writes,
    native_semantic_step_exact=True, training_frames=96, actual_dataset_optimizer_steps=3,
    formal_training_optimizer_steps=0, training_labels=counts, mean_training_loss=sum(losses)/len(losses),
    last_gradients_l1=gradient_l1, base_parameters_and_buffers_unchanged=True, base_state_sha256=before,
    production_sources_sha256=sha(root / 'integration.json'), causal_source_sha256=sha(root / 'causal_training.py'),
    checker_sha256=sha(__file__), legacy_text_used_only_for_fit_smoke=True,
    formal_training_has_started=False, smoke_weights_saved=False,
    parity_seconds=parity_elapsed, training_seconds=time.time() - training_started,
    elapsed_seconds=time.time() - started, measured_learned_tracking_performance=False)
(root / 'causal_smoke_result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
