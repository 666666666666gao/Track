"""Freeze the matched fine-tuning budget before either retained training run."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root
p = json.loads((root / 'preparation.json').read_text())
cases = json.loads(Path(p['fitting_manifest']).read_text())
fit = sorted({x['sequence'] for x in cases if x['split'] == 'fit'})
dev = sorted({x['sequence'] for x in cases if x['split'] == 'development'})
assert len(fit) == 63 and len(dev) == 22 and not (set(fit) & set(dev))
assert not (root / 'training_spec.json').exists()
spec = dict(
    experiment='M55 TSG clone paired bounded DepthTrack fine-tuning',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    preparation_sha256=sha(root / 'preparation.json'),
    trainer_sha256=sha(root / 'train.py'),
    checkpoint=p['checkpoint'], checkpoint_sha256=p['checkpoint_sha256'],
    fitting_manifest=p['fitting_manifest'], fitting_manifest_sha256=p['fitting_manifest_sha256'],
    dataset_root=p['dataset_root'], fit_sequences=fit, development_sequences=dev,
    fit_gt_sha256={x: sha(Path(p['dataset_root']) / x / 'groundtruth.txt') for x in fit},
    model_seed=2026, data_seed=55000000, epochs=15, samples_per_epoch=2048,
    clips_per_arm=30720, search_frames_per_arm=122880, microbatch_size=2,
    gradient_accumulation=4, effective_batch_size=8, optimizer_steps_per_arm=3840,
    workers=4, mean=[.485, .456, .406, .449, .449, .449],
    std=[.229, .224, .225, .226, .226, .226], optimizer='AdamW',
    learning_rate=1e-4, backbone_multiplier=.1, weight_decay=1e-4,
    gradient_clip_norm=.1, lr_drop_after_epoch=10, lr_decay=.1,
    keep_rate_epochs_1_to_9=[1.], keep_rate_epochs_10_to_15=[1., .85, .7],
    keep_rate_implementation='Captured adjust_keep_rate; values retain its float32 rounding.',
    loss='Sum over four search frames of 2*GIoU + 5*L1 + focal; average four microbatches before clipping and stepping.',
    sampler='Original causal-with-replacement sampler, no sorting/deduplication; native ViPTProcessing and augmentation. Explicit invalid-crop resampling retained; broad exception swallowing omitted.',
    predicted_crop_training=False, template_update_training=False,
    initialization='Strict load of all base state_dict tensors; same model initialization and indexed sample seeds.',
    only_arm_model_difference='temp_x_flip=temp_x.clone() and temp_r_flip=temp_r.clone() in isolated clone model source.',
    batchnorm='Training BatchNorm uses microbatch 2. Accumulation does not reproduce physical batch-8 BatchNorm; identical in both arms.',
    budget_scope='Bounded fine-tuning from existing weights, not the original 15*15000 clips budget or training from scratch.',
    final_weight_selection='Only epoch 15; no development best-epoch selection or loss-based promotion.',
    termination='Execution/data error or nonfinite loss/gradient ends the run with a nonzero exit; no automatic relaunch, early stopping or fallback.',
    temporary_checkpoint_root='/dev/shm/sttrack_m55_training_checkpoints_20260906',
    checkpoint_policy='One rolling optimizer checkpoint per arm in shared memory, volatile across reboot. Final model-only weights and JSONL evidence on /root/autodl-tmp, then locally archived. No old artifacts deleted.',
    contract={'epochs': [1, 10], 'clips_per_epoch': 8, 'optimizer_steps': 2, 'weights_retained': False},
    recursive_validation={
        'split': 'Fixed 22 development sequences within DepthTrack Train, all with prior development use; full causal rollout of both final arms and independent native default on the same GT-valid mask.',
        'primary_metric': 'valid-frame pooled mean IoU',
        'low_overlap': 'IoU <= 0.1 on valid GT frames',
        'H10': 'At least 10 consecutive low-overlap frames; use the bound existing evaluator and disclose invalid-frame handling.',
        'clone_required_gain_vs_native': .005,
        'clone_required_gain_vs_control': .002,
        'secondary_requirements': [
            'low-overlap frames no greater than either native or control',
            'H10 episodes no greater than either native or control',
            'no new H10 episode on a sequence with native H10=0'],
        'acceptance_requires_all': True, 'report_all_22': True},
    public_progression='After paired recursive gate: freeze low22 evaluation; after clear low22 improvement with protection, evaluate one identical weight/runtime bundle on DepthTrack Test, CDTB and VOT full127. Language, association and reader off in this primary pair.',
    timing_estimate_hours_per_arm=[4.5, 7.], long_poll_seconds=240)
(root / 'training_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
print(json.dumps({'spec_sha256': sha(root / 'training_spec.json'), 'fit': len(fit), 'development': len(dev), 'optimizer_steps_per_arm': 3840}))
