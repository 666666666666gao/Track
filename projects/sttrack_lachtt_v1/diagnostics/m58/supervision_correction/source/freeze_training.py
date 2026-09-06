from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

root = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory = json.loads((root / 'data_inventory.json').read_text())
text = json.loads((root / 'text_preparation.json').read_text())
smoke = json.loads((root / 'causal_smoke_result.json').read_text())
assert smoke['status'] == 'causal_wiring_and_fit_smoke_complete'
assert smoke['base_parameters_and_buffers_unchanged'] and smoke['native_semantic_step_exact']
assert smoke['causal_source_sha256'] == sha(root / 'causal_training.py')
assert text['sequences'] == {'fit': 130, 'development': 22}
for name, digest in text['file_sha256'].items():
    assert sha(root / name) == digest
fit = [r for r in inventory['sequences_detail'] if r['split'] == 'fit']
development = [r['sequence'] for r in inventory['sequences_detail'] if r['split'] == 'development']
assert len(fit) == 130 and len(development) == 22
random.Random(2026).shuffle(fit)
calls = sum(r['rgb_frames']-1 for r in fit)
assert calls == 186694
baseline = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906/recursive_result.json')
assert sha(baseline) == '5ed4a20beae88a8e8b85513000a8f020311e934b9206678f7d670b50b0737064'
original = json.loads(baseline.read_text())['aggregates']['native']
spec = dict(status='frozen_before_formal_training', observed_utc=datetime.now(timezone.utc).isoformat(),
    revision='native_supervision_round_v2', preceding_training_spec_sha256='42f618b8d3af47b9e5d5cff83e19b113a0bae74417a7046d95ea96626e7dbb53', supervision_coordinate_convention='Native transform_image_to_crop; rounded centre, constrained to the representable 0..15 grid; same-state diagnostic precedes restart', supervision_diagnostic_sha256='896e9a80cfa26443dc66f76bce35e628debbe010f7fc87a72adcf37c95059d68', primary_arm='text', control_arm='visual', seed=2026, epochs=1,
    hypothesis='An initial-instance-bound dense text/RGB-D residual improves native recursive tracking relative to an equal-capacity same-slot empty-text control.',
    architecture='semantic_spatial_v1', learned_parameters=289154, visual_base_frozen_eval=True,
    native_checkpoint=inventory['native_checkpoint'], native_checkpoint_sha256=inventory['native_checkpoint_sha256'],
    integration_sha256=sha(root/'integration.json'), causal_script_sha256=sha(root/'causal_training.py'),
    training_script_sha256=sha(root/'train_causal.py'), freeze_script_sha256=sha(__file__),
    inventory_sha256=sha(root/'data_inventory.json'), text_preparation_sha256=sha(root/'text_preparation.json'),
    text_fit_sha256=sha(root/'text_fit.pt'), text_development_sha256=sha(root/'text_development.pt'),
    initial_checkpoint_sha256=sha(root/'native_parity/text_zero.pth'),
    causal_smoke_result_sha256=sha(root/'causal_smoke_result.json'),
    dataset_root=inventory['dataset_root'], sequence_order=fit, development_sequences=development,
    total_training_image_frames=sum(r['rgb_frames'] for r in fit), total_training_track_calls=calls,
    maximum_optimizer_steps=sum(math.ceil((r['rgb_frames']-1)/32) for r in fit),
    optimizer='AdamW', learning_rate=1e-4, weight_decay=1e-4, gradient_accumulation_frames=32, gradient_clip=1.,
    loss='focal centre heatmap + 2 GIoU + 5 XYXY L1 at the GT centre cell; GT never selects a public box',
    invalid_gt='No loss; advance the predicted crop/query/template state normally.',
    target_centre_outside_crop='All-zero centre heatmap and no box regression; no GT reset or widened crop.',
    state_protocol='Initialize from first RGB-D/box only; chronological full sequence; actual predicted boxes, queries and default template updates feed the next frame. Optimizer updates every 32 observed frames with valid-loss averaging. State tensors are detached.',
    backprop_through_time_or_discrete_crops=False, native_update_interval=50, native_update_threshold=.75,
    caption_protocol='Frozen local Qwen2.5-VL first-frame full image with target mark plus target crop; no sequence/category hint; raw automatic captions retained, including observed errors.',
    text_control='Same slots/mask/initial references/weights/order/budget; replace all valid lexical vectors with the frozen CLIP empty embedding.',
    annotation_limitations=['Single-model automatic captions are not semantic ground truth.',
        'Executor first-frame visual inspection of 24 seed-sampled fit plus all 22 reused development initial crops found errors, including flower-as-insect and package-as-phone.',
        'No manual wording changes or new caption generation selected from tracking outcomes.',
        'Control retains valid-slot structure metadata. One seed. Development22 has been repeatedly used.'],
    frozen_native_development_aggregate=original, native_result_sha256=sha(baseline),
    promotion_gates=dict(text_pooled_mean_vs_native_minimum=.002, text_pooled_mean_vs_visual_minimum=.001,
        text_low_frames_no_more_than_native=True, text_H10_no_more_than_native=True,
        no_new_failure_on_native_zero_H10_sequences=True),
    after_training='Seal both fixed-final-head full development22 predictions before GT metric analysis; then same-head content counterfactuals only if primary gates pass. Low22 follows; no full public dataset until prior gates pass.',
    checkpoint_retention='One latest and one fixed final adapter+optimizer per arm; no base copy or per-epoch history.',
    runtime_estimate_seconds_per_arm=calls*smoke['training_seconds']/smoke['training_frames']*1.15,
    monitoring='First numerical/coverage check after 240 seconds; then near estimated finish, or only when a logged error warrants intervention.',
    independent_review_pass=False, independent_reviewer_status='Astra/max agent quota unavailable; executor checks only, no independent verdict.',
    optimizer_stop_conditions=['Non-finite loss/gradient, source mismatch, runtime exception; preserve terminal evidence.'],
    final_three_dataset_metrics_exist=False)
assert not (root / 'training_spec.json').exists()
(root / 'training_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
print(json.dumps(dict(status=spec['status'], spec_sha256=sha(root/'training_spec.json'),
    track_calls_per_arm=calls, maximum_optimizer_steps=spec['maximum_optimizer_steps'],
    estimated_hours_per_arm=spec['runtime_estimate_seconds_per_arm']/3600), indent=2))
