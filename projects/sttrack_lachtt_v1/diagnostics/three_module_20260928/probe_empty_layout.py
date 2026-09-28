"""Replay the failed M99 epoch order; compare one Empty tensor-layout change."""

import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F

from train_ab_visual_control import batch, check_frozen, freeze_unused, load_inputs
from instance_ab_prototype import InstanceCandidatePrototype
from analyze_train_states import sha


def nonfinite_named(values):
    return [name for name, value in values if value is not None and not bool(torch.isfinite(value).all())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path('/root/autodl-tmp')
    panel = load_inputs(root/'sttrack_m90_train_states_20260928',
                        root/'sttrack_m98_train_contexts_20260928',
                        root/'sttrack_m95_initial_origins_20260928', False)['fit']
    empty = torch.load(root/'sttrack_full152_paired_20260925/text_full152.pt', map_location='cpu')['empty'].cuda().float()
    records = []
    for canonical in (False, True):
        random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
        model = InstanceCandidatePrototype().cuda()
        frozen = freeze_unused(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        with torch.no_grad():
            initial = model(batch(panel, torch.arange(64), empty, torch.device('cuda')))
        assert torch.equal(initial['selection_logits'], panel['base_scores'][:64].cuda())
        assert bool((initial['selected_index'] == 0).all())
        indices_list = torch.randperm(len(panel['key']), generator=torch.Generator().manual_seed(2027)).split(64)
        steps, failed, max_semantic, max_phrase, max_scores = 0, None, 0., 0., 0.
        batch_checks = []
        for at, indices in enumerate(indices_list):
            data = batch(panel, indices, empty, torch.device('cuda'))
            original_stride = list(data['text'].stride())
            if canonical:
                data['text'] = data['text'].contiguous()
            model.train(); optimizer.zero_grad(set_to_none=True)
            parameters_before = nonfinite_named(model.named_parameters())
            output = model(data)
            forward_finite = {name:bool(torch.isfinite(output[name]).all()) for name in
                              ('semantic_delta', 'phrase_delta', 'selection_logits', 'visual_selection_logits', 'quality_logits')}
            finite = all(forward_finite.values())
            delta = output['selection_logits'] - output['visual_selection_logits']
            equal = torch.equal(output['selection_logits'], output['visual_selection_logits'])
            checks = dict(batch=at, parameters_nonfinite_before=parameters_before,
                          forward_finite=forward_finite, score_equal=equal,
                          loss_finite=None, gradients_nonfinite=None, parameters_nonfinite_after=None)
            batch_checks.append(checks)
            max_semantic = max(max_semantic, float(output['semantic_delta'].abs().max()))
            max_phrase = max(max_phrase, float(output['phrase_delta'].abs().max()))
            max_scores = max(max_scores, float(delta.abs().max()))
            if parameters_before or not finite or not equal:
                different = torch.nonzero(output['selection_logits'] != output['visual_selection_logits']).cpu().tolist()
                failed = dict(batch=at, optimizer_steps_before_failure=steps, finite=finite,
                              score_equality_assertion_failed=not equal,
                              original_text_stride=original_stride, used_text_stride=list(data['text'].stride()),
                              semantic_delta_max=float(output['semantic_delta'].abs().max()),
                              phrase_delta_max=float(output['phrase_delta'].abs().max()),
                              score_delta_max=float(delta.abs().max()),
                              affected=[dict(key=panel['key'][int(indices[row])], candidate=column,
                                             score=float(output['selection_logits'][row,column]),
                                             visual_score=float(output['visual_selection_logits'][row,column]))
                                        for row,column in different])
                break
            loss = F.binary_cross_entropy_with_logits(output['visual_selection_logits'], panel['iou'][indices].cuda()) + F.binary_cross_entropy_with_logits(output['quality_logits'], panel['iou'][indices].cuda())
            checks['loss_finite'] = bool(torch.isfinite(loss))
            if not checks['loss_finite']:
                failed = dict(batch=at, reason='nonfinite_loss', score_equality_assertion_failed=False)
                break
            loss.backward()
            checks['gradients_nonfinite'] = nonfinite_named((name, p.grad) for name,p in model.named_parameters())
            if checks['gradients_nonfinite']:
                failed = dict(batch=at, reason='nonfinite_gradient', score_equality_assertion_failed=False)
                break
            optimizer.step(); steps += 1
            checks['parameters_nonfinite_after'] = nonfinite_named(model.named_parameters())
            if checks['parameters_nonfinite_after']:
                failed = dict(batch=at, reason='nonfinite_update', score_equality_assertion_failed=False)
                break
        check_frozen(model, frozen)
        records.append(dict(canonical_text_contiguous=canonical, optimizer_steps=steps, failure=failed,
                            batch_checks=batch_checks,
                            maximum_semantic_delta=max_semantic, maximum_phrase_delta=max_phrase,
                            maximum_score_delta=max_scores))
    result = dict(status='complete_Empty_layout_diagnostic_only', seed=2027, batch_size=64,
                  source_sha256=sha(Path(__file__)), prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),
                  records=records, no_checkpoint=True, no_development_or_public_evaluation=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    assert records[0]['failure'] is not None and records[0]['failure']['score_equality_assertion_failed'], 'Original score-equality assertion not reproduced'
    assert records[1]['failure'] is None and records[1]['optimizer_steps'] == 40, 'Layout change did not fix the replay'


if __name__ == '__main__':
    main()
