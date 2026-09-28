"""Regression replay for M99's actual expanded-Empty equality failure."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path('/root/autodl-tmp')
    fit = load_inputs(root/'sttrack_m90_train_states_20260928',
                      root/'sttrack_m98_train_contexts_20260928',
                      root/'sttrack_m95_initial_origins_20260928', False)['fit']
    empty = torch.load(root/'sttrack_full152_paired_20260925/text_full152.pt', map_location='cpu')['empty'].cuda().float()
    random.seed(2027); np.random.seed(2027); torch.manual_seed(2027); torch.cuda.manual_seed_all(2027)
    model = InstanceCandidatePrototype().cuda()
    frozen = freeze_unused(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    with torch.no_grad():
        initial = model(batch(fit, torch.arange(64), empty, torch.device('cuda')))
    assert torch.equal(initial['selection_logits'], fit['base_scores'][:64].cuda())
    assert bool((initial['selected_index'] == 0).all())
    steps, max_semantic, max_phrase = 0, 0., 0.
    for indices in torch.randperm(len(fit['key']), generator=torch.Generator().manual_seed(2027)).split(64):
        data = batch(fit, indices, empty, torch.device('cuda'))
        assert data['text'].stride() == (0, 0, 1)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        assert torch.equal(output['selection_logits'], output['visual_selection_logits'])
        max_semantic = max(max_semantic, float(output['semantic_delta'].abs().max()))
        max_phrase = max(max_phrase, float(output['phrase_delta'].abs().max()))
        target = fit['iou'][indices].cuda()
        loss = F.binary_cross_entropy_with_logits(output['visual_selection_logits'], target) + F.binary_cross_entropy_with_logits(output['quality_logits'], target)
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        steps += 1
    check_frozen(model, frozen)
    assert steps == 40
    assert max_semantic == max_phrase == 0.
    result = dict(status='complete_actual_Empty_regression_only', seed=2027,
                  optimizer_steps=steps, original_expanded_input_unchanged=True,
                  exact_semantic_phrase_score_cancellation=True,
                  prototype_sha256=sha(Path(__file__).with_name('instance_ab_prototype.py')),
                  source_sha256=sha(Path(__file__)), no_checkpoint=True,
                  no_development_or_public_evaluation=True)
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
