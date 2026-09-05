"""Exploratory static readout of trained affinity; never changes M52 policy."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    torch.set_num_threads(1)
    plan = json.loads((root/'spec.json').read_text())
    parent = Path(plan['source_root'])
    repository = Path(json.loads((parent/'spec.json').read_text())['repository'])
    sys.path.insert(0, str(repository))
    from tools.train_sttrack_m52 import check_sources, load_data, FIELDS, sha
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation, select_candidate
    plan, parent, spec = check_sources(root)
    tensors, keys, previous_choice, ious, target, previous_target, fit, policy_fit, dev, audit = load_data(root, plan, parent, spec)
    assert audit == json.loads((root/'data_audit.json').read_text())
    results = {}
    for arm in ['control', 'mixed']:
        assert (root/('training_'+arm+'.exit')).read_text().strip() == '0'
        training = json.loads((root/arm/'training_result.json').read_text())
        checkpoint = root/arm/'geometry_final.pth'
        assert sha(checkpoint) == training['checkpoint_sha256']
        net = CandidateSetAssociation(True).eval()
        net.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=True)
        scopes = {}
        for scope, index in [('native_fit', fit), ('policy_fit', policy_fit), ('development', dev)]:
            logits, affinities = [], []
            with torch.no_grad():
                for batch in index.split(32):
                    scores, affinity = net(*[tensors[k][batch].float() for k in FIELDS], previous_choice[batch])
                    logits.append(scores); affinities.append(affinity)
            logits, affinity = torch.cat(logits), torch.cat(affinities)
            classifier = select_candidate(logits)
            expected = torch.tensor([r['chosen'] for r in training[scope]['rows']])
            assert [keys[i] for i in index] == [r['key'] for r in training[scope]['rows']]
            assert torch.equal(classifier, expected), (arm, scope, 'CPU classifier differs from recorded GPU classifier')
            # No GT enters this readout: use the actual previous-choice input.
            column = previous_choice[index][:, None, None].expand(-1, 11, 1)
            causal_raw = affinity.gather(2, column).squeeze(-1).argmax(1)
            causal = torch.where(causal_raw == 10, torch.zeros_like(causal_raw), causal_raw)
            # Explicit offline oracle: known previous GT candidate, never deployable.
            oracle_column = previous_target[index][:, None, None].expand(-1, 11, 1)
            oracle_raw = affinity.gather(2, oracle_column).squeeze(-1).argmax(1)
            oracle = torch.where(oracle_raw == 10, torch.zeros_like(oracle_raw), oracle_raw)
            def metrics(chosen, mask):
                subset = ious[index][mask]
                selected = subset.gather(1, chosen[mask, None]).flatten()
                baseline = subset[:, 0]
                return dict(events=int(mask.sum()), mean_iou=float(selected.mean()),
                            default_mean_iou=float(baseline.mean()), correct=int((selected >= .5).sum()),
                            changes=int((chosen[mask] != 0).sum()),
                            rescues=int(((baseline <= .1) & (selected >= .5)).sum()),
                            breaks=int(((baseline >= .5) & (selected <= .1)).sum()))
            all_events = torch.ones(len(index), dtype=torch.bool)
            previous_available = previous_target[index] < 10
            assert previous_available.any()
            scopes[scope] = dict(
                classifier_cpu_matches_recorded_gpu=True,
                classifier=metrics(classifier, all_events),
                causal_previous_choice_affinity=metrics(causal, all_events),
                causal_unmatched=int((causal_raw == 10).sum()),
                oracle_subset_classifier=metrics(classifier, previous_available),
                oracle_subset_causal=metrics(causal, previous_available),
                oracle_previous_gt_affinity=metrics(oracle, previous_available),
                rows=[dict(key=keys[i], previous_choice=int(previous_choice[i]), classifier=int(c),
                           causal_affinity=int(a), causal_unmatched=bool(n == 10),
                           oracle_previous_gt_affinity=int(o) if bool(valid) else None)
                      for i, c, a, n, o, valid in zip(index, classifier, causal, causal_raw, oracle, previous_available)])
        results[arm] = dict(checkpoint_sha256=sha(checkpoint), scopes=scopes)
    check_sources(root)
    output = dict(status='complete', arms=results, source_sha256=sha(Path(__file__)),
                  spec_sha256=sha(root/'spec.json'), data_audit_sha256=sha(root/'data_audit.json'),
                  optimizer_steps=0, recursive_policy_changed=False,
                  scope='Post-training exploratory static diagnostic on original-native and fixed-M45 caches. Not KeepTrack reproduction, not online ID propagation, not a calibrated affinity probability, and not a promotion gate. Oracle previous-GT readout is explicitly privileged. Unmatched maps to candidate zero for output comparison only.')
    (root/'affinity_readout_diagnostic.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({arm: {scope: {k: v for k, v in value.items() if k != 'rows'}
                            for scope, value in arm_result['scopes'].items()}
                      for arm, arm_result in results.items()}, indent=2))


if __name__ == '__main__':
    main()
