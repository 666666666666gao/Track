"""Posthoc scope audit of sealed M57 content interventions; no new inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(pairs):
    count = len(pairs)
    return dict(events=count, sequences=sorted({real['sequence'] for real, _ in pairs}),
        valid_gt_events=sum(real['valid_gt'] for real, _ in pairs),
        real_mean_iou=sum(real['selected_iou'] for real, _ in pairs) / count if count else None,
        altered_mean_iou=sum(altered['selected_iou'] for _, altered in pairs) / count if count else None,
        choice_changes=sum(real['candidate'] != altered['candidate'] for real, altered in pairs),
        real_correct_altered_severe=sum(real['selected_iou'] >= .5 and altered['selected_iou'] <= .1 for real, altered in pairs),
        real_severe_altered_correct=sum(real['selected_iou'] <= .1 and altered['selected_iou'] >= .5 for real, altered in pairs))


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--text-manifest', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
assert sha(args.root / 'static_result.json') == '1987764e5db554232289366600cbc06c979767f105ca886d246b121a45057556'
assert sha(args.text_manifest) == '97e18d86b6f9406d198d682c9e3c91cb2fc5edcceaf33985bb03674c5f139db7'
spec = json.loads((args.root / 'static_spec.json').read_text())
result = json.loads((args.root / 'static_result.json').read_text())
assert result['static_spec_sha256'] == sha(args.root / 'static_spec.json')
manifest = {row['sequence']: row for row in json.loads(args.text_manifest.read_text())}
same_category_names = sorted(name for name, donor in spec['same_mask_donors'].items()
                             if manifest[name]['category'] == manifest[donor]['category'])
reports = {}
for arm in ['candidate_text', 'initial_text']:
    real = {row['key']: row for row in result['variants'][arm]['rows']}
    pairs = [(real[row['key']], row) for row in result['variants'][arm + '_same_mask_shuffled']['rows']]
    subsets = {
        'all_matched_masks': pairs,
        'same_category_different_description': [(a, b) for a, b in pairs if a['sequence'] in same_category_names],
        'different_category': [(a, b) for a, b in pairs if a['sequence'] not in same_category_names],
        'same_category_default_below_half_oracle_at_least_half': [(a, b) for a, b in pairs
            if a['sequence'] in same_category_names and a['default_iou'] < .5 and a['oracle_iou'] >= .5],
    }
    severe_rows = []
    for a, b in pairs:
        if ((a['selected_iou'] >= .5 and b['selected_iou'] <= .1)
                or (a['selected_iou'] <= .1 and b['selected_iou'] >= .5)):
            name, donor = a['sequence'], spec['same_mask_donors'][a['sequence']]
            severe_rows.append(dict(key=a['key'], sequence=name, donor=donor,
                real_iou=a['selected_iou'], altered_iou=b['selected_iou'],
                real_choice=a['candidate'], altered_choice=b['candidate'],
                same_category=name in same_category_names, original_phrases=manifest[name]['phrases'],
                altered_phrases=manifest[donor]['phrases'], original_review_status=manifest[name]['review_status']))
    reports[arm] = dict(subsets={name: summary(values) for name, values in subsets.items()}, severe_direction_changes=severe_rows)
pair_descriptions = [dict(sequence=name, donor=spec['same_mask_donors'][name],
    category=manifest[name]['category'], original_phrases=manifest[name]['phrases'],
    altered_phrases=manifest[spec['same_mask_donors'][name]]['phrases']) for name in same_category_names]
output = dict(status='complete_posthoc_static_content_scope_audit',
    static_result_sha256=sha(args.root / 'static_result.json'), static_spec_sha256=sha(args.root / 'static_spec.json'),
    text_manifest_sha256=sha(args.text_manifest), auditor_sha256=sha(Path(__file__)),
    category_pairing_predeclared=False, original_donors_and_weights_changed=False,
    full_recursive_predictions_or_gt_opened=False, new_model_inference=False,
    reports=reports, same_category_pair_descriptions=pair_descriptions,
    historical_review_status_counts=dict(Counter(row['review_status'] for row in manifest.values())),
    interpretation='This posthoc subgroup examines changed decisions, not unchanged logits or a causal identity guarantee. Different descriptions from the same category are not independently verified false-attribute labels. The principal severe positive cases are not independent sequence-level replicates.')
args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps(dict(status=output['status'], output_sha256=sha(args.output),
    reports={name: value['subsets'] for name, value in reports.items()}), indent=2))
