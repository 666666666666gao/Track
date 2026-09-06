"""Freeze the three-arm design after the real CPU engineering contract."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert not (root / 'spec.json').exists()
    parent = Path('/root/autodl-tmp/sttrack_m44_candidate_set_v1_20260905')
    p = json.loads((parent / 'spec.json').read_text())
    binding = json.loads((parent / 'training_binding.json').read_text())
    repo = Path(p['repository'])
    source = {**p['source_sha256'], **binding['source_sha256']}
    for name, digest in source.items():
        assert sha(repo / name) == digest, name
    for name in ['lib/models/sttrack/lachtt_attribute_candidate_set.py', 'lib/test/tracker/sttrack_attribute_candidate_set.py']:
        source[name] = sha(repo / name)
    assert source['lib/models/sttrack/lachtt_attribute_candidate_set.py'] == sha(root / 'model.py')
    contract = json.loads((root / 'contract.json').read_text())
    assert contract['status'] == 'pass' and contract['model_sha256'] == sha(root / 'model.py')
    assert contract['parameter_count'] == 484387 and contract['added_parameter_count'] == 35648
    assert contract['checker_sha256'] == sha(root / 'check.py')
    preparation = json.loads((root / 'text_preparation.json').read_text())
    assert preparation['status'] == 'complete' and preparation['preparer_sha256'] == sha(root / 'prepare.py')
    for name, digest in preparation['files'].items():
        assert sha(root / name) == digest
    manifest = json.loads((root / 'text_manifest.json').read_text())
    # Controls use text metadata only and are fixed before optimizer execution.
    colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'brown', 'orange', 'pink', 'purple', 'gray', 'grey']
    control_rows = {}
    for split in ['fit', 'development']:
        rows = [row for row in manifest if row['split'] == split]
        count = len(rows)
        shifts = [shift for shift in range(1, count)
                  if all(row['phrases'] != rows[(i + shift) % count]['phrases'] for i, row in enumerate(rows))]
        assert shifts
        shift = shifts[0]
        for i, row in enumerate(rows):
            recipient_colors = set(re.findall(r'\b(?:' + '|'.join(colors) + r')\b', ' '.join(row['stable_attributes']).lower()))
            donors = []
            for candidate in rows:
                candidate_colors = set(re.findall(r'\b(?:' + '|'.join(colors) + r')\b', ' '.join(candidate['stable_attributes']).lower()))
                if (candidate['category'] == row['category'] and recipient_colors and candidate_colors
                        and recipient_colors.isdisjoint(candidate_colors)):
                    donors.append(candidate['sequence'])
            control_rows[row['sequence']] = dict(split=split, shuffled_donor=rows[(i + shift) % count]['sequence'],
                same_category_conflict_donor=min(donors) if donors else None,
                annotated_color_words=sorted(recipient_colors))
    controls = dict(rows=control_rows, color_vocabulary=colors,
        shuffled_rule='First sorted cyclic derangement with all phrase bundles different, separately within fit/development',
        conflict_rule='Same annotated category, nonempty disjoint annotated color words; lexicographically first donor',
        conflict_scope='Conflict with existing first-frame annotations, not independent proof that every visible surface has the opposite color',
        eligible_development_sequences=[name for name, value in control_rows.items()
            if value['split'] == 'development' and value['same_category_conflict_donor'] is not None])
    by_name = {row['sequence']: row for row in manifest}
    for name, value in control_rows.items():
        recipient = by_name[name]
        shuffled = by_name[value['shuffled_donor']]
        value['shuffled_category_changed'] = recipient['category'] != shuffled['category']
        value['shuffled_phrase_count_changed'] = len(recipient['phrases']) != len(shuffled['phrases'])
        donor = value['same_category_conflict_donor']
        value['conflict_phrase_count_changed'] = None if donor is None else len(recipient['phrases']) != len(by_name[donor]['phrases'])
    controls['predefined_stratification'] = 'Report all controls, plus phrase-count-matched subsets; shuffled text also stratified by category preservation'
    (root / 'text_controls.json').write_text(json.dumps(controls, indent=2) + '\n')
    names = {row['sequence'] for row in manifest if row['split'] == 'fit'}
    receipts = []
    for shard in [0, 1]:
        receipt = json.loads((parent / ('shard%d_receipt.json' % shard)).read_text())
        assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(parent / 'spec.json')
        receipts.extend(receipt['sequences'])
    features = [dict(sequence=row['sequence'], events=row['events'], sha256=row['feature_sha256'])
                for row in sorted(receipts, key=lambda x: x['sequence']) if row['sequence'] in names]
    assert len(features) == 63 and sum(row['events'] for row in features) == 1511
    files = [root / name for name in ['text_preparation.json', 'text_manifest.json', 'text_controls.json', 'text_bank.pt', 'fit_labels.json', 'contract.json', 'EXPERIMENT_PLAN.md']]
    files += [parent / name for name in ['spec.json', 'training_binding.json', 'shard0_receipt.json', 'shard1_receipt.json']]
    files += [Path(p['checkpoint'])]
    spec = dict(schema='sttrack_m56_attribute_local_alignment_v1', frozen_unix=time.time(),
        repository=str(repo), source_root=str(parent), dataset_root=p['dataset_root'],
        parent_spec_sha256=sha(parent / 'spec.json'), source_sha256=source,
        input_sha256={str(path): sha(path) for path in files},
        trainer_sha256=sha(root / 'train.py'), static_analyzer_sha256=sha(root / 'analyze_static.py'), binder_sha256=sha(__file__),
        base_checkpoint=p['checkpoint'], base_checkpoint_sha256=p['checkpoint_sha256'],
        fit_features=features, parameters=484387, added_parameters=35648,
        variants=['attributes', 'pooled', 'empty'],
        optimization=dict(seed=2026, epochs=20, batch_size=32, lr=.0003, weight_decay=.01,
                          grad_clip=1., optimizer='AdamW', optimizer_steps=960, device='cpu', cpu_threads=4,
                          checkpoint='fixed final epoch only'),
        model=dict(phrase_dimension=768, phrase_slots=5, local_channels=32, cross_attention_heads=4,
                   roi_shape=[2, 16, 768], text_reads_visual=True, visual_reads_conditioned_text=True,
                   zero_initial_residual=True, backbone_frozen=True, text_encoder_frozen=True,
                   native_default_template_updates=True, candidate_generation_uses_language=False),
        training_target='M45 default-priority labels; current and previous IoU>=.5; partial target-only matching',
        state_distribution='Native STTrack trajectories only, previous_choice=0; no claim of policy-state augmentation',
        annotation_scope='Existing Train category and stable attributes; seven of 85 carry historical cross-frame/adjudication review provenance',
        train_label_scope='Preparation parses existing mixed label file and writes only 1511 fit labels; trainer cannot read development numerical targets',
        recursive_gate=dict(mean_gain_vs_native=.002, mean_gain_vs_empty=.001, mean_gain_vs_pooled=.001,
                            low_frames_no_increase_vs_all=True, H10_no_increase_vs_all=True,
                            protect_native_zero_H10=True, all_22_sequences_required=True),
        low22_gate=dict(EAO_gain_pp=.2, ROB_gain_pp=.2, ACC_min_delta_pp=-.1,
                        fewer_than_124_failures=True, protect_all_7_zero_failure_sequences=True),
        public_evaluation_requires_recursive_pass=True,
        claim='Frozen language hypothesis, no M56 performance yet; modest development gates do not imply original final targets reached')
    (root / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(dict(status='frozen', spec_sha256=sha(root / 'spec.json'), parameters=484387,
        fit_events=1511, steps_per_arm=960, estimated_cpu_seconds=contract['estimated_three_arm_train_seconds'])), flush=True)


if __name__ == '__main__':
    main()
