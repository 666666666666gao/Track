"""Bind actual current-region model review to exact phrase and cached candidate.

No physical-instance label or GT-derived semantic negative is produced.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


CLASSES = ['supported', 'conflicting', 'unknown']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ['review', 'reviewer-input', 'private-binding', 'weak-labels', 'output', 'summary']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.summary.exists()
    packet = json.loads(args.reviewer_input.read_text(encoding='utf-8'))
    bindings = json.loads(args.private_binding.read_text(encoding='utf-8'))
    weak = json.loads(args.weak_labels.read_text(encoding='utf-8'))
    initial = {row['sequence']: row for row in weak['initial']}
    assert len(packet['cases']) == len(bindings) == 24
    binding_by_id = {row['audit_id']: row for row in bindings}
    expected = {}
    for case in packet['cases']:
        bound = binding_by_id[case['audit_id']]
        assert bound['split'] == initial[bound['sequence']]['split'] == 'fit'
        assert case['phrases'] == bound['phrases'] == initial[bound['sequence']]['phrases']
        for candidate in ['A', 'B']:
            for slot, query in enumerate(case['phrases']):
                expected[(case['audit_id'], candidate, slot)] = query
    with args.review.open(encoding='utf-8-sig', newline='') as handle:
        reviewed = list(csv.DictReader(handle))
    assert len(reviewed) == len(expected) == 202
    rows, seen = [], set()
    for row in reviewed:
        key = (row['audit_id'], row['candidate'], int(row['slot']))
        assert key in expected and key not in seen
        seen.add(key)
        assert row['query'] == expected[key]
        assert row['label'] in CLASSES and row['evidence_note'].strip() and row['reviewer'].strip()
        assert row['human_confirmed'].strip().lower() == 'false'
        bound = binding_by_id[row['audit_id']]
        candidate_index = bound['a_index'] if row['candidate'] == 'A' else bound['b_index']
        rows.append(dict(audit_id=row['audit_id'], sequence=bound['sequence'], frame=bound['frame'],
                         key=bound['sequence'] + '@' + str(bound['frame']), candidate=row['candidate'],
                         candidate_index=candidate_index, slot=key[2], query=row['query'],
                         label=CLASSES.index(row['label']), label_name=row['label'],
                         evidence_note=row['evidence_note'], reviewer=row['reviewer'], split='fit'))
    assert seen == set(expected)
    counts = Counter(row['label_name'] for row in rows)
    by_slot = {str(slot): dict(Counter(row['label_name'] for row in rows if row['slot'] == slot))
               for slot in range(5)}
    paired = {(row['audit_id'], row['slot'], row['candidate']): row['label_name'] for row in rows}
    pair_counts = Counter((paired[(audit_id, slot, 'A')], paired[(audit_id, slot, 'B')])
                         for audit_id, candidate, slot in expected if candidate == 'A')
    source_hashes = dict(review=sha(args.review), reviewer_input=sha(args.reviewer_input),
                         private_binding=sha(args.private_binding), weak_labels=sha(args.weak_labels))
    manifest = dict(status='prepared_M112_current_phrase_model_weak_labels', human_confirmed=False,
                    classes=CLASSES, rows=rows, optimized_split='fit', input_sha256=source_hashes,
                    physical_instance_labels=False, depth_semantic_labels=False,
                    protocol='Current RGB phrase support/conflict/unknown. Train temporal context is teacher-only; '
                             'unknown is not a negative identity label. No GT/IoU semantic labels.')
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = {key: value for key, value in manifest.items() if key != 'rows'}
    summary.update(events=24, candidates=48, phrase_candidate_rows=len(rows),
                   development_label_events=0, label_counts=dict(counts), by_slot=by_slot,
                   phrase_pair_label_counts={'/'.join(pair): count for pair, count in pair_counts.items()},
                   manifest_sha256=sha(args.output), source_sha256=sha(Path(__file__)))
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
