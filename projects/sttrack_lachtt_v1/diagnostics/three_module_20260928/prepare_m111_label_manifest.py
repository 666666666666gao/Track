"""Bind original category queries to the authorized, private initial review."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


CLASSES = ['supported', 'conflicting', 'uncertain']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ['review', 'labels', 'output', 'summary']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and not a.summary.exists()
    labels = json.loads(a.labels.read_text(encoding='utf-8'))
    assert not labels['human_confirmed']
    assert labels['input_sha256'][a.review.name] == sha(a.review)
    with a.review.open(encoding='utf-8-sig', newline='') as f:
        reviewed = {r['audit_id']: r for r in csv.DictReader(f)}
    assert len(reviewed) == len(labels['initial']) == 152
    eligible = []
    counts = {s: Counter() for s in ['fit', 'development']}
    omitted = {s: Counter() for s in ['fit', 'development']}
    for bound in labels['initial']:
        r = reviewed[bound['audit_id']]
        assert r['record_type'] == 'initial'
        assert r['reviewed_category_status'] == bound['category_status']
        assert r['first_frame_category_observability'] == bound['first_frame_observability']
        status = r['reviewed_category_status']
        assert status in CLASSES and bound['split'] in counts
        if bound['first_frame_observability'] != 'clear':
            omitted[bound['split']][bound['first_frame_observability']] += 1
            continue
        query = r['generated_category'].strip()
        assert query
        eligible.append(dict(audit_id=bound['audit_id'], sequence=bound['sequence'],
                             split=bound['split'], query=query, label=CLASSES.index(status),
                             category_status=status, first_frame_observability='clear'))
        counts[bound['split']][status] += 1
    assert counts['fit'] == Counter(supported=80, conflicting=17, uncertain=1)
    assert counts['development'] == Counter(supported=15, conflicting=3, uncertain=1)
    assert len(eligible) == len({r['sequence'] for r in eligible}) == 117
    result = dict(status='prepared_M111_first_frame_model_weak_categories', human_confirmed=False,
                  classes=CLASSES, labels_sha256=sha(a.labels), review_sha256=sha(a.review),
                  rows=eligible, optimized_split='fit', later_frames_as_inputs=False,
                  protocol='Original generated category, not the corrected full phrase; clear initial observations only.')
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary = dict(status=result['status'], human_confirmed=False, classes=CLASSES,
                   manifest_sha256=sha(a.output), labels_sha256=sha(a.labels),
                   review_sha256=sha(a.review), source_sha256=sha(__file__),
                   eligible_counts={s: dict(c) for s, c in counts.items()},
                   excluded_observability={s: dict(c) for s, c in omitted.items()},
                   candidate_support_labels_used=False, corrected_phrase_labels_inherited=False,
                   unknown_fitting_examples=1, no_claim_of_current_candidate_visibility=True)
    a.summary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
