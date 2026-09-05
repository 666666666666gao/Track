"""Score sealed local VLM localization suggestions using posthoc GT only."""
import argparse
import json
from pathlib import Path
from statistics import mean
from analyze_m41 import iou


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    spec = json.loads((root / 'spec.json').read_text())
    cases = {c['key']: c for c in json.loads((root / 'inputs.json').read_text())}
    responses = [json.loads(line) for line in (root / 'responses.jsonl').read_text().splitlines()]
    assert len(responses) == 72
    assert {(r['key'], r['variant']) for r in responses} == {(k, v) for k in cases for v in spec['variants']}
    rows = []
    for response in responses:
        case = cases[response['key']]
        raw = response['raw']; first, last = raw.find('{'), raw.rfind('}')
        assert first >= 0 and last > first
        parsed = json.loads(raw[first:last + 1])
        gt_path = Path(spec['sequence_root']) / case['sequence'] / 'groundtruth.txt'
        gt = [float(v) for v in gt_path.read_text().splitlines()[case['current_frame']].split(',')]
        accepted = parsed['target_visible'] and not parsed['identity_uncertain'] and parsed['bbox_1000'] is not None
        overlap = 0.
        if accepted:
            x1, y1, x2, y2 = parsed['bbox_1000']; w, h = response['image_size']
            assert 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000
            overlap = iou([x1 * w / 1000, y1 * h / 1000, (x2 - x1) * w / 1000, (y2 - y1) * h / 1000], gt)
        rows.append(dict(key=response['key'], sequence=case['sequence'], variant=response['variant'], accepted=bool(accepted),
                         iou=overlap, correct=bool(accepted and overlap >= .5), wrong=bool(accepted and overlap <= .1),
                         protected_iou=iou(case['protected_bbox'], gt), dynamic_step=case['dynamic_step'],
                         latency_seconds=response['latency_seconds'], parsed=parsed))
    summaries = {}
    for variant in spec['variants']:
        chosen = [r for r in rows if r['variant'] == variant]
        summaries[variant] = dict(cases=len(chosen), accepted=sum(r['accepted'] for r in chosen),
            correct=sum(r['correct'] for r in chosen), wrong=sum(r['wrong'] for r in chosen),
            mean_iou=mean(r['iou'] for r in chosen), mean_latency_seconds=mean(r['latency_seconds'] for r in chosen),
            per_sequence={seq: dict(correct=sum(r['correct'] for r in chosen if r['sequence'] == seq),
                                    wrong=sum(r['wrong'] for r in chosen if r['sequence'] == seq)) for seq in sorted({r['sequence'] for r in chosen})})
    base, relative = summaries['default_templates_identity'], summaries['default_templates_relative']
    improved_sequences = [seq for seq in base['per_sequence'] if relative['per_sequence'][seq]['correct'] > base['per_sequence'][seq]['correct']]
    result = dict(status='complete', summaries=summaries, improved_sequences=improved_sequences,
        relative_gate_passed=relative['correct'] > base['correct'] and relative['wrong'] <= base['wrong'] and len(improved_sequences) >= 2,
        rank_accuracy_not_available=True, tracker_commits=0, training_steps=0,
        claim_limit='Small GT-timed causal-information pilot, not an online tracker result or a VOT metric.', rows=rows)
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
