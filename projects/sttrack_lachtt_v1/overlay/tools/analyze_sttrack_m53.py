"""Post-seal GT capacity analysis; never selects a deployed template."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(box, gt):
    width = max(0., min(box[0]+box[2], gt[0]+gt[2])-max(box[0], gt[0]))
    height = max(0., min(box[1]+box[3], gt[1]+gt[3])-max(box[1], gt[1]))
    intersection = width*height
    return float(intersection/(box[2]*box[3]+gt[2]*gt[3]-intersection))


def valid_gt(box):
    return bool(np.isfinite(box).all() and box[2] > 0 and box[3] > 0)


def event_capacity(event, gt, write_quality):
    """Current control is always available; only past alternatives are filtered."""
    control = event['baseline']
    all_views = [dict(template_frame=event['active_template_frame'], **control)] + event['alternatives']
    evaluated = [dict(template_frame=view['template_frame'], write_iou=write_quality[view['template_frame']],
                      top1=overlap(view['bbox'], gt),
                      top10=max(overlap(c['bbox'], gt) for c in view['candidates'])) for view in all_views]
    trusted = [evaluated[0]] + [v for v in evaluated[1:]
                                if write_quality[v['template_frame']] is not None
                                and write_quality[v['template_frame']] >= .5]
    modes = {}
    for name, views in [('default', evaluated[:1]), ('all_past', evaluated), ('valid_past', trusted)]:
        best_one = max(views, key=lambda v: v['top1'])
        best_ten = max(views, key=lambda v: v['top10'])
        modes[name] = dict(oracle_top1_iou=best_one['top1'], union_top10_iou=best_ten['top10'],
                           best_top1_template_frame=best_one['template_frame'],
                           best_top10_template_frame=best_ten['template_frame'])
    healthy = evaluated[0]['top1'] >= .5
    return dict(modes=modes, views=evaluated, past_reads=len(evaluated)-1, valid_past_reads=len(trusted)-1,
                harmful_past_reads=sum(healthy and v['top1'] <= .1 for v in evaluated[1:]),
                harmful_valid_past_reads=sum(healthy and v['top1'] <= .1 for v in trusted[1:]),
                healthy_past_reads=(len(evaluated)-1) if healthy else 0,
                healthy_valid_past_reads=(len(trusted)-1) if healthy else 0)


def summarize(rows, mode):
    evaluated = [r for r in rows if r['valid_gt']]
    result = dict(events=len(rows), valid_events=len(evaluated), invalid_gt_events=len(rows)-len(evaluated),
                  top1_correct=0, top10_available=0, severe_events_recovered=0,
                  current_candidate_rank_improved=0, new_candidate_events=0)
    recovered_sequences = set()
    values = []
    for row in evaluated:
        base = row['modes']['default']; current = row['modes'][mode]
        one, ten = current['oracle_top1_iou'], current['union_top10_iou']
        values.append(one)
        result['top1_correct'] += one >= .5
        result['top10_available'] += ten >= .5
        recovered = base['oracle_top1_iou'] <= .1 and one >= .5
        result['severe_events_recovered'] += recovered
        result['current_candidate_rank_improved'] += base['oracle_top1_iou'] < .5 and base['union_top10_iou'] >= .5 and one >= .5
        result['new_candidate_events'] += base['union_top10_iou'] < .5 and ten >= .5
        if recovered:
            recovered_sequences.add(row['sequence'])
    result['oracle_top1_iou_sum'] = math.fsum(values)
    result['mean_oracle_top1_iou'] = math.fsum(values)/len(values) if values else None
    result['recovered_sequences'] = sorted(recovered_sequences)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    # The exits and every sealed output are checked before any GT is opened.
    assert (root/'collection.exit').read_text().strip() == '0'
    assert (root/'controller.exit').read_text().strip() == '0'
    plan = json.loads((root/'spec.json').read_text())
    binding = json.loads((root/'analysis_binding.json').read_text())
    assert binding['spec_sha256'] == sha(root/'spec.json')
    assert binding['analysis_source_sha256'] == sha(Path(__file__))
    assert binding['execution_binding_sha256'] == sha(root/'execution_binding.json')
    parent = Path(plan['source_root'])
    source = json.loads((parent/'spec.json').read_text())
    repository = Path(source['repository'])
    old_binding = json.loads((parent/'training_binding.json').read_text())
    assert binding['native_training_binding_sha256'] == sha(parent/'training_binding.json')
    assert old_binding['spec_sha256'] == plan['source_spec_sha256'] == sha(parent/'spec.json')

    def check_sources():
        for name, digest in {**source['source_sha256'], **old_binding['source_sha256'], **plan['source_sha256']}.items():
            assert sha(repository/name) == digest, name
        assert sha(source['checkpoint']) == source['checkpoint_sha256']
        assert sha(parent/'inference_inputs.json') == source['inference_inputs_sha256']
        assert sha(root/'EXPERIMENT_PLAN.md') == plan['experiment_plan_sha256']
        assert sha(Path(plan['m52_root'])/'recursive_result.json') == plan['m52_result_sha256']

    check_sources()
    cases = {c['sequence']: c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split'] == 'fit'}
    assert sorted(cases) == plan['sequences'] and len(cases) == 63
    receipt = json.loads((root/'collection_receipt.json').read_text())
    assert receipt['status'] == 'complete' and receipt['source_unchanged']
    assert not receipt['labels_opened'] and receipt['optimizer_steps'] == 0
    assert receipt['spec_sha256'] == sha(root/'spec.json')
    assert receipt['checkpoint_sha256'] == source['checkpoint_sha256']
    assert receipt['current_template_replay_exact'] and receipt['public_state_unchanged']
    assert receipt['events'] == 1511 and receipt['frames'] == 93362
    execution = json.loads((root/'execution_binding.json').read_text())
    assert receipt['past_template_shadows'] == execution['expected_past_template_shadows_from_native_predictions']
    assert len(receipt['sequences']) == 63 and {r['sequence'] for r in receipt['sequences']} == set(cases)
    collected = {}
    for item in receipt['sequences']:
        name = item['sequence']; path = root/'events'/(name+'.json')
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
        data = json.loads(path.read_text()); case = cases[name]
        assert data['sequence'] == name and data['split'] == 'fit' and data['fold'] == case['fold']
        assert data['fold'] in source['fit_folds']
        assert item['max_bbox_error_px'] <= 1e-4 and item['max_score_error'] <= 1e-6
        assert [e['frame'] for e in data['events']] == sorted(case['event_frames'])
        assert len(data['events']) == item['events'] and max(case['event_frames']) == item['frames']
        writes = data['template_writes']; frames = [w['frame'] for w in writes]
        assert frames[0] == 0 and frames == sorted(set(frames))
        assert len(writes)-1 == item['template_updates']
        assert all(w['frame'] % 50 == 0 and w['score'] > .75 for w in writes[1:])
        assert sum(len(e['alternatives']) for e in data['events']) == item['past_template_shadows']
        for event in data['events']:
            assert event['key'] == '{}@{}'.format(name, event['frame'])
            assert event['current_template_replay_exact'] and event['public_state_unchanged']
            available = [f for f in frames if f < event['frame']]
            assert event['active_template_frame'] == available[-1]
            assert [v['template_frame'] for v in event['alternatives']] == available[:-1]
            for view in [event['baseline']] + event['alternatives']:
                boxes = np.asarray([view['bbox']] + [c['bbox'] for c in view['candidates']])
                assert boxes.shape == (11, 4) and np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                assert view['candidates'][0]['bbox'] == view['bbox']
            expected = case['expected_rows'][event['frame']]
            assert np.max(np.abs(np.asarray(event['baseline']['bbox'])-expected['bbox'])) <= 1e-4
            assert abs(event['baseline']['score']-expected['score']) <= 1e-6
        collected[name] = data

    assert sum(len(d['events']) for d in collected.values()) == receipt['events']
    assert sum(len(d['template_writes'])-1 for d in collected.values()) == receipt['native_template_updates']
    screen = plan['capacity_screen']
    assert screen['severe_default_iou_at_most'] == .1 and screen['past_top1_iou_at_least'] == .5
    # All collection files are verified above. GT is first opened below.
    rows = []
    groundtruth_sha256 = {}
    write_counts = dict(total_including_initial=0, valid_gt=0, overlap_at_least_half=0)
    for name in sorted(collected):
        data = collected[name]
        path = Path(source['dataset_root'])/name/'groundtruth.txt'
        groundtruth_sha256[name] = sha(path)
        gt = np.loadtxt(path, delimiter=',')[:cases[name]['frames']]
        assert gt.shape == (cases[name]['frames'], 4)
        quality = {}
        for write in data['template_writes']:
            target = gt[write['frame']]
            quality[write['frame']] = overlap(write['bbox'], target) if valid_gt(target) else None
            write_counts['total_including_initial'] += 1
            if quality[write['frame']] is not None:
                write_counts['valid_gt'] += 1
                write_counts['overlap_at_least_half'] += quality[write['frame']] >= .5
        for event in data['events']:
            target = gt[event['frame']]
            row = dict(sequence=name, key=event['key'], frame=event['frame'], valid_gt=valid_gt(target),
                       available_past_reads=len(event['alternatives']))
            if row['valid_gt']:
                row.update(event_capacity(event, target, quality))
                base = row['modes']['default']['oracle_top1_iou']
                row['current_overlap_stratum'] = 'severe' if base <= .1 else ('correct' if base >= .5 else 'partial')
            else:
                row['current_overlap_stratum'] = 'invalid_gt'
            rows.append(row)

    modes = ['default', 'all_past', 'valid_past']
    summary = {mode: summarize(rows, mode) for mode in modes}
    strata = {stratum: {mode: summarize([r for r in rows if r['current_overlap_stratum'] == stratum], mode)
                       for mode in modes} for stratum in ['severe', 'partial', 'correct', 'invalid_gt']}
    per_sequence = {name: {mode: summarize([r for r in rows if r['sequence'] == name], mode) for mode in modes}
                    for name in sorted(collected)}
    harm_fields = ['past_reads', 'valid_past_reads', 'harmful_past_reads', 'harmful_valid_past_reads',
                   'healthy_past_reads', 'healthy_valid_past_reads']
    harms = {field: sum(r[field] for r in rows if r['valid_gt']) for field in harm_fields}
    harms['invalid_event_gt_past_reads'] = sum(r['available_past_reads'] for r in rows if not r['valid_gt'])
    harms['healthy_events_with_harmful_past_read'] = sum(r['harmful_past_reads'] > 0 for r in rows if r['valid_gt'])
    harms['healthy_events_with_harmful_valid_past_read'] = sum(r['harmful_valid_past_reads'] > 0 for r in rows if r['valid_gt'])
    assert harms['past_reads'] + harms['invalid_event_gt_past_reads'] == receipt['past_template_shadows']
    passed = (summary['all_past']['severe_events_recovered'] >= screen['events_at_least']
              and len(summary['all_past']['recovered_sequences']) >= screen['sequences_at_least'])
    check_sources()
    assert binding['analysis_source_sha256'] == sha(Path(__file__))
    result = dict(status='complete', scope='Privileged single-frame historical-template capacity on fixed fitting events',
                  spec_sha256=sha(root/'spec.json'), analysis_binding_sha256=sha(root/'analysis_binding.json'),
                  analysis_source_sha256=sha(Path(__file__)), collection_receipt_sha256=sha(root/'collection_receipt.json'),
                  groundtruth_sha256=groundtruth_sha256, optimizer_steps=0, public_evaluation=False,
                  capacity_screen_pass=passed, capacity_screen=screen, summary=summary,
                  current_overlap_strata=strata, individual_read_harms=harms, archived_write_quality=write_counts,
                  per_sequence=per_sequence, events=rows,
                  limitations=[
                      'Best-view selection and filtering historical writes by GT are privileged; no causal reader was evaluated.',
                      'The current control is available in every oracle mode, so oracle means cannot reveal harmful selection.',
                      'Top1 and top10 best views may differ; their template frame IDs are reported separately.',
                      'All past native writes are archived; this does not establish capacity of a bounded memory bank.',
                      'Invalid GT events are excluded from overlap means and capacity counts, and reported separately.',
                      'Fixed training events were selected by earlier diagnostics; they are not a deployable event trigger.',
                      'Capacity-screen success permits reader design only; it is not a recursive or benchmark promotion.'])
    (root/'capacity_result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    columns = ['sequence', 'mode', 'events', 'valid_events', 'invalid_gt_events', 'top1_correct', 'top10_available',
               'severe_events_recovered', 'current_candidate_rank_improved', 'new_candidate_events',
               'oracle_top1_iou_sum', 'mean_oracle_top1_iou']
    with (root/'per_sequence.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for name, values in per_sequence.items():
            for mode, metrics in values.items():
                writer.writerow(dict(sequence=name, mode=mode, **{k: metrics[k] for k in columns[2:]}))
    print(json.dumps(dict(status=result['status'], capacity_screen_pass=passed, summary=summary,
                          individual_read_harms=harms), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
