"""Read-only equal-window-budget attribution of the frozen M70 cross-region probes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE = Path('/root/autodl-tmp')
INVENTORY = BASE / 'sttrack_m70_recovery_window_inventory_20260907'
PARENT = INVENTORY / 'candidate_capacity'
ROOT = BASE / 'sttrack_m71_equal_budget_routing_20260907'
M70 = BASE / 'm70_candidate_capacity_20260907.py'
M70_SHA = 'e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977'
PARENT_SHA = '796a2756db4c2be0b5282d567294a76d494ad031cc4e186ba96deb3228c00fa5'
STRATA = ['all', 'H10', 'H10_local_inside', 'H10_local_outside', 'H10_outside_visible_centre', 'valid_after_invalid', 'healthy']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def parent():
    assert sha(M70) == M70_SHA and sha(PARENT / 'spec.json') == PARENT_SHA
    spec = importlib.util.spec_from_file_location('m71_parent_capacity', str(M70))
    app = importlib.util.module_from_spec(spec); spec.loader.exec_module(app)
    g, training, frozen = app.checked()
    return app, g, training, frozen


def member(row, group):
    inside = row['null_local_centre_inside']; failure = 'H10' in row['tags']
    return {'all': True, 'H10': failure, 'H10_local_inside': failure and inside,
        'H10_local_outside': failure and not inside,
        'H10_outside_visible_centre': failure and not inside and row['GT_centre_inside_image'],
        'valid_after_invalid': 'valid_after_invalid' in row['tags'], 'healthy': 'healthy' in row['tags']}[group]


def prior_order(windows, previous):
    centre = np.asarray(previous[:2]) + np.asarray(previous[2:]) / 2
    distances = []
    for i, window in enumerate(windows[3:], 3):
        x, y, w, h = window['rectangle']
        distances.append((float(np.square(np.asarray([x + w / 2, y + h / 2]) - centre).sum()), i))
    return [i for _, i in sorted(distances)]


def geometry_row(event, app, g):
    previous = event['arms']['null']['previous_bbox']; width = event['image_width']; height = event['image_height']
    windows = app.windows(previous, width, height, g); order = prior_order(windows, previous)
    target = event['GT_bbox']; coarse_side = windows[2]['rectangle'][2]
    row = dict(sequence=event['sequence'], frame=event['frame'], tags=event['tags'],
        null_local_centre_inside=event['arms']['null']['local']['centre_inside'],
        GT_centre_inside_image=event['GT_vs_image']['centre_inside'], grid_windows=len(order),
        coarse_nominal_target_width=target[2] * 256 / coarse_side, coarse_nominal_target_height=target[3] * 256 / coarse_side,
        prior_windows={})
    for k in [1, 2, 3]:
        chosen = order[:k]; values = [g.coverage(windows[i]['rectangle'], target) for i in chosen]
        row['prior_windows'][str(k)] = dict(actual_windows=len(chosen), indices=chosen,
            centre_covered=any(v['centre_inside'] for v in values), full_box_covered=any(v['full_box_inside'] for v in values))
    return row


def geometry_summary(rows):
    summaries = {}
    for group in STRATA:
        selected = [r for r in rows if member(r, group)]
        short = [min(r['coarse_nominal_target_width'], r['coarse_nominal_target_height']) for r in selected]
        summaries[group] = dict(events=len(selected), sequences=len({r['sequence'] for r in selected}),
            grid_windows_total=sum(r['grid_windows'] for r in selected),
            grid_windows_median=float(np.median([r['grid_windows'] for r in selected])) if selected else None,
            coarse_nominal_short_side_median=float(np.median(short)) if short else None,
            coarse_nominal_short_side_below={str(p): sum(x < p for x in short) for p in [4, 8, 16]},
            prior_window_coverage={str(k): dict(centre=sum(r['prior_windows'][str(k)]['centre_covered'] for r in selected),
                full_box=sum(r['prior_windows'][str(k)]['full_box_covered'] for r in selected),
                actual_windows=sum(r['prior_windows'][str(k)]['actual_windows'] for r in selected)) for k in [1, 2, 3]})
    return summaries


def prepare():
    app, g, training, frozen = parent()
    assert not (PARENT / 'result.json').exists() and not (PARENT / 'shard0/receipt.json').exists() and not (PARENT / 'shard1/receipt.json').exists()
    inventory = read(INVENTORY / 'result.json')
    assert sha(INVENTORY / 'geometry_events.json') == inventory['events_sha256']
    events = read(INVENTORY / 'geometry_events.json')['events']
    assert {(e['sequence'], e['frame']) for e in events} == {(e['sequence'], e['frame']) for e in frozen['events']}
    rows = [geometry_row(e, app, g) for e in events]
    assert len(rows) == 216 and sum(r['grid_windows'] for r in rows) == 10907
    ROOT.mkdir()
    spec = dict(status='frozen_before_M70_model_outputs', observed_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        parent_source_sha256=M70_SHA, parent_spec_sha256=PARENT_SHA, head_sha256=frozen['head_sha256'],
        inventory_result_sha256=sha(INVENTORY / 'result.json'), geometry_events_sha256=sha(INVENTORY / 'geometry_events.json'),
        budgets=[1, 2, 3], primary='All216 events at common budget min(3, category_route_length, empty_route_length); report H10 crop-out with GT centre in image and healthy strata.',
        fixed_budget_sensitivity='For each k in1,2,3 use only events where both text routes supply at least k unique windows. Report all excluded events; never pad or duplicate windows.',
        geometry_control='Rank the same grid windows by squared distance from the previous public bbox centre; exact ties use original grid index. No GT, current image features, or word input.',
        fine_conditions=['category', 'empty'], routing_conditions=['category', 'empty', 'prior'],
        primary_response='raw NMS top10; report Hann and dense capacities as predefined additional readouts.',
        contrasts=['category_route/category_fine vs empty_route/category_fine: routing words',
                   'category_route/category_fine vs category_route/empty_fine: fine words',
                   'category_route/category_fine vs empty_route/empty_fine: combined content',
                   'category_route/category_fine vs prior_route/category_fine: extra observation selection over history geometry'],
        stratification=STRATA, denominators='Both event-weighted and sequence-equal; paired gains/losses, not independent-frame significance.',
        cost='Route and fine text variants receive identical numbers of fine windows. Category/empty need one coarse pass; prior needs no coarse pass and is cheaper. Forward counts are operation budgets, not measured deployment FPS.',
        score_argmax='Diagnostic only; cross-window response calibration and deployed commit safety are not established.',
        geometry_scope='Use already sealed M70 GT geometry for offline resize/coverage accounting; nominal box size is not visible pixel support. No new labels or new model outputs.',
        new_tracking_calls=0, new_optimizer_steps=0, model_or_parent_queue_modified=False, public_evaluation_allowed=False,
        no_promotion_from_this_diagnostic=True, independent_model_review_pass=False)
    write(ROOT / 'spec.json', spec)
    write(ROOT / 'geometry_accounting.json', dict(status='completed_actual_M70_window_geometry_accounting',
        spec_sha256=sha(ROOT / 'spec.json'), events=rows, summaries=geometry_summary(rows),
        model_outputs_read=False, new_tracking_calls=0, new_GT_files_opened=False, subsequent_GT_used='Previously sealed M70 geometry only.'))
    print(json.dumps(dict(spec_sha256=sha(ROOT / 'spec.json'), geometry_result_sha256=sha(ROOT / 'geometry_accounting.json'), summaries=geometry_summary(rows)), indent=2))


def checked():
    spec = read(ROOT / 'spec.json'); app, g, training, frozen = parent()
    assert spec['source_sha256'] == sha(__file__) and spec['parent_spec_sha256'] == PARENT_SHA
    assert sha(INVENTORY / 'geometry_events.json') == spec['geometry_events_sha256']
    return spec, app, g, training, frozen


def seal():
    spec, app, g, training, frozen = checked()
    for name in ['controller', 'analysis', 'shard0', 'shard1']:
        assert (PARENT / (name + '.exit')).read_text().strip() == '0'
    result = read(PARENT / 'result.json')
    assert result['status'] == 'completed_fixed_state_cross_region_candidate_capacity'
    assert result['source_sha256'] == M70_SHA and result['spec_sha256'] == PARENT_SHA
    assert result['all_public_boxes_and_scores_exact'] and result['shadow_states_never_committed']
    assert result['event_metrics_sha256'] == sha(PARENT / 'event_metrics.json')
    sealed = {}; times = []
    for shard in [0, 1]:
        folder = PARENT / ('shard' + str(shard)); receipt = read(folder / 'receipt.json')
        assert sha(folder / 'receipt.json') == result['receipts'][str(shard)]
        assert receipt['spec_sha256'] == PARENT_SHA and receipt['source_sha256'] == M70_SHA and receipt['head_sha256'] == spec['head_sha256']
        assert receipt['status'] == 'completed_exact_public_replay_with_noncommitting_cross_region_probes'
        assert receipt['all_public_boxes_and_scores_exact'] and receipt['state_unchanged_after_each_shadow'] and receipt['local_maps_exact_to_public']
        assert [c['sequence'] for c in receipt['sequences']] == frozen['shard_sequences'][shard]
        times.append(dict(shard=shard, elapsed_seconds=receipt['elapsed_seconds'], public_calls=receipt['public_track_calls'], shadow_calls=receipt['shadow_forward_calls']))
        for case in receipt['sequences']:
            for event in case['events']:
                key = (case['sequence'], event['frame']); stem = key[0] + '__' + str(key[1]); assert key not in sealed
                meta_path = folder / (stem + '.json'); array_path = folder / (stem + '.npz')
                assert sha(meta_path) == event['metadata_sha256'] and sha(array_path) == event['array_sha256']
                meta = read(meta_path)
                assert meta['array_sha256'] == event['array_sha256'] and meta['spec_sha256'] == PARENT_SHA and not meta['GT_opened']
                assert (meta['sequence'], meta['frame']) == key and len(meta['windows']) == event['windows']
                sealed[key] = (meta, array_path)
    assert set(sealed) == {(e['sequence'], e['frame']) for e in frozen['events']}
    assert sum(r['public_calls'] for r in times) == 33108 and sum(r['shadow_calls'] for r in times) == 23110
    return spec, app, g, training, frozen, sealed, times


def readout(arrays, ious, fine, indices):
    values = ious[fine]; out = dict(dense=dict(best_iou=float(values[indices].max())))
    out['dense']['capacity'] = out['dense']['best_iou'] >= .5
    for response in ['raw', 'hann']:
        peaks = arrays[fine + '_nms_' + response][indices]
        local_iou = values[np.asarray(indices)[:, None], peaks]
        scores = arrays[fine + '_' + response][np.asarray(indices)[:, None], peaks]
        best = float(local_iou.max()); chosen = float(local_iou.reshape(-1)[int(scores.argmax())])
        out[response] = dict(best_iou=best, capacity=best >= .5, chosen_iou=chosen, correct=chosen >= .5, severe=chosen <= .1)
    return out


def paired(rows, left, right, response, metric):
    changes = [float(r['readouts'][left][response][metric]) - float(r['readouts'][right][response][metric]) for r in rows]
    by_sequence = {}
    for r, delta in zip(rows, changes): by_sequence.setdefault(r['sequence'], []).append(delta)
    return dict(events=len(rows), sequences=len(by_sequence), increases=sum(d > 0 for d in changes), decreases=sum(d < 0 for d in changes),
        ties=sum(d == 0 for d in changes), event_mean_delta=float(np.mean(changes)) if changes else None,
        sequence_equal_mean_delta=float(np.mean([np.mean(x) for x in by_sequence.values()])) if by_sequence else None,
        per_sequence_mean_delta={name: float(np.mean(values)) for name, values in sorted(by_sequence.items())})


def analyze():
    spec, app, g, training, frozen, sealed, times = seal()
    # Every response family has now sealed. Only this phase accesses GT and event tags.
    labels = {(e['sequence'], e['frame']): e for e in read(INVENTORY / 'geometry_events.json')['events']}
    groundtruth = {}
    for case in frozen['cases']:
        path = Path(training['dataset_root']) / case['sequence'] / 'groundtruth.txt'
        assert sha(path) == case['gt_sha256']
        groundtruth[case['sequence']] = np.loadtxt(path, delimiter=',').reshape(-1, 4)
    old_metrics = {(r['sequence'], r['frame']): r for r in read(PARENT / 'event_metrics.json')['events']}
    records = []; route_counts = Counter()
    for key, (meta, array_path) in sorted(sealed.items()):
        event = labels[key]; gt = groundtruth[key[0]][key[1]]; assert gt.tolist() == event['GT_bbox']
        windows = app.windows(meta['previous_bbox'], meta['image_shape'][1], meta['image_shape'][0], g)
        assert windows == meta['windows'] and meta['previous_bbox'] == event['arms']['null']['previous_bbox']
        n = len(windows); order = prior_order(windows, meta['previous_bbox'])
        with np.load(array_path) as arrays:
            ious = {}; routes = {}
            centres = np.asarray([[w['rectangle'][0] + w['rectangle'][2] / 2, w['rectangle'][1] + w['rectangle'][3] / 2] for w in windows[3:]])
            for fine in spec['fine_conditions']:
                boxes = arrays[fine + '_boxes']; assert boxes.shape == (n, 256, 4) and np.isfinite(boxes).all()
                ious[fine] = app.overlaps(boxes, gt, np)
                route = []
                for peak in arrays[fine + '_nms_raw'][2]:
                    box = boxes[2, int(peak)]; centre = box[:2] + box[2:] / 2
                    index = int(np.argmin(np.square(centres - centre).sum(1))) + 3
                    if index not in route: route.append(index)
                    if len(route) == 3: break
                assert route == meta['routes'][fine] and 1 <= len(route) <= 3
                routes[fine] = route
            route_counts[str(len(routes['category'])) + '/' + str(len(routes['empty']))] += 1
            # Recompute original unequal-budget readouts first and compare them to M70.
            for fine in spec['fine_conditions']:
                for route_name in spec['fine_conditions']:
                    values = readout(arrays, ious, fine, routes[route_name])
                    for response, value in values.items():
                        old = old_metrics[key]['readouts'][fine + '/' + route_name + '_route3/' + response]
                        assert value['best_iou'] == old['best_iou'] and value['capacity'] == old['correct_candidate_exists']
                        if response != 'dense': assert value['chosen_iou'] == old['score_argmax_iou']
            common = min(len(x) for x in routes.values())
            cohorts = [('common', common)] + [('k' + str(k), k) for k in [1, 2, 3] if common >= k]
            for cohort, k in cohorts:
                choices = {name: ids[:k] for name, ids in routes.items()}; choices['prior'] = order[:k]
                assert all(len(ids) == len(set(ids)) == k for ids in choices.values())
                row = dict(sequence=key[0], frame=key[1], tags=event['tags'], cohort=cohort, fine_windows=k,
                    null_local_centre_inside=event['arms']['null']['local']['centre_inside'], GT_centre_inside_image=event['GT_vs_image']['centre_inside'],
                    selected_windows=choices, coverage={}, readouts={})
                for route_name, ids in choices.items():
                    covered = [g.coverage(windows[i]['rectangle'], gt) for i in ids]
                    row['coverage'][route_name] = dict(centre=any(v['centre_inside'] for v in covered), full_box=any(v['full_box_inside'] for v in covered))
                    for fine in spec['fine_conditions']: row['readouts'][route_name + '/' + fine] = readout(arrays, ious, fine, ids)
                records.append(row)
    contrasts = {'route_content': ('category/category', 'empty/category'), 'fine_content': ('category/category', 'category/empty'),
        'combined_content': ('category/category', 'empty/empty'), 'route_vs_history_geometry': ('category/category', 'prior/category')}
    summaries = {}
    for cohort in ['common', 'k1', 'k2', 'k3']:
        rows = [r for r in records if r['cohort'] == cohort]; summaries[cohort] = {}
        included = {(r['sequence'], r['frame']) for r in rows}
        for group in STRATA:
            points = [r for r in rows if member(r, group)]
            strata_events = [e for e in read(ROOT / 'geometry_accounting.json')['events'] if member(e, group)]
            summary = dict(events=len(points), excluded_events=[dict(sequence=r['sequence'], frame=r['frame']) for r in strata_events if (r['sequence'], r['frame']) not in included],
                fine_forward_calls_per_condition=sum(r['fine_windows'] for r in points), coarse_calls_per_text_route=len(points), prior_coarse_calls=0,
                coverage={name: {m: sum(r['coverage'][name][m] for r in points) for m in ['centre', 'full_box']} for name in ['category', 'empty', 'prior']},
                contrasts={})
            for name, (left, right) in contrasts.items():
                summary['contrasts'][name] = {response: {metric: paired(points, left, right, response, metric)
                    for metric in (['capacity', 'best_iou'] if response == 'dense' else ['capacity', 'best_iou', 'correct', 'severe', 'chosen_iou'])}
                    for response in ['raw', 'hann', 'dense']}
            summaries[cohort][group] = summary
    write(ROOT / 'matched_event_metrics.json', dict(spec_sha256=sha(ROOT / 'spec.json'), events=records))
    result = dict(status='completed_equal_window_budget_routing_attribution', observed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), spec_sha256=sha(ROOT / 'spec.json'), parent_result_sha256=sha(PARENT / 'result.json'),
        parent_event_metrics_sha256=sha(PARENT / 'event_metrics.json'), matched_metrics_sha256=sha(ROOT / 'matched_event_metrics.json'),
        original_events=len(sealed), route_count_pairs=dict(sorted(route_counts.items())), summaries=summaries,
        actual_parent_shard_times=times, parent_times_include_replay_copying_serialization_and_IO=True,
        deployment_FPS_measured=False, new_tracking_calls=0, new_optimizer_steps=0, independent_model_review_pass=False,
        public_evaluation_allowed=False, hypothesis_scope='Same frozen model and protected history. No recovered trajectories, causal trigger, wrong-category control, or promotion claim.')
    write(ROOT / 'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'summaries'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'check', 'analyze']); args = parser.parse_args()
    {'prepare': prepare, 'check': checked, 'analyze': analyze}[args.action]()
