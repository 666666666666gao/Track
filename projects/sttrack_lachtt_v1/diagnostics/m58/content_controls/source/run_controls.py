"""Fixed-text-head content interventions after the frozen M58 development gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path('/root/autodl-tmp/sttrack_m58_content_controls_v1_20260906')
PARENT = Path('/root/autodl-tmp/sttrack_m58_semantic_spatial_v2_20260906')
INTERFACE = Path('/root/autodl-tmp/sttrack_m58_evaluation_preparation_20260906')
TRAINING_SHA = 'c592109d10579efac4dce5d3dd6f4881900c2c264acc3db3b63a6021855ea3b7'
RECURSIVE_SHA = '29aeeaa353d1d898f4793744375238b2212f865557df64efd5f981c1d62c027e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def parent_plans():
    assert sha(PARENT / 'training_spec.json') == TRAINING_SHA
    assert sha(PARENT / 'recursive_spec.json') == RECURSIVE_SHA
    training = json.loads((PARENT / 'training_spec.json').read_text())
    recursive = json.loads((PARENT / 'recursive_spec.json').read_text())
    assert sha(PARENT / 'integration.json') == training['integration_sha256']
    integration = json.loads((PARENT / 'integration.json').read_text())
    for name, digest in integration['source_sha256'].items():
        assert sha(PARENT / 'code' / name) == digest
    for name, key in [('run_recursive.py', 'runner_sha256'), ('recursive_metric.py', 'metric_sha256')]:
        assert sha(PARENT / name) == recursive[key]
    return training, recursive, integration


def plans():
    spec = json.loads((ROOT / 'spec.json').read_text())
    parent_plans()
    for name, digest in spec['source_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, digest in spec['interface_sha256'].items():
        assert sha(INTERFACE / name) == digest
    for name, digest in spec['bank_sha256'].items():
        assert sha(ROOT / (name + '.pt')) == digest
    assert sha(INTERFACE / 'text_protocol.json') == spec['text_protocol_sha256']
    assert sha(ROOT / 'mappings.json') == spec['mappings_sha256']
    return spec


def parent_ready():
    spec = plans()
    assert (PARENT / 'recursive_analysis.exit').read_text().strip() == '0'
    result = json.loads((PARENT / 'recursive_result.json').read_text())
    assert result['status'] == 'complete_recursive_development'
    assert result['recursive_spec_sha256'] == RECURSIVE_SHA
    assert result['training_spec_sha256'] == TRAINING_SHA
    assert result['primary_pass'] and all(result['gates'].values())
    sys.path.insert(0, str(PARENT))
    from run_recursive import trained
    _, training, training_results = trained()
    return spec, training, result, training_results


def seal_bundle():
    spec, training, parent, results = parent_ready()
    _, _, integration = parent_plans()
    assert not (ROOT / 'bundle.json').exists()
    bundle = dict(architecture='semantic_spatial_v1', repository=str(PARENT / 'code'),
        configuration='experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml',
        source_sha256=integration['source_sha256'], interface_sha256=spec['interface_sha256'],
        base_checkpoint=training['native_checkpoint'], base_checkpoint_sha256=training['native_checkpoint_sha256'],
        adapter_checkpoint=str(PARENT / 'training/text/final.pth'),
        adapter_checkpoint_sha256=results['text']['final_checkpoint_sha256'],
        training_spec_sha256=TRAINING_SHA, use_text=True, seed=training['seed'],
        text_protocol_path=str(INTERFACE / 'text_protocol.json'),
        text_protocol_sha256=spec['text_protocol_sha256'])
    write(ROOT / 'bundle.json', bundle)
    for name in ['original'] + list(spec['controls']):
        write(ROOT / (name + '_plan.json'), dict(bundle_path=str(ROOT / 'bundle.json'),
            bundle_sha256=sha(ROOT / 'bundle.json'), text_bank_path=str(ROOT / (name + '.pt')),
            text_bank_sha256=spec['bank_sha256'][name]))
    write(ROOT / 'bundle_receipt.json', dict(status='candidate_bundle_sealed_after_parent_gate',
        bundle_sha256=sha(ROOT / 'bundle.json'), content_spec_sha256=sha(ROOT / 'spec.json'),
        parent_recursive_result_sha256=sha(PARENT / 'recursive_result.json'),
        head_sha256=bundle['adapter_checkpoint_sha256'],
        plan_sha256={n:sha(ROOT / (n + '_plan.json')) for n in ['original'] + list(spec['controls'])},
        formal_three_dataset_promotion=False))


def runtime(name):
    spec, training, parent, results = parent_ready()
    receipt = json.loads((ROOT / 'bundle_receipt.json').read_text())
    assert receipt['content_spec_sha256'] == sha(ROOT / 'spec.json')
    assert receipt['bundle_sha256'] == sha(ROOT / 'bundle.json')
    assert receipt['parent_recursive_result_sha256'] == sha(PARENT / 'recursive_result.json')
    assert receipt['head_sha256'] == results['text']['final_checkpoint_sha256']
    assert sha(ROOT / (name + '_plan.json')) == receipt['plan_sha256'][name]
    sys.path.insert(0, str(INTERFACE))
    from semantic_runtime import checked_plan, make_tracker, text_bank
    plan, bundle = checked_plan(ROOT / (name + '_plan.json'))
    return spec, training, make_tracker(bundle), text_bank(plan, bundle)


def load_frame(folder, index):
    from lib.train.dataset.depth_utils import get_rgbd_frame
    stem = '%08d' % (index + 1)
    rgb = folder / 'color' / (stem + '.jpg')
    image = get_rgbd_frame(str(rgb), str(folder / 'depth' / (stem + '.png')),
                           dtype='rgbcolormap', depth_clip=True)
    return rgb, image


def prefix_parity():
    import numpy as np
    spec, training, tracker, bank = runtime('original')
    parent_receipt = json.loads((PARENT / 'text_recursive_receipt.json').read_text())
    assert parent_receipt['head_sha256'] == json.loads((ROOT / 'bundle.json').read_text())['adapter_checkpoint_sha256']
    sealed = {r['sequence']:r for r in parent_receipt['sequences']}
    reports = []
    started = time.time()
    for name in spec['prefix_parity_sequences']:
        case = next(c for c in spec['cases'] if c['sequence'] == name)
        prediction_path = PARENT / 'recursive/text' / (name + '.json')
        assert sha(prediction_path) == sealed[name]['sha256']
        expected = json.loads(prediction_path.read_text())['rows']
        assert len(expected) == case['frames']
        folder = Path(training['dataset_root']) / name
        rgb, image = load_frame(folder, 0)
        tracker.initialize(image, bank.info(rgb, case['init_bbox']))
        assert list(tracker.state) == expected[0]['bbox'] == case['init_bbox']
        box_error = score_error = 0.
        updates = 0
        for i in range(1, spec['prefix_parity_frames']):
            old_template = tracker.z_dict[1]
            _, image = load_frame(folder, i)
            output = tracker.track(image)
            box_error = max(box_error, float(np.abs(np.asarray(output['target_bbox']) - expected[i]['bbox']).max()))
            score_error = max(score_error, abs(float(output['best_score']) - expected[i]['score']))
            updates += tracker.z_dict[1] is not old_template
        assert box_error <= 1e-4 and score_error <= 1e-6
        reports.append(dict(sequence=name, frames=spec['prefix_parity_frames'],
            max_bbox_error_px=box_error, max_score_error=score_error, template_updates=updates))
    write(ROOT / 'prefix_parity_result.json', dict(status='shared_runtime_prefix_parity_passed',
        bundle_sha256=sha(ROOT / 'bundle.json'), content_spec_sha256=sha(ROOT / 'spec.json'),
        reference_receipt_sha256=sha(PARENT / 'text_recursive_receipt.json'), sequences=reports,
        elapsed_seconds=time.time() - started, subsequent_gt_opened=False,
        full_sequence_reproduction_verified=False, trax_server_exchange_verified=False))


def run(name):
    spec, training, tracker, bank = runtime(name)
    parity = json.loads((ROOT / 'prefix_parity_result.json').read_text())
    assert parity['status'] == 'shared_runtime_prefix_parity_passed'
    assert parity['bundle_sha256'] == sha(ROOT / 'bundle.json')
    cases = [c for c in spec['cases'] if c['sequence'] in spec['controls'][name]['sequences']]
    output = ROOT / 'predictions' / name
    output.mkdir(parents=True)
    started = time.time()
    receipts = []
    for case in cases:
        folder = Path(training['dataset_root']) / case['sequence']
        rgb, image = load_frame(folder, 0)
        tracker.initialize(image, bank.info(rgb, case['init_bbox']))
        rows = [dict(frame=0, bbox=list(tracker.state), score=None)]
        for i in range(1, case['frames']):
            _, image = load_frame(folder, i)
            prediction = tracker.track(image)
            rows.append(dict(frame=i, bbox=list(prediction['target_bbox']), score=float(prediction['best_score'])))
        path = output / (case['sequence'] + '.json')
        write(path, dict(sequence=case['sequence'], control=name, rows=rows))
        item = dict(sequence=case['sequence'], frames=len(rows), sha256=sha(path), elapsed_seconds=time.time() - started)
        receipts.append(item)
        print(json.dumps(dict(control=name, **item)), flush=True)
    plans()
    write(ROOT / (name + '_receipt.json'), dict(status='complete', control=name,
        content_spec_sha256=sha(ROOT / 'spec.json'), bundle_sha256=sha(ROOT / 'bundle.json'),
        bank_sha256=spec['bank_sha256'][name], sequences=receipts,
        frames=sum(r['frames'] for r in receipts), subsequent_gt_opened=False,
        optimizer_steps=0, online_text_updates=False, elapsed_seconds=time.time() - started))


def aggregate(rows):
    totals = {key:sum(row[key] for row in rows.values()) for key in
              ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
    totals['mean_iou'] = totals['iou_sum'] / totals['valid_frames']
    totals['macro_sequence_mean_iou'] = sum(r['mean_iou'] for r in rows.values()) / len(rows)
    totals['sequences'] = len(rows)
    return totals


def analyze():
    import numpy as np
    spec, training, parent, results = parent_ready()
    for worker in [0, 1]:
        assert (ROOT / ('worker%d.exit' % worker)).read_text().strip() == '0'
    predicted = {}
    receipt_shas = {}
    for name in ['original'] + list(spec['controls']):
        original = name == 'original'
        receipt_path = PARENT / 'text_recursive_receipt.json' if original else ROOT / (name + '_receipt.json')
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'complete'
        if original:
            assert receipt['head_sha256'] == results['text']['final_checkpoint_sha256']
            assert receipt['recursive_spec_sha256'] == RECURSIVE_SHA
            selected = [c['sequence'] for c in spec['cases']]
        else:
            assert receipt['content_spec_sha256'] == sha(ROOT / 'spec.json')
            assert receipt['bundle_sha256'] == sha(ROOT / 'bundle.json')
            assert receipt['bank_sha256'] == spec['bank_sha256'][name]
            selected = spec['controls'][name]['sequences']
        assert [r['sequence'] for r in receipt['sequences']] == selected
        frame_key = 'total_frames' if original else 'frames'
        assert receipt[frame_key] == sum(c['frames'] for c in spec['cases'] if c['sequence'] in selected)
        predicted[name] = {}
        for row in receipt['sequences']:
            folder = PARENT / 'recursive/text' if original else ROOT / 'predictions' / name
            path = folder / (row['sequence'] + '.json')
            assert sha(path) == row['sha256']
            value = json.loads(path.read_text())
            case = next(c for c in spec['cases'] if c['sequence'] == row['sequence'])
            assert row['frames'] == case['frames']
            assert value['sequence'] == case['sequence']
            kind_key, kind_value = ('arm', 'text') if original else ('control', name)
            assert value[kind_key] == kind_value
            assert [r['frame'] for r in value['rows']] == list(range(case['frames']))
            boxes = np.asarray([r['bbox'] for r in value['rows']])
            assert np.array_equal(boxes[0], case['init_bbox'])
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            predicted[name][case['sequence']] = boxes
        receipt_shas[name] = sha(receipt_path)
    # All original/control predictions are sealed before this analysis opens GT.
    sys.path.insert(0, str(PARENT))
    from recursive_metric import statistics
    per_sequence = {name:{} for name in predicted}
    first_differences = {name:{} for name in spec['controls']}
    for case in spec['cases']:
        name = case['sequence']
        gt_path = Path(training['dataset_root']) / name / 'groundtruth.txt'
        assert sha(gt_path) == case['gt_sha256']
        gt = np.loadtxt(gt_path, delimiter=',').reshape(-1, 4)
        assert len(gt) == case['frames']
        for control in predicted:
            if name not in predicted[control]:
                continue  # The matched-caption-category arm has a frozen 14-case subset.
            per_sequence[control][name] = statistics(predicted[control][name], gt)
            if control != 'original':
                diff = np.max(np.abs(predicted[control][name] - predicted['original'][name]), axis=1)
                indices = np.flatnonzero(diff > 1e-4)
                first_differences[control][name] = dict(changed_frames=int(len(indices)),
                    first_bbox_difference_frame=int(indices[0]) if len(indices) else None)
        prior = parent['per_sequence']['text'][name]
        current = per_sequence['original'][name]
        for key in ['valid_frames', 'low_iou_frames', 'failure_episodes']:
            assert current[key] == prior[key]
        assert abs(current['iou_sum'] - prior['iou_sum']) < 1e-8
    aggregates = {name:aggregate(rows) for name, rows in per_sequence.items()}
    aggregates['original_for_attributes'] = aggregate({n:per_sequence['original'][n]
        for n in spec['controls']['attributes']['sequences']})
    comparisons = {}
    for control, settings in spec['controls'].items():
        original = aggregates['original_for_attributes'] if control == 'attributes' else aggregates['original']
        modified = aggregates[control]
        comparisons[control] = dict(sequences=modified['sequences'], valid_frames=modified['valid_frames'],
            original_minus_control_mean_iou=original['mean_iou'] - modified['mean_iou'],
            original_minus_control_low_frames=original['low_iou_frames'] - modified['low_iou_frames'],
            original_minus_control_H10=original['failure_episodes'] - modified['failure_episodes'],
            original_better_sequences=sum(per_sequence['original'][n]['mean_iou'] > per_sequence[control][n]['mean_iou'] + 1e-8 for n in settings['sequences']),
            original_worse_sequences=sum(per_sequence['original'][n]['mean_iou'] < per_sequence[control][n]['mean_iou'] - 1e-8 for n in settings['sequences']))
    gates = {name:comparisons[name]['original_minus_control_mean_iou'] >= threshold
             for name, threshold in spec['primary_content_gates'].items()}
    result = dict(status='completed_same_head_content_controls', observed_utc=datetime.now(timezone.utc).isoformat(),
        content_spec_sha256=sha(ROOT / 'spec.json'), bundle_sha256=sha(ROOT / 'bundle.json'),
        head_sha256=results['text']['final_checkpoint_sha256'], parent_result_sha256=sha(PARENT / 'recursive_result.json'),
        receipt_sha256=receipt_shas, aggregates=aggregates, per_sequence=per_sequence,
        comparisons=comparisons, first_bbox_differences=first_differences,
        gates=gates, primary_content_pass=all(gates.values()), attributes_descriptive_only=True,
        wrong_attribute_ground_truth_verified=False, independent_review_pass=False,
        scope='Reused DepthTrack Train development22; attribute arm is a frozen matched-caption-category subset.',
        next='Technical low22 entry validation before public evaluation' if all(gates.values()) else 'Stop this revision before public evaluation; diagnose fixed content effects')
    write(ROOT / 'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence', 'first_bbox_differences']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seal', action='store_true')
    mode.add_argument('--parity', action='store_true')
    mode.add_argument('--worker', type=int, choices=[0, 1])
    mode.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.seal:
        seal_bundle()
    elif args.parity:
        prefix_parity()
    elif args.analyze:
        analyze()
    else:
        spec = plans()
        for name in spec['workers'][str(args.worker)]:
            run(name)
