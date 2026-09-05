"""Audit frozen M44 weights and sealed recursive trajectories on CPU."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert (root / 'controller.exit').read_text().strip() == '0'
    spec = json.loads((root / 'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.audit_sttrack_m43 import independent_overlap, sha
    from tools.train_sttrack_m44 import tensor_sha
    from lib.models.sttrack.lachtt_candidate_set import CandidateSetAssociation

    torch.set_num_threads(1)
    binding = json.loads((root / 'training_binding.json').read_text())
    training = json.loads((root / 'training_result.json').read_text())
    result = json.loads((root / 'recursive_result.json').read_text())
    assert training['status'] == result['status'] == 'complete'
    assert training['primary'] == result['primary'] == spec['primary'] == 'geometry'
    assert binding['spec_sha256'] == training['spec_sha256'] == result['spec_sha256'] == sha(root / 'spec.json')
    assert result['training_result_sha256'] == sha(root / 'training_result.json')
    assert training['source_binding_sha256'] == sha(root / 'training_binding.json')
    for name, digest in {**spec['source_sha256'], **binding['source_sha256']}.items():
        assert sha(repo / name) == digest, name
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    assert sha(root / 'inference_inputs.json') == spec['inference_inputs_sha256']
    assert sha(root / 'training_labels.json') == spec['labels_sha256']
    inputs = json.loads((root / 'inference_inputs.json').read_text())
    cases = {c['sequence']: c for c in inputs if c['split'] == 'development'}
    assert len(cases) == binding['recursive_sequences']
    assert sum(c['frames'] for c in cases.values()) == binding['recursive_frames_per_arm']
    fit_names = {c['sequence'] for c in inputs if c['split'] == 'fit'}
    assert fit_names.isdisjoint(cases)
    collection = []
    for shard in [0, 1]:
        assert (root / f'collect_s{shard}.exit').read_text().strip() == '0'
        receipt = json.loads((root / f'shard{shard}_receipt.json').read_text())
        assert receipt['status'] == 'complete' and receipt['labels_opened'] is False
        assert receipt['source_unchanged'] and receipt['spec_sha256'] == sha(root / 'spec.json')
        collection.extend(receipt['sequences'])
    assert len(collection) == len({r['sequence'] for r in collection}) == spec['sequences']
    assert sum(r['events'] for r in collection) == spec['events']
    assert sum(r['frames'] for r in collection) == sum(spec['shard_frames'])
    for row in collection:
        assert sha(root / 'features' / (row['sequence'] + '.pt')) == row['feature_sha256']

    weights = {}
    seals = {}
    steps = spec['optimization']['epochs'] * math.ceil(binding['fit_events'] / spec['optimization']['batch_size'])
    for arm in spec['variants']:
        values = training['variants'][arm]
        assert values == json.loads((root / (arm + '_result.json')).read_text())
        path = root / (arm + '_final.pth')
        assert sha(path) == values['checkpoint_sha256']
        saved = torch.load(path, map_location='cpu')
        assert saved['variant'] == arm and saved['base_checkpoint_sha256'] == spec['checkpoint_sha256']
        assert saved['spec_sha256'] == sha(root / 'spec.json')
        assert saved['binding_sha256'] == values['source_binding_sha256'] == sha(root / 'training_binding.json')
        assert saved['epochs'] == values['epochs'] == spec['optimization']['epochs']
        assert saved['optimizer_steps'] == values['optimizer_steps'] == binding['optimizer_steps_per_arm'] == steps
        assert [r['epoch'] for r in values['losses']] == list(range(1, saved['epochs'] + 1))
        assert all(math.isfinite(r[k]) for r in values['losses'] for k in ['loss', 'identity_loss', 'matching_loss'])
        assert values['reload_logits_exact']
        assert values['fit']['events'] == binding['fit_events']
        assert values['development']['events'] == binding['development_events']
        torch.manual_seed(spec['optimization']['seed'])
        model = CandidateSetAssociation(arm == 'geometry')
        initial = {k: tensor_sha(v) for k, v in model.state_dict().items()}
        assert initial == values['initial_state_sha256']
        model.load_state_dict(saved['model'], strict=True)
        assert all(torch.isfinite(v).all() for v in saved['model'].values())
        changed = [k for k, v in saved['model'].items() if tensor_sha(v) != initial[k]]
        assert changed == values['changed_tensors'] and 'identity.weight' in changed
        assert sum(v.numel() for v in model.parameters()) == values['parameters']
        weights[arm] = dict(sha256=sha(path), parameters=values['parameters'], optimizer_steps=steps,
                            changed_tensors=changed, first_loss=values['losses'][0], final_loss=values['losses'][-1])
        receipt = json.loads((root / (arm + '_recursive_receipt.json')).read_text())
        assert receipt['status'] == 'complete' and receipt['source_unchanged']
        assert receipt['ground_truth_files_opened'] is False and receipt['checkpoint_sha256'] == sha(path)
        seals[arm] = {r['sequence']: r for r in receipt['sequences']}
        assert len(receipt['sequences']) == len(cases) and set(seals[arm]) == set(cases)
    a, b = [training['variants'][arm] for arm in spec['variants']]
    assert a['initial_state_sha256'] == b['initial_state_sha256']
    assert a['sample_order_sha256'] == b['sample_order_sha256']

    baseline = defaultdict(list)
    for path, digest in spec['baseline_trace_sha256'].items():
        assert sha(path) == digest
        for row in json.loads(Path(path).read_text())['rows']:
            if row['sequence'] in cases:
                baseline[row['sequence']].append(row)
    per = {arm: {} for arm in ['default'] + spec['variants']}
    details = []
    csvrows = []
    total = 0
    for name, case in cases.items():
        base = sorted(baseline[name], key=lambda r: r['frame_index'])
        assert [r['frame_index'] for r in base] == list(range(case['frames']))
        boxes0 = np.asarray([r['public_bbox'] for r in base], dtype=np.float64)
        gt = np.loadtxt(Path(spec['dataset_root']) / name / 'groundtruth.txt', delimiter=',')[:len(base)]
        assert len(gt) == len(base)
        values0, per['default'][name] = independent_overlap(boxes0, gt)
        sequence = dict(sequence=name, frames=len(base))
        for arm in spec['variants']:
            path = root / 'recursive' / (arm + '_' + name + '.json')
            assert sha(path) == seals[arm][name]['sha256']
            data = json.loads(path.read_text())
            rows = data['rows']
            assert data['sequence'] == name and data['arm'] == arm
            assert len(rows) == case['frames'] and [r['frame'] for r in rows] == list(range(len(rows)))
            boxes = np.asarray([r['bbox'] for r in rows], dtype=np.float64)
            assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
            assert all(math.isfinite(r['score']) and 0 <= r['choice'] < 10 and (not r['none'] or r['choice'] == 0) for r in rows)
            assert np.array_equal(boxes[0], np.asarray(case['init_bbox']))
            values, per[arm][name] = independent_overlap(boxes, gt)
            changed = [i for i, r in enumerate(rows) if r['choice'] != 0]
            assert len(changed) == seals[arm][name]['changes'] == result['per_sequence'][arm][name]['changes']
            first = changed[0] if changed else len(rows)
            prefix_error = float(np.abs(boxes[:first] - boxes0[:first]).max())
            assert prefix_error == 0., (name, arm, first, prefix_error)
            item = dict(sequence=name, arm=arm, changes=len(changed), trajectory_sha256=sha(path),
                        first_override_frame_zero_based=first if changed else None, prefix_max_bbox_error_px=prefix_error)
            if changed:
                start, end = first + 1, min(first + 11, len(rows))
                valid = np.isfinite(values0[start:end])
                item.update(first_choice=rows[first]['choice'], first_score=rows[first]['score'],
                            first_default_iou=float(values0[first]) if np.isfinite(values0[first]) else None,
                            first_selected_iou=float(values[first]) if np.isfinite(values[first]) else None,
                            next10_valid_frames=int(valid.sum()),
                            next10_mean_gain=float((values[start:end][valid] - values0[start:end][valid]).mean()) if valid.any() else None)
            details.append(item)
            sequence[arm + '_changes'] = len(changed)
            total += len(rows)
        for arm in per:
            for key, value in per[arm][name].items():
                assert math.isclose(value, result['per_sequence'][arm][name][key], rel_tol=1e-12, abs_tol=1e-10), (name, arm, key)
                sequence[arm + '_' + key] = value
        csvrows.append(sequence)

    aggregates = {}
    for arm, rows in per.items():
        sums = {k: sum(x[k] for x in rows.values()) for k in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
        sums['mean_iou'] = sums['iou_sum'] / sums['valid_frames']
        sums['macro_sequence_mean_iou'] = float(np.mean([x['mean_iou'] for x in rows.values()]))
        for key, value in sums.items():
            assert math.isclose(value, result['aggregates'][arm][key], rel_tol=1e-12, abs_tol=1e-10), (arm, key)
        aggregates[arm] = sums
    rule = spec['recursive_performance_gate']
    base = aggregates['default']
    for arm in spec['variants']:
        value = aggregates[arm]
        positive = sum(per[arm][n]['mean_iou'] > per['default'][n]['mean_iou'] for n in cases)
        broken = sorted(n for n in cases if per['default'][n]['failure_episodes'] == 0 and per[arm][n]['failure_episodes'] > 0)
        gates = dict(mean_iou=value['mean_iou'] >= base['mean_iou'] + rule['mean_iou_gain_at_least'],
                     fewer_low_frames=value['low_iou_frames'] < base['low_iou_frames'],
                     no_episode_increase=value['failure_episodes'] <= base['failure_episodes'],
                     sequence_coverage=positive >= rule['positive_sequences_at_least'],
                     successful_sequence_protection=not broken)
        assert result['comparisons'][arm] == dict(gates=gates, pass_gate=all(gates.values()), positive_sequences=positive, new_failure_sequences=broken)
    assert result['primary_pass'] == result['comparisons']['geometry']['pass_gate']
    assert math.isclose(result['geometry_incremental_mean_gain'], aggregates['geometry']['mean_iou'] - aggregates['appearance']['mean_iou'], abs_tol=1e-12)
    audit = dict(status='complete', integrity_pass=True, sequences=len(cases), trajectory_files=len(details),
                 frames_including_initialization=total, primary_pass=result['primary_pass'],
                 source_and_weights_unchanged=True, independent_metrics_and_gates_match=True,
                 exact_default_prefix_before_first_override=True, first_overrides=details, weights=weights,
                 collection_sequences=len(collection), collection_pairs=sum(r['events'] for r in collection),
                 collection_frames=sum(r['frames'] for r in collection), feature_bytes=sum(r['bytes'] for r in collection),
                 maximum_bbox_error_px=max(r['max_bbox_error_px'] for r in collection),
                 maximum_score_error=max(r['max_score_error'] for r in collection),
                 default_template_writes=sum(r['template_updates'] for r in collection),
                 result_sha256=sha(root / 'recursive_result.json'), auditor_sha256=sha(__file__),
                 independent_overlap_source_sha256=sha(repo / 'tools/audit_sttrack_m43.py'),
                 scope='Previously used Train development sequences only. Scalar IoU and H10 recomputation reuse the verified M43 auditor; this is not a second-person review or a public metric. No new model selection or gate.')
    (root / 'terminal_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    with (root / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csvrows[0]))
        writer.writeheader()
        writer.writerows(csvrows)
    print(json.dumps(dict(status='complete', integrity_pass=True, frames=total, primary_pass=result['primary_pass'], result_sha256=audit['result_sha256'])), flush=True)


if __name__ == '__main__':
    main()
