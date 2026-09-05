"""Run both fixed M52 heads through complete causal recursion and compare them."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=['control', 'mixed'])
    parser.add_argument('--shard', type=int, choices=[0, 1])
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root/'spec.json').read_text())
    parent = Path(plan['source_root'])
    spec = json.loads((parent/'spec.json').read_text())
    repo = Path(spec['repository'])
    sys.path.insert(0, str(repo))
    from tools.train_sttrack_m52 import sha, check_sources
    from tools.audit_sttrack_m43 import independent_overlap
    check_sources(root)
    binding = json.loads((root/'training_binding.json').read_text())
    assert sha(parent/'inference_inputs.json') == spec['inference_inputs_sha256']
    cases = {c['sequence']: c for c in json.loads((parent/'inference_inputs.json').read_text()) if c['split'] == 'development'}
    assert len(cases) == 22 and set(sum(binding['shards'], [])) == set(cases)
    assert sum(len(s) for s in binding['shards']) == 22
    training = {}
    for arm in ['control', 'mixed']:
        result = json.loads((root/arm/'training_result.json').read_text())
        assert result['status'] == 'complete' and result['spec_sha256'] == sha(root/'spec.json')
        assert result['optimizer_steps'] == 1900 and result['parameters'] == 448739 and result['reload_logits_exact']
        assert result['training_binding_sha256'] == sha(root/'training_binding.json')
        assert result['data_audit_sha256'] == sha(root/'data_audit.json')
        assert sha(root/arm/'geometry_final.pth') == result['checkpoint_sha256']
        training[arm] = result
    assert training['control']['initial_state_sha256'] == training['mixed']['initial_state_sha256']
    assert training['control']['logical_sample_order_sha256'] == training['mixed']['logical_sample_order_sha256']
    contract = json.loads((root/'runtime_contract.json').read_text())
    assert contract['status'] == 'PASS' and contract['spec_sha256'] == sha(root/'spec.json')
    for arm in training:
        assert contract['arms'][arm]['checkpoint_sha256'] == training[arm]['checkpoint_sha256']
    if args.analyze:
        assert args.arm is None and args.shard is None
        receipts = {}
        for arm in ['control', 'mixed']:
            receipts[arm] = []
            for shard in [0, 1]:
                assert (root/f'recursive_{arm}_s{shard}.exit').read_text().strip() == '0'
                receipt = json.loads((root/arm/f'shard{shard}_receipt.json').read_text())
                assert receipt['status'] == 'complete' and receipt['spec_sha256'] == sha(root/'spec.json')
                assert receipt['checkpoint_sha256'] == training[arm]['checkpoint_sha256']
                receipts[arm].extend(receipt['sequences'])
            assert len(receipts[arm]) == 22 and {r['sequence'] for r in receipts[arm]} == set(cases)
        old = json.loads((parent/'recursive_result.json').read_text())
        assert sha(parent/'recursive_result.json') == binding['native_recursive_result_sha256']
        paired = json.loads((Path(plan['policy_root'])/'recursive_result.json').read_text())
        assert sha(Path(plan['policy_root'])/'recursive_result.json') == binding['m45_recursive_result_sha256']
        baseline = defaultdict(list)
        for path, digest in spec['baseline_trace_sha256'].items():
            assert sha(path) == digest
            for row in json.loads(Path(path).read_text())['rows']:
                if row['sequence'] in cases:
                    baseline[row['sequence']].append(row)
        gt_by_sequence, boxes_by_sequence = {}, {}
        for name, case in cases.items():
            base = sorted(baseline[name], key=lambda r: r['frame_index'])
            assert [r['frame_index'] for r in base] == list(range(case['frames']))
            boxes = np.asarray([r['public_bbox'] for r in base])
            gt = np.loadtxt(Path(spec['dataset_root'])/name/'groundtruth.txt', delimiter=',')[:case['frames']]
            assert len(gt) == len(boxes)
            _, metrics = independent_overlap(boxes, gt)
            for key, value in metrics.items():
                assert math.isclose(value, old['per_sequence']['default'][name][key], rel_tol=1e-12, abs_tol=1e-10)
            gt_by_sequence[name], boxes_by_sequence[name] = gt, boxes
        per, totals, gates, details, table = {}, {}, {}, {}, []
        rule = plan['performance_gate']
        for arm in ['control', 'mixed']:
            per[arm], details[arm] = {}, []
            for item in receipts[arm]:
                name = item['sequence']
                path = root/arm/'recursive'/(name+'.json')
                assert sha(path) == item['sha256']
                data = json.loads(path.read_text()); rows = data['rows']
                assert data['sequence'] == name and [r['frame'] for r in rows] == list(range(cases[name]['frames']))
                boxes = np.asarray([r['bbox'] for r in rows])
                assert np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all()
                assert np.array_equal(boxes[0], cases[name]['init_bbox'])
                assert all(math.isfinite(r['score']) and 0 <= r['choice'] < 10 and (not r['none'] or r['choice'] == 0) for r in rows)
                values, metrics = independent_overlap(boxes, gt_by_sequence[name])
                metrics['low_iou_frames'] = int(metrics['low_iou_frames'])
                per[arm][name] = metrics
                changed = [i for i, r in enumerate(rows) if r['choice'] != 0]
                assert len(changed) == item['changes']
                first = changed[0] if changed else len(rows)
                assert np.array_equal(boxes[:first], boxes_by_sequence[name][:first]), (arm, name)
                base = old['per_sequence']['default'][name]
                table.append(dict(arm=arm, sequence=name, **metrics, changes=len(changed),
                                  mean_iou_gain=metrics['mean_iou']-base['mean_iou'],
                                  low_frame_delta=metrics['low_iou_frames']-base['low_iou_frames'],
                                  episode_delta=metrics['failure_episodes']-base['failure_episodes']))
                details[arm].append(dict(sequence=name, first_override=first if changed else None,
                                         changes=len(changed), native_prefix_exact=True, trajectory_sha256=sha(path)))
            total = {key: sum(x[key] for x in per[arm].values()) for key in ['valid_frames', 'iou_sum', 'low_iou_frames', 'failure_episodes']}
            total['mean_iou'] = total['iou_sum']/total['valid_frames']
            total['macro_sequence_mean_iou'] = float(np.mean([x['mean_iou'] for x in per[arm].values()]))
            totals[arm] = total
            positive = sum(x['mean_iou'] > old['per_sequence']['default'][n]['mean_iou'] for n, x in per[arm].items())
            broken = sorted(n for n, x in per[arm].items() if old['per_sequence']['default'][n]['failure_episodes'] == 0 and x['failure_episodes'] > 0)
            base = old['aggregates']['default']
            gates[arm] = dict(mean_iou=total['mean_iou'] >= base['mean_iou']+rule['mean_iou_gain_at_least'],
                              fewer_low_frames=total['low_iou_frames'] < base['low_iou_frames'],
                              no_episode_increase=total['failure_episodes'] <= base['failure_episodes'],
                              sequence_coverage=positive >= rule['positive_sequences_at_least'],
                              successful_sequence_protection=not broken)
            details[arm] = dict(first_overrides=details[arm], positive_sequences=positive, new_failure_sequences=broken)
        incremental = dict(mean_iou=totals['mixed']['mean_iou'] > totals['control']['mean_iou'],
                           low_frames=totals['mixed']['low_iou_frames'] <= totals['control']['low_iou_frames'],
                           episodes=totals['mixed']['failure_episodes'] <= totals['control']['failure_episodes'])
        data_pass = all(gates['mixed'].values()) and all(incremental.values())
        control_pass = all(gates['control'].values())
        selected = 'mixed' if data_pass else ('control' if control_pass else None)
        check_sources(root)
        result = dict(status='complete', integrity_pass=True, primary='policy_state_data', primary_pass=data_pass,
                      gates=gates, incremental_gates=incremental, extra_training_control_pass=control_pass,
                      advancing_arm=selected, aggregates=dict(default=old['aggregates']['default'], m45=paired['aggregates']['m45'], **totals),
                      per_sequence=per, details=details, frames_per_arm=33130, total_frames=66260,
                      matched_initialization_and_order=True, native_baseline_recomputed=True,
                      checkpoint_sha256={a: r['checkpoint_sha256'] for a, r in training.items()},
                      training_result_sha256={a: sha(root/a/'training_result.json') for a in training},
                      spec_sha256=sha(root/'spec.json'), training_binding_sha256=sha(root/'training_binding.json'),
                      data_audit_sha256=sha(root/'data_audit.json'),
                      scope='Repeated Train development. Data attribution requires mixed improvement over paired control; a qualifying control is reported as extra-training benefit only.')
        (root/'recursive_result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        with (root/'per_sequence.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        print(json.dumps({k: v for k, v in result.items() if k not in ['per_sequence', 'details']}, indent=2))
        return
    assert args.arm in ['control', 'mixed'] and args.shard in [0, 1]
    from lib.config.sttrack.config import cfg, update_config_from_file
    from lib.test.tracker.sttrack_candidate_set import STTrackCandidateSet
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1)
    update_config_from_file(str(repo/'experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params = SimpleNamespace(cfg=cfg, checkpoint=spec['checkpoint'], template_factor=2., template_size=128,
                             search_factor=4., search_size=256, save_all_boxes=False, debug=0)
    checkpoint = root/args.arm/'geometry_final.pth'
    saved = torch.load(checkpoint, map_location='cpu')
    assert saved['m52_spec_sha256'] == sha(root/'spec.json') and saved['arm'] == args.arm
    assert saved['optimizer_steps'] == 1900 and saved['base_checkpoint_sha256'] == spec['checkpoint_sha256']
    tracker = STTrackCandidateSet(params, checkpoint)
    folder = root/args.arm/'recursive'; folder.mkdir(exist_ok=True)
    receipts = []; started = time.time()
    for name in binding['shards'][args.shard]:
        case = cases[name]; data = Path(spec['dataset_root'])/name
        def image_at(i):
            return get_rgbd_frame(str(data/'color'/f'{i+1:08d}.jpg'), str(data/'depth'/f'{i+1:08d}.png'),
                                  dtype='rgbcolormap', depth_clip=True)
        tracker.initialize(image_at(0), dict(init_bbox=case['init_bbox']))
        rows = [dict(frame=0, bbox=case['init_bbox'], score=1., choice=0, none=False)]
        for frame in range(1, case['frames']):
            out = tracker.track(image_at(frame))
            rows.append(dict(frame=frame, bbox=out['target_bbox'], score=float(out['best_score']),
                             choice=out['association_candidate'], none=out['association_none']))
        path = folder/(name+'.json')
        path.write_text(json.dumps(dict(sequence=name, rows=rows), allow_nan=False)+'\n')
        item = dict(sequence=name, frames=len(rows), changes=sum(r['choice'] != 0 for r in rows),
                    sha256=sha(path), elapsed_seconds=time.time()-started)
        receipts.append(item); print(json.dumps(dict(arm=args.arm, **item)), flush=True)
    check_sources(root)
    (root/args.arm/f'shard{args.shard}_receipt.json').write_text(json.dumps(dict(status='complete', sequences=receipts,
        checkpoint_sha256=sha(checkpoint), spec_sha256=sha(root/'spec.json'), ground_truth_files_opened=False), indent=2)+'\n')


if __name__ == '__main__': main()
