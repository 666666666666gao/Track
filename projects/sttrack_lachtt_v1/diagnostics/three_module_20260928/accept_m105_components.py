"""CPU arithmetic acceptance for M105 outputs; never run or alter a model."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).parent
ROOT = HERE / 'm105_completed'
VARIANTS = ['parent', 'full', 'center_only', 'size_only']


def read(path):
    return json.loads(path.read_text())


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def overlap(box, target):
    box = list(map(f32, box)); target = list(map(f32, target))
    extent = [max(f32(min(f32(box[i]+box[i+2]), f32(target[i]+target[i+2])) - max(box[i], target[i])), 0.) for i in (0, 1)]
    intersection = f32(extent[0]*extent[1])
    union = f32(f32(f32(box[2]*box[3])+f32(target[2]*target[3]))-intersection)
    return f32(intersection/union)


def summarize(rows):
    values = {}
    for name in VARIANTS:
        parent = [r['selected_iou']['parent'] for r in rows]
        current = [r['selected_iou'][name] for r in rows]
        before_best = [max(r['candidate_iou']['parent']) for r in rows]
        after_best = [max(r['candidate_iou'][name]) for r in rows]
        values[name] = dict(events=len(rows), selected_correct=sum(v>=.5 for v in current),
                            selected_mean_iou=sum(current)/len(rows),
                            parent_rescues=sum(a<.5<=b for a,b in zip(parent,current)),
                            parent_breaks=sum(b<.5<=a for a,b in zip(parent,current)),
                            oracle_correct=sum(v>=.5 for v in after_best),
                            original_miss_hit=sum(a<.5<=b for a,b in zip(before_best,after_best)),
                            original_hit_miss=sum(b<.5<=a for a,b in zip(before_best,after_best)))
    return values


def main():
    driver = read(ROOT/'driver.json')
    assert driver['exit_code'] == 0 and (ROOT/'driver.exit').read_text().strip() == '0'
    assert [(r['split'],r['gpu'],r['exit_code']) for r in driver['runs']] == [('fit',0,0),('development',1,0)]
    original = read(HERE/'m104_completed/train/result.json')
    inputs = HERE/'m105_inputs'
    labels = read(inputs/'training_labels.json')
    assert hashlib.sha256((inputs/'training_labels.json').read_bytes()).hexdigest() == read(inputs/'preparation.json')['training_labels_sha256']
    accepted = {}; component_changes = {}; selected_boxes_checked = 0
    for split, n in [('fit',2544),('development',495)]:
        assert (ROOT/(split+'.exit')).read_text().strip() == '0'
        result = read(ROOT/split/'result.json')
        rows = [json.loads(line) for line in (ROOT/split/'events.jsonl').read_text().splitlines()]
        old = [json.loads(line) for line in (HERE/'m104_completed/train'/ (split+'_events.jsonl')).read_text().splitlines()]
        assert result['status'] == 'complete_same_final_geometry_components'
        assert len(rows) == result['events'] == len(old) == n
        assert result['variants'] == VARIANTS
        assert result['optimizer_steps'] == 0 and not result['checkpoint_saved']
        assert result['refiner_weight_sha256'] == original['final_weight_sha256']
        assert result['parent_weight_sha256'] == original['parent_weight_sha256']
        for flag in ['full_replay_original_rows_and_summaries_exact','actual_all_zero_geometry_exact','parent_and_refiner_parameters_buffers_unchanged','no_GT_in_geometry_forward','no_tracker_memory_semantic_or_public_action']:
            assert result[flag] is True
        groups = defaultdict(list); joints = defaultdict(list)
        for r, previous in zip(rows, old):
            assert (r['key'],r['strata'],r['selected'],r['native_iou']) == (previous['key'],previous['strata'],previous['selected'],previous['native_iou'])
            assert r['selected_iou']['parent'] == previous['parent_iou']
            assert r['selected_iou']['full'] == previous['selected_iou']
            assert max(r['candidate_iou']['parent']) == previous['native_top10_iou']
            assert max(r['candidate_iou']['full']) == previous['oracle_iou']
            target = labels[r['key']]['current']
            assert labels[r['key']]['split'] == split and target is not None
            for variant in VARIANTS:
                assert len(r['candidate_iou'][variant]) == 10
                assert r['selected_iou'][variant] == r['candidate_iou'][variant][r['selected']]
                assert overlap(r['selected_boxes'][variant], target) == r['selected_iou'][variant], (r['key'],variant)
                selected_boxes_checked += 1
            groups['all'].append(r)
            for tag in r['strata']: groups[tag].append(r)
            joints['|'.join(sorted(r['strata']))].append(r)
        assert {k:summarize(v) for k,v in groups.items()} == result['summaries']
        assert {k:summarize(v) for k,v in joints.items()} == result['joint_profiles']
        assert sum(len(v) for v in joints.values()) == n
        changes = {}
        for variant in VARIANTS:
            changes[variant] = dict(
                rescues=[r for r in rows if r['selected_iou']['parent']<.5<=r['selected_iou'][variant]],
                breaks=[r for r in rows if r['selected_iou'][variant]<.5<=r['selected_iou']['parent']])
        accepted[split] = result['summaries']
        component_changes[split] = changes
    acceptance = dict(status='complete_local_m105_arithmetic_acceptance',actual_events=3039,selected_boxes_GT_IoU_exact_checks=selected_boxes_checked,
                      complete_full_rows_replay_exact=True,complete_marginal_and_joint_summaries_exact=True,
                      two_gpu_readouts_actual_exit0=True,optimizer_steps=0,checkpoint_saved=False,
                      ground_truth='Dataset-provided DepthTrack Train current boxes from hash-bound original training_labels.json; not predictions or human identity truth',
                      scope='Fixed cached Train state component interventions; not retrained ablation, recursive tracking, semantic contribution, or official metrics',
                      accepted_summaries=accepted,all_variant_parent_changes=component_changes)
    (ROOT/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    print(json.dumps(dict(status=acceptance['status'],events=3039,selected_box_checks=selected_boxes_checked,all={split:values['all'] for split,values in accepted.items()})))


if __name__ == '__main__':
    main()
