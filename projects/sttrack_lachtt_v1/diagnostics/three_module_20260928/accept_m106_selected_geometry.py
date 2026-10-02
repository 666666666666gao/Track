"""Independent CPU arithmetic acceptance of all M106 cached geometry rows."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math
import struct

HERE = Path(__file__).parent
ROOT = HERE/'m106_completed'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]


def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def overlap(b,t):
    b=list(map(f32,b));t=list(map(f32,t))
    extent=[max(f32(min(f32(b[i]+b[i+2]),f32(t[i]+t[i+2]))-max(b[i],t[i])),0.) for i in (0,1)]
    intersection=f32(extent[0]*extent[1])
    union=f32(f32(f32(b[2]*b[3])+f32(t[2]*t[3]))-intersection)
    return f32(intersection/union)


def summary(values):
    return dict(valid_gt=len(values),
                native_iou50=sum(r['native_iou']>=.5 for r in values),
                selected_iou50=sum(r['selected_iou']>=.5 for r in values),
                oracle_iou50=sum(r['oracle_iou']>=.5 for r in values),
                rescues=sum(r['native_iou']<.5<=r['selected_iou'] for r in values),
                breaks=sum(r['selected_iou']<.5<=r['native_iou'] for r in values),
                native_mean_iou=sum(r['native_iou'] for r in values)/len(values),
                selected_mean_iou=sum(r['selected_iou'] for r in values)/len(values),
                parent_correct=sum(r['parent_iou']>=.5 for r in values),
                parent_mean_iou=sum(r['parent_iou'] for r in values)/len(values),
                parent_rescues=sum(r['parent_iou']<.5<=r['selected_iou'] for r in values),
                parent_breaks=sum(r['selected_iou']<.5<=r['parent_iou'] for r in values),
                original_top10_correct=sum(r['native_top10_iou']>=.5 for r in values),
                original_miss_refined_hit=sum(r['native_top10_iou']<.5<=r['oracle_iou'] for r in values))


def main():
    download=read(ROOT/'download_receipt.json')
    for r in download['files']:
        p=ROOT/r['relative']
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    driver=read(ROOT/'driver.json')
    assert driver['exit_code']==0 and (ROOT/'driver.exit').read_text().strip()=='0'
    assert [(r['mode'],r['weight'],r['gpu'],r['exit_code']) for r in driver['runs']]==[('sanity',0,0,0),('sanity',1,1,0),('train',0,0,0),('train',1,1,0)]
    replay=read(ROOT/'verification/result.json')
    assert replay['status']=='complete_m106_readonly_checkpoint_replay'
    for key in ['control_M104_all_tensors_equal','control_M104_all3039_rows_and_all_summaries_equal','both_weights_all3039_final_replay_equal','all_replayed_states_unchanged']:
        assert replay[key] is True
    assert sha(ROOT/'verification/geometry_boxes.jsonl')==replay['geometry_boxes_sha256']
    assert sha(HERE/'verify_m106_checkpoint_replay.py')==replay['source_sha256']
    labels=read(HERE/'m105_inputs/training_labels.json')
    assert sha(HERE/'m105_inputs/training_labels.json')==replay['training_labels_sha256']==read(HERE/'m105_inputs/preparation.json')['training_labels_sha256']
    geometry=rows(ROOT/'verification/geometry_boxes.jsonl')
    assert len(geometry)==3039 and len({r['key'] for r in geometry})==3039
    by_key={r['key']:r for r in geometry}
    old_result=read(HERE/'m104_completed/train/result.json')
    assert sha(HERE/'m104_completed/train/final.pt')==sha(ROOT/'train_weight0/final.pt')==replay['M104_final_sha256']
    collected={};changes={};profiles={};table=[];numbers=0;pair_changes={}
    for weight in [0,1]:
        w=str(weight); report=read(ROOT/('train_weight'+w)/'result.json')
        sanity=read(ROOT/('sanity_weight'+w)/'result.json')
        assert report['status']=='complete_selected_geometry_preservation' and report['mode']=='train'
        assert report['seed']==2027 and report['eligible_fit_events']==2033 and report['optimizer_steps']==384
        assert report['optimized_parameters']==59540 and report['learning_rate']==3e-4 and report['batch_size']==64
        assert report['selected_preservation_weight']==weight and len(report['history'])==12
        assert all(r['optimizer_steps']==32 and math.isfinite(r['mean_loss']) for r in report['history'])
        assert sanity['mode']=='sanity' and sanity['optimizer_steps']==2 and sanity['checkpoint_saved'] is False
        assert all(v>0 and math.isfinite(v) for v in sanity['gradient_norms'].values())
        assert report['parent_weight_sha256']==replay['parent_weight_sha256']
        assert sha(ROOT/('train_weight'+w)/'final.pt')==report['final_weight_sha256']==replay['final_weights_sha256'][w]
        assert report['source_sha256']==sha(HERE/'train_selected_geometry_preservation.py')
        for flag in ['actual3039zero_geometry_exact','actual495parent_selection_exact','parent_parameters_buffers_unchanged','no_GT_forward_gate','no_semantic_identity_labels_or_public_or_recursive_action','final_reload_all3039rows_exact']:
            assert report[flag] is True
        collected[w]={};changes[w]={};profiles[w]={}
        for split,n in [('fit',2544),('development',495)]:
            events=rows(ROOT/('train_weight'+w)/(split+'_events.jsonl'))
            assert len(events)==n
            if weight==0:
                assert events==rows(HERE/'m104_completed/train'/(split+'_events.jsonl'))
                assert report[split]==old_result[split]
            groups=defaultdict(list);joint=defaultdict(list)
            for r in events:
                g=by_key[r['key']]; label=labels[r['key']]
                assert (g['split'],g['selected'],g['strata'])==(split,r['selected'],r['strata'])
                assert label['split']==split and label['current'] is not None
                ious={name:[overlap(box,label['current']) for box in values] for name,values in g['boxes'].items()}
                assert all(len(v)==10 for v in ious.values())
                assert all(math.isfinite(v) and 0<=v<=1 for values in ious.values() for v in values)
                s=r['selected'];now=ious['weight'+w];parent=ious['parent']
                assert (r['native_iou'],r['parent_iou'],r['native_top10_iou'],r['selected_iou'],r['oracle_iou'])==(parent[0],parent[s],max(parent),now[s],max(now)),r['key']
                numbers+=30
                groups['all'].append(r)
                for tag in r['strata']:groups[tag].append(r)
                joint['|'.join(sorted(r['strata']))].append(r)
            derived={tag:summary(v) for tag,v in groups.items()}
            assert derived==report[split]
            profiles[w][split]={tag:summary(v) for tag,v in joint.items()}
            assert sum(v['valid_gt'] for v in profiles[w][split].values())==n
            collected[w][split]=derived
            changes[w][split]=dict(rescues=[r for r in events if r['parent_iou']<.5<=r['selected_iou']],
                                   breaks=[r for r in events if r['selected_iou']<.5<=r['parent_iou']])
            for tag,values in derived.items():table.append(dict(weight=weight,split=split,group=tag,**values))
        d=collected[w]['development']
        gates=dict(correct_exceeds_parent=d['all']['selected_iou50']>d['all']['parent_correct'],
                   mean_iou_exceeds_parent=d['all']['selected_mean_iou']>d['all']['parent_mean_iou'],
                   healthy_new_breaks_zero=d['healthy']['parent_breaks']==0,
                   transition_at_least_parent=d['transition']['selected_iou50']>=d['transition']['parent_correct'])
        assert gates==report['fixed_state_checks']
    for split in ['fit','development']:
        zero=rows(ROOT/'train_weight0'/(split+'_events.jsonl'));one=rows(ROOT/'train_weight1'/(split+'_events.jsonl'))
        assert [r['key'] for r in zero]==[r['key'] for r in one]
        paired=[dict(key=a['key'],strata=a['strata'],control_iou=a['selected_iou'],preservation_iou=b['selected_iou']) for a,b in zip(zero,one)]
        pair_changes[split]=dict(gained_correct=[r for r in paired if r['control_iou']<.5<=r['preservation_iou']],
                                lost_correct=[r for r in paired if r['preservation_iou']<.5<=r['control_iou']],
                                iou_improved=sum(r['preservation_iou']>r['control_iou'] for r in paired),
                                iou_decreased=sum(r['preservation_iou']<r['control_iou'] for r in paired),
                                iou_equal=sum(r['preservation_iou']==r['control_iou'] for r in paired))
    acceptance=dict(status='complete_local_m106_arithmetic_acceptance',events=3039,independent_candidate_IoU_checks=numbers,
                    unique_candidate_IoU_checks=91170,control_M104_rows_summaries_and_checkpoint_bytes_exact=True,
                    actual_four_children_exit0=True,all_marginal_summaries_exact=True,all_original_gates_exact=True,
                    source_sha256=sha(Path(__file__)),GT_provenance='Hash-bound dataset-provided DepthTrack Train current boxes; not predictions, semantic labels, or physical identity truth.',
                    scope='Fixed Train cache geometry learning; no recursive/public/semantic claim.',
                    summaries=collected,joint_profiles=profiles,all_parent_changes=changes,all_control_pair_changes=pair_changes)
    (ROOT/'acceptance.json').write_bytes((json.dumps(acceptance,indent=2)+'\n').encode())
    with (ROOT/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(dict(status=acceptance['status'],events=3039,unique_IoU_checks=91170,
                         all={w:{s:g['all'] for s,g in v.items()} for w,v in collected.items()},
                         paired={s:{k:len(v) if isinstance(v,list) else v for k,v in p.items()} for s,p in pair_changes.items()})))


if __name__=='__main__':
    main()
