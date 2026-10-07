"""CPU-only recount of each actual final; no new policy or model execution."""
from pathlib import Path
from datetime import datetime
import hashlib, json

root=Path(__file__).resolve().parents[4]
d=root/'projects/sttrack_lachtt_v1/diagnostics/three_module_20260928'
completed=d/'m118_completed'
driver=json.loads((completed/'result.json').read_text(encoding='utf-8'))
assert driver['status']=='complete_M118_regional_suite'
assert all(r['exit']==0 for r in driver['launches'])
mirror=json.loads((d/'M118_ARTIFACT_MIRROR.json').read_text(encoding='utf-8'))
for item in mirror['files']:
    b=(completed/item['path']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==item['sha256'] and len(b)==item['bytes']
read=lambda p:[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
summary={}
for arm in ['human_text','visual_query','generic']:
    report=json.loads((completed/('train_'+arm)/'result.json').read_text(encoding='utf-8'))
    assert report['optimizer_steps']==480 and report['optimized_parameters']==150528
    weight=root/'.aris/m118_regional_decoder_20261007/finals'/('train_'+arm)/'final.pt'
    assert hashlib.sha256(weight.read_bytes()).hexdigest()==report['final_sha256']
    summary[arm]={}
    for condition in ['empty','generic','visual_query','human_text']:
        rows=read(completed/('train_'+arm)/(condition+'_development.jsonl'))
        assert len(rows)==495
        out=dict(states=495,correct=0,parent_correct=0,mean_iou=0.,parent_mean_iou=0.,
                 cross_half_rescues=0,cross_half_breaks=0,severe_rescues=0,severe_harms=0,
                 both_qualified=0,both_qualified_better=0,both_qualified_worse=0,both_qualified_tied=0,
                 both_qualified_iou_delta=0.,healthy=0,healthy_correct=0,healthy_mean_iou=0.,
                 transition=0,transition_correct=0)
        for row in rows:
            chosen=row['selected'];teacher=row['parent_selected'];iou=row['candidate_iou']
            assert iou[chosen]==row['selected_iou'] and iou[teacher]==row['parent_iou']
            assert max(range(10),key=row['scores'].__getitem__)==chosen
            assert max(range(10),key=row['parent_scores'].__getitem__)==teacher
            a,b=iou[chosen],iou[teacher]
            out['correct']+=a>=.5;out['parent_correct']+=b>=.5
            out['mean_iou']+=a/495;out['parent_mean_iou']+=b/495
            out['cross_half_rescues']+=a>=.5>b;out['cross_half_breaks']+=b>=.5>a
            out['severe_rescues']+=a>=.5 and b<=.1;out['severe_harms']+=b>=.5 and a<=.1
            if min(a,b)>=.5:
                delta=a-b;out['both_qualified']+=1
                out['both_qualified_better']+=delta>0;out['both_qualified_worse']+=delta<0
                out['both_qualified_tied']+=delta==0;out['both_qualified_iou_delta']+=delta
            if 'healthy' in row['strata']:
                out['healthy']+=1;out['healthy_correct']+=a>=.5;out['healthy_mean_iou']+=a
            if 'transition' in row['strata']:
                out['transition']+=1;out['transition_correct']+=a>=.5
        out['healthy_mean_iou']/=out['healthy']
        assert out['correct']-out['parent_correct']==out['cross_half_rescues']-out['cross_half_breaks']
        assert out['correct']==report['content_conditions'][condition]['all']['selected_iou50']
        assert abs(out['mean_iou']-report['content_conditions'][condition]['all']['selected_mean_iou'])<1e-12
        summary[arm][condition]=out
main=summary['human_text']['human_text']
controls={name:summary[name][name] for name in ['visual_query','generic']}
checks=json.loads((completed/'train_human_text/result.json').read_text())['fixed_state_checks']
checks.update({name+'_correct_exceeds':main['correct']>control['correct'] for name,control in controls.items()})
checks.update({name+'_mean_exceeds':main['mean_iou']>control['mean_iou'] for name,control in controls.items()})
result=dict(status='complete_M118_final_CPU_recount',observed_at=datetime.now().astimezone().isoformat(),
    summary=summary,main_checks=checks,main_all_checks=all(checks.values()),all_three_finals_preserved=True,
    no_policy_fitted=True,no_model_replay=True,no_recursive_or_public_metrics=True,
    metric_definitions='cross_half uses0.5 boundary against Parent; severe uses<=0.1 versus>=0.5; both-qualified precision is separate. Native summary fields not relabelled Parent.')
path=d/'M118_COMPARISON.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(main_checks=checks,matched_arms={arm:summary[arm][arm] for arm in summary}),indent=2))
