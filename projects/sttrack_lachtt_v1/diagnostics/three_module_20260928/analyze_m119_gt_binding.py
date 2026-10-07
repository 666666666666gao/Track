"""Recount fixed readouts from real saved GT; no model or policy fitting."""
import argparse, json, statistics
from pathlib import Path


def stats(values):
    return dict(count=len(values), positive=sum(x>0 for x in values), mean=statistics.mean(values),
                median=statistics.median(values)) if values else dict(count=0,positive=0,mean=None,median=None)


def scores(row, condition):
    if condition=='initial_instance':
        return row['candidate_instance']
    region=[x[0] for x in row['candidate_region']]
    if condition=='category_empty':
        return [a-b[6] for a,b in zip(region,row['candidate_region'])]
    if condition=='category_ring':
        return [a-b[0] for a,b in zip(region,row['candidate_ring'])]
    assert condition=='category_raw'
    return region


def selected_summary(rows, condition):
    result=dict(events=len(rows),correct=0,parent_correct=0,oracle_correct=0,no_qualified_candidates=0,
        parent_miss_with_qualified=0,readout_miss_with_qualified=0,mean_iou=0.,parent_mean_iou=0.,
        rescues=0,breaks=0,severe_rescues=0,severe_harms=0,healthy=0,healthy_correct=0,
        both_qualified=0,both_qualified_worse=0,both_qualified_better=0,both_qualified_iou_delta=0.,
        healthy_parent_correct=0,healthy_rescues=0,healthy_breaks=0,healthy_iou=0.,healthy_parent_iou=0.)
    margins=[];qualified_deltas=[]
    for row in rows:
        values=scores(row,condition);i=max(range(10),key=values.__getitem__)
        a=row['candidate_iou'][i];b=row['candidate_iou'][row['parent_selected']]
        qualified=max(row['candidate_iou'])>=.5
        result['oracle_correct']+=qualified;result['no_qualified_candidates']+=not qualified
        result['parent_miss_with_qualified']+=b<.5 and qualified
        result['readout_miss_with_qualified']+=a<.5 and qualified
        result['correct']+=a>=.5;result['parent_correct']+=b>=.5
        result['mean_iou']+=a/len(rows);result['parent_mean_iou']+=b/len(rows)
        result['rescues']+=a>=.5>b;result['breaks']+=b>=.5>a
        result['severe_rescues']+=a>=.5 and b<=.1;result['severe_harms']+=b>=.5 and a<=.1
        if min(a,b)>=.5:
            result['both_qualified']+=1;result['both_qualified_worse']+=a<b;result['both_qualified_better']+=a>b
            result['both_qualified_iou_delta']+=a-b;qualified_deltas.append(a-b)
        if 'healthy' in row['strata']:
            result['healthy']+=1;result['healthy_correct']+=a>=.5;result['healthy_iou']+=a
            result['healthy_parent_correct']+=b>=.5;result['healthy_parent_iou']+=b
            result['healthy_rescues']+=a>=.5>b;result['healthy_breaks']+=b>=.5>a
        good=[values[j] for j,q in enumerate(row['candidate_iou']) if q>=.5]
        poor=[values[j] for j,q in enumerate(row['candidate_iou']) if q<=.1]
        if good and poor:
            margins.extend(a-b for a in good for b in poor)
    result['healthy_mean_iou']=result.pop('healthy_iou')/result['healthy']
    result['healthy_parent_mean_iou']=result.pop('healthy_parent_iou')/result['healthy']
    result['both_qualified_precision_delta']=stats(qualified_deltas)
    result['good_poor_geometry_margin']=stats(margins)
    assert result['correct']-result['parent_correct']==result['rescues']-result['breaks']
    assert result['events']==result['oracle_correct']+result['no_qualified_candidates']
    assert result['healthy_correct']-result['healthy_parent_correct']==result['healthy_rescues']-result['healthy_breaks']
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert (args.input/'driver.exit').read_text().strip()=='0'
    driver=json.loads((args.input/'result.json').read_text())
    assert driver['status']=='complete_M119_GT_binding_pair' and all(r['exit']==0 for r in driver['launches'])
    rows=[];initial=[]
    for shard in [0,1]:
        folder=args.input/('full_shard'+str(shard))
        rows.extend(json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines())
        initial.extend(json.loads(line) for line in (folder/'initial.jsonl').read_text().splitlines())
    assert len(rows)==len({r['key'] for r in rows})==3502
    assert len(initial)==152 and len({r['sequence'] for r in initial})==152
    analysis={}
    for split,expected in [('fit',2544),('development',495)]:
        valid=[r for r in rows if r['split']==split and r['gt_valid']]
        init=[r for r in initial if r['split']==split]
        assert len(valid)==expected
        counts=dict(valid_gt=len(valid),invalid_gt=sum(r['split']==split and not r['gt_valid'] for r in rows),
            gt_center_in_crop=sum(r['gt_center_in_crop'] for r in valid),
            gt_whole_in_crop=sum(r['gt_whole_in_crop'] for r in valid),
            gt_whole_in_observed_image=sum(r['gt_whole_in_observed_image'] for r in valid))
        gt_groups={}
        for name,group in [('all',valid),('center_in_crop',[r for r in valid if r['gt_center_in_crop']]),
            ('whole_in_observed_image',[r for r in valid if r['gt_whole_in_observed_image']]),
            ('center_out_crop',[r for r in valid if not r['gt_center_in_crop']])]:
            gt_groups[name]=dict(states=len(group),
                category_region_minus_ring=stats([r['gt_region'][0]-r['gt_ring'][0] for r in group]),
                category_region_minus_empty=stats([r['gt_region'][0]-r['gt_region'][6] for r in group]),
                initial_instance_cosine=stats([r['gt_instance'] for r in group]),
                observed_fraction=stats([r['gt_observed_fraction'] for r in group]))
        comparisons={condition:selected_summary(valid,condition) for condition in
            ['category_raw','category_empty','category_ring','initial_instance']}
        paired=dict(both_correct=0,category_only_correct=0,instance_only_correct=0,neither_correct=0)
        for row in valid:
            a=scores(row,'category_raw');b=scores(row,'initial_instance')
            ca=row['candidate_iou'][max(range(10),key=a.__getitem__)]>=.5
            cb=row['candidate_iou'][max(range(10),key=b.__getitem__)]>=.5
            key='both_correct' if ca and cb else ('category_only_correct' if ca else ('instance_only_correct' if cb else 'neither_correct'))
            paired[key]+=1
        analysis[split]=dict(counts=counts,gt_regions=gt_groups,readouts=comparisons,paired_class_instance=paired,
            initial_region_minus_ring=stats([r['region_cosine'][0]-r['ring_cosine'][0] for r in init]))
    assert analysis['fit']['counts']['invalid_gt']+analysis['development']['counts']['invalid_gt']==463
    result=dict(status='complete_M119_CPU_saved_GT_region_analysis',analysis=analysis,
        no_model_execution=True,no_optimization=True,no_tracker_actions=True,no_current_attribute_visibility_truth=True,
        category_and_instance_cosines_not_calibrated_identity_probability=True,
        prior_geometry_GT_not_deployment_input=True,no_new_public_metrics=True)
    args.output.mkdir()
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':
    main()
