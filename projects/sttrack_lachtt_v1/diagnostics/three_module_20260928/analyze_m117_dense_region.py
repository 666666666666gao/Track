"""Matched fixed-state dense encoding analysis, with every declared readout retained."""
import argparse,json,math
from pathlib import Path
from collections import Counter


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def table(observed,gt,centered):
    summary={};events=[]
    for split in ['fit','development']:
        selected=[r for r in observed if not r['is_initialization'] and r['key'] in gt and r['split']==split]
        summary[split]={}
        for kind in ['region_cosine','region_minus_ring']:
            summary[split][kind]={}
            for content in ['human_category','human_phrases','generic','empty']:
                margins=[];correct=total=0;iou_sum=0.;quality_pairs=quality_concordant=quality_tied=zero_scores=0
                for r in selected:
                    truth=gt[r['key']];iou=truth['candidate_iou'];assert truth['split']==split
                    cols=[0] if content=='human_category' else [i for i,v in enumerate(r['phrase_mask']) if v] if content=='human_phrases' else [5] if content=='generic' else [6]
                    values=[sum(v[j]-(v[6] if centered else 0.) for j in cols)/len(cols) for v in r[kind]]
                    assert all(math.isfinite(v) for v in values)
                    best=max(range(10),key=values.__getitem__)
                    zero_scores+=not any(values)
                    correct+=iou[best]>=.5;total+=1;iou_sum+=iou[best]
                    good=[i for i,v in enumerate(iou) if v>=.5];poor=[i for i,v in enumerate(iou) if v<=.1]
                    if good and poor:
                        margin=max(values[i] for i in good)-max(values[i] for i in poor)
                        margins.append(margin);events.append(dict(key=r['key'],split=split,kind=kind,content=content,empty_centered=centered,good_poor_margin=margin))
                    for i in good:
                        for j in good:
                            if i>=j or iou[i]==iou[j]:continue
                            direction=(iou[i]-iou[j])*(values[i]-values[j]);quality_pairs+=1
                            quality_concordant+=direction>0;quality_tied+=direction==0
                assert margins and quality_pairs
                summary[split][kind][content]=dict(states=total,good_poor_states=len(margins),
                    positive_margin=sum(m>0 for m in margins),mean_margin=sum(margins)/len(margins),
                    read_only_argmax_correct=correct,read_only_argmax_mean_iou=iou_sum/total,
                    qualified_candidate_pairs=quality_pairs,quality_concordant=quality_concordant,
                    quality_tied=quality_tied,all_zero_score_states=zero_scores)
    return summary,events


def load(root,status):
    observed=[]
    for shard in [0,1]:
        folder=root/('full_shard'+str(shard));receipt=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        assert receipt['status']==status and receipt['model_updates']==0
        assert receipt['human_confirmed'] and not receipt['GT_loaded'] and receipt['frozen_state_exact']
        observed+=read_rows(folder/'phrase_responses.jsonl')
    assert sum(r['is_initialization'] for r in observed)==152
    assert len({r['key'] for r in observed if not r['is_initialization']})==3502
    assert len(observed)==3654
    for r in observed:
        assert len(r['phrase_mask'])==5 and r['phrase_mask'][0]
        n=1 if r['is_initialization'] else 10
        for name in ['region_cosine','ring_cosine','region_minus_ring']:
            assert len(r[name])==n and all(len(v)==7 for v in r[name])
            assert all(math.isfinite(x) for v in r[name] for x in v)
        for k in range(n):
            for j in range(7):assert abs(r['region_cosine'][k][j]-r['ring_cosine'][k][j]-r['region_minus_ring'][k][j])<1e-7
    return observed


def main():
    p=argparse.ArgumentParser()
    for name in ['root','baseline','baseline-analysis','gt-responses','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    gt={r['key']:r for r in read_rows(a.gt_responses) if r['condition']=='empty'}
    assert Counter(r['split'] for r in gt.values())=={'fit':2544,'development':495}
    models={'ordinary':load(a.baseline,'complete_M116_full_shard'),
            'dense':load(a.root,'complete_M117_full_shard')}
    assert [r['key'] for r in models['ordinary']]==[r['key'] for r in models['dense']]
    assert set(gt)<=set(r['key'] for r in models['dense'])
    for old,new in zip(models['ordinary'],models['dense']):
        for field in ['sequence','split','is_initialization','phrase_mask','global_cosine','observed_fraction','ring_observed_fraction']:
            assert old[field]==new[field],(old['key'],field)
    summaries={};events=[];initial={}
    for name,observed in models.items():
        summaries[name]={};initial[name]={}
        for centered in [False,True]:
            condition='empty_centered' if centered else 'raw'
            summaries[name][condition],rows=table(observed,gt,centered)
            events+=[dict(r,encoder=name) for r in rows]
            init=[r for r in observed if r['is_initialization']]
            values=[r['region_minus_ring'][0][0]-(r['region_minus_ring'][0][6] if centered else 0.) for r in init]
            initial[name][condition]=dict(initializations=152,category_region_exceeds_ring=sum(v>0 for v in values),mean_category_region_minus_ring=sum(values)/152)
    baseline=json.loads(a.baseline_analysis.read_text(encoding='utf-8'))['summary']
    for split,kinds in baseline.items():
        for kind,contents in kinds.items():
            for content,values in contents.items():
                assert all(summaries['ordinary']['raw'][split][kind][content][k]==v for k,v in values.items())
    result=dict(status='complete_M117_dense_region_CPU_analysis',summaries=summaries,initial=initial,
        matched_baseline_recount_exact=True,matched_CLS_mask_content_exact=True,all_declared_readouts_reported=True,
        valid_GT_states=3039,excluded_invalid_GT=463,optimizer_steps=0,policy_fitted=False,tracker_state_committed=False,
        empty_centered_empty_tie='All ten scores are zero; argmax returns the first native candidate. This is not learned semantic gain.',
        limits='GT IoU groups are localization only; no physical-instance or current phrase visibility truth. Raw-versus-centered is a separate declared comparison. No official benchmarks.')
    a.output.mkdir();(a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    (a.output/'margin_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events),encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=result['status'],initial=initial,development={m:v['raw']['development'] for m,v in summaries.items()})))


if __name__=='__main__':main()
