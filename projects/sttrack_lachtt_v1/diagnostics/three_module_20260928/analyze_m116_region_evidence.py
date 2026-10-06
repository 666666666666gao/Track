"""CPU-only localization grouping of measured spatial phrase scores; no policy fitting."""
import argparse,json,math
from pathlib import Path
from collections import Counter


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gt-responses',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    gt={r['key']:r for r in rows(a.gt_responses) if r['condition']=='empty'}
    assert Counter(r['split'] for r in gt.values())=={'fit':2544,'development':495}
    observed=[]
    for shard in [0,1]:
        folder=a.root/('full_shard'+str(shard));receipt=json.loads((folder/'result.json').read_text())
        assert receipt['status']=='complete_M116_full_shard' and receipt['model_updates']==0
        assert receipt['human_confirmed'] and not receipt['GT_loaded'] and receipt['frozen_state_exact']
        observed+=rows(folder/'phrase_responses.jsonl')
    initial=[r for r in observed if r['is_initialization']]
    current=[r for r in observed if not r['is_initialization']]
    assert len(initial)==152 and len(current)==3502
    assert len({r['key'] for r in current})==3502
    assert set(gt)<=set(r['key'] for r in current)
    for r in observed:
        assert len(r['phrase_mask'])==5 and r['phrase_mask'][0]
        n=1 if r['is_initialization'] else 10
        for name in ['region_cosine','ring_cosine','region_minus_ring']:
            assert len(r[name])==n and all(len(v)==7 for v in r[name])
            assert all(math.isfinite(x) for v in r[name] for x in v)
        for k in range(n):
            for j in range(7):assert abs(r['region_cosine'][k][j]-r['ring_cosine'][k][j]-r['region_minus_ring'][k][j])<1e-7
        assert all(0<=v<=1 for v in r['observed_fraction']+r['ring_observed_fraction'])
    summary={};events=[]
    for split in ['fit','development']:
        selected=[r for r in current if r['key'] in gt and r['split']==split]
        summary[split]={}
        for kind in ['region_cosine','region_minus_ring']:
            summary[split][kind]={}
            for content in ['human_category','human_phrases','generic','empty']:
                margins=[];correct=0;total=0;iou_sum=0.;quality_pairs=quality_concordant=quality_tied=0
                for r in selected:
                    truth=gt[r['key']];iou=truth['candidate_iou'];assert truth['split']==split
                    cols=[0] if content=='human_category' else [i for i,v in enumerate(r['phrase_mask']) if v] if content=='human_phrases' else [5] if content=='generic' else [6]
                    values=[sum(v[j] for j in cols)/len(cols) for v in r[kind]]
                    best=max(range(10),key=values.__getitem__)
                    correct+=iou[best]>=.5;total+=1;iou_sum+=iou[best]
                    good=[i for i,v in enumerate(iou) if v>=.5];poor=[i for i,v in enumerate(iou) if v<=.1]
                    if good and poor:
                        margin=max(values[i] for i in good)-max(values[i] for i in poor)
                        margins.append(margin)
                        events.append(dict(key=r['key'],split=split,kind=kind,content=content,good_poor_margin=margin))
                    for i in good:
                        for j in good:
                            if i>=j or iou[i]==iou[j]:continue
                            direction=(iou[i]-iou[j])*(values[i]-values[j]);quality_pairs+=1
                            quality_concordant+=direction>0;quality_tied+=direction==0
                assert margins and quality_pairs
                summary[split][kind][content]=dict(states=total,good_poor_states=len(margins),
                    positive_margin=sum(m>0 for m in margins),mean_margin=sum(margins)/len(margins),
                    read_only_argmax_correct=correct,read_only_argmax_mean_iou=iou_sum/total,
                    qualified_candidate_pairs=quality_pairs,quality_concordant=quality_concordant,quality_tied=quality_tied)
    result=dict(status='complete_M116_cpu_region_localization_analysis',summary=summary,
        initializations=152,observed_current_events=3502,valid_GT_current_events=len(gt),excluded_invalid_GT=3502-len(gt),
        optimizer_steps=0,policy_fitted=False,tracking_state_committed=False,
        identity_or_attribute_visibility_truth_used=False,
        limits='CLIP patch projection is a hypothesis; all argmaxes are only frozen-state readouts. IoU groups are localization labels, not distinct-instance truth. No formal benchmark metrics.')
    a.output.mkdir();(a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (a.output/'margin_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
