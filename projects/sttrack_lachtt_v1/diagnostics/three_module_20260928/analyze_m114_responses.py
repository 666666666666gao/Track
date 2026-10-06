"""Analyze exact cached candidate scores against the persisted dataset GT IoUs."""
import argparse,json,math
from pathlib import Path
from collections import defaultdict
from statistics import median


def bce(score,target):return max(score,0)-target*score+math.log1p(math.exp(-abs(score)))


def summarize(rows):
    count=len(rows);candidate_count=sum(len(r['scores']) for r in rows)
    common_energy=0.;centered_energy=0.;total_energy=0.;losses=[0.,0.,0.]
    margins=[];selection_changes=0;improvements=0;worsenings=0;no_good=0;no_bad=0
    for row in rows:
        scores=row['scores'];empty=row['empty_scores'];iou=row['candidate_iou']
        assert len(scores)==len(empty)==len(iou) and len(scores)>1
        assert all(math.isfinite(x) for x in scores+empty+iou)
        delta=[s-e for s,e in zip(scores,empty)];common=sum(delta)/len(delta)
        total_energy+=sum(d*d for d in delta)
        common_energy+=len(delta)*common*common
        centered_energy+=sum((d-common)**2 for d in delta)
        for s,e,t in zip(scores,empty,iou):
            losses[0]+=bce(e,t);losses[1]+=bce(s,t);losses[2]+=bce(e+common,t)
        selected=max(range(len(scores)),key=scores.__getitem__)
        empty_selected=max(range(len(empty)),key=empty.__getitem__)
        assert selected==row['selected'] and iou[selected]==row['selected_iou']
        selection_changes+=selected!=empty_selected
        improvements+=iou[selected]>iou[empty_selected]
        worsenings+=iou[selected]<iou[empty_selected]
        good=[i for i,x in enumerate(iou) if x>=.5];bad=[i for i,x in enumerate(iou) if x<=.1]
        no_good+=not bool(good);no_bad+=not bool(bad)
        if good and bad:
            before=max(empty[i] for i in good)-max(empty[i] for i in bad)
            after=max(scores[i] for i in good)-max(scores[i] for i in bad)
            margins.append((before,after))
    assert math.isclose(total_energy,common_energy+centered_energy,rel_tol=1e-10,abs_tol=1e-12)
    return dict(states=count,candidates=candidate_count,
        selected_iou50=sum(r['selected_iou']>=.5 for r in rows),
        empty_iou50=sum(r['candidate_iou'][max(range(len(r['empty_scores'])),key=r['empty_scores'].__getitem__)]>=.5 for r in rows),
        selected_mean_iou=sum(r['selected_iou'] for r in rows)/count,
        score_delta_energy=dict(total=total_energy,common=common_energy,candidate_specific=centered_energy),
        localization_bce=dict(empty=losses[0]/candidate_count,full=losses[1]/candidate_count,common_only=losses[2]/candidate_count),
        changed_selections=selection_changes,iou_improvements=improvements,iou_worsenings=worsenings,
        no_good_candidate=no_good,no_bad_candidate=no_bad,
        margin_eligible=len(margins),margin_improvements=sum(after>before for before,after in margins),
        margin_worsenings=sum(after<before for before,after in margins),
        margin_crossings_to_nonnegative=sum(before<0<=after for before,after in margins),
        margin_crossings_to_negative=sum(after<0<=before for before,after in margins),
        margin_mean_before=sum(before for before,after in margins)/len(margins),
        margin_mean_after=sum(after for before,after in margins)/len(margins),
        median_margin_delta=median(after-before for before,after in margins),
        semantic_delta_norm_mean=sum(r['semantic_delta_norm'] for r in rows)/count,
        phrase_delta_norm_mean=sum(r['phrase_delta_norm'] for r in rows)/count)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();pair=json.loads((a.root/'result.json').read_text())
    assert pair['status']=='complete_M114_response_pair' and pair['optimizer_steps']==0
    assert (a.root/'driver.exit').read_text().strip()=='0'
    result={}
    for arm in ['human_text','generic']:
        values=[json.loads(line) for line in (a.root/('full_'+arm)/'responses.jsonl').read_text().splitlines()]
        groups=defaultdict(list)
        for row in values:groups[(row['split'],row['condition'])].append(row)
        assert sum(len(v) for v in groups.values())==3*(2544+495)
        result[arm]={}
        for split in ['fit','development']:
            baseline=groups[(split,'empty')]
            assert len(baseline)==(2544 if split=='fit' else 495)
            assert len({r['key'] for r in baseline})==len(baseline)
            result[arm][split]={}
            for condition in ['empty','generic','human_text']:
                rows=groups[(split,condition)]
                assert [r['key'] for r in rows]==[r['key'] for r in baseline]
                assert all(r['candidate_iou']==e['candidate_iou'] and r['empty_scores']==e['scores'] for r,e in zip(rows,baseline))
                result[arm][split][condition]=summarize(rows)
    receipt=dict(status='complete_M114_GT_response_analysis',arms=result,optimizer_steps=0,
        no_identity_truth_inferred_from_iou=True,no_recursive_or_official_metrics=True,
        common_shift_does_not_establish_discrimination=True)
    a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
