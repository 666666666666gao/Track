"""Stdlib-only paired localization reports; no model replay or identity labels."""
import argparse,hashlib,json,math
from collections import defaultdict
from pathlib import Path


ARMS=['human_text','visual_query','generic']
CONDITIONS=['human_text','empty','generic','visual_query']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    assert len({r['key'] for r in rows})==len(rows)
    return rows


def paired(a,b):
    assert len(a)==len(b)>0
    assert all(math.isfinite(x) and math.isfinite(y) and 0<=x<=1 and 0<=y<=1 for x,y in zip(a,b))
    delta=[x-y for x,y in zip(a,b)]
    fine=[d for x,y,d in zip(a,b,delta) if min(x,y)>=.5]
    result=dict(events=len(a),candidate_iou50=sum(x>=.5 for x in a),reference_iou50=sum(y>=.5 for y in b),
        candidate_mean_iou=sum(a)/len(a),reference_mean_iou=sum(b)/len(b),mean_iou_delta=sum(delta)/len(delta),
        cross_half_rescues=sum(x>=.5>y for x,y in zip(a,b)),cross_half_breaks=sum(y>=.5>x for x,y in zip(a,b)),
        severe_rescues=sum(x>=.5 and y<=.1 for x,y in zip(a,b)),severe_harms=sum(y>=.5 and x<=.1 for x,y in zip(a,b)),
        precision_better=sum(d>0 for d in delta),precision_worse=sum(d<0 for d in delta),precision_tied=sum(d==0 for d in delta),
        both_qualified=len(fine),both_qualified_better=sum(d>0 for d in fine),
        both_qualified_worse=sum(d<0 for d in fine),both_qualified_tied=sum(d==0 for d in fine),
        both_qualified_mean_iou_delta=sum(fine)/len(fine) if fine else None)
    assert result['candidate_iou50']-result['reference_iou50']==result['cross_half_rescues']-result['cross_half_breaks']
    return result


def panel(rows,reference):
    keyed={r['key']:r for r in reference}
    assert set(keyed)=={r['key'] for r in rows}
    groups=defaultdict(list)
    for r in rows:
        assert r['strata']==keyed[r['key']]['strata']
        groups['all'].append(r)
        for tag in r['strata']:groups[tag].append(r)
    return {tag:paired([r['selected_iou'] for r in group],[keyed[r['key']]['selected_iou'] for r in group])
        for tag,group in groups.items()}


def native_reference(rows):
    return [dict(key=r['key'],strata=r['strata'],selected_iou=r['native_iou']) for r in rows]


def describe(rows,empty,parent):
    groups=defaultdict(list)
    sequences=defaultdict(list)
    for r in rows:
        groups['all'].append(r)
        for tag in r['strata']:groups[tag].append(r)
        sequences[r['key'].rsplit('@',1)[0]].append(r)
    geometry={tag:dict(
        fixed_native_position_refinement=paired([r['native_position_refined_iou'] for r in group],[r['native_iou'] for r in group]),
        full256_oracle_refinement=paired([r['oracle_iou'] for r in group],[r['native_full256_oracle_iou'] for r in group]),
        new_events_with_qualified_candidates=sum(r['oracle_iou']>=.5>r['native_full256_oracle_iou'] for r in group),
        lost_events_with_qualified_candidates=sum(r['native_full256_oracle_iou']>=.5>r['oracle_iou'] for r in group),
        static_no_qualified_candidate=sum(r['native_full256_oracle_iou']<.5 for r in group),
        refined_no_qualified_candidate=sum(r['oracle_iou']<.5 for r in group),
        refined_candidate_present_but_not_selected=sum(r['oracle_iou']>=.5>r['selected_iou'] for r in group))
        for tag,group in groups.items()}
    empty_map={r['key']:r for r in empty};parent_map={r['key']:r for r in parent}
    return dict(vs_native=panel(rows,native_reference(rows)),vs_own_empty=panel(rows,empty),vs_M101_reference=panel(rows,parent),
        geometry=geometry,per_sequence={name:dict(
            vs_native=paired([r['selected_iou'] for r in group],[r['native_iou'] for r in group]),
            vs_own_empty=paired([r['selected_iou'] for r in group],[empty_map[r['key']]['selected_iou'] for r in group]),
            vs_M101_reference=paired([r['selected_iou'] for r in group],[parent_map[r['key']]['selected_iou'] for r in group]),
            fixed_native_position_refinement=paired([r['native_position_refined_iou'] for r in group],[r['native_iou'] for r in group]),
            full256_oracle_refinement=paired([r['oracle_iou'] for r in group],[r['native_full256_oracle_iou'] for r in group]))
            for name,group in sequences.items()})


def compile_report(root,parent_path):
    parent=json.loads(parent_path.read_text())
    assert parent['status']=='complete_visual_control' and parent['seed']==2027
    assert parent['final_weights_sha256']=='1d187d78c1dd76ebf59e41fd31af16fa69cecc37b4cb6c74fbdfe3ed5c237293'
    parent_rows=parent['development_rows'];assert len(parent_rows)==495
    reports={};rows_by={};summary={};artifacts=[]
    for arm in ARMS:
        location=root/('train_'+arm)
        assert (root/('train_'+arm+'.exit')).read_text().strip()=='0'
        report=json.loads((location/'result.json').read_text());reports[arm]=report
        assert report['status']=='complete_M121_training' and report['arm']==arm and report['seed']==2027
        assert report['optimizer_steps']==640 and report['final_state_roundtrip_exact']
        assert report['source']['sequences']==152 and report['source']['valid_gt']==3039
        assert report['source']['invalid_gt_excluded']==463 and report['no_recursive_or_public_evaluation']
        assert sha(location/'final.pt')==report['final_sha256']
        rows_by[arm]={c:read_rows(location/(c+'_development.jsonl')) for c in CONDITIONS}
        assert all(len(r)==495 for r in rows_by[arm].values())
        empty_map={r['key']:r for r in rows_by[arm]['empty']}
        fit=read_rows(location/'fit_events.jsonl');assert len(fit)==2544
        assert not set(r['key'] for r in fit)&set(r['key'] for r in parent_rows)
        for c,rows in rows_by[arm].items():
            assert sum(r['selected_iou']>=.5 for r in rows)==report['content_conditions'][c]['all']['selected_iou50']
            assert abs(sum(r['selected_iou'] for r in rows)/495-report['content_conditions'][c]['all']['selected_mean_iou'])<1e-12
            assert all(0<=r['selected']<256 and len(r['selected_box'])==4 and len(r['semantic_delta'])==256 for r in rows)
            for r in rows:
                empty=empty_map[r['key']]
                assert all(r[k]==empty[k] for k in ['native_iou','native_full256_oracle_iou','oracle_iou',
                    'native_position_refined_iou','observation_probability','GT_box_intersects_observed_window'])
            if c=='empty':assert all(not any(r['semantic_delta']) for r in rows)
        summary[arm]={c:describe(r,rows_by[arm]['empty'],parent_rows) for c,r in rows_by[arm].items()}
        summary[arm]['fit_diagnostic']=dict(vs_native=panel(fit,native_reference(fit)),
            fixed_native_position_refinement=paired([r['native_position_refined_iou'] for r in fit],[r['native_iou'] for r in fit]),
            full256_oracle_refinement=paired([r['oracle_iou'] for r in fit],[r['native_full256_oracle_iou'] for r in fit]))
        for name in ['result.json','fit_events.jsonl','final.pt']+[c+'_development.jsonl' for c in CONDITIONS]:
            path=location/name;artifacts.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path)))
    for key in ['initial_state_sha256','optimized_parameters','bank_sha256','labels_sha256',
                'source_sha256','model_source_sha256','loader_source_sha256']:
        assert len({r[key] for r in reports.values()})==1,key
    assert len({json.dumps(r['source'],sort_keys=True) for r in reports.values()})==1
    main=rows_by['human_text']['human_text']
    references=dict(native=native_reference(main),own_empty=rows_by['human_text']['empty'],
        visual_capacity=rows_by['visual_query']['visual_query'],generic_capacity=rows_by['generic']['generic'])
    comparison={name:panel(main,reference) for name,reference in references.items()}
    gates={}
    for name,pairs in comparison.items():
        gates[name+'_more_qualified']=pairs['all']['candidate_iou50']>pairs['all']['reference_iou50']
        gates[name+'_higher_mean']=pairs['all']['mean_iou_delta']>0
        gates[name+'_healthy_no_cross_half_breaks']=pairs['healthy']['cross_half_breaks']==0
        gates[name+'_healthy_mean_preserved']=pairs['healthy']['mean_iou_delta']>=0
        gates[name+'_transition_count_preserved']=pairs['transition']['candidate_iou50']>=pairs['transition']['reference_iou50']
        gates[name+'_transition_mean_preserved']=pairs['transition']['mean_iou_delta']>=0
        fine=pairs['all']['both_qualified_mean_iou_delta']
        gates[name+'_both_qualified_precision_preserved']=fine is not None and fine>=0
    return dict(status='complete_M121_CPU_paired_report',main_comparisons=comparison,summary=summary,
        main_progression_checks=gates,main_all_progression_checks=all(gates.values()),
        source_receipts=artifacts,M101_reference_result_sha256=sha(parent_path),
        model_replays=0,no_policy_fitted=True,no_recursive_or_public_metrics=True,
        GT_usage='Cached genuine Train boxes only; overlap/intersection measures are not physical identity or phrase visibility labels.',
        reference_boundary='M101 is historical paired localization only; three M121 arms are the matched capacity controls.',
        precision_definition='Exact stored IoU difference; both-qualified means both IoUs >=0.5; cross-half and severe <=0.1 are separate.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--parent-result',type=Path,required=True);args=parser.parse_args()
    path=args.root/'comparison.json';assert not path.exists()
    report=compile_report(args.root,args.parent_result);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report['main_progression_checks'],
        main_all_progression_checks=report['main_all_progression_checks'])),flush=True)


if __name__=='__main__':main()
