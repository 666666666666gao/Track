"""Post-seal ROB accounting, following the existing exact full127 analysis.

This module is independent of the running training/evaluation source gate.
"""
import argparse,csv,hashlib,inspect,json,math
from collections import defaultdict
from pathlib import Path

NATIVE=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905/result.json')
NATIVE_SHA='51213ef55085d270a56ae2952b8871c06557891d418ae572ec3fb932102f1d8b'
OFFICIAL_SOURCE_SHA='5a09065e2315387405f4cb8f96c0b8fd32d7428996a34eedd92ac4e2a4deeb02'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def account(native,models):
    anchors=native['failure_outcomes'];assert len(anchors)==1765
    groups={};lengths={}
    for name,result in dict(native=native,**models).items():
        outcomes=result['failure_outcomes'];assert set(outcomes)==set(anchors)
        assert result['failure_settings']==native['failure_settings']
        by_sequence=defaultdict(list)
        for key,item in outcomes.items():
            assert all(item[field]==anchors[key][field] for field in ['sequence','anchor','direction','run_length'])
            assert 0<=item['progress']<=item['run_length'] and item['failed']==(item['progress']<item['run_length'])
            by_sequence[item['sequence']].append(item)
        assert len(by_sequence)==127;groups[name]={}
        for sequence,items in by_sequence.items():
            first=outcomes[sequence+'@0F'];assert first['anchor']==0 and first['direction']=='forward'
            length=first['run_length'];lengths[sequence]=length
            survived=sum(item['progress'] for item in items);available=sum(item['run_length'] for item in items)
            failures=sum(item['failed'] for item in items)
            assert failures==result['per_sequence_failures'][sequence]['confirmed_failures']
            assert len(items)==result['per_sequence_failures'][sequence]['anchors']
            groups[name][sequence]=dict(anchors=len(items),survived=survived,available=available,failures=failures,rob=survived/available)
    denominator=sum(lengths.values());assert denominator==80741
    reconstructed={}
    for name,values in groups.items():
        value=100*sum(lengths[s]*v['rob'] for s,v in values.items())/denominator
        recorded=native['metrics_percent']['rob'] if name=='native' else models[name]['metrics_percent']['ROB']
        assert math.isfinite(value) and abs(value-recorded)<1e-10,(name,value,recorded)
        reconstructed[name]=value
    comparisons={}
    for name,result in models.items():
        outcomes=result['failure_outcomes'];rows=[];anchor_rows=[]
        for sequence in sorted(lengths):
            old,new=groups['native'][sequence],groups[name][sequence];weight=lengths[sequence]/denominator
            rows.append(dict(sequence=sequence,sequence_frames=lengths[sequence],weight=weight,anchors=new['anchors'],
                native_ROB_percent=100*old['rob'],model_ROB_percent=100*new['rob'],ROB_contribution_pp=100*weight*(new['rob']-old['rob']),
                native_failures=old['failures'],model_failures=new['failures'],native_survived_anchor_frames=old['survived'],
                model_survived_anchor_frames=new['survived'],available_anchor_frames=new['available']))
        for key in sorted(anchors):
            old,new=anchors[key],outcomes[key];sequence=new['sequence']
            contribution=100*lengths[sequence]/denominator*(new['progress']-old['progress'])/groups[name][sequence]['available']
            anchor_rows.append(dict(anchor_key=key,sequence=sequence,anchor=new['anchor'],direction=new['direction'],run_length=new['run_length'],
                native_failed=old['failed'],model_failed=new['failed'],native_progress=old['progress'],model_progress=new['progress'],
                ROB_contribution_pp=contribution,new_failure=bool(new['failed'] and not old['failed']),
                rescued_failure=bool(old['failed'] and not new['failed'])))
        delta=reconstructed[name]-reconstructed['native']
        assert abs(sum(row['ROB_contribution_pp'] for row in rows)-delta)<1e-10
        assert abs(sum(row['ROB_contribution_pp'] for row in anchor_rows)-delta)<1e-10
        assert sum(r['model_failures'] for r in rows)==result['confirmed_failures']
        ranked=sorted(rows,key=lambda row:(row['ROB_contribution_pp'],row['sequence']))
        comparisons[name]=dict(net_delta_pp=delta,negative_sum_pp=sum(r['ROB_contribution_pp'] for r in rows if r['ROB_contribution_pp']<0),
            positive_sum_pp=sum(r['ROB_contribution_pp'] for r in rows if r['ROB_contribution_pp']>0),
            new_failed_anchors=sum(r['new_failure'] for r in anchor_rows),rescued_failed_anchors=sum(r['rescued_failure'] for r in anchor_rows),
            worst_ten=ranked[:10],best_ten=list(reversed(ranked[-10:])),sequences=rows,anchors=anchor_rows)
    return dict(reconstructed_ROB_percent=reconstructed,total_sequence_frames=denominator,comparisons_vs_native=comparisons)


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(args):
    # Open no result/GT before the complete two-final, six-evaluation seal exists.
    suite=read(args.suite/'all_results.json');selection=read(args.suite/'selection.json')
    assert suite['status']=='six_M122_full_evaluations_complete' and len(suite['results'])==6
    assert suite['selection_sha256']==sha(args.suite/'selection.json')
    assert [r['name'] for r in selection['models']]==['precision0','precision1']
    assert sha(NATIVE)==NATIVE_SHA
    import vot.analysis.multistart as official
    official_path=Path(inspect.getsourcefile(official));assert sha(official_path)==OFFICIAL_SOURCE_SHA
    models={};sources={str(NATIVE):sha(NATIVE),str(official_path):sha(official_path),str(Path(__file__)):sha(__file__)}
    for model in selection['models']:
        name=model['name'];path=args.suite/name/'vot/result.json';result=read(path)
        saved=next(r for r in suite['results'] if r['model']==name and r['dataset']=='vot')
        assert saved['result_path']==str(path) and saved['result_sha256']==sha(path)
        assert result['status']=='complete_full127' and result['model']==name and result['final_sha256']==model['final_sha256']
        assert result['bundle_sha256']==model['bundle_sha256'] and result['bank_sha256']==selection['bank_sha256']
        assert result['binding_sha256']==selection['binding_sha256'] and result['anchors']==1765 and result['sequences']==127
        assert result['external_optimizer_steps']==0 and result['metrics_percent']==saved['metrics_percent']
        assert sha(args.suite/name/'vot/run/merge_result.json')==result['merge_sha256']
        assert sha(args.suite/name/'vot/run/master/analysis'/(name+'_full127.json'))==result['analysis_sha256']
        merge=read(args.suite/name/'vot/run/merge_result.json');assert merge['result_file_count']==5295 and merge['anchor_count']==1765
        assert len(merge['result_sha256'])==5295
        for relative,digest in merge['result_sha256'].items():assert sha(args.suite/name/'vot/run/master'/relative)==digest
        assert sha(args.suite/name/'bundle.json')==model['bundle_sha256']
        bundle=read(args.suite/name/'bundle.json');assert sha(bundle['final_path'])==model['final_sha256']
        models[name]=result;sources[str(path)]=sha(path)
    result=account(read(NATIVE),models);assert not args.output.exists();args.output.mkdir()
    for name,comparison in result['comparisons_vs_native'].items():
        csv_write(args.output/(name+'_sequence_ROB_contributions.csv'),comparison['sequences'])
        csv_write(args.output/(name+'_anchor_ROB_contributions.csv'),comparison['anchors'])
    result.update(status='complete_M122_sealed_ROB_accounting',source_sha256=sources,all_results_sha256=sha(args.suite/'all_results.json'),
        scope='Exact additive accounting of official ROB; not EAO/ACC contribution, deployment improvement, or causal module attribution.',
        formula='100*sum(sequence_length/sum_length*sum_anchor(progress)/sum_anchor(run_length))',
        sequences=127,anchors=1765,neural_executions=0,optimizer_steps=0)
    (args.output/'ROB_accounting.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],reconstructed=result['reconstructed_ROB_percent'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
