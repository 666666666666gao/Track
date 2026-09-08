"""Recompute sealed M79 low22 content controls without new inference."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,importlib.util,io,json,shutil,sys,tarfile

B=Path('/root/autodl-tmp');R=B/'sttrack_m79_vot_content_diagnostic_20260908'
P=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation';O=R/'completed_evidence'
SOURCE=B/'m79_vot_content_diagnostic_20260908.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_bytes((json.dumps(x,indent=2,allow_nan=False)+'\n').encode('utf-8'))

def main():
    import torch
    torch.set_num_threads(1)
    assert not O.exists()
    for name in ['binding','empty_tracking','swapped_tracking','empty_analysis','swapped_analysis','empty','swapped','analysis','controller']:
        assert (R/(name+'.exit')).read_text().strip()=='0',name
    spec=importlib.util.spec_from_file_location('m79_completion_audit',str(SOURCE))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.checked();parent,_=m.parent()
    summary=read(R/'result.json')
    assert summary['status']=='complete_M79_fixed_M78_head_VOT_content_diagnostic'
    assert summary['spec_sha256']==sha(R/'spec.json') and summary['head_sha256']==s['head_sha256']
    assert summary['base_sha256']==s['base_sha256'] and summary['seed']==2027 and summary['additional_seeds']==[]
    assert not summary['model_promotion_allowed'] and not summary['full_three_dataset_evaluation_allowed']
    sys.path.insert(0,str(P/'interface'));from semantic_runtime import checked_plan,text_bank
    plan,bundle=checked_plan(P/'low22_plan.json');original=torch.load(plan['text_bank_path'],map_location='cpu')
    mapping=read(R/'donor_mapping.json');assert len(mapping)==303 and {r['key'] for r in mapping}==set(original['keys'])
    donor={r['key']:r['donor_key'] for r in mapping}
    captions=read(B/'sttrack_m64_category_candidate_20260907/low22_captions/plan.json')
    for arm in ['empty','swapped']:
        ap,ab=checked_plan(R/arm/'plan.json');bank=torch.load(ap['text_bank_path'],map_location='cpu');router=text_bank(ap,ab)
        assert bank['keys']==original['keys']
        for name in ['mask','empty']:assert torch.equal(bank[name],original[name])
        expected=original['tokens'].clone()
        if arm=='empty':expected[original['mask']]=original['empty']
        else:
            for i,key in enumerate(original['keys']):expected[i,0]=original['tokens'][original['keys'].index(donor[key]),0]
        assert torch.equal(bank['tokens'],expected)
        for row in captions['rows']:
            i=bank['keys'].index(row['key']);info=router.info(row['image'],row['init_bbox'])
            assert torch.equal(info['text_tokens'],expected[i]) and torch.equal(info['text_mask'],original['mask'][i])
        for k,v in bundle.items():
            if k not in ['text_protocol_path','text_protocol_sha256']:assert ab[k]==v,k
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    all_results={};reports={};files={}
    for arm in ['category','empty','swapped']:
        root=P if arm=='category' else R/arm;run=root/('low22_run' if arm=='category' else 'run')
        result_path=root/('low22_result.json' if arm=='category' else 'result.json');result=read(result_path)
        assert summary['result_sha256'][arm]==sha(result_path)
        tracker=parent.TRACKER if arm=='category' else s['arms'][arm]['tracker']
        merge=read(run/'merge_result.json');assert merge['anchor_count']==303 and merge['result_file_count']==len(merge['result_sha256'])==909
        assert result['merge_sha256']==sha(run/'merge_result.json')
        expected_manifest=read(P/'low22_execution.json')['manifest_sha256'] if arm=='category' else s['arms'][arm]['manifest_sha256']
        assert merge['source_manifest_sha256']==expected_manifest
        for rel,h in merge['result_sha256'].items():assert sha(run/'master'/rel)==h,rel
        name='m78_category_low22_analysis' if arm=='category' else 'm79_'+arm+'_low22_analysis'
        analysis=run/'master/analysis'/(name+'.json');assert sha(analysis)==result['analysis_sha256']
        d=read(analysis)['results']['baseline']['results']
        metrics=dict(EAO=float(d[0][0][0])*100,ACC=float(d[2][0][0])*100,ROB=float(d[2][0][1])*100)
        outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',tracker,expected_anchors=303)
        assert metrics==result['metrics_percent'] and failures==result['confirmed_failures']
        assert outcomes==result['failure_outcomes'] and per==result['per_sequence_failures'] and settings==result['failure_settings']
        assert sum(v['run_length'] for v in outcomes.values())==220483
        aggregate=dict(metrics_percent=metrics,confirmed_failures=failures)
        assert summary['aggregates'][arm]==aggregate and summary['per_sequence'][arm]==per
        all_results[arm]=result;reports[arm]=dict(aggregate,result_sha256=sha(result_path),verified_saved_files=909,verified_anchors=303,frame_positions=220483)
        files[arm+'__result.json']=result_path;files[arm+'__merge.json']=run/'merge_result.json';files[arm+'__analysis.json']=analysis
    native=read(parent.M39/'m39_result.json')['arms']['default'];primary=all_results['category']
    for arm,result in all_results.items():
        for reference,base in [('category',primary),('native',native)]:
            bm=base['metrics_percent'];bm={k:bm[k.lower()] for k in ['EAO','ACC','ROB']} if reference=='native' else bm
            expected=dict(delta_metrics_percent={k:result['metrics_percent'][k]-bm[k] for k in bm},
                delta_failures=result['confirmed_failures']-base['confirmed_failures'],
                rescued_anchors=[k for k,v in result['failure_outcomes'].items() if not v['failed'] and base['failure_outcomes'][k]['failed']],
                newly_failed_anchors=[k for k,v in result['failure_outcomes'].items() if v['failed'] and not base['failure_outcomes'][k]['failed']])
            assert summary['comparisons'][arm][reference]==expected
    O.mkdir()
    audit=dict(status='completed_M79_all_saved_content_evidence_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
        auditor_sha256=sha(__file__),spec_sha256=sha(R/'spec.json'),result_sha256=sha(R/'result.json'),head_sha256=s['head_sha256'],base_sha256=s['base_sha256'],
        conditions=reports,total_saved_files_verified=2727,total_anchor_trajectories_verified=909,total_frame_positions=661449,
        all606_control_initializations_rebound=True,all_control_tensor_interventions_exact=True,all_official_metrics_and_failure_comparisons_recomputed=True,
        new_inference_calls=0,new_training_steps=0,new_caption_calls=0,official_analysis_rerun=False,seed=2027,additional_seeds=[],
        model_promotion_allowed=False,full_three_dataset_evaluation_allowed=False,independent_model_review_pass=False,goal_achieved=False)
    write(O/'audit.json',audit)
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['sequence','anchors','native_failures','category_failures','empty_failures','swapped_failures'])
    for seq in sorted(primary['per_sequence_failures']):
        writer.writerow([seq,primary['per_sequence_failures'][seq]['anchors'],native['per_sequence_failures'][seq]['confirmed_failures']]+[all_results[arm]['per_sequence_failures'][seq]['confirmed_failures'] for arm in ['category','empty','swapped']])
    (O/'per_sequence_failures.csv').write_bytes(stream.getvalue().encode('utf-8'))
    stream=io.StringIO(newline='');writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(['anchor_key','sequence','anchor','direction','run_length','native_failed','category_failed','empty_failed','swapped_failed'])
    for key,v in primary['failure_outcomes'].items():
        writer.writerow([key,v['sequence'],v['anchor'],v['direction'],v['run_length'],native['failure_outcomes'][key]['failed']]+[all_results[arm]['failure_outcomes'][key]['failed'] for arm in ['category','empty','swapped']])
    (O/'anchor_comparison.csv').write_bytes(stream.getvalue().encode('utf-8'))
    files.update({name:O/name for name in ['audit.json','per_sequence_failures.csv','anchor_comparison.csv']})
    files.update({name:R/name for name in ['result.json','spec.json','controller.exit','empty.exit','swapped.exit','analysis.exit']})
    files.update({Path(__file__).name:Path(__file__),SOURCE.name:SOURCE})
    out=O/'files';out.mkdir();manifest=[]
    for name,src in files.items():
        dest=out/name;shutil.copyfile(src,dest);manifest.append(dict(path=name,bytes=dest.stat().st_size,sha256=sha(dest)))
    write(out/'manifest.json',manifest);archive=O/'published.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for f in sorted(out.iterdir()):
            assert f.is_file() and f.stat().st_size<3000000
            tar.add(f,arcname=f.name)
    print(json.dumps(dict(audit=audit,archive=str(archive),archive_sha256=sha(archive)),indent=2))

if __name__=='__main__':main()
