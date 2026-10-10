"""Collect one complete nine-metric row for the fixed composite final."""
import argparse,csv,math
from pathlib import Path
from bind_m122_official_initializations import sha
from prepare_m122_evaluation_suite import read,write


def main(root):
    selection=read(root/'selection.json');rows=[];gates=[]
    assert selection['status']=='M123_ABC_joint_composite_final_fixed_before_metrics' and len(selection['models'])==1
    for model in selection['models']:
        name=model['name'];target=root;bundle=read(target/'bundle.json');metrics={}
        assert sha(target/'bundle.json')==model['bundle_sha256'] and sha(bundle['final_path'])==model['final_sha256']
        for path,digest in bundle['source_sha256'].items():assert sha(path)==digest,path
        for dataset,count,frames in [('depthtrack_test',50,76373),('cdtb',80,101956)]:
            plan=model['plans'][dataset];folder=target/dataset/'predictions'
            result=read(folder/'metrics.json');receipt=read(folder/'receipt.json')
            assert result['status']=='complete' and result['final_sha256']==model['final_sha256'] and result['bundle_sha256']==model['bundle_sha256']
            assert result['plan_sha256']==sha(plan['path'])==plan['sha256'] and result['receipt_sha256']==sha(folder/'receipt.json')
            assert receipt['bank_sha256']==selection['bank_sha256'] and len(receipt['sequences'])==count and receipt['frames']==frames
            assert receipt['frozen_before_after_exact'] and receipt['optimizer_steps']==0 and not receipt['subsequent_GT_opened']
            for r in receipt['sequences']:
                assert sha(folder/(r['sequence']+'.txt'))==r['bbox_sha256']
                assert sha(folder/(r['sequence']+'_all_scores.txt'))==r['confidence_sha256']
            values=result['metrics'];assert values['sequences']==count and values['frames']==frames
            metrics[dataset]={key:values[key+'_percent'] for key in ['precision','recall','f_score']}
            rows.append(dict(model=name,dataset=dataset,final_sha256=model['final_sha256'],bundle_sha256=model['bundle_sha256'],
                result_path=str(folder/'metrics.json'),result_sha256=sha(folder/'metrics.json'),metrics_percent=metrics[dataset]))
        vot=read(target/'vot/result.json')
        assert vot['status']=='complete_full127' and vot['final_sha256']==model['final_sha256'] and vot['bundle_sha256']==model['bundle_sha256']
        assert vot['bank_sha256']==selection['bank_sha256'] and vot['binding_sha256']==selection['binding_sha256']
        assert vot['anchors']==1765 and vot['sequences']==127 and vot['external_optimizer_steps']==0
        merge=read(target/'vot/run/merge_result.json');assert vot['merge_sha256']==sha(target/'vot/run/merge_result.json')
        for rel,digest in merge['result_sha256'].items():assert sha(target/'vot/run/master'/rel)==digest
        assert vot['analysis_sha256']==sha(target/'vot/run/master/analysis'/(name+'_full127.json'))
        metrics['vot']=vot['metrics_percent']
        rows.append(dict(model=name,dataset='vot',final_sha256=model['final_sha256'],bundle_sha256=model['bundle_sha256'],
            result_path=str(target/'vot/result.json'),result_sha256=sha(target/'vot/result.json'),metrics_percent=metrics['vot'],confirmed_failures=vot['confirmed_failures']))
        assert all(math.isfinite(v) and 0<=v<=100 for group in metrics.values() for v in group.values())
        dt,cd,v=metrics['depthtrack_test'],metrics['cdtb'],metrics['vot']
        passed=dict(depthtrack_P=dt['precision']>=65.2,depthtrack_R=dt['recall']>=64.9,depthtrack_F=dt['f_score']>=65.1,
            cdtb_P=cd['precision']>=72.9,cdtb_R=cd['recall']>=75.6,cdtb_F=cd['f_score']>=74.2,
            vot_EAO=v['EAO']>77.9,vot_ACC=v['ACC']>82.1,vot_ROB=v['ROB']>93.7)
        gates.append(dict(model=name,final_sha256=model['final_sha256'],metrics_percent=metrics,passed=passed,joint_pass=all(passed.values())))
    assert len(rows)==3
    write(root/'all_results.json',dict(status='three_M123_ABC_joint_full_evaluations_complete',selection_sha256=sha(root/'selection.json'),models=selection['models'],
        results=rows,gates=gates,any_joint_pass=any(g['joint_pass'] for g in gates),external_metric_checkpoint_selection=False,
        human_review_used_multiframe_aids=True,external_optimizer_steps=0,independent_completed_audit=False))
    with (root/'metrics.csv').open('w',newline='') as f:
        fields=['model','final_sha256','DepthTrack_P','DepthTrack_R','DepthTrack_F','CDTB_P','CDTB_R','CDTB_F','VOT_EAO','VOT_ACC','VOT_ROB','joint_pass']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for g in gates:
            m=g['metrics_percent'];writer.writerow(dict(model=g['model'],final_sha256=g['final_sha256'],joint_pass=g['joint_pass'],
                DepthTrack_P=m['depthtrack_test']['precision'],DepthTrack_R=m['depthtrack_test']['recall'],DepthTrack_F=m['depthtrack_test']['f_score'],
                CDTB_P=m['cdtb']['precision'],CDTB_R=m['cdtb']['recall'],CDTB_F=m['cdtb']['f_score'],
                VOT_EAO=m['vot']['EAO'],VOT_ACC=m['vot']['ACC'],VOT_ROB=m['vot']['ROB']))
    print('complete: '+str([dict(model=g['model'],joint_pass=g['joint_pass']) for g in gates]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
