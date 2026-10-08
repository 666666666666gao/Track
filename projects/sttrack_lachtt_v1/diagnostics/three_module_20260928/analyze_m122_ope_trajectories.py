"""Read-only diagnosis of sealed OPE boxes; never imports a tracker or changes a score."""
import csv,hashlib,importlib.util,json,math
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np

ROOT=Path('/root/autodl-tmp/sttrack_m122_official_evaluation_20261007')
OUTPUT=Path('/root/autodl-tmp/sttrack_m122_ope_posthoc_20261008')
METRIC_SHA='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
EXPECTED_METRICS={
    ('precision0','depthtrack_test'):'04edc9994cc488e8cf7496f2243a58aed18b0777ebbbd8a991b936694154047d',
    ('precision0','cdtb'):'cbfb344f6e05077dbf061669b74fb8d25d618ce425834e2923d9d8d472ee29d7',
    ('precision1','depthtrack_test'):'88002b20f3db413ad09512b29119e5c3121f3c288693c6c78294523f1fb1bce4',
    ('precision1','cdtb'):'fd4c545f30963277fef3e6eb111a7f6e67052912a370bfa7c615d08d90aba7a6'}
inputs={}


def pinned(path,expected=None):
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:assert digest==expected,str(path)
    inputs[str(path)]=digest
    return path


def read(path,expected=None):
    return json.loads(pinned(path,expected).read_text(encoding='utf-8'))


def low_segments(mask):
    ends=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(ends==1),np.flatnonzero(ends==-1)) if b-a>=10]


def main():
    assert not OUTPUT.exists()
    selection=read(ROOT/'selection.json')
    assert selection['status']=='M122_two_actual_third_pass_finals_frozen_before_metrics'
    assert [m['name'] for m in selection['models']]==['precision0','precision1']
    assert not selection['external_metric_checkpoint_selection'] and selection['external_optimizer_steps']==0
    assert not selection['GT_used_for_optimization']
    datasets=[]
    for model in selection['models']:
        name=model['name']
        bundle=read(ROOT/name/'bundle.json',model['bundle_sha256'])
        assert bundle['final_sha256']==model['final_sha256']
        assert bundle['human_review_used_multiframe_aids']
        pinned(bundle['final_path'],model['final_sha256'])
        for dataset,count,frames,target_R in [('depthtrack_test',50,76373,64.9),('cdtb',80,101956,75.6)]:
            root=ROOT/name/dataset
            plan=read(root/'plan.json',model['plans'][dataset]['sha256'])
            assert plan['dataset']==dataset and plan['bundle_sha256']==model['bundle_sha256']
            assert plan['bank_sha256']==selection['bank_sha256']
            assert plan['binding_sha256']==selection['binding_sha256']
            pinned(plan['bank_path'],plan['bank_sha256'])
            pinned(plan['binding_path'],plan['binding_sha256'])
            cases=read(plan['cases_path'],plan['cases_sha256'])
            saved=read(Path(plan['output'])/'metrics.json',EXPECTED_METRICS[(name,dataset)])
            receipt=read(Path(plan['output'])/'receipt.json',saved['receipt_sha256'])
            assert saved['status']=='complete' and receipt['status']=='complete_M122_official_OPE_predictions'
            assert saved['final_sha256']==receipt['final_sha256']==model['final_sha256']
            assert saved['bundle_sha256']==receipt['bundle_sha256']==model['bundle_sha256']
            assert saved['plan_sha256']==receipt['plan_sha256']==model['plans'][dataset]['sha256']
            assert receipt['bank_sha256']==selection['bank_sha256']
            assert not receipt['subsequent_GT_opened'] and receipt['optimizer_steps']==0 and not receipt['text_updated_online']
            assert receipt['frozen_before_after_exact']
            assert len(cases)==len(receipt['sequences'])==saved['metrics']['sequences']==count
            assert sum(c['frames'] for c in cases)==receipt['frames']==saved['metrics']['frames']==frames
            assert saved['metric_source_sha256']==plan['metric_source_sha256']==METRIC_SHA
            spec=importlib.util.spec_from_file_location('M122_locked_OPE_posthoc_metric',str(pinned(plan['metric_source'],METRIC_SHA)))
            metric=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metric)
            threshold=saved['metrics']['threshold']
            assert threshold is not None and math.isfinite(threshold)
            rows=[]
            for case,record in zip(cases,receipt['sequences']):
                seq=case['sequence']
                assert record['sequence']==seq and record['frames']==case['frames']
                folder=Path(plan['dataset_root'])/seq
                boxes=metric._load_rows(pinned(Path(plan['output'])/(seq+'.txt'),record['bbox_sha256']),4)
                score=metric._load_rows(pinned(Path(plan['output'])/(seq+'_all_scores.txt'),record['confidence_sha256']),1).reshape(-1)
                gt=metric._load_rows(pinned(folder/'groundtruth.txt',case['gt_sha256']),4)
                assert len(boxes)==len(score)==len(gt)==case['frames'] and np.isfinite(score).all()
                first=next(iter(sorted((folder/'color').glob('*'))))
                image=cv2.imread(str(pinned(first)))
                assert image is not None
                height,width=image.shape[:2]
                overlaps,visible=metric._vot_overlaps(boxes,gt,width,height)
                assert visible.any() and np.isfinite(overlaps).all() and ((overlaps>=0)&(overlaps<=1)).all()
                selected=score>=threshold
                precision=float(overlaps[selected].mean()) if selected.any() else 1.
                recall=float(overlaps[selected].sum()/visible.sum())
                all_recall=float(overlaps.sum()/visible.sum())
                severe=visible&(overlaps<=.1)
                segments=low_segments(severe)
                rows.append(dict(sequence=seq,frames=len(boxes),valid_GT_frames=int(visible.sum()),selected_frames=int(selected.sum()),
                    reported_precision_percent=100*precision,reported_recall_percent=100*recall,
                    reported_sequence_F_percent=100*(2*precision*recall/(precision+recall) if precision+recall>0 else 0.),
                    all_boxes_recall_percent=100*all_recall,score_selection_cost_pp=100*(all_recall-recall),
                    valid_GT_mean_IoU=float(overlaps[visible].mean()),severe_low_overlap_frames=int(severe.sum()),
                    H10_segments=len(segments),longest_H10_frames=max([b-a for a,b in segments]+[0]),
                    H10_zero_based_half_open=segments))
            mean_P=float(np.mean([row['reported_precision_percent'] for row in rows]))
            mean_R=float(np.mean([row['reported_recall_percent'] for row in rows]))
            mean_F=2*mean_P*mean_R/(mean_P+mean_R)
            assert abs(mean_P-saved['metrics']['precision_percent'])<1e-8
            assert abs(mean_R-saved['metrics']['recall_percent'])<1e-8
            assert abs(mean_F-saved['metrics']['f_score_percent'])<1e-8
            cap=float(np.mean([row['all_boxes_recall_percent'] for row in rows]))
            datasets.append(dict(model=name,dataset=dataset,sequences=count,frames=frames,official_threshold=threshold,
                official_P=mean_P,official_R=mean_R,official_F=mean_F,all_boxes_R=cap,
                score_selection_cost_pp=cap-mean_R,target_R=target_R,fixed_saved_boxes_can_reach_target_R=cap>=target_R,
                metric_sha256=EXPECTED_METRICS[(name,dataset)],per_sequence=rows))
    contrasts=[]
    for dataset,count in [('depthtrack_test',50),('cdtb',80)]:
        a=next(x for x in datasets if x['model']=='precision0' and x['dataset']==dataset)
        b=next(x for x in datasets if x['model']=='precision1' and x['dataset']==dataset)
        assert [x['sequence'] for x in a['per_sequence']]==[x['sequence'] for x in b['per_sequence']]
        ledger=[dict(sequence=x['sequence'],P0_all_boxes_R=x['all_boxes_recall_percent'],P1_all_boxes_R=y['all_boxes_recall_percent'],
            delta_all_boxes_R_pp=y['all_boxes_recall_percent']-x['all_boxes_recall_percent'],
            macro_R_contribution_pp=(y['all_boxes_recall_percent']-x['all_boxes_recall_percent'])/count) for x,y in zip(a['per_sequence'],b['per_sequence'])]
        assert abs(sum(x['macro_R_contribution_pp'] for x in ledger)-(b['all_boxes_R']-a['all_boxes_R']))<1e-8
        contrasts.append(dict(dataset=dataset,scope='signed arithmetic difference between two whole recursive trajectories; no causal module attribution',per_sequence=ledger))
    for path,digest in inputs.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    report=dict(status='complete_M122_sealed_OPE_readonly_trajectory_diagnosis',created_at=datetime.now(timezone.utc).isoformat(),
        scope='Fixed saved boxes and trajectories; confidence filtering removed only for the recall cap. Per-sequence PR/F use each model-dataset original official threshold. Invalid GT means invalid/unknown, not necessarily physical absence.',
        optimizer_steps=0,neural_calls=0,predictions_or_scores_changed=False,exact_crop_reconstructed=False,
        full_VOT_claim=False,text_causal_increment_claim=False,H10_is_not_VOT_ROB=True,human_review_used_multiframe_aids=True,
        dataset_aggregated_F_is_harmonic_of_macro_P_R_not_mean_sequence_F=True,
        datasets=datasets,contrasts=contrasts,input_sha256=inputs,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUTPUT.mkdir()
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns=['model','dataset','sequence','frames','valid_GT_frames','selected_frames','reported_precision_percent','reported_recall_percent',
        'reported_sequence_F_percent','all_boxes_recall_percent','score_selection_cost_pp','valid_GT_mean_IoU','severe_low_overlap_frames','H10_segments','longest_H10_frames']
    with (OUTPUT/'per_sequence.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        for item in datasets:
            for row in item['per_sequence']:writer.writerow(dict(model=item['model'],dataset=item['dataset'],**{key:row[key] for key in columns[2:]}))
    print(json.dumps(dict(status=report['status'],datasets=[{k:v for k,v in item.items() if k!='per_sequence'} for item in datasets]),indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
