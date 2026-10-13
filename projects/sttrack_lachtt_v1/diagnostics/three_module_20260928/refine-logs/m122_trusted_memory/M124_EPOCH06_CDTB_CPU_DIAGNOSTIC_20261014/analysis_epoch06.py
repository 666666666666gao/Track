"""CPU-only diagnosis of sealed M124 epoch-06 CDTB predictions; no tracker replay."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
import cv2,numpy as np

ROOT=Path('/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-06/evaluation/cdtb')
FINAL='4d5c7e411a047fa996835647d1efa3bb94f4184fdc3a592380523be8c04156eb'
METRIC='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
SAVED='dc5d6fd0a0ff931923c4bfdb43bee75a3d8f85195f849938f4e6d2340aa85ada'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def runs(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def main():
    plan_path=ROOT/'plan.json'
    plan=json.loads(plan_path.read_text())
    metric_path=Path(plan['metric_source'])
    assert sha(metric_path)==plan['metric_source_sha256']==METRIC
    spec=importlib.util.spec_from_file_location('locked_M124_epoch06_CDTB_metric',metric_path)
    metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    cases_path=Path(plan['cases_path'])
    cases=json.loads(cases_path.read_text())
    assert len(cases)==80 and sum(c['frames'] for c in cases)==101956
    assert len({c['sequence'] for c in cases})==80 and sha(cases_path)==plan['cases_sha256']
    folder=ROOT/'predictions'
    assert Path(plan['output'])==folder
    saved_path=folder/'metrics.json';receipt_path=folder/'receipt.json'
    saved=json.loads(saved_path.read_text());receipt=json.loads(receipt_path.read_text())
    assert sha(saved_path)==SAVED and saved['status']=='complete'
    assert saved['final_sha256']==receipt['final_sha256']==FINAL
    assert saved['plan_sha256']==receipt['plan_sha256']==sha(plan_path)
    assert saved['receipt_sha256']==sha(receipt_path)
    assert saved['bundle_sha256']==receipt['bundle_sha256']==plan['bundle_sha256']
    assert receipt['status']=='complete_M122_official_OPE_predictions'
    assert receipt['bank_sha256']==plan['bank_sha256']
    assert saved['metrics']['frames']==receipt['frames']==101956
    assert saved['metrics']['sequences']==len(receipt['sequences'])==80
    assert saved['metric_source_sha256']==METRIC
    threshold=saved['metrics']['threshold'];assert threshold is not None
    rows=[];inputs={str(plan_path):sha(plan_path),str(cases_path):sha(cases_path),str(saved_path):sha(saved_path),str(receipt_path):sha(receipt_path),str(metric_path):sha(metric_path)}
    for case,proof in zip(cases,receipt['sequences']):
        name=case['sequence'];assert proof['sequence']==name and proof['frames']==case['frames']
        dataset=Path(plan['dataset_root'])/name
        pred=folder/(name+'.txt');score=folder/(name+'_all_scores.txt');gt=dataset/'groundtruth.txt'
        assert sha(pred)==proof['bbox_sha256'] and sha(score)==proof['confidence_sha256'] and sha(gt)==case['gt_sha256']
        for path in [pred,score,gt]:inputs[str(path)]=sha(path)
        boxes=metric._load_rows(pred,4);truth=metric._load_rows(gt,4);confidence=metric._load_rows(score,1).reshape(-1)
        assert len(boxes)==len(truth)==len(confidence)==case['frames'] and np.isfinite(confidence).all()
        first=next(iter(sorted((dataset/'color').glob('*'))));image=cv2.imread(str(first));assert image is not None
        inputs[str(first)]=sha(first)
        height,width=image.shape[:2]
        overlap,visible=metric._vot_overlaps(boxes,truth,width,height)
        assert visible.sum()>0 and np.isfinite(overlap).all() and ((overlap>=0)&(overlap<=1)).all()
        selected=confidence>=threshold
        precision=float(overlap[selected].mean()) if selected.any() else 1.
        recall=float(overlap[selected].sum()/visible.sum());full=float(overlap.sum()/visible.sum())
        low=visible&(overlap<=.1);inside=np.zeros(len(boxes),dtype=bool)
        for frame in range(1,len(boxes)):
            if not visible[frame]:continue
            x,y,w,h=map(float,boxes[frame-1]);side=math.ceil(math.sqrt(w*h)*4.)
            left,top=round(x+.5*w-.5*side),round(y+.5*h-.5*side)
            cx,cy=truth[frame,:2]+.5*truth[frame,2:]
            inside[frame]=left<=cx<left+side and top<=cy<top+side
        inside[0]=bool(visible[0])
        segments=[dict(start=int(a),end=int(b),length=int(b-a),mean_IoU=float(overlap[a:b].mean()),start_center_in_crop=bool(inside[a]),center_outside_frames=int((~inside[a:b]).sum())) for a,b in runs(low) if b-a>=10]
        reappear=np.flatnonzero(visible[1:]&~visible[:-1])+1
        rows.append(dict(model='M124_epoch06',sequence=name,frames=len(boxes),valid_GT_frames=int(visible.sum()),P_percent=precision*100,R_percent=recall*100,all_saved_boxes_R_percent=full*100,confidence_R_loss_pp=(full-recall)*100,serious_low_IoU_frames=int(low.sum()),serious_low_IoU_crop_inside_frames=int((low&inside).sum()),serious_low_IoU_crop_outside_frames=int((low&~inside).sum()),long_low_IoU_segments=segments,valid_GT_reappearance_count=len(reappear),reappearance_center_outside_count=int((~inside[reappear]).sum()),valid_GT_reappearance_events=[dict(frame=int(f),center_in_crop=bool(inside[f]),IoU=float(overlap[f])) for f in reappear],bbox_sha256=proof['bbox_sha256'],confidence_sha256=proof['confidence_sha256'],GT_sha256=case['gt_sha256'],image_shape=[height,width]))
    P=float(np.mean([r['P_percent'] for r in rows]));R=float(np.mean([r['R_percent'] for r in rows]));F=2*P*R/(P+R)
    assert abs(P-saved['metrics']['precision_percent'])<1e-8 and abs(R-saved['metrics']['recall_percent'])<1e-8 and abs(F-saved['metrics']['f_score_percent'])<1e-8
    aggregate=dict(P=P,R=R,F=F,threshold=threshold,all_saved_boxes_R=float(np.mean([r['all_saved_boxes_R_percent'] for r in rows])),serious_low_IoU_frames=sum(r['serious_low_IoU_frames'] for r in rows),low_crop_outside_frames=sum(r['serious_low_IoU_crop_outside_frames'] for r in rows),long_low_IoU_segment_count=sum(len(r['long_low_IoU_segments']) for r in rows))
    assert aggregate['all_saved_boxes_R']>=R
    value=dict(status='actual_complete_M124_epoch06_CDTB_saved_trajectory_diagnosis',observed_at=datetime.now(timezone.utc).isoformat(),sequences=80,frames=101956,final_sha256=FINAL,metric_source_sha256=METRIC,saved_metrics_sha256=SAVED,aggregate=aggregate,per_sequence=rows,inputs_sha256=inputs,analysis_source_sha256=sha(__file__),evaluation_type='real_gt',NN_calls=0,GPU_queries=0,training_progress_queries=0,limits='Sealed epoch-06 saved predictions and dataset GT only. This is CPU trajectory diagnosis, not a NN replay, text-content intervention, module causality test, full-nine audit or training-time input. Invalid GT does not prove physical absence. Crop centres are reconstructed from saved previous boxes with native factor4 geometry. Segment onset need not be the first causal divergence. All-box R is a fixed-trajectory recall upper bound for confidence-only changes, not a deployable P/F gain.')
    print(json.dumps(value,allow_nan=False),flush=True)

if __name__=='__main__':main()
