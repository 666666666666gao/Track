"""Complete OPE predictions/locked metrics and TraX multi-start entry with no inference GT."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np
from m122_current_C_official_runtime import checked_plan,OfficialDenseTracker
from bind_m122_official_initializations import sha


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def ope_track(path):
    plan,bundle=checked_plan(path);assert plan['dataset'] in ['depthtrack_test','cdtb']
    assert sha(plan['cases_path'])==plan['cases_sha256'];cases=json.loads(Path(plan['cases_path']).read_text())
    assert len(cases)=={'depthtrack_test':50,'cdtb':80}[plan['dataset']]
    assert sum(c['frames'] for c in cases)=={'depthtrack_test':76373,'cdtb':101956}[plan['dataset']]
    actor=OfficialDenseTracker(plan,bundle);frozen=actor.actor.frozen_digest()
    from train_template_write_C import state_digest
    C_before=state_digest(actor.trusted.writer)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    output=Path(plan['output']);output.mkdir(exist_ok=False);started=time.time();receipts=[]
    for case in cases:
        folder=Path(plan['dataset_root'])/case['sequence'];boxes=[case['init_bbox']];scores=[1.]
        for frame in range(case['frames']):
            stem='%08d'%(frame+1);rgb=folder/'color'/(stem+'.jpg');depth=folder/'depth'/(stem+'.png')
            image=get_rgbd_frame(str(rgb),str(depth),dtype='rgbcolormap',depth_clip=True)
            if frame==0:actor.initialize(image,case['init_bbox'],rgb)
            else:
                prediction=actor.track(image);boxes.append(prediction['target_bbox']);scores.append(prediction['best_score'])
        boxes=np.asarray(boxes);scores=np.asarray(scores)
        assert boxes.shape==(case['frames'],4) and np.isfinite(boxes).all() and np.isfinite(scores).all() and (boxes[:,2:]>0).all()
        box_file=output/(case['sequence']+'.txt');score_file=output/(case['sequence']+'_all_scores.txt')
        np.savetxt(box_file,boxes,fmt='%.6f',delimiter=',');np.savetxt(score_file,scores,fmt='%.6f')
        assert np.abs(np.loadtxt(box_file,delimiter=',').reshape(-1,4)-boxes).max()<=5.01e-7
        assert np.abs(np.loadtxt(score_file).reshape(-1)-scores).max()<=5.01e-7
        record=dict(sequence=case['sequence'],frames=case['frames'],bbox_sha256=sha(box_file),confidence_sha256=sha(score_file),elapsed_seconds=time.time()-started)
        receipts.append(record);print(json.dumps(record),flush=True)
    assert actor.actor.frozen_digest()==frozen and state_digest(actor.trusted.writer)==C_before;checked_plan(path)
    write(output/'receipt.json',dict(status='complete_M122_official_OPE_predictions',plan_sha256=sha(path),final_sha256=bundle['final_sha256'],
        bank_sha256=plan['bank_sha256'],bundle_sha256=plan['bundle_sha256'],sequences=receipts,frames=sum(r['frames'] for r in receipts),
        subsequent_GT_opened=False,optimizer_steps=0,text_updated_online=False,frozen_before_after_exact=True,elapsed_seconds=time.time()-started))


def ope_analyze(path):
    plan,bundle=checked_plan(path);output=Path(plan['output']);receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['status']=='complete_M122_official_OPE_predictions' and receipt['plan_sha256']==sha(path)
    assert receipt['final_sha256']==bundle['final_sha256'] and receipt['bank_sha256']==plan['bank_sha256']
    assert sha(plan['cases_path'])==plan['cases_sha256'];cases=json.loads(Path(plan['cases_path']).read_text())
    assert [r['sequence'] for r in receipt['sequences']]==[r['sequence'] for r in cases]
    for case,row in zip(cases,receipt['sequences']):
        assert case['frames']==row['frames']
        assert sha(output/(case['sequence']+'.txt'))==row['bbox_sha256'] and sha(output/(case['sequence']+'_all_scores.txt'))==row['confidence_sha256']
        # Current/future evaluation annotations are opened only after all predictions are sealed.
        assert sha(Path(plan['dataset_root'])/case['sequence']/'groundtruth.txt')==case['gt_sha256']
    assert sha(plan['metric_source'])==plan['metric_source_sha256']=='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
    spec=importlib.util.spec_from_file_location('locked_M122_OPE_metric',plan['metric_source']);metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    values=metric.evaluate_depthtrack_results(plan['dataset_root'],output,resolution=100,sequence_names=[c['sequence'] for c in cases])
    assert values['sequences']==len(cases) and values['frames']==receipt['frames']
    write(output/'metrics.json',dict(status='complete',metrics=values,final_sha256=bundle['final_sha256'],bundle_sha256=plan['bundle_sha256'],
        plan_sha256=sha(path),receipt_sha256=sha(output/'receipt.json'),metric_source_sha256=plan['metric_source_sha256']))
    print(json.dumps(values,indent=2),flush=True)


def vot_track(path):
    plan,bundle=checked_plan(path);actor=OfficialDenseTracker(plan,bundle)
    bridge_spec=importlib.util.spec_from_file_location('M122_locked_vot_bridge',bundle['vot_bridge_path'])
    vot=importlib.util.module_from_spec(bridge_spec);bridge_spec.loader.exec_module(vot)
    from lib.train.dataset.depth_utils import get_rgbd_frame
    handle=vot.VOT('rectangle',channels='rgbd');bbox=list(handle.region());files=handle.frame()
    assert isinstance(files,list) and len(files)==2
    image=get_rgbd_frame(files[0],files[1],dtype='rgbcolormap',depth_clip=True);actor.initialize(image,bbox,files[0])
    while True:
        files=handle.frame()
        if files is None:break
        assert isinstance(files,list) and len(files)==2
        image=get_rgbd_frame(files[0],files[1],dtype='rgbcolormap',depth_clip=True);out=actor.track(image)
        handle.report(vot.Rectangle(*out['target_bbox']),out['best_score'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--mode',choices=['ope_track','ope_analyze','vot_track'],required=True);args=parser.parse_args()
    {'ope_track':ope_track,'ope_analyze':ope_analyze,'vot_track':vot_track}[args.mode](args.plan)
