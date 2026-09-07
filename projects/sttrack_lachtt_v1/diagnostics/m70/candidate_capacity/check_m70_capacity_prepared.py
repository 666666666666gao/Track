from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,math,subprocess,sys
import cv2
import numpy as np
import torch

BASE=Path('/root/autodl-tmp');INVENTORY=BASE/'sttrack_m70_recovery_window_inventory_20260907'
ROOT=INVENTORY/'candidate_capacity';PARENT=BASE/'sttrack_m65_category_null_support_20260907'
SCRIPT=BASE/'m70_candidate_capacity_20260907.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
s=read(ROOT/'spec.json');assert s['source_sha256']==sha(SCRIPT)
spec=importlib.util.spec_from_file_location('m70_capacity_check_target',str(SCRIPT));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
g,t,checked=m.checked();assert s==checked
subprocess.run(['bash','-n',str(ROOT/'run_capacity.sh')],check=True)
compile(SCRIPT.read_text(),str(SCRIPT),'exec')
code=PARENT/'code';integration=read(PARENT/'integration.json')['source_sha256'];namespace=dict(torch=torch,np=np,math=math,cv=cv2,F=torch.nn.functional)
def original_function(relative,name,cls=None):
    p=code/relative;assert sha(p)==integration[relative];tree=ast.parse(p.read_text())
    nodes=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),namespace)
    return namespace[name]
cal_bbox=original_function('lib/models/layers/head.py','cal_bbox','CenterPredictor')
clip=original_function('lib/utils/box_ops.py','clip_box')
map_back=original_function('lib/test/tracker/sttrack.py','map_box_back','STTrack')
map_local=original_function('lib/test/tracker/sttrack_lachtt_observation.py','_map_local_box')
sample=original_function('lib/train/data/processing_utils.py','sample_target')
torch.manual_seed(7067);torch.set_num_threads(1)
sizes=torch.rand(1,2,16,16);offsets=torch.rand(1,2,16,16);identity=torch.eye(256).reshape(256,1,16,16)
head=SimpleNamespace(feat_sz=16)
bulk=cal_bbox(head,identity,sizes.expand(256,-1,-1,-1),offsets.expand(256,-1,-1,-1))
single=torch.cat([cal_bbox(head,identity[i:i+1],sizes,offsets) for i in range(256)])
assert torch.equal(bulk,single)
# This is an arithmetic contract check using synthetic maps, never an experiment metric.
inventory=read(INVENTORY/'result.json');assert sha(INVENTORY/'geometry_events.json')==inventory['events_sha256']
events=read(INVENTORY/'geometry_events.json')['events'];references={}
for case in s['cases']:
    p=PARENT/'recursive/null'/(case['sequence']+'.json')
    receipt=read(PARENT/'null_recursive_receipt.json');assert sha(p)==next(x['sha256'] for x in receipt['sequences'] if x['sequence']==case['sequence'])
    references[case['sequence']]=read(p)['rows']
window_total=0;crop_checks=0;mapping_checks=0;max_mapping_error=0.;metric_checks=0;max_metric_error=0.
for e in events:
    seq,f=e['sequence'],e['frame'];prior=references[seq][f-1]['bbox']
    assert prior==e['arms']['null']['previous_bbox']
    width,height=e['image_width'],e['image_height'];ws=m.windows(prior,width,height,g)
    assert len(ws)-3==e['null_scale_grid']['windows'];window_total+=len(ws)
    image=cv2.imread(str(Path(t['dataset_root'])/seq/'color'/('%08d.jpg'%(f+1))))
    assert image.shape[:2]==(height,width)
    for wi in sorted(set([0,1,2,3,len(ws)-1])):
        w=ws[wi];patch,resize,mask=sample(image,w['prior'],w['factor'],output_sz=256)
        assert patch.shape==(256,256,3) and mask.shape==(256,256)
        assert resize==256/w['rectangle'][2];crop_checks+=1
        local=(bulk*256/resize).tolist();tracker=SimpleNamespace(state=w['prior'],params=SimpleNamespace(search_size=256))
        for box in local:
            official=clip(map_back(tracker,box,resize),height,width,margin=10)
            diagnostic=map_local(box,w['prior'],256,resize,height,width)
            error=float(np.max(np.abs(np.asarray(official)-np.asarray(diagnostic))));assert error<1e-9
            max_mapping_error=max(max_mapping_error,error);mapping_checks+=1
    gt=np.asarray(e['GT_bbox'])
    for arm in ['null','control','native']:
        # Reference scalar IoUs were already computed from these annotations after sealing.
        if arm=='null':bbox=references[seq][f]['bbox']
        elif arm=='control':bbox=read(PARENT/'recursive/control'/(seq+'.json'))['rows'][f]['bbox']
        else:continue
        actual=float(m.overlaps(np.asarray(bbox).reshape(1,4),gt,np)[0]);reference=e['arms'][arm]['current_iou']
        error=abs(actual-reference);assert error<1e-12;metric_checks+=1;max_metric_error=max(max_metric_error,error)
assert window_total==s['shadow_calls_per_text']==11555
assert s['expected_total_formal_forward_calls']==56218 and s['public_track_calls']==33108
assert not torch.cuda.is_initialized()
for name in ['smoke','shard0','shard1']:assert not (ROOT/name).exists()
assert not (ROOT/'result.json').exists()
r=dict(status='M70_cross_region_capacity_preparation_CPU_checked',observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__),source_sha256=sha(SCRIPT),spec_sha256=sha(ROOT/'spec.json'),queue_sha256=sha(ROOT/'run_capacity.sh'),
    actual_native_functions_used=['sample_target','CenterPredictor.cal_bbox','STTrack.map_box_back','clip_box','_map_local_box'],
    native_bulk_vs_single_peak_boxes_exact_256=True,head_arithmetic_check_uses_synthetic_maps_not_experiment_results=True,
    actual_Train_event_images_checked=len(events),actual_crop_contract_checks=crop_checks,
    native_mapping_checks=mapping_checks,max_mapping_error=max_mapping_error,
    sealed_annotation_metric_checks=metric_checks,max_metric_error=max_metric_error,
    annotations_used_only_for_CPU_metric_contract=True,GT_files_opened=False,
    planned_windows_per_text=window_total,planned_formal_forward_calls=56218,queue_syntax_valid=True,
    learned_head_loaded=False,CUDA_initialized=False,GPU_shadow_state_parity_verified=False,
    new_tracking_calls=0,new_optimizer_steps=0,candidate_capacity_result_exists=False,
    public_evaluation_allowed=False,independent_model_review_pass=False)
(ROOT/'preparation_check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
