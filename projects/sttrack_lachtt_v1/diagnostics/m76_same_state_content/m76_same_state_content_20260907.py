"""Exact Category prefix replay with noncommitting same-state content/head probes."""
import argparse,copy,hashlib,importlib.util,json,math,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

B=Path('/root/autodl-tmp');P=B/'sttrack_m73_paired_lexical_replication_20260907/seed2027'
M74=B/'sttrack_m74_m73_content_diagnostic_20260907';M75=B/'sttrack_m75_m73_failure_geometry_20260907'
R=B/'sttrack_m76_same_state_content_20260907'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def many_iou(boxes,g):
 a=np.asarray(boxes,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
 wh=np.maximum(0,np.minimum(a[:,:2]+a[:,2:],g[:2]+g[2:])-np.maximum(a[:,:2],g[:2]))
 inter=wh[:,0]*wh[:,1];return inter/(a[:,2]*a[:,3]+g[2]*g[3]-inter)
def parents():
 source=B/'m74_m73_content_diagnostic_20260907.py'
 assert sha(source)=='819b5a629bd59715a0ce7a106e3dbfcc7780c4d0078fed9731cfb6a1a8d891eb'
 modspec=importlib.util.spec_from_file_location('m76_parents74',str(source));m=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(m)
 s,t,a=m.checked()
 assert sha(M74/'result.json')=='cad42b93712f017af9b79c615f7e8264de614c0b982d0e32be9d398927089820'
 assert sha(M74/'completed_evidence_audit.json')=='ddba08da45fddf83b2ddd56195a01a72c8af6bc7971a83f27f26569f3e5b9d84'
 assert sha(M75/'result.json')=='18d79edfb53ee3e908d98a9b8011163272efa13958fac08b4de9005636ecf268'
 return s,t,a
def prepare():
 import torch
 s,t,a=parents();assert not R.exists()
 events=[v for v in read(M75/'result.json')['events'] if v['strict_content_harm'] and v['arms']['category']['GT_center_inside_search']]
 assert len(events)==4
 cases=[];windows={};selected=[]
 for e in events:
  seq=e['sequence'];case=next(c for c in s['cases'] if c['sequence']==seq)
  gp=Path(t['dataset_root'])/seq/'groundtruth.txt';assert sha(gp)==case['gt_sha256'];gt=np.loadtxt(gp,delimiter=',').reshape(-1,4)
  rows=read(P/'recursive/category'/(seq+'.json'))['rows'];valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
  earlier=[i for i in range(e['start']) if valid[i] and many_iou([rows[i]['bbox']],gt[i])[0]>=.5]
  assert earlier
  last=earlier[-1];start=max(1,last-2);end=e['end_exclusive']
  assert seq not in windows
  windows[seq]=[start,end];cases.append(dict(case,stop_frame_exclusive=end))
  selected.append(dict(sequence=seq,H10_start=e['start'],H10_end_exclusive=end,last_valid_correct_frame=last,capture_start=start,capture_end_exclusive=end))
 head=torch.load(P/'training/category/final.pth',map_location='cpu')
 assert head['status']=='complete' and sum(v.numel() for v in head['model'].values())==289154
 assert all(torch.isfinite(v).all() for v in head['model'].values())
 R.mkdir()
 queue='''#!/bin/bash
set -u
cd /root/autodl-tmp/sttrack_m76_same_state_content_20260907 || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m76_same_state_content_20260907.py run > replay.log 2>&1
status=$?; printf '%s\\n' "$status" > replay.exit
if [ "$status" -ne 0 ]; then printf '%s\\n' "$status" > controller.exit; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/m76_same_state_content_20260907.py analyze > analysis.log 2>&1
status=$?; printf '%s\\n' "$status" > analysis.exit; printf '%s\\n' "$status" > controller.exit
exit "$status"
'''
 (R/'run.sh').write_text(queue);subprocess.run(['bash','-n',str(R/'run.sh')],check=True)
 spec=dict(status='frozen_M76_post_result_same_state_content_probe',observed_utc=now(),source_sha256=sha(__file__),queue_sha256=sha(R/'run.sh'),M74_spec_sha256=sha(M74/'spec.json'),M74_result_sha256=sha(M74/'result.json'),M74_audit_sha256=sha(M74/'completed_evidence_audit.json'),M75_result_sha256=sha(M75/'result.json'),head_sha256=sha(P/'training/category/final.pth'),cases=cases,capture_windows=windows,selected_intervals=selected,banks=s['banks'],seed=2027,
  capture_rule='Four strict M74 harms whose GT centers remain in the Category search at H10 onset. Capture from two frames before the latest valid correct Category frame through the H10 end, plus every scheduled template-check frame in each replayed prefix.',
  new_track_calls=sum(c['stop_frame_exclusive']-1 for c in cases),estimated_runtime_seconds=450,poll_seconds=240,
  heads=['category','empty','swapped','unadapted_same_state'],GT_policy='GT selects diagnostic windows during CPU preparation and scores sealed probes afterward. Replay/probes receive only frozen initialization boxes, images and frozen text banks.',
  state_policy='Run the original Category prefix exactly. Clone the actual six pre-adapter inputs. Read Empty/Swapped using the same trained adapter and unadapted head using the same pre-adapter fused features; never commit probe bbox/query/template/semantic state.',
  interpretation='These are one-frame counterfactuals on the Category trajectory, not alternative recursive recoveries or an independent native baseline. Deliberately selected four reused Train development harms, not an unbiased distribution.',
  new_optimizer_steps=0,new_training_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False)
 write(R/'spec.json',spec)
 write(R/'preparation.json',dict(status='M76_CPU_bound_sources_banks_checkpoint_and_windows_verified',observed_utc=now(),spec_sha256=sha(R/'spec.json'),source_sha256=sha(__file__),finite_learned_parameters=289154,new_tracking_calls=0,new_optimizer_steps=0,new_training_seeds=[]))
 print(json.dumps(dict(spec_sha256=sha(R/'spec.json'),track_calls=spec['new_track_calls'],selected=selected),indent=2))
def checked():
 old,t,a=parents();s=read(R/'spec.json')
 assert s['source_sha256']==sha(__file__) and s['queue_sha256']==sha(R/'run.sh')
 for n,b in s['banks'].items():assert sha(b['path'])==b['sha256']
 assert s['seed']==2027 and not s['new_training_seeds'] and not s['public_evaluation_allowed']
 return s,t
def run():
 s,t=checked()
 import torch
 from types import SimpleNamespace
 sys.path.insert(0,str(P/'code'))
 from lib.config.sttrack.config import cfg,update_config_from_file
 from lib.test.tracker.sttrack_semantic import STTrackSemantic
 from lib.test.tracker.sttrack_lachtt_observation import decode_nms_candidates,_map_local_box
 from lib.train.dataset.depth_utils import get_rgbd_frame
 torch.set_num_threads(1);torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
 update_config_from_file(str(P/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
 params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
 tracker=STTrackSemantic(params,str(P/'training/category/final.pth'));assert not tracker.network.training and tracker.use_text
 banks={n:torch.load(b['path'],map_location='cpu') for n,b in s['banks'].items()}
 capture={'active':False};real=tracker.network.forward
 def public(*args,**kwargs):
  out=real(*args,**kwargs)
  if capture['active']:capture['out']={k:out[0][k].detach().clone() for k in ['score_map','size_map','offset_map']}
  return out
 def before(module,args):
  if capture['active']:
   assert len(args)==6;capture['inputs']=tuple(v.detach().clone() for v in args)
 tracker.network.forward=public;hook=tracker.network.semantic_adapter.register_forward_pre_hook(before)
 eye=torch.eye(256,device='cuda').reshape(256,1,16,16)
 def decode(out,prior,shape):
  side=math.ceil(math.sqrt(prior[2]*prior[3])*4.);resize=256./side;response=tracker.output_window*out['score_map']
  nms=decode_nms_candidates(response,out['size_map'],out['offset_map'],[prior],[resize],shape,256,10)
  local=tracker.network.box_head.cal_bbox(eye,out['size_map'].expand(256,-1,-1,-1),out['offset_map'].expand(256,-1,-1,-1))
  dense=[_map_local_box(v,prior,256,resize,shape[0],shape[1]) for v in (local*256/resize).cpu().tolist()]
  index=int(response.flatten().argmax());assert np.max(np.abs(np.asarray(dense[index])-nms[0]['bbox']))<1e-4
  return dict(nms=nms,dense_boxes=dense,hann_scores=response.flatten().cpu().tolist(),raw_scores=out['score_map'].flatten().cpu().tolist(),selected_index=index,raw_selected_index=int(out['score_map'].flatten().argmax()))
 def equal(a,b):
  if torch.is_tensor(a):assert torch.equal(a,b)
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for u,v in zip(a,b):equal(u,v)
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:equal(a[k],b[k])
  else:assert a==b
 output=R/'replay';output.mkdir();started=time.time();receipts=[]
 original_receipt=read(P/'category_recursive_receipt.json');refs={v['sequence']:v for v in original_receipt['sequences']}
 for case in s['cases']:
  seq=case['sequence'];folder=Path(t['dataset_root'])/seq;path=P/'recursive/category'/(seq+'.json');assert sha(path)==refs[seq]['sha256'];reference=read(path)['rows']
  def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
  j=banks['category']['sequences'].index(seq);capture['active']=False
  tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=banks['category']['tokens'][j],text_mask=banks['category']['mask'][j],empty_text=banks['category']['empty']))
  alt={n:banks[n]['tokens'][banks[n]['sequences'].index(seq)].float().cuda().unsqueeze(0) for n in ['empty','swapped']}
  probes=[];geometry=[];begin,end=s['capture_windows'][seq]
  for i in range(1,case['stop_frame_exclusive']):
   prior=list(tracker.state);image=frame(i);picked=begin<=i<end or i%50==0;capture['active']=picked
   prediction=tracker.track(image);capture['active']=False
   assert list(prediction['target_bbox'])==reference[i]['bbox'] and float(prediction['best_score'])==reference[i]['score'],(seq,i)
   side=math.ceil(math.sqrt(prior[2]*prior[3])*4.);crop=[round(prior[0]+prior[2]/2-side/2),round(prior[1]+prior[3]/2-side/2),side,side]
   geometry.append(dict(frame=i,previous_bbox=prior,bbox=list(tracker.state),score=float(prediction['best_score']),search_rectangle=crop,template_write=i%50==0 and float(prediction['best_score'])>.75))
   if picked:
    state=[tracker.state,tracker.track_query_before,tracker.z_dict,tracker.box_mask_z,tracker.frame_id,tracker.semantic_context];saved=copy.deepcopy(state)
    inputs=capture['inputs'];heads={}
    with torch.no_grad():
     heads['category']=decode(capture['out'],prior,image.shape)
     heads['unadapted_same_state']=decode(tracker.network.forward_head(inputs[2].clone()),prior,image.shape)
     for name in ['empty','swapped']:
      enhanced,_=tracker.network.semantic_adapter(inputs[0],inputs[1],inputs[2],inputs[3],alt[name],inputs[5])
      heads[name]=decode(tracker.network.forward_head(enhanced),prior,image.shape)
    equal(state,saved)
    own=heads['category'];assert abs(own['hann_scores'][own['selected_index']]-reference[i]['score'])<1e-8
    assert np.max(np.abs(np.asarray(own['dense_boxes'][own['selected_index']])-reference[i]['bbox']))<1e-4
    probes.append(dict(frame=i,heads=heads))
  target=output/(seq+'.json');write(target,dict(sequence=seq,reference_sha256=sha(path),all_public_boxes_scores_exact=True,stop_frame_exclusive=case['stop_frame_exclusive'],geometry=geometry,probes=probes))
  item=dict(sequence=seq,frames=case['stop_frame_exclusive'],probe_frames=len(probes),sha256=sha(target),elapsed_seconds=time.time()-started);receipts.append(item);print(json.dumps(item),flush=True)
 hook.remove();checked()
 receipt=dict(status='complete_exact_Category_prefixes_and_noncommitting_content_probes',source_sha256=sha(__file__),spec_sha256=sha(R/'spec.json'),head_sha256=s['head_sha256'],sequences=receipts,new_track_calls=sum(v['frames']-1 for v in receipts),all_public_boxes_scores_exact=True,probe_state_unchanged=True,subsequent_GT_opened=False,new_optimizer_steps=0,new_training_seeds=[],elapsed_seconds=time.time()-started)
 write(R/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
def analyze():
 s,t=checked();assert (R/'replay.exit').read_text().strip()=='0'
 receipt=read(R/'receipt.json');assert receipt['spec_sha256']==sha(R/'spec.json') and receipt['new_track_calls']==s['new_track_calls']
 assert receipt['all_public_boxes_scores_exact'] and receipt['probe_state_unchanged'] and not receipt['subsequent_GT_opened']
 data={}
 for item,case in zip(receipt['sequences'],s['cases']):
  path=R/'replay'/(case['sequence']+'.json');assert sha(path)==item['sha256'];v=read(path)
  assert v['sequence']==item['sequence']==case['sequence'] and v['stop_frame_exclusive']==case['stop_frame_exclusive']
  assert len(v['geometry'])==case['stop_frame_exclusive']-1 and len(v['probes'])==item['probe_frames'];data[case['sequence']]=v
 # Later GT opens only after the complete, parity-checked prefix/probe family seals.
 evaluated={};onsets=[];summary={}
 for case in s['cases']:
  seq=case['sequence'];p=Path(t['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256'];gt=np.loadtxt(p,delimiter=',').reshape(-1,4)
  valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1);rows=[]
  for probe in data[seq]['probes']:
   i=probe['frame'];geometry=data[seq]['geometry'][i-1];assert geometry['frame']==i
   base=dict(frame=i,valid_GT=bool(valid[i]),scheduled_template_check=i%50==0,actual_template_write=geometry['template_write'],heads={})
   for name,head in probe['heads'].items():
    h=head['selected_index'];c=probe['heads']['category']['selected_index'];score=head['hann_scores'][h]
    out=dict(selected_index=h,selected_score=score,peak_changed_vs_category=h!=c,scheduled_write_if_this_head=i%50==0 and score>.75)
    if valid[i]:
     v=many_iou(head['dense_boxes'],gt[i]);n=many_iou([z['bbox'] for z in head['nms']],gt[i]);hit=np.flatnonzero(n>=.5)
     out.update(selected_iou=float(v[h]),raw_selected_iou=float(v[head['raw_selected_index']]),dense_best_iou=float(v.max()),top10_best_iou=float(n.max()),first_correct_rank=int(hit[0])+1 if len(hit) else None)
    base['heads'][name]=out
   if valid[i]:
    c=geometry['search_rectangle'];g=gt[i];cx,cy=g[:2]+g[2:]/2
    base.update(GT=g.tolist(),search_rectangle=c,center_inside=bool(c[0]<=cx<c[0]+c[2] and c[1]<=cy<c[1]+c[3]))
    assert abs(base['heads']['category']['selected_iou']-many_iou([geometry['bbox']],g)[0])<1e-5
   rows.append(base)
  evaluated[seq]=rows;by_frame={v['frame']:v for v in rows};selection=next(v for v in s['selected_intervals'] if v['sequence']==seq)
  onsets.append(dict(selection,probe=by_frame[selection['H10_start']]))
  window=[v for v in rows if selection['capture_start']<=v['frame']<selection['capture_end_exclusive'] and v['valid_GT']]
  low=[v for v in window if v['heads']['category']['selected_iou']<=.1];good=[v for v in window if v['heads']['category']['selected_iou']>=.5]
  summary[seq]=dict(valid_window_probes=len(window),category_low_probes=len(low),category_correct_probes=len(good),category_low_center_inside=sum(v['center_inside'] for v in low),heads={name:dict(low_probe_top1_correct=sum(v['heads'][name]['selected_iou']>=.5 for v in low),low_probe_top10_correct=sum(v['heads'][name]['top10_best_iou']>=.5 for v in low),low_probe_dense_correct=sum(v['heads'][name]['dense_best_iou']>=.5 for v in low),correct_probe_severely_broken=sum(v['heads'][name]['selected_iou']<=.1 for v in good)) for name in s['heads']})
 result=dict(status='completed_M76_post_sealing_same_state_content_diagnostic',observed_utc=now(),source_sha256=sha(__file__),spec_sha256=sha(R/'spec.json'),receipt_sha256=sha(R/'receipt.json'),summary=summary,onsets=onsets,new_tracking_calls=0,new_optimizer_steps=0,new_training_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False,scope=s['interpretation'])
 write(R/'evaluated_probes.json',evaluated);write(R/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze','check']);a=p.parse_args()
 {'prepare':prepare,'run':run,'analyze':analyze,'check':checked}[a.action]()
