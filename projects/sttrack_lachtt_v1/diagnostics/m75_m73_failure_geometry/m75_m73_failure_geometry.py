"""Post-result CPU geometry census of all M73 Category H10 onsets."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
from PIL import Image

B=Path('/root/autodl-tmp');P=B/'sttrack_m73_paired_lexical_replication_20260907/seed2027'
M74=B/'sttrack_m74_m73_content_diagnostic_20260907';R=B/'sttrack_m75_m73_failure_geometry_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def intervals(mask):
 d=np.diff(np.r_[False,mask,False].astype(np.int8));starts=np.flatnonzero(d==1);ends=np.flatnonzero(d==-1)
 return [(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=10]
def ious(boxes,gt):
 valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1)
 a=boxes[valid];b=gt[valid];v=np.full(len(gt),np.nan)
 extent=np.maximum(0,np.minimum(a[:,:2]+a[:,2:],b[:,:2]+b[:,2:])-np.maximum(a[:,:2],b[:,:2]))
 inter=extent[:,0]*extent[:,1];v[valid]=inter/(a[:,2]*a[:,3]+b[:,2]*b[:,3]-inter)
 return v,valid
def crop(prior):
 side=math.ceil(math.sqrt(prior[2]*prior[3])*4.)
 return [round(prior[0]+.5*prior[2]-.5*side),round(prior[1]+.5*prior[3]-.5*side),side,side]
def overlap_fraction(g,c):
 area=max(0,min(g[0]+g[2],c[0]+c[2])-max(g[0],c[0]))*max(0,min(g[1]+g[3],c[1]+c[3])-max(g[1],c[1]))
 return float(area/(g[2]*g[3]))
def geometry(prior,box,score,g,image_size):
 c=crop(prior);cx=g[0]+g[2]/2;cy=g[1]+g[3]/2
 return dict(previous_bbox=prior.tolist(),bbox=box.tolist(),score=score,search_rectangle=c,
  GT_center_inside_search=bool(c[0]<=cx<c[0]+c[2] and c[1]<=cy<c[1]+c[3]),
  GT_box_inside_search=bool(g[0]>=c[0] and g[1]>=c[1] and g[0]+g[2]<=c[0]+c[2] and g[1]+g[3]<=c[1]+c[3]),
  GT_area_fraction_in_search=overlap_fraction(g,c),GT_center_inside_image=bool(0<=cx<image_size[0] and 0<=cy<image_size[1]),
  predicted_center_displacement=float(np.linalg.norm(box[:2]+box[2:]/2-prior[:2]-prior[2:]/2)))

def main():
 assert not R.exists()
 assert sha(M74/'result.json')=='cad42b93712f017af9b79c615f7e8264de614c0b982d0e32be9d398927089820'
 assert sha(M74/'completed_evidence_audit.json')=='ddba08da45fddf83b2ddd56195a01a72c8af6bc7971a83f27f26569f3e5b9d84'
 assert (M74/'completion_audit_runner/controller.exit').read_text().strip()=='0'
 x=read(M74/'result.json');audit=read(M74/'completed_evidence_audit.json');rs=read(P/'recursive_spec.json');training=read(P/'training_spec.json')
 code=P/'code/lib/train/data/processing_utils.py';integration=read(P/'integration.json')
 assert sha(code)==integration['source_sha256']['lib/train/data/processing_utils.py']
 source=code.read_text();assert 'crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)' in source
 assert 'x1 = round(x + 0.5 * w - crop_sz * 0.5)' in source and 'y1 = round(y + 0.5 * h - crop_sz * 0.5)' in source
 R.mkdir()
 spec=dict(status='frozen_post_result_Train_geometry_census',observed_utc=now(),source_sha256=sha(__file__),M74_result_sha256=sha(M74/'result.json'),M74_audit_sha256=sha(M74/'completed_evidence_audit.json'),sampling_source_sha256=sha(code),seed=2027,new_training_seeds=[],selection='Every Category H10 onset in all 22 reused Train development sequences; mark the already sealed M74 strict content-harm intervals. No sequence or failure subset is dropped.',
  comparisons='At the same physical Category-failure time, compare each content condition own prior bbox/search state. These are different complete trajectories, not same-state head probes.',
  crop='Exact native ceil(sqrt(w*h)*4), round(center-halfside) geometry from prior public xywh.',
  GT_policy='Posthoc diagnostics only, not a deployable trigger; invalid GT never automatically means absent target.',
  limitations=['Search inclusion does not prove a correct candidate exists.','H10 onset is the first frame of a sustained low-overlap interval, not necessarily the first earlier tracking error.','Reused development events are not an unbiased held-out test.'],
  new_tracking_calls=0,new_optimizer_steps=0,public_evaluation_allowed=False)
 write(R/'spec.json',spec)
 strict={(v['sequence'],v['start'],v['end_exclusive']) for v in audit['sustained_category_H10_with_reference_correct_every_frame']}
 data={};receipts={}
 for arm in ['category','empty','swapped']:
  folder=P/'recursive/category' if arm=='category' else M74/arm
  rp=P/'category_recursive_receipt.json' if arm=='category' else folder/'receipt.json'
  assert sha(rp)==x['receipts'][arm];receipt=read(rp);receipts[arm]=sha(rp);data[arm]={}
  for item in receipt['sequences']:
   path=folder/(item['sequence']+'.json');assert sha(path)==item['sha256']
   data[arm][item['sequence']]=read(path)['rows']
 events=[]
 for case in rs['cases']:
  seq=case['sequence'];folder=Path(training['dataset_root'])/seq;gp=folder/'groundtruth.txt';assert sha(gp)==case['gt_sha256']
  gt=np.loadtxt(gp,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
  boxes={a:np.asarray([row['bbox'] for row in data[a][seq]],dtype=np.float64) for a in data}
  series={a:ious(b,gt)[0] for a,b in boxes.items()};_,valid=ious(boxes['category'],gt)
  metric_valid=valid.copy();metric_valid[0]=False
  segments=intervals(metric_valid&(series['category']<=.1))
  assert len(segments)==x['per_sequence']['category'][seq]['failure_episodes']
  for start,end in segments:
   with Image.open(folder/'color'/('%08d.jpg'%(start+1))) as image:image_size=image.size
   arms={a:geometry(boxes[a][start-1],boxes[a][start],float(data[a][seq][start]['score']),gt[start],image_size) for a in data}
   for a in arms:
    arms[a]['iou']=float(series[a][start]);arms[a]['prior_iou']=float(series[a][start-1]) if valid[start-1] else None
   event=dict(sequence=seq,start=start,end_exclusive=end,frames=end-start,GT=gt[start].tolist(),image_size=list(image_size),previous_GT_valid=bool(valid[start-1]),strict_content_harm=(seq,start,end) in strict,arms=arms)
   if valid[start-1]:
    scale=math.sqrt(gt[start-1,2]*gt[start-1,3]);motion=float(np.linalg.norm(gt[start,:2]+gt[start,2:]/2-gt[start-1,:2]-gt[start-1,2:]/2))
    event.update(GT_center_displacement=motion,GT_center_displacement_over_prior_scale=motion/scale,GT_linear_scale_factor=math.sqrt(gt[start,2]*gt[start,3])/scale)
   events.append(event)
 assert len(events)==71 and sum(v['strict_content_harm'] for v in events)==len(strict)
 def counts(rows):
  return dict(events=len(rows),prior_GT_invalid=sum(not v['previous_GT_valid'] for v in rows),
   by_content={a:dict(center_inside=sum(v['arms'][a]['GT_center_inside_search'] for v in rows),box_inside=sum(v['arms'][a]['GT_box_inside_search'] for v in rows),correct_IoU_at_least_half=sum(v['arms'][a]['iou']>=.5 for v in rows)) for a in data})
 result=dict(status='completed_M75_CPU_exact_search_geometry_census',observed_utc=now(),source_sha256=sha(__file__),spec_sha256=sha(R/'spec.json'),receipts=receipts,all_onsets=counts(events),strict_content_harm_onsets=counts([v for v in events if v['strict_content_harm']]),events=events,new_tracking_calls=0,new_optimizer_steps=0,new_training_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False)
 write(R/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
 for v in events:
  if v['strict_content_harm']:print(json.dumps(v))

if __name__=='__main__':main()
