"""Verify sealed M74 outputs and independently recompute continuous-box scalars."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

B=Path('/root/autodl-tmp');R=B/'sttrack_m74_m73_content_diagnostic_20260907'
P=B/'sttrack_m73_paired_lexical_replication_20260907/seed2027'
SOURCE_SHA='819b5a629bd59715a0ce7a106e3dbfcc7780c4d0078fed9731cfb6a1a8d891eb'
SPEC_SHA='636baf52a2418cf01879d3a440afa5a6b2f039c454a2771296d000e357a6427a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def intervals(mask):
    padded=np.r_[False,mask,False].astype(np.int8)
    starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
    return [(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=10]

def main():
    target=R/'completed_evidence_audit.json';assert not target.exists()
    result=read(R/'result.json');old=sha(R/'result.json')
    assert result['status']=='completed_diagnostic_only_M73_Category_fixed_head_content'
    assert sha(R/'spec.json')==SPEC_SHA==result['spec_sha256']
    source=B/'m74_m73_content_diagnostic_20260907.py';assert sha(source)==SOURCE_SHA==result['source_sha256']
    for name in ['eligibility','device_check','prefix','empty','swapped','analysis','controller']:
        assert (R/(name+'.exit')).read_text().strip()=='0'
    module_spec=importlib.util.spec_from_file_location('m74_sealed_inputs',str(source))
    app=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(app)
    spec,training,audit=app.checked()
    assert spec['seed']==2027 and spec['additional_training_seeds']==[]
    assert not result['low22_candidate_preparation_allowed'] and not result['full_three_dataset_evaluation_allowed']
    families={};per={};writes={};traces={}
    for name in ['category','empty','swapped']:
        directory=P/'recursive/category' if name=='category' else R/name
        receipt_path=P/'category_recursive_receipt.json' if name=='category' else directory/'receipt.json'
        assert sha(receipt_path)==result['receipts'][name]
        receipt=read(receipt_path);assert receipt['head_sha256']==app.HEAD_SHA and receipt['status']=='complete'
        assert receipt['total_frames']==33130 and len(receipt['sequences'])==22
        if name!='category':
            assert receipt['spec_sha256']==SPEC_SHA and receipt['bank_sha256']==spec['banks'][name]['sha256']
            assert receipt['variant']==name and not receipt['subsequent_GT_opened'] and receipt['optimizer_steps']==0
        records={item['sequence']:item for item in receipt['sequences']}
        traces[name]={};writes[name]=0
        for case in spec['cases']:
            path=directory/(case['sequence']+'.json');assert sha(path)==records[case['sequence']]['sha256']
            value=read(path);rows=value['rows']
            assert value['sequence']==case['sequence'] and value['arm']==name
            assert len(rows)==case['frames'] and [row['frame'] for row in rows]==list(range(case['frames']))
            assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None)
            boxes=np.asarray([row['bbox'] for row in rows],dtype=np.float64)
            scores=np.asarray([row['score'] for row in rows[1:]],dtype=np.float64)
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all()
            traces[name][case['sequence']]=boxes
            writes[name]+=sum(row['frame']%50==0 and row['score']>.75 for row in rows[1:])
        families[name]=dict(receipt_sha256=sha(receipt_path),head_sha256=app.HEAD_SHA,track_calls=33108,sequences=22)
    # All family bindings, scores and boxes are validated before reading later GT.
    series={name:{} for name in traces};harms=[]
    per={name:{} for name in traces}
    for case in spec['cases']:
        seq=case['sequence'];gt_path=Path(training['dataset_root'])/seq/'groundtruth.txt'
        assert sha(gt_path)==case['gt_sha256']
        gt=np.loadtxt(gt_path,delimiter=',',dtype=np.float64).reshape(-1,4)
        assert len(gt)==case['frames']
        valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1);valid[0]=False
        for name in traces:
            boxes=traces[name][seq];iou=np.full(len(gt),np.nan,dtype=np.float64)
            g=gt[valid];b=boxes[valid]
            overlap=np.maximum(0,np.minimum(g[:,:2]+g[:,2:],b[:,:2]+b[:,2:])-np.maximum(g[:,:2],b[:,:2]))
            inter=overlap[:,0]*overlap[:,1]
            iou[valid]=inter/(g[:,2]*g[:,3]+b[:,2]*b[:,3]-inter)
            low=valid&(iou<=.1);episodes=intervals(low)
            row=dict(valid_frames=int(valid.sum()),iou_sum=float(iou[valid].sum()),mean_iou=float(iou[valid].mean()),low_iou_frames=int(low.sum()),failure_episodes=len(episodes))
            for key,v in row.items():assert abs(v-result['per_sequence'][name][seq][key])<1e-8,(name,seq,key)
            per[name][seq]=row;series[name][seq]=iou
        for a,b in intervals(valid&(series['category'][seq]<=.1)):
            for ref in ['empty','swapped']:
                if np.all(series[ref][seq][a:b]>=.5):harms.append(dict(sequence=seq,reference=ref,start=a,end_exclusive=b,frames=b-a))
    aggregates={}
    for name,values in per.items():
        row={key:sum(v[key] for v in values.values()) for key in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        row.update(mean_iou=row['iou_sum']/row['valid_frames'],macro_sequence_mean_iou=float(np.mean([v['mean_iou'] for v in values.values()])))
        for key,v in row.items():assert abs(v-result['aggregates'][name][key])<1e-8,(name,key)
        assert writes[name]==result['reconstructed_template_writes'][name]
        aggregates[name]=row
    flags={};c=aggregates['category']
    for name in ['empty','swapped']:
        o=aggregates[name]
        flags[name]=dict(pooled_margin=c['mean_iou']>=o['mean_iou']+.001,macro=c['macro_sequence_mean_iou']>=o['macro_sequence_mean_iou'],low_frames=c['low_iou_frames']<=o['low_iou_frames'],H10=c['failure_episodes']<=o['failure_episodes'])
    assert flags==result['descriptive_lexical_criteria']
    assert all(v for row in flags.values() for v in row.values())==result['descriptive_criteria_pass']
    assert sha(R/'result.json')==old
    report=dict(status='M74_completed_artifacts_and_independent_scalar_recomputation',observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),source_sha256=SOURCE_SHA,spec_sha256=SPEC_SHA,result_sha256=old,families=families,recomputed_aggregates=aggregates,recomputed_lexical_criteria=flags,reconstructed_template_writes=writes,sustained_category_H10_with_reference_correct_every_frame=harms,additional_seeds=[],new_optimizer_steps=0,new_tracking_calls=0,sealed_outputs_modified=False,original_M73_primary_pass=False,public_evaluation_allowed=False,independent_model_review_pass=False,scope='Continuous-box metric independently implemented here; source, receipt and input bindings verified. Same reused Train development22, no statistical seed replication or independent learned-model review.')
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
