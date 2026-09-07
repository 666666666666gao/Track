"""Deferred M78 final-checkpoint audit and same-head lexical controls; no training."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys,time
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';O=R/'content_followup'
PY=B/'envs/sttrack/bin/python';AUDITOR=B/'audit_m78_completed_20260908.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def plans():
    s=read(O/'spec.json');t=read(R/'training_spec.json');rs=read(R/'recursive_spec.json')
    assert s['source_sha256']==sha(__file__) and s['auditor_sha256']==sha(AUDITOR)
    assert s['training_spec_sha256']==sha(R/'training_spec.json') and s['recursive_spec_sha256']==sha(R/'recursive_spec.json')
    assert t['seed']==s['seed']==2027 and not s['additional_seeds']
    for n,k in [('train_causal.py','training_script_sha256'),('causal_training.py','causal_script_sha256'),('window_competition.py','window_loss_sha256'),('run_pair.sh','run_queue_sha256')]:assert sha(R/n)==t[k]
    assert sha(R/'run_recursive.py')==rs['runner_sha256'] and sha(R/'recursive_metric.py')==rs['metric_sha256']
    assert sha(R/'integration.json')==t['integration_sha256']
    for n,h in read(R/'integration.json')['source_sha256'].items():assert sha(R/'code'/n)==h
    for b in s['banks'].values():assert sha(b['path'])==b['sha256']
    assert not s['public_evaluation_allowed']
    return s,t,rs

def prepare():
    import torch
    import numpy as np
    assert not O.exists()
    assert sha(R/'training_spec.json')=='57cdd314efd5359fa2e16be4f364c5568f1ca498b41df718517173610fe4d865'
    assert sha(R/'recursive_spec.json')=='fcef09d58ec79f690d6088e7f8fbc16a4ac42ef3c2a5b3c1c1cef389ed6a6f9f'
    t=read(R/'training_spec.json');rs=read(R/'recursive_spec.json')
    previous=read(B/'sttrack_m77_window_competition_20260907/content_followup/spec.json')
    banks=previous['banks'];values={n:torch.load(v['path'],map_location='cpu') for n,v in banks.items()}
    c=values['category'];assert c['tokens'].shape==(22,5,768)
    assert banks['category']==t['banks']['development']['category'] and banks['empty']==t['banks']['development']['empty']
    for n,v in values.items():
        assert sha(banks[n]['path'])==banks[n]['sha256']
        assert v['sequences']==c['sequences'] and torch.equal(v['mask'],c['mask']) and torch.equal(v['empty'],c['empty'])
        assert torch.equal(v['tokens'][~c['mask']],c['tokens'][~c['mask']])
    assert torch.equal(values['empty']['tokens'][c['mask']],c['empty'].expand_as(c['tokens'][c['mask']]))
    assert torch.equal(values['swapped']['tokens'][:,1:],c['tokens'][:,1:])
    assert bool((values['swapped']['tokens'][:,0]!=c['tokens'][:,0]).any(1).all())
    launch=read(R/'launch.json');pid=launch['controller_pid'];parent=Path('/proc')/str(pid)
    assert parent.exists() and (parent/'cwd').resolve()==R
    ticks=(parent/'stat').read_text().split()[21]
    argv=[x.decode() for x in (parent/'cmdline').read_bytes().split(b'\0') if x]
    assert argv==['bash',str(R/'run_pair.sh')] and not (R/'controller.exit').exists()
    helper=module('m78_preparation_audit',AUDITOR);_,scalar=helper.helpers()
    old=B/'sttrack_m77_window_competition_20260907';prior=read(old/'recursive_result.json')
    g=helper.gates(prior['per_sequence'],prior['aggregates'],prior['parent_M73_aggregates'],t)
    assert len(g)==10 and g=={k:v for k,v in prior['gates'].items() if k in g}
    # Real sealed rows exercise the interface that failed in M77, before any M78 output exists.
    seq=rs['cases'][0]['sequence'];rows=read(old/'recursive/category'/ (seq+'.json'))['rows']
    gt=np.loadtxt(Path(t['dataset_root'])/seq/'groundtruth.txt',delimiter=',').reshape(-1,4)
    independent=scalar.scalar_metric();iou,metrics,spans=independent(rows,gt)
    sys.path.insert(0,str(R));from recursive_metric import statistics
    reference=statistics(np.asarray([r['bbox'] for r in rows]),gt)
    assert all(abs(metrics[k]-reference[k])<1e-8 for k in reference)
    O.mkdir()
    spec=dict(status='frozen_M78_followup_before_final_outputs',observed_utc=now(),source_sha256=sha(__file__),auditor_sha256=sha(AUDITOR),
        training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),seed=2027,additional_seeds=[],
        parent_pid=pid,parent_start_ticks=ticks,parent_argv=argv,poll_seconds=240,banks=banks,cases=rs['cases'],prefix_sequences=previous['prefix_sequences'],prefix_frames=102,
        head_selection='Fixed M78 Category final; no metric checkpoint selection, regardless of primary gate.',
        descriptive_lexical_criteria=previous['descriptive_lexical_criteria'],new_full_track_calls=66216,prefix_track_calls=303,new_optimizer_steps=0,
        public_evaluation_allowed=False,independent_model_review_pass=False,estimated_content_seconds=2100)
    write(O/'spec.json',spec)
    write(O/'preparation.json',dict(source_sha256=sha(__file__),spec_sha256=sha(O/'spec.json'),auditor_sha256=sha(AUDITOR),historical_gate_contract=g,
        rows_metric_interface_verified=True,reference_sequence=seq,reference_metrics=reference,prepared_before_current_final_outputs=True,current_GPU_calls=0))
    print(json.dumps(dict(spec_sha256=sha(O/'spec.json'),preparation_sha256=sha(O/'preparation.json'))))

def checked():
    s,t,rs=plans();a=read(O/'activation.json');audit=read(O/'completed_training_audit.json')
    assert a['spec_sha256']==sha(O/'spec.json') and a['audit_sha256']==sha(O/'completed_training_audit.json')
    assert audit['auditor_sha256']==s['auditor_sha256'] and audit['content_diagnostic_allowed']
    assert audit['result_sha256']==sha(R/'recursive_result.json')
    assert a['head_sha256']==sha(R/'training/category/final.pth')==audit['training']['category']['head_sha256']
    assert a['category_receipt_sha256']==sha(R/'category_recursive_receipt.json')
    return s,t,rs,a,audit

def track(name,prefix=False):
    s,t,rs,a,audit=checked()
    if not prefix:
        assert (O/'prefix.exit').read_text().strip()=='0'
        p=read(O/'prefix/receipt.json');assert p['exact_category_prefix_parity'] and p['head_sha256']==a['head_sha256'] and p['spec_sha256']==sha(O/'spec.json')
    import torch
    sys.path.insert(0,str(R/'code'))
    from lib.config.sttrack.config import cfg,update_config_from_file
    from lib.test.tracker.sttrack_semantic import STTrackSemantic
    from lib.train.dataset.depth_utils import get_rgbd_frame
    torch.set_num_threads(1);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    update_config_from_file(str(R/'code/experiments/sttrack/deep_rgbd_256_lachtt_v1.yaml'))
    params=SimpleNamespace(cfg=cfg,checkpoint=t['native_checkpoint'],base_checkpoint_sha256=t['native_checkpoint_sha256'],template_factor=2.,template_size=128,search_factor=4.,search_size=256,save_all_boxes=False,debug=0)
    assert sha(t['native_checkpoint'])==t['native_checkpoint_sha256']
    tracker=STTrackSemantic(params,str(R/'training/category/final.pth'));assert tracker.use_text and tracker.network.semantic_adapter.null_support and not tracker.network.training
    bank=torch.load(s['banks'][name]['path'],map_location='cpu')
    out=O/('prefix' if prefix else name);out.mkdir()
    refs={v['sequence']:v for v in read(R/'category_recursive_receipt.json')['sequences']};receipts=[];started=time.time()
    for case in s['cases']:
        seq=case['sequence']
        if prefix and seq not in s['prefix_sequences']:continue
        folder=Path(t['dataset_root'])/seq
        def frame(i):return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        j=bank['sequences'].index(seq)
        tracker.initialize(frame(0),dict(init_bbox=case['init_bbox'],text_tokens=bank['tokens'][j],text_mask=bank['mask'][j],empty_text=bank['empty']))
        rows=[dict(frame=0,bbox=list(tracker.state),score=None)];count=102 if prefix else case['frames']
        for i in range(1,count):
            pred=tracker.track(frame(i));rows.append(dict(frame=i,bbox=list(pred['target_bbox']),score=float(pred['best_score'])))
        if prefix:
            p=R/'recursive/category'/(seq+'.json');assert sha(p)==refs[seq]['sha256'] and rows==read(p)['rows'][:count]
        p=out/(seq+'.json');write(p,dict(sequence=seq,arm=name,rows=rows))
        row=dict(sequence=seq,frames=count,sha256=sha(p),elapsed_seconds=time.time()-started);receipts.append(row);print(json.dumps(row),flush=True)
    checked()
    receipt=dict(status='complete',variant=name,prefix_only=prefix,head_sha256=a['head_sha256'],spec_sha256=sha(O/'spec.json'),activation_sha256=sha(O/'activation.json'),
        bank_sha256=s['banks'][name]['sha256'],sequences=receipts,total_frames=sum(v['frames'] for v in receipts),exact_category_prefix_parity=prefix,subsequent_GT_opened=False,new_optimizer_steps=0,elapsed_seconds=time.time()-started)
    write(out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='sequences'}),flush=True)

def analyze():
    import numpy as np
    s,t,rs,a,audit=checked();data={};receipts={}
    for name in ['category','empty','swapped']:
        if name=='category':directory=R/'recursive/category';rp=R/'category_recursive_receipt.json'
        else:
            assert (O/(name+'.exit')).read_text().strip()=='0';directory=O/name;rp=directory/'receipt.json'
        receipt=read(rp);assert receipt['status']=='complete' and receipt['head_sha256']==a['head_sha256'] and receipt['total_frames']==33130
        assert [v['sequence'] for v in receipt['sequences']]==[v['sequence'] for v in s['cases']]
        if name!='category':
            assert receipt['variant']==name and not receipt['prefix_only'] and not receipt['subsequent_GT_opened']
            assert receipt['spec_sha256']==sha(O/'spec.json') and receipt['bank_sha256']==s['banks'][name]['sha256']
            assert receipt['activation_sha256']==sha(O/'activation.json')
        data[name]={}
        for case,v in zip(s['cases'],receipt['sequences']):
            p=directory/(case['sequence']+'.json');assert sha(p)==v['sha256'];x=read(p);rows=x['rows']
            assert x['arm']==name and x['sequence']==case['sequence']
            assert len(rows)==case['frames'] and [r['frame'] for r in rows]==list(range(case['frames']))
            assert rows[0]==dict(frame=0,bbox=case['init_bbox'],score=None)
            boxes=np.asarray([r['bbox'] for r in rows]);scores=np.asarray([r['score'] for r in rows[1:]])
            assert np.isfinite(boxes).all() and (boxes[:,2:]>0).all() and np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
            data[name][case['sequence']]=rows
        receipts[name]=sha(rp)
    # Open later-frame metric GT only after every prediction family is fully sealed.
    sys.path.insert(0,str(R));from recursive_metric import statistics
    helper=module('m78_audit_for_content',AUDITOR);_,independent=helper.helpers();scalar=independent.scalar_metric()
    per={n:{} for n in data};writes={n:0 for n in data};overlaps={n:{} for n in data};segments={}
    for case in s['cases']:
        seq=case['sequence'];p=Path(t['dataset_root'])/seq/'groundtruth.txt';assert sha(p)==case['gt_sha256'];gt=np.loadtxt(p,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        for n in data:
            rows=data[n][seq];boxes=np.asarray([r['bbox'] for r in rows]);v=statistics(boxes,gt);iou,recomputed,spans=scalar(rows,gt)
            for k in v:assert abs(v[k]-recomputed[k])<1e-8,(n,seq,k)
            per[n][seq]=v;overlaps[n][seq]=iou
            if n=='category':segments[seq]=spans
            writes[n]+=sum(r['frame']%50==0 and r['score']>.75 for r in rows[1:])
    aggregates={}
    for n,values in per.items():
        v={k:sum(r[k] for r in values.values()) for k in ['valid_frames','iou_sum','low_iou_frames','failure_episodes']}
        v.update(mean_iou=v['iou_sum']/v['valid_frames'],macro_sequence_mean_iou=float(np.mean([r['mean_iou'] for r in values.values()])))
        assert v['valid_frames']==28897;aggregates[n]=v
    for k,v in aggregates['category'].items():assert abs(v-audit['aggregates']['category'][k])<1e-8
    c=aggregates['category'];criteria={};harms=[]
    for n in ['empty','swapped']:
        v=aggregates[n];criteria[n]=dict(pooled=c['mean_iou']>=v['mean_iou']+.001,macro=c['macro_sequence_mean_iou']>=v['macro_sequence_mean_iou'],low=c['low_iou_frames']<=v['low_iou_frames'],H10=c['failure_episodes']<=v['failure_episodes'])
        for seq,spans in segments.items():
            for start,end in spans:
                if bool((overlaps[n][seq][start:end]>=.5).all()):harms.append(dict(reference=n,sequence=seq,start=int(start),end_exclusive=int(end),frames=int(end-start)))
    result=dict(status='completed_M78_fixed_Category_head_content_diagnostic',observed_utc=now(),source_sha256=sha(__file__),spec_sha256=sha(O/'spec.json'),activation_sha256=sha(O/'activation.json'),head_sha256=a['head_sha256'],
        receipts=receipts,aggregates=aggregates,per_sequence=per,reconstructed_template_writes=writes,descriptive_criteria=criteria,descriptive_criteria_pass=all(v for x in criteria.values() for v in x.values()),
        development_gate_pass=audit['development_gates_pass'],strict_category_H10_reference_correct_every_frame=harms,independent_scalar_recomputation=True,
        seed=2027,new_full_tracking_calls=66216,new_optimizer_steps=0,additional_seeds=[],public_evaluation_allowed=False,independent_model_review_pass=False,
        scope='One fixed Category head, reused Train development22, original/empty/swapped word content. Automatic category truth and same-class identity are not established. Failed development conditions remain failed regardless of content diagnostics.')
    write(O/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['per_sequence','strict_category_H10_reference_correct_every_frame']},indent=2))

def controller():
    s,t,rs=plans();assert not (O/'controller.started').exists();write(O/'controller.started',dict(observed_utc=now(),pid=os.getpid()))
    def stage(name,command,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        write(O/'stage.json',dict(stage=name,observed_utc=now()))
        with (O/(name+'.log')).open('x') as log:status=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (O/(name+'.exit')).write_text(str(status)+'\n');assert status==0,(name,status)
    while not (R/'controller.exit').exists():
        parent=Path('/proc')/str(s['parent_pid']);assert parent.exists() and (parent/'cwd').resolve()==R
        st=(parent/'stat').read_text().split();assert st[21]==s['parent_start_ticks'] and st[2]!='Z'
        assert [x.decode() for x in (parent/'cmdline').read_bytes().split(b'\0') if x]==s['parent_argv']
        write(O/'stage.json',dict(stage='waiting_for_original_training_and_recursion',observed_utc=now(),verified_parent_pid=s['parent_pid']))
        time.sleep(s['poll_seconds'])
    assert (R/'controller.exit').read_text().strip()=='0'
    stage('training_audit',[str(PY),str(AUDITOR)],'')
    audit=read(O/'completed_training_audit.json');assert audit['content_diagnostic_allowed']
    write(O/'activation.json',dict(observed_utc=now(),spec_sha256=sha(O/'spec.json'),audit_sha256=sha(O/'completed_training_audit.json'),head_sha256=audit['training']['category']['head_sha256'],category_receipt_sha256=sha(R/'category_recursive_receipt.json'),development_gates_pass=audit['development_gates_pass'],selection='fixed_category_final_without_metric_selection'))
    used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(used)==2 and max(used)<500 and shutil.disk_usage(R).free>700_000_000
    write(O/'device_preflight.json',dict(observed_utc=now(),GPU_MiB=used,free_bytes=shutil.disk_usage(R).free))
    stage('prefix',[str(PY),str(Path(__file__)), 'prefix'],'0')
    write(O/'stage.json',dict(stage='running_fixed_head_empty_and_swapped',observed_utc=now()))
    jobs=[]
    for name,gpu in [('empty','0'),('swapped','1')]:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        log=(O/(name+'.log')).open('x');p=subprocess.Popen([str(PY),'-u',str(Path(__file__)),name],env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((name,p,log))
    status=[]
    for name,p,log in jobs:
        code=p.wait();log.close();(O/(name+'.exit')).write_text(str(code)+'\n');status.append(code)
    assert status==[0,0],status
    stage('content_analysis',[str(PY),str(Path(__file__)),'analyze'],'')
    write(O/'stage.json',dict(stage='complete_content_diagnostic',observed_utc=now(),result_sha256=sha(O/'result.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','controller','prefix','empty','swapped','analyze']);a=p.parse_args()
    if a.action=='prefix':track('category',True)
    elif a.action in ['empty','swapped']:track(a.action)
    else:{'prepare':prepare,'check':plans,'controller':controller,'analyze':analyze}[a.action]()
