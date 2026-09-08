"""Fixed M78 weights: complete low22 lexical controls, diagnostic only."""
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

B=Path('/root/autodl-tmp');P=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation'
R=B/'sttrack_m79_vot_content_diagnostic_20260908';I=P/'interface'
PY='/root/miniconda3/envs/mplt/bin/python';RUNNER=R/'run_vot_failure_family_shards.py'
ARMS=['empty','swapped'];PARENT_RESULT='64f23763ee6df91d80e81754e7db3e654196eb377b624e258768fddd970d5fc3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def parent():
    path=B/'m78_vot_low22_20260908.py'
    assert sha(path)=='7fe95f000d8eb0df1268a40641124520ba1d8d78176d47f941c59847987dcfb5'
    spec=importlib.util.spec_from_file_location('m79_parent_evaluation',str(path))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.checked_execution()
    assert sha(P/'low22_result.json')==PARENT_RESULT
    a=read(P/'low22_completed_evidence/audit.json')
    assert a['status']=='completed_low22_saved_evidence_verified' and a['result_sha256']==PARENT_RESULT
    assert not a['full_three_dataset_evaluation_allowed']
    assert (P/'full_followup/controller.exit').read_text().strip()=='0'
    assert read(P/'full_followup/result.json')['status']=='complete_low22_did_not_qualify_for_full'
    assert not (P/'full_evaluation').exists()
    return m,a

def prepare():
    import torch
    torch.set_num_threads(1)
    m,a=parent();assert not R.exists()
    memory=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(memory)==2 and max(memory)<500 and shutil.disk_usage(B).free>1000000000
    sys.path.insert(0,str(I));from semantic_runtime import checked_plan,text_bank
    pp,bundle=checked_plan(P/'low22_plan.json')
    original=torch.load(pp['text_bank_path'],map_location='cpu')
    captions=read(B/'sttrack_m64_category_candidate_20260907/low22_captions/plan.json')
    cases={c['key']:c['id'].rsplit('_',1)[0] for c in captions['cases']}
    assert len(original['keys'])==len(cases)==303 and original['mask'][:,0].all()
    order=sorted(range(303),key=lambda i:original['keys'][i]);donors={}
    for pos,j in enumerate(order):
        k=next(order[(pos+offset)%303] for offset in range(1,303)
            if cases[original['keys'][order[(pos+offset)%303]]]!=cases[original['keys'][j]]
            and not torch.equal(original['tokens'][order[(pos+offset)%303],0],original['tokens'][j,0]))
        donors[j]=k
    R.mkdir();shutil.copyfile(P/'run_vot_failure_family_shards.py',RUNNER)
    mapping=[dict(key=original['keys'][j],sequence=cases[original['keys'][j]],donor_key=original['keys'][donors[j]],donor_sequence=cases[original['keys'][donors[j]]]) for j in range(303)]
    write(R/'donor_mapping.json',mapping)
    frozen=read(m.FROZEN);files=[Path(__file__),B/'m79_vot_content_diagnostic_20260908.sh',RUNNER,R/'donor_mapping.json']
    specs={}
    for gpu,arm in enumerate(ARMS):
        root=R/arm;root.mkdir()
        protocol=read(bundle['text_protocol_path'])
        protocol['diagnostic_intervention']=dict(variant=arm,parent_protocol_sha256=bundle['text_protocol_sha256'],
            rule='All valid slots receive the frozen CLIP empty embedding.' if arm=='empty' else 'Only slot0 receives a fixed different-vector category from a different sequence; no guaranteed semantic contradiction.',
            donor_mapping_sha256=sha(R/'donor_mapping.json'),diagnostic_only=True,model_promotion_allowed=False)
        protocol['selection_provenance']='M79 fixed M78 Category final after failed low22; content diagnosis only. No new seed, training, caption generation, or checkpoint selection.'
        write(root/'text_protocol.json',protocol)
        b=dict(original);b['tokens']=original['tokens'].clone();b['protocol_sha256']=sha(root/'text_protocol.json')
        if arm=='empty':b['tokens'][b['mask']]=b['empty']
        else:
            for j,k in donors.items():b['tokens'][j,0]=original['tokens'][k,0]
        assert torch.equal(b['mask'],original['mask']) and torch.equal(b['tokens'][~b['mask']],original['tokens'][~original['mask']])
        assert torch.equal(b['tokens'][:,1:],original['tokens'][:,1:])
        assert b['keys']==original['keys'] and torch.equal(b['empty'],original['empty'])
        if arm=='swapped':assert (b['tokens'][:,0]!=original['tokens'][:,0]).any(1).all()
        torch.save(b,root/'text_bank.pt')
        variant_bundle=dict(bundle,text_protocol_path=str(root/'text_protocol.json'),text_protocol_sha256=sha(root/'text_protocol.json'))
        write(root/'bundle.json',variant_bundle)
        for k,v in bundle.items():
            if k not in ['text_protocol_path','text_protocol_sha256']:assert variant_bundle[k]==v
        write(root/'plan.json',dict(bundle_path=str(root/'bundle.json'),bundle_sha256=sha(root/'bundle.json'),text_bank_path=str(root/'text_bank.pt'),text_bank_sha256=sha(root/'text_bank.pt')))
        plan,bb=checked_plan(root/'plan.json');router=text_bank(plan,bb)
        for row in captions['rows']:
            info=router.info(row['image'],row['init_bbox']);j=original['keys'].index(row['key'])
            assert torch.equal(info['text_tokens'],b['tokens'][j]) and torch.equal(info['text_mask'],original['mask'][j])
        tracker='sttrack_m79_'+arm+'_low22';wrapper=root/'m79_semantic_vot.py'
        wrapper.write_text("import sys\nsys.path.insert(0, '"+str(I)+"')\nfrom run_semantic_vot import run\nrun('"+str(root/'plan.json')+"')\n")
        ini=(f'[{tracker}]\nlabel = M79 M78 fixed head {arm} diagnostic\nprotocol = traxpython\ncommand = m79_semantic_vot\npaths = {root}\n'
            f'python = /root/autodl-tmp/envs/sttrack/bin/python\nenv_CUDA_VISIBLE_DEVICES = {gpu}\nenv_PYTHONPATH = {root}\n'
            'env_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
        run=root/'run';run.mkdir();shards=[]
        for old in frozen['shards']:
            source=Path(old['root']);dest=run/('shard-%02d'%old['index'])
            assert sha(source/'config.yaml')==old['config_sha256'] and sha(source/'sequences/list.txt')==old['list_sha256']
            shutil.copytree(source/'sequences',dest/'sequences',symlinks=True);shutil.copyfile(source/'config.yaml',dest/'config.yaml')
            (dest/'trackers.ini').write_text(ini)
            for name in frozen['sequences']:
                x=source/'sequences'/name/'anchor.value';y=dest/'sequences'/name/'anchor.value'
                if x.exists():assert sha(x)==sha(y)
            shards.append(dict(old,root=str(dest),gpu=gpu,trackers_sha256=sha(dest/'trackers.ini')))
            files.extend(dest/n for n in ['trackers.ini','config.yaml','sequences/list.txt'])
        write(run/'shard_manifest.json',dict(frozen,schema='m79_fixed_head_content_low22_v1',tracker=tracker,gpu_count=1,shards=shards))
        files.extend(root/n for n in ['text_protocol.json','text_bank.pt','bundle.json','plan.json','m79_semantic_vot.py','run/shard_manifest.json'])
        specs[arm]=dict(root=str(root),tracker=tracker,gpu=gpu,manifest_sha256=sha(run/'shard_manifest.json'),bank_sha256=sha(root/'text_bank.pt'),bundle_sha256=sha(root/'bundle.json'))
    for p in [Path(__file__)]:compile(p.read_text(),str(p),'exec')
    subprocess.run(['bash','-n',str(B/'m79_vot_content_diagnostic_20260908.sh')],check=True)
    s=dict(status='frozen_M79_diagnostic_before_tracking',observed_utc=now(),source_sha256={str(p):sha(p) for p in files},
        parent_result_sha256=PARENT_RESULT,parent_audit_sha256=sha(P/'low22_completed_evidence/audit.json'),parent_bundle_sha256=sha(P/'bundle.json'),
        base_sha256=bundle['base_checkpoint_sha256'],head_sha256=bundle['adapter_checkpoint_sha256'],seed=2027,additional_seeds=[],arms=specs,
        category_predictions_reused=True,sequences_per_arm=22,anchors_per_arm=303,frame_positions_per_arm=220483,total_new_frame_positions=440966,
        workers_per_arm=4,poll_seconds=240,new_training_steps=0,new_parameters=0,new_captions=0,new_embeddings=0,
        control_bank_keys_masks_padding_and_attribute_slots_exact=True,changed_category_vectors_swapped=303,
        donor_rule='Sort initialization keys, take the first cyclic donor from another sequence with a different slot0 vector. No GT or tracking result informs this mapping.',
        interpretation='Compare EAO/ACC/ROB, all303 paired failures, per-sequence rescues and harms under the same learned weights. Different control protocol hashes explicitly declare changed content.',
        descriptive_comparisons='Report all metric deltas and rescue/new-failure counts against Category and native. No optimized threshold or new promotion gate.',
        limitations=['Repeatedly used VOT low22 development subset, not an untouched test.','Automatic original categories are not semantic truth.','Different donor embedding is not a verified attribute contradiction.','Recursive crop/query/template states can diverge; content effect includes their consequences.'],
        model_promotion_allowed=False,full_three_dataset_evaluation_allowed=False,independent_model_review_pass=False,goal_achieved=False)
    write(R/'spec.json',s)
    write(R/'preparation_result.json',dict(status='complete_CPU_content_and_all606_initialization_routes_verified',spec_sha256=sha(R/'spec.json'),cuda_initialized=torch.cuda.is_initialized(),new_tracking_calls=0,all_model_and_runtime_bytes_unchanged=True))
    assert not torch.cuda.is_initialized()
    print(json.dumps({k:v for k,v in s.items() if k!='source_sha256'},indent=2))

def checked():
    parent();s=read(R/'spec.json')
    for p,h in s['source_sha256'].items():assert sha(p)==h,p
    assert sha(P/'low22_completed_evidence/audit.json')==s['parent_audit_sha256']
    sys.path.insert(0,str(I));from semantic_runtime import checked_plan
    for arm in ARMS:
        _,b=checked_plan(R/arm/'plan.json')
        assert b['adapter_checkpoint_sha256']==s['head_sha256'] and b['base_checkpoint_sha256']==s['base_sha256']
    return s

def analyze_arm(arm):
    s=checked();root=R/arm;run=root/'run';spec=s['arms'][arm]
    assert (R/(arm+'_tracking.exit')).read_text().strip()=='0'
    merge=read(run/'merge_result.json');assert merge['anchor_count']==303 and merge['result_file_count']==909
    assert merge['source_manifest_sha256']==spec['manifest_sha256']
    for rel,h in merge['result_sha256'].items():assert sha(run/'master'/rel)==h
    name='m79_'+arm+'_low22_analysis'
    with (root/'toolkit_analysis.log').open('w') as log:
        subprocess.run([PY,'-m','vot','analysis','--workspace',str(run/'master'),'--format','json','--name',name,spec['tracker']],env=dict(os.environ,PYTHONPATH='/home/SUTrack_RGBD_L'),stdout=log,stderr=subprocess.STDOUT,check=True)
    path=run/'master/analysis'/(name+'.json');d=read(path)['results']['baseline']['results']
    metrics=dict(EAO=float(d[0][0][0])*100,ACC=float(d[2][0][0])*100,ROB=float(d[2][0][1])*100)
    sys.path.insert(0,'/home/SUTrack_RGBD_L');from tools.finalize_vot_transaction_low22 import collect_confirmed_failure_outcomes
    outcomes,failures,per,settings=collect_confirmed_failure_outcomes(run/'master',spec['tracker'],expected_anchors=303)
    primary=read(P/'low22_result.json');assert set(outcomes)==set(primary['failure_outcomes'])
    assert sum(v['run_length'] for v in outcomes.values())==220483
    result=dict(status='complete_M79_content_arm',observed_utc=now(),variant=arm,spec_sha256=sha(R/'spec.json'),head_sha256=s['head_sha256'],
        bundle_sha256=spec['bundle_sha256'],text_bank_sha256=spec['bank_sha256'],metrics_percent=metrics,confirmed_failures=failures,
        per_sequence_failures=per,failure_outcomes=outcomes,failure_settings=settings,merge_sha256=sha(run/'merge_result.json'),analysis_sha256=sha(path),
        model_promotion_allowed=False,full_three_dataset_evaluation_allowed=False)
    write(root/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['failure_outcomes','per_sequence_failures']},indent=2))

def analyze():
    s=checked();primary=read(P/'low22_result.json');results={'category':primary}
    m,_=parent();native=read(m.M39/'m39_result.json')['arms']['default'];comparisons={}
    for arm in ARMS:
        assert (R/(arm+'.exit')).read_text().strip()=='0'
        r=read(R/arm/'result.json');assert r['spec_sha256']==sha(R/'spec.json') and r['head_sha256']==s['head_sha256']
        results[arm]=r
    for arm,r in results.items():
        comparisons[arm]={}
        for reference,base in [('category',primary),('native',native)]:
            bm=base['metrics_percent'];bm={k:bm[k.lower()] for k in ['EAO','ACC','ROB']} if reference=='native' else bm
            comparisons[arm][reference]=dict(delta_metrics_percent={k:r['metrics_percent'][k]-bm[k] for k in bm},
                delta_failures=r['confirmed_failures']-base['confirmed_failures'],
                rescued_anchors=[k for k,v in r['failure_outcomes'].items() if not v['failed'] and base['failure_outcomes'][k]['failed']],
                newly_failed_anchors=[k for k,v in r['failure_outcomes'].items() if v['failed'] and not base['failure_outcomes'][k]['failed']])
    out=dict(status='complete_M79_fixed_M78_head_VOT_content_diagnostic',observed_utc=now(),spec_sha256=sha(R/'spec.json'),
        head_sha256=s['head_sha256'],base_sha256=s['base_sha256'],seed=2027,additional_seeds=[],
        result_sha256={arm:sha(P/'low22_result.json') if arm=='category' else sha(R/arm/'result.json') for arm in results},
        aggregates={arm:dict(metrics_percent=r['metrics_percent'],confirmed_failures=r['confirmed_failures']) for arm,r in results.items()},
        per_sequence={arm:r['per_sequence_failures'] for arm,r in results.items()},comparisons=comparisons,
        all606_new_anchor_outputs_verified=True,category303_sealed_outputs_reused=True,new_training_steps=0,new_captions=0,
        original_M78_failed_gate_unchanged=True,model_promotion_allowed=False,full_three_dataset_evaluation_allowed=False,goal_achieved=False)
    write(R/'result.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ['per_sequence','comparisons']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','empty','swapped','analyze']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':checked();print('M79_FIXED_WEIGHTS_AND_CONTENT_INPUTS_VERIFIED')
    elif a.action in ARMS:analyze_arm(a.action)
    else:analyze()
