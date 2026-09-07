"""Wait for the live M77 content queue, then execute only prospectively eligible stages."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,subprocess,time

B=Path('/root/autodl-tmp');R=B/'sttrack_m77_window_competition_20260907'
C=R/'content_followup';E=R/'candidate_evaluation';O=E/'deferred_execution'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def module(name,p):
    s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def parent_identity(spec):
    p=Path('/proc')/str(spec['parent_pid']);v=(p/'stat').read_text().split()
    assert v[21]==spec['parent_start_ticks'] and v[2]!='Z'
    assert str((p/'cwd').resolve())==spec['parent_cwd']
    argv=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
    assert argv==spec['parent_argv']
    return dict(pid=spec['parent_pid'],start_ticks=v[21],state=v[2],argv=argv)

def inputs():
    assert sha(E/'spec.json')=='7626558230f864237b49aa4e77bd2cf7fe32c8af46d46cb3753584ac4db9483d'
    assert sha(E/'full_preparation_readiness.json')=='1877fa45c8b67ed06ee48a2bf332aa066146cecfb5784802f065b955bfe1dc17'
    entry=read(E/'spec.json');full=read(E/'full_preparation_readiness.json')
    for n,h in entry['source_sha256'].items():assert sha(B/n)==h
    for p,h in full['full_source_sha256'].items():assert sha(p)==h
    candidate=module('m77_queued_candidate',B/'m77_candidate_evaluation_20260907.py')
    assert candidate.checked_preparation()['candidate_seed']==2027
    return entry,full,candidate

def prepare():
    entry,full,_=inputs();assert not O.exists()
    assert not (C/'controller.exit').exists() and not (C/'activation.json').exists()
    assert not (E/'bundle.json').exists() and not (R/'training/category/final.pth').exists()
    assert sha(C/'spec.json')==entry['content_spec_sha256']
    wrapper=B/'m77_deferred_evaluation_20260907.sh'
    subprocess.run(['bash','-n',str(wrapper)],check=True)
    spec=dict(status='M77_deferred_evaluation_frozen_before_final_results',observed_utc=now(),source_sha256=sha(__file__),wrapper_sha256=sha(wrapper),
        entry_spec_sha256=sha(E/'spec.json'),full_readiness_sha256=sha(E/'full_preparation_readiness.json'),content_spec_sha256=sha(C/'spec.json'),
        parent_pid=496607,parent_start_ticks='4581958397',parent_cwd=str(C),
        parent_argv=[str(B/'envs/sttrack/bin/python'),'-u',str(B/'m77_content_followup_20260907.py'),'controller'],poll_seconds=240,
        seed=2027,additional_seeds=[],head_selection='Only the prospectively fixed M77 Category final; no metric-based checkpoint or seed selection.',
        steps={'entry':'m77_candidate_entry_20260907.sh','low22':'m77_candidate_low22_20260907.sh','full':'m77_full_evaluation_20260907.sh'},
        required_development_conditions=15,required_content_comparisons=8,low22_gate=entry['full_evaluation_gate'],
        execution='Wait for the original content controller identity and zero exit. Failed development/content conditions stop before binding. Failed low22 conditions stop before any full-dataset captions or tracking. No restart or threshold change.',
        preserve_original_training_and_content_controllers=True,new_optimizer_steps=0,new_caption_protocol=False,
        goal_completion_claimed=False,independent_model_review_pass=False)
    identity=parent_identity(spec);O.mkdir();write(O/'spec.json',spec)
    write(O/'preparation.json',dict(status='CPU_sources_and_live_parent_identity_verified',observed_utc=now(),spec_sha256=sha(O/'spec.json'),parent=identity,
        new_tracking_calls=0,new_optimizer_steps=0,new_caption_calls=0,final_checkpoint_opened=False,public_evaluation_started=False))
    print(json.dumps(dict(status='prepared',spec_sha256=sha(O/'spec.json'),preparation_sha256=sha(O/'preparation.json'))))

def checked():
    s=read(O/'spec.json');assert s['source_sha256']==sha(__file__)
    assert s['wrapper_sha256']==sha(B/'m77_deferred_evaluation_20260907.sh')
    assert s['entry_spec_sha256']==sha(E/'spec.json') and s['full_readiness_sha256']==sha(E/'full_preparation_readiness.json')
    assert s['content_spec_sha256']==sha(C/'spec.json') and s['seed']==2027 and s['additional_seeds']==[]
    entry,full,candidate=inputs();return s,candidate

def controller():
    s,candidate=checked();assert not (O/'controller.started').exists()
    write(O/'controller.started',dict(observed_utc=now(),pid=os.getpid(),spec_sha256=sha(O/'spec.json')))
    def stage(name):
        checked();write(O/'stage.json',dict(stage='running_'+name,observed_utc=now()))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (O/(name+'.log')).open('x') as log:
            code=subprocess.run(['bash',str(B/s['steps'][name])],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        (O/(name+'.exit')).write_text(str(code)+'\n');assert code==0,(name,code)
    def finish(status,evidence):
        report=dict(status=status,observed_utc=now(),source_sha256=sha(__file__),spec_sha256=sha(O/'spec.json'),seed=2027,additional_seeds=[],
            evidence=evidence,goal_completion_claimed=False,independent_model_review_pass=False)
        write(O/'result.json',report);write(O/'stage.json',dict(stage=status,observed_utc=now(),result_sha256=sha(O/'result.json')))
        print(json.dumps(report),flush=True)
    while not (C/'controller.exit').exists():
        identity=parent_identity(s)
        write(O/'stage.json',dict(stage='waiting_for_original_content_controller',observed_utc=now(),verified_parent=identity))
        time.sleep(s['poll_seconds'])
    assert (C/'controller.exit').read_text().strip()=='0'
    followup=module('m77_final_content_for_queue',B/'m77_content_followup_20260907.py')
    _,_,_,_,audit=followup.checked();content=read(C/'result.json');parent_stage=read(C/'stage.json')
    assert parent_stage['stage']=='complete_content_diagnostic' and parent_stage['result_sha256']==sha(C/'result.json')
    assert content['status']=='completed_M77_fixed_Category_head_content_diagnostic' and content['spec_sha256']==s['content_spec_sha256']
    assert content['source_sha256']==sha(B/'m77_content_followup_20260907.py')
    assert len(audit['gates'])==s['required_development_conditions']
    criteria=content['descriptive_criteria'];assert set(criteria)=={'empty','swapped'}
    assert all(set(v)=={'pooled','macro','low','H10'} for v in criteria.values())
    assert sum(len(v) for v in criteria.values())==s['required_content_comparisons']
    assert audit['development_gates_pass']==all(audit['gates'].values())==content['development_gate_pass']
    assert content['descriptive_criteria_pass']==all(v for c in criteria.values() for v in c.values())
    evidence=dict(training_audit_sha256=sha(C/'completed_training_audit.json'),content_result_sha256=sha(C/'result.json'),
        development_gates=audit['gates'],content_comparisons=criteria)
    if not audit['development_gates_pass'] or not content['descriptive_criteria_pass']:
        assert not (E/'bundle.json').exists()
        finish('completed_without_public_evaluation_due_to_frozen_conditions',evidence);return
    candidate.gates()
    stage('entry');assert (E/'entry_parity/controller.exit').read_text().strip()=='0'
    evidence['entry_result_sha256']=sha(E/'entry_parity/result.json')
    stage('low22');assert (E/'low22_controller.exit').read_text().strip()=='0'
    low=read(E/'low22_result.json');assert low['status']=='completed_low22_same_bundle_evaluation'
    assert low['full_three_dataset_evaluation_allowed']==all(low['gate_checks'].values())
    evidence['low22_result_sha256']=sha(E/'low22_result.json');evidence['low22_gates']=low['gate_checks']
    if not low['full_three_dataset_evaluation_allowed']:
        assert not (E/'full_evaluation').exists()
        finish('completed_low22_without_full_evaluation_due_to_frozen_conditions',evidence);return
    stage('full');assert (E/'full_evaluation/controller.exit').read_text().strip()=='0'
    final=read(E/'full_evaluation/same_bundle_verification.json')
    assert final['status']=='complete_saved_output_and_same_bundle_verification'
    evidence['same_bundle_verification_sha256']=sha(E/'full_evaluation/same_bundle_verification.json')
    evidence['all_numeric_targets_pass']=final['all_numeric_targets_pass']
    finish('completed_three_dataset_pipeline_pending_final_handoff_and_goal_audit',evidence)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','controller']);a=p.parse_args()
    {'prepare':prepare,'check':checked,'controller':controller}[a.action]()
