"""Wait for the registered live M67 controller, then run frozen authorized diagnostics once."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_post_m67_diagnostic_queue_20260907'
M67=BASE/'sttrack_m67_supervised_semantic_support_20260907'
M68=BASE/'sttrack_m68_reported_confidence_20260907'
M69=BASE/'sttrack_m69_m65_content_diagnostic_20260907'
M70=BASE/'sttrack_m70_recovery_window_inventory_20260907/candidate_capacity'
SOURCE_HASHES={
    BASE/'audit_m67_completed_20260907.py':'1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf',
    BASE/'m67_content_counterfactuals_20260907.py':'40b765b22c398bc18ed4ac3f5ef33edc7b9611cf1c8d3172062f518d435f8285',
    BASE/'m68_reported_confidence_20260907.py':'7faf987f68c26739cd3666428e1c73e42704a4d208ffe29572863ca7b808f4d9',
    BASE/'m69_m65_content_diagnostic_20260907.py':'c3feaa2f1507dc5174331d0a1ebd0fbf96032fc5434f9f82675bd83cc7292ce0',
    BASE/'m70_candidate_capacity_20260907.py':'e0f02783e2410f745a40465799407a4cef218993e1d8eee6fc745d0029f53977',
    M67/'run_m67.sh':'546d4ac1dedf38a6381e8fcc79038dcd73dfc6d7a424126852ffdc1bce6c95a3',
    M67/'training_spec.json':'2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e',
    M67/'recursive_spec.json':'d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5',
    M67/'content_counterfactuals/spec.json':'c107a0b08a6825803f6a57f0c7ab5f469e06288c3df203dfbb5ce880a4f6c400',
    M68/'spec.json':'e2106dd1604d627c02d0589b91224c4901bfbca2906ffbbf241a3f380b7cfdf5',
    M69/'spec.json':'1edc827b9f5aaf9dffee70b4f85e456b6f66c9a2e0dfab5c189bb4132ceeab8e',
    M70/'spec.json':'796a2756db4c2be0b5282d567294a76d494ad031cc4e186ba96deb3228c00fa5'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()

def identity(pid):
    p=Path('/proc')/str(pid);assert p.exists(),('process_missing',pid)
    stat=(p/'stat').read_text().rsplit(')',1)[1].split();assert stat[0]!='Z',('process_zombie',pid)
    return dict(pid=pid,start_ticks=stat[19],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],cwd=str((p/'cwd').resolve()))

def plans():
    for p,h in SOURCE_HASHES.items():assert sha(p)==h,str(p)
    return [
        dict(name='M67_content',queue=str(M67/'content_counterfactuals/run_controls.sh'),result=str(M67/'content_counterfactuals/result.json'),
             when='M67_original_development_gate_pass',expected_status='completed_fixed_head_category_content_controls'),
        dict(name='M69_M65_content',queue=str(M69/'run_controls.sh'),result=str(M69/'result.json'),when='always_diagnostic_only',
             expected_status='completed_diagnostic_only_M65_fixed_head_content_controls'),
        dict(name='M68_confidence',queue=str(M68/'run_diagnostic.sh'),result=str(M68/'result.json'),when='always_diagnostic_only',
             expected_status='completed_Train_confidence_readout_diagnostic'),
        dict(name='M70_capacity',queue=str(M70/'run_capacity.sh'),result=str(M70/'result.json'),when='always_diagnostic_only',
             expected_status='completed_fixed_state_cross_region_candidate_capacity')]

def prepare():
    stages=plans();parent=identity(465920)
    assert parent==dict(pid=465920,start_ticks='4577349167',argv=['bash',str(M67/'run_m67.sh')],cwd=str(M67))
    assert not (M67/'controller.exit').exists() and not (M67/'completed_evidence_audit.json').exists()
    ROOT.mkdir()
    for stage in stages:
        assert not Path(stage['result']).exists()
        stage['queue_sha256']=sha(stage['queue'])
        subprocess.run(['bash','-n',stage['queue']],check=True)
    launch=ROOT/'launch.sh'
    launch.write_text('''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_post_m67_diagnostic_queue_20260907
cd "$root" || exit 1
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/autodl-tmp/envs/sttrack/bin/python -u /root/autodl-tmp/post_m67_diagnostic_queue_20260907.py run > controller.log 2>&1
status=$?
printf '%s\\n' "$status" > controller.exit
exit "$status"
''')
    s=dict(status='frozen_before_post_M67_diagnostics',created_utc=now(),source_sha256=sha(__file__),launch_sha256=sha(launch),
        parent_controller=parent,parent_queue_sha256=sha(M67/'run_m67.sh'),poll_seconds=240,
        dependencies={str(p):h for p,h in SOURCE_HASHES.items()},stages=stages,
        ready_files=['controller.exit','training_control.exit','training_support.exit','control_recursive.exit','support_recursive.exit','recursive_analysis.exit'],
        first_action='Run the already frozen M67 completed artifact and original performance-gate audit.',
        policy='Original M67 content stage only if its original gate passes. M69 is explicitly authorized despite the old M65 failure; M68 and M70 are pure diagnostics. Any execution/parity/artifact failure stops the sequence without retries.',
        automatic_public_evaluation=False,automatic_promotion=False,training_restart_allowed=False,
        parent_training_or_sources_modified=False,goal_completed=False,independent_model_review_pass=False)
    write(ROOT/'spec.json',s);print(json.dumps(dict(spec_sha256=sha(ROOT/'spec.json'),source_sha256=sha(__file__),parent=parent,poll_seconds=240),indent=2))

def checked():
    s=read(ROOT/'spec.json');assert sha(__file__)==s['source_sha256'] and sha(ROOT/'launch.sh')==s['launch_sha256']
    for p,h in s['dependencies'].items():assert sha(p)==h,p
    for stage in s['stages']:assert sha(stage['queue'])==stage['queue_sha256']
    assert s['poll_seconds']==240 and not s['automatic_public_evaluation'] and not s['automatic_promotion']
    return s

def idle():
    used=[int(x.strip()) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(used)==2 and max(used)<500,used
    free=shutil.disk_usage(BASE).free;assert free>1_000_000_000,free
    return dict(gpu_memory_MiB=used,disk_free_bytes=free)

def log(event,**values):
    row=dict(time=now(),event=event,**values)
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(ROOT/'latest.json',row);print(json.dumps(row),flush=True)

def step(name,argv):
    checked();assert not (ROOT/(name+'.exit')).exists()
    log('stage_started',stage=name,argv=argv)
    with (ROOT/(name+'.log')).open('w') as f:
        result=subprocess.run(argv,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    (ROOT/(name+'.exit')).write_text(str(result.returncode)+'\n')
    log('stage_exited',stage=name,exit_code=result.returncode)
    assert result.returncode==0,(name,result.returncode)

def run():
    s=checked();assert not (ROOT/'running_identity.json').exists()
    write(ROOT/'running_identity.json',dict(observed_utc=now(),identity=identity(os.getpid()),source_sha256=sha(__file__),spec_sha256=sha(ROOT/'spec.json')))
    while not (M67/'controller.exit').exists():
        current=identity(s['parent_controller']['pid']);assert current==s['parent_controller']
        log('waiting_for_registered_M67_controller',parent_identity=current,poll_seconds=s['poll_seconds'])
        time.sleep(s['poll_seconds'])
    for name in s['ready_files']:assert (M67/name).read_text().strip()=='0',name
    checked();log('M67_training_and_recursive_controller_completed')
    step('M67_completion_audit',[sys.executable,str(BASE/'audit_m67_completed_20260907.py'),'completed'])
    audit=read(M67/'completed_evidence_audit.json');assert audit['status']=='completed_M67_artifacts_and_development_audited'
    assert audit['auditor_sha256']==SOURCE_HASHES[BASE/'audit_m67_completed_20260907.py']
    outcomes={};performance=bool(audit['paired_development_gate_pass'])
    log('M67_original_gate_audited',paired_development_gate_pass=performance,audit_sha256=sha(M67/'completed_evidence_audit.json'))
    for stage in s['stages']:
        if stage['when']=='M67_original_development_gate_pass' and not performance:
            outcomes[stage['name']]=dict(status='skipped_by_original_M67_development_gate')
            log('stage_skipped',stage=stage['name'],reason='Original M67 paired development gate failed; no threshold changes.')
            continue
        assert not Path(stage['result']).exists()
        resources=idle();log('resources_free',stage=stage['name'],**resources)
        step(stage['name'],['bash',stage['queue']])
        result=read(stage['result']);assert result['status']==stage['expected_status'],stage['name']
        outcomes[stage['name']]=dict(status=result['status'],result_sha256=sha(stage['result']))
        log('stage_result_sealed',stage=stage['name'],**outcomes[stage['name']])
    final=dict(status='completed_authorized_post_M67_diagnostic_sequence',observed_utc=now(),source_sha256=sha(__file__),
        spec_sha256=sha(ROOT/'spec.json'),M67_audit_sha256=sha(M67/'completed_evidence_audit.json'),
        M67_original_development_gate_pass=performance,outcomes=outcomes,
        automatic_public_evaluation=False,automatic_promotion=False,goal_completed=False,independent_model_review_pass=False)
    write(ROOT/'result.json',final);log('authorized_diagnostics_completed',result_sha256=sha(ROOT/'result.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='check':checked();print('QUEUE_AND_FROZEN_DEPENDENCIES_VERIFIED_NO_EXECUTION')
    else:run()
