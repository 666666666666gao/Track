"""Continue the already frozen full evaluation only after a complete low22 pass."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time

B=Path('/root/autodl-tmp');E=B/'sttrack_m78_raw_competition_20260908/candidate_evaluation';Q=E/'full_followup'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
now=lambda:datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    spec=read(Q/'spec.json')
    assert spec['source_sha256']==sha(__file__) and spec['poll_seconds']==240 and spec['seed']==2027
    assert not (Q/'controller.started.json').exists()
    write(Q/'controller.started.json',dict(pid=os.getpid(),observed_utc=now(),spec_sha256=sha(Q/'spec.json')))
    terminal=E/'low22_controller.exit'
    while not terminal.exists():
        p=Path('/proc')/str(spec['parent_pid']);st=(p/'stat').read_text().split()
        assert st[21]==spec['parent_start_ticks'] and st[2]!='Z'
        assert sha(p/'cmdline')==spec['parent_command_sha256']
        write(Q/'stage.json',dict(stage='waiting_for_complete_low22',observed_utc=now(),verified_parent_pid=spec['parent_pid']))
        time.sleep(spec['poll_seconds'])
    code=int(terminal.read_text().strip())
    if code:
        write(Q/'result.json',dict(status='low22_execution_failed_no_full_run',observed_utc=now(),low22_exit_code=code,full_evaluation_started=False))
        return code
    result=read(E/'low22_result.json')
    assert result['status']=='completed_low22_same_bundle_evaluation'
    assert result['bundle_sha256']==spec['bundle_sha256']==sha(E/'bundle.json')
    assert result['execution_sha256']==spec['low22_execution_sha256']==sha(E/'low22_execution.json')
    assert result['full_three_dataset_evaluation_allowed']==all(result['gate_checks'].values())
    if not result['full_three_dataset_evaluation_allowed']:
        write(Q/'result.json',dict(status='complete_low22_did_not_qualify_for_full',observed_utc=now(),low22_result_sha256=sha(E/'low22_result.json'),
            metrics=result['metrics_percent'],gate_checks=result['gate_checks'],full_evaluation_started=False,goal_achieved=False))
        write(Q/'stage.json',dict(stage='complete_without_full_evaluation',observed_utc=now()))
        return 0
    script=Path(spec['full_script'])
    sources=E/'prepared_entry_and_full.json'
    assert sha(sources)==spec['evaluation_sources_manifest_sha256']
    assert sha(script)==spec['full_script_sha256']==read(sources)['source_files'][script.name]
    assert not (E/'full_evaluation').exists()
    write(Q/'activation.json',dict(observed_utc=now(),low22_result_sha256=sha(E/'low22_result.json'),bundle_sha256=spec['bundle_sha256'],full_script_sha256=sha(script)))
    with (Q/'full_execution.log').open('w') as log:
        child=subprocess.Popen(['bash',str(script)],cwd=str(E),stdout=log,stderr=subprocess.STDOUT)
        write(Q/'stage.json',dict(stage='running_frozen_full_three_dataset_evaluation',observed_utc=now(),pid=child.pid))
        code=child.wait()
    (Q/'full_execution.exit').write_text(str(code)+'\n')
    if code:
        write(Q/'result.json',dict(status='full_evaluation_execution_failed',observed_utc=now(),full_exit_code=code,goal_achieved=False))
        return code
    final=E/'full_evaluation/same_bundle_verification.json';v=read(final)
    assert v['status']=='complete_saved_output_and_same_bundle_verification' and v['bundle_sha256']==spec['bundle_sha256']
    write(Q/'result.json',dict(status='complete_full_evaluation_requires_executor_audit',observed_utc=now(),verification_sha256=sha(final),
        metrics=v['metrics'],all_numeric_targets_pass=v['all_numeric_targets_pass'],goal_completion_claimed=False))
    write(Q/'stage.json',dict(stage='complete_full_evaluation',observed_utc=now()))
    return 0
if __name__=='__main__':raise SystemExit(main())
