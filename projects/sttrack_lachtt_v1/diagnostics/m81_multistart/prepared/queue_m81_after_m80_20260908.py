"""Wait for the live M80 controller, then launch the frozen M81 pair on idle GPUs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess,time
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';D=B/'sttrack_m80_block_text_dropout_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def checked():
    spec=read(R/'queue_spec.json');f=read(R/'frozen.json');t=read(R/'training_spec.json')
    assert sha(__file__)==spec['source_sha256']
    assert sha(R/'frozen.json')==spec['frozen_sha256']
    assert sha(R/'training_spec.json')==f['training_spec_sha256']
    assert sha(R/'evaluation_spec.json')==f['evaluation_spec_sha256']
    assert sha(R/'train_causal.py')==t['training_script_sha256'] and sha(R/'run.sh')==t['run_queue_sha256']
    assert sha(D/'run.sh')==spec['dependency_run_sha256']
    assert sha(D/'training_spec.json')==spec['dependency_training_sha256']
    assert not (R/'training').exists() and not (R/'launch.json').exists()
    return spec,t
spec,t=checked()
while True:
    checked()
    state=dict(observed_utc=datetime.now(timezone.utc).isoformat(),scheduler_pid=__import__('os').getpid(),dependency=str(D),poll_seconds=240,training_started=False)
    exit_path=D/'controller.exit'
    if not exit_path.exists():
        screens=subprocess.run(['screen','-ls'],capture_output=True,text=True)
        live='.sttrack_m80_dropout_20260908' in screens.stdout
        state.update(status='waiting_for_live_M80',dependency_screen_live=live)
        write(R/'queue_state.json',state);print(json.dumps(state),flush=True)
        assert live,'M80 has no exit receipt and its screen is absent; inspect before rescheduling.'
        time.sleep(240);continue
    assert exit_path.read_text().strip()=='0','M80 terminated abnormally; preserve evidence and inspect.'
    result=read(D/'result.json');assert result['status']=='completed_M80_development_and_content'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(x.split(',')[0]):int(x.split(',')[1]) for x in gpu.strip().splitlines()}
    assert set(memory)=={0,1}
    if any(v>=500 for v in memory.values()):
        state.update(status='waiting_for_free_GPUs_after_M80',memory_mib=memory,dependency_exit=0)
        write(R/'queue_state.json',state);print(json.dumps(state),flush=True);time.sleep(240);continue
    assert shutil.disk_usage(R).free>700000000
    for arm in ['category','empty']:
        bank=t['banks']['fit'][arm];assert sha(bank['path'])==bank['sha256']
    subprocess.run(['screen','-dmS','sttrack_m81_pair_20260908','bash',str(R/'run.sh')],check=True)
    state.update(status='M81_pair_detached_launch_requested',training_started=True,seed=2027,training_spec_sha256=sha(R/'training_spec.json'),run_sha256=sha(R/'run.sh'),dependency_result_sha256=sha(D/'result.json'),memory_before_launch_mib=memory,free_disk_bytes=shutil.disk_usage(R).free)
    write(R/'launch.json',state);write(R/'queue_state.json',state);print(json.dumps(state),flush=True)
    break
