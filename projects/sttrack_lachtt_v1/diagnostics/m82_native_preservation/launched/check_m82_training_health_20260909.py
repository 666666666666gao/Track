from pathlib import Path
from datetime import datetime,timezone,timedelta
import hashlib,json,shutil,subprocess
R=Path('/root/autodl-tmp/sttrack_m82_native_preservation_20260909');B=R.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
launch=json.loads((R/'launch.json').read_text());now=datetime.now(timezone.utc)
assert (now-datetime.fromisoformat(launch['observed_utc'])).total_seconds()>=240
f=json.loads((R/'frozen.json').read_text())
for n,h in f['source_sha256'].items():assert sha(R/n)==h
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits']).decode().strip()
processes=[]
for line in raw.splitlines():
    pid,memory=[x.strip() for x in line.split(',')]
    p=Path('/proc')/pid
    if (p/'cwd').resolve()!=R:continue
    args=(p/'cmdline').read_bytes().decode().split('\x00')
    if 'train_causal.py' not in args:continue
    arm=args[args.index('--arm')+1]
    processes.append(dict(pid=int(pid),arm=arm,gpu_memory_mib=int(memory)))
arms={}
for arm in ['category','empty']:
    e=R/('training_'+arm+'.exit');log=R/('training_'+arm+'.log');folder=R/'training'/arm
    q=folder/'sequence_log.jsonl'
    records=[json.loads(x) for x in q.read_text().splitlines() if x.strip()] if q.exists() else []
    arm_record=dict(exit=e.read_text().strip() if e.exists() else None,completed_sequences=len(records),latest_record=records[-1] if records else None,log_tail=log.read_text()[-1800:] if not records else None)
    if records:
        last=records[-1];rate=last['total_track_calls']/last['elapsed_seconds']
        arm_record.update(observed_track_calls_per_second=rate,estimated_remaining_training_seconds=(186694-last['total_track_calls'])/rate,estimated_training_finish_utc=(now+timedelta(seconds=(186694-last['total_track_calls'])/rate)).isoformat())
        assert last['cumulative_native_eligible_frames']>0 and last['cumulative_preservation_kl_sum']>0
        assert last['total_optimizer_steps']>0
    arms[arm]=arm_record
qwen={}
for name in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']:
    files=sorted((B/'qwen'/name).rglob('*.safetensors'))
    qwen[name]=dict(shards=len(files),bytes=sum(p.stat().st_size for p in files))
    assert files
record=dict(status='first_paired_training_health_observation',observed_utc=now.isoformat(),seconds_since_launch=(now-datetime.fromisoformat(launch['observed_utc'])).total_seconds(),processes=processes,arms=arms,free_disk_bytes=shutil.disk_usage(R).free,qwen_retained=qwen,source_hashes_unchanged=True,performance_results_exist=(R/'recursive_result.json').exists(),audit_queued=not (R/'postrun_audit.exit').exists())
(R/'first_training_health.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
assert {p['arm'] for p in processes}=={'category','empty'}
assert all(x['exit'] is None and x['completed_sequences']>=1 for x in arms.values())
