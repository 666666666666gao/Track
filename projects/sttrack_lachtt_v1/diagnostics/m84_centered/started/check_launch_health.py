from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
R=Path(__file__).parent
launch=json.loads((R/'launch.json').read_text())
pid=launch['training_pid'];cmd=Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
assert 'train_causal.py --arm category' in cmd
assert not (R/'training_category.exit').exists()
path=R/'training/category/sequence_log.jsonl'
rows=[json.loads(line) for line in path.read_text().splitlines()]
assert rows
traces=R/'training/category/sampled_state_trace.jsonl'
last_trace=json.loads(traces.read_text().splitlines()[-1])
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
health=dict(status='training_alive_with_completed_sequence',observed_utc=datetime.now(timezone.utc).isoformat(),
    training_pid=pid,completed_sequences=len(rows),last_sequence=rows[-1],last_sampled_trace=last_trace,
    free_disk_bytes=shutil.disk_usage(R).free,gpu=gpu,formal_performance_available=False,
    observed_launch_sha256=hashlib.sha256((R/'launch.json').read_bytes()).hexdigest())
(R/'launch_health.json').write_text(json.dumps(health,indent=2)+'\n')
print(json.dumps(health,indent=2))
