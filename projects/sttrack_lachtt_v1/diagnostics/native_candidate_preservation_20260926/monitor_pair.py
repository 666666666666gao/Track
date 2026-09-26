"""Observe the fixed training pair once per hour; never restart jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')


def main():
    launch = json.loads((ROOT / 'training_launch.json').read_text())
    next_check = datetime.fromisoformat(launch['started_utc']).timestamp() + 3600
    while True:
        time.sleep(max(0, next_check - time.time()))
        snapshot = dict(observed_utc=datetime.now(timezone.utc).isoformat(), arms={})
        for name, pid in zip(('control', 'candidate'), launch['pids']):
            command = Path(f'/proc/{pid}/cmdline')
            live = command.exists() and str(ROOT / 'train_candidate.py').encode() in command.read_bytes()
            output = ROOT / 'training' / name
            results = output / 'result.json'
            log = output / 'sequence_log.jsonl'
            lines = log.read_text().splitlines() if log.exists() else []
            last = json.loads(lines[-1]) if lines else None
            complete = results.exists() and json.loads(results.read_text())['status'] == 'one_full_causal_fit_pass_complete'
            snapshot['arms'][name] = dict(pid=pid, live=live, complete=complete, completed_sequences=len(lines), last_sequence=last)
        snapshot['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], text=True).splitlines()
        with (ROOT / 'hourly_status.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot) + '\n')
        print(json.dumps(snapshot), flush=True)
        assert all(arm['live'] or arm['complete'] for arm in snapshot['arms'].values()), 'A training arm exited without a complete result; inspect its existing log.'
        if all(arm['complete'] for arm in snapshot['arms'].values()):
            (ROOT / 'monitor_complete.json').write_text(json.dumps(snapshot, indent=2) + '\n')
            return
        next_check = time.time() + 3600


if __name__ == '__main__':
    main()
