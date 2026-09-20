from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import time

root = Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
controller = Path('/proc/6605')
terminal = root / 'controller.exit'

def snapshot():
    rows = [json.loads(line) for line in (root / 'training/category/sequence_log.jsonl').read_text().splitlines()]
    last = rows[-1]
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), completed_sequences=len(rows),
                track_calls=last['total_track_calls'], optimizer_steps=last['total_optimizer_steps'],
                free_disk_bytes=shutil.disk_usage(root).free,
                stage_exits={p.name: p.read_text().strip() for p in root.glob('*.exit')})

if not terminal.exists():
    command = controller.joinpath('cmdline').read_bytes()
    started = controller.joinpath('stat').read_text().split()[21]
    assert b'run_m84.sh' in command
    print(json.dumps(dict(phase='before_wait', **snapshot())), flush=True)
    time.sleep(240)
    if not terminal.exists():
        assert controller.joinpath('cmdline').read_bytes() == command
        fields = controller.joinpath('stat').read_text().split()
        assert fields[21] == started and fields[2] != 'Z'
print(json.dumps(dict(phase='after_wait', controller_terminal=terminal.exists(), **snapshot())), flush=True)
