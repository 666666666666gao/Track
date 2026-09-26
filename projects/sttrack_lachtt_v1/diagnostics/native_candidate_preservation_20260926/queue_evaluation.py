"""Wait for the identified M89 training controller, then run evaluation once."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/sttrack_native_candidate_preservation_20260926')


def main():
    package = json.loads((ROOT / 'evaluation_package.json').read_text())
    for path, digest in package['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    launch = json.loads((ROOT / 'training_launch.json').read_text())
    assert launch['controller_pid'] == package['training_controller_pid']
    assert launch['experiment_spec_sha256'] == package['training_spec_sha256']
    record = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  package_sha256=hashlib.sha256((ROOT / 'evaluation_package.json').read_bytes()).hexdigest())
    with (ROOT / 'evaluation_queue_launch.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)
    next_check = datetime.fromisoformat(launch['started_utc']).timestamp() + 3600
    while True:
        time.sleep(max(0, next_check - time.time()))
        next_check = time.time() + 3600
        completion = ROOT / 'training.exit'
        if completion.exists():
            assert completion.read_text().strip() == '0'
            for arm in ['control', 'candidate']:
                assert (ROOT / f'training_{arm}.exit').read_text().strip() == '0'
            break
        proc = Path('/proc') / str(launch['controller_pid'])
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert str(ROOT / 'run_pair.py') in command and '--phase training' in command, command
        assert (proc / 'stat').read_text().split(') ', 1)[1][0] != 'Z'
        print(json.dumps(dict(status='waiting_for_training', observed_utc=datetime.now(timezone.utc).isoformat())), flush=True)
    for path, digest in package['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
    assert len(memory) == 2 and all(int(value) < 500 for value in memory), memory
    with (ROOT / 'evaluation_controller.log').open('x') as log:
        job = subprocess.Popen(['bash', str(ROOT / 'run_evaluation.sh')], stdout=log, stderr=subprocess.STDOUT)
        with (ROOT / 'evaluation_process.json').open('x') as stream:
            json.dump(dict(pid=job.pid, started_utc=datetime.now(timezone.utc).isoformat()), stream, indent=2)
        result = job.wait()
    (ROOT / 'evaluation_queue.exit').write_text(str(result) + '\n')
    assert result == 0, result


if __name__ == '__main__':
    main()
