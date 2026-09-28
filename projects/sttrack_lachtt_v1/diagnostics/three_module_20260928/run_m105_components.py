"""Run the two fixed M105 readouts once, using the completed M104 final."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path('/home/SUTrack_RGBD_L/m105_geometry_components_20260928')
    assert not root.exists()
    root.mkdir()
    python = '/root/autodl-tmp/envs/sttrack/bin/python'
    script = str(Path(__file__).with_name('diagnose_learned_geometry_components.py'))
    common = [python, '-u', script,
              '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
              '--contexts', '/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
              '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
              '--empty-bank', '/root/autodl-tmp/sttrack_full152_paired_20260925/text_full152.pt',
              '--geometry-probe', '/root/autodl-tmp/sttrack_m103_geometry_probe_20260928',
              '--parent-weight', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
              '--parent-result', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
              '--m104-directory', '/root/autodl-tmp/sttrack_m104_visual_geometry_20260928/train',
              '--device', 'cuda']
    started = time.monotonic()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), runs=[])
    processes = []
    for split, gpu in [('fit', 0), ('development', 1)]:
        command = common + ['--split', split, '--output', str(root / split)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        log = (root / (split + '.log')).open('wb')
        process = subprocess.Popen(command, env=env, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT)
        run = dict(split=split, gpu=gpu, pid=process.pid, command=command)
        report['runs'].append(run)
        processes.append((process, log, run))
    (root / 'launch.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    for process, log, run in processes:
        run['exit_code'] = process.wait()
        log.close()
        (root / (run['split'] + '.exit')).write_text(str(run['exit_code']) + '\n')
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic() - started,
                  exit_code=0 if all(run['exit_code'] == 0 for run in report['runs']) else 1)
    (root / 'driver.json').write_text(json.dumps(report, indent=2) + '\n')
    (root / 'driver.exit').write_text(str(report['exit_code']) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(report['exit_code'])


if __name__ == '__main__':
    main()
