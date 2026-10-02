"""One two-GPU M106 sanity/full pair; no retries or promotion."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path('/root/autodl-tmp/sttrack_m106_selected_geometry_20260928')
    assert not root.exists()
    root.mkdir()
    python = '/root/autodl-tmp/envs/sttrack/bin/python'
    script = str(Path(__file__).with_name('train_selected_geometry_preservation.py'))
    common = [python, '-u', script,
              '--cache', '/root/autodl-tmp/sttrack_m90_train_states_20260928',
              '--contexts', '/root/autodl-tmp/sttrack_m98_train_contexts_20260928',
              '--origins', '/root/autodl-tmp/sttrack_m95_initial_origins_20260928',
              '--empty-bank', '/root/autodl-tmp/sttrack_full152_paired_20260925/text_full152.pt',
              '--geometry-probe', '/root/autodl-tmp/sttrack_m103_geometry_probe_20260928',
              '--parent-weight', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt',
              '--parent-result', '/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/result.json',
              '--repository', '/home/SUTrack_RGBD_L', '--device', 'cuda']
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), runs=[])
    started = time.monotonic()
    for mode in ['sanity', 'train']:
        processes = []
        for weight, gpu in [(0, 0), (1, 1)]:
            name = mode + '_weight' + str(weight)
            command = common + ['--mode', mode, '--selected-preservation-weight', str(weight),
                                '--output', str(root/name)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            log = (root/(name+'.log')).open('wb')
            process = subprocess.Popen(command, env=env, cwd=Path(__file__).parent,
                                       stdout=log, stderr=subprocess.STDOUT)
            row = dict(mode=mode, weight=weight, gpu=gpu, pid=process.pid, command=command)
            report['runs'].append(row); processes.append((process, log, row, name))
        (root/'launch.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
        for process, log, row, name in processes:
            row['exit_code'] = process.wait(); log.close()
            (root/(name+'.exit')).write_text(str(row['exit_code'])+'\n')
        if any(row['exit_code'] != 0 for _, _, row, _ in processes):
            break
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-started,
                  exit_code=0 if len(report['runs'])==4 and all(r['exit_code']==0 for r in report['runs']) else 1)
    (root/'driver.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'driver.exit').write_text(str(report['exit_code'])+'\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(report['exit_code'])


if __name__ == '__main__':
    main()
