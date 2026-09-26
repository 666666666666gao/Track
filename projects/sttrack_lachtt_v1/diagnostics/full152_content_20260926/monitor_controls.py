"""Hourly observation and CPU analysis of the existing content-control job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


ROOT = Path('/root/autodl-tmp/sttrack_full152_content_20260926')
METRIC_PYTHON = '/root/miniconda3/envs/mplt/bin/python'


def main(controller_pid, not_before):
    next_check = datetime.fromisoformat(not_before).timestamp()
    analyzed = set()
    jobs = [(m, v, d) for m in ['M82', 'M67'] for v in ['empty', 'swapped'] for d in ['depthtrack', 'cdtb']]
    while True:
        time.sleep(max(0, next_check - time.time()))
        observed = datetime.now(timezone.utc).isoformat()
        command_path = Path(f'/proc/{controller_pid}/cmdline')
        command = command_path.read_bytes().replace(b'\0', b' ').decode() if command_path.exists() else None
        live = command is not None and 'rgbd_full152_content_20260926' in command
        exit_path = ROOT / 'controls.exit'
        exit_code = exit_path.read_text().strip() if exit_path.exists() else None
        ready = [job for job in jobs if (ROOT / job[0] / job[2] / job[1] / 'predictions/metrics.json').exists()]
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True).stdout.strip().splitlines()
        processes = subprocess.run(['ps', '-eo', 'pid,etimes,args'], check=True, capture_output=True, text=True).stdout
        workers = [line.strip() for line in processes.splitlines()
                   if str(ROOT) in line and 'run_semantic_ope.py' in line]
        progress = {}
        for model, variant, dataset in jobs:
            log = ROOT / 'logs' / f'{model}_{dataset}_{variant}_track.log'
            if log.exists():
                receipts = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{"sequence":')]
                progress[f'{model}/{dataset}/{variant}'] = dict(completed_sequences=len(receipts),
                    completed_frames=sum(r['frames'] for r in receipts),
                    cumulative_seconds=receipts[-1]['cumulative_seconds'] if receipts else None)
        snapshot = dict(observed_utc=observed, controller_pid=controller_pid, controller_live=live,
                        controls_exit=exit_code, gpu=gpu, workers=workers, progress=progress,
                        metrics_ready=['/'.join(job) for job in ready])
        with (ROOT / 'hourly_status.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot) + '\n')
        print(json.dumps(snapshot), flush=True)
        for model, variant, dataset in ready:
            job = (model, variant, dataset)
            if job not in analyzed:
                log_path = ROOT / 'logs' / f'{model}_{dataset}_{variant}_posthoc.log'
                with log_path.open('w') as log:
                    subprocess.run([METRIC_PYTHON, str(ROOT / 'analyze_content.py'), '--model', model,
                                    '--dataset', dataset, '--variant', variant], check=True, stdout=log, stderr=subprocess.STDOUT)
                analyzed.add(job)
        if exit_code == '0':
            assert len(ready) == len(analyzed) == 8
            (ROOT / 'analysis_complete.json').write_text(json.dumps(dict(status='complete',
                observed_utc=observed, analyzed_jobs=['/'.join(job) for job in sorted(analyzed)]), indent=2) + '\n')
            return
        if not live:
            raise RuntimeError('The content controller is no longer live and has no success exit.')
        next_check = time.time() + 3600


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--not-before', required=True)
    args = parser.parse_args()
    main(args.controller_pid, args.not_before)
