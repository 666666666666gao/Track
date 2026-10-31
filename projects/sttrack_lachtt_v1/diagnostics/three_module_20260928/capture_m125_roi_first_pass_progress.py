"""Capture CPU-only evidence for the already running ROI first training pass."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(pid):
    root = Path('/proc') / str(pid)
    command = root.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    fields = dict(line.split(':', 1) for line in root.joinpath('status').read_text().splitlines())
    assert not fields['State'].strip().startswith('Z')
    return dict(pid=pid, ppid=int(fields['PPid']), state=fields['State'].strip(),
                command=[part.decode() for part in command])


def main(root, output):
    launch = read_json(root / 'actual_launch.json')
    stage = read_json(root / 'passes/pass-01/train_fixed_pass.launch.json')
    gate = read_json(root / 'gate.json')
    assert launch['region_arm'] == gate['region_arm'] == 'roi'
    assert launch['fixed_budget_passes'] == launch['fixed_final_pass'] == 6
    assert digest(root / 'gate.json') == launch['gate_sha256']
    supervisor = process(launch['supervisor_pid'])
    assert supervisor['command'] == launch['process_command'] == launch['command']
    processes = [supervisor]
    for _ in range(2):
        parent = processes[-1]['pid']
        children = Path('/proc/%d/task/%d/children' % (parent, parent)).read_text().split()
        assert len(children) == 1, children
        child = process(int(children[0]))
        assert child['ppid'] == parent
        processes.append(child)
    assert processes[1]['command'][-2:] == [str(root / 'code/run_m125_region_fixed_suite.py'), str(root)]
    assert processes[2]['command'] == stage['command']
    log_path = root / 'passes/pass-01/fit/sequence_log.jsonl'
    log_bytes = log_path.read_bytes()
    assert log_bytes.endswith(b'\n')
    rows = [json.loads(line) for line in log_bytes.splitlines()]
    assert [row['sequence_index'] for row in rows] == list(range(len(rows)))
    assert all(row['epoch'] == 1 and row['sequences_per_epoch'] == 152 for row in rows)
    assert all(math.isfinite(row[key]) for row in rows for key in ['AB_loss_mean', 'C_MSE_mean'])
    last = rows[-1]
    assert sum(row['track_calls'] for row in rows) == last['total_track_calls']
    assert not root.joinpath('passes/pass-01/train_fixed_pass.exit').exists()
    seconds_per_call = last['elapsed_seconds'] / last['total_track_calls']
    started = datetime.fromisoformat(stage['started_at'])
    now = datetime.now(timezone.utc)
    source_checks = {path: digest(Path(path)) == expected for path, expected in gate['source_sha256'].items()}
    assert all(source_checks.values())
    output.mkdir()
    output.joinpath('sequence_log.jsonl').write_bytes(log_bytes)
    for path in [root / 'actual_launch.json', root / 'gate.json', root / 'passes/pass-01/train_fixed_pass.launch.json', Path(__file__)]:
        shutil.copyfile(str(path), str(output / path.name))
    summary = dict(status='live_first_pass_observation_not_final_result', observed_at_utc=now.isoformat(),
                   hostname=socket.gethostname(), root=str(root), processes=processes,
                   completed_sequences=len(rows), sequences_per_pass=152, last_completed_sequence=last,
                   all_recorded_losses_finite=True, source_checks=source_checks,
                   seconds_per_call_observed=seconds_per_call, expected_calls_per_pass=219802,
                   estimated_first_pass_end_utc=(started + timedelta(seconds=seconds_per_call * 219802)).isoformat(),
                   estimated_six_pass_training_end_utc=(started + timedelta(seconds=seconds_per_call * 1318812)).isoformat(),
                   estimate_scope='Prefix throughput extrapolation only; excludes remaining suite evaluation and is not a completion claim.',
                   no_neural_execution=True, no_optimizer_updates=True, no_training_changes=True,
                   evidence_sha256={path.name: digest(path) for path in output.iterdir()})
    output.joinpath('summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: summary[key] for key in ['status', 'observed_at_utc', 'completed_sequences',
          'seconds_per_call_observed', 'estimated_first_pass_end_utc', 'estimated_six_pass_training_end_utc']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
