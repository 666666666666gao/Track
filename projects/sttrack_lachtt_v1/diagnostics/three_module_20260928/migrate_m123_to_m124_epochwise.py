"""One requested handoff: retain original evidence; resume saved whole-sequence work."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(states):
    h = hashlib.sha256()
    for name, value in states.items():
        h.update(name.encode()); h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    root = Path(sys.argv[1]); request = json.loads((root / 'MIGRATION_REQUEST.json').read_text())
    assert datetime.now(timezone.utc) >= datetime.fromisoformat(request['next_NN_observation_not_before'])
    assert request['maximum_epochs'] == 20 and request['source_review_complete']
    old = Path(request['old_root']); old_gate = json.loads((old / 'gate.json').read_text())
    for path, value in request['source_sha256'].items():
        assert sha(path) == value, path
    for path, value in old_gate['source_sha256'].items():
        assert sha(path) == value, path
    for path, value in old_gate['input_sha256'].items():
        assert sha(path) == value, path
    report = Path(request['source_review_report_path'])
    assert sha(report) == request['source_review_report_sha256']
    assert not (root / 'actual_launch.json').exists() and not (root / 'handoff').exists()
    pid = request['old_trainer_pid']; process = Path('/proc') / str(pid) / 'cmdline'
    command = process.read_bytes().replace(b'\0', b' ').decode()
    assert str(old / 'code/train_m123_ABC_joint_full152.py') in command
    assert '--output ' + str(old / 'fit') in command
    assert (old / 'fit/latest.pt').is_file()
    import torch
    latest = old / 'fit/latest.pt'
    before = latest.stat()
    checkpoint_bytes = latest.read_bytes()
    after = latest.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    assert len(checkpoint_bytes) == after.st_size
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu')
    assert saved['epoch'] == 1 and 0 < saved['completed_sequences'] < 152
    log_bytes = (old / 'fit/sequence_log.jsonl').read_bytes()
    prefix = log_bytes.splitlines()[:saved['completed_sequences']]
    assert len(prefix) == saved['completed_sequences']
    records = [json.loads(line) for line in prefix]
    spec = json.loads(Path(old_gate['spec']).read_text())
    for index, row in enumerate(records):
        case = spec['sequence_order'][index]
        assert row['epoch'] == 1 and row['sequence_index'] == index and row['sequence'] == case['sequence']
        assert row['track_calls'] == case['rgb_frames'] - 1
    assert sum(row['track_calls'] for row in records) == saved['track_calls']
    assert records[-1]['total_optimizer_steps'] == saved['optimizer_steps']
    assert records[-1]['total_C_optimizer_steps'] == saved['C_optimizer_steps']
    (root / 'handoff').mkdir()
    original = root / 'handoff/prestop_latest.pt'
    original.write_bytes(checkpoint_bytes)
    (root / 'handoff/prestop_sequence_log_original.jsonl').write_bytes(log_bytes)
    (root / 'handoff/prestop_sequence_log_committed_prefix.jsonl').write_bytes(b'\n'.join(prefix) + b'\n')
    assert sha(original) == hashlib.sha256(checkpoint_bytes).hexdigest()
    provenance = dict(original_checkpoint_sha256=sha(original), original_root=str(old),
                      original_trainer_pid=pid, original_command=command,
                      saved_complete_sequences=saved['completed_sequences'], saved_track_calls=saved['track_calls'],
                      saved_optimizer_steps=saved['optimizer_steps'],
                      original_sequence_log_sha256=sha(root / 'handoff/prestop_sequence_log_original.jsonl'),
                      committed_sequence_log_sha256=sha(root / 'handoff/prestop_sequence_log_committed_prefix.jsonl'),
                      resume_source='sealed_pre_stop_checkpoint_and_committed_sequence_log',
                      reason='user requested max20, per-epoch Test50/CDTB80, patience2, best then VOT',
                      RNG_missing_in_original_checkpoint=True, complete_sequence_boundary_only=True,
                      partial_sequence_repeated_from_its_first_frame=True)
    canonical = dict(saved, format='M124_joint_sequence_boundary_v1',
                     supervised_frames=sum(row['supervised_frames'] for row in records),
                     C_supervised_frames=sum(row['C_supervised_frames'] for row in records),
                     sequence_records=records, spec_sha256=sha(old_gate['spec']), frozen_state_sha256=None,
                     model_state_sha256=dict(A_B=digest(saved['A_B']), C=digest(saved['C'])),
                     resume_RNG_saved=False, legacy_handoff_RNG_exact=False, legacy_handoff=provenance)
    checkpoint = root / 'handoff/resume.pt'; torch.save(canonical, checkpoint)
    restored = torch.load(checkpoint, map_location='cpu')
    assert restored['track_calls'] == saved['track_calls'] and restored['optimizer_steps'] == saved['optimizer_steps']
    for kind in ['A_B', 'C']:
        assert all(torch.equal(value, restored[kind][name]) for name, value in saved[kind].items())
    assert restored['optimizer']['param_groups'] == saved['optimizer']['param_groups']
    for key, state in saved['optimizer']['state'].items():
        for name, value in state.items():
            other = restored['optimizer']['state'][key][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    # Prepare and verify the complete model/Adam recovery before stopping the old task.
    # This is a deliberate user-requested protocol change, never a timeout restart.
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while not (old / 'controller.supervised.exit').exists() and time.monotonic() < deadline:
        time.sleep(1)
    assert (old / 'controller.supervised.exit').is_file()
    assert (old / 'train_ABC_complete_all152.exit').read_text().strip() == '-15'
    assert (old / 'controller.supervised.exit').read_text().strip() != '0'
    assert not process.exists()
    for name in ['train_ABC_complete_all152.receipt.json', 'train_ABC_complete_all152.exit',
                 'controller.supervisor.terminal.json', 'controller.supervised.exit',
                 'train_ABC_complete_all152.log', 'fit/sequence_log.jsonl', 'fit/sampled_state_trace.jsonl', 'fit/latest.pt']:
        target = root / 'handoff' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / name, target)
        assert sha(old / name) == sha(target)
    provenance['post_stop_checkpoint_retained_not_used_for_resume'] = True
    gate = dict(old_gate, status='reviewed_M124_epochwise_joint_full152_source_gate',
                code_root=str(root / 'code'), root=str(root), maximum_epochs=20, early_stop_patience=2,
                handoff_checkpoint=str(checkpoint), initial_OPE_F_percent=request['initial_OPE_F_percent'],
                source_sha256=dict(old_gate['source_sha256'], **request['source_sha256']),
                input_sha256=dict(old_gate['input_sha256'], **{str(checkpoint): sha(checkpoint)}),
                source_review_report_path=str(report), source_review_report_sha256=sha(report),
                external_metric_checkpoint_selection=True, Test50_CDTB80_used_for_checkpoint_selection=True)
    (root / 'gate.json').write_text(json.dumps(gate, indent=2) + '\n')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    assert len(gpu.strip().splitlines()) == 2
    assert all(int(line.split(',')[1]) < 500 for line in gpu.strip().splitlines())
    command = [gate['python'], '-B', '-u', str(root / 'code/m124_epochwise_process_supervisor.py'), str(root)]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([gate['code_root'], gate['base_code_root'], '/home/SUTrack_RGBD_L']),
                       CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONOPTIMIZE='0', PYTHONDONTWRITEBYTECODE='1')
    with (root / 'controller.log').open('xb') as stream:
        child = subprocess.Popen(command, cwd=gate['code_root'], env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(status='actual_M124_epochwise_supervisor_spawned_from_saved_M123_progress',
                  supervisor_pid=child.pid, command=command, initial_poll_exit=child.poll(),
                  started_at=datetime.now(timezone.utc).isoformat(), GPU_snapshot=gpu,
                  GPU_queries=1, NN_progress_snapshot_count=1, root=str(root),
                  handoff=provenance, handoff_checkpoint_sha256=sha(checkpoint), gate_sha256=sha(root / 'gate.json'),
                  maximum_epochs=20, training_completed=False, no_development_split=True)
    (root / 'actual_launch.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
