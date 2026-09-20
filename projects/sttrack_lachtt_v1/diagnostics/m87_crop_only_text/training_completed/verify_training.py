from pathlib import Path
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
import torch

ROOT = Path('/root/autodl-tmp/sttrack_m87_crop_only_text_20260921')
OUT = ROOT / 'training_completion'
TRAIN = ROOT / 'training/category'
EXPECTED_SPEC = 'd14946d36262ee9c23153af8927bce7f3d07aa0383ae2f1c2c02933662448860'
EXPECTED_FINAL = 'dbf86cdf5fc5f9b30863725ca0e95cb0cbbde7e65977a5bfb02387d89fbc756b'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_finite(value):
    if torch.is_tensor(value):
        assert bool(torch.isfinite(value).all())
    elif isinstance(value, dict):
        for item in value.values():
            check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)

assert not (OUT / 'verification.json').exists()
assert (ROOT / 'training_category.exit').read_text().strip() == '0'
assert sha(ROOT / 'training_spec.json') == EXPECTED_SPEC
spec = json.loads((ROOT / 'training_spec.json').read_text())
result = json.loads((TRAIN / 'result.json').read_text())
rows = [json.loads(line) for line in (TRAIN / 'sequence_log.jsonl').read_text().splitlines()]
assert len(rows) == 130
assert [row['sequence'] for row in rows] == [row['sequence'] for row in spec['sequence_order']]
assert [row['sequence_index'] for row in rows] == list(range(130))
assert sum(row['track_calls'] for row in rows) == 186694
assert rows[-1]['total_track_calls'] == result['total_track_calls'] == 186694
assert rows[-1]['total_optimizer_steps'] == result['optimizer_steps'] == 5798
assert result['status'] == 'one_full_causal_fit_pass_complete'
assert result['training_spec_sha256'] == EXPECTED_SPEC
assert sha(TRAIN / 'final.pth') == result['final_checkpoint_sha256'] == EXPECTED_FINAL
assert sha(TRAIN / 'sequence_log.jsonl') == result['sequence_log_sha256']
assert sha(TRAIN / 'sampled_state_trace.jsonl') == result['sampled_trace_sha256']
assert sum(result['training_label_counts'].values()) == 186694
check_finite(rows)
checkpoint = torch.load(str(TRAIN / 'final.pth'), map_location='cpu')
assert checkpoint['architecture'] == 'semantic_spatial_centered_v1'
assert checkpoint['status'] == 'complete'
assert checkpoint['seed'] == 2027
assert checkpoint['arm'] == 'category'
assert checkpoint['completed_sequences'] == 130
assert checkpoint['frame_count'] == 186694
assert checkpoint['optimizer_steps'] == checkpoint['actual_dataset_optimizer_steps'] == 5798
assert checkpoint['training_spec_sha256'] == EXPECTED_SPEC
assert checkpoint['base_checkpoint_sha256'] == spec['native_checkpoint_sha256']
check_finite(checkpoint['model'])
check_finite(checkpoint['optimizer'])
processes = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,args'], universal_newlines=True)
evaluation_processes = [line.strip() for line in processes.splitlines()
                        if 'run_recursive.py --arm' in line and 'python' in line]
assert evaluation_processes
verification = dict(
    status='saved_complete_training_verified_development_pending',
    observed_utc=datetime.now(timezone.utc).isoformat(),
    root=str(ROOT), training_exit=0, sequences=130, track_calls=186694,
    optimizer_steps=5798, seed=2027,
    final_checkpoint_sha256=EXPECTED_FINAL,
    final_checkpoint_bytes=(TRAIN / 'final.pth').stat().st_size,
    training_spec_sha256=EXPECTED_SPEC,
    files_sha256={name: sha(TRAIN / name) for name in
                  ['final.pth', 'result.json', 'sequence_log.jsonl', 'sampled_state_trace.jsonl']},
    sequence_order_and_counts_match=True, model_optimizer_and_log_values_finite=True,
    base_preservation='Training routine reports unchanged parameters and buffers; this collection does not independently replay the full base state.',
    live_evaluation_processes=evaluation_processes,
    development_metrics_read=False,
    verification_source_sha256=sha(Path(__file__)),
)
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
print(json.dumps(verification))
