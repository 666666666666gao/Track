"""Publishable M57 completed preparation/training/static evidence, recursion pending."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tarfile

root = Path('/root/autodl-tmp/sttrack_m57_initial_binding_v1_20260906')
destination = Path('/root/autodl-tmp/sttrack_m57_training_static_completed_20260906')
sys.path.insert(0, str(root))
from run_recursive import bound
from train import sha, write

spec, recursive, training, cases = bound(root)
for phase in ['collection', 'runtime_contract', 'training', 'static']:
    assert (root / (phase + '.exit')).read_text().strip() == '0'
contract = json.loads((root / 'runtime_contract/contract.json').read_text())
assert contract['status'] == 'complete_runtime_input_contract'
assert len(contract['arms']) == 4 and all(row['runtime_input_tensors_exactly_equal'] for row in contract['arms'])
static = json.loads((root / 'static_result.json').read_text())
assert static['status'] == 'complete_static_diagnostic' and not static['promotion_decision']
assert static['static_spec_sha256'] == sha(root / 'static_spec.json')
assert static['training_result_sha256'] == sha(root / 'training_result.json')
assert len(static['variants']) == 8 and static['original_slot_masks_unchanged']
import torch
heads = {}
for arm in spec['variants']:
    path = root / 'training' / (arm + '_final.pth')
    saved = torch.load(path, map_location='cpu')
    assert saved['m57_spec_sha256'] == sha(root / 'spec.json') and saved['variant'] == arm
    assert saved['epochs'] == 20 and saved['optimizer_steps'] == 960
    assert saved['base_checkpoint_sha256'] == spec['base_checkpoint_sha256']
    heads[arm] = dict(sha256=sha(path), bytes=path.stat().st_size, variant=saved['variant'],
        reference_mode=saved['reference_mode'], use_text=saved['use_text'], epochs=20, optimizer_steps=960)
    assert heads[arm]['sha256'] == training['variants'][arm]['checkpoint_sha256']
source = ['train.py', 'freeze_training.py', 'run_recursive.py', 'freeze_recursive.py', 'queue_recursive.py',
    'finish_recursive.py', 'analyze_static.py', 'freeze_static.py', 'collect_initial.py',
    'prepare_collection.py', 'check_runtime_inputs.py', 'runtime.py']
files = [
    'base_decision.json', 'EXPERIMENT_PLAN.md', 'collection_spec.json',
    'initial_references/receipt.json', 'collection_launch.json', 'collection.log', 'collection.exit',
    'run_collection.sh', 'runtime_contract_spec.json', 'runtime_contract/contract.json',
    'runtime_contract_launch.json', 'runtime_contract.log', 'runtime_contract.exit', 'run_runtime_contract.sh',
    'spec.json', 'training_result.json', 'training_seal.json', 'training_launch.json',
    'training.log', 'training.exit', 'run_training.sh',
    'static_spec.json', 'static_result.json', 'static_launch.json', 'static.log', 'static.exit', 'run_static.sh',
    'recursive_spec.json', 'recursive_queue0_launch.json', 'recursive_queue1_launch.json',
    'run_recursive_queue0.sh', 'run_recursive_queue1.sh', 'analysis_scheduler_launch.json',
    'analysis_wait_record.json', 'run_analysis_scheduler.sh']
files += ['training/' + arm + '_result.json' for arm in spec['variants']]
files += [arm + '_recursive_launch.json' for arm in ['initial_text', 'candidate_text']]
destination.mkdir()
for name in files:
    path = destination / name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / name, path)
(destination / 'source').mkdir()
for name in source:
    shutil.copyfile(root / name, destination / 'source' / name)
for name in ['lachtt_initial_instance_alignment.py', 'sttrack_initial_instance_observation.py']:
    folder = 'lib/models/sttrack' if name.startswith('lachtt_') else 'lib/test/tracker'
    shutil.copyfile(Path(spec['repository']) / folder / name, destination / 'source' / name)
shutil.copyfile(__file__, destination / 'source/export_training_static.py')
progress = {}
for gpu in [0, 1]:
    name = 'recursive_queue%d' % gpu
    path = root / (name + '.exit')
    assert not path.exists(), 'Export intended for the running snapshot; use a completion export if finished'
for arm in spec['variants']:
    folder = root / 'recursive' / arm
    paths = sorted(folder.glob('*.json')) if folder.exists() else []
    progress[arm] = dict(completed_sequences=len(paths),
        completed_frames=sum(len(json.loads(path.read_text())['rows']) for path in paths),
        terminal_receipt_present=(root / (arm + '_recursive_receipt.json')).exists())
record = dict(status='completed_collection_contract_training_static_recursion_running',
    observed_utc=datetime.now(timezone.utc).isoformat(), base_checkpoint_sha256=spec['base_checkpoint_sha256'],
    source_files_verified=len(spec['source_sha256']), initial_reference_sequences=85,
    runtime_contract_sha256=sha(root / 'runtime_contract/contract.json'),
    training_spec_sha256=sha(root / 'spec.json'), training_result_sha256=sha(root / 'training_result.json'),
    static_spec_sha256=sha(root / 'static_spec.json'), static_result_sha256=sha(root / 'static_result.json'),
    recursive_spec_sha256=sha(root / 'recursive_spec.json'), head_metadata_verified=heads,
    recursive_progress_snapshot=progress, recursive_metrics_available=False,
    public_evaluation=False, adopted_weight=False, new_independent_result_review=False,
    omitted='No checkpoint tensors, text/feature banks, dataset images, numeric training labels or raw predictions in this bundle')
write(destination / 'completion_binding.json', record)
manifest = [dict(path=path.relative_to(destination).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(destination.rglob('*')) if path.is_file()]
write(destination / 'bundle_manifest.json', manifest)
archive = destination.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as stream:
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            stream.add(path, arcname=path.relative_to(destination).as_posix())
print(json.dumps(dict(status=record['status'], archive=str(archive), archive_bytes=archive.stat().st_size,
    archive_sha256=sha(archive), record_files=len(manifest), static_result_sha256=record['static_result_sha256'],
    recursive_progress=progress), indent=2))
