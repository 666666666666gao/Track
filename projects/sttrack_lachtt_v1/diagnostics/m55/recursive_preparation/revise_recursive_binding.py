"""Add direct checks of existing training metadata before any M55 evaluation."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

root = Path('/root/autodl-tmp/sttrack_m55_tsg_direction_v2_20260906')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


old_spec_sha = '13fc33cbe44ea998ae2ecda22d920e64920049093d10f2eb8b3a94aca3d89897'
old_runner_sha = '02a532765ec89324ed2059d17241aa8f22888480d804472ca841858d4ed495ed'
assert sha(root / 'recursive_spec.json') == old_spec_sha
assert sha(root / 'run_recursive.py') == old_runner_sha
assert not (root / 'recursive').exists()
assert not (root / 'recursive_queue_launch.json').exists()
archive = root / 'recursive_binding_v1'
archive.mkdir()
for name in ['recursive_spec.json', 'run_recursive.py']:
    shutil.copyfile(root / name, archive / name)
candidate = root / 'run_recursive_metadata_checked.py'
spec = json.loads((root / 'recursive_spec.json').read_text())
spec['schema'] = 'sttrack_m55_final_epoch_paired_development_recursion_v2'
spec['parent_recursive_spec_sha256'] = old_spec_sha
spec['binding_revision_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
spec['binding_revision_script_sha256'] = sha(Path(__file__))
spec['runner_sha256'] = sha(candidate)
shutil.copyfile(candidate, root / 'run_recursive.py')
(root / 'recursive_spec.json').write_text(json.dumps(spec, indent=2) + '\n')
record = dict(status='prepared_not_executed', old_spec_sha256=old_spec_sha,
    old_runner_sha256=old_runner_sha, new_spec_sha256=sha(root / 'recursive_spec.json'),
    new_runner_sha256=sha(root / 'run_recursive.py'), revision_script_sha256=sha(Path(__file__)),
    changed='Verify existing checkpoint and execution-binding metadata before inference',
    training_changed=False, gate_changed=False, inputs_changed=False, recursive_predictions_created=False)
(root / 'recursive_binding_revision.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
