from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess, tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
CONTENT = ROOT / 'content_counterfactuals'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
assert sha(ROOT / 'recursive_spec.json') == 'd4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
assert sha(BASE / 'audit_m67_completed_20260907.py') == '1773887deb3be27aa11a4409aa77bd7581f32b8d891ecab2a03454d277e769bf'
assert sha(BASE / 'm67_content_counterfactuals_20260907.py') == '40b765b22c398bc18ed4ac3f5ef33edc7b9611cf1c8d3172062f518d435f8285'
assert (ROOT / 'completion_auditor_reference.exit').read_text().strip() == '0'
assert (ROOT / 'content_preparation.exit').read_text().strip() == '0'
subprocess.run(['/root/autodl-tmp/envs/sttrack/bin/python', str(BASE / 'm67_content_counterfactuals_20260907.py'), 'check'], check=True)
subprocess.run(['bash', '-n', str(CONTENT / 'run_controls.sh')], check=True)
assert not (ROOT / 'completed_evidence_audit.json').exists()
assert not (CONTENT / 'result.json').exists()
for name in ['prefix', 'empty', 'swapped']:
    assert not (CONTENT / name).exists()
receipt = dict(status='completion_tools_prepared_not_executed_on_M67_final', observed_utc=datetime.now(timezone.utc).isoformat(),
    auditor_sha256=sha(BASE / 'audit_m67_completed_20260907.py'),
    reference_receipt_sha256=sha(ROOT / 'completion_auditor_reference.json'),
    content_source_sha256=sha(BASE / 'm67_content_counterfactuals_20260907.py'),
    content_spec_sha256=sha(CONTENT / 'spec.json'), content_preparation_sha256=sha(CONTENT / 'preparation_result.json'),
    content_source_inputs_check=True, conditional_queue_bash_syntax_check=True,
    M67_completion_audited=False, content_tracking_started=False, training_or_inference_source_modified=False,
    new_training_steps=0, new_tracking_calls=0, independent_model_review_pass=False)
(ROOT / 'completion_tools_check.json').write_text(json.dumps(receipt, indent=2) + '\n')
out = ROOT / 'completion_tools_publication_evidence'
out.mkdir()
for n in ['completion_auditor_reference.json', 'completion_auditor_reference.log', 'completion_auditor_reference.exit',
          'content_preparation.log', 'content_preparation.exit', 'completion_tools_check.json']:
    shutil.copyfile(ROOT / n, out / n)
for n in ['audit_m67_completed_20260907.py', 'm67_content_counterfactuals_20260907.py',
          'prepare_m67_auditor_20260907.py', 'prepare_m67_content_20260907.py']:
    shutil.copyfile(BASE / n, out / n)
for n in ['spec.json', 'preparation_result.json', 'run_controls.sh']:
    shutil.copyfile(CONTENT / n, out / ('content_' + n))
shutil.copyfile(__file__, out / 'export_m67_completion_tools.py')
(out / 'evidence_manifest.json').write_text(json.dumps([
    dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'completion_tools_evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for p in sorted(out.iterdir()):
        t.add(p, arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, receipt=receipt), indent=2))
