from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, subprocess, tarfile

BASE = Path('/root/autodl-tmp')
ROOT = BASE / 'sttrack_m68_reported_confidence_20260907'
PARENT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
SOURCE = BASE / 'm68_reported_confidence_20260907.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE) == '7faf987f68c26739cd3666428e1c73e42704a4d208ffe29572863ca7b808f4d9'
assert sha(ROOT / 'spec.json') == 'e2106dd1604d627c02d0589b91224c4901bfbca2906ffbbf241a3f380b7cfdf5'
assert not (ROOT / 'control').exists() and not (ROOT / 'support').exists() and not (ROOT / 'result.json').exists()
assert not (PARENT / 'recursive_result.json').exists()
checks = []
for interpreter in [BASE / 'envs/sttrack/bin/python', Path('/root/miniconda3/envs/mplt/bin/python')]:
    result = subprocess.run([str(interpreter), str(SOURCE), 'check'], check=True, capture_output=True, text=True)
    checks.append(dict(interpreter=str(interpreter), action='check', exit_code=result.returncode, output=result.stdout.strip()))
    subprocess.run([str(interpreter), '-m', 'py_compile', str(SOURCE)], check=True)
subprocess.run(['bash', '-n', str(ROOT / 'run_diagnostic.sh')], check=True)
metric_check = subprocess.run(['/root/miniconda3/envs/mplt/bin/python', '-c',
    "import importlib.util; p='/home/SRTrack_RGBD_L/lib/test/analysis/depthtrack_pr.py'; s=importlib.util.spec_from_file_location('m68_pr_check',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert callable(m.evaluate_depthtrack_results); print('FROZEN_PR_MODULE_IMPORT_OK_NO_EVALUATION')"],
    check=True, capture_output=True, text=True)
receipt = dict(status='M68_source_input_and_environment_preflight_pass_no_replay', observed_utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(SOURCE), spec_sha256=sha(ROOT / 'spec.json'), queue_sha256=sha(ROOT / 'run_diagnostic.sh'),
    checks=checks, metric_import_output=metric_check.stdout.strip(),
    py_compile_both_environments=True, queue_syntax_check=True,
    actual_hook_replay_parity_checked=False, final_heads_loaded=False, new_GT_opened=False, new_tracking_calls=0,
    training_source_modified=False, public_evaluation_allowed=False, independent_model_review_pass=False)
(ROOT / 'preflight_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
progress = PARENT / 'progress_1120_check.json'
item = json.loads(progress.read_text())
assert sha(PARENT / 'training_spec.json') == '2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
item['training_spec_sha256'] = sha(PARENT / 'training_spec.json')
item['spec_verified_utc'] = datetime.now(timezone.utc).isoformat()
item['filename_note'] = 'Filename is only a label; the actual process observation was 2026-09-07T03:10:01Z.'
progress.write_text(json.dumps(item, indent=2) + '\n')
out = ROOT / 'publication_evidence'; out.mkdir()
for name in ['spec.json', 'preparation_result.json', 'preflight_receipt.json', 'run_diagnostic.sh']:
    shutil.copyfile(ROOT / name, out / name)
shutil.copyfile(SOURCE, out / SOURCE.name)
shutil.copyfile(__file__, out / 'export_m68_prepared.py')
shutil.copyfile(progress, out / 'M67_progress_1110_CST.json')
(out / 'evidence_manifest.json').write_text(json.dumps([
    dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'prepared_evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for p in sorted(out.iterdir()):
        t.add(p, arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size, receipt=receipt), indent=2))
