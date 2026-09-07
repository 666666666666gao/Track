from pathlib import Path
import hashlib, json, shutil, tarfile

BASE = Path('/root/autodl-tmp')
PARENT = BASE / 'sttrack_m67_supervised_semantic_support_20260907'
ROOT = PARENT / 'evaluation_interface'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (PARENT / 'evaluation_interface_preparation.exit').read_text().strip() == '0'
assert sha(ROOT / 'spec.json') == '323620eab82b163a3a9e6de54ea8e17da5dbc0cf7336b3717ef443082b59e07b'
spec = json.loads((ROOT / 'spec.json').read_text())
check = json.loads((ROOT / 'cpu_contract_result.json').read_text())
assert check['actual_Train_initializations_checked'] == check['exact_token_mask_bbox_matches'] == 152
assert not check['cuda_initialized'] and not check['final_bundle_created']
for name, digest in spec['interface_sha256'].items():
    assert sha(ROOT / name) == digest
out = ROOT / 'publication_evidence'; out.mkdir()
for name in list(spec['interface_sha256']) + ['runtime_delta.patch', 'text_protocol.json', 'spec.json', 'cpu_contract_result.json']:
    shutil.copyfile(ROOT / name, out / name)
for name in ['evaluation_interface_preparation.log', 'evaluation_interface_preparation.exit']:
    shutil.copyfile(PARENT / name, out / name)
shutil.copyfile(BASE / 'prepare_m67_evaluation_interface_20260907.py', out / 'prepare_m67_evaluation_interface.py')
shutil.copyfile(__file__, out / 'export_m67_evaluation_interface.py')
(out / 'evidence_manifest.json').write_text(json.dumps([
    dict(path=p.name, bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())], indent=2) + '\n')
archive = ROOT / 'evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for p in sorted(out.iterdir()):
        t.add(p, arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
    runtime_sha256=spec['interface_sha256']['semantic_runtime.py'], text_protocol_sha256=spec['text_protocol_sha256'],
    fixture_sha256=spec['train_fixture_sha256'], check_sha256=sha(ROOT / 'cpu_contract_result.json')), indent=2))
