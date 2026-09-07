from pathlib import Path
import difflib,hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m69_m65_content_diagnostic_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((ROOT/'preparation_check.json').read_text())
assert r['status']=='M69_diagnostic_only_preparation_CPU_checked'
assert r['source_sha256']==sha(BASE/'m69_m65_content_diagnostic_20260907.py')
assert r['spec_sha256']==sha(ROOT/'spec.json')
assert not r['CUDA_initialized'] and not r['low22_candidate_preparation_allowed']
out=ROOT/'publication_evidence';out.mkdir()
for name in ['spec.json','run_controls.sh','preparation_result.json','preparation_check.json','check_check.log','check_check.exit','eligible_check.log','eligible_check.exit']:
    shutil.copyfile(ROOT/name,out/name)
for source,target in [('m69_m65_content_diagnostic_20260907.py','m69_m65_content_diagnostic.py'),('check_m69_prepared_20260907.py','check_m69_prepared.py')]:
    shutil.copyfile(BASE/source,out/target)
shutil.copyfile(__file__,out/'export_m69_prepared.py')
original=(BASE/'m65_content_counterfactuals_20260907.py').read_text()
current=(BASE/'m69_m65_content_diagnostic_20260907.py').read_text()
(out/'diagnostic_scope_delta.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),current.splitlines(True),fromfile='M65_original_gated_content',tofile='M69_diagnostic_only_content')))
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,check_sha256=sha(ROOT/'preparation_check.json')),indent=2))
