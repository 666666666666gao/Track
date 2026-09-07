"""Bounded export of conditional full-evaluation helpers and their real CPU checks."""
from datetime import datetime,timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

B=Path('/root/autodl-tmp');M=B/'sttrack_m73_paired_lexical_replication_20260907';R=M/'candidate_evaluation'
OUT=R/'full_prepared_evidence';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
source=B/'m73_full_preparation_20260907.py'
s=importlib.util.spec_from_file_location('m73_full_export_contract',str(source));helper=importlib.util.module_from_spec(s);s.loader.exec_module(helper)
helper.contracts();proof=read(R/'full_preparation_readiness.json');check=read(R/'full_helper_check.json')
for p,h in proof['full_source_sha256'].items():assert sha(p)==h
assert check['helper_sha256']==sha(source) and check['readiness_sha256']==sha(R/'full_preparation_readiness.json')
assert check['checker_sha256']==sha(B/'check_m73_full_helpers_20260907.py')
assert not (R/'full_evaluation').exists() and not (R/'bundle.json').exists()
jobs=[]
for pid,ticks in [(482884,'4579784392'),(482891,'4579784502'),(482892,'4579784502'),(484228,'4579959586')]:
    p=Path('/proc')/str(pid);fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    assert fields[19]==ticks and fields[0]!='Z'
    jobs.append(dict(pid=pid,start_ticks=ticks,cwd=str((p/'cwd').resolve()),argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]))
assert not (M/'controller.exit').exists() and not (M/'completion_queue/controller.exit').exists()
report=dict(status='M73_full_helpers_verified_original_training_live',observed_utc=datetime.now(timezone.utc).isoformat(),
    jobs=jobs,disk_free_bytes=shutil.disk_usage(str(B)).free,readiness_sha256=sha(R/'full_preparation_readiness.json'),
    CPU_check_sha256=sha(R/'full_helper_check.json'),full_evaluation_directory_created=False,candidate_bundle_created=False,
    new_model_calls=0,new_caption_calls=0,independent_model_review_pass=False,exporter_sha256=sha(__file__))
files={Path(p).name:Path(p) for p in proof['full_source_sha256']}
files['check_m73_full_helpers_20260907.py']=B/'check_m73_full_helpers_20260907.py'
files['export_m73_full_prepared_20260907.py']=Path(__file__)
for n in ['full_preparation_readiness.json','full_helper_check.json','premature_full_pipeline.log']:files[n]=R/n
OUT.mkdir()
for n,p in files.items():(OUT/n).write_bytes(p.read_bytes())
for current,old in [('m73_full_preparation_20260907.py','m64_full_preparation_20260907.py'),('m73_full_vot_20260907.py','m64_full_vot_20260907.py'),('verify_m73_three_datasets_20260907.py','verify_m64_three_datasets_20260907.py')]:
    (OUT/(current+'.diff')).write_text(''.join(difflib.unified_diff((B/old).read_text().splitlines(True),(B/current).read_text().splitlines(True),fromfile='M64/'+old,tofile='M73/'+current)))
(OUT/'running_verification.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir())],indent=2)+'\n')
archive=R/'full_prepared_evidence.tar.gz'
with tarfile.open(str(archive),'w:gz') as t:
    for p in sorted(OUT.iterdir()):t.add(str(p),arcname=p.name)
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(list(OUT.iterdir())),report=report)))
