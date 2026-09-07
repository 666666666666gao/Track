"""Seal CPU-only M73 evaluation entry preparation; no final checkpoint or results."""
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

B=Path('/root/autodl-tmp');M=B/'sttrack_m73_paired_lexical_replication_20260907'
R=M/'candidate_evaluation';OUT=R/'publication_evidence'
assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
source=B/'m73_candidate_evaluation_20260907.py'
s=importlib.util.spec_from_file_location('m73_candidate_export_binding',str(source));module=importlib.util.module_from_spec(s);s.loader.exec_module(module)
spec=module.checked_preparation()
assert not (R/'bundle.json').exists() and not (R/'entry_parity').exists() and not (R/'low22_run').exists()
assert not (M/'result.json').exists()
qsource=B/'m73_completion_queue_20260907.py'
assert sha(qsource)=='d9704553563e5a26d84306481a8fc13f145d93dd0449dc18860652178033de21'
s=importlib.util.spec_from_file_location('m73_candidate_waiter_verification',str(qsource));q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
qs=q.checked();registered=read(M/'completion_queue/running_identity.json')
assert q.identity(registered['identity']['pid'])==registered['identity']
assert q.identity(qs['parent_identity']['pid'])==qs['parent_identity']
assert not (M/'controller.exit').exists() and not (M/'completion_queue/controller.exit').exists()
retained={}
for name in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']:
    files=list((B/'qwen'/name).glob('*.safetensors'));retained[name]=dict(files=len(files),bytes=sum(p.stat().st_size for p in files))
assert retained['Qwen3_8B']==dict(files=5,bytes=16381516776)
assert retained['Qwen2.5-VL-3B-Instruct']==dict(files=2,bytes=7509337976)
report=dict(status='M73_CPU_candidate_preparation_sealed_training_and_follower_live',observed_utc=datetime.now(timezone.utc).isoformat(),
    spec_sha256=sha(R/'spec.json'),parent_training_identity=qs['parent_identity'],follower_identity=registered['identity'],
    follower_latest=read(M/'completion_queue/latest.json'),disk_free_bytes=shutil.disk_usage(str(B)).free,retained_qwen=retained,
    candidate_bundle_exists=False,actual_entry_parity_run=False,low22_tracking_started=False,new_model_calls=0,
    original_training_and_follower_sources_unchanged=True,exporter_sha256=sha(__file__),independent_model_review_pass=False)
files={name:B/name for name in spec['source_sha256']}
files['export_m73_candidate_prepared_20260907.py']=Path(__file__)
for name in ['spec.json','cpu_preparation_result.json','premature_entry_check.json','premature_entry_check.log']:files[name]=R/name
for name in list(spec['interface_sha256'])+['text_protocol.json']:files['interface__'+name]=R/'interface'/name
OUT.mkdir()
for name,path in files.items():(OUT/name).write_bytes(path.read_bytes())
for name in ['m73_learned_entry_parity_20260907.py','m73_vot_low22_20260907.py','m73_candidate_entry_20260907.sh','m73_candidate_low22_20260907.sh']:
    old=B/name.replace('m73','m67');current=B/name
    (OUT/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),current.read_text().splitlines(True),fromfile='M67/'+old.name,tofile='M73/'+name)))
(OUT/'running_verification.json').write_text(json.dumps(report,indent=2)+'\n')
manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir())]
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'prepared_evidence.tar.gz'
with tarfile.open(str(archive),'w:gz') as t:
    for p in sorted(OUT.iterdir()):t.add(str(p),arcname=p.name)
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(list(OUT.iterdir())),report=report)))
