"""Update the remaining candidate shell dependency before any final model exists."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

B=Path('/root/autodl-tmp'); R=B/'sttrack_m73_paired_lexical_replication_20260907'
A=R/'single_seed_amendment'; C=R/'candidate_evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

assert not (R/'result.json').exists() and not (C/'bundle.json').exists()
entry=B/'m73_candidate_entry_20260907.sh';full=B/'m73_full_preparation_20260907.py'
paths=[entry, C/'spec.json', full,C/'full_preparation_readiness.json',A/'patch_receipt.json',A/'controller/spec.json']
backup=A/'entry_link_previous';backup.mkdir()
for p in paths:
 q=backup/p.relative_to(B);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
text=entry.read_text();old='sttrack_m73_paired_lexical_replication_20260907/completion_queue';assert old in text
entry.write_text(text.replace(old,'sttrack_m73_paired_lexical_replication_20260907/single_seed_amendment/controller'))
subprocess.run(['bash','-n',str(entry)],check=True)
oldspecsha=sha(C/'spec.json');spec=read(C/'spec.json');spec['source_sha256'][entry.name]=sha(entry);write(C/'spec.json',spec)
text=full.read_text();assert oldspecsha in text;full.write_text(text.replace(oldspecsha,sha(C/'spec.json')))
compile(full.read_text(),str(full),'exec');mod('fixed73',full).contracts()
proof=read(C/'full_preparation_readiness.json');proof['candidate_spec_sha256']=sha(C/'spec.json');proof['preparation_source_sha256']=sha(full)
proof['full_source_sha256'][str(full)]=sha(full);write(C/'full_preparation_readiness.json',proof)
receipt=read(A/'patch_receipt.json')
receipt['before_sha256'][str(entry)]=sha(backup/entry.relative_to(B))
for p in paths[:4]:receipt['after_sha256'][str(p)]=sha(p)
receipt['candidate_shell_link_corrected_utc']=datetime.now(timezone.utc).isoformat()
write(A/'patch_receipt.json',receipt)
spec=read(A/'controller/spec.json');spec['patch_receipt_sha256']=sha(A/'patch_receipt.json');write(A/'controller/spec.json',spec)
mod('single73',B/'m73_single_seed_controller_20260907.py').checked()
r=subprocess.run(['bash',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
assert r.returncode!=0 and 'M73 completion audit is not yet available' in (C/'binding.log').read_text()
for name in ['binding.log','binding.exit']:
 p=C/name;q=A/('expected_premature_'+name);assert not q.exists();p.rename(q)
write(A/'entry_link_correction.json',dict(status='candidate_shell_and_dependency_hashes_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
 actual_entry_shell_exit=r.returncode,expected_missing_audit=True,new_model_calls=0,controller_training_pair_changed=False,
 patch_receipt_sha256=sha(A/'patch_receipt.json'),controller_spec_sha256=sha(A/'controller/spec.json')))
print((A/'entry_link_correction.json').read_text())
