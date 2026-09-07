"""Exercise the real category transform and reject an unqualified full pipeline."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import torch

B=Path('/root/autodl-tmp')
ROOT=B/'sttrack_m73_paired_lexical_replication_20260907/candidate_evaluation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=B/'m73_full_preparation_20260907.py'
loader=importlib.util.spec_from_file_location('full73_prelaunch_check',str(source))
module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
spec,_,_,_=module.contracts()
assert not (ROOT/'full_helper_check.json').exists()
raw_path=B/'sttrack_m64_category_candidate_20260907/low22_captions/text_bank.pt'
raw_sha=sha(raw_path);raw=torch.load(raw_path,map_location='cpu')
expected=torch.load(ROOT/'low22_category.pt',map_location='cpu')
actual=module.retain_category(raw,spec['text_protocol_sha256'])
assert actual['keys']==expected['keys'] and actual['format']==expected['format']
assert actual['protocol_sha256']==expected['protocol_sha256']
for key in ['tokens','mask','empty']:assert torch.equal(actual[key],expected[key])
assert sha(raw_path)==raw_sha and len(actual['keys'])==303
before=sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
result=subprocess.run(['bash',str(B/'m73_full_evaluation_20260907.sh')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
assert result.returncode!=0 and 'M73 completion audit is not yet available' in result.stdout
assert before==sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*'))
assert not (ROOT/'full_evaluation').exists() and not torch.cuda.is_initialized()
(ROOT/'premature_full_pipeline.log').write_text(result.stdout)
report=dict(status='M73_full_helpers_actual_bank_and_premature_pipeline_checked',checker_sha256=sha(__file__),
    helper_sha256=sha(source),readiness_sha256=sha(ROOT/'full_preparation_readiness.json'),raw_bank_sha256=raw_sha,
    actual_existing_low22_bank_cases=303,category_transform_matches_prepared_M73_bank=True,source_raw_bank_unchanged=True,
    full_pipeline_returncode=result.returncode,expected_gate_reason_matched=True,no_full_evaluation_directory_created=True,
    new_caption_calls=0,new_model_calls=0,new_optimizer_steps=0,cuda_initialized=False,public_evaluation_started=False)
(ROOT/'full_helper_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
