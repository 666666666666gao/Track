"""CPU-only real-bank transform and premature full-run rejection checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,subprocess
import torch

B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';E=R/'candidate_evaluation'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=B/'m78_full_preparation_20260908.py'
loader=importlib.util.spec_from_file_location('m78_prepared_full_check',str(source))
helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
spec,_,_,_=helper.contracts()
assert spec['candidate_seed']==2027 and spec['additional_seeds']==[]
assert spec['training_spec_sha256']=='57cdd314efd5359fa2e16be4f364c5568f1ca498b41df718517173610fe4d865'
assert not (R/'training/category/final.pth').exists() and not (R/'recursive_result.json').exists()
assert not (R/'content_followup/activation.json').exists()
assert not (E/'evaluation_preparation_check.json').exists()
raw_path=B/'sttrack_m64_category_candidate_20260907/low22_captions/text_bank.pt'
raw_digest=sha(raw_path);raw=torch.load(raw_path,map_location='cpu')
expected=torch.load(E/'low22_category.pt',map_location='cpu')
actual=helper.retain_category(raw,spec['text_protocol_sha256'])
assert actual['keys']==expected['keys'] and len(actual['keys'])==303
assert actual['format']==expected['format'] and actual['protocol_sha256']==expected['protocol_sha256']
for name in ['tokens','mask','empty']:assert torch.equal(actual[name],expected[name])
assert sha(raw_path)==raw_digest
before=sorted(str(p.relative_to(E)) for p in E.rglob('*'))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
result=subprocess.run(['bash',str(B/'m78_full_evaluation_20260908.sh')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
assert result.returncode!=0 and 'FileNotFoundError' in result.stdout and str(R/'content_followup/activation.json') in result.stdout
assert before==sorted(str(p.relative_to(E)) for p in E.rglob('*'))
assert not (E/'full_evaluation').exists() and not (E/'bundle.json').exists() and not torch.cuda.is_initialized()
(E/'premature_full_pipeline.log').write_text(result.stdout)
proof=dict(status='M78_CPU_real_category_transform_and_premature_full_pipeline_rejection_verified',observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__),helper_sha256=sha(source),readiness_sha256=sha(E/'full_preparation_readiness.json'),entry_spec_sha256=sha(E/'spec.json'),
    existing_low22_observations_checked=303,raw_bank_sha256=raw_digest,source_raw_bank_unchanged=True,category_transform_exact=True,
    full_pipeline_returncode=result.returncode,expected_missing_activation_matched=True,no_evaluation_files_created=True,
    seed=2027,additional_seeds=[],new_caption_calls=0,new_tracking_calls=0,new_optimizer_steps=0,cuda_initialized=False,
    actual_GPU_entry_parity_checked=False,public_evaluation_started=False,independent_model_review_pass=False)
(E/'evaluation_preparation_check.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps(proof))
