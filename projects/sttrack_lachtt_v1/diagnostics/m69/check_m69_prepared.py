from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,subprocess,sys
import torch

BASE=Path('/root/autodl-tmp');OUT=BASE/'sttrack_m69_m65_content_diagnostic_20260907'
OLD=BASE/'sttrack_m65_category_null_support_20260907'
script=BASE/'m69_m65_content_diagnostic_20260907.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
s=read(OUT/'spec.json')
assert sha(script)==s['source_sha256']
original=BASE/'m65_content_counterfactuals_20260907.py'
assert sha(original)==s['original_content_source_sha256']
old_ast,new_ast=ast.parse(original.read_text()),ast.parse(script.read_text())
for name in ['track','parents','sha','read','write']:
    a=next(n for n in old_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
    b=next(n for n in new_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
for action in ['check','eligible']:
    result=subprocess.run([sys.executable,str(script),action],text=True,capture_output=True)
    (OUT/(action+'_check.log')).write_text(result.stdout+result.stderr)
    (OUT/(action+'_check.exit')).write_text(str(result.returncode)+'\n')
    assert result.returncode==0,result.stderr
subprocess.run(['bash','-n',str(OUT/'run_controls.sh')],check=True)
compile(script.read_text(),str(script),'exec')
compile(Path(__file__).read_text(),__file__,'exec')
old_spec=read(OLD/'content_counterfactuals/spec.json')
assert old_spec['head_selection']=='Only M65 null final after completed evidence audit and every original development gate passes.'
for name in ['empty','swapped']:
    assert sha(s['banks'][name]['path'])==old_spec['banks'][name]['sha256']
banks={k:torch.load(v['path'],map_location='cpu') for k,v in s['banks'].items()}
category=banks['category']
assert category['tokens'].shape==(22,5,768)
for name,b in banks.items():
    assert b['sequences']==category['sequences']
    assert torch.equal(b['mask'],category['mask']) and torch.equal(b['empty'],category['empty'])
    assert torch.equal(b['tokens'][~b['mask']],category['tokens'][~category['mask']])
    if name=='empty':assert torch.equal(b['tokens'][b['mask']],b['empty'].expand_as(b['tokens'][b['mask']]))
    elif name=='swapped':
        assert torch.equal(b['tokens'][:,1:],category['tokens'][:,1:])
        assert bool((b['tokens'][:,0]!=category['tokens'][:,0]).any(1).all())
for n in ['prefix','empty','swapped']:assert not (OUT/n).exists()
assert not (OUT/'result.json').exists()
audit=read(OLD/'completed_evidence_audit.json')
assert sha(OLD/'completed_evidence_audit.json')==s['completed_M65_audit_sha256']
assert not audit['paired_development_gate_pass'] and not audit['content_counterfactuals_allowed']
assert not s['low22_candidate_preparation_allowed'] and not s['public_full_evaluation_allowed']
assert not torch.cuda.is_initialized()
r=dict(status='M69_diagnostic_only_preparation_CPU_checked',observed_utc=datetime.now(timezone.utc).isoformat(),
    checker_sha256=sha(__file__),source_sha256=sha(script),spec_sha256=sha(OUT/'spec.json'),queue_sha256=sha(OUT/'run_controls.sh'),
    preparation_sha256=sha(OUT/'preparation_result.json'),fixed_head_sha256=s['fixed_null_head_sha256'],
    original_tracking_function_AST_identical=True,original_parent_integrity_checks_AST_identical=True,
    original_M65_source_and_gate_unchanged=True,banks_byte_identical_to_original_frozen_M65_controls=True,
    slots=5,sequences=22,changed_category_vectors=22,masks_and_padding_exact=True,
    queue_syntax_valid=True,diagnostic_artifact_eligibility_pass=True,GPU_inference_parity_verified=False,
    CUDA_initialized=False,learned_head_loaded=False,new_tracking_calls=0,new_GT_opened=False,
    low22_candidate_preparation_allowed=False,full_three_dataset_evaluation_allowed=False,independent_model_review_pass=False)
(OUT/'preparation_check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
