"""Validate new scalar audit against already sealed M78 data, never M82 outputs."""
from pathlib import Path
import ast,hashlib,json
from datetime import datetime,timezone
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908'
source=B/'audit_m82_completed_20260909.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(R/'recursive_result.json')=='37e01fd17c9b747dc5c5d7654ef66fc60b8ba00e799d0a59fc71e7dfba82e16a'
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['equal','statistics']]
assert len(nodes)==2
namespace={}
exec('import math',namespace)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
statistics=namespace['statistics'];equal=namespace['equal']
spec=json.loads((R/'recursive_spec.json').read_text());train=json.loads((R/'training_spec.json').read_text());result=json.loads((R/'recursive_result.json').read_text())
receipts={a:json.loads((R/(a+'_recursive_receipt.json')).read_text()) for a in ['category','empty']}
for arm,receipt in receipts.items():
    assert (R/(arm+'_recursive.exit')).read_text().strip()=='0'
    assert receipt['status']=='complete' and len(receipt['sequences'])==22
    for row in receipt['sequences']:assert sha(R/'recursive'/arm/(row['sequence']+'.json'))==row['sha256']
positions=0
for case in spec['cases']:
    p=Path(train['dataset_root'])/case['sequence']/'groundtruth.txt';assert sha(p)==case['gt_sha256']
    gt=[[float(v) for v in line.split(',')] for line in p.read_text().splitlines() if line.strip()]
    for arm in receipts:
        rows=json.loads((R/'recursive'/arm/(case['sequence']+'.json')).read_text())['rows']
        stats=statistics(rows,gt);positions+=len(rows)
        for key,value in stats.items():equal(value,result['per_sequence'][arm][case['sequence']][key])
record=dict(status='scalar_auditor_verified_against_sealed_M78',observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(source),validation_source_sha256=sha(__file__),reference_result_sha256=sha(R/'recursive_result.json'),families=2,sequences=44,positions=positions,current_M82_predictions_or_development_GT_read=False,current_M82_performance_claim=False)
out=B/'sttrack_m82_native_preservation_20260909'/'auditor_reference_validation.json'
assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
