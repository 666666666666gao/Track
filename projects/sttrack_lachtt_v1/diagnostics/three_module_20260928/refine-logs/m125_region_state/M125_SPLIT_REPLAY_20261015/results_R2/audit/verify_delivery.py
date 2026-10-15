from pathlib import Path
import hashlib
import json

out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inventory=json.loads((out/'input_inventory.json').read_bytes())
report=json.loads((out/'EXPERIMENT_AUDIT.json').read_bytes())
det=json.loads((out/'deterministic_checks.json').read_bytes())
trace=Path(report['trace_path'])
assert all(sha(r['path'])==r['sha256'] for r in inventory)
assert det['failed']==0 and det['passed']==3798
assert report['review_independence']=='same-family' and report['acceptance_status']=='provisional'
assert report['reviewer_model_actual']=='UNATTESTED' and report['verdict']=='WARN'
assert (trace/'001-result-audit.response.md').read_bytes()==(out/'EXPERIMENT_AUDIT.md').read_bytes()
assert not (out.parent/'M125_R2_RESULT_INPUTS_20261015/__pycache__').exists()
artifacts=[p for p in out.rglob('*') if p.is_file() and p.name not in ['DELIVERY_MANIFEST.json','delivery_verification.json']]
manifest=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(artifacts)]
(out/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
result=dict(status='PASS',audited_inputs=len(inventory),inputs_unchanged=True,deterministic_checks=det['passed'],failed=0,trace_matches_report=True,
    input_directory_cache_absent=True,audit_json_sha256=sha(out/'EXPERIMENT_AUDIT.json'),audit_md_sha256=sha(out/'EXPERIMENT_AUDIT.md'),
    deterministic_script_sha256=sha(out/'deterministic_checks.py'),artifact_count=len(manifest))
(out/'delivery_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
