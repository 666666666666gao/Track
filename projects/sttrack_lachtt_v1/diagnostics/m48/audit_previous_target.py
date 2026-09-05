"""Post-protocol fitting diagnosis; it does not modify the fixed M48 policy."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
audit_path = root / 'native_continuity_audit.json'
fit_path = root.parent / 'm45' / 'geometry_result.json'
assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == '969a982bf7633b31d25569b2cdd4bcf6e957b45934705fceb808085b3fa24950'
assert hashlib.sha256(fit_path.read_bytes()).hexdigest() == 'e2647ea8d9738632d93f99d0108e638025b1c38894f8560cc9a532eb4f13f39d'
rows = json.loads(audit_path.read_text(encoding='utf-8'))['rows']
targets = {r['key']: r['previous_target'] for r in json.loads(fit_path.read_text(encoding='utf-8'))['fit']['rows']}
groups = dict(vetoed_rescues=[r for r in rows if r['vetoed'] and r['default_iou'] <= .1 and r['proposal_iou'] >= .5],
    retained_rescues=[r for r in rows if r['default_iou'] <= .1 and r['selected_iou'] >= .5])
summary = {name: dict(events=len(values), previous_default_valid=sum(targets[r['key']] == 0 for r in values),
    previous_alternative_valid=sum(0 < targets[r['key']] < 10 for r in values),
    previous_no_valid_candidate=sum(targets[r['key']] == 10 for r in values)) for name, values in groups.items()}
result = dict(status='complete', summary=summary,
    scope='Fitting-only descriptive analysis after M48 policy and recursive protocol were frozen. No margin change or new training.',
    definition='Reuse M45 default-priority previous_target labels:0 means previous default IoU>=.5;1to9 mean another box reaches.5;10 means no valid candidate under the existing annotation rule.',
    interpretation='20of21vetoed rescues have no valid previous default. Continuity to a previous mistake can obstruct recovery. This is an explanatory fitting association, not a deployable GT condition or proof of recursive causation.',
    audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(), fit_result_sha256=hashlib.sha256(fit_path.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    rows={name: [dict(key=r['key'], previous_target=targets[r['key']]) for r in values] for name, values in groups.items()})
(root / 'previous_target_diagnosis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(summary, indent=2))
