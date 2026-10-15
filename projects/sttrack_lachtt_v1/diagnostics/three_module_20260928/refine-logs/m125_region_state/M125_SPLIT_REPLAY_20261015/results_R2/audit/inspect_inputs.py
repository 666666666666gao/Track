from pathlib import Path
import json, hashlib

BASE = Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
OUT = BASE / 'M125_R2_RESULT_AUDIT_20261015'
INPUT = BASE / 'M125_R2_RESULT_INPUTS_20261015'
manifest = json.loads((INPUT / 'INPUTS.json').read_text(encoding='utf-8-sig'))
inventory = []
for record in manifest:
    p = Path(record['path'])
    raw = p.read_bytes()
    text = raw.decode('utf-8-sig')
    digest = hashlib.sha256(raw).hexdigest()
    row = dict(record, bytes=len(raw), lines=len(text.splitlines()), actual_sha256=digest, matches=digest == record['sha256'])
    if p.suffix == '.json':
        obj = json.loads(text)
        row['structure'] = {k: ('dict:' + str(len(v)) if isinstance(v, dict) else 'list:' + str(len(v)) if isinstance(v, list) else v) for k, v in obj.items()} if isinstance(obj, dict) else 'list:' + str(len(obj))
    inventory.append(row)
(OUT / 'initial_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n', encoding='utf-8')
for row in inventory:
    print(Path(row['path']).name, row['bytes'], row['lines'], row['matches'])
    if 'structure' in row:
        print(json.dumps(row['structure']))
print('ARCHIVE')
archive = json.loads((BASE / 'M125_R2_complete_replay_primary_inputs_retry1_20261015.json').read_text(encoding='utf-8-sig'))
print({k: ('dict:' + str(len(v)) if isinstance(v,dict) else 'list:'+str(len(v)) if isinstance(v,list) else v) for k,v in archive.items()})
print('COVERAGE SUMMARY')
coverage = json.loads((BASE / 'M125_R2_complete_box_coverage_summary_20261015.json').read_text(encoding='utf-8-sig'))
print(json.dumps({k: v for k,v in coverage.items() if k != 'states'}, indent=2))

