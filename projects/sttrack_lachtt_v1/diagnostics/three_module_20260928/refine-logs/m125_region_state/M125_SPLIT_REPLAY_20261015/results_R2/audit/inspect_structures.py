from pathlib import Path
import json
BASE=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M125_R2_RESULT_INPUTS_20261015')
for prefix in ['006','016','026','036']:
    p=next(BASE.glob(prefix+'*'))
    v=json.loads(p.read_text())
    print(p.name)
    print(json.dumps({k:v[k] for k in ['plan','case','selected_best']},indent=2))
    if prefix=='006':
        print('BUNDLE',json.dumps(v['bundle'],indent=2))
for prefix in ['007','017','027','037']:
    p=next(BASE.glob(prefix+'*'))
    v=json.loads(p.read_text())
    print(p.name,'ACTION FIRST/LAST',json.dumps([v['actions'][0],v['actions'][-1]]))
    for s in v['sampled_candidates']:
        print('SNAPSHOT',json.dumps({k:v for k,v in s.items() if not isinstance(v,list) or len(v)<5}))
for prefix in ['008','018','028','038']:
    p=next(BASE.glob(prefix+'*'))
    v=json.loads(p.read_text())
    print(p.name)
    print(json.dumps([{k:v for k,v in d.items() if not isinstance(v,list)} for d in v['diagnostics']],indent=2))
for p in sorted(BASE.glob('*.log')):
    print(p.name,p.read_text())

