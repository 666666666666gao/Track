from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
A=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
root=Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING\实验诊断\M124E5_CDTB原始输入_20261013')
manifest=json.loads((root/'PRIMARY_INPUT_MANIFEST.json').read_bytes())
items=manifest['files']
def read(origin):
 p=Path(items[origin]['local_path']);data=p.read_bytes()
 assert hashlib.sha256(data).hexdigest()==items[origin]['sha256']
 return data
base='/root/autodl-tmp/m124_ABC_joint_epoch20_20261011/epochs/epoch-05/evaluation/cdtb'
plan=json.loads(read(base+'/plan.json'));cases=json.loads(read(plan['cases_path']))
receipt=json.loads(read(base+'/predictions/receipt.json'))
def rows(origin,columns):
 data=read(origin).decode('utf-8-sig')
 values=[tuple(map(float,s.replace(',',' ').split())) for s in data.splitlines() if s.strip()]
 assert all(len(r)==columns for r in values)
 return values
checks=[]
for case,proof in zip(cases,receipt['sequences']):
 name=case['sequence'];assert proof['sequence']==name
 bbox=rows(base+'/predictions/'+name+'.txt',4)
 confidence=rows(base+'/predictions/'+name+'_all_scores.txt',1)
 gt=rows(plan['dataset_root'].rstrip('/')+'/'+name+'/groundtruth.txt',4)
 assert len(bbox)==len(confidence)==len(gt)==case['frames']
 assert all(all(math.isfinite(x) for x in row) and row[2]>=0 and row[3]>=0 for row in bbox)
 assert all(math.isfinite(row[0]) for row in confidence)
 valid=sum(all(math.isfinite(x) for x in row) and row[2]>0 and row[3]>0 for row in gt)
 checks.append(dict(sequence=name,rows=len(gt),valid_GT_rows=valid,invalid_GT_rows=len(gt)-valid,initial_score=confidence[0][0]))
assert len(checks)==80 and sum(x['rows'] for x in checks)==101956
assert sum(x['valid_GT_rows'] for x in checks)==91300
result=dict(status='actual_local_primary_numeric_rows_checked',at=datetime.now(timezone.utc).isoformat(),numeric_files_checked=240,sequences=80,rows=101956,valid_GT_rows=91300,per_sequence=checks,new_NN_calls=0,new_GPU_queries=0,new_SSH_calls=0,scope='All raw GT/bbox/score rows parsed and hashed locally; no overlaps, PR curves, crop replay, hidden-state or historical execution attestation.')
(A/'M124_epoch05_CDTB_local_primary_numeric_row_checks_20261013.json').write_bytes((json.dumps(result,indent=2)+chr(10)).encode())
print(json.dumps({k:v for k,v in result.items() if k!='per_sequence'},indent=2));print(datetime.now().astimezone().isoformat())