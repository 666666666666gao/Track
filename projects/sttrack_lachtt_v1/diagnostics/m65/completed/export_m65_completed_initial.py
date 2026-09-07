from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile

ROOT=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'recursive_result.json')=='0101fa8185855044dda7462df2f9606b6e58c88248e8a7f1cf87497fb055b150'
assert sha(ROOT/'completed_evidence_audit.json')=='26a79a424ca8b0dec01ff3c12f8cda6653b4a156c12e5c3b297112f52cf69f88'
for name in ['training_control','training_null','control_recursive','null_recursive','recursive_analysis','controller','completed_evidence_audit']:
    assert (ROOT/(name+'.exit')).read_text().strip()=='0'
r=json.loads((ROOT/'recursive_result.json').read_text())
a=json.loads((ROOT/'completed_evidence_audit.json').read_text())
assert not r['primary_pass'] and not a['paired_development_gate_pass']
assert r['broken_control_success_sequences']==['glass03_indoor']
content=ROOT/'content_counterfactuals'
assert not (content/'prefix_result.json').exists() and not (content/'recursive').exists()
out=ROOT/'completed_publication_evidence';out.mkdir()
files=['recursive_result.json','completed_evidence_audit.json','completed_evidence_audit.log','completed_evidence_audit.exit',
       'control_recursive_receipt.json','null_recursive_receipt.json','control_recursive.log','null_recursive.log',
       'control_recursive.exit','null_recursive.exit','recursive_analysis.log','recursive_analysis.exit','controller.exit',
       'training_control.exit','training_null.exit','training_complete_development_running_check.json']
for name in files:shutil.copyfile(ROOT/name,out/name)
for arm in ['control','null']:
    for name in ['result.json','sequence_log.jsonl']:
        shutil.copyfile(ROOT/'training'/arm/name,out/('training_'+arm+'_'+name))
    shutil.copyfile(ROOT/('training_'+arm+'.log'),out/('training_'+arm+'.log'))
lines=['sequence,valid_frames,native_mean_iou,control_mean_iou,null_mean_iou,native_low,control_low,null_low,native_H10,control_H10,null_H10']
for name,v in r['per_sequence']['native'].items():
    xs=[r['per_sequence'][k][name] for k in ['native','control','null']]
    lines.append(','.join([name,str(v['valid_frames'])]+[str(x['mean_iou']) for x in xs]+[str(x['low_iou_frames']) for x in xs]+[str(x['failure_episodes']) for x in xs]))
(out/'per_sequence.csv').write_text('\n'.join(lines)+'\n')
receipt=dict(status='complete_evidence_export',observed_utc=datetime.now(timezone.utc).isoformat(),
    exporter_sha256=sha(__file__),result_sha256=sha(ROOT/'recursive_result.json'),audit_sha256=sha(ROOT/'completed_evidence_audit.json'),
    template_writes={k:sum(a['families'][k]['reconstructed_template_writes'].values()) for k in ['control','null']},
    original_content_stage_executed=False,public_evaluation_started=False,new_training_calls=0,new_tracking_calls=0,
    independent_model_review_pass=False)
(out/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(__file__,out/'export_m65_completed.py')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'completed_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,receipt=receipt),indent=2))
