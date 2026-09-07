"""Collect the complete raw/Hann comparison and fixed-head content evidence."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,shutil,tarfile
B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908';C=R/'content_followup';O=R/'completed_comparison_20260908'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not O.exists()
t=read(R/'training_spec.json');d=read(R/'recursive_result.json');a=read(C/'completed_training_audit.json');content=read(C/'result.json')
assert sha(R/'training_spec.json')==read(R/'frozen.json')['training_spec_sha256']
assert a['result_sha256']==sha(R/'recursive_result.json') and a['gates']==d['gates'] and len(d['gates'])==10
assert d['primary_pass']==all(d['gates'].values())
assert content['development_gate_pass']==d['primary_pass'] and content['independent_scalar_recomputation']
assert content['source_sha256']==read(C/'spec.json')['source_sha256']==sha(B/'m78_content_followup_20260908.py')
assert content['descriptive_criteria_pass']==all(v for g in content['descriptive_criteria'].values() for v in g.values())
for n in ['controller.exit','training_category.exit','training_empty.exit','category_recursive.exit','empty_recursive.exit','recursive_analysis.exit']:
    assert (R/n).read_text().strip()=='0'
for n in ['controller.exit','training_audit.exit','prefix.exit','empty.exit','swapped.exit','content_analysis.exit']:
    assert (C/n).read_text().strip()=='0'
for arm in ['category','empty']:
    train=read(R/'training'/arm/'result.json')
    assert train['sequences']==130 and train['total_track_calls']==186694 and train['optimizer_steps']==5798
    assert sha(R/'training'/arm/'final.pth')==train['final_checkpoint_sha256']==a['training'][arm]['head_sha256']
assert content['head_sha256']==a['training']['category']['head_sha256']
assert content['aggregates']['category']==d['aggregates']['category']
assert sha(Path(t['parent_recursive_result_path']))==t['parent_recursive_result_sha256']
assert sha(Path(t['hann_recursive_result_path']))==t['hann_recursive_result_sha256']
previous=read(t['parent_recursive_result_path']);hann=read(t['hann_recursive_result_path'])
assert previous['aggregates']['native']==hann['aggregates']['native']==d['aggregates']['native']
O.mkdir()
models=[('native','visual',d,'native'),('M73','category',previous,'category'),('M73','empty_trained',previous,'empty'),
    ('M77_Hann','category',hann,'category'),('M77_Hann','empty_trained',hann,'empty'),
    ('M78_raw','category',d,'category'),('M78_raw','empty_trained',d,'empty'),
    ('M78_category_head','empty_content',content,'empty'),('M78_category_head','swapped_content',content,'swapped')]
out=io.StringIO(newline='');w=csv.writer(out);w.writerow(['model','condition','mean_iou','macro_mean_iou','low_iou_frames','H10','valid_frames'])
for name,condition,data,key in models:
    v=data['aggregates'][key];w.writerow([name,condition,v['mean_iou'],v['macro_sequence_mean_iou'],v['low_iou_frames'],v['failure_episodes'],v['valid_frames']])
(O/'aggregate_comparison.csv').write_bytes(out.getvalue().encode())
out=io.StringIO(newline='');w=csv.writer(out);w.writerow(['sequence','model','condition','mean_iou','low_iou_frames','H10','valid_frames'])
for seq in d['per_sequence']['category']:
    for name,condition,data,key in models:
        v=data['per_sequence'][key][seq];w.writerow([seq,name,condition,v['mean_iou'],v['low_iou_frames'],v['failure_episodes'],v['valid_frames']])
(O/'per_sequence_comparison.csv').write_bytes(out.getvalue().encode())
keys=['mean_iou','macro_sequence_mean_iou','low_iou_frames','failure_episodes']
deltas={}
for arm in ['category','empty']:
    for label,reference in [('M73_no_competition',previous),('M77_Hann_competition',hann)]:
        deltas[arm+'_minus_'+label]={k:d['aggregates'][arm][k]-reference['aggregates'][arm][k] for k in keys}
unique_harms={}
for row in a['strict_category_H10_reference_correct_every_frame']:
    key=(row['sequence'],row['start'],row['end_exclusive']);unique_harms[key]=row['frames']
summary=dict(status='completed_M78_paired_training_development_and_fixed_head_content',observed_utc=datetime.now(timezone.utc).isoformat(),
    collector_sha256=sha(__file__),training_spec_sha256=sha(R/'training_spec.json'),recursive_result_sha256=sha(R/'recursive_result.json'),
    audit_sha256=sha(C/'completed_training_audit.json'),content_result_sha256=sha(C/'result.json'),
    development_passes=sum(d['gates'].values()),development_conditions=10,development_pass=d['primary_pass'],
    content_passes=sum(v for g in content['descriptive_criteria'].values() for v in g.values()),content_conditions=8,content_pass=content['descriptive_criteria_pass'],
    paired_aggregates=d['aggregates'],same_head_content_aggregates=content['aggregates'],same_head_template_writes=content['reconstructed_template_writes'],
    deltas=deltas,unique_development_harm_intervals=len(unique_harms),unique_development_harm_frames=sum(unique_harms.values()),
    head_sha256={arm:a['training'][arm]['head_sha256'] for arm in ['category','empty']},seed=2027,additional_seeds=[],
    source_references={'M73':dict(path=t['parent_recursive_result_path'],sha256=t['parent_recursive_result_sha256']),
        'M77':dict(path=t['hann_recursive_result_path'],sha256=t['hann_recursive_result_sha256'])},
    public_evaluation_started=(R/'candidate_evaluation/bundle.json').exists(),independent_model_review_pass=False,goal_achieved=False,
    interpretation_limits=['Same-seed budget-matched training interventions, not random-seed replication.',
        'Each trained policy visits its own states; do not attribute independent-head differences to one word.',
        'Hard-negative eligibility is fixed but raw versus Hann ordering may select different negatives.',
        'Fixed-head content tests do not retroactively override failed development gates.',
        'Reused DepthTrack Train development22, not official three-dataset results.'])
write(O/'comparison.json',summary)
files={n:O/n for n in ['comparison.json','aggregate_comparison.csv','per_sequence_comparison.csv']}
files['collect_m78_completed_20260908.py']=Path(__file__)
for n in ['recursive_result.json','category_recursive_receipt.json','empty_recursive_receipt.json','controller.exit','training_category.exit','training_empty.exit','category_recursive.exit','empty_recursive.exit','recursive_analysis.exit']:
    files[n]=R/n
for n in ['completed_training_audit.json','activation.json','result.json','controller.exit','training_audit.exit','prefix.exit','empty.exit','swapped.exit','content_analysis.exit']:
    files['content__'+n]=C/n
for arm in ['category','empty']:
    for n in ['result.json','sequence_log.jsonl']:files['training__'+arm+'__'+n]=R/'training'/arm/n
for arm in ['prefix','empty','swapped']:files['content__'+arm+'__receipt.json']=C/arm/'receipt.json'
published=O/'published';published.mkdir()
manifest=[]
for n,p in files.items():
    q=published/n;shutil.copyfile(p,q);manifest.append(dict(path=n,bytes=q.stat().st_size,sha256=sha(q)))
write(published/'manifest.json',manifest)
archive=O/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(published.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
print(json.dumps(dict(summary=summary,archive_sha256=sha(archive),files=len(files)),indent=2))
