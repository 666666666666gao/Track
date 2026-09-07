from pathlib import Path
import csv, hashlib, json, shutil, tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'
Q=ROOT/'quarter_prefix_34';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (ROOT/'quarter_prefix_audit.exit').read_text().strip()=='0'
r=json.loads((Q/'result.json').read_text())
assert r['status']=='completed_first34_sampled_training_mechanism_audit'
assert r['source_sha256']==sha(BASE/'m67_quarter_prefix_audit_20260907.py')
assert r['spec_sha256']==sha(Q/'spec.json')
assert r['new_tracking_calls']==r['optimizer_steps_added']==0
assert not r['training_source_modified'] and not r['checkpoint_selected']
for arm, files in r['snapshots'].items():
    assert sha(Q/(arm+'_records.json'))==files['records_sha256']
    assert sha(Q/(arm+'_sampled_trace.json'))==files['trace_sha256']
out=Q/'publication_evidence';out.mkdir()
for name in ['spec.json','result.json']:shutil.copyfile(Q/name,out/name)
for name in ['quarter_prefix_audit.log','quarter_prefix_audit.exit']:shutil.copyfile(ROOT/name,out/name)
shutil.copyfile(BASE/'m67_quarter_prefix_audit_20260907.py',out/'m67_quarter_prefix_audit.py')
shutil.copyfile(__file__,out/'export_m67_quarter_prefix.py')
with (out/'training_bins.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['step_start','step_end_exclusive','samples','centre_inside','centre_outside','invalid','centre_null_median','outside_null_median','paired_background_greater_fraction'])
    for b in r['bins']:
        s=b['statistics'];w.writerow([b['steps_start_inclusive'],b['steps_end_exclusive'],s['samples']]+[s['labels'].get(k,0) for k in ['centre_inside','centre_outside','invalid']]+[s['GT_centre_null_mass']['median'],s['outside_box_mean_null_mass']['median'],s['paired_background_greater_fraction']])
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=Q/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,result_sha256=sha(Q/'result.json')),indent=2))
