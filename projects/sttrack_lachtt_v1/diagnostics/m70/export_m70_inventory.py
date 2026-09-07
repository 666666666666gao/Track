from pathlib import Path
import csv,hashlib,json,shutil,tarfile

BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m70_recovery_window_inventory_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert (ROOT/'run.exit').read_text().strip()=='0'
r=json.loads((ROOT/'result.json').read_text())
assert r['source_sha256']==sha(BASE/'m70_recovery_window_inventory_20260907.py')
assert r['spec_sha256']==sha(ROOT/'spec.json') and r['events_sha256']==sha(ROOT/'geometry_events.json')
assert r['replay_cases_sha256']==sha(ROOT/'events_for_replay.json')
assert r['all68_Null_H10_included'] and r['new_tracking_calls']==0 and r['geometry_is_not_recognition']
events=json.loads((ROOT/'geometry_events.json').read_text())['events']
H10=[e for e in events if 'H10' in e['tags']]
outside=[e for e in H10 if not e['arms']['null']['local']['centre_inside']]
extra=dict(H10_after_at_least10_invalid_frames=sum('valid_after_invalid' in e['tags'] for e in H10),
    outside_H10_after_at_least10_invalid_frames=sum('valid_after_invalid' in e['tags'] for e in outside),
    grid_centre_misses=[dict(sequence=e['sequence'],frame=e['frame'],GT_centre_inside_actual_image=e['GT_vs_image']['centre_inside']) for e in events if not e['null_scale_grid']['centre_covered']],
    observations='Valid GT can extend beyond the actual image. Invalid GT is not an absence label. Geometry is not candidate recognition.')
assert extra['H10_after_at_least10_invalid_frames']==43 and extra['outside_H10_after_at_least10_invalid_frames']==40
out=ROOT/'publication_evidence';out.mkdir()
for name in ['spec.json','result.json','events_for_replay.json','run.log','run.exit']:shutil.copyfile(ROOT/name,out/name)
shutil.copyfile(BASE/'m70_recovery_window_inventory_20260907.py',out/'m70_recovery_window_inventory.py')
shutil.copyfile(__file__,out/'export_m70_inventory.py')
(out/'annotation_boundary_summary.json').write_text(json.dumps(extra,indent=2)+'\n')
with (out/'coverage_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['stratum','arm','events','sequences','local_centre_inside','local_full_box_inside','factor7_centre_inside'])
    for tag,v in r['summary'].items():
        for arm,x in v['arms'].items():w.writerow([tag,arm,v['events'],v['sequences'],x['local_centre_inside'],x['local_full_box_inside'],x['wide_centre_inside']])
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
archive=ROOT/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,result_sha256=sha(ROOT/'result.json')),indent=2))
