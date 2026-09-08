"""Publish sealed M78 results and a separately reviewed factual handoff append."""
from pathlib import Path
import hashlib,json,shutil,tarfile,subprocess
from datetime import datetime,timezone

B=Path('/root/autodl-tmp');R=B/'sttrack_m78_raw_competition_20260908'
O=R/'completed_comparison_20260908';D=R/'completed_publication_20260908_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert not D.exists()
s=read(O/'comparison.json');d=read(R/'recursive_result.json');c=read(R/'content_followup/result.json')
assert s['status']=='completed_M78_paired_training_development_and_fixed_head_content'
assert s['recursive_result_sha256']==sha(R/'recursive_result.json')
assert s['content_result_sha256']==sha(R/'content_followup/result.json')
assert s['development_pass']==d['primary_pass'] and s['content_pass']==c['descriptive_criteria_pass']
assert (R/'controller.exit').read_text().strip()=='0'
assert (R/'content_followup/controller.exit').read_text().strip()=='0'
note=B/'handoff_append_m78_completed_20260908.md'
note_bytes=note.read_bytes()
assert note_bytes.decode('utf-8').startswith('\n\n## 5.154 ')
assert s['head_sha256']['category'].encode() in note_bytes and s['head_sha256']['empty'].encode() in note_bytes
assert s['recursive_result_sha256'].encode() in note_bytes and s['content_result_sha256'].encode() in note_bytes
master=Path('/home/SUTrack_RGBD_L/docs/RGBD_LANGUAGE_TRACKING_PROJECT_MASTER.md')
old=master.read_bytes()
assert sha(master)=='3f49f0ac12c68953910779581fda2422fc1db6b0563ed3a39d78eaf1bc72dba8'
assert b'\n## 5.154 ' not in old
D.mkdir();published=D/'published';published.mkdir()
manifest=read(O/'published/manifest.json')
files={}
for row in manifest:
    p=O/'published'/row['path']
    assert Path(row['path']).name==row['path'] and sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    files[row['path']]=p
files['handoff_append.md']=note
files[Path(__file__).name]=Path(__file__)
files['failed_exporter_preserved_20260908.py']=B/'finalize_m78_completed_20260908.py'
recovery=dict(failed_exporter_sha256=sha(B/'finalize_m78_completed_20260908.py'),failed_archive_sha256=sha(R/'completed_publication_20260908/published.tar.gz'),failure='TypeError: cannot concatenate a metrics dictionary with handoff bytes; variable name reused',fix='Keep handoff bytes in note_bytes; write to a separate v2 publication directory',training_or_inference_changed=False,master_unchanged_before_recovery=sha(master))
(D/'export_recovery.json').write_text(json.dumps(recovery,indent=2)+'\n');files['export_recovery.json']=D/'export_recovery.json'
E=R/'candidate_evaluation';entry=read(E/'entry_parity/result.json')
assert entry['status']=='real_learned_OPE_and_TraX_entries_verified'
assert entry['head_sha256']==s['head_sha256']['category'] and entry['bundle_sha256']==sha(E/'bundle.json')
assert (E/'entry_parity/controller.exit').read_text().strip()=='0'
assert (E/'low22_preparation.exit').read_text().strip()=='0' and (E/'low22_binding.exit').read_text().strip()=='0'
assert not (E/'low22_controller.exit').exists()
for name in ['binding_result.json','bundle.json','entry_launch.json','low22_launch.json','low22_execution.json','binding.exit','entry_preparation.exit','low22_preparation.exit','low22_binding.exit']:
    files['evaluation__'+name]=E/name
for name in ['result.json','spec.json','ope.exit','trax.exit','verification.exit','controller.exit','resource_check.exit']:
    files['entry__'+name]=E/'entry_parity'/name
files['entry__OPE_receipt.json']=E/'entry_parity/ope/receipt.json'
files['entry__TraX_receipt.json']=E/'entry_parity/trax/receipt.json'
processes=[]
for line in subprocess.check_output(['ps','-eo','pid=,ppid=,stat=,args='],text=True).splitlines():
    parts=line.strip().split(None,3)
    if len(parts)!=4 or not parts[3].startswith(('/root/autodl-tmp/envs/sttrack/bin/python','/root/miniconda3/envs/mplt/bin/python')):continue
    cwd=str((Path('/proc')/parts[0]/'cwd').resolve())
    if str(E) in parts[3] or cwd.startswith(str(E/'low22_run')+'/'):
        processes.append(dict(pid=int(parts[0]),parent_pid=int(parts[1]),state=parts[2],cwd=cwd,command=parts[3]))
assert any('run_vot_failure_family_shards.py' in row['command'] for row in processes)
snapshot=dict(observed_utc=datetime.now(timezone.utc).isoformat(),processes=processes,low22_complete=False,full_evaluation_started=False,
    disk_free_bytes=shutil.disk_usage(R).free,entry_result_sha256=sha(E/'entry_parity/result.json'),bundle_sha256=sha(E/'bundle.json'))
(D/'low22_start_snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n');files['low22_start_snapshot.json']=D/'low22_start_snapshot.json'
concentration={}
for label,data,refs in [('paired',d,['native','empty']),('same_head',c,['empty','swapped'])]:
    for ref in refs:
        ac,ar=data['aggregates']['category'],data['aggregates'][ref];loo={};gain={}
        for seq,pc in data['per_sequence']['category'].items():
            pr=data['per_sequence'][ref][seq];n=pc['valid_frames'];assert n==pr['valid_frames']
            gain[seq]=(pc['mean_iou']-pr['mean_iou'])*n
            loo[seq]=100*((ac['iou_sum']-pc['mean_iou']*n)/(ac['valid_frames']-n)-(ar['iou_sum']-pr['mean_iou']*n)/(ar['valid_frames']-n))
        concentration[label+'_category_minus_'+ref]=dict(positive_sequences=sum(v>0 for v in gain.values()),leave_one_out_pp=loo,iou_sum_contributions=gain)
(D/'concentration.json').write_text(json.dumps(concentration,indent=2)+'\n');files['concentration.json']=D/'concentration.json'
tracker='''# M78 completed raw-response competition experiment

This completion supersedes preserved RUNNING and QUEUED launch snapshots. Only seed2027 was used. Category and independently trained Empty each completed130 fit sequences,186694 tracking calls and5798 optimizer steps, followed by complete development22. Fixed-Category-head Empty and Swapped content controls also completed; no additional training was performed for those controls.

'''
tracker+=f"Development conditions: {s['development_passes']}/{s['development_conditions']}; pass={s['development_pass']}. Same-head content conditions: {s['content_passes']}/{s['content_conditions']}; pass={s['content_pass']}.\n\n"
tracker+='The aggregate and per-sequence tables distinguish separately trained heads from fixed-head content interventions. Raw versus Hann competition preserves negative eligibility, but ranking and therefore the selected hard-negative set can differ. All training policies visit their own predicted states. No new official three-dataset result is included. Deterministic artifact and scalar checks are not independent model review. See master section5.154 for interpretation and the next step.\n'
tracker+='\nAfter the comparison snapshot, the predetermined Category final was bound, actual OPE/TraX parity passed, and303-anchor VOT low22 started. See entry__result.json and low22_start_snapshot.json. The earlier comparison public_evaluation_started=false records its own pre-binding timestamp. No completed VOT result or full benchmark is included in this snapshot.\n'
(D/'README.md').write_bytes(tracker.encode('utf-8'));files['README.md']=D/'README.md'
rows=[]
for name,p in files.items():
    q=published/name;shutil.copyfile(p,q)
    rows.append(dict(path=name,bytes=q.stat().st_size,sha256=sha(q)))
(published/'manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
archive=D/'published.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(published.iterdir()):
        assert p.is_file() and p.stat().st_size<3000000
        tar.add(p,arcname=p.name)
master.write_bytes(old+note_bytes)
record=dict(archive=str(archive),archive_sha256=sha(archive),master_sha256=sha(master),master_bytes=master.stat().st_size,files=len(files))
(D/'publication_record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
