"""Preserve interrupted CDTB output and launch only the unfinished fixed queue."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/root/autodl-tmp/sttrack_selected_full_evaluation_20260921')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())

assert not (ROOT/'resume_launch.json').exists()
selection=read(ROOT/'selection.json')
for f,h in selection['source_sha256'].items():assert sha(ROOT/f)==h,f
assert sha(ROOT/'EXPERIMENT_PLAN.md')==selection['plan_sha256']
for name in ['M67','M82']:
    p=ROOT/name/'bundle.json';assert sha(p)==selection['models'][name]['bundle_sha256']
    b=read(p)
    for key in ['adapter_checkpoint','base_checkpoint']:assert sha(b[key])==b[key+'_sha256']
    for f,h in b['source_sha256'].items():assert sha(Path(b['repository'])/f)==h
    for f,h in b['interface_sha256'].items():assert sha(ROOT/'interface'/f)==h
    assert sha(b['text_protocol_path'])==b['text_protocol_sha256']
for folder in ['inputs_bfloat16/depthtrack/captions','inputs_bfloat16/cdtb/captions','vot_inputs/all_initializations']:
    p=ROOT/folder;g=read(p/'generation_result.json');e=read(p/'encoding_result.json')
    assert g['generation_dtype']=='bfloat16' and g['all_generation_logits_finite']
    assert sha(p/'records.jsonl')==g['records_sha256'] and sha(p/'text_bank.pt')==e['bank_sha256']
assert (ROOT/'full_text.exit').read_text().strip()=='0'
done=ROOT/'M67/depthtrack/predictions';receipt=read(done/'receipt.json');metrics=read(done/'metrics.json')
assert receipt['status']=='complete' and receipt['frames']==76373 and len(receipt['sequences'])==50
assert metrics['receipt_sha256']==sha(done/'receipt.json') and metrics['status']=='complete'
assert metrics['bundle_sha256']==selection['models']['M67']['bundle_sha256']
for row in receipt['sequences']:
    assert sha(done/(row['sequence']+'.txt'))==row['bbox_sha256']
    assert sha(done/(row['sequence']+'_all_scores.txt'))==row['confidence_sha256']
assert not (ROOT/'M67/cdtb/predictions/receipt.json').exists()
assert not (ROOT/'M67/cdtb_track.exit').exists()
assert not (ROOT/'M67/vot').exists()
assert not (ROOT/'M82/depthtrack').exists() and not (ROOT/'M82/cdtb').exists() and not (ROOT/'M82/vot').exists()
used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()]
assert len(used)==2 and max(used)<500,used
assert shutil.disk_usage(ROOT).free>1000000000
# Explicitly check actual process arguments, not stale PID files.
for p in Path('/proc').iterdir():
    if p.name.isdigit() and (p/'cmdline').is_file():
        raw=(p/'cmdline').read_bytes().split(b'\0')
        if len(raw)>1 and any(x.endswith((b'run_semantic_ope.py',b'run_semantic_vot.py',b'run_vot_failure_family_shards.py',b'run_full.sh',b'resume_20260924.sh')) for x in raw):
            raise RuntimeError('Tracking controller already active: PID '+p.name)
subprocess.run(['bash','-n',str(ROOT/'resume_20260924.sh')],check=True)
archive=ROOT/'M67/interrupted_cdtb_20260924';archive.mkdir()
partial=ROOT/'M67/cdtb/predictions'
partial_hashes={p.name:sha(p) for p in partial.iterdir() if p.is_file()}
partial.rename(archive/'predictions')
(ROOT/'M67/cdtb_track.log').rename(archive/'cdtb_track.log')
source_hashes={n:sha(ROOT/n) for n in ['resume_20260924.sh','migration_preflight_20260924.py','launch_resume_20260924.py']}
with (ROOT/'resume_controller.log').open('x') as log:
    process=subprocess.Popen(['bash',str(ROOT/'resume_20260924.sh')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(status='remaining_full_evaluations_launched',observed_utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,
    endpoint_port=43811,actual_vot_poll_seconds=3600,original_execution_metadata_poll_seconds=240,source_sha256=source_hashes,selection_sha256=sha(ROOT/'selection.json'),
    preserved_DepthTrack_metric_sha256=sha(done/'metrics.json'),preserved_partial_files=partial_hashes,
    partial_archive=str(archive),no_live_previous_tracker=True,new_training_steps=0,
    queue=['M67 CDTB','M67 VOT full127','M82 DepthTrack Test','M82 CDTB','M82 VOT full127'])
(ROOT/'resume_launch.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
