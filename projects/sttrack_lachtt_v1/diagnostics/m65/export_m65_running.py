"""Export immutable plan/source plus an explicitly running-state receipt."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,tarfile,shutil,os,subprocess
R=Path('/root/autodl-tmp/sttrack_m65_category_null_support_20260907')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((R/'training_spec.json').read_text());p=json.loads((R/'preparation.json').read_text())
assert sha(R/'train_causal.py')==s['training_script_sha256']
assert sha(R/'run_m65.sh')==s['run_queue_sha256']
assert (R/'causal_smoke.exit').read_text().strip()=='0'
launch=json.loads((R/'launch_receipt.json').read_text());assert launch['training_spec_sha256']==sha(R/'training_spec.json')
progress={}
for arm in ['control','null']:
    path=R/'training'/arm/'sequence_log.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert rows and not (R/('training_'+arm+'.exit')).exists()
    pid=launch['training_pids'][arm];assert Path('/proc') .joinpath(str(pid),'cmdline').exists()
    command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'train_causal.py --arm '+arm in command
    progress[arm]=dict(pid=pid,completed_sequences=len(rows),last_sequence=rows[-1],
       current_latest_checkpoint_sha256=sha(R/'training'/arm/'latest.pth'),
       status='running',complete_metrics=False)
qwen={}
for name in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']:
    files=list((Path('/root/autodl-tmp/qwen')/name).glob('*.safetensors'));qwen[name]=dict(files=len(files),bytes=sum(f.stat().st_size for f in files))
assert qwen['Qwen3_8B']==dict(files=5,bytes=16381516776)
assert qwen['Qwen2.5-VL-3B-Instruct']==dict(files=2,bytes=7509337976)
receipt=dict(status='paired_formal_training_running',observed_utc=datetime.now(timezone.utc).isoformat(),
    training_spec_sha256=sha(R/'training_spec.json'),recursive_spec_sha256=sha(R/'recursive_spec.json'),
    launch_receipt_sha256=sha(R/'launch_receipt.json'),progress=progress,disk_free_bytes=shutil.disk_usage(R).free,
    qwen_preserved=qwen,formal_dataset_metrics_exist=False,development_complete=False,
    actual_dataset_optimizer_steps=sum(x['last_sequence']['total_optimizer_steps'] for x in progress.values()),
    independent_model_review_pass=False,new_caption_calls=0,new_learned_parameters_each=289154)
(R/'running_evidence_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
dest=R/'running_publication_evidence';dest.mkdir()
paths=['preparation.json','integration.json','training_spec.json','recursive_spec.json','causal_smoke_result.json',
 'causal_smoke.exit','launch_receipt.json','running_evidence_receipt.json','train_causal.py','causal_training.py',
 'run_recursive.py','recursive_metric.py','check_m65.py','freeze_m65.py','run_m65.sh',
 'code/lib/models/sttrack/semantic_spatial_adapter.py','code/lib/test/tracker/sttrack_semantic.py']
for n in paths:shutil.copyfile(R/n,dest/Path(n).name)
for name in ['prepare_m65.py','export_m65_running.py']:shutil.copyfile(Path('/root/autodl-tmp')/name,dest/name)
for arm in progress:
    # Capture already completed sequence rows rather than racing a growing log copy.
    (dest/(arm+'_completed_prefix.json')).write_text(json.dumps(progress[arm],indent=2)+'\n')
manifest=[dict(path=f.name,bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(dest.iterdir())]
(dest/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=R/'running_publication_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    for f in sorted(dest.iterdir()):t.add(f,arcname=f.name)
print(json.dumps(dict(receipt=receipt,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size),indent=2))
