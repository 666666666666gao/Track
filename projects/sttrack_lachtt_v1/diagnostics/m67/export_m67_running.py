from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,tarfile,subprocess
BASE=Path('/root/autodl-tmp');ROOT=BASE/'sttrack_m67_supervised_semantic_support_20260907'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ROOT/'training_spec.json')=='2027b1893343e5e1520c447bd82a5c45468542cecaab0f59b3ec72a345e48b2e'
assert sha(ROOT/'recursive_spec.json')=='d4dd7ca74b8f4b318f8e32f961bc21a10b95aa3802137dd293b49948d83411c5'
assert (ROOT/'causal_smoke.exit').read_text().strip()=='0'
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or not (p/'cmdline').is_file():continue
    b=(p/'cmdline').read_bytes()
    if b'train_causal.py\x00--arm\x00' in b and p.joinpath('cwd').resolve()==ROOT:
        processes.append(dict(pid=int(p.name),command=b.replace(b'\x00',b' ').decode().strip()))
assert len(processes)==2
counts={}
for arm in ['control','support']:
    assert not (ROOT/('training_'+arm+'.exit')).exists()
    f=ROOT/'training'/arm/'sequence_log.jsonl';lines=f.read_text().splitlines()
    rows=[json.loads(x) for x in lines]
    counts[arm]=dict(completed_sequences=len(rows),last_completed_sequence=rows[-1] if rows else None)
qwen={}
for name in ['Qwen3_8B','Qwen2.5-VL-3B-Instruct']:
    files=list((BASE/'qwen'/name).glob('*.safetensors'));assert files
    qwen[name]=dict(shards=len(files),bytes=sum(f.stat().st_size for f in files))
r=dict(status='paired_training_running',observed_utc=datetime.now(timezone.utc).isoformat(),processes=processes,completed_sequence_prefixes=counts,
    launch_sha256=sha(ROOT/'launch_receipt.json'),free_bytes=shutil.disk_usage(ROOT).free,preserved_qwen=qwen,
    full_training_results_exist=False,formal_three_dataset_metrics_exist=False,independent_model_review_pass=False)
(ROOT/'running_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
out=ROOT/'running_publication_evidence';out.mkdir()
names=['training_spec.json','recursive_spec.json','prepared_training_spec.json','preparation.json','integration.json','support_loss.py',
       'causal_training.py','train_causal.py','run_recursive.py','run_m67.sh','freeze_m67.py','check_m67.py',
       'causal_smoke_result.json','causal_smoke.log','causal_smoke.exit','launch_receipt.json','running_receipt.json',
       'initial_check_m67.py','initial_causal_smoke.log','initial_causal_smoke.exit','second_check_m67.py','second_causal_smoke.log','second_causal_smoke.exit',
       'third_check_m67.py','third_causal_smoke.log','third_causal_smoke.exit','control_mismatch_diagnostic.json',
       'diagnose_control.py','original_repeat_diagnostic.json','original_repeat_diagnostic.log','original_repeat_diagnostic.exit','diagnose_original_repeat.py',
       'initial_causal_training.py','initial_train_causal.py','initial_preparation.json','initial_prepared_training_spec.json']
for n in names:shutil.copyfile(ROOT/n,out/n)
shutil.copyfile(BASE/'prepare_m67_20260907.py',out/'prepare_m67_initial.py');shutil.copyfile(__file__,out/'export_m67_running.py')
(out/'evidence_manifest.json').write_text(json.dumps([dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())],indent=2)+'\n')
a=ROOT/'running_evidence.tar.gz'
with tarfile.open(a,'w:gz') as t:
    for p in sorted(out.iterdir()):t.add(p,arcname=p.name)
print(json.dumps(dict(sha256=sha(a),bytes=a.stat().st_size,receipt=r),indent=2))
