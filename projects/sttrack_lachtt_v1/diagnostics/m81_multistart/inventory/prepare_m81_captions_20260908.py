"""Prepare only the new Train initialization observations using the sealed captioner."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
B=Path('/root/autodl-tmp');R=B/'sttrack_m81_multistart_20260908';G=B/'sttrack_m58_initialization_generator_20260906'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
s=read(R/'inventory_spec.json')
assert s['status']=='initialization_inventory_prepared_no_training'
assert sha(G/'initialization_captions.py')=='eeb15e8bee4a40692cfb19e8ec73329e232f3a8c25c96f789a03b991d71ac655'
assert not (R/'captions').exists() and not (R/'caption_inputs.json').exists()
cases=[]
for split in ['fit','development']:
    assert sha(R/(split+'_initializations.json'))==s[split+'_manifest_sha256']
    for row in read(R/(split+'_initializations.json'))['episodes']:
        if row['start_frame']==0:continue
        assert Path('/root/autodl-tmp/depthtrack/train/sequences') in Path(row['image']).parents
        assert sha(row['image'])==row['image_sha256']
        cases.append(dict(id=row['id'],image=row['image'],bbox=row['init_bbox']))
assert len(cases)==348 and len({r['id'] for r in cases})==348
write(R/'caption_inputs.json',dict(coordinate_convention='ope_raw_xywh',cases=cases))
sys.path.insert(0,str(G));import initialization_captions as app
plan=app.prepare(R/'caption_inputs.json',R/'captions')
assert len(plan['cases'])==348 and len(plan['rows'])<=348
queue='''#!/bin/bash
set -u
root=/root/autodl-tmp/sttrack_m81_multistart_20260908
generator=/root/autodl-tmp/sttrack_m58_initialization_generator_20260906/initialization_captions.py
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=1 /home/qwen25_env/bin/python -u "$generator" generate --plan "$root/captions/plan.json" > "$root/caption_generate.log" 2>&1
status=$?
printf '%s\\n' "$status" > "$root/caption_generate.exit"
if [ "$status" -ne 0 ]; then printf '%s\\n' "$status" > "$root/caption_controller.exit"; exit "$status"; fi
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -u "$generator" encode --plan "$root/captions/plan.json" > "$root/caption_encode.log" 2>&1
status=$?
printf '%s\\n' "$status" > "$root/caption_encode.exit"
printf '%s\\n' "$status" > "$root/caption_controller.exit"
exit "$status"
'''
(R/'run_captions.sh').write_text(queue);subprocess.run(['bash','-n',str(R/'run_captions.sh')],check=True)
record=dict(status='caption_plan_prepared_not_started',observed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
    inventory_spec_sha256=sha(R/'inventory_spec.json'),caption_inputs_sha256=sha(R/'caption_inputs.json'),caption_plan_sha256=sha(R/'captions/plan.json'),generator_sha256=sha(G/'initialization_captions.py'),queue_sha256=sha(R/'run_captions.sh'),
    cases=len(cases),unique_observations=len(plan['rows']),fit_later_initializations=296,development_later_initializations=52,
    model='existing_Qwen2.5-VL-3B-Instruct',generation_gpu=1,encoding_device='cpu',training_seed=2027,additional_training_seeds=[],
    protocol='Existing initialization-only full target-marked image plus target crop; unchanged prompt/generation/processor. No sequence/category hint or later image.',
    outputs='Raw category/attributes preserved; downstream category arm blanks valid attribute content, Empty arm blanks all valid content. No semantic correctness guarantee.',
    training_has_started=False,external_images_used=False,new_model_downloads=False)
write(R/'caption_preparation.json',record);print(json.dumps(record))
