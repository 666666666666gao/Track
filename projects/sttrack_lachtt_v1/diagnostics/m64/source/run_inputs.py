from pathlib import Path
import subprocess, os, json
from datetime import datetime, timezone
R=Path('/root/autodl-tmp/sttrack_m64_category_candidate_20260907')
G='/root/autodl-tmp/sttrack_m58_initialization_generator_20260906/initialization_captions.py'
S='/root/autodl-tmp/m64_category_candidate_20260907.py'
P='/root/autodl-tmp/envs/sttrack/bin/python'
Q='/home/qwen25_env/bin/python'
stages=[
 ('train_generate','1',[Q,G,'generate','--plan',str(R/'train_generator_replay/plan.json')]),
 ('train_verify','',[P,S,'verify_replay']),
 ('low22_prepare','',[Q,S,'low22_prepare']),
 ('low22_generate','1',[Q,G,'generate','--plan',str(R/'low22_captions/plan.json')]),
 ('low22_encode','',[P,G,'encode','--plan',str(R/'low22_captions/plan.json')]),
 ('low22_bank','',[P,S,'low22_bank'])]
for name,gpu,cmd in stages:
 (R/'input_stage.json').write_text(json.dumps(dict(stage=name,observed_utc=datetime.now(timezone.utc).isoformat())))
 with (R/(name+'.log')).open('w') as log:
  code=subprocess.run(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT).returncode
 (R/(name+'.exit')).write_text(str(code)+'\n')
 if code:
  (R/'input_job.exit').write_text(str(code)+'\n')
  raise SystemExit(code)
(R/'input_job.exit').write_text('0\n')
