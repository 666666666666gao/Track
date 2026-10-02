"""Durable M107 controller. Preserves child terminal exit without retry."""
import json,subprocess,sys,time
from pathlib import Path
root=Path('/root/autodl-tmp/sttrack_m107_model_weak_pair_20260928')
d=Path('/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928')
args=['/root/autodl-tmp/envs/sttrack/bin/python','-u',str(d/'run_m107_weak_pair.py'),'--output',str(root),'--labels',str(d/'m107_weak_inputs/weak_labels.json')]
code=subprocess.call(args)
root.mkdir(exist_ok=True)
(root/'controller.exit').write_text(str(code)+'\n')
print(json.dumps({'status':'terminal','exit':code,'finished':time.time()}),flush=True)
raise SystemExit(code)
