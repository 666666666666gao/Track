"""Durable M109 controller preserving actual terminal exit; no retry."""
import json,subprocess,time
from pathlib import Path
root=Path('/root/autodl-tmp/sttrack_m109_fit_gradients_20261002')
d=Path('/home/SUTrack_RGBD_L/projects/sttrack_lachtt_v1/diagnostics/three_module_20260928')
code=subprocess.call(['/root/autodl-tmp/envs/sttrack/bin/python','-u',str(d/'run_m109_pair.py'),
                      '--output',str(root),'--labels',str(d/'m107_weak_inputs/weak_labels.json')])
root.mkdir(exist_ok=True)
(root/'controller.exit').write_text(str(code)+'\n')
print(json.dumps({'status':'terminal','exit':code,'finished':time.time()}),flush=True)
raise SystemExit(code)
