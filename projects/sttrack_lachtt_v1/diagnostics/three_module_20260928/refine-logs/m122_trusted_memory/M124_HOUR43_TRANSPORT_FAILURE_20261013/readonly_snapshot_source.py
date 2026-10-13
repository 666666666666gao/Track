from datetime import datetime,timezone
from pathlib import Path
import json,subprocess,sys,zlib
assert datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-13T16:06:18.103772+00:00')
root=Path('/root/autodl-tmp/m124_ABC_joint_epoch20_20261011');launch=json.loads((root/'actual_launch.json').read_text())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
paths=['actual_launch.json','controller.supervisor.launch.json','controller.supervised.exit','controller.supervisor.terminal.json','current_training_state.json','best.json','epoch_history.json','result.json'];files={}
for name in paths:
 p=root/name
 if p.is_file():files[name]=p.read_text()
for folder in sorted((root/'epochs').glob('epoch-*')):
 for name in ['train.launch.json','train.exit','train.receipt.json','fit/result.json','depthtrack_test_track.exit','depthtrack_test_track.receipt.json','cdtb_track.exit','cdtb_track.receipt.json']:
  p=folder/name
  if p.is_file():files[str(p.relative_to(root))]=p.read_text()
 p=folder/'fit/sequence_log.jsonl'
 if p.is_file():
  lines=p.read_bytes().split(b'\n')[:-1];files[str(p.relative_to(root))]=b'\n'.join(lines).decode()+'\n'
 for dataset in ['depthtrack_test','cdtb']:
  pred=folder/'evaluation'/dataset/'predictions'
  if pred.is_dir():
   files[str((pred/'bbox_file_count').relative_to(root))]=str(len([p for p in pred.glob('*.txt') if not p.name.endswith('_all_scores.txt')]))
   if (folder/(dataset+'_analyze.exit')).is_file() and (folder/(dataset+'_analyze.exit')).read_text().strip()=='0':files[str((pred/'metrics.json').relative_to(root))]=(pred/'metrics.json').read_text()
proc=Path('/proc')/str(launch['supervisor_pid'])/'cmdline';command=proc.read_bytes().replace(bytes([0]),b' ').decode() if proc.exists() else None
value=dict(status='actual_M124_fortyfourth_hourly_readonly_observation',observed_at=datetime.now(timezone.utc).isoformat(),root=str(root),GPU_snapshot=gpu,GPU_queries=1,NN_progress_queries=1,NN_forward_calls=0,controller_cmdline=command,files=files,new_training_launched=False,restarted=False)
sys.stdout.buffer.write(zlib.compress((json.dumps(value)+chr(10)).encode('utf-8')));sys.stdout.buffer.flush()
