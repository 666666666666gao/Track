"""One read-only original R0 observation at the predeclared Beijing end node."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import base64
import json
import os
import subprocess
import zlib

A = Path(__file__).resolve().parent
node = json.loads((A / 'M125_R0_suite_end_node_20261018.json').read_bytes())
output = A / 'M125_R0_suite_end_observation_20261018.json'
error = A / 'M125_R0_suite_end_observation_20261018_stderr.log'
assert not output.exists() and not error.exists(), 'The one-shot observation already has an attempt receipt.'
assert datetime.now(timezone.utc) >= datetime.fromisoformat(node['not_before'])
with output.with_name(output.stem + '_attempt.json').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),node=node)) + '\n')
remote = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,zlib
node=NODE
assert datetime.now(timezone.utc)>=datetime.fromisoformat(node['not_before'])
root=Path(node['root']);files={};passes=[]
def capture(name):
    path=root/name
    if path.is_file():
        data=path.read_bytes()
        files[name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=data.decode())
def committed(name):
    path=root/name
    if not path.is_file():return []
    return [json.loads(line) for line in path.read_bytes().split(b'\\n')[:-1]]
for name in ['actual_launch.json','controller.supervisor.launch.json','controller.supervised.exit',
             'controller.supervisor.terminal.json','controller.exit','current_training_state.json',
             'result.json','evaluation/all_results.json','evaluation/selection.json']:
    capture(name)
for epoch in range(1,7):
    prefix='passes/pass-%02d/'%epoch
    folder=root/prefix
    if not folder.is_dir():continue
    for suffix in ['train_fixed_pass.launch.json','train_fixed_pass.exit',
                   'train_fixed_pass.receipt.json','fit/result.json']:
        capture(prefix+suffix)
    checkpoint=folder/'fit/latest.pt'
    metadata=dict(exists=checkpoint.is_file())
    if checkpoint.is_file():metadata.update(bytes=checkpoint.stat().st_size,mtime_ns=checkpoint.stat().st_mtime_ns)
    trace=committed(prefix+'fit/sampled_state_trace.jsonl')
    log=folder/'train_fixed_pass.log'
    passes.append(dict(pass_index=epoch,sequence_records=committed(prefix+'fit/sequence_log.jsonl'),
        last_sampled_trace=trace[-1:],checkpoint_metadata=metadata,
        log_tail=log.read_bytes()[-8000:].decode() if log.is_file() else ''))
for stage in ['prepare','depthtrack_test_track','cdtb_track','depthtrack_test_analyze','cdtb_analyze',
              'vot_track','vot_official_analysis','vot_seal','collect']:
    for suffix in ['.launch.json','.exit','.receipt.json']:capture(stage+suffix)
    log=root/(stage+'.log')
    if log.is_file():
        data=log.read_bytes()[-5000:]
        files[stage+'.log_tail']=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=data.decode())
for name in ['evaluation/vot/run/shard_manifest.json','evaluation/vot/run/launch.json','evaluation/vot/run/merge_result.json',
             'evaluation/depthtrack_test/predictions/receipt.json','evaluation/cdtb/predictions/receipt.json']:
    capture(name)
manifest_path=root/'evaluation/vot/run/shard_manifest.json'
if manifest_path.is_file():
    manifest=json.loads(manifest_path.read_bytes());proxy=[]
    for shard in manifest['shards']:
        capture('evaluation/vot/run/shard-%02d.exit'%shard['index'])
        complete=0
        for name in shard['expected_trajectories']:
            folder=Path(shard['root'])/'results'/manifest['tracker']/'baseline'/name.rsplit('_',1)[0]
            complete+=all((folder/(name+suffix)).is_file() and (folder/(name+suffix)).stat().st_size>0
                          for suffix in ['.bin','_confidence.value','_time.value'])
        proxy.append(dict(index=shard['index'],expected=shard['anchor_count'],nonempty_triplet_proxy=complete))
    data=json.dumps(proxy).encode()
    files['evaluation/vot/run/nonempty_triplet_proxy.json']=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),text=data.decode())
process=Path('/proc')/str(node['original_supervisor_pid'])/'cmdline'
command=process.read_bytes().replace(bytes([0]),b' ').decode() if process.is_file() else ''
value=dict(status='actual_original_M125_R0_six_pass_end_node_observation',
    observed_at=datetime.now(timezone.utc).isoformat(),node=node,files=files,passes=passes,
    original_supervisor_cmdline=command,NN_calls=0,GPU_queries=0,automatic_retries=0)
print(base64.b64encode(zlib.compress(json.dumps(value).encode())).decode())
'''.replace('NODE', repr(node))
tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                   SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
assert environment['RGBD_GOAL_SSH_PASSWORD']
completed = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
                           input=remote.encode(), capture_output=True, env=environment)
with error.open('xb') as stream: stream.write(completed.stderr)
with (output.with_suffix('.stdout.b64')).open('xb') as stream: stream.write(completed.stdout)
assert completed.returncode == 0, completed.stderr.decode()
value = json.loads(zlib.decompress(base64.b64decode(completed.stdout)))
with output.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(json.dumps(value, indent=2) + '\n')
print(json.dumps(dict(status=value['status'],observed_at=value['observed_at'],
    original_supervisor_cmdline=value['original_supervisor_cmdline'],
    passes=[dict(pass_index=row['pass_index'],complete_sequences=len(row['sequence_records']),
        last_sequence=row['sequence_records'][-1:] ,last_trace=row['last_sampled_trace']) for row in value['passes']],
    terminal_files=[name for name in value['files'] if name.endswith('.exit') or name=='result.json'])))
