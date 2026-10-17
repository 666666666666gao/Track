from datetime import datetime, timezone, timedelta
from pathlib import Path
import ast
import hashlib
import json

A = Path(__file__).resolve().parent
raw_path = A / 'M125_R0_training_end_observation_20261017.json'
raw = json.loads(raw_path.read_bytes())
for name, item in raw['files'].items():
    assert hashlib.sha256(item['text'].encode()).hexdigest() == item['sha256'], name
launch = json.loads(raw['files']['actual_launch.json']['text'])
launch_stdout_path = A / 'M125_R0_fixed6_actual_launch_20261015_stdout.json'
assert hashlib.sha256(launch_stdout_path.read_bytes()).hexdigest() == raw['node']['original_launch_sha256']
assert json.loads(launch_stdout_path.read_bytes()) == launch
first_observation = json.loads((A / 'M125_R0_first_training_commit_observation_20261015.json').read_bytes())
assert raw['files']['actual_launch.json']['sha256'] == first_observation['files']['actual_launch.json']['sha256']
assert raw['original_supervisor_cmdline'].split() == launch['command']
assert launch['supervisor_pid'] == raw['node']['original_supervisor_pid'] == 895667
assert [row['pass_index'] for row in raw['passes']] == [1, 2, 3, 4, 5, 6]
rows = []
for row in raw['passes']:
    epoch = row['pass_index']; records = row['sequence_records']
    assert [item['sequence_index'] for item in records] == list(range(len(records)))
    assert all(item['epoch'] == epoch for item in records)
    assert [item['sequence'] for item in records] == [item['sequence'] for item in raw['passes'][0]['sequence_records'][:len(records)]]
    calls = sum(item['track_calls'] for item in records)
    assert records[-1]['total_track_calls'] == (epoch - 1) * 219802 + calls
    completed = epoch <= 5
    if completed:
        assert len(records) == 152 and calls == 219802
        prefix = 'passes/pass-%02d/' % epoch
        assert raw['files'][prefix + 'train_fixed_pass.exit']['text'].strip() == '0'
        assert json.loads(raw['files'][prefix + 'train_fixed_pass.receipt.json']['text'])['exit_code'] == 0
        result = json.loads(raw['files'][prefix + 'fit/result.json']['text'])
        assert result['epochs'] == epoch and result['sequence_runs'] == epoch * 152
        assert result['track_calls'] == epoch * 219802
        assert result['optimizer_steps'] == records[-1]['total_optimizer_steps']
        assert result['C_optimizer_steps'] == records[-1]['total_C_optimizer_steps']
        assert not result['fixed_final_after_complete_budget']
    rows.append(dict(pass_index=epoch, complete_sequences=len(records), train_track_calls_in_pass=calls,
                     complete_pass=completed, cumulative_train_track_calls=records[-1]['total_track_calls'],
                     cumulative_AB_updates=records[-1]['total_optimizer_steps'],
                     cumulative_C_updates=records[-1]['total_C_optimizer_steps'],
                     committed_elapsed_seconds=records[-1]['elapsed_seconds']))
state = json.loads(raw['files']['current_training_state.json']['text'])
assert state['pass_completed'] == 5 and state['fixed_budget_passes'] == state['final_pass'] == 6
assert rows[-1]['complete_sequences'] == 141
assert not any(name in raw['files'] for name in ['controller.supervised.exit', 'controller.exit', 'result.json',
                                               'evaluation/all_results.json', 'evaluation/selection.json'])
observed = datetime.fromisoformat(raw['observed_at'])
partial = rows[-1]
remaining = 219802 - partial['train_track_calls_in_pass']
speed = partial['train_track_calls_in_pass'] / partial['committed_elapsed_seconds']
train_eta = observed + timedelta(seconds=remaining / speed)
old_path = A / 'M124_E6_terminal_audit_artifacts_20261015.json'
old = json.loads(old_path.read_bytes())
old_root = '/root/autodl-tmp/m124_ABC_joint_epoch20_20261011'
ope = {}
for dataset in ['depthtrack_test', 'cdtb']:
    key = old_root + '/epochs/epoch-06/evaluation/' + dataset + '/predictions/receipt.json'
    ope[dataset] = json.loads(old['files'][key]['text'])['elapsed_seconds']
old_result = json.loads(old['files'][old_root + '/result.json']['text'])
vot_stages = [row for row in old_result['stages'] if row['stage'] in
              ['prepare_best_vot', 'best_vot_track', 'best_vot_official_analysis', 'best_vot_seal', 'collect_best']]
assert len(vot_stages) == 5 and all(row['exit_code'] == 0 for row in vot_stages)
vot_and_seal_seconds = sum((datetime.fromisoformat(row['finished_at']) - datetime.fromisoformat(row['started_at'])).total_seconds()
                          for row in vot_stages)
evaluation_seconds = max(ope.values()) + vot_and_seal_seconds
suite_eta = train_eta + timedelta(seconds=evaluation_seconds)
node = dict(status='one_planned_R0_complete_suite_end_observation', not_before='2026-10-18T12:15:00+08:00',
            original_supervisor_pid=895667, root=raw['node']['root'],
            original_launch_stdout_sha256=raw['node']['original_launch_sha256'],
            original_remote_launch_sha256=raw['files']['actual_launch.json']['sha256'],
            training_progress_polling_before_node=False, new_local_timer=False,
            wait_method='root interruptible calendar wait then the one guarded read; old missing Windows helper retired',
            last_actual_observation_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
summary = dict(status='verified_original_R0_near_final_pass_progress_only', observed_at=raw['observed_at'],
    observation_sha256=node['last_actual_observation_sha256'], original_supervisor_exact_cmdline_verified=True,
    original_launch_stdout_object_matches_remote_receipt=True,
    original_remote_launch_bytes_equal_first_observation=True,
    original_launch_stdout_sha256=node['original_launch_stdout_sha256'],
    original_remote_launch_sha256=node['original_remote_launch_sha256'],
    previous_hidden_waiter_pid=49616, previous_hidden_waiter_absent_at_due_node=True,
    previous_hidden_waiter_terminal_exit_code=None, due_read_original_reviewed_entrypoint=True,
    due_read_root_unified_session=2052, due_read_observed_exit_code=0,
    original_NN_restarted=False, original_scientific_sources_modified=False,
    completed_passes=5, planned_passes=6, complete_sequence_runs=901, planned_sequence_runs=912,
    cumulative_committed_train_track_calls=1302307, planned_train_track_calls=1318812,
    cumulative_committed_AB_updates=40328, cumulative_committed_C_updates=40328,
    current_pass=6, current_pass_complete_sequences=141, current_pass_total_sequences=152,
    pass_progress=rows, remaining_train_calls_from_committed_prefix=remaining,
    estimated_train_calls_per_second=speed, estimated_training_end_UTC=train_eta.isoformat(),
    historical_E6_ope_elapsed_seconds=ope, historical_E6_VOT_and_final_CPU_seconds=vot_and_seal_seconds,
    estimated_full_evaluation_seconds_excluding_small_new_prepare_OPE_analysis=evaluation_seconds,
    estimated_complete_suite_end_UTC=suite_eta.isoformat(),
    ETA_is_estimate_not_terminal_evidence=True, actual_new_final_available=False, actual_new_nine_metrics=False,
    next_node=node, actual_current_observation_NN_calls=0, actual_current_observation_GPU_queries=0)
for name, value in [('M125_R0_due_node_progress_summary_20261017.json', summary),
                    ('M125_R0_suite_end_node_20261018.json', node)]:
    with (A / name).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
observer = (A / 'observe_M125_R0_training_end_R2_once_20261017.py').read_bytes().decode()
observer = observer.replace('M125_R0_training_end_node_20261017.json', 'M125_R0_suite_end_node_20261018.json')
observer = observer.replace('M125_R0_training_end_observation_20261017', 'M125_R0_suite_end_observation_20261018')
needle = "process=Path('/proc')/str(node['original_supervisor_pid'])/'cmdline'"
assert observer.count(needle) == 1
additional = '''for stage in ['prepare','depthtrack_test_track','cdtb_track','depthtrack_test_analyze','cdtb_analyze',
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
'''
observer = observer.replace(needle, additional + needle)
tree = ast.parse(observer, feature_version=(3, 8))
remote = next(n.value.func.value.value for n in tree.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'remote')
ast.parse(remote, feature_version=(3, 8))
with (A / 'observe_M125_R0_suite_end_once_20261018.py').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(observer)
print(json.dumps(dict(status=summary['status'], completed_passes=5, pass6_sequences=141,
    committed_track_calls=1302307, AB_updates=40328, C_updates=40328,
    estimated_training_end_UTC=train_eta.isoformat(), estimated_complete_suite_end_UTC=suite_eta.isoformat(),
    next_observation=node['not_before'], new_NN_calls=0)))
