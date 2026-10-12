from pathlib import Path
from datetime import datetime,timezone
import json

A=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
owner=json.loads((A/'M124_thirteenth_hour_original_wait_owner_20261012.json').read_bytes())
assert owner['native_session_id']==61182 and owner['process_id']==48256
deadline=owner['deadline']
assert deadline=='2026-10-12T06:24:38.140404+00:00' and datetime.now(timezone.utc)<datetime.fromisoformat(deadline)
loss=dict(status='actual_original_M124_thirteenth_local_wait_handle_and_PID_missing',observed_local_time='2026-10-12T13:54:11.1503208+08:00',original_native_session_id=61182,original_process_id=48256,original_functions_cell_id='1746',functions_handle_error='exec cell 1746 not found',native_handle_error='write_stdin failed: Unknown process id 61182',actual_CIM_chunk_id='6f2429',original_PID_present=False,original_terminal_file_present=False,original_join_receipt_present=False,original_exit_code=None,original_deadline=deadline,server_queries=0,NN_progress_queries=0,GPU_queries=0,training_restarted=False,training_stop_inferred=False,replacement_local_wait_only=True)
path=A/'M124_thirteenth_hour_original_wait_handle_loss_20261012.json'
assert not path.exists()
path.write_text(json.dumps(loss,indent=2)+'\n',encoding='utf-8')
timer=A/'M124_passive_thirteenth_hourly_wait_replacement1_20261012.py'
assert not timer.exists()
source="""from datetime import datetime,timezone
import json,os,time
from pathlib import Path
deadline=datetime.fromisoformat('2026-10-12T06:24:38.140404+00:00')
value=dict(status='actual_local_M124_thirteenth_replacement1_wait_started',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),deadline=deadline.isoformat(),sleep_interval_seconds=300,server_queries=0,NN_progress_queries=0,restarted=True,original_native_session_id=61182,original_process_id=48256,reason='Original process handle unknown and exact PID absent; no original terminal record; no training restart')
print(json.dumps(value),flush=True)
while datetime.now(timezone.utc)<deadline:
    time.sleep(min(300,max(.01,(deadline-datetime.now(timezone.utc)).total_seconds())))
value.update(status='actual_local_M124_thirteenth_replacement1_wait_deadline_reached',finished_at=datetime.now(timezone.utc).isoformat())
Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007\M124_thirteenth_replacement1_wait_terminal_20261012.json').write_text(json.dumps(value,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(value),flush=True)
"""
timer.write_text(source,encoding='utf-8')
state_path=A/'CURRENT_STATUS.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(M124_thirteenth_hour_original_wait_handle_loss=loss,M124_thirteenth_replacement1_wait_prepared_not_launched=True,next_NN_observation_not_before=deadline,previous_goal_turn_classification='VERIFIED_WAIT_THEN_ACTUAL_LOCAL_HANDLE_LOSS_NOT_REMOTE_TRAINING_FAILURE')
state['M124_thirteenth_hour_original_wait_owner'].update(handle_lost=True,process_missing=True,consumed=False)
state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
line='\nRGB-D Oct12 13:54 actual original localwait61182/PID48256 andfunctions1746 missing before14:24:38 deadline; originalterminal/joins absent, originalexit unknown. ActualCIM6f2429 PIDabsent/native61182UnknownProcess/cell1746notfound; NOT inferred remotefailure/NOT restarted training. One replacement local-only countdown source prepared, originaldeadline2026-10-12T06:24:38.140404+00:00 unchanged, no SSH/GPU/NN/progressquery. Preserve lossJSON/originalowner; replacementnotlaunchedyet. GoalACTIVE_UNMET.\n'
for path in [Path(r'C:\Users\gb\memory\2026-10-12.md'),Path(r'C:\Users\gb\MEMORY.md')]:
    with path.open('a',encoding='utf-8') as stream:stream.write(line)
print(json.dumps(loss,indent=2))
