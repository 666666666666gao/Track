from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,zipfile
A=Path(r'C:\Users\gb\.codex_track_publish_m29_20260902\.aris\m122_full_causal_20261007')
archive=A/'M124_epoch05_CDTB_primary_inputs_original_20261013.zip'
expected=json.loads((A/'M124_epoch05_CDTB_CPU_diagnostic_original_stdout_20261013.json').read_bytes())['inputs_sha256']
local=Path(r'C:\Users\gb\Desktop\document\RGBD_LANGUAGE_TRACKING\实验诊断\M124E5_CDTB原始输入_20261013')
assert not local.exists()
entries=[]
with zipfile.ZipFile(archive) as package:
 manifest=json.loads(package.read('PRIMARY_INPUT_MANIFEST.json'));assert set(manifest['files'])==set(expected) and len(expected)==325
 for i,(origin,item) in enumerate(manifest['files'].items(),1):
  data=package.read(item['member']);assert hashlib.sha256(data).hexdigest()==item['sha256']==expected[origin] and len(data)==item['bytes']
  target=local/(str(i).zfill(3)+'_'+Path(origin).name)
  assert len(str(target))<240
  entries.append((origin,item,target,data))
 local.mkdir()
 copies={}
 for origin,item,target,data in entries:
  target.write_bytes(data);assert hashlib.sha256(target.read_bytes()).hexdigest()==expected[origin]
  copies[origin]=dict(item,local_path=str(target))
 (local/'PRIMARY_INPUT_MANIFEST.json').write_bytes((json.dumps(dict(manifest,files=copies),indent=2)+chr(10)).encode())
raw=archive.read_bytes()
receipt=dict(status='actual_epoch05_CDTB_primary_input_local_copies_verified',completed_at=datetime.now(timezone.utc).isoformat(),archive=str(archive),archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),local_directory=str(local),primary_files_checked=325,all_primary_hashes_match_previous_captured_diagnosis=True,first_parent_exit_code=1,first_parent_exit_scope='SSH packaging completed before Windows path-length error during local extraction; no independent subprocess completion transcript for SSH alone',original_native_session=18079,original_invoke_chunk='28d795',original_auth_chunk='3248d4',original_parent_terminal_chunk='050442',new_SSH_calls=0,new_NN_calls=0,new_GPU_queries=0,new_training_progress_queries=0,scope='Primary byte copies only; no independent IoU/PR recomputation, historical runtime attestation or checkpoint tensor verification. Original partial extraction preserved.')
(A/'M124_epoch05_CDTB_primary_inputs_archive_receipt_20261013.json').write_bytes((json.dumps(receipt,indent=2)+chr(10)).encode())
print(json.dumps(receipt,indent=2),flush=True)