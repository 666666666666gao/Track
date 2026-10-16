"""One reviewed CAD-suite admission; no calls before the original R0 suite ends."""
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import ast
import base64
import hashlib
import json
import os
import subprocess
import zlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert datetime.now(timezone.utc) >= datetime.fromisoformat('2026-10-17T12:45:00+08:00')
    A = Path(__file__).resolve().parent.parent
    source = A / 'M125_CAD_UNIFIED_SOURCE_R2_20261016'
    audit = A / 'M125_CAD_UNIFORM_SOURCE_AUDIT_R2_20261016'
    review = json.loads((audit / 'EXPERIMENT_AUDIT.json').read_bytes())
    candidate = json.loads((source / 'CANDIDATE.json').read_bytes())
    layout = json.loads((A / 'M125_CAD_R2_final66_actual_CPU_layout_20261016.json').read_bytes())
    assert review['source_verdict'] == 'PASS_SOURCE_ONLY' and review['blocking_findings'] == []
    assert review['candidate_sha256'] == sha(source / 'CANDIDATE.json')
    assert review['audited_input_hashes'][str(Path(__file__).resolve())] == 'sha256:' + sha(__file__)
    assert layout['code_files'] == 66 and layout['CAD_entrypoints'] == 12
    assert len(layout['rows']) == 12 and all(row['exit_code'] == 0 for row in layout['rows'])
    assert review['audited_staged_source_sha256'] == layout['source_sha256']
    manifest = json.loads((source / 'INPUT_MANIFEST.json').read_bytes())
    for path, digest in manifest.items():
        assert sha(path) == digest, path
        assert review['audited_input_hashes'][path] == 'sha256:' + digest, path
    for name, digest in layout['locked_inputs_sha256'].items():
        assert sha(source / name) == digest, name
    future = {PurePosixPath(path).name: digest for path, digest in candidate['source_sha256'].items()
              if PurePosixPath(path).parent == PurePosixPath(candidate['code_root'])}
    assert len(future) == 66
    assert future == {PurePosixPath(path).name: digest for path, digest in layout['source_sha256'].items()}
    request = dict(staging_root=layout['root'], code_sha256=layout['source_sha256'],
                   locked_inputs_sha256=layout['locked_inputs_sha256'],
                   source_review_json=(audit / 'EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'),
                   source_review_md=(audit / 'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8'),
                   source_review_sha256=sha(audit / 'EXPERIMENT_AUDIT.json'),
                   candidate_sha256=sha(source / 'CANDIDATE.json'), invocation_source_sha256=sha(__file__))
    stem = 'M125_CAD_uniform_suite_actual_invocation_20261016'
    attempt = dict(status='one_remote_CAD_suite_attempt_started', started_at=datetime.now(timezone.utc).isoformat(),
                   automatic_retries=0, source_review_sha256=request['source_review_sha256'],
                   invocation_source_sha256=request['invocation_source_sha256'])
    with (A / (stem + '_attempt.json')).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(attempt, indent=2) + '\n')
    remote = '''from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,os,subprocess,sys,zlib
assert datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-17T12:45:00+08:00')
p=json.loads(zlib.decompress(base64.b64decode('PAYLOAD')))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
staging=Path(p['staging_root']);inputs=staging/'locked_inputs'
for path,digest in p['code_sha256'].items():assert sha(path)==digest,path
for name,digest in p['locked_inputs_sha256'].items():assert sha(inputs/name)==digest,name
assert hashlib.sha256(p['source_review_json'].encode()).hexdigest()==p['source_review_sha256']
candidate=json.loads((inputs/'CANDIDATE.json').read_bytes())
review=json.loads(p['source_review_json'])
assert review['source_verdict']=='PASS_SOURCE_ONLY' and review['blocking_findings']==[]
assert review['candidate_sha256']==sha(inputs/'CANDIDATE.json')==p['candidate_sha256']
assert review['audited_staged_source_sha256']==p['code_sha256']
root=Path(candidate['root']);code=Path(candidate['code_root'])
assert root==Path('/root/autodl-tmp/m125_CAD_same_R0_uniform_profiles_20261016') and code==root/'code'
assert candidate['profiles']==['U0','U1','U2'] and not candidate['per_dataset_profile_selection']
assert candidate['same_R0_final_for_all_profiles'] and not candidate['new_training']
future={Path(path).name:digest for path,digest in candidate['source_sha256'].items() if Path(path).parent==code}
assert len(future)==66 and future=={Path(path).name:digest for path,digest in p['code_sha256'].items()}
for path,digest in candidate['source_sha256'].items():
    if Path(path).parent!=code:assert sha(path)==digest,path
for path,digest in candidate['input_sha256'].items():assert sha(path)==digest,path
sys.path.insert(0,str(staging/'code'))
from deploy_m125_region_fixed import verify_predecessor,resource_admission
predecessor=verify_predecessor(candidate)
gpu_rows=resource_admission(candidate)
root.mkdir();code.mkdir()
for name,digest in future.items():
    with (code/name).open('xb') as stream:stream.write((staging/'code'/name).read_bytes())
    assert sha(code/name)==digest,name
for name,data in [('source_review.json',p['source_review_json'].encode()),('source_review.md',p['source_review_md'].encode()),
                  ('candidate_gate.json',(inputs/'CANDIDATE.json').read_bytes())]:
    with (root/name).open('xb') as stream:stream.write(data)
gate=dict(candidate,status='reviewed_M125_CAD_uniform_inference_source_gate',
    source_review_report_path=str(root/'source_review.json'),source_review_report_sha256=p['source_review_sha256'],
    source_review_candidate_sha256=p['candidate_sha256'],admission_operator_sha256=p['invocation_source_sha256'])
with (root/'gate.json').open('x') as stream:stream.write(json.dumps(gate,indent=2)+chr(10))
command=[gate['python'],'-B','-u',str(code/'m125_CAD_process_supervisor.py'),str(root)]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONPATH=os.pathsep.join([str(code),'/home/SUTrack_RGBD_L']),
    PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (root/'controller.log').open('xb') as stream:
    child=subprocess.Popen(command,cwd=code,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='actual_M125_CAD_uniform_inference_pipeline_launched',root=str(root),
    started_at=datetime.now(timezone.utc).isoformat(),supervisor_pid=child.pid,command=command,
    gate_sha256=sha(root/'gate.json'),source_review_sha256=p['source_review_sha256'],
    admission_operator_sha256=p['invocation_source_sha256'],completed_predecessor=predecessor,
    GPU_memory_admission=gpu_rows,GPU_query_fields='index,memory.used,memory.total',automatic_retries=0,
    profiles=['U0','U1','U2'],new_training=False,real64_complete_at_launch=False,formal_metrics_available_at_launch=False)
with (root/'actual_launch.json').open('x') as stream:stream.write(json.dumps(receipt,indent=2)+chr(10))
print(json.dumps(receipt),flush=True)
'''.replace('PAYLOAD', base64.b64encode(zlib.compress(json.dumps(request).encode())).decode())
    tree = ast.parse((A / 'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py').read_text())
    args = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'args'))[:-1]
    environment = dict(os.environ, SSH_ASKPASS=str(A / 'rgbd_goal_resume_askpass_20261014.cmd'),
                       SSH_ASKPASS_REQUIRE='force', DISPLAY='codex')
    assert environment['RGBD_GOAL_SSH_PASSWORD']
    completed = subprocess.run(args + ["CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/envs/sttrack/bin/python -B -"],
                               input=remote.encode(), capture_output=True, env=environment)
    for suffix, data in [('_stdout.json', completed.stdout), ('_stderr.log', completed.stderr)]:
        with (A / (stem + suffix)).open('xb') as stream:
            stream.write(data)
    terminal = dict(attempt, status='one_remote_CAD_suite_attempt_terminal', remote_exit_code=completed.returncode,
                    finished_at=datetime.now(timezone.utc).isoformat())
    with (A / (stem + '_terminal.json')).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(terminal, indent=2) + '\n')
    assert completed.returncode == 0, completed.stderr.decode()
    print(completed.stdout.decode())


if __name__ == '__main__':
    main()
