"""Stdlib-only source/interface fixtures. All generated data are synthetic."""
import argparse, ast, base64, contextlib, csv, difflib, hashlib, io, json, math, os, shutil, sys, types, zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

OUT=Path(__file__).resolve().parent; BASE=OUT.parent
CAD=BASE/'M125_CAD_UNIFIED_SOURCE_R2_20261016'
OLD=BASE/'M125_CAD_UNIFIED_SOURCE_20261016'
R0=BASE/'M125_R0_FIXED_SOURCE_20261015'; SNAP=R0/'frozen_dependency_snapshots'
OP=BASE/'M125_REGION_DEPLOYMENT_SOURCE_R2_20261015/deploy_m125_region_fixed.py'
INVOKER=BASE/'M125_CAD_DEPLOYMENT_SOURCE_20261016/invoke_M125_CAD_reviewed_suite.py'
FIX=OUT/'fixtures'; FIX.mkdir()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_bytes((json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8'))
checks=[]
def check(name,value,**extra):
    checks.append(dict(name=name,passed=bool(value),**extra)); assert value,name
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8-sig'),filename=str(p))
def fn(p,name):return next(n for n in tree(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def functions(p,names,scope):
    nodes=[n for n in tree(p).body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),scope)
    return scope

inputs=read(OUT/'input_hashes_before.json')['files']+read(OUT/'additional_dependency_hashes_before.json')['files']
for row in inputs:check('input_seal:'+row['path'],sha(row['path'])==row['sha256'])
candidate=read(CAD/'CANDIDATE.json'); old_candidate=read(OLD/'CANDIDATE.json')
layout=read(BASE/'M125_CAD_R2_final66_actual_CPU_layout_20261016.json')
revision=read(BASE/'M125_CAD_R2_initial_source_revision_20261016.json')
spec=read(CAD/'SOURCE_SPEC.json'); manifest=read(CAD/'INPUT_MANIFEST.json')
sources=list(CAD.glob('*.py'))+list(R0.glob('*.py'))+list(SNAP.glob('*.py'))+[OP]
stage={p.name:p for p in sources}; imported={PurePosixPath(p).name:d for p,d in layout['source_sha256'].items()}
check('stage_66_no_collisions',len(stage)==len(sources)==66)
check('receipt_names_exact',set(stage)==set(imported))
future={PurePosixPath(p).name:d for p,d in candidate['source_sha256'].items() if PurePosixPath(p).parent==PurePosixPath(candidate['code_root'])}
check('candidate_and_receipt_66_exact',future==imported)
for n,p in stage.items():
    check('stage_hash:'+n,sha(p)==imported[n]);tree(p);check('AST:'+n,True)
for p,d in manifest.items():check('manifest:'+Path(p).name,sha(p)==d)
for n,d in layout['locked_inputs_sha256'].items():check('locked_input:'+n,sha(CAD/n)==d)
for n,d in spec['source_sha256'].items():check('source_spec:'+n,sha(CAD/n)==d)
check('receipt_12_zero_main_zero_real',len(layout['rows'])==12 and all(r['exit_code']==0 for r in layout['rows']) and layout['entrypoint_main_executions']==layout['real_tracking_calls']==0)
for row in layout['rows']:
    check('origin_json:'+row['source'],json.loads(row['stdout'])==row['verified_packaged_module_origins'])
    for m,p in row['verified_packaged_module_origins'].items():check('origin:'+row['source']+':'+m,p==layout['root']+'/code/'+m+'.py' and m+'.py' in stage)
diff=[];changed=[]
for p in sorted(CAD.glob('*.py')):
    new_text=p.read_text(encoding='utf-8');old_text=(OLD/p.name).read_text(encoding='utf-8')
    if new_text!=old_text:changed.append(p.name)
    diff.extend(difflib.unified_diff(old_text.splitlines(True),new_text.splitlines(True),fromfile='original/'+p.name,tofile='R2/'+p.name))
check('exact_two_source_changes',changed==['collect_m125_CAD_results.py','prepare_m125_CAD_evaluation.py'])
for row in revision['files']:
    check('revision_record:'+row['source'],sha(OLD/row['source'])==row['original_sha256'] and sha(CAD/row['source'])==row['revised_sha256'] and row['changed']==(row['source'] in changed))
(OUT/'direct_R2_diffs.patch').write_bytes(''.join(diff).encode())
candidate_changes=[p for p,d in candidate['source_sha256'].items() if old_candidate['source_sha256'][p]!=d]
check('candidate_changes_only_two_future_digests',len(candidate_changes)==2 and {PurePosixPath(p).name for p in candidate_changes}==set(changed))
check('candidate_non_source_fields_unchanged',{k:v for k,v in candidate.items() if k!='source_sha256'}=={k:v for k,v in old_candidate.items() if k!='source_sha256'})
check('R2_repeated_hash_disclosure',spec['inherited_baseline_plan_checks_rehash_assets_per_invocation'] is True and 'one_time_asset_hashing_before_jobs' not in spec and 'no_rehash_of_unused_training_warm_per_VOT_anchor' not in spec)
tree(INVOKER);check('invoker_AST',True)

# Load only the pure profile module. No torch/numpy/project-module imports.
policy=types.ModuleType('cad_template_profiles');policy.__file__=str(CAD/'cad_template_profiles.py')
sys.modules[policy.__name__]=policy
exec(compile((CAD/'cad_template_profiles.py').read_text(),policy.__file__,'exec'),policy.__dict__)

# Complete unchanged prepare main plus actual CAD bundle/plan functions over fixture R0 metadata.
reference=FIX/'reference';reference.mkdir(); training=FIX/'training';training.mkdir()
save(training/'result.json',dict(evaluation_type='simulation_only',fixture=True));(training/'final.pt').write_bytes(b'SYNTHETIC FIXTURE NOT A CHECKPOINT')
reference_bundle=dict(schema='M125_R0_fixed6_Train152_official_v1',final_path=str(training/'final.pt'),final_sha256=sha(training/'final.pt'),
    training_result_path=str(training/'result.json'),training_result_sha256=sha(training/'result.json'))
save(reference/'bundle.json',reference_bundle)
for dataset in ['depthtrack_test','cdtb','vot']:
    folder=reference/dataset;folder.mkdir()
    plan=dict(dataset=dataset,bundle_path=str(reference/'bundle.json'),bundle_sha256=sha(reference/'bundle.json'),
              bank_sha256='fixture-bank',binding_sha256='fixture-binding')
    if dataset!='vot':plan['output']=str(folder/'predictions')
    save(folder/'plan.json',plan)
run=reference/'vot/run';run.mkdir();shards=[];sealed={}
for i in range(4):
    p=run/('shard-%02d'%i);(p/'sequences').mkdir(parents=True)
    (p/'sequences/list.txt').write_bytes(b'fixture_sequence\n');(p/'sequences/anchor.value').write_bytes(b'fixture-anchor\n')
    (p/'config.yaml').write_bytes(b'stack: vot2022/rgbd\n')
    sealed[str(p/'sequences/anchor.value')]=sha(p/'sequences/anchor.value')
    shards.append(dict(root=str(p),index=i,gpu=i%2,anchor_count=441 if i<3 else 442,expected_trajectories=['fixture']))
anchor_root=FIX/'anchor_source';anchor_root.mkdir();(anchor_root/'anchor.value').write_bytes(b'fixture\n')
manifest_fixture=dict(schema='M122_full127_frozen_shards_v1',sequences=['fixture_%03d'%i for i in range(127)],
    total_anchor_count=1765,shards=shards,source={'fixture':dict(root=str(anchor_root),anchor_sha256=sha(anchor_root/'anchor.value'))})
save(run/'shard_manifest.json',manifest_fixture);save(reference/'vot/execution.json',dict(source_sha256=sealed))
gate_path=FIX/'source_gate.json';gate=dict(status='reviewed_M125_CAD_uniform_inference_source_gate',profiles=['U0','U1','U2'],
    per_dataset_profile_selection=False,source_sha256={str(CAD/'m125_CAD_official_runtime.py'):sha(CAD/'m125_CAD_official_runtime.py')},base_code_root=str(R0))
save(gate_path,gate)
class FutureClock:
    @staticmethod
    def now(tz):return datetime(2026,10,18,tzinfo=timezone.utc)
    fromisoformat=staticmethod(datetime.fromisoformat)
def fixture_R0_checked(p):
    plan=read(p);bundle=read(plan['bundle_path']);bundle['base_bundle']={'fixture':True};return plan,bundle
runtime_scope=functions(CAD/'m125_CAD_official_runtime.py',['make_bundle','checked_plan'],dict(datetime=FutureClock,timezone=timezone,
    Path=Path,json=json,sha=sha,profile_dict=policy.profile_dict,SCHEMA='M125_CAD_same_R0_final_uniform_profile_v1',
    checked_R0_plan=fixture_R0_checked,verify_predecessor=lambda gate:dict(root=str(FIX),final_sha256=reference_bundle['final_sha256'])))
# The bound root is FIX/evaluation by the real make_bundle contract.
reference.rename(FIX/'evaluation');reference=FIX/'evaluation'
# Rebind only synthetic path strings after moving the fixture reference tree.
for p in reference.rglob('*.json'):
    p.write_bytes(p.read_bytes().replace(str(FIX/'reference').encode().replace(b'\\',b'\\\\'),str(reference).encode().replace(b'\\',b'\\\\')))
for d in ['depthtrack_test','cdtb','vot']:
    plan=read(reference/d/'plan.json');plan['bundle_sha256']=sha(reference/'bundle.json');save(reference/d/'plan.json',plan)
prep_scope=functions(CAD/'prepare_m125_CAD_evaluation.py',['main'],dict(Path=Path,json=json,shutil=shutil,sha=sha,read=read,write=save,
    checked_R0_plan=fixture_R0_checked,make_bundle=runtime_scope['make_bundle'],checked_plan=runtime_scope['checked_plan'],
    __file__=str(CAD/'prepare_m125_CAD_evaluation.py')))
prepared={}
for profile in ['U0','U1','U2']:
    output=FIX/('prepared_'+profile)
    with contextlib.redirect_stdout(io.StringIO()):prep_scope['main'](argparse.Namespace(source_gate=gate_path,reference_evaluation=reference,profile=profile,output=output))
    selection=read(output/'selection.json');prepared[profile]=output
    check('prepare_full_main:'+profile,selection['training_result_sha256']==sha(training/'result.json') and selection['inference_profile_id']==profile and len(selection['plans'])==3)
    for dataset in ['depthtrack_test','cdtb','vot']:
        plan,bundle=runtime_scope['checked_plan'](output/dataset/'plan.json')
        check('bundle_plan_exact:'+profile+':'+dataset,bundle['final_sha256']==reference_bundle['final_sha256'] and bundle['inference_profile']==policy.profile_dict(profile))
    check('prepare_shard_config:'+profile,all('restart = false' in (output/'vot/run'/('shard-%02d'%i)/'trackers.ini').read_text() for i in range(4)))

# Complete collector main over in-memory synthetic receipts, with actual CSV/result output in FIX.
collection_results={}
for profile in ['U1','U2']:
    root=prepared[profile];selection=read(root/'selection.json');bundle=read(root/'bundle.json');model=selection['models'][0]
    digest='a'*64;objects={str(root/'selection.json'):selection,str(root/'bundle.json'):bundle}
    for dataset,count,frames in [('depthtrack_test',50,76373),('cdtb',80,101956)]:
        folder=root/dataset/'predictions';plan=model['plans'][dataset]
        rows=[dict(sequence='synthetic_%03d'%i,frames=frames//count+(i<frames%count),inference_profile_id=profile,
            template_counts=dict(tracking_calls=frames//count+(i<frames%count)-1),bbox_sha256=digest,confidence_sha256=digest) for i in range(count)]
        objects[str(folder/'receipt.json')]=dict(inference_profile_id=profile,inference_profile=bundle['inference_profile'],one_profile_for_all_datasets=True,
            bank_sha256=selection['bank_sha256'],sequences=rows,frames=frames,frozen_before_after_exact=True,optimizer_steps=0,subsequent_GT_opened=False)
        objects[str(folder/'metrics.json')]=dict(status='complete',final_sha256=model['final_sha256'],bundle_sha256=model['bundle_sha256'],
            inference_profile_id=profile,inference_profile=bundle['inference_profile'],plan_sha256=plan['sha256'],receipt_sha256=digest,
            metrics=dict(sequences=count,frames=frames,precision_percent=50.,recall_percent=50.,f_score_percent=50.))
    objects[str(root/'vot/result.json')]=dict(status='complete_full127',final_sha256=model['final_sha256'],bundle_sha256=model['bundle_sha256'],
        bank_sha256=selection['bank_sha256'],binding_sha256=selection['binding_sha256'],anchors=1765,sequences=127,external_optimizer_steps=0,
        inference_profile_id=profile,inference_profile=bundle['inference_profile'],profile_event_files_sha256={('%064x.json'%i):digest for i in range(1765)},
        merge_sha256=digest,analysis_sha256=digest,metrics_percent=dict(EAO=50.,ACC=50.,ROB=50.),confirmed_failures=0)
    objects[str(root/'vot/run/merge_result.json')]=dict(result_sha256={})
    expected=root/'vot/run/master/analysis'/(model['name']+'_full127.json');analysis_requests=[]
    def fixture_sha(p):
        p=Path(p)
        if p.parent.name=='analysis':analysis_requests.append(str(p));assert p==expected
        return sha(p) if p.is_file() else digest
    scope=functions(CAD/'collect_m125_CAD_results.py',['main'],dict(Path=Path,math=math,csv=csv,sha=fixture_sha,
        read=lambda p:objects[str(p)],write=save))
    with contextlib.redirect_stdout(io.StringIO()):scope['main'](root)
    result=read(root/'all_results.json')
    check('collector_complete_main:'+profile,len(result['results'])==3 and len(result['gates'])==1 and all(r['model']==model['name'] for r in result['results']) and result['gates'][0]['model']==model['name'])
    check('collector_analysis_identity:'+profile,analysis_requests==[str(expected)])
    check('collector_CSV_identity:'+profile,list(csv.DictReader((root/'metrics.csv').open()))[0]['model']==model['name'])
    collection_results[profile]=dict(result_path=str(root/'all_results.json'),analysis_requested=analysis_requests,fixture=True,evaluation_type='simulation_only')

# Exact time assertions, including the embedded remote program, without any transport.
inv_tree=tree(INVOKER);inv_main=fn(INVOKER,'main')
remote_assign=next(n for n in inv_main.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='remote')
remote_template=remote_assign.value.func.value.value
remote_tree=ast.parse(remote_template)
(OUT/'embedded_remote_source.txt').write_bytes(remote_template.encode())
check('embedded_remote_AST',True)
local_time=inv_main.body[0];remote_time=next(n for n in remote_tree.body if isinstance(n,ast.Assert))
class BoundaryClock:
    value=datetime.fromisoformat('2026-10-17T12:44:59+08:00')
    @classmethod
    def now(cls,tz):return cls.value
    fromisoformat=staticmethod(datetime.fromisoformat)
for side,node in [('local',local_time),('remote',remote_time)]:
    for permitted,timestamp in [(False,'2026-10-17T12:44:59+08:00'),(True,'2026-10-17T12:45:00+08:00')]:
        BoundaryClock.value=datetime.fromisoformat(timestamp)
        try:exec(compile(ast.Module(body=[node],type_ignores=[]),str(INVOKER),'exec'),dict(datetime=BoundaryClock,timezone=timezone));accepted=True
        except AssertionError:accepted=False
        check('time_boundary:'+side+':'+str(permitted),accepted==permitted)
check('local_time_before_file_reads',isinstance(inv_main.body[0],ast.Assert))
remote_effects=[n for n in remote_tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
check('remote_time_before_payload_and_file_reads',remote_effects[0] is remote_time)
local_asserts={n.lineno:n for n in inv_main.body if isinstance(n,ast.Assert)}
remote_asserts={n.lineno:n for n in remote_tree.body if isinstance(n,ast.Assert)}
check('local_attempt_is_exclusive',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='open' and n.lineno==51 and n.args[0].value=='x' for n in ast.walk(inv_main)))
check('remote_root_mkdir_before_Popen',remote_template.index('root.mkdir();code.mkdir()')<remote_template.index('child=subprocess.Popen('))
check('predecessor_then_resource_then_root',remote_template.index('predecessor=verify_predecessor(candidate)')<remote_template.index('gpu_rows=resource_admission(candidate)')<remote_template.index('root.mkdir();code.mkdir()'))
check('no_invoker_retry_loop',not any(isinstance(n,ast.While) for n in ast.walk(inv_main)) and not any(isinstance(n,ast.While) for n in ast.walk(remote_tree)))
check('remote_launches_CAD_supervisor',"str(code/'m125_CAD_process_supervisor.py')" in remote_template and "PYTHONOPTIMIZE='0'" in remote_template)
check('launch_receipt_does_not_claim_completion',"real64_complete_at_launch=False,formal_metrics_available_at_launch=False" in remote_template)

# Full supervisor main with one stubbed subprocess and preserved true terminal codes.
supervisor_results=[]
for code in [0,7]:
    root=FIX/('supervisor_exit_'+str(code));root.mkdir();save(root/'gate.json',dict(python='fixture-python',code_root=str(CAD)))
    calls=[]
    def fake_run(*args,**kwargs):calls.append([args,kwargs]);return types.SimpleNamespace(returncode=code)
    scope=functions(CAD/'m125_CAD_process_supervisor.py',['main'],dict(Path=Path,json=json,datetime=FutureClock,timezone=timezone,os=types.SimpleNamespace(getpid=lambda:123),
        sys=types.SimpleNamespace(argv=['fixture-supervisor',str(root)]),subprocess=types.SimpleNamespace(run=fake_run)))
    with contextlib.redirect_stdout(io.StringIO()):returned=scope['main']()
    terminal=read(root/'controller.supervisor.terminal.json')
    check('supervisor_exit_preserved:'+str(code),returned==code and terminal['exit_code']==code and len(calls)==1 and (root/'controller.supervised.exit').read_text().strip()==str(code))
    supervisor_results.append(dict(exit_code=code,child_calls=len(calls),fixture=True))

# Source ordering evidence and current frozen prerequisites are read, never executed.
op_text=OP.read_text();check('frozen_predecessor_full6_and_full3',all(s in op_text for s in ["range(1, 7)","len(result['stages']) == 15", "len(combined['results']) == 3", "{'depthtrack_test', 'cdtb', 'vot'}"]))
check('resource_memory_only_and_fixed_two',"--query-gpu=index,memory.used,memory.total" in op_text and "[row[0] for row in rows] == [0, 1]" in op_text)
observer=BASE/'M124_fiftyfirst_hourly_compressed_native_observer_20261014.py'
obs_tree=tree(observer)
args=ast.literal_eval(next(n.value for n in obs_tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='args'))
check('transport_prefix_single_ssh_attempt',args[0].endswith('/ssh.exe') and 'ConnectionAttempts=1' in args and isinstance(args[-1],str))
check('askpass_env_reference','RGBD_GOAL_SSH_PASSWORD' in (BASE/'rgbd_goal_resume_askpass_20261014.cmd').read_text())

save(OUT/'fixture_results.json',dict(evaluation_type='simulation_only',fixture=True,preparation_profiles=['U0','U1','U2'],
    complete_prepare_main_executions=3,complete_collector_main_executions=2,collection_results=collection_results,
    supervisor_results=supervisor_results,actual_neural_executions=0,actual_transport_calls=0,actual_GPU_queries=0))
save(OUT/'verified_staged_source_sha256.json',layout['source_sha256'])
save(OUT/'deterministic_checks.json',dict(checks=checks,total=len(checks),passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],
    input_files=len(inputs),candidate_sha256=sha(CAD/'CANDIDATE.json'),source_changes=changed,
    execution_type='stdlib_AST_and_synthetic_interfaces_only',actual_NN_calls=0,actual_transport_calls=0))
print(json.dumps(dict(total=len(checks),passed=sum(c['passed'] for c in checks),candidate_sha256=sha(CAD/'CANDIDATE.json'),
    changed=changed,prepare_mains=3,collector_mains=2,supervisor_stub_mains=2,actual_NN_calls=0,actual_transport_calls=0),indent=2))
