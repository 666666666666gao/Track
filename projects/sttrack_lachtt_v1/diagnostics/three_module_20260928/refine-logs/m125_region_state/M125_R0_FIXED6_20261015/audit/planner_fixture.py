"""Run actual preparation/check functions on synthetic local metadata, never data/NN/evaluation."""
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

OUT=Path(__file__).resolve().parent
BASE=OUT.parent
SRC=BASE/'M125_R0_FIXED_SOURCE_20261015'
DEP=SRC/'frozen_dependency_snapshots'
ROOT=OUT/'SYNTHETIC_CONTROL_FLOW_NO_NN/planner'
ROOT.mkdir(exist_ok=False)
checks=[]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,data): Path(path).write_text(json.dumps(dict(data,fixture_only=True,NN_calls=0),indent=2)+'\n')
def check(name,actual,expected=True): checks.append(dict(name=name,status='PASS' if actual==expected else 'FAIL',actual=actual,expected=expected))
def functions(path,env):
    tree=ast.parse(path.read_text())
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),str(path),'exec'),env)
    return env


base=ROOT/'base';base.mkdir()
training=ROOT/'training';training.mkdir()
output=ROOT/'prepared'
basefinal=base/'NOT_MODEL_WEIGHTS.txt';basefinal.write_text('fixture baseline; no model weights\n')
final=training/'final.pt';final.write_text('fixture only; no actual tensor/checkpoint/training\n')
base_bundle=dict(schema='M122_full152_final_official_v1',final_sha256=sha(basefinal),precision_weight=1)
trained=dict(status='complete_M125_R0_fixed_Train152_pass',seed=2027,epochs=6,fixed_budget_passes=6,fixed_final_pass=6,
    training_sequences=152,sequence_runs=912,track_calls=1318812,parameters=450888,
    optimizer_steps=1,C_optimizer_steps=1,no_development_split=True,no_cached_feature_fit=True,
    frozen_before_after_exact=True,final_state_roundtrip_exact=True,final_sha256=sha(final),
    external_metric_checkpoint_selection=False,fixed_final_after_complete_budget=True,fresh_common_warm_start=True,
    first_pass_optimizer_fresh=True,exact_seeded_training_start=True,no_external_test_cdtb_vot_gradient_training=True,
    Test50_CDTB80_used_for_checkpoint_selection=False)
write(training/'result.json',trained)
source_map={str(p):sha(p) for p in list(SRC.glob('*.py'))+list(DEP.glob('*.py'))}
gate=dict(status='reviewed_M125_R0_fixed6_source_gate',source_sha256=source_map,base_code_root=str(DEP))
write(ROOT/'gate.json',gate)

base_plans={}
for dataset in ['depthtrack_test','cdtb','vot']:
    folder=base/dataset;folder.mkdir()
    plan=dict(dataset=dataset,bank_path='SYNTHETIC_NO_BANK',bank_sha256='fixture-bank',
        binding_path='SYNTHETIC_NO_BINDING',binding_sha256='fixture-binding',
        cases_path='SYNTHETIC_NO_CASES',cases_sha256='fixture-cases',dataset_root='SYNTHETIC_NO_DATA',
        metric_source='SYNTHETIC_NEVER_EXECUTED_METRIC',metric_source_sha256='fixture-metric')
    write(folder/'plan.json',plan)
    base_plans[str(folder/'plan.json')]=read(folder/'plan.json')

index=read(OUT/'metric_operator_checks.json')['inherited_sources']
old=next(v for k,v in index.items() if k.endswith('/shard_manifest.json'))
manifest=read(old['local_snapshot'])
source_trajectories=[list(s['expected_trajectories']) for s in manifest['shards']]
oldrun=base/'vot/run';oldrun.mkdir()
old_execution=dict(source_sha256={})
for i,info in enumerate(manifest['source'].values()):
    anchor=ROOT/'anchor_metadata'/str(i)/'anchor.value';anchor.parent.mkdir(parents=True)
    anchor.write_text('SYNTHETIC anchor file; copied/hash checked only\n')
    info['root']=str(anchor.parent);info['anchor_sha256']=sha(anchor)
for shard in manifest['shards']:
    oldshard=oldrun/('shard-%02d'%shard['index']);(oldshard/'sequences').mkdir(parents=True)
    listing=oldshard/'sequences/list.txt';listing.write_text('SYNTHETIC metadata only\n')
    (oldshard/'config.yaml').write_text('registry:\n- ./trackers.ini\nsequences: ./sequences\nstack: vot2022/rgbd\n')
    shard['root']=str(oldshard)
    old_execution['source_sha256'][str(listing)]=sha(listing)
write(oldrun/'shard_manifest.json',manifest)
write(base/'vot/execution.json',old_execution)


def checked_base(path):
    return base_plans[str(path)],base_bundle


runtime=functions(SRC/'m125_R0_fixed_official_runtime.py',dict(json=json,Path=Path,sha=sha,checked_base_plan=checked_base))
prepare=functions(SRC/'prepare_m125_R0_fixed_evaluation.py',dict(json=json,Path=Path,sha=sha,shutil=shutil,
    checked_base_plan=checked_base,checked_plan=runtime['checked_plan'],read=read,write=write,
    __file__=str(SRC/'prepare_m125_R0_fixed_evaluation.py')))
prepare['main'](SimpleNamespace(source_gate=ROOT/'gate.json',training=training,base_evaluation=base,output=output))
bundle=read(output/'bundle.json');selection=read(output/'selection.json')
check('prepare_single_final_model',len(selection['models']),1)
check('prepare_final_fixed_pass_six',selection['fixed_final_pass'],6)
check('prepare_all_three_datasets',sorted(selection['plans']),['cdtb','depthtrack_test','vot'])
check('prepare_all_53_source_bindings',bundle['source_sha256'],source_map)
check('prepare_final_hash_bound',bundle['final_sha256'],sha(final))
check('prepare_no_external_selection',bundle['external_metric_checkpoint_selection'],False)
for dataset in ['depthtrack_test','cdtb','vot']:
    plan,checked=runtime['checked_plan'](output/dataset/'plan.json')
    check('runtime_checked_plan_succeeds:'+dataset,checked['final_sha256'],sha(final))
    check('runtime_same_legal_initialization_bank:'+dataset,plan['bank_sha256'],'fixture-bank')
wrapper=output/'vot/selected_m125_R0_fixed_vot.py'
tree=ast.parse(wrapper.read_text())
check('wrapper_imports_only_R0_runtime',[(n.module,[a.name for a in n.names]) for n in tree.body if isinstance(n,ast.ImportFrom)],
      [('run_m125_R0_fixed_official',['vot_track'])])
check('wrapper_invokes_bound_R0_plan',ast.literal_eval(tree.body[-1].value.args[0]),str(output/'vot/plan.json'))
prepared_manifest=read(output/'vot/run/shard_manifest.json')
check('prepare_preserves_all_1765_trajectory_names',[list(s['expected_trajectories']) for s in prepared_manifest['shards']],source_trajectories)
check('prepare_preserves_all_127_sequences',prepared_manifest['sequences'],manifest['sequences'])
check('prepare_new_tracker_identity',prepared_manifest['tracker'],'sttrack_m125_R0_fixed6')
check('prepare_manifest_source_binding',prepared_manifest['source_manifest_sha256'],sha(oldrun/'shard_manifest.json'))
for row in prepared_manifest['shards']:
    target=Path(row['root']);ini=(target/'trackers.ini').read_text()
    check('prepare_isolated_shard:'+str(row['index']),target.parent,output/'vot/run')
    check('prepare_ini_no_restart:'+str(row['index']),'restart = false\n' in ini)
    check('prepare_ini_correct_wrapper:'+str(row['index']),'command = selected_m125_R0_fixed_vot\n' in ini)
execution=read(output/'vot/execution.json')
check('prepare_all_execution_hashes_valid',all(sha(path)==digest for path,digest in execution['source_sha256'].items()))
check('prepare_execution_declares_full_counts',[execution[k] for k in ['anchors','sequences','result_files']], [1765,127,5295])

# Rebind only synthetic metadata hashes to demonstrate the semantic final-six guard, not merely SHA rejection.
trained['epochs']=5;write(training/'result.json',trained)
bundle['training_result_sha256']=sha(training/'result.json');write(output/'bundle.json',bundle)
path=output/'vot/plan.json';plan=read(path);plan['bundle_sha256']=sha(output/'bundle.json');write(path,plan)
rejected=False
try:
    runtime['checked_plan'](path)
except AssertionError:
    rejected=True
check('runtime_rejects_rehashed_non_final_pass5',rejected)
# The tampered synthetic files are intentionally retained as negative-test evidence.
result=dict(scope='Actual preparation/check/wrapper logic with synthetic metadata and fake upstream base-plan verifier; no GT, checkpoints, torch, NN, subprocesses or metrics executed',
    checks=checks,count=len(checks),failed=sum(r['status']!='PASS' for r in checks))
(OUT/'planner_checks.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(dict(checks=len(checks),failures=[r['name'] for r in checks if r['status']!='PASS'])))
