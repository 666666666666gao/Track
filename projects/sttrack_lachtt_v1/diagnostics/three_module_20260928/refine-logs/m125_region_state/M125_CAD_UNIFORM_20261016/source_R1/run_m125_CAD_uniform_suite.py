"""U0 original complete R0 reference, then complete uniform U1 and U2 inference."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import os
import subprocess
import sys

from bind_m122_official_initializations import sha
from cad_template_profiles import profile_dict
from deploy_m125_region_fixed import verify_predecessor,resource_admission


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(root,gate,name,python,arguments,gpu=None):
    environment=dict(os.environ,PYTHONPATH=os.pathsep.join([gate['code_root'],gate['base_code_root'],'/home/SUTrack_RGBD_L']),
        CUDA_VISIBLE_DEVICES='0,1' if gpu is None else str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[python,'-B','-u']+arguments
    row=dict(stage=name,command=command,gpu=gpu,started_at=datetime.now(timezone.utc).isoformat())
    write(root/(name+'.launch.json'),row)
    with (root/(name+'.log')).open('xb') as stream:
        completed=subprocess.run(command,cwd=gate['code_root'],env=environment,stdout=stream,stderr=subprocess.STDOUT)
    row.update(exit_code=completed.returncode,finished_at=datetime.now(timezone.utc).isoformat())
    (root/(name+'.exit')).write_text(str(completed.returncode)+'\n')
    write(root/(name+'.receipt.json'),row)
    assert completed.returncode==0,'Read the original failure; no automatic retry.'
    return row


def main():
    assert sys.flags.optimize==0
    assert datetime.now(timezone.utc)>=datetime.fromisoformat('2026-10-17T12:45:00+08:00')
    root=Path(sys.argv[1]);gate=read(root/'gate.json')
    assert gate['status']=='reviewed_M125_CAD_uniform_inference_source_gate'
    assert gate['profiles']==['U0','U1','U2'] and not gate['per_dataset_profile_selection']
    for path,digest in gate['source_sha256'].items():assert sha(path)==digest,path
    for path,digest in gate['input_sha256'].items():assert sha(path)==digest,path
    predecessor=verify_predecessor(gate)
    assert Path(gate['reference_evaluation'])==Path(predecessor['root'])/'evaluation'
    resource_rows=resource_admission(gate)
    write(root/'actual_resource_admission.json',dict(rows=resource_rows,fields='index,memory.used,memory.total',completed_predecessor=predecessor))
    code=Path(gate['code_root']);st=gate['python'];metric=gate['metric_python']
    reference=read(Path(gate['reference_evaluation'])/'all_results.json')
    assert reference['status']=='three_M125_R0_fixed6_full_evaluations_complete' and len(reference['gates'])==1
    assert reference['gates'][0]['final_sha256']==predecessor['final_sha256']
    stages=[];profiles=root/'profiles';profiles.mkdir();qualification=root/'qualification';qualification.mkdir()
    for name in gate['profiles']:
        evaluation=profiles/name
        stages.append(run(root,gate,'prepare_'+name,st,[str(code/'prepare_m125_CAD_evaluation.py'),
            '--reference-evaluation',gate['reference_evaluation'],'--source-gate',str(root/'gate.json'),
            '--profile',name,'--output',str(evaluation)]))
        stages.append(run(root,gate,'qualify_'+name,st,[str(code/'check_m125_CAD_real64.py'),
            '--plan',str(evaluation/'depthtrack_test/plan.json'),'--output',str(qualification/name)],0))
        witness=read(qualification/name/'result.json')
        assert witness['status']=='complete_M125_CAD_real64_inference_interface_witness'
        assert witness['inference_profile_id']==name and witness['inference_profile']==profile_dict(name)
        assert witness['track_calls']==64 and witness['final_sha256']==predecessor['final_sha256']
        assert witness['frozen_before_after_exact'] and witness['A_B_C_states_unchanged']
        assert witness['optimizer_steps']==0 and not witness['subsequent_GT_opened'] and not witness['new_checkpoint_created']
        if name=='U0':assert witness['U0_original_prefix_parity']['all65_box_and_score_rows_match_original_six_decimal_files']
    rows=[dict(inference_profile_id='U0',inference_profile=profile_dict('U0'),
        execution_origin='original complete fixed6 R0 suite; not a new full U0 runtime run',
        source_result_path=str(Path(gate['reference_evaluation'])/'all_results.json'),
        source_result_sha256=sha(Path(gate['reference_evaluation'])/'all_results.json'),
        nine_metric_gate=reference['gates'][0],new_runtime_real64_prefix_result_sha256=sha(qualification/'U0/result.json'))]
    for name in ['U1','U2']:
        resource_admission(gate)
        evaluation=profiles/name
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(run,root,gate,name+'_'+dataset+'_track',st,
                [str(code/'run_m125_CAD_official.py'),'--mode','ope_track','--plan',str(evaluation/dataset/'plan.json')],gpu)
                for gpu,dataset in enumerate(['depthtrack_test','cdtb'])]
            stages.extend(job.result() for job in jobs)
        for dataset in ['depthtrack_test','cdtb']:
            stages.append(run(root,gate,name+'_'+dataset+'_analyze',metric,
                [str(code/'run_m125_CAD_official.py'),'--mode','ope_analyze','--plan',str(evaluation/dataset/'plan.json')]))
        vot=evaluation/'vot';manifest=read(vot/'run/shard_manifest.json')
        stages.append(run(root,gate,name+'_vot_track',st,[str(code/'run_m123_joint_vot_shards.py'),'--root',str(vot/'run')]))
        (evaluation/'vot.exit').write_text('0\n')
        model_name=read(evaluation/'selection.json')['models'][0]['name']
        stages.append(run(root,gate,name+'_vot_official_analysis',metric,
            ['-m','vot','analysis','--workspace',str(vot/'run/master'),'--format','json','--name',model_name+'_full127',manifest['tracker']]))
        stages.append(run(root,gate,name+'_vot_seal',metric,[str(code/'analyze_m125_CAD_vot.py'),'--root',str(evaluation)]))
        stages.append(run(root,gate,name+'_collect',metric,[str(code/'collect_m125_CAD_results.py'),'--root',str(evaluation)]))
        result=read(evaluation/'all_results.json')
        assert result['status']=='three_M125_CAD_uniform_profile_full_evaluations_complete'
        assert result['inference_profile_id']==name and result['inference_profile']==profile_dict(name)
        assert result['one_profile_for_all_datasets'] and not result['per_dataset_profile_selection']
        assert len(result['gates'])==1 and result['gates'][0]['final_sha256']==predecessor['final_sha256']
        rows.append(dict(inference_profile_id=name,inference_profile=profile_dict(name),execution_origin='new complete three-dataset inference',
            source_result_path=str(evaluation/'all_results.json'),source_result_sha256=sha(evaluation/'all_results.json'),
            nine_metric_gate=result['gates'][0],new_runtime_real64_prefix_result_sha256=sha(qualification/name/'result.json')))
    assert len(rows)==3 and {row['nine_metric_gate']['final_sha256'] for row in rows}=={predecessor['final_sha256']}
    result=dict(status='complete_M125_CAD_same_weight_uniform_profile_comparison',final_sha256=predecessor['final_sha256'],
        rows=rows,stages=stages,source_gate_sha256=sha(root/'gate.json'),automatic_retries=0,
        new_training=False,per_dataset_profile_selection=False,one_profile_per_complete_nine_metric_row=True,
        inference_strategy_engineering_comparison=True,independent_completed_result_audit=False,
        any_joint_pass=any(row['nine_metric_gate']['joint_pass'] for row in rows),finished_at=datetime.now(timezone.utc).isoformat())
    write(root/'result.json',result)
    with (root/'comparison_metrics.csv').open('x',newline='') as stream:
        fields=['profile','final_sha256','DepthTrack_P','DepthTrack_R','DepthTrack_F','CDTB_P','CDTB_R','CDTB_F','VOT_EAO','VOT_ACC','VOT_ROB','joint_pass']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            gate_row=row['nine_metric_gate'];m=gate_row['metrics_percent']
            writer.writerow(dict(profile=row['inference_profile_id'],final_sha256=gate_row['final_sha256'],
                DepthTrack_P=m['depthtrack_test']['precision'],DepthTrack_R=m['depthtrack_test']['recall'],DepthTrack_F=m['depthtrack_test']['f_score'],
                CDTB_P=m['cdtb']['precision'],CDTB_R=m['cdtb']['recall'],CDTB_F=m['cdtb']['f_score'],
                VOT_EAO=m['vot']['EAO'],VOT_ACC=m['vot']['ACC'],VOT_ROB=m['vot']['ROB'],joint_pass=gate_row['joint_pass']))
    (root/'controller.exit').write_text('0\n')


if __name__=='__main__':main()
