"""Read-only preflight and completed-artifact audit for the frozen M80/M81 runs."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,math,sys
import numpy as np
import torch

B=Path('/root/autodl-tmp')
ROOTS={'m80':B/'sttrack_m80_block_text_dropout_20260908','m81':B/'sttrack_m81_multistart_20260908'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def model_hash(state):
    h=hashlib.sha256()
    for name,tensor in state.items():
        h.update(('semantic_adapter.'+name).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def main(experiment,complete):
    torch.set_num_threads(1);root=ROOTS[experiment];multi=experiment=='m81'
    frozen=read(root/'frozen.json');spec=read(root/'training_spec.json')
    assert sha(root/'training_spec.json')==frozen['training_spec_sha256']
    assert sha(root/'train_causal.py')==spec['training_script_sha256']
    assert sha(root/'causal_training.py')==spec['causal_script_sha256']
    assert sha(root/'window_competition.py')==spec['window_loss_sha256']
    assert sha(root/'integration.json')==spec['integration_sha256']
    for name,h in read(root/'integration.json')['source_sha256'].items():assert sha(root/'code'/name)==h
    assert spec['seed']==2027 and spec['learned_parameters']==289154
    if multi:
        assert sha(root/'fit_initializations.json')==spec['fit_initializations_sha256']
        assert sha(root/'multistart.py')==spec['multistart_source_sha256']
        manifest=read(root/'fit_initializations.json')
    else:
        assert sha(root/'text_schedule.json')==spec['text_schedule_sha256']
        assert sha(root/'text_dropout.py')==spec['text_dropout_source_sha256']
        schedule=read(root/'text_schedule.json')
    expected={};all_calls=all_steps=all_inits=0;all_modes=Counter()
    for row in spec['sequence_order']:
        name=row['sequence'];n=row['rgb_frames'];path=Path(spec['dataset_root'])/name/'groundtruth.txt'
        assert sha(path)==row['groundtruth_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4)
        if name=='toy07_indoor_320':
            assert n==1367 and len(gt)==1406 and row['groundtruth_sha256']=='683e8ae7ae401b71b8d10e9bb489c3956a150163606f5bac925a911f395444e2'
            gt=gt[:n]
        assert len(gt)==n
        valid=np.isfinite(gt).all(1)&(gt[:,2:]>0).all(1);valid[0]=False
        increments={};steps=0
        for end in list(range(32,n,32))+([] if (n-1)%32==0 else [n-1]):
            start=((end-1)//32)*32+1
            increments[end]=int(valid[start:end+1].any());steps+=increments[end]
        if multi:
            eps=[x for x in manifest['episodes'] if x['sequence']==name]
            assert eps[0]['start_frame']==0 and eps[-1]['end_frame_inclusive']==n-1
            assert sum(x['track_calls'] for x in eps)==n-1
            for i,ep in enumerate(eps):
                assert ep['start_frame']%32==0 and np.array_equal(gt[ep['start_frame']],ep['init_bbox'])
                assert sha(ep['image'])==ep['image_sha256'] and sha(ep['depth'])==ep['depth_sha256']
                if i:assert eps[i-1]['end_frame_inclusive']==ep['start_frame']
            starts={x['start_frame']:x for x in eps}
        else:starts={0:dict(id=name+':0',start_frame=0,init_bbox=row['first_box'])}
        modes=Counter()
        if not multi:
            assert len(schedule['sequences'][name])==math.ceil((n-1)/32)
            for i in range(1,n):modes['empty' if schedule['sequences'][name][(i-1)//32] else 'category']+=1
            all_modes.update(modes)
        expected[name]=dict(row=row,n=n,gt=gt,valid=valid,steps=steps,increments=increments,starts=starts,modes=dict(modes),steps_before=all_steps,calls_before=all_calls)
        all_calls+=n-1;all_steps+=steps;all_inits+=len(starts)
    assert len(expected)==130 and all_calls==186694 and all_steps==5798 and all_inits==(426 if multi else 130)
    if not multi:assert all_modes==Counter(category=149777,empty=36917)
    arms=['category','empty'] if multi else ['category']
    for arm in arms:
        b=spec['banks']['fit'][arm];assert sha(b['path'])==b['sha256']
        bank=torch.load(b['path'],map_location='cpu')
        assert set(bank['sequences'])==set(expected) and bank['tokens'].shape==((426 if multi else 130),5,768)
        assert torch.isfinite(bank['tokens']).all()
        initial=root/'native_parity'/(arm+'_zero.pth');assert sha(initial)==spec['initial_checkpoint_sha256'][arm]
        assert model_hash(torch.load(initial,map_location='cpu')['model'])=='56d03acff2972871788dd51f0eff40d0b6d888aad79fe74db4c44cdfb276ca42'
    report=dict(status='preflight_frozen_training_inputs_verified',experiment=experiment,observed_utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),training_spec_sha256=sha(root/'training_spec.json'),fit_sequences=130,tracking_calls=all_calls,valid_loss_optimizer_windows=all_steps,initializations=all_inits,text_condition_calls=dict(all_modes) if not multi else 'fixed_per_training_arm',formal_training_completion_verified=False,tracking_inference_calls=0,optimizer_steps_executed_by_auditor=0,independent_model_review_pass=False)
    if complete:
        sys.path.insert(0,str(root/'code'))
        from lib.train.data.processing_utils import transform_image_to_crop
        audited={}
        for arm in arms:
            assert (root/('training_'+arm+'.exit')).read_text().strip()=='0'
            folder=root/'training'/arm;result=read(folder/'result.json');checkpoint=torch.load(folder/'final.pth',map_location='cpu')
            assert result['status']=='one_full_causal_fit_pass_complete' and result['training_spec_sha256']==sha(root/'training_spec.json')
            assert result['total_track_calls']==all_calls and result['optimizer_steps']==all_steps
            assert sha(folder/'final.pth')==result['final_checkpoint_sha256']
            assert checkpoint['status']=='complete' and checkpoint['seed']==2027 and checkpoint['completed_sequences']==130
            assert checkpoint['frame_count']==all_calls and checkpoint['optimizer_steps']==all_steps
            assert checkpoint['training_spec_sha256']==sha(root/'training_spec.json') and checkpoint['base_checkpoint_sha256']==spec['native_checkpoint_sha256']
            assert sum(x.numel() for x in checkpoint['model'].values())==289154 and all(torch.isfinite(x).all() for x in checkpoint['model'].values())
            assert result['base_parameters_and_buffers_unchanged'] and result['base_state_before_sha256']=='c07022117c6efa8399b6df57e275ec81c6f3efce44eca5e4bc0282b52dd0dae0'
            assert sha(spec['native_checkpoint'])==spec['native_checkpoint_sha256']
            assert sha(folder/'sequence_log.jsonl')==result['sequence_log_sha256'] and sha(folder/'sampled_state_trace.jsonl')==result['sampled_trace_sha256']
            logs=[json.loads(x) for x in (folder/'sequence_log.jsonl').read_text().splitlines()]
            assert [x['sequence'] for x in logs]==list(expected)
            traces=[json.loads(x) for x in (folder/'sampled_state_trace.jsonl').read_text().splitlines()]
            trace_pairs=[];expected_pairs=[]
            for name,d in expected.items():
                expected_pairs.extend((name,i) for i in range(1,d['n']) if i%50==0 or i==d['n']-1 or (i-1 in d['starts'] if multi else i==1))
            for index,log in enumerate(logs):
                d=expected[log['sequence']]
                assert log['sequence_index']==index and log['track_calls']==d['n']-1
                assert log['supervised_frames']==int(d['valid'].sum())
                assert log['label_counts'].get('invalid',0)==d['n']-1-int(d['valid'].sum())
                assert sum(log['label_counts'].values())==d['n']-1
                assert log['total_optimizer_steps']==d['steps_before']+d['steps'] and log['total_track_calls']==d['calls_before']+d['n']-1
                if multi:assert log['initializations']==[x['id'] for x in d['starts'].values()]
                else:assert log['text_condition_calls']==d['modes']
            for row in traces:
                name=row['sequence'];i=row['frame_index'];d=expected[name];trace_pairs.append((name,i))
                begin=max(x for x in d['starts'] if x<i);relative=i-begin
                assert row['optimizer_steps_before_update']==d['steps_before']+sum(v for k,v in d['increments'].items() if k<i)
                if i==begin+1:assert row['previous_bbox']==d['starts'][begin]['init_bbox']
                if multi:assert row['episode_start']==begin and row['episode_frame_id']==relative
                else:assert row['text_dropped']==schedule['sequences'][name][(i-1)//32]
                assert bool(row['template_write'])==(relative%50==0 and row['best_score']>.75)
                assert np.isfinite(row['bbox']).all() and min(row['bbox'][2:])>0
                assert math.isfinite(row['best_score']) and math.isfinite(row['resize_factor'])
                if not d['valid'][i]:assert row['label']=='invalid'
                else:
                    target=transform_image_to_crop(torch.tensor(d['gt'][i],dtype=torch.float32),torch.tensor(row['previous_bbox'],dtype=torch.float32),row['resize_factor'],torch.tensor([256,256],dtype=torch.float32),normalize=True)
                    center=target[:2]+target[2:]/2
                    label='centre_inside' if bool(((center>=0)&(center<1)).all()) else 'centre_outside'
                    assert row['label']==label
            assert trace_pairs==expected_pairs
            if multi:assert result['initializations']==checkpoint['initializations']==426
            else:assert result['actual_text_condition_calls']==dict(all_modes)
            audited[arm]=dict(final_checkpoint_sha256=sha(folder/'final.pth'),final_adapter_tensor_sha256=model_hash(checkpoint['model']),sequence_rows=len(logs),sampled_trace_rows=len(traces),all_scheduled_trace_positions_matched=True,actual_optimizer_steps_verified=5798,actual_initialization_protocol_verified=True)
        report.update(status='completed_saved_training_artifacts_verified',formal_training_completion_verified=True,arms=audited,base_preservation_evidence='Completed trainer source assertions, recorded tensor fingerprint, unchanged checkpoint file; no separately stored post-training base copy.')
    assert not torch.cuda.is_initialized()
    output=root/('saved_training_audit.json' if complete else 'saved_training_audit_preflight.json')
    assert not output.exists();write(output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=list(ROOTS),required=True);p.add_argument('--complete',action='store_true');a=p.parse_args();main(a.experiment,a.complete)
