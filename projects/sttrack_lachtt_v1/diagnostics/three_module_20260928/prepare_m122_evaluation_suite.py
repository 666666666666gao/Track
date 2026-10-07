"""Bind actual third-pass finals and confirmed text to the frozen full evaluation."""
import argparse,json,shutil
from pathlib import Path
from bind_m122_official_initializations import sha

NATIVE=Path('/root/autodl-tmp/sttrack_default_rgbd_ope_v1_20260906')
FROZEN=Path('/root/autodl-tmp/sttrack_default_full127_v1_20260905/run/shard_manifest.json')
VOT_METADATA=Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/vot_inputs/initializations/metadata_sha256.json')
REPOSITORY=Path('/root/autodl-tmp/rgbd_baselines/STTrack_lachtt_v1')
CHECKPOINT=Path('/root/autodl-tmp/sttrack_checkpoints/STTrack_Vot22.pth.tar')
CLIP=Path('/root/autodl-tmp/sutrack_assets/weights/ViT-L-14.pt')
BRIDGE=Path('/root/autodl-tmp/sttrack_full152_evaluation_20260925/interface/m39_vot_bridge.py')
FAILURES=Path('/home/SUTrack_RGBD_L/tools/finalize_vot_transaction_low22.py')
FAILURE_COMMON=FAILURES.with_name('finalize_vot_full127.py')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def checked_sources(path):
    gate=read(path);assert gate['status']=='reviewed_M122_complete_evaluation_source_gate'
    for name,digest in gate['source_sha256'].items():assert sha(name)==digest,name
    return gate


def prepare(args):
    import torch
    from dense_target_decoder import DenseTargetDecoder
    gate=checked_sources(args.source_gate);assert not args.output.exists()
    pair=read(args.training_root/'result.json')
    assert pair['status']=='complete_M122_full152_human_pair' and (args.training_root/'driver.exit').read_text().strip()=='0'
    assert all(x['exit']==0 for x in pair['launches'])
    assert len(pair['full'])==2 and {r['precision_weight'] for r in pair['full']}=={0,1}
    assert len({r['warm_final_sha256'] for r in pair['full']})==len({r['initial_state_sha256'] for r in pair['full']})==1
    binding=read(args.binding);bank=torch.load(args.bank,map_location='cpu')
    assert binding['status']=='complete_M122_human_official_initialization_binding'
    assert binding['counts']==dict(depthtrack_test=50,cdtb=80,vot=1765)
    assert bank['format']=='M122_human_official_initialization_v1' and bank['human_confirmed']
    assert bank['keys']==[r['key'] for r in binding['rows']] and len(set(bank['keys']))==1895
    assert bank['datasets']==[r['dataset'] for r in binding['rows']] and bank['binding_sha256']==sha(args.binding)
    assert bank['tokens'].shape==(1895,5,768) and bank['mask'].shape==(1895,5) and bank['mask'].dtype==torch.bool
    assert bank['encoder_sha256']==sha(CLIP)=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    assert bool(bank['mask'][:,0].all()) and bool(torch.isfinite(bank['tokens']).all())
    assert sha(CHECKPOINT)=='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98'
    assert sha(NATIVE/'inputs.json')=='61541e35f7b9e3c40427df79067fc0be20b8622cf275e93025e4a1547bf68601'
    assert sha(FROZEN)=='8e76256f1c7c135a65a1b262506356769b59557db60eb83bc21ef1392890dd01'
    assert sha(VOT_METADATA)=='1633e2f543dd97fa0b2bff989a84e5d9b4e3ceb1537a0a42d084249c90f4b855'
    metadata=read(VOT_METADATA)
    for name,digest in metadata.items():assert sha(name)==digest,name
    native=read(NATIVE/'spec.json');inputs=read(NATIVE/'inputs.json');frozen=read(FROZEN)
    assert frozen['total_anchor_count']==1765 and len(frozen['sequences'])==127 and len(frozen['shards'])==4
    assert len({t for s in frozen['shards'] for t in s['expected_trajectories']})==1765
    for name,info in frozen['source'].items():assert sha(Path(info['root'])/'anchor.value')==info['anchor_sha256'],name
    assert native['metric_source_sha256']=='05879f2e732aed982fbcbebd9756ce063ed0fa945c1f6b0c04092c3e487466cc'
    assert sha(native['metric_source'])==native['metric_source_sha256']
    assert sha(BRIDGE)=='230acf10f378a6babfacf9979ea07a1ce89c34952cc0c6b9568376249e265316'
    assert sha(FAILURES)=='e96a375a3792a80ab12e5671db468f559d514558a05d1181fe7afec48bcb514a'
    assert sha(FAILURE_COMMON)=='cbbe55132a4e64157011c0cbfa9162bd583f09588e3e497dcf741a8d5174a6bb'
    args.output.mkdir();models=[];source_dir=Path(__file__).parent
    for weight in [0,1]:
        trained=args.training_root/('train_precision'+str(weight));result=read(trained/'result.json')
        assert result==next(r for r in pair['full'] if r['precision_weight']==weight)
        assert result['status']=='complete_M122_full152_training' and result['epochs']==3 and result['sequence_runs']==456 and result['track_calls']==659406
        assert result['seed']==2027 and result['frozen_before_after_exact'] and result['buffers_exact'] and result['final_state_roundtrip_exact']
        assert result['GT_reinitializations_after_first_frame']==0 and result['no_external_test_cdtb_vot_optimization'] and result['no_best_selection']
        assert result['source_sha256']==sha(source_dir/'train_m122_full_causal.py') and result['runtime_source_sha256']==sha(source_dir/'full_dense_tracker.py')
        assert result['final_sha256']==sha(trained/'final.pt')
        model=DenseTargetDecoder();model.load_state_dict(torch.load(trained/'final.pt',map_location='cpu'),strict=True)
        assert sum(p.numel() for p in model.parameters())==result['parameters']==380167;del model
        name='precision'+str(weight);target=args.output/name;target.mkdir()
        bundle=dict(schema='M122_full152_final_official_v1',precision_weight=weight,repository=str(REPOSITORY),source_sha256=gate['source_sha256'],
            final_path=str(trained/'final.pt'),final_sha256=sha(trained/'final.pt'),training_result_path=str(trained/'result.json'),training_result_sha256=sha(trained/'result.json'),
            native_checkpoint_path=str(CHECKPOINT),native_checkpoint_sha256=sha(CHECKPOINT),clip_weight_path=str(CLIP),clip_weight_sha256=sha(CLIP),
            vot_bridge_path=str(BRIDGE),failure_source_path=str(FAILURES),failure_common_path=str(FAILURE_COMMON),
            source_gate_path=str(args.source_gate),source_gate_sha256=sha(args.source_gate),human_review_used_multiframe_aids=True,
            external_metric_checkpoint_selection=False,external_optimizer_steps=0)
        write(target/'bundle.json',bundle)
        shared=dict(bundle_path=str(target/'bundle.json'),bundle_sha256=sha(target/'bundle.json'),bank_path=str(args.bank),bank_sha256=sha(args.bank),
            binding_path=str(args.binding),binding_sha256=sha(args.binding))
        plans={}
        for dataset,native_name,count,frames in [('depthtrack_test','depthtrack',50,76373),('cdtb','cdtb',80,101956)]:
            root=target/dataset;root.mkdir();reference=read(NATIVE/('metrics_'+native_name+'.json'))
            rows=inputs[native_name];assert len(rows)==count and sum(r['frames'] for r in rows)==frames
            cases=[dict(sequence=r['sequence'],frames=r['frames'],init_bbox=r['init_bbox'],gt_sha256=reference['groundtruth_sha256'][r['sequence']]) for r in rows]
            write(root/'cases.json',cases)
            write(root/'plan.json',dict(shared,dataset=dataset,cases_path=str(root/'cases.json'),cases_sha256=sha(root/'cases.json'),
                dataset_root=native['datasets'][native_name]['root'],output=str(root/'predictions'),metric_source=native['metric_source'],metric_source_sha256=native['metric_source_sha256']))
            plans[dataset]=dict(path=str(root/'plan.json'),sha256=sha(root/'plan.json'))
        root=target/'vot';root.mkdir();write(root/'plan.json',dict(shared,dataset='vot'))
        run=root/'run';run.mkdir();tracker='sttrack_m122_'+name+'_human_full127'
        wrapper=root/'selected_m122_vot.py'
        wrapper.write_text('import sys\nsys.path.insert(0,'+repr(str(source_dir))+')\nfrom run_m122_official import vot_track\nvot_track('+repr(str(root/'plan.json'))+')\n')
        shards=[]
        for s in frozen['shards']:
            src=Path(s['root']);dest=run/('shard-%02d'%s['index']);gpu=s['index']%2
            assert sha(src/'config.yaml')==s['config_sha256'] and sha(src/'sequences/list.txt')==s['list_sha256']
            shutil.copytree(src/'sequences',dest/'sequences',symlinks=True);shutil.copyfile(src/'config.yaml',dest/'config.yaml')
            ini=(f'[{tracker}]\nlabel = M122 {name} human Full152\nprotocol = traxpython\ncommand = selected_m122_vot\npaths = {root}\n'
                f'python = /root/autodl-tmp/envs/sttrack/bin/python\nenv_CUDA_VISIBLE_DEVICES = {gpu}\nenv_PYTHONPATH = {root}\n'
                'env_TOKENIZERS_PARALLELISM = false\nenv_PYTHONDONTWRITEBYTECODE = 1\ntimeout = 600\nrestart = false\n')
            (dest/'trackers.ini').write_text(ini);shards.append(dict(s,root=str(dest),gpu=gpu,trackers_sha256=sha(dest/'trackers.ini')))
        write(run/'shard_manifest.json',dict(frozen,schema='M122_full127_frozen_shards_v1',tracker=tracker,gpu_count=2,shards=shards))
        files=[target/'bundle.json',root/'plan.json',wrapper,run/'shard_manifest.json']
        files += [Path(info['root'])/'anchor.value' for info in frozen['source'].values()]
        files += [Path(s['root'])/n for s in shards for n in ['config.yaml','trackers.ini','sequences/list.txt']]
        # Freeze copied anchor overlays and all metadata, without changing their values.
        for s,old in zip(shards,frozen['shards']):
            for original,digest in metadata.items():
                old_root=Path(old['root']);path=Path(original)
                if old_root in path.parents:
                    copied=Path(s['root'])/path.relative_to(old_root);assert sha(copied)==digest;files.append(copied)
        write(root/'execution.json',dict(status='frozen_before_M122_VOT_tracking',source_sha256={str(p):sha(p) for p in files},
            anchors=1765,sequences=127,result_files=5295,original_manifest_sha256=sha(FROZEN),metadata_sha256=sha(VOT_METADATA),
            plans_sha256=sha(root/'plan.json'),workers=2,poll_seconds=300,reused_prediction_anchors=0))
        plans['vot']=dict(path=str(root/'plan.json'),sha256=sha(root/'plan.json'),run=str(run),execution_sha256=sha(root/'execution.json'))
        models.append(dict(name=name,precision_weight=weight,final_sha256=bundle['final_sha256'],bundle_sha256=sha(target/'bundle.json'),plans=plans))
    checked_sources(args.source_gate)
    write(args.output/'selection.json',dict(status='M122_two_actual_third_pass_finals_frozen_before_metrics',models=models,
        training_pair_sha256=sha(args.training_root/'result.json'),source_gate_sha256=sha(args.source_gate),bank_sha256=sha(args.bank),binding_sha256=sha(args.binding),
        external_metric_checkpoint_selection=False,external_optimizer_steps=0,GT_used_for_optimization=False))
    print(json.dumps(dict(status='prepared_M122_complete_six_evaluation_plans',models=[r['name'] for r in models])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['training-root','binding','bank','source-gate','output']:p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
