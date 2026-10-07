"""Three complete human-text Train152 passes on own crop/query/template history."""
import argparse,hashlib,json,random,time
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
from analyze_train_states import sha
from dense_target_decoder import DenseTargetDecoder
from full_dense_tracker import FullDenseTracker,state_digest
from train_m121_dense_target import objective,box_overlap,PARAMETER_GROUPS


def load_truth(folder,row):
    assert sha(folder/'groundtruth.txt')==row['groundtruth_sha256']
    target=np.loadtxt(folder/'groundtruth.txt',delimiter=',').reshape(-1,4)
    if row['sequence']=='toy07_indoor_320':
        assert len(target)==1406 and row['rgb_frames']==1367
        target=target[:1367]
    assert len(target)==row['rgb_frames']==row['depth_frames']
    assert np.array_equal(target[0],np.asarray(row['first_box']))
    return target


def precision_objective(out,data,target,weight):
    measured=dict(out,boxes=out['boxes'].float())
    loss,terms,counts=objective(measured,data,target)
    native,_=box_overlap(data['boxes'],target);refined,_=box_overlap(measured['boxes'],target)
    qualified=native.detach()>=.5
    protection=(torch.relu(native.detach()-refined)*qualified).sum()/qualified.sum().clamp_min(1)
    return loss+weight*protection,torch.cat((terms,protection[None])),counts


def main():
    parser=argparse.ArgumentParser()
    for name in ['spec','repository','checkpoint','clip-weight','bank','labels','warm-final','warm-result','cache-plan','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=['sanity','train'],required=True)
    parser.add_argument('--precision-weight',type=int,choices=[0,1],required=True)
    args=parser.parse_args();assert not args.output.exists()
    spec=json.loads(args.spec.read_text());bank=torch.load(args.bank,map_location='cpu');labels=json.loads(args.labels.read_text())
    warm=json.loads(args.warm_result.read_text());plans=json.loads(args.cache_plan.read_text())
    assert spec['seed']==2027 and spec['total_training_track_calls']==219802
    assert len(spec['sequence_order'])==len(bank['sequences'])==len(labels['initial'])==152
    assert bank['human_confirmed'] and labels['human_confirmed'] and bank['dataset']==labels['dataset']=='depthtrack'
    assert bank['labels_sha256']==sha(args.labels)=='6ffb6e9907fee6a520e31fa78b4d3ff044a46a7daf8ad1aaa42c4d638b8c50c2'
    assert sha(args.bank)==warm['bank_sha256']=='a599e063b79b9458aab9e63ec21b9f4ac9735e420bceeec95275bfd1289603b2'
    assert sha(args.warm_final)==warm['final_sha256']=='d4843b7fc98471f3ef3d189a476c179ceb63f5f36b5fb4d19356a0e1314f7304'
    assert warm['status']=='complete_M121_training' and warm['arm']=='human_text' and warm['optimizer_steps']==640
    assert sha(args.checkpoint)==spec['native_checkpoint_sha256']=='cacbd799115be1aaeb049cee0db89270851e3b6dd68997553b4c2c31c1104f98'
    assert sha(args.clip_weight)==bank['encoder_sha256']=='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
    assert {r['sequence'] for r in spec['sequence_order']}==set(bank['sequences'])
    assert sum(r['rgb_frames']-1 for r in spec['sequence_order'])==219802
    torch.set_num_threads(1);random.seed(2027);np.random.seed(2027);torch.manual_seed(2027);torch.cuda.manual_seed_all(2027)
    sys.path.insert(0,str(args.repository))
    from lib.train.dataset.depth_utils import get_rgbd_frame
    model=DenseTargetDecoder().cuda();model.load_state_dict(torch.load(args.warm_final,map_location='cpu'),strict=True)
    actor=FullDenseTracker(args.repository,args.checkpoint,args.clip_weight,bank,model)
    frozen=actor.frozen_digest();initial=state_digest(model);parameters={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    args.output.mkdir();started=time.time();steps=calls=supervised=0;visited=hashlib.sha256();sequence_records=[];gradient_groups=[]
    if args.mode=='sanity':
        # Real zero-delta trajectory check includes frame50 template writing; no training GT is consulted.
        zero=DenseTargetDecoder().cuda().eval();actor.decoder=zero
        case=spec['sequence_order'][0];folder=Path(spec['dataset_root'])/case['sequence']
        expected=next(r for r in plans if r['sequence']==case['sequence'])['expected_rows']
        def first_image(i):
            return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
        actor.initialize(first_image(0),case['first_box'],bank['sequences'].index(case['sequence']))
        with torch.no_grad():
            for frame in range(1,65):
                out,data,state=actor.step(first_image(frame));chosen=state['selected']
                assert torch.equal(out['response'],data['native_response']*.5)
                assert state['bbox']==expected[frame]['bbox'],(frame,state['bbox'],expected[frame]['bbox'])
                assert state['native_same_position_response']==expected[frame]['score']
        actor.decoder=model;del zero
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    with (args.output/'sequence_log.jsonl').open('w') as log,(args.output/'sampled_state_trace.jsonl').open('w') as trace:
        for epoch in range(3 if args.mode=='train' else 1):
            order=spec['sequence_order'] if args.mode=='train' else spec['sequence_order'][:1]
            for seq_index,case in enumerate(order):
                folder=Path(spec['dataset_root'])/case['sequence'];truth=load_truth(folder,case)
                def image(i):
                    return get_rgbd_frame(str(folder/'color'/('%08d.jpg'%(i+1))),str(folder/'depth'/('%08d.png'%(i+1))),dtype='rgbcolormap',depth_clip=True)
                actor.initialize(image(0),case['first_box'],bank['sequences'].index(case['sequence']));model.train()
                pending=grad_frames=writes=local_supervised=0;terms_sum=np.zeros(7);counts=Counter();optimizer.zero_grad(set_to_none=True)
                limit=case['rgb_frames'] if args.mode=='train' else 65
                for frame in range(1,limit):
                    out,data,state=actor.step(image(frame));calls+=1;pending+=1;writes+=state['template_write']
                    valid=bool(np.isfinite(truth[frame]).all() and (truth[frame,2:]>0).all())
                    if valid:
                        target=torch.tensor(truth[frame:frame+1],device='cuda',dtype=torch.float32)
                        loss,terms,diagnostic=precision_objective(out,data,target,args.precision_weight)
                        assert bool(torch.isfinite(loss));(loss/32).backward();grad_frames+=1;supervised+=1;local_supervised+=1
                        terms_sum+=terms.detach().cpu().numpy();counts.update(diagnostic)
                    visited.update(json.dumps([epoch,case['sequence'],frame,state],separators=(',',':')).encode())
                    if frame%500==0 or frame==1 or frame==limit-1:
                        trace.write(json.dumps(dict(epoch=epoch+1,sequence=case['sequence'],GT_valid_for_loss=valid,**state))+'\n');trace.flush()
                    if pending==32 or frame==limit-1:
                        if grad_frames:
                            for parameter in model.parameters():
                                if parameter.grad is not None:parameter.grad.mul_(32/grad_frames)
                            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                            if args.mode=='sanity':
                                gradient_groups.append({name:sum(float(p.grad.norm()) for k,p in model.named_parameters()
                                    if k.startswith(name+'.') and p.grad is not None) for name in PARAMETER_GROUPS})
                            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert bool(torch.isfinite(norm))
                            optimizer.step();steps+=1
                        optimizer.zero_grad(set_to_none=True);pending=grad_frames=0
                    del out,data
                record=dict(epoch=epoch+1,sequence=case['sequence'],sequence_index=seq_index,sequences_per_epoch=len(order),track_calls=limit-1,
                    supervised_frames=local_supervised,template_writes=writes,loss_means=(terms_sum/local_supervised).tolist() if local_supervised else None,
                    counts=dict(counts),total_track_calls=calls,total_optimizer_steps=steps,elapsed_seconds=time.time()-started)
                sequence_records.append(record);log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
                if args.mode=='train':torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
                    completed_sequences=seq_index+1,total_track_calls=calls,total_optimizer_steps=steps),args.output/'latest.pt')
    assert actor.frozen_digest()==frozen
    assert all(p.grad is None for p in actor.native.network.parameters()) and all(p.grad is None for p in actor.clip.parameters())
    assert all(torch.equal(v.cpu(),buffers[k]) for k,v in model.named_buffers())
    changes={name:sum(int(p.detach().cpu().ne(parameters[k]).sum()) for k,p in model.named_parameters() if k.startswith(name+'.')) for name in PARAMETER_GROUPS}
    assert all(changes.values())
    result=dict(status='complete_M122_full152_training' if args.mode=='train' else 'complete_M122_recursive_sanity',mode=args.mode,
        precision_weight=args.precision_weight,seed=2027,epochs=3 if args.mode=='train' else 1,sequence_runs=len(sequence_records),
        track_calls=calls,supervised_frames=supervised,optimizer_steps=steps,gradient_accumulation=32,learning_rate=3e-5,weight_decay=.01,
        initial_state_sha256=initial,warm_final_sha256=sha(args.warm_final),bank_sha256=sha(args.bank),labels_sha256=sha(args.labels),
        spec_sha256=sha(args.spec),source_sha256=sha(__file__),runtime_source_sha256=sha(Path(__file__).with_name('full_dense_tracker.py')),
        parameters=sum(p.numel() for p in model.parameters()),parameter_changes=changes,frozen_before_after_exact=True,buffers_exact=True,
        GT_reinitializations_after_first_frame=0,GT_used_after_action_for_loss_only=True,state_detached_no_temporal_backprop=True,
        template_rule='original native response at the SAME selected cell; interval50/threshold0.75 unchanged scale; no new C claim',
        visited_sha256=visited.hexdigest(),elapsed_seconds=time.time()-started,sequence_records=sequence_records,
        no_external_test_cdtb_vot_optimization=True,no_best_selection=True,no_public_evaluation_yet=True)
    if args.mode=='sanity':
        assert calls==64 and steps==2 and all(x>0 for x in gradient_groups[-1].values())
        result.update(gradient_groups=gradient_groups,native_zero_64frame_trajectory_exact=True,final_saved=False)
    else:
        assert calls==3*219802 and len(sequence_records)==3*152
        torch.save(model.state_dict(),args.output/'final.pt');reloaded=torch.load(args.output/'final.pt',map_location='cpu')
        assert all(torch.equal(v.cpu(),reloaded[k]) for k,v in model.state_dict().items())
        result.update(final_sha256=sha(args.output/'final.pt'),final_state_roundtrip_exact=True)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],calls=calls,updates=steps)),flush=True)


if __name__=='__main__':main()
