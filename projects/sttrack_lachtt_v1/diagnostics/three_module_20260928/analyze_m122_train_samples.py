"""CPU-only diagnosis of sealed Train152 logs and sparse historical states."""
import argparse,csv,hashlib,json,math,struct
from collections import defaultdict
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]

def valid(box):
    return all(math.isfinite(x) for x in box) and box[2]>0 and box[3]>0

def overlap(box,target,loss_float32=False):
    cast=f32 if loss_float32 else float
    box=list(map(cast,box));target=list(map(cast,target))
    extent=[max(cast(min(cast(box[i]+box[i+2]),cast(target[i]+target[i+2]))-max(box[i],target[i])),0.) for i in [0,1]]
    intersection=cast(extent[0]*extent[1])
    union=cast(cast(cast(box[2]*box[3])+cast(target[2]*target[3]))-intersection)
    return cast(intersection/union)

def intersects(target,origin,height,width):
    target=list(map(f32,target))
    left=[max(target[i],origin[i],0.) for i in [0,1]]
    right=[min(f32(target[i]+target[i+2]),f32(origin[i]+origin[2]),float(size-1)) for i,size in enumerate([width,height])]
    return all(right[i]>left[i] for i in [0,1])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    assert sha(Path(__file__))==plan['analysis_source_sha256']
    paths={name:Path(path) for name,path in plan['paths'].items()}
    for path,digest in plan['input_sha256'].items():assert sha(Path(path))==digest,path
    spec=json.loads(paths['spec'].read_text(encoding='utf-8'))
    assert spec['seed']==2027 and len(spec['sequence_order'])==152
    cases={x['sequence']:x for x in spec['sequence_order']}
    images=json.loads(paths['image_dimensions'].read_text(encoding='utf-8'))
    shapes={(x['sequence'],x['frame']):(x['height'],x['width']) for x in images}
    assert len(shapes)==len(images)==666
    truths={}
    for sequence,case in cases.items():
        path=paths['GT_root']/(sequence+'.txt')
        assert sha(path)==case['groundtruth_sha256']
        truth=[[float(v) for v in line.split(',')] for line in path.read_text(encoding='utf-8').splitlines()]
        assert all(len(x)==4 for x in truth)
        if sequence=='toy07_indoor_320':
            assert len(truth)==1406 and case['rgb_frames']==1367
            truth=truth[:1367]
        assert len(truth)==case['rgb_frames']==case['depth_frames']
        assert truth[0]==case['first_box']
        truths[sequence]=truth
    output=[];summaries=[];per_sequence=[]
    for model in ['precision0','precision1']:
        records=[json.loads(line) for line in paths[model+'_log'].read_text(encoding='utf-8').splitlines()]
        samples=[json.loads(line) for line in paths[model+'_trace'].read_text(encoding='utf-8').splitlines()]
        training=json.loads(paths[model+'_result'].read_text(encoding='utf-8'))
        assert records==training['sequence_records']
        assert len(records)==456 and training['track_calls']==659406 and training['supervised_frames']==610128 and training['optimizer_steps']==20421
        assert training['seed']==2027 and training['epochs']==3 and training['frozen_before_after_exact'] and training['buffers_exact']
        expected={(epoch,seq,frame) for epoch in [1,2,3] for seq,case in cases.items() for frame in ({1,case['rgb_frames']-1}|set(range(500,case['rgb_frames'],500)))}
        assert len(samples)==len(expected)==1998
        assert {(x['epoch'],x['sequence'],x['frame']) for x in samples}==expected
        grouped=defaultdict(list)
        for source in samples:
            seq=source['sequence'];frame=source['frame'];target=truths[seq][frame]
            assert source['GT_valid_for_loss']==valid(target)
            assert source['template_write']==(frame%50==0 and source['native_same_position_response']>.75)
            assert 0<=source['selected']<256 and valid(source['bbox'])
            prior=source['previous_bbox'];side=math.ceil(math.sqrt(prior[2]*prior[3])*4.)
            assert source['crop_origin']==[round(prior[0]+.5*prior[2]-.5*side),round(prior[1]+.5*prior[3]-.5*side),side]
            assert source['resize_factor']==256./side
            assert all(0<=source[k]<=1 for k in ['best_score','selected_quality','native_same_position_response','observation_intersection_probability'])
            row=dict(model=model,**source)
            if source['GT_valid_for_loss']:
                height,width=shapes[(seq,frame)]
                row.update(selected_iou_float64=overlap(source['bbox'],target),
                    selected_iou_loss_float32_CPU=overlap(source['bbox'],target,True),
                    GT_box_intersects_observed_window=intersects(target,source['crop_origin'],height,width))
                origin=source['crop_origin'];cx=target[0]+target[2]/2;cy=target[1]+target[3]/2
                row['GT_center_in_crop_square']=origin[0]<=cx<origin[0]+origin[2] and origin[1]<=cy<origin[1]+origin[2]
            grouped[(source['epoch'],seq)].append(row)
            output.append(row)
        for epoch in [1,2,3]:
            logs=[x for x in records if x['epoch']==epoch]
            assert [x['sequence'] for x in logs]==[x['sequence'] for x in spec['sequence_order']]
            assert len(logs)==152 and sum(x['track_calls'] for x in logs)==219802 and sum(x['supervised_frames'] for x in logs)==203376
            rows=[x for x in output if x['model']==model and x['epoch']==epoch]
            known=[x for x in rows if x['GT_valid_for_loss']]
            nonintersection=[x for x in known if not x['GT_box_intersects_observed_window']]
            bins=defaultdict(list)
            for x in known:bins[min(9,int(x['selected_quality']*10))].append(x)
            summaries.append(dict(model=model,epoch=epoch,full_track_calls=219802,full_supervised_frames=203376,
                full_GT_invalid_frames=219802-203376,full_GT_window_intersection_frames=sum(x['counts']['observation_intersection'] for x in logs),
                full_template_writes=sum(x['template_writes'] for x in logs),
                samples=len(rows),valid_GT_samples=len(known),invalid_GT_samples=len(rows)-len(known),
                severe_samples=sum(x['selected_iou_float64']<=.1 for x in known),
                crop_center_out_samples=sum(not x['GT_center_in_crop_square'] for x in known),
                GT_nonintersection_samples=len(nonintersection),
                nonintersection_predicted_present_at_half=sum(x['observation_intersection_probability']>=.5 for x in nonintersection),
                sampled_writes_valid_GT=sum(x['template_write'] for x in known),
                sampled_writes_severe=sum(x['template_write'] and x['selected_iou_float64']<=.1 for x in known),
                sampled_writes_invalid_GT=sum(x['template_write'] for x in rows if not x['GT_valid_for_loss']),
                sample_mean_iou=sum(x['selected_iou_float64'] for x in known)/len(known),
                quality_sample_absolute_error_mean=sum(abs(x['selected_quality']-x['selected_iou_loss_float32_CPU']) for x in known)/len(known),
                observation_sample_brier=sum((x['observation_intersection_probability']-int(x['GT_box_intersects_observed_window']))**2 for x in known)/len(known),
                quality_bins=[dict(bin_index=i,count=len(xs),mean_quality=sum(x['selected_quality'] for x in xs)/len(xs),
                    mean_loss_iou_CPU=sum(x['selected_iou_loss_float32_CPU'] for x in xs)/len(xs)) for i,xs in sorted(bins.items())]))
            for log in logs:
                rows=grouped[(epoch,log['sequence'])];known=[x for x in rows if x['GT_valid_for_loss']]
                per_sequence.append(dict(model=model,epoch=epoch,sequence=log['sequence'],
                    full_track_calls=log['track_calls'],full_supervised_frames=log['supervised_frames'],
                    full_GT_window_intersection_frames=log['counts']['observation_intersection'],
                    full_template_writes=log['template_writes'],samples=len(rows),valid_GT_samples=len(known),
                    severe_samples=sum(x['selected_iou_float64']<=.1 for x in known),
                    crop_center_out_samples=sum(not x['GT_center_in_crop_square'] for x in known),
                    GT_nonintersection_samples=sum(not x['GT_box_intersects_observed_window'] for x in known),
                    sampled_writes_severe=sum(x['template_write'] and x['selected_iou_float64']<=.1 for x in known),
                    sampled_writes_invalid_GT=sum(x['template_write'] for x in rows if not x['GT_valid_for_loss'])))
    for path,digest in plan['input_sha256'].items():assert sha(Path(path))==digest,path
    result=dict(status='complete_M122_sparse_training_diagnosis',evaluation_type='real_gt_training_diagnostic',
        neural_executions=0,neural_progress_queries=0,source_sha256=sha(Path(__file__)),plan_sha256=sha(args.plan),
        input_sha256=plan['input_sha256'],summaries=summaries,
        scope='Own histories DURING three training passes, not inference with the final checkpoint or held-out benchmark results.',
        limitations=['Samples are frame1, every500th frame and last frame; not a representative all-frame IoU or all-write error rate.',
            'Invalid GT is unknown, not an absence or identity-negative label.',
            'CPU float32 overlap is a reimplementation for diagnosis; no GPU bitwise replay claim.',
            'Recorded crop origins are used; no rounded-OPE-box reconstruction.',
            'RGB dimensions/hashes were read on the remote CPU; RGB image bytes are not in this local package.',
            'No text/module causal attribution, no inference threshold changes and no new C module efficacy claim.'])
    args.output.mkdir(exist_ok=False)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (args.output/'sample_rows.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    with (args.output/'per_sequence.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_sequence[0]))
        writer.writeheader();writer.writerows(per_sequence)
    print(json.dumps(dict(status=result['status'],rows=len(output),sequence_epoch_rows=len(per_sequence),summaries=summaries)))

if __name__=='__main__':
    main()
