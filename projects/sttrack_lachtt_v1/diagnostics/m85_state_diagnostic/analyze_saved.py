"""Posthoc dense capacity on sealed M85 trajectories; no tracking or training."""
import hashlib,json
from pathlib import Path
import numpy as np

R=Path(__file__).parent
P=Path('/root/autodl-tmp/sttrack_m84_centered_20260920')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def dense_boxes(maps, row):
    grid=np.arange(256,dtype=np.float32)
    m=maps.reshape(5,256)
    normalized=np.stack([(grid%16+m[3])/16,(grid//16+m[4])/16,m[1],m[2]],axis=1)
    values=(normalized*np.float32(256)/np.float32(256./row['search_side'])).astype(np.float64)
    previous=row['previous_bbox'];h,w=row['image_hw']
    values[:,0]+=previous[0]+.5*previous[2]-.5*row['search_side']
    values[:,1]+=previous[1]+.5*previous[3]-.5*row['search_side']
    xy=values[:,:2]-.5*values[:,2:];end=xy+values[:,2:]
    xy=np.maximum(0,xy);xy=np.minimum(xy,[w-10,h-10])
    end=np.minimum(np.maximum(10,end),[w,h])
    return np.concatenate([xy,np.maximum(10,end-xy)],axis=1)

def iou(boxes,gt):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(b[:,:2]+b[:,2:],gt[:2]+gt[2:])-np.maximum(b[:,:2],gt[:2])).prod(1)
    return intersection/(b[:,2:].prod(1)+gt[2]*gt[3]-intersection)

def low_runs(records,arm):
    runs=[];start=None
    for row in records+[dict(frame=records[-1]['frame']+1,valid=False)]:
        low=row['valid'] and row['variants'][arm]['hann_iou']<=.1
        if low and start is None:start=row['frame']
        if not low and start is not None:
            if row['frame']-start>=10:runs.append([start,row['frame']])
            start=None
    return runs

def main():
    assert (R/'controller.exit').read_text().strip()=='0'
    plan=read(R/'spec.json');receipt=read(R/'predictions/receipt.json')
    assert sha(Path(__file__))==plan['analysis_sha256']
    assert receipt['status']=='complete' and receipt['positions']==2616 and receipt['mode']=='selected3'
    assert receipt['spec_sha256']==sha(R/'spec.json') and receipt['source_sha256']==sha(R/'same_state.py')
    assert len(receipt['sequences'])==len(plan['cases'])==3
    # Seal all sources and predictions before reading any subsequent GT.
    for c,item in zip(plan['cases'],receipt['sequences']):
        assert c['sequence']==item['sequence'] and item['positions']==c['frames']-1
        for suffix,key in [('.json','sha256'),('.npz','dense_sha256')]:
            assert sha(R/'predictions'/(c['sequence']+suffix))==item[key]
    dataset=Path(read(P/'training_spec.json')['dataset_root']);reports=[]
    for case in plan['cases']:
        seq=case['sequence'];path=dataset/seq/'groundtruth.txt';assert sha(path)==case['gt_sha256']
        gt=np.loadtxt(path,delimiter=',').reshape(-1,4);assert len(gt)==case['frames']
        data=read(R/'predictions'/(seq+'.json'));assert data['sequence']==seq
        assert [r['frame'] for r in data['rows']]==list(range(1,len(gt)))
        with np.load(R/'predictions'/(seq+'.npz'),allow_pickle=False) as archive:
            maps={k:archive[k] for k in ['category','swapped','native']};window=archive['window'].reshape(256)
        assert all(m.shape==(len(gt)-1,5,16,16) and np.isfinite(m).all() for m in maps.values())
        records=[];max_decode_error=0.
        for j,row in enumerate(data['rows']):
            i=row['frame'];g=gt[i];valid=bool(np.isfinite(g).all() and (g[2:]>0).all())
            origin=np.asarray(row['crop_origin']);end=origin+row['search_side']
            center=g[:2]+g[2:]/2
            record=dict(frame=i,valid=valid,center_in_crop=bool(valid and ((center>=origin)&(center<end)).all()),
                full_box_in_crop=bool(valid and (g[:2]>=origin).all() and (g[:2]+g[2:]<=end).all()),variants={})
            for arm,values in maps.items():
                m=values[j];boxes=dense_boxes(m,row);raw=m[0].reshape(256);hann=raw*window
                v=row['variants'][arm];assert int(raw.argmax())==v['raw_peak'] and int(hann.argmax())==v['hann_peak']
                for kind in ['raw','hann']:
                    error=float(np.abs(boxes[v[kind+'_peak']]-np.asarray(v[kind+'_bbox'])).max())
                    max_decode_error=max(max_decode_error,error);assert error<=1e-4,(seq,i,arm,kind,error)
                if valid:
                    overlaps=iou(boxes,g)
                    record['variants'][arm]=dict(raw_iou=float(iou([v['raw_bbox']],g)[0]),
                        hann_iou=float(iou([v['hann_bbox']],g)[0]),dense_best_iou=float(overlaps.max()),
                        dense_correct_count=int((overlaps>=.5).sum()),dense_best_index=int(overlaps.argmax()))
            records.append(record)
        valid_rows=[r for r in records if r['valid']]
        bad=[r for r in valid_rows if r['variants']['category']['hann_iou']<=.1]
        summary=dict(valid_frames=len(valid_rows),category_low_frames=len(bad),
            low_center_outside=sum(not r['center_in_crop'] for r in bad),
            low_center_inside_no_correct_dense=sum(r['center_in_crop'] and r['variants']['category']['dense_correct_count']==0 for r in bad),
            low_center_inside_correct_dense=sum(r['center_in_crop'] and r['variants']['category']['dense_correct_count']>0 for r in bad),
            raw_rescue=sum(r['variants']['category']['raw_iou']>=.5 for r in bad),
            raw_harm=sum(r['variants']['category']['hann_iou']>=.5 and r['variants']['category']['raw_iou']<=.1 for r in valid_rows))
        for arm in ['swapped','native']:
            summary[arm+'_rescue']=sum(r['variants'][arm]['hann_iou']>=.5 for r in bad)
            summary[arm+'_harm']=sum(r['variants']['category']['hann_iou']>=.5 and r['variants'][arm]['hann_iou']<=.1 for r in valid_rows)
            summary[arm+'_correct_dense_on_category_low']=sum(r['variants'][arm]['dense_correct_count']>0 for r in bad)
        summary['low_all_heads_no_correct_dense']=sum(all(v['dense_correct_count']==0 for v in r['variants'].values()) for r in bad)
        assert summary['low_center_outside']+summary['low_center_inside_no_correct_dense']+summary['low_center_inside_correct_dense']==len(bad)
        reports.append(dict(sequence=seq,summary=summary,H10={arm:low_runs(records,arm) for arm in maps},max_selected_decoder_error=max_decode_error,rows=records))
    result=dict(status='complete_selected_diagnostic',spec_sha256=sha(R/'spec.json'),source_sha256=sha(Path(__file__)),receipt_sha256=sha(R/'predictions/receipt.json'),
        scope='Three posthoc selected sequences; same M84 Category state. Dense candidate capacity is hindsight, not recovery. Empty maps exactly equal native in replay.',sequences=reports)
    (R/'diagnostic_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(sequence=r['sequence'],**r['summary'],max_selected_decoder_error=r['max_selected_decoder_error']) for r in reports],indent=2))

if __name__=='__main__':main()
