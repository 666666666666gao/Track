"""Only complete, source-bound native/CLIP Train grids and confirmed initial words."""
import json
import torch
from analyze_train_states import sha


def load_panel(args,bank):
    prep=json.loads((args.cache/'preparation.json').read_text())
    assert prep['training_labels_sha256']==sha(args.cache/'training_labels.json')
    assert prep['inference_inputs_sha256']==sha(args.cache/'inference_inputs.json')
    labels=json.loads((args.cache/'training_labels.json').read_text())
    plans={r['sequence']:r for r in json.loads((args.cache/'inference_inputs.json').read_text())}
    assert (args.native/'driver.exit').read_text().strip()=='0'
    native=json.loads((args.native/'result.json').read_text())
    assert native['status']=='complete_M120_native_grid_pair' and native['optimizer_steps']==0
    assert sum(x['events'] for x in native['full'])==3502 and sum(x['frames'] for x in native['full'])==219194
    assert sum(len(x['sequences']) for x in native['full'])==152
    assert all(x['frozen_state_exact'] and not x['GT_loaded'] and not x['text_loaded'] for x in native['full'])
    assert (args.dense/'driver.exit').read_text().strip()=='0'
    dense=json.loads((args.dense/'result.json').read_text())
    assert dense['status']=='complete_M117_region_pair' and dense['optimizer_steps']==0
    assert all(x['bank_sha256']==sha(args.bank) and x['encoder_sha256']==bank['encoder_sha256'] for x in dense['full'])
    counts={split:sum(r['current'] is not None and r['split']==split for r in labels.values()) for split in ['fit','development']}
    assert counts==dict(fit=2544,development=495)
    panel={};initial={}
    for split,n in counts.items():
        panel[split]={k:torch.empty(n,256,768,dtype=torch.float16) for k in ['rgb','depth','fused','clip']}
        panel[split].update(key=[],strata=[],target=torch.empty(n,4),initial_index=torch.empty(n,dtype=torch.long),
            boxes=torch.empty(n,256,4),native_valid=torch.empty(n,256),clip_valid=torch.empty(n,256),
            native_score=torch.empty(n,256),native_response=torch.empty(n,256),origin=torch.empty(n,3),
            image_shape=torch.empty(n,2))
    for kind in ['rgb','depth','fused','clip']:initial[kind]=torch.empty(152,256,768,dtype=torch.float16)
    initial.update(native_valid=torch.empty(152,256),clip_valid=torch.empty(152,256),origin=torch.empty(152,3),box=torch.empty(152,4))
    at={k:0 for k in panel};seen=set();invalid=0
    for shard in [0,1]:
        original=json.loads((args.cache/('collect_shard'+str(shard)+'.json')).read_text())
        assert original['status']=='complete' and not original['smoke']
        assert original['preparation_sha256']==sha(args.cache/'preparation.json')
        assert native['full'][shard]['checkpoint_sha256']==original['checkpoint_sha256']
        native_rows={r['sequence']:r for r in native['full'][shard]['sequences']}
        dense_rows={r['sequence']:r for r in dense['full'][shard]['sequences']}
        for row in original['sequences']:
            name=row['sequence'];assert name not in seen;seen.add(name)
            paths=[args.cache/'features'/(name+'.pt'),args.native/('full_shard'+str(shard))/'features'/(name+'.pt'),
                args.dense/('full_shard'+str(shard))/'features'/(name+'.pt')]
            digests=[row['feature_sha256'],native_rows[name]['feature_sha256'],dense_rows[name]['feature_sha256']]
            assert all(sha(p)==h for p,h in zip(paths,digests))
            old,full,clip=[torch.load(p,map_location='cpu') for p in paths]
            assert full['original_feature_sha256']==clip['original_feature_sha256']==row['feature_sha256']
            assert not old['labels_loaded'] and not full['GT_loaded'] and not clip['GT_loaded']
            assert not full['auxiliary_state_committed'] and not clip['tracker_state_committed']
            assert full['event_frames']==clip['event_frames']==old['event_frames']==plans[name]['event_frames']
            assert full['split']==clip['split']==old['split']==plans[name]['split']
            i=bank['sequences'].index(name);assert bank['splits'][i]==full['split']
            initial['box'][i]=torch.tensor(full['initial_bbox'])
            initial['origin'][i]=torch.tensor(full['initial_crop_origin'])
            assert full['initial_crop_origin']==clip['crop_origins'][0]
            initial['native_valid'][i]=full['initial_observed_fraction']
            initial['clip_valid'][i]=clip['initial_observed_fraction']
            for kind in ['rgb','depth','fused']:initial[kind][i]=full['initial_grids'][kind]
            initial['clip'][i]=clip['initial_search_tokens']
            for j,frame in enumerate(full['event_frames']):
                key=name+'@'+str(frame);label=labels[key];split=label['split']
                assert split==full['split']
                assert torch.equal(full['prior_bbox'][j].float(),old['prior_bbox'][j])
                assert float(full['resize_factor'][j].float())==float(old['resize_factor'][j])
                assert torch.equal(full['crop_origin'][j],torch.tensor(clip['crop_origins'][j+1],dtype=torch.float32))
                if label['current'] is None:invalid+=1;continue
                p=panel[split];k=at[split];at[split]+=1
                p['key'].append(key);p['strata'].append(label['strata'])
                for kind in ['rgb','depth','fused']:p[kind][k]=full[kind][j]
                p['clip'][k]=clip['search_tokens'][j]
                p['native_valid'][k]=full['observed_fraction'][j];p['clip_valid'][k]=clip['observed_fraction'][j]
                p['boxes'][k]=old['dense_boxes'][j];p['native_score'][k]=old['score_map'][j].reshape(256)
                p['native_response'][k]=old['response_map'][j].reshape(256)
                p['target'][k]=torch.tensor(label['current']);p['initial_index'][k]=i
                p['origin'][k]=full['crop_origin'][j];p['image_shape'][k]=old['image_shape'][j]
    assert len(seen)==152 and invalid==463 and at==counts
    assert all(bool(torch.isfinite(v).all()) for p in [initial]+list(panel.values()) for v in p.values() if torch.is_tensor(v))
    return panel,initial,dict(sequences=152,valid_gt=3039,invalid_gt_excluded=invalid,
        native_receipt_sha256=sha(args.native/'result.json'),dense_receipt_sha256=sha(args.dense/'result.json'),
        preparation_sha256=sha(args.cache/'preparation.json'),half_precision_feature_storage=True)


def inputs(part,indices,initial,bank,condition,device):
    data={k:v[indices].to(device).float() for k,v in part.items() if torch.is_tensor(v) and k not in ['target','initial_index']}
    ids=part['initial_index'][indices]
    for key,value in initial.items():data['initial_'+key]=value[ids].to(device).float()
    query=bank['tokens'][ids].to(device).float();mask=bank['mask'][ids].to(device)
    assert bool(mask[:,0].all())
    if condition in ['empty','generic']:
        query=bank['empty' if condition=='empty' else 'generic'].to(device).float()[None,None].expand_as(query)
    data.update(query=query,query_mask=mask,empty_query=bank['empty'].to(device).float(),condition=condition)
    return data


def observation_targets(target,origin,image_shape):
    left=torch.maximum(torch.maximum(target[:,:2],origin[:,:2]),torch.zeros_like(target[:,:2]))
    image_xy=image_shape.flip(-1)-1
    right=torch.minimum(torch.minimum(target[:,:2]+target[:,2:],origin[:,:2]+origin[:,2:]),image_xy)
    extent=(right-left).clamp_min(0);present=(extent>0).all(-1)
    center=(left+right)*.5;point=(center-origin[:,:2])/origin[:,2:]*16
    index=point.floor().clamp(0,15).long();cell=index[:,1]*16+index[:,0]
    axis=torch.arange(16,device=target.device,dtype=target.dtype)
    y,x=torch.meshgrid(axis,axis,indexing='ij');xy=torch.stack((x,y),dim=-1).reshape(256,2)
    sigma=(extent.min(-1).values/origin[:,2]*16/6).clamp_min(1)
    heat=torch.exp(-((xy[None]-index[:,None])**2).sum(-1)/(2*sigma[:,None]**2))*present[:,None]
    return heat,cell,present
