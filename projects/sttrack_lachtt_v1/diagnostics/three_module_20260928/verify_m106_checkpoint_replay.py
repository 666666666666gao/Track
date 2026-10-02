"""Read-only M106 checkpoint parity and complete cached geometry replay."""
from pathlib import Path
import hashlib
import json
import sys
import torch
sys.path.insert(0, '/home/SUTrack_RGBD_L')
from train_selected_geometry_preservation import LocalVisualRefiner, InstanceCandidatePrototype, load_inputs, metadata, encoded_inputs, decode, evaluate

ROOT = Path('/root/autodl-tmp/sttrack_m106_selected_geometry_20260928')
CACHE = Path('/root/autodl-tmp/sttrack_m90_train_states_20260928')
OLD = Path('/root/autodl-tmp/sttrack_m104_visual_geometry_20260928/train')
PARENT = Path('/root/autodl-tmp/sttrack_m101_ab_native_pair_20260928/train_weight1/final.pt')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    driver = json.loads((ROOT/'driver.json').read_text())
    assert driver['exit_code']==0 and (ROOT/'driver.exit').read_text().strip()=='0'
    assert [(r['mode'],r['weight'],r['gpu'],r['exit_code']) for r in driver['runs']]==[('sanity',0,0,0),('sanity',1,1,0),('train',0,0,0),('train',1,1,0)]
    old_state = torch.load(OLD/'final.pt',map_location='cpu')
    control_state = torch.load(ROOT/'train_weight0/final.pt',map_location='cpu')
    assert old_state.keys()==control_state.keys()
    assert all(torch.equal(v,control_state[k]) for k,v in old_state.items())
    old_result = json.loads((OLD/'result.json').read_text())
    results = {str(w):json.loads((ROOT/('train_weight'+str(w))/'result.json').read_text()) for w in [0,1]}
    for split in ['fit','development']:
        assert old_result[split]==results['0'][split]
        assert (OLD/(split+'_events.jsonl')).read_bytes()==(ROOT/'train_weight0'/(split+'_events.jsonl')).read_bytes()
    panel = load_inputs(CACHE,Path('/root/autodl-tmp/sttrack_m98_train_contexts_20260928'),Path('/root/autodl-tmp/sttrack_m95_initial_origins_20260928'),False)
    meta = metadata(panel,CACHE,Path('/root/autodl-tmp/sttrack_m103_geometry_probe_20260928'))
    empty = torch.load('/root/autodl-tmp/sttrack_full152_paired_20260925/text_full152.pt',map_location='cpu')['empty'].cuda().float()
    parent=InstanceCandidatePrototype().cuda()
    parent.load_state_dict(torch.load(PARENT,map_location='cuda'))
    parent.eval().requires_grad_(False)
    parent_before={k:v.detach().cpu().clone() for k,v in parent.state_dict().items()}
    encoded={s:encoded_inputs(parent,g,empty,torch.device('cuda')) for s,g in panel.items()}
    models={}
    before={}
    for w in ['0','1']:
        p=ROOT/('train_weight'+w)/'final.pt'
        assert sha(p)==results[w]['final_weight_sha256']
        model=LocalVisualRefiner().cuda()
        model.load_state_dict(torch.load(p,map_location='cuda'))
        model.eval().requires_grad_(False)
        models[w]=model
        before[w]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    out=ROOT/'verification'; assert not out.exists();out.mkdir()
    rows=[]
    with torch.no_grad():
        for split,group in panel.items():
            for w,model in models.items():
                summary,actual=evaluate(model,group,encoded[split],meta[split],torch.device('cuda'))
                saved=[json.loads(s) for s in (ROOT/('train_weight'+w)/(split+'_events.jsonl')).read_text().splitlines()]
                assert summary==results[w][split] and actual==saved
            for start in range(0,len(group['key']),64):
                at=torch.arange(start,min(start+64,len(group['key'])))
                boxes=group['boxes'][at].cuda().float()
                tokens=encoded[split]['tokens'][at].cuda()
                valid=encoded[split]['valid'][at].cuda()
                geometry=group['geometry'][at].cuda().float()
                shapes=meta[split]['image_shape'][at].cuda()
                variants={'parent':boxes.cpu()}
                for w,model in models.items():
                    variants['weight'+w]=decode(boxes,model(tokens,valid,geometry),shapes).cpu()
                for offset,index in enumerate(at.tolist()):
                    rows.append(dict(key=group['key'][index],split=split,strata=group['strata'][index],
                                     selected=int(encoded[split]['selected'][index]),
                                     boxes={name:value[offset].tolist() for name,value in variants.items()}))
    assert len(rows)==3039
    assert all(torch.equal(v.detach().cpu(),parent_before[k]) for k,v in parent.state_dict().items())
    for w,model in models.items():
        assert all(torch.equal(v.detach().cpu(),before[w][k]) for k,v in model.state_dict().items())
    (out/'geometry_boxes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=dict(status='complete_m106_readonly_checkpoint_replay',
                control_M104_all_tensors_equal=True,control_M104_all3039_rows_and_all_summaries_equal=True,
                both_weights_all3039_final_replay_equal=True,all_replayed_states_unchanged=True,
                events=len(rows),serialized_candidate_boxes=3039*10*3,optimizer_steps=0,
                checkpoints_created=False,parent_weight_sha256=sha(PARENT),
                training_labels_sha256=sha(CACHE/'training_labels.json'),
                M104_final_sha256=sha(OLD/'final.pt'),
                final_weights_sha256={w:sha(ROOT/('train_weight'+w)/'final.pt') for w in ['0','1']},
                source_sha256=sha(Path(__file__)),geometry_boxes_sha256=sha(out/'geometry_boxes.jsonl'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
