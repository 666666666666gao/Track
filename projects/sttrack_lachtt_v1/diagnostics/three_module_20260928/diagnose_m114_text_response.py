"""Read M113 candidate responses; GT IoU is localization evidence, not identity."""
import argparse,json,time
from pathlib import Path
import torch
from analyze_train_states import sha
from instance_ab_prototype import InstanceCandidatePrototype
from train_ab_visual_control import load_inputs
from train_m110_no_weak_rank import inputs


def main():
    p=argparse.ArgumentParser()
    for name in ['cache','contexts','origins','bank','labels','checkpoint','reference','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=['human_text','generic'],required=True)
    p.add_argument('--mode',choices=['sanity','full'],required=True)
    a=p.parse_args();assert not a.output.exists();started=time.time()
    torch.set_num_threads(1);device=torch.device('cuda')
    reference=json.loads((a.reference/'result.json').read_text())
    digest=sha(a.checkpoint);assert digest==reference['final_sha256']
    panel=load_inputs(a.cache,a.contexts,a.origins,False)
    bank=torch.load(a.bank,map_location='cpu');labels=json.loads(a.labels.read_text())
    assert bank['human_confirmed'] and labels['human_confirmed']
    assert bank['labels_sha256']==reference['labels_sha256']==sha(a.labels)
    assert sha(a.bank)==reference['bank_sha256']
    assert bank['sequences']==[r['sequence'] for r in labels['initial']]
    assert bank['splits']==[r['split'] for r in labels['initial']]
    splits={r['sequence']:r['split'] for r in labels['initial']}
    model=InstanceCandidatePrototype().to(device)
    saved=torch.load(a.checkpoint,map_location='cpu')
    model.load_state_dict(saved,strict=True);model.requires_grad_(False);model.eval()
    rows=[]
    with torch.no_grad():
        for split in ['fit','development']:
            data=panel[split]
            assert all(splits[k.rsplit('@',1)[0]]==split for k in data['key'])
            expected={condition:{r['key']:r for r in map(json.loads,(a.reference/(condition+'_development.jsonl')).read_text().splitlines())}
                for condition in ['empty','generic','human_text']} if split=='development' else {
                    a.arm:{r['key']:r for r in map(json.loads,(a.reference/'fit_events.jsonl').read_text().splitlines())}}
            count=len(data['key']) if a.mode=='full' else 64
            for start in range(0,count,64):
                ids=torch.arange(start,min(start+64,count));outputs={}
                for condition in ['empty','generic','human_text']:
                    out=model(inputs(data,ids,bank,condition,device))
                    assert all(bool(torch.isfinite(out[name]).all()) for name in ['selection_logits','quality_logits','semantic_delta','phrase_delta'])
                    if condition=='empty':
                        assert torch.equal(out['selection_logits'],out['visual_selection_logits'])
                        assert not bool(out['semantic_delta'].any()) and not bool(out['phrase_delta'].any())
                    outputs[condition]={name:out[name].cpu() for name in ['selection_logits','quality_logits','semantic_delta','phrase_delta']}
                baseline=outputs['empty']
                for condition,out in outputs.items():
                    assert torch.equal(out['quality_logits'],baseline['quality_logits'])
                    for j,index in enumerate(ids.tolist()):
                        key=data['key'][index];score=out['selection_logits'][j]
                        iou=data['iou'][index];selected=int(score.argmax())
                        if condition in expected:
                            old=expected[condition][key]
                            assert selected==old['selected'],(condition,key,selected,old['selected'])
                            assert float(iou[selected])==old['selected_iou'],key
                        rows.append(dict(key=key,split=split,strata=data['strata'][index],condition=condition,
                            candidate_iou=iou.tolist(),native_scores=data['base_scores'][index].tolist(),
                            empty_scores=baseline['selection_logits'][j].tolist(),scores=score.tolist(),
                            selected=selected,selected_iou=float(iou[selected]),
                            semantic_delta_norm=float(out['semantic_delta'][j].norm()),
                            phrase_delta_norm=float(out['phrase_delta'][j].norm())))
    assert all(torch.equal(v.cpu(),saved[k]) for k,v in model.state_dict().items())
    assert sha(a.checkpoint)==digest
    a.output.mkdir()
    events=a.output/'responses.jsonl'
    events.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=dict(status='complete_M114_response_'+a.mode,arm=a.arm,mode=a.mode,optimizer_steps=0,
        human_confirmed=True,checkpoint_sha256=digest,bank_sha256=sha(a.bank),labels_sha256=sha(a.labels),
        source_sha256=sha(__file__),events_sha256=sha(events),conditions=['empty','generic','human_text'],
        states_per_split={s:sum(r['split']==s and r['condition']=='empty' for r in rows) for s in ['fit','development']},
        stored_selection_replay_matches=True,replay_scope='first_batch_per_split' if a.mode=='sanity' else 'all_cached_states',
        weights_buffers_unchanged=True,empty_exact_visual=True,
        geometry_quality_unchanged=True,no_GT_in_model_inputs=True,localization_not_identity_labels=True,
        no_recursive_or_official_evaluation=True,elapsed_seconds=time.time()-started)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
